"""
Advanced features for agenteval — 2026 Standard.

Covers: Caching, Pipeline, Validation & Schema, Async & Concurrency,
Observability, Streaming & Storage, Diff & Regression, Security & Cost.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

from trajscore.exceptions import BudgetExceededError, EvaluationError
from trajscore.models import EvaluationResult, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CACHING
# ─────────────────────────────────────────────

class TrajectoryCache:
    """LRU + TTL cache for TrajectoryScore results, keyed by SHA-256."""

    def __init__(self, max_size: int = 256, ttl: float = 300.0) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self._store: Dict[str, Tuple[TrajectoryScore, float]] = {}
        self._order: deque = deque()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(trajectory: Trajectory) -> str:
        raw = json.dumps(trajectory.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, trajectory: Trajectory) -> Optional[TrajectoryScore]:
        """Return cached score or None if expired / not found."""
        k = self._key(trajectory)
        with self._lock:
            if k in self._store:
                score, ts = self._store[k]
                if time.time() - ts <= self.ttl:
                    self._hits += 1
                    return score
                del self._store[k]
            self._misses += 1
        return None

    def put(self, trajectory: Trajectory, score: TrajectoryScore) -> None:
        """Insert or update a cached score, evicting LRU if at capacity."""
        k = self._key(trajectory)
        with self._lock:
            if k in self._store:
                self._order.remove(k)
            elif len(self._store) >= self.max_size:
                oldest = self._order.popleft()
                self._store.pop(oldest, None)
            self._store[k] = (score, time.time())
            self._order.append(k)

    def memoize(self, evaluate_fn: Callable[[Trajectory], TrajectoryScore]) -> Callable[[Trajectory], TrajectoryScore]:
        """Decorator: cache the result of evaluate_fn."""
        def wrapper(trajectory: Trajectory) -> TrajectoryScore:
            cached = self.get(trajectory)
            if cached is not None:
                return cached
            result = evaluate_fn(trajectory)
            self.put(trajectory, result)
            return result
        return wrapper

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def save(self, path: str) -> None:
        """Persist cache to a JSON file."""
        with self._lock:
            data = {k: (v[0].model_dump(), v[1]) for k, v in self._store.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("TrajectoryCache saved to %s", path)

    def load(self, path: str) -> None:
        """Load cache from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            for k, (score_dict, ts) in data.items():
                self._store[k] = (TrajectoryScore(**score_dict), ts)
                self._order.append(k)
        logger.info("TrajectoryCache loaded from %s", path)


# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────

@dataclass
class _PipelineStep:
    name: str
    fn: Callable[[List[Trajectory]], List[Trajectory]]
    retries: int = 0


class EvalPipeline:
    """Fluent, auditable pipeline for trajectory pre-processing and evaluation."""

    def __init__(self) -> None:
        self._steps: List[_PipelineStep] = []
        self._audit_log: List[Dict[str, Any]] = []

    def map(self, name: str, fn: Callable[[Trajectory], Trajectory]) -> "EvalPipeline":
        """Apply fn to each trajectory."""
        self._steps.append(_PipelineStep(name, lambda ts: [fn(t) for t in ts]))
        return self

    def filter(self, name: str, fn: Callable[[Trajectory], bool]) -> "EvalPipeline":
        """Keep only trajectories for which fn returns True."""
        self._steps.append(_PipelineStep(name, lambda ts: [t for t in ts if fn(t)]))
        return self

    def branch(self, condition: Callable[[Trajectory], bool],
               true_fn: Callable[[Trajectory], Trajectory],
               false_fn: Callable[[Trajectory], Trajectory]) -> "EvalPipeline":
        """Route trajectories to true_fn or false_fn based on condition."""
        def _branch(ts: List[Trajectory]) -> List[Trajectory]:
            return [true_fn(t) if condition(t) else false_fn(t) for t in ts]
        self._steps.append(_PipelineStep("branch", _branch))
        return self

    def with_retry(self, step_name: str, retries: int = 3) -> "EvalPipeline":
        """Set retry count for a named step."""
        for step in self._steps:
            if step.name == step_name:
                step.retries = retries
        return self

    def run(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Execute the pipeline synchronously."""
        result = list(trajectories)
        for step in self._steps:
            start = time.time()
            attempt = 0
            last_exc: Optional[Exception] = None
            while attempt <= step.retries:
                try:
                    result = step.fn(result)
                    break
                except Exception as exc:
                    last_exc = exc
                    attempt += 1
                    logger.warning("Pipeline step '%s' attempt %d failed: %s", step.name, attempt, exc)
            else:
                raise EvaluationError(f"Pipeline step '{step.name}' failed after {step.retries + 1} attempts") from last_exc
            elapsed = time.time() - start
            self._audit_log.append({"step": step.name, "output_count": len(result), "elapsed_s": elapsed})
        return result

    async def arun(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Execute the pipeline asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, trajectories)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Return step audit log."""
        return list(self._audit_log)


# ─────────────────────────────────────────────
# 3. VALIDATION & SCHEMA
# ─────────────────────────────────────────────

@dataclass
class TrajectoryRule:
    """A single declarative validation rule for a trajectory."""
    name: str
    check: Callable[[Trajectory], bool]
    message: str


class TrajectoryValidator:
    """Declarative trajectory validator."""

    def __init__(self) -> None:
        self._rules: List[TrajectoryRule] = []

    def add_rule(self, rule: TrajectoryRule) -> "TrajectoryValidator":
        """Register a validation rule."""
        self._rules.append(rule)
        return self

    def validate(self, trajectory: Trajectory) -> List[str]:
        """Return list of violation messages; empty = valid."""
        violations = []
        for rule in self._rules:
            try:
                if not rule.check(trajectory):
                    violations.append(rule.message)
            except Exception as exc:
                violations.append(f"Rule '{rule.name}' error: {exc}")
        return violations

    def is_valid(self, trajectory: Trajectory) -> bool:
        """Return True if all rules pass."""
        return len(self.validate(trajectory)) == 0


class ConfidenceScorer:
    """Heuristic 0–1 confidence score for a TrajectoryScore."""

    def score(self, ts: TrajectoryScore) -> float:
        """Return confidence based on metric variance and step score spread."""
        if not ts.metric_scores:
            return 0.0
        scores = list(ts.metric_scores.values())
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        # Low variance = high confidence
        confidence = max(0.0, 1.0 - variance)
        return round(confidence, 4)


# ─────────────────────────────────────────────
# 4. ASYNC & CONCURRENCY
# ─────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter (sync + async)."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> bool:
        """Synchronously acquire tokens. Returns False if denied."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
        return False

    async def aacquire(self, tokens: float = 1.0) -> bool:
        """Async acquire (non-blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire, tokens)


class CancellationToken:
    """Token for cooperative cancellation of async batch evaluation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Signal cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


async def abatch_evaluate(
    trajectories: List[Trajectory],
    evaluate_fn: Callable[[Trajectory], TrajectoryScore],
    concurrency: int = 8,
    token: Optional[CancellationToken] = None,
) -> List[TrajectoryScore]:
    """Async concurrent trajectory evaluation with optional cancellation."""
    sem = asyncio.Semaphore(concurrency)

    async def _eval(t: Trajectory) -> TrajectoryScore:
        if token and token.is_cancelled:
            raise asyncio.CancelledError()
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, evaluate_fn, t)

    return list(await asyncio.gather(*[_eval(t) for t in trajectories]))


def batch_evaluate(
    trajectories: List[Trajectory],
    evaluate_fn: Callable[[Trajectory], TrajectoryScore],
    max_workers: int = 4,
) -> List[TrajectoryScore]:
    """Synchronous concurrent trajectory evaluation using thread pool."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(evaluate_fn, trajectories))


def evaluate_with_budget(
    trajectories: List[Trajectory],
    evaluate_fn: Callable[[Trajectory], TrajectoryScore],
    budget_seconds: float = 30.0,
) -> List[TrajectoryScore]:
    """Evaluate trajectories within a wall-clock time budget."""
    results: List[TrajectoryScore] = []
    deadline = time.monotonic() + budget_seconds
    for t in trajectories:
        if time.monotonic() >= deadline:
            raise BudgetExceededError(f"Time budget of {budget_seconds}s exceeded after {len(results)} trajectories.")
        results.append(evaluate_fn(t))
    return results


# ─────────────────────────────────────────────
# 5. OBSERVABILITY
# ─────────────────────────────────────────────

class EvaluationProfiler:
    """Tracks timing and memory per evaluation call."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def profile(self, evaluate_fn: Callable[[Trajectory], TrajectoryScore]) -> Callable[[Trajectory], TrajectoryScore]:
        """Decorator that records call timing."""
        def wrapper(trajectory: Trajectory) -> TrajectoryScore:
            start = time.perf_counter()
            result = evaluate_fn(trajectory)
            elapsed = time.perf_counter() - start
            self._records.append({
                "trajectory_id": trajectory.trajectory_id,
                "elapsed_s": round(elapsed, 6),
                "overall_score": result.overall_score,
                "passed": result.passed,
            })
            return result
        return wrapper

    def report(self) -> Dict[str, Any]:
        """Return profiling summary."""
        if not self._records:
            return {"calls": 0}
        elapsed_vals = [r["elapsed_s"] for r in self._records]
        return {
            "calls": len(self._records),
            "mean_elapsed_s": sum(elapsed_vals) / len(elapsed_vals),
            "max_elapsed_s": max(elapsed_vals),
            "min_elapsed_s": min(elapsed_vals),
            "records": self._records,
        }


class DriftDetector:
    """Detect metric drift across evaluation runs."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self._baseline: Optional[Dict[str, float]] = None

    def set_baseline(self, result: EvaluationResult) -> None:
        """Set the baseline metric means."""
        self._baseline = dict(result.metric_means)
        logger.info("DriftDetector baseline set: %s", self._baseline)

    def detect(self, result: EvaluationResult) -> Dict[str, float]:
        """Return dict of metric → drift (absolute diff). Empty if no baseline."""
        if not self._baseline:
            return {}
        drifts: Dict[str, float] = {}
        for metric, val in result.metric_means.items():
            baseline_val = self._baseline.get(metric, val)
            drift = abs(val - baseline_val)
            if drift >= self.threshold:
                drifts[metric] = round(drift, 4)
                logger.warning("Drift detected in '%s': %.4f", metric, drift)
        return drifts


class EvaluationReport:
    """Export EvaluationResult to JSON, CSV, or Markdown."""

    def __init__(self, result: EvaluationResult) -> None:
        self._result = result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self._result.model_dump(), indent=indent, default=str)

    def to_csv(self) -> str:
        """Serialize per-trajectory scores to CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        if self._result.scores:
            metric_names = list(self._result.scores[0].metric_scores.keys())
            writer.writerow(["trajectory_id", "task", "overall_score", "passed"] + metric_names)
            for s in self._result.scores:
                row = [s.trajectory_id, s.task, s.overall_score, s.passed]
                row += [s.metric_scores.get(m, "") for m in metric_names]
                writer.writerow(row)
        return buf.getvalue()

    def to_markdown(self) -> str:
        """Render a Markdown summary table."""
        r = self._result
        lines = [
            "# Evaluation Report",
            f"**Trajectories evaluated:** {r.trajectories_evaluated}",
            f"**Mean overall score:** {r.mean_overall:.3f}",
            f"**Pass rate:** {r.pass_rate:.1%}",
            "",
            "## Metric Means",
            "| Metric | Mean Score |",
            "|--------|-----------|",
        ]
        for metric, val in r.metric_means.items():
            lines.append(f"| {metric} | {val:.3f} |")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 6. STREAMING & STORAGE
# ─────────────────────────────────────────────

def stream_scores(
    trajectories: List[Trajectory],
    evaluate_fn: Callable[[Trajectory], TrajectoryScore],
) -> Generator[TrajectoryScore, None, None]:
    """Generator that yields TrajectoryScore one at a time (streaming)."""
    for t in trajectories:
        yield evaluate_fn(t)


def scores_to_ndjson(
    trajectories: List[Trajectory],
    evaluate_fn: Callable[[Trajectory], TrajectoryScore],
) -> Generator[str, None, None]:
    """Stream NDJSON lines of TrajectoryScore dicts."""
    for score in stream_scores(trajectories, evaluate_fn):
        yield json.dumps(score.model_dump(), default=str)


# ─────────────────────────────────────────────
# 7. DIFF & REGRESSION
# ─────────────────────────────────────────────

@dataclass
class ScoreDiff:
    """Diff between two EvaluationResults."""
    added_trajectories: List[str]
    removed_trajectories: List[str]
    improved: Dict[str, float]  # trajectory_id → score delta
    regressed: Dict[str, float]
    unchanged: List[str]

    def summary(self) -> str:
        return (
            f"Added: {len(self.added_trajectories)}, "
            f"Removed: {len(self.removed_trajectories)}, "
            f"Improved: {len(self.improved)}, "
            f"Regressed: {len(self.regressed)}, "
            f"Unchanged: {len(self.unchanged)}"
        )

    def to_json(self) -> str:
        return json.dumps({
            "added": self.added_trajectories,
            "removed": self.removed_trajectories,
            "improved": self.improved,
            "regressed": self.regressed,
            "unchanged": self.unchanged,
        }, indent=2)


def diff_results(a: EvaluationResult, b: EvaluationResult) -> ScoreDiff:
    """Compute diff between two EvaluationResults."""
    a_map = {s.trajectory_id: s.overall_score for s in a.scores}
    b_map = {s.trajectory_id: s.overall_score for s in b.scores}

    added = [tid for tid in b_map if tid not in a_map]
    removed = [tid for tid in a_map if tid not in b_map]
    improved: Dict[str, float] = {}
    regressed: Dict[str, float] = {}
    unchanged: List[str] = []

    for tid in a_map:
        if tid in b_map:
            delta = b_map[tid] - a_map[tid]
            if delta > 0.01:
                improved[tid] = round(delta, 4)
            elif delta < -0.01:
                regressed[tid] = round(delta, 4)
            else:
                unchanged.append(tid)

    return ScoreDiff(added_trajectories=added, removed_trajectories=removed,
                     improved=improved, regressed=regressed, unchanged=unchanged)


class RegressionTracker:
    """Track score trends and detect regressions across evaluation runs."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._history: deque = deque(maxlen=window)

    def record(self, result: EvaluationResult) -> None:
        """Record an evaluation result."""
        self._history.append(result)

    def trend(self) -> str:
        """Return 'improving', 'declining', or 'stable'."""
        if len(self._history) < 2:
            return "stable"
        scores = [r.mean_overall for r in self._history]
        deltas = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
        mean_delta = sum(deltas) / len(deltas)
        if mean_delta > 0.01:
            return "improving"
        if mean_delta < -0.01:
            return "declining"
        return "stable"

    def latest_regression(self) -> Optional[ScoreDiff]:
        """Return diff between last two runs, or None if fewer than 2."""
        if len(self._history) < 2:
            return None
        a, b = list(self._history)[-2], list(self._history)[-1]
        return diff_results(a, b)


# ─────────────────────────────────────────────
# 8. SECURITY & COST
# ─────────────────────────────────────────────

class AuditLog:
    """Append-only audit log for evaluation events."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log(self, event: str, data: Dict[str, Any]) -> None:
        """Append an audit entry."""
        entry = {"event": event, "timestamp": time.time(), **data}
        with self._lock:
            self._entries.append(entry)

    def to_json(self, indent: int = 2) -> str:
        with self._lock:
            return json.dumps(self._entries, indent=indent, default=str)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)


@dataclass
class CostLedger:
    """Track evaluation cost per run (token counts or arbitrary units)."""
    _entries: List[Dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, trajectory_id: str, tokens: int, cost_usd: float) -> None:
        with self._lock:
            self._entries.append({
                "trajectory_id": trajectory_id,
                "tokens": tokens,
                "cost_usd": cost_usd,
                "timestamp": time.time(),
            })

    def total_cost(self) -> float:
        with self._lock:
            return sum(e["cost_usd"] for e in self._entries)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(e["tokens"] for e in self._entries)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self._entries),
                "total_tokens": self.total_tokens(),
                "total_cost_usd": self.total_cost(),
            }
