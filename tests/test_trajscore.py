"""Tests for agenteval core and advanced features."""
import asyncio
import pytest
from trajscore.models import StepType, TrajectoryStep, Trajectory
from trajscore.evaluator import TrajectoryEvaluator
from trajscore.watcher import TrajectoryWatcher
from trajscore.exceptions import EvaluationError, MetricNotFoundError, BudgetExceededError
from trajscore.advanced import (
    TrajectoryCache, EvalPipeline, TrajectoryValidator, TrajectoryRule,
    ConfidenceScorer, RateLimiter, CancellationToken, abatch_evaluate,
    batch_evaluate, evaluate_with_budget, EvaluationProfiler, DriftDetector,
    EvaluationReport, stream_scores, scores_to_ndjson, diff_results,
    RegressionTracker, AuditLog, CostLedger,
)


def make_trajectory(tid: str = "t1", task: str = "Find the capital of France") -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        task=task,
        steps=[
            TrajectoryStep(step_index=0, step_type=StepType.THOUGHT, content="I need to look up the capital of France."),
            TrajectoryStep(step_index=1, step_type=StepType.TOOL_CALL, content="search", tool_name="search", tool_args={"query": "capital France"}),
            TrajectoryStep(step_index=2, step_type=StepType.OBSERVATION, content="Paris is the capital of France."),
            TrajectoryStep(step_index=3, step_type=StepType.FINAL_ANSWER, content="The capital of France is Paris."),
        ],
        final_answer="The capital of France is Paris.",
        expected_tools=["search"],
    )


# ──────────────── Core ────────────────

def test_evaluator_single():
    ev = TrajectoryEvaluator()
    t = make_trajectory()
    score = ev.evaluate(t)
    assert 0.0 <= score.overall_score <= 1.0
    assert score.trajectory_id == "t1"
    assert isinstance(score.passed, bool)

def test_evaluator_batch():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"t{i}") for i in range(5)]
    result = ev.evaluate_batch(trajs)
    assert result.trajectories_evaluated == 5
    assert 0.0 <= result.mean_overall <= 1.0
    assert 0.0 <= result.pass_rate <= 1.0

def test_metric_not_found():
    ev = TrajectoryEvaluator(metrics=["nonexistent_metric"])
    with pytest.raises(MetricNotFoundError):
        ev.evaluate(make_trajectory())

def test_watcher_finish():
    watcher = TrajectoryWatcher("w1", "test task")
    step = TrajectoryStep(step_index=0, step_type=StepType.THOUGHT, content="thinking...")
    watcher.add_step(step)
    traj = watcher.finish("done")
    assert traj.trajectory_id == "w1"
    assert len(traj.steps) == 1
    assert traj.final_answer == "done"

def test_selected_metrics():
    ev = TrajectoryEvaluator(metrics=["goal_completion", "tool_accuracy"])
    score = ev.evaluate(make_trajectory())
    assert "goal_completion" in score.metric_scores
    assert "tool_accuracy" in score.metric_scores


# ──────────────── Async ────────────────

def test_async_evaluate():
    ev = TrajectoryEvaluator()
    t = make_trajectory()
    score = asyncio.run(ev.aevaluate(t))
    assert 0.0 <= score.overall_score <= 1.0

def test_async_batch():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"a{i}") for i in range(3)]
    result = asyncio.run(ev.aevaluate_batch(trajs))
    assert result.trajectories_evaluated == 3


# ──────────────── Cache ────────────────

def test_trajectory_cache_hit_miss():
    cache = TrajectoryCache(max_size=10, ttl=60)
    t = make_trajectory()
    ev = TrajectoryEvaluator()
    score = ev.evaluate(t)
    assert cache.get(t) is None
    cache.put(t, score)
    assert cache.get(t) is not None
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1

def test_cache_memoize():
    cache = TrajectoryCache()
    ev = TrajectoryEvaluator()
    memoized = cache.memoize(ev.evaluate)
    t = make_trajectory()
    r1 = memoized(t)
    r2 = memoized(t)
    assert r1.overall_score == r2.overall_score
    assert cache.stats()["hits"] >= 1


# ──────────────── Pipeline ────────────────

def test_eval_pipeline_map_filter():
    pipe = EvalPipeline()
    pipe.map("tag", lambda t: t)
    pipe.filter("has_steps", lambda t: len(t.steps) > 0)
    trajs = [make_trajectory(f"p{i}") for i in range(3)]
    result = pipe.run(trajs)
    assert len(result) == 3
    assert len(pipe.audit_log) == 2

def test_eval_pipeline_arun():
    pipe = EvalPipeline()
    pipe.map("identity", lambda t: t)
    trajs = [make_trajectory()]
    result = asyncio.run(pipe.arun(trajs))
    assert len(result) == 1


# ──────────────── Validation ────────────────

def test_trajectory_validator():
    v = TrajectoryValidator()
    v.add_rule(TrajectoryRule("has_steps", lambda t: len(t.steps) > 0, "Must have steps"))
    v.add_rule(TrajectoryRule("has_task", lambda t: bool(t.task), "Must have task"))
    t = make_trajectory()
    assert v.is_valid(t)

def test_trajectory_validator_violation():
    v = TrajectoryValidator()
    v.add_rule(TrajectoryRule("no_steps", lambda t: len(t.steps) == 0, "Should have no steps"))
    t = make_trajectory()
    violations = v.validate(t)
    assert len(violations) == 1

def test_confidence_scorer():
    ev = TrajectoryEvaluator()
    score = ev.evaluate(make_trajectory())
    cs = ConfidenceScorer()
    conf = cs.score(score)
    assert 0.0 <= conf <= 1.0


# ──────────────── Rate Limiter ────────────────

def test_rate_limiter_sync():
    rl = RateLimiter(rate=100, capacity=10)
    assert rl.acquire(5) is True
    assert rl.acquire(5) is True

def test_rate_limiter_async():
    rl = RateLimiter(rate=100, capacity=10)
    result = asyncio.run(rl.aacquire(3))
    assert result is True


# ──────────────── Batch & Budget ────────────────

def test_batch_evaluate():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"b{i}") for i in range(4)]
    scores = batch_evaluate(trajs, ev.evaluate)
    assert len(scores) == 4

def test_abatch_evaluate():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"ab{i}") for i in range(3)]
    scores = asyncio.run(abatch_evaluate(trajs, ev.evaluate))
    assert len(scores) == 3

def test_budget_exceeded():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"bud{i}") for i in range(100)]
    with pytest.raises(BudgetExceededError):
        evaluate_with_budget(trajs, ev.evaluate, budget_seconds=0.000001)


# ──────────────── Observability ────────────────

def test_evaluation_profiler():
    profiler = EvaluationProfiler()
    ev = TrajectoryEvaluator()
    profiled = profiler.profile(ev.evaluate)
    profiled(make_trajectory())
    report = profiler.report()
    assert report["calls"] == 1
    assert "mean_elapsed_s" in report

def test_drift_detector():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"d{i}") for i in range(3)]
    r1 = ev.evaluate_batch(trajs)
    dd = DriftDetector(threshold=0.0)
    dd.set_baseline(r1)
    drifts = dd.detect(r1)
    assert isinstance(drifts, dict)

def test_evaluation_report_formats():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"r{i}") for i in range(2)]
    result = ev.evaluate_batch(trajs)
    report = EvaluationReport(result)
    assert '"trajectories_evaluated"' in report.to_json()
    assert "trajectory_id" in report.to_csv()
    assert "# Evaluation Report" in report.to_markdown()


# ──────────────── Streaming ────────────────

def test_stream_scores():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"s{i}") for i in range(3)]
    scores = list(stream_scores(trajs, ev.evaluate))
    assert len(scores) == 3

def test_scores_to_ndjson():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"n{i}") for i in range(2)]
    lines = list(scores_to_ndjson(trajs, ev.evaluate))
    assert len(lines) == 2
    import json
    for line in lines:
        obj = json.loads(line)
        assert "overall_score" in obj


# ──────────────── Diff & Regression ────────────────

def test_diff_results():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"dr{i}") for i in range(3)]
    r1 = ev.evaluate_batch(trajs)
    r2 = ev.evaluate_batch(trajs)
    diff = diff_results(r1, r2)
    assert diff.summary() is not None
    assert isinstance(diff.to_json(), str)

def test_regression_tracker():
    ev = TrajectoryEvaluator()
    trajs = [make_trajectory(f"rt{i}") for i in range(2)]
    tracker = RegressionTracker(window=5)
    r1 = ev.evaluate_batch(trajs)
    r2 = ev.evaluate_batch(trajs)
    tracker.record(r1)
    tracker.record(r2)
    assert tracker.trend() in ("improving", "declining", "stable")
    diff = tracker.latest_regression()
    assert diff is not None


# ──────────────── Security & Cost ────────────────

def test_audit_log():
    log = AuditLog()
    log.log("eval_start", {"trajectory_id": "t1"})
    assert len(log.entries) == 1
    assert "eval_start" in log.to_json()

def test_cost_ledger():
    ledger = CostLedger()
    ledger.record("t1", tokens=500, cost_usd=0.01)
    ledger.record("t2", tokens=300, cost_usd=0.006)
    s = ledger.summary()
    assert s["calls"] == 2
    assert s["total_tokens"] == 800
    assert abs(s["total_cost_usd"] - 0.016) < 1e-9
