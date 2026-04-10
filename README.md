# trajscore

**Production-grade agentic trajectory evaluation for multi-step AI agents.**

Score any AI agent run on 6 built-in metrics, detect regressions, stream results, and integrate into CI/CD — with zero vendor lock-in.

```bash
pip install trajscore
```

---

## Why trajscore?

In 2026, every team building agentic AI faces the same problem: **you can't improve what you can't measure.** Agents fail in subtle ways — they loop, misuse tools, hallucinate answers unsupported by observations, or take twice as many steps as needed. No single library evaluated full multi-step trajectories with structured, auditable metrics.

`trajscore` fixes this.

---

## Quickstart

```python
from trajscore import (
    Trajectory, TrajectoryStep, StepType,
    TrajectoryEvaluator,
)

trajectory = Trajectory(
    trajectory_id="run-001",
    task="What is the capital of France?",
    steps=[
        TrajectoryStep(step_index=0, step_type=StepType.THOUGHT,
                       content="I should look this up."),
        TrajectoryStep(step_index=1, step_type=StepType.TOOL_CALL,
                       content="search", tool_name="search",
                       tool_args={"query": "capital of France"}),
        TrajectoryStep(step_index=2, step_type=StepType.OBSERVATION,
                       content="Paris is the capital of France."),
        TrajectoryStep(step_index=3, step_type=StepType.FINAL_ANSWER,
                       content="The capital of France is Paris."),
    ],
    final_answer="The capital of France is Paris.",
    expected_tools=["search"],
)

evaluator = TrajectoryEvaluator()
score = evaluator.evaluate(trajectory)

print(f"Overall: {score.overall_score:.3f}  Passed: {score.passed}")
print(score.metric_scores)
```

---

## Built-in Metrics

| Metric | Description |
|--------|-------------|
| `goal_completion` | Did the agent produce a relevant final answer? |
| `tool_accuracy` | Did it use the right tools? (F1 vs expected_tools) |
| `step_efficiency` | Did it reach the goal without unnecessary steps? |
| `reasoning_coherence` | Do thoughts lead logically to actions? |
| `loop_detection` | Did the agent repeat actions or thoughts? |
| `answer_faithfulness` | Is the final answer grounded in observations? |

---

## Batch & Async Evaluation

```python
from trajscore import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()

# Synchronous batch
result = evaluator.evaluate_batch(trajectories, max_workers=8)

# Async batch
import asyncio
result = asyncio.run(evaluator.aevaluate_batch(trajectories))

print(f"Pass rate: {result.pass_rate:.1%}")
print(f"Mean score: {result.mean_overall:.3f}")
```

---

## Advanced Features

### Caching (LRU + TTL + SHA-256)

```python
from trajscore.advanced import TrajectoryCache

cache = TrajectoryCache(max_size=512, ttl=600)
memoized_eval = cache.memoize(evaluator.evaluate)
score = memoized_eval(trajectory)    # cached on second call
print(cache.stats())
```

### Evaluation Pipeline

```python
from trajscore.advanced import EvalPipeline

pipeline = (
    EvalPipeline()
    .filter("non_empty", lambda t: len(t.steps) > 0)
    .map("tag_metadata", lambda t: t)
    .with_retry("tag_metadata", retries=2)
)
cleaned = pipeline.run(trajectories)
print(pipeline.audit_log)

# Async
import asyncio
cleaned = asyncio.run(pipeline.arun(trajectories))
```

### Declarative Validation

```python
from trajscore.advanced import TrajectoryValidator, TrajectoryRule

validator = (
    TrajectoryValidator()
    .add_rule(TrajectoryRule("has_steps", lambda t: len(t.steps) > 0, "Need steps"))
    .add_rule(TrajectoryRule("has_task", lambda t: bool(t.task), "Need task"))
)
violations = validator.validate(trajectory)
```

### Rate Limiter (sync + async)

```python
from trajscore.advanced import RateLimiter

limiter = RateLimiter(rate=10, capacity=10)  # 10 evals/s
if limiter.acquire():
    score = evaluator.evaluate(trajectory)
```

### Budget-Controlled Evaluation

```python
from trajscore.advanced import evaluate_with_budget
scores = evaluate_with_budget(trajectories, evaluator.evaluate, budget_seconds=5.0)
```

### Streaming Results

```python
from trajscore.advanced import stream_scores, scores_to_ndjson

for score in stream_scores(trajectories, evaluator.evaluate):
    print(score.trajectory_id, score.overall_score)

# NDJSON stream
for line in scores_to_ndjson(trajectories, evaluator.evaluate):
    print(line)
```

### Diff & Regression Tracking

```python
from trajscore.advanced import diff_results, RegressionTracker

tracker = RegressionTracker(window=10)
tracker.record(result_v1)
tracker.record(result_v2)
print(tracker.trend())          # "improving" / "declining" / "stable"
diff = tracker.latest_regression()
print(diff.summary())
print(diff.to_json())
```

### Observability

```python
from trajscore.advanced import EvaluationProfiler, DriftDetector, EvaluationReport

profiler = EvaluationProfiler()
scored = profiler.profile(evaluator.evaluate)(trajectory)
print(profiler.report())

detector = DriftDetector(threshold=0.05)
detector.set_baseline(result_v1)
print(detector.detect(result_v2))

report = EvaluationReport(result)
print(report.to_json())
print(report.to_csv())
print(report.to_markdown())
```

### Audit Log & Cost Ledger

```python
from trajscore.advanced import AuditLog, CostLedger

log = AuditLog()
log.log("eval_start", {"run_id": "ci-42"})

ledger = CostLedger()
ledger.record("t1", tokens=1200, cost_usd=0.024)
print(ledger.summary())
```

---

## Live Trajectory Watcher

```python
from trajscore import TrajectoryWatcher, TrajectoryStep, StepType

watcher = TrajectoryWatcher(
    trajectory_id="live-001",
    task="Summarize the paper",
    on_step=lambda step, idx: print(f"Step {idx}: {step.step_type}"),
)

watcher.add_step(TrajectoryStep(step_index=0, step_type=StepType.THOUGHT, content="Reading..."))
trajectory = watcher.finish("Summary complete.")
score = evaluator.evaluate(trajectory)
```

---

## Installation

```bash
pip install trajscore
```

Python 3.8+ · No external dependencies (stdlib + pydantic)

---

## Changelog

### v1.1.3 (2026-04-10)
- Added Changelog section to README for release traceability
- SEO improvements: trajectory evaluation, agentic benchmark, multi-step AI scoring

### v1.1.0
- Renamed module to `trajscore` to match PyPI package name
- Fixed all internal import references

### v1.0.0
- Initial release: 6 built-in agentic metrics, regression detection, CI/CD integration, streaming

## License

MIT

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository on [GitHub](https://github.com/maheshmakvana/trajscore)
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

Please open an issue first for major changes to discuss the approach.

## Author

**Mahesh Makvana** — [GitHub](https://github.com/maheshmakvana) · [PyPI](https://pypi.org/user/maheshmakvana/)

MIT License
