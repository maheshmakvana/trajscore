"""
trajscore — Agentic Multi-Step Trajectory Evaluation.

Evaluate any AI agent trajectory with production-grade metrics:
goal completion, tool accuracy, step efficiency, reasoning coherence,
loop detection, and answer faithfulness.
"""
from trajscore.models import (
    StepType,
    TrajectoryStep,
    Trajectory,
    StepScore,
    TrajectoryScore,
    EvaluationResult,
)
from trajscore.evaluator import TrajectoryEvaluator
from trajscore.watcher import TrajectoryWatcher
from trajscore.exceptions import (
    TrajscoreError,
    TrajectoryValidationError,
    EvaluationError,
    MetricNotFoundError,
    BudgetExceededError,
    SchemaViolationError,
)
from trajscore.metrics import (
    GoalCompletionMetric,
    ToolAccuracyMetric,
    StepEfficiencyMetric,
    ReasoningCoherenceMetric,
    LoopDetectionMetric,
    AnswerFaithfulnessMetric,
)
from trajscore.advanced import (
    TrajectoryCache,
    EvalPipeline,
    TrajectoryRule,
    TrajectoryValidator,
    ConfidenceScorer,
    RateLimiter,
    CancellationToken,
    abatch_evaluate,
    batch_evaluate,
    evaluate_with_budget,
    EvaluationProfiler,
    DriftDetector,
    EvaluationReport,
    stream_scores,
    scores_to_ndjson,
    ScoreDiff,
    diff_results,
    RegressionTracker,
    AuditLog,
    CostLedger,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "StepType", "TrajectoryStep", "Trajectory", "StepScore",
    "TrajectoryScore", "EvaluationResult",
    # Core
    "TrajectoryEvaluator", "TrajectoryWatcher",
    # Exceptions
    "TrajscoreError", "TrajectoryValidationError", "EvaluationError",
    "MetricNotFoundError", "BudgetExceededError", "SchemaViolationError",
    # Metrics
    "GoalCompletionMetric", "ToolAccuracyMetric", "StepEfficiencyMetric",
    "ReasoningCoherenceMetric", "LoopDetectionMetric", "AnswerFaithfulnessMetric",
    # Advanced
    "TrajectoryCache", "EvalPipeline", "TrajectoryRule", "TrajectoryValidator",
    "ConfidenceScorer", "RateLimiter", "CancellationToken",
    "abatch_evaluate", "batch_evaluate", "evaluate_with_budget",
    "EvaluationProfiler", "DriftDetector", "EvaluationReport",
    "stream_scores", "scores_to_ndjson", "ScoreDiff", "diff_results",
    "RegressionTracker", "AuditLog", "CostLedger",
]
