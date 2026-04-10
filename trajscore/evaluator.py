"""Core trajectory evaluator."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Type

from trajscore.exceptions import EvaluationError, MetricNotFoundError
from trajscore.metrics.base import BaseMetric
from trajscore.metrics.goal_completion import GoalCompletionMetric
from trajscore.metrics.tool_accuracy import ToolAccuracyMetric
from trajscore.metrics.step_efficiency import StepEfficiencyMetric
from trajscore.metrics.reasoning_coherence import ReasoningCoherenceMetric
from trajscore.metrics.loop_detection import LoopDetectionMetric
from trajscore.metrics.answer_faithfulness import AnswerFaithfulnessMetric
from trajscore.models import EvaluationResult, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)

_BUILTIN_METRICS: Dict[str, BaseMetric] = {
    "goal_completion": GoalCompletionMetric(),
    "tool_accuracy": ToolAccuracyMetric(),
    "step_efficiency": StepEfficiencyMetric(),
    "reasoning_coherence": ReasoningCoherenceMetric(),
    "loop_detection": LoopDetectionMetric(),
    "answer_faithfulness": AnswerFaithfulnessMetric(),
}


class TrajectoryEvaluator:
    """Evaluate agent trajectories using one or more metrics."""

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        self._metrics: Dict[str, BaseMetric] = dict(_BUILTIN_METRICS)
        self._active_metric_names: List[str] = metrics or list(self._metrics.keys())

    def register_metric(self, metric: BaseMetric) -> None:
        """Register a custom metric."""
        self._metrics[metric.name] = metric
        logger.info("Registered metric: %s", metric.name)

    def evaluate(self, trajectory: Trajectory) -> TrajectoryScore:
        """Evaluate a single trajectory and return aggregated score."""
        all_step_scores = []
        metric_scores: Dict[str, float] = {}

        for name in self._active_metric_names:
            if name not in self._metrics:
                raise MetricNotFoundError(f"Metric '{name}' not found")
            try:
                result = self._metrics[name].score_trajectory(trajectory)
                metric_scores[name] = result.overall_score
                all_step_scores.extend(result.step_scores)
            except Exception as exc:
                logger.error("Metric '%s' failed: %s", name, exc)
                raise EvaluationError(f"Metric '{name}' raised: {exc}") from exc

        overall = sum(metric_scores.values()) / max(len(metric_scores), 1)
        passed = overall >= 0.5
        return TrajectoryScore(
            trajectory_id=trajectory.trajectory_id,
            task=trajectory.task,
            metric_scores=metric_scores,
            step_scores=all_step_scores,
            overall_score=overall,
            passed=passed,
        )

    def evaluate_batch(self, trajectories: List[Trajectory], max_workers: int = 4) -> EvaluationResult:
        """Evaluate multiple trajectories concurrently."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            scores: List[TrajectoryScore] = list(pool.map(self.evaluate, trajectories))
        return self._aggregate(scores)

    async def aevaluate(self, trajectory: Trajectory) -> TrajectoryScore:
        """Async single-trajectory evaluation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, trajectory)

    async def aevaluate_batch(self, trajectories: List[Trajectory]) -> EvaluationResult:
        """Async batch evaluation."""
        scores = await asyncio.gather(*[self.aevaluate(t) for t in trajectories])
        return self._aggregate(list(scores))

    def _aggregate(self, scores: List[TrajectoryScore]) -> EvaluationResult:
        if not scores:
            return EvaluationResult(
                trajectories_evaluated=0,
                scores=[],
                mean_overall=0.0,
                pass_rate=0.0,
                metric_means={},
            )
        mean_overall = sum(s.overall_score for s in scores) / len(scores)
        pass_rate = sum(1 for s in scores if s.passed) / len(scores)

        all_metric_names = set()
        for s in scores:
            all_metric_names.update(s.metric_scores.keys())

        metric_means: Dict[str, float] = {}
        for m in all_metric_names:
            vals = [s.metric_scores[m] for s in scores if m in s.metric_scores]
            metric_means[m] = sum(vals) / max(len(vals), 1)

        return EvaluationResult(
            trajectories_evaluated=len(scores),
            scores=scores,
            mean_overall=mean_overall,
            pass_rate=pass_rate,
            metric_means=metric_means,
        )
