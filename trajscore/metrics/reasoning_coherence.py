"""Reasoning coherence metric — checks thought → action flow."""
from __future__ import annotations

import logging
from typing import List

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, StepType, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class ReasoningCoherenceMetric(BaseMetric):
    """Score how well the thought steps lead logically to actions."""

    name = "reasoning_coherence"
    description = "Measures thought→action coherence and reasoning completeness."

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        step_scores: List[StepScore] = []
        steps = trajectory.steps
        n = len(steps)
        total_score = 0.0
        scored = 0

        for i, step in enumerate(steps):
            if step.step_type == StepType.THOUGHT:
                # Check if next step is action/tool_call (coherent)
                next_step = steps[i + 1] if i + 1 < n else None
                is_coherent = next_step is not None and next_step.step_type in (
                    StepType.ACTION, StepType.TOOL_CALL, StepType.FINAL_ANSWER
                )
                # Length heuristic: too short = low quality
                content_score = min(1.0, len(step.content) / 80)
                s = (0.6 if is_coherent else 0.2) * content_score + 0.4 * content_score
                s = min(1.0, s)
                step_scores.append(StepScore(
                    step_index=i,
                    metric=self.name,
                    score=s,
                    reason=f"Thought coherent={is_coherent}, length={len(step.content)}",
                    passed=s >= self.threshold,
                ))
                total_score += s
                scored += 1

        score = (total_score / scored) if scored > 0 else 0.5
        passed = score >= self.threshold
        logger.debug("ReasoningCoherence %s: score=%.3f passed=%s", trajectory.trajectory_id, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
