"""Goal completion metric — checks if the final answer addresses the task."""
from __future__ import annotations

import logging
from typing import List

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, StepType, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class GoalCompletionMetric(BaseMetric):
    """Score whether the trajectory reaches a final answer that addresses the task."""

    name = "goal_completion"
    description = "Checks whether the agent produced a relevant final answer."

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        step_scores: List[StepScore] = []
        has_final = any(s.step_type == StepType.FINAL_ANSWER for s in trajectory.steps)
        final_content = trajectory.final_answer or ""

        # Basic keyword overlap between task and final answer
        task_words = set(trajectory.task.lower().split())
        answer_words = set(final_content.lower().split())
        overlap = len(task_words & answer_words) / max(len(task_words), 1)

        score = 0.0
        if has_final and final_content:
            score = min(1.0, 0.5 + overlap * 0.5)
        elif has_final:
            score = 0.3
        else:
            score = 0.0

        for i, step in enumerate(trajectory.steps):
            if step.step_type == StepType.FINAL_ANSWER:
                step_scores.append(StepScore(
                    step_index=i,
                    metric=self.name,
                    score=score,
                    reason=f"Final answer overlap with task: {overlap:.2f}",
                    passed=score >= self.threshold,
                ))

        passed = score >= self.threshold
        logger.debug("GoalCompletion %s: score=%.3f passed=%s", trajectory.trajectory_id, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
