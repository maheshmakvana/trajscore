"""Step efficiency metric — penalizes unnecessary steps."""
from __future__ import annotations

import logging
from typing import List

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class StepEfficiencyMetric(BaseMetric):
    """Score how efficiently the agent reached the goal (fewer steps = better)."""

    name = "step_efficiency"
    description = "Penalizes trajectories with more steps than the expected minimum."

    def __init__(self, expected_steps: int = 5, threshold: float = 0.6) -> None:
        self.expected_steps = expected_steps
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        n = len(trajectory.steps)
        step_scores: List[StepScore] = []

        if n == 0:
            score = 0.0
        elif n <= self.expected_steps:
            score = 1.0
        else:
            # Decay: each extra step costs 10%
            excess = n - self.expected_steps
            score = max(0.0, 1.0 - excess * 0.1)

        for i, _ in enumerate(trajectory.steps):
            step_scores.append(StepScore(
                step_index=i,
                metric=self.name,
                score=score,
                reason=f"{n} steps vs expected {self.expected_steps}",
                passed=score >= self.threshold,
            ))

        passed = score >= self.threshold
        logger.debug("StepEfficiency %s: n=%d score=%.3f passed=%s", trajectory.trajectory_id, n, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
