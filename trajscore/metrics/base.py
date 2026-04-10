"""Base class for all trajectory metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from trajscore.models import StepScore, Trajectory, TrajectoryScore


class BaseMetric(ABC):
    """Abstract base for trajectory evaluation metrics."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        """Score a full trajectory and return a TrajectoryScore."""

    def _make_score(
        self,
        trajectory: Trajectory,
        metric_score: float,
        step_scores: List[StepScore],
        passed: bool,
    ) -> TrajectoryScore:
        return TrajectoryScore(
            trajectory_id=trajectory.trajectory_id,
            task=trajectory.task,
            metric_scores={self.name: metric_score},
            step_scores=step_scores,
            overall_score=metric_score,
            passed=passed,
        )
