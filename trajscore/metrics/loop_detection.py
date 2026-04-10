"""Loop detection metric — identifies repetitive agent behavior."""
from __future__ import annotations

import logging
from collections import Counter
from typing import List

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, StepType, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class LoopDetectionMetric(BaseMetric):
    """Penalize trajectories where the agent repeats the same actions/tools."""

    name = "loop_detection"
    description = "Detects and penalizes repetitive tool calls or thought patterns."

    def __init__(self, repeat_threshold: int = 2, threshold: float = 0.6) -> None:
        self.repeat_threshold = repeat_threshold
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        step_scores: List[StepScore] = []
        tool_calls = [s.tool_name for s in trajectory.steps if s.step_type == StepType.TOOL_CALL and s.tool_name]
        thoughts = [s.content[:50] for s in trajectory.steps if s.step_type == StepType.THOUGHT]

        tool_counts = Counter(tool_calls)
        thought_counts = Counter(thoughts)

        tool_loops = sum(1 for c in tool_counts.values() if c >= self.repeat_threshold)
        thought_loops = sum(1 for c in thought_counts.values() if c >= self.repeat_threshold)
        total_loops = tool_loops + thought_loops

        # Score: 1.0 for no loops, decays per loop
        score = max(0.0, 1.0 - total_loops * 0.25)

        for i, step in enumerate(trajectory.steps):
            if step.step_type == StepType.TOOL_CALL and step.tool_name:
                looping = tool_counts[step.tool_name] >= self.repeat_threshold
                step_scores.append(StepScore(
                    step_index=i,
                    metric=self.name,
                    score=0.0 if looping else 1.0,
                    reason=f"Tool '{step.tool_name}' used {tool_counts[step.tool_name]}x",
                    passed=not looping,
                ))

        passed = score >= self.threshold
        logger.debug("LoopDetection %s: loops=%d score=%.3f passed=%s", trajectory.trajectory_id, total_loops, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
