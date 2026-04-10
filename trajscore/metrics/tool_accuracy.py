"""Tool accuracy metric — checks correct tool usage."""
from __future__ import annotations

import logging
from typing import List, Optional, Set

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, StepType, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class ToolAccuracyMetric(BaseMetric):
    """Score how accurately the agent used the expected tools."""

    name = "tool_accuracy"
    description = "Measures correctness of tool selection against expected tools."

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        step_scores: List[StepScore] = []
        expected: Optional[Set[str]] = (
            set(trajectory.expected_tools) if trajectory.expected_tools else None
        )

        tool_steps = [s for s in trajectory.steps if s.step_type == StepType.TOOL_CALL]

        if not tool_steps:
            # No tools used — full score if none expected
            score = 1.0 if not expected else 0.0
            return self._make_score(trajectory, score, [], score >= self.threshold)

        used_tools: Set[str] = {s.tool_name for s in tool_steps if s.tool_name}

        if expected is None:
            # No expectation — reward any tool use
            score = 1.0
        else:
            correct = used_tools & expected
            false_positive = used_tools - expected
            false_negative = expected - used_tools
            precision = len(correct) / max(len(used_tools), 1)
            recall = len(correct) / max(len(expected), 1)
            score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        for i, step in enumerate(tool_steps):
            step_idx = trajectory.steps.index(step)
            correct_tool = expected is None or (step.tool_name in expected if step.tool_name else False)
            step_scores.append(StepScore(
                step_index=step_idx,
                metric=self.name,
                score=1.0 if correct_tool else 0.0,
                reason=f"Tool '{step.tool_name}' {'in' if correct_tool else 'not in'} expected set",
                passed=correct_tool,
            ))

        passed = score >= self.threshold
        logger.debug("ToolAccuracy %s: score=%.3f passed=%s", trajectory.trajectory_id, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
