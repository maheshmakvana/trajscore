"""Answer faithfulness metric — checks answer grounding in observations."""
from __future__ import annotations

import logging
from typing import List

from trajscore.metrics.base import BaseMetric
from trajscore.models import StepScore, StepType, Trajectory, TrajectoryScore

logger = logging.getLogger(__name__)


class AnswerFaithfulnessMetric(BaseMetric):
    """Score whether the final answer is grounded in trajectory observations."""

    name = "answer_faithfulness"
    description = "Measures how well the final answer is supported by observations."

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score_trajectory(self, trajectory: Trajectory) -> TrajectoryScore:
        step_scores: List[StepScore] = []
        final = trajectory.final_answer or ""

        observation_text = " ".join(
            s.content for s in trajectory.steps if s.step_type == StepType.OBSERVATION
        )

        if not final or not observation_text:
            score = 0.5  # Neutral when no observations or no answer
        else:
            answer_words = set(final.lower().split())
            obs_words = set(observation_text.lower().split())
            overlap = len(answer_words & obs_words) / max(len(answer_words), 1)
            score = min(1.0, overlap + 0.2)  # Small base credit

        final_steps = [s for s in trajectory.steps if s.step_type == StepType.FINAL_ANSWER]
        for step in final_steps:
            idx = trajectory.steps.index(step)
            step_scores.append(StepScore(
                step_index=idx,
                metric=self.name,
                score=score,
                reason=f"Answer-observation overlap score: {score:.2f}",
                passed=score >= self.threshold,
            ))

        passed = score >= self.threshold
        logger.debug("AnswerFaithfulness %s: score=%.3f passed=%s", trajectory.trajectory_id, score, passed)
        return self._make_score(trajectory, score, step_scores, passed)
