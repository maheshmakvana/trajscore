"""Real-time trajectory watcher for streaming agent evaluation."""
from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

from trajscore.exceptions import TrajectoryValidationError
from trajscore.models import Trajectory, TrajectoryStep

logger = logging.getLogger(__name__)

OnStepCallback = Callable[[TrajectoryStep, int], None]
OnCompleteCallback = Callable[[Trajectory], None]


class TrajectoryWatcher:
    """Thread-safe live trajectory builder with hook callbacks."""

    def __init__(
        self,
        trajectory_id: str,
        task: str,
        on_step: Optional[OnStepCallback] = None,
        on_complete: Optional[OnCompleteCallback] = None,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.task = task
        self._on_step = on_step
        self._on_complete = on_complete
        self._steps: List[TrajectoryStep] = []
        self._lock = threading.Lock()
        self._complete = False

    def add_step(self, step: TrajectoryStep) -> None:
        """Append a step to the live trajectory."""
        if self._complete:
            raise TrajectoryValidationError("Cannot add steps to a completed trajectory.")
        with self._lock:
            self._steps.append(step)
        if self._on_step:
            self._on_step(step, len(self._steps) - 1)
        logger.debug("Step added: index=%d type=%s", step.step_index, step.step_type)

    def finish(self, final_answer: Optional[str] = None) -> Trajectory:
        """Finalize the trajectory and trigger the on_complete callback."""
        with self._lock:
            self._complete = True
            traj = Trajectory(
                trajectory_id=self.trajectory_id,
                task=self.task,
                steps=list(self._steps),
                final_answer=final_answer,
            )
        if self._on_complete:
            self._on_complete(traj)
        return traj

    @property
    def steps(self) -> List[TrajectoryStep]:
        """Current steps snapshot."""
        with self._lock:
            return list(self._steps)
