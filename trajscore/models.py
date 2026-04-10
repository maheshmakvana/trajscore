"""Pydantic models for trajscore."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Supported agent step types."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"


class TrajectoryStep(BaseModel):
    """A single step in an agent trajectory."""
    step_index: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Full agent trajectory from input to final answer."""
    trajectory_id: str
    task: str
    steps: List[TrajectoryStep]
    final_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    expected_tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepScore(BaseModel):
    """Score for a single trajectory step."""
    step_index: int
    metric: str
    score: float  # 0.0 – 1.0
    reason: str
    passed: bool


class TrajectoryScore(BaseModel):
    """Aggregate score for a full trajectory."""
    trajectory_id: str
    task: str
    metric_scores: Dict[str, float]
    step_scores: List[StepScore]
    overall_score: float
    passed: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating one or more trajectories."""
    trajectories_evaluated: int
    scores: List[TrajectoryScore]
    mean_overall: float
    pass_rate: float
    metric_means: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
