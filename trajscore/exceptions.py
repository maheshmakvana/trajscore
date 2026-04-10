"""Exceptions for trajscore."""


class AgentEvalError(Exception):
    """Base exception for trajscore."""


class TrajectoryValidationError(AgentEvalError):
    """Raised when a trajectory is malformed or invalid."""


class EvaluationError(AgentEvalError):
    """Raised when an evaluation step fails."""


class MetricNotFoundError(AgentEvalError):
    """Raised when a requested metric is not registered."""


class BudgetExceededError(AgentEvalError):
    """Raised when evaluation exceeds cost/token budget."""


class SchemaViolationError(AgentEvalError):
    """Raised when a trajectory step violates schema constraints."""
