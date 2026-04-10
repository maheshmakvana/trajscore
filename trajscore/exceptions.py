"""Exceptions for trajscore."""


class TrajscoreError(Exception):
    """Base exception for trajscore."""


class TrajectoryValidationError(TrajscoreError):
    """Raised when a trajectory is malformed or invalid."""


class EvaluationError(TrajscoreError):
    """Raised when an evaluation step fails."""


class MetricNotFoundError(TrajscoreError):
    """Raised when a requested metric is not registered."""


class BudgetExceededError(TrajscoreError):
    """Raised when evaluation exceeds cost/token budget."""


class SchemaViolationError(TrajscoreError):
    """Raised when a trajectory step violates schema constraints."""
