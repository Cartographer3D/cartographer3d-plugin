from __future__ import annotations


class ProbeTriggerError(RuntimeError):
    """Raised when probe triggers unexpectedly (e.g., before movement)."""
