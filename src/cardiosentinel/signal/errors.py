"""Lightweight signal-layer exceptions without numerical dependencies."""


class SignalValidationError(ValueError):
    """Raised when physical waveform or signal-processing metadata is invalid."""
