"""Strict configuration objects for causal ECG preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cardiosentinel.signal.errors import SignalValidationError


@dataclass(frozen=True)
class HighPassConfig:
    enabled: bool = False
    cutoff_hz: float | None = None
    order: int | None = None

    def __post_init__(self) -> None:
        _validate_ordered_filter("high-pass", self.enabled, self.cutoff_hz, self.order)


@dataclass(frozen=True)
class LowPassConfig:
    enabled: bool = False
    cutoff_hz: float | None = None
    order: int | None = None

    def __post_init__(self) -> None:
        _validate_ordered_filter("low-pass", self.enabled, self.cutoff_hz, self.order)


@dataclass(frozen=True)
class NotchConfig:
    enabled: bool = False
    frequency_hz: float | None = None
    quality_factor: float | None = None

    def __post_init__(self) -> None:
        if self.enabled:
            if (
                self.frequency_hz is None
                or self.frequency_hz <= 0
                or self.quality_factor is None
                or self.quality_factor <= 0
            ):
                raise SignalValidationError(
                    "Enabled notch requires positive frequency_hz and quality_factor."
                )
        elif self.frequency_hz is not None or self.quality_factor is not None:
            raise SignalValidationError(
                "Disabled notch must not retain frequency or quality settings."
            )


def _validate_ordered_filter(
    name: str, enabled: bool, cutoff_hz: float | None, order: int | None
) -> None:
    if enabled:
        if cutoff_hz is None or cutoff_hz <= 0:
            raise SignalValidationError(f"Enabled {name} requires a positive cutoff.")
        if isinstance(order, bool) or order is None or order <= 0:
            raise SignalValidationError(f"Enabled {name} requires a positive order.")
    elif cutoff_hz is not None or order is not None:
        raise SignalValidationError(
            f"Disabled {name} must not retain cutoff or order settings."
        )


@dataclass(frozen=True)
class FilterProfile:
    """A named causal filter chain; raw is an explicit identity profile."""

    name: str = "raw"
    highpass: HighPassConfig = HighPassConfig()
    lowpass: LowPassConfig = LowPassConfig()
    notch: NotchConfig = NotchConfig()

    def __post_init__(self) -> None:
        if self.name not in {"raw", "st_preserving", "custom"}:
            raise SignalValidationError(
                "Filter profile must be raw, st_preserving, or custom."
            )
        enabled = self.highpass.enabled or self.lowpass.enabled or self.notch.enabled
        if self.name == "raw" and enabled:
            raise SignalValidationError("The raw profile cannot enable filters.")
        if (
            self.name == "st_preserving"
            and self.highpass.enabled
            and self.highpass.cutoff_hz is not None
            and self.highpass.cutoff_hz > 0.05
        ):
            raise SignalValidationError(
                "ST-preserving high-pass cutoff cannot exceed 0.05 Hz."
            )

    @property
    def is_raw(self) -> bool:
        return self.name == "raw"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def raw_profile() -> FilterProfile:
    return FilterProfile(name="raw")
