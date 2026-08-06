"""Typed, validated configuration loading for reproducible research runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


class ConfigValidationError(ValueError):
    """Raised when a configuration violates a Phase 0 research invariant."""


@dataclass(frozen=True)
class ProjectConfig:
    """Project identity and scope metadata."""

    name: str
    profile: str
    research_only: bool


@dataclass(frozen=True)
class PathsConfig:
    """External locations for data and generated artefacts."""

    dataset_root: Path
    output_root: Path


@dataclass(frozen=True)
class DataConfig:
    """Dataset metadata requirements before signal processing."""

    sampling_frequency_hz: float | None
    require_record_metadata: bool
    subject_id_field: str


@dataclass(frozen=True)
class WindowConfig:
    """Windowing parameters, intentionally unset until protocol review."""

    length_seconds: float | None
    overlap_seconds: float | None


@dataclass(frozen=True)
class SplitConfig:
    """Subject-wise data partition ratios."""

    strategy: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class SignalPreprocessingConfig:
    """Future signal-preprocessing selection without an implementation."""

    enabled: bool
    filter_profile: str | None


@dataclass(frozen=True)
class TrainingConfig:
    """Future training state, disabled in Phase 0."""

    enabled: bool


@dataclass(frozen=True)
class CalibrationConfig:
    """Future calibration state and routing safeguard."""

    method: str | None
    routing_requires_calibrated_confidence: bool


@dataclass(frozen=True)
class PersonalizationConfig:
    """Future patient-adaptation state, disabled in Phase 0."""

    enabled: bool


@dataclass(frozen=True)
class EpisodeConfig:
    """Future temporal episode-detection state, disabled in Phase 0."""

    enabled: bool


@dataclass(frozen=True)
class EdgeConfig:
    """Future edge benchmark state, without deployment behavior."""

    benchmark_enabled: bool
    hardware_target: str | None


@dataclass(frozen=True)
class CardioSentinelConfig:
    """Complete validated configuration for a future reproducible run."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    window: WindowConfig
    random_seed: int
    split: SplitConfig
    signal_preprocessing: SignalPreprocessingConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    personalization: PersonalizationConfig
    episodes: EpisodeConfig
    edge: EdgeConfig


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.groups()
            environment_value = os.getenv(name)
            if environment_value is not None:
                return environment_value
            if default is not None:
                return default
            raise ConfigValidationError(f"Environment variable {name} is required.")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{name} must be a mapping.")
    return value


def _expect_keys(mapping: Mapping[str, Any], name: str, expected: set[str]) -> None:
    unknown = set(mapping) - expected
    missing = expected - set(mapping)
    if unknown:
        raise ConfigValidationError(f"{name} has unknown keys: {sorted(unknown)}.")
    if missing:
        raise ConfigValidationError(f"{name} is missing keys: {sorted(missing)}.")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(_mapping(merged[key], key), value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path, ancestors: set[Path]) -> dict[str, Any]:
    resolved_path = path.resolve()
    if resolved_path in ancestors:
        raise ConfigValidationError(f"Circular configuration inheritance at {path}.")
    if not resolved_path.is_file():
        raise ConfigValidationError(f"Configuration file does not exist: {path}.")
    with resolved_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}
    config = _mapping(raw, str(path))
    parent_name = config.get("extends")
    if parent_name is None:
        return dict(config)
    if not isinstance(parent_name, str):
        raise ConfigValidationError("extends must be a string.")
    parent_path = Path(parent_name)
    if parent_path.suffix == "":
        parent_path = parent_path.with_suffix(".yaml")
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    parent = _read_config(parent_path, ancestors | {resolved_path})
    return _deep_merge(parent, config)


def _as_optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigValidationError(f"{name} must be a positive number or null.")
    return float(value)


def _as_fraction(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ConfigValidationError(f"{name} must be a fraction between 0 and 1.")
    return float(value)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{name} must be true or false.")
    return value


def _as_string(value: Any, name: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if allow_none else ""
        raise ConfigValidationError(f"{name} must be a non-empty string{suffix}.")
    return value


def _build_config(raw: Mapping[str, Any]) -> CardioSentinelConfig:
    expected_sections = {
        "project", "paths", "data", "window", "random_seed", "split",
        "signal_preprocessing", "training", "calibration", "personalization",
        "episodes", "edge",
    }
    _expect_keys(raw, "configuration", expected_sections)

    project = _mapping(raw["project"], "project")
    _expect_keys(project, "project", {"name", "profile", "research_only"})
    project_config = ProjectConfig(
        name=_as_string(project["name"], "project.name"),
        profile=_as_string(project["profile"], "project.profile"),
        research_only=_as_bool(project["research_only"], "project.research_only"),
    )
    if not project_config.research_only:
        raise ConfigValidationError("project.research_only must remain true.")

    paths = _mapping(raw["paths"], "paths")
    _expect_keys(paths, "paths", {"dataset_root", "output_root"})
    paths_config = PathsConfig(
        dataset_root=Path(_as_string(paths["dataset_root"], "paths.dataset_root")),
        output_root=Path(_as_string(paths["output_root"], "paths.output_root")),
    )

    data = _mapping(raw["data"], "data")
    _expect_keys(
        data,
        "data",
        {"sampling_frequency_hz", "require_record_metadata", "subject_id_field"},
    )
    data_config = DataConfig(
        sampling_frequency_hz=_as_optional_positive_float(
            data["sampling_frequency_hz"], "data.sampling_frequency_hz"
        ),
        require_record_metadata=_as_bool(
            data["require_record_metadata"], "data.require_record_metadata"
        ),
        subject_id_field=_as_string(data["subject_id_field"], "data.subject_id_field"),
    )
    if not data_config.require_record_metadata:
        raise ConfigValidationError("data.require_record_metadata must remain true.")

    window = _mapping(raw["window"], "window")
    _expect_keys(window, "window", {"length_seconds", "overlap_seconds"})
    window_config = WindowConfig(
        length_seconds=_as_optional_positive_float(
            window["length_seconds"], "window.length_seconds"
        ),
        overlap_seconds=_as_optional_positive_float(
            window["overlap_seconds"], "window.overlap_seconds"
        ),
    )
    if (window_config.length_seconds is None) != (
        window_config.overlap_seconds is None
    ):
        raise ConfigValidationError(
            "window length and overlap must be set together or both null."
        )
    if (
        window_config.length_seconds is not None
        and window_config.overlap_seconds is not None
        and window_config.overlap_seconds >= window_config.length_seconds
    ):
        raise ConfigValidationError(
            "window.overlap_seconds must be less than window.length_seconds."
        )

    random_seed = raw["random_seed"]
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ConfigValidationError("random_seed must be a non-negative integer.")

    split = _mapping(raw["split"], "split")
    _expect_keys(
        split,
        "split",
        {"strategy", "train_fraction", "validation_fraction", "test_fraction"},
    )
    split_config = SplitConfig(
        strategy=_as_string(split["strategy"], "split.strategy"),
        train_fraction=_as_fraction(split["train_fraction"], "split.train_fraction"),
        validation_fraction=_as_fraction(
            split["validation_fraction"], "split.validation_fraction"
        ),
        test_fraction=_as_fraction(split["test_fraction"], "split.test_fraction"),
    )
    if split_config.strategy != "subject_wise":
        raise ConfigValidationError("split.strategy must be subject_wise.")
    if not isclose(
        split_config.train_fraction
        + split_config.validation_fraction
        + split_config.test_fraction,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ConfigValidationError("split fractions must sum to 1.0.")

    preprocessing = _mapping(raw["signal_preprocessing"], "signal_preprocessing")
    _expect_keys(preprocessing, "signal_preprocessing", {"enabled", "filter_profile"})
    preprocessing_config = SignalPreprocessingConfig(
        enabled=_as_bool(preprocessing["enabled"], "signal_preprocessing.enabled"),
        filter_profile=_as_string(
            preprocessing["filter_profile"],
            "signal_preprocessing.filter_profile",
            allow_none=True,
        ),
    )

    training = _mapping(raw["training"], "training")
    _expect_keys(training, "training", {"enabled"})
    training_config = TrainingConfig(
        enabled=_as_bool(training["enabled"], "training.enabled")
    )

    calibration = _mapping(raw["calibration"], "calibration")
    _expect_keys(
        calibration,
        "calibration",
        {"method", "routing_requires_calibrated_confidence"},
    )
    calibration_config = CalibrationConfig(
        method=_as_string(calibration["method"], "calibration.method", allow_none=True),
        routing_requires_calibrated_confidence=_as_bool(
            calibration["routing_requires_calibrated_confidence"],
            "calibration.routing_requires_calibrated_confidence",
        ),
    )
    if not calibration_config.routing_requires_calibrated_confidence:
        raise ConfigValidationError("routing must require calibrated confidence.")

    personalization = _mapping(raw["personalization"], "personalization")
    _expect_keys(personalization, "personalization", {"enabled"})
    personalization_config = PersonalizationConfig(
        enabled=_as_bool(personalization["enabled"], "personalization.enabled")
    )

    episodes = _mapping(raw["episodes"], "episodes")
    _expect_keys(episodes, "episodes", {"enabled"})
    episode_config = EpisodeConfig(
        enabled=_as_bool(episodes["enabled"], "episodes.enabled")
    )

    edge = _mapping(raw["edge"], "edge")
    _expect_keys(edge, "edge", {"benchmark_enabled", "hardware_target"})
    edge_config = EdgeConfig(
        benchmark_enabled=_as_bool(
            edge["benchmark_enabled"], "edge.benchmark_enabled"
        ),
        hardware_target=_as_string(
            edge["hardware_target"], "edge.hardware_target", allow_none=True
        ),
    )

    return CardioSentinelConfig(
        project=project_config,
        paths=paths_config,
        data=data_config,
        window=window_config,
        random_seed=random_seed,
        split=split_config,
        signal_preprocessing=preprocessing_config,
        training=training_config,
        calibration=calibration_config,
        personalization=personalization_config,
        episodes=episode_config,
        edge=edge_config,
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> CardioSentinelConfig:
    """Load, merge, expand, and validate a CardioSentinel YAML configuration."""
    raw = _expand_environment(_read_config(Path(path), set()))
    return _build_config(_mapping(raw, "configuration"))
