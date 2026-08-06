from pathlib import Path

import pytest

from cardiosentinel.config import (
    DEFAULT_CONFIG_PATH,
    ConfigValidationError,
    load_config,
)


def test_base_configuration_loads() -> None:
    config = load_config()

    assert config.project.name == "CardioSentinel"
    assert config.project.profile == "base"
    assert config.project.research_only is True
    assert config.data.sampling_frequency_hz is None
    assert config.split.strategy == "subject_wise"
    assert config.signal.canonical_unit == "mV"
    assert config.preprocessing.name == "raw"
    assert config.preprocessing.highpass.enabled is False
    assert config.preprocessing.lowpass.enabled is False
    assert config.preprocessing.notch.enabled is False
    assert config.windowing.length_seconds is None
    assert config.windowing.stride_seconds is None


def test_invalid_sampling_frequency_is_rejected(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\ndata:\n  sampling_frequency_hz: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="sampling_frequency_hz"):
        load_config(invalid_config)


def test_invalid_split_strategy_is_rejected(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\nsplit:\n  strategy: window_wise\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="subject_wise"):
        load_config(invalid_config)


def test_st_preserving_configuration_rejects_aggressive_highpass(
    tmp_path: Path,
) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\n"
        "preprocessing:\n"
        "  profile: st_preserving\n"
        "  highpass:\n"
        "    enabled: true\n"
        "    cutoff_hz: 0.5\n"
        "    order: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="cannot exceed 0.05"):
        load_config(invalid_config)


def test_window_length_and_stride_must_be_set_together(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\nwindowing:\n  length_seconds: 10\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="length and stride"):
        load_config(invalid_config)
