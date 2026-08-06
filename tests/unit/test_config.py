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


def test_invalid_sampling_frequency_is_rejected(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\n"
        "data:\n"
        "  sampling_frequency_hz: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="sampling_frequency_hz"):
        load_config(invalid_config)


def test_invalid_split_strategy_is_rejected(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        f"extends: {DEFAULT_CONFIG_PATH}\n"
        "split:\n"
        "  strategy: window_wise\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError, match="subject_wise"):
        load_config(invalid_config)

