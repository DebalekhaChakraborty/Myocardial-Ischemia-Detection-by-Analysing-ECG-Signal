"""SYNTHETIC waveform-only feature schema, causality, and firewall tests."""

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.features.morphology import extract_morphology_features
from cardiosentinel.features.schema import COMBINED_V1, MORPHOLOGY_V1, SIGNAL_V1
from cardiosentinel.features.signal_features import extract_signal_features
from cardiosentinel.signal.models import CausalWindow


def window(values: np.ndarray, fs: float = 250.0) -> CausalWindow:
    values = np.asarray(values, dtype=np.float64)
    return CausalWindow(
        "record",
        "subject",
        0,
        "I",
        "I",
        "mV",
        fs,
        0,
        len(values),
        0.0,
        len(values) / fs,
        len(values),
        False,
        values,
    )


def synthetic_periodic_ecg(fs: float = 250.0) -> np.ndarray:
    time = np.arange(round(10 * fs)) / fs
    values = 0.03 * np.sin(2 * np.pi * 0.3 * time)
    for peak_time in np.arange(0.6, 9.7, 1.0):
        values += 1.2 * np.exp(-0.5 * ((time - peak_time) / 0.018) ** 2)
        values -= 0.25 * np.exp(-0.5 * ((time - peak_time + 0.035) / 0.012) ** 2)
        values += 0.18 * np.exp(-0.5 * ((time - peak_time - 0.25) / 0.06) ** 2)
    return values


def test_feature_schema_order_and_hash_are_frozen() -> None:
    assert len(SIGNAL_V1.names) == 22
    assert len(MORPHOLOGY_V1.names) == 18
    assert len(COMBINED_V1.names) == 40
    assert COMBINED_V1.names == SIGNAL_V1.names + MORPHOLOGY_V1.names
    assert SIGNAL_V1.sha256 == (
        "25d05f8716340e0fcc9950590025e7c58dccbfe8d0e0475ccd36bd629d4c57d4"
    )
    assert MORPHOLOGY_V1.sha256 == (
        "13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434"
    )
    assert COMBINED_V1.sha256 == (
        "6b1517cb6ffd5d113a385bb252a90630f75beec4da7345185e74dda0eff98a34"
    )


def test_signal_features_retain_physical_derivative_units() -> None:
    fs = 250.0
    time = np.arange(2_500) / fs
    features = extract_signal_features(window(np.sin(2 * np.pi * 3 * time), fs))
    by_name = dict(zip(SIGNAL_V1.names, features, strict=True))
    assert features.shape == (22,)
    assert by_name["robust_derivative_scale_mv_per_s"] > 0
    assert by_name["median_absolute_derivative_mv_per_s"] > 0
    definitions = {item.name: item.unit for item in SIGNAL_V1.features}
    assert definitions["robust_derivative_scale_mv_per_s"] == "mV/s"
    assert definitions["median_amplitude_mv"] == "mV"


def test_xqrs_morphology_on_synthetic_periodic_waveform() -> None:
    features = extract_morphology_features(window(synthetic_periodic_ecg()))
    by_name = dict(zip(MORPHOLOGY_V1.names, features, strict=True))
    assert by_name["detected_r_peak_count"] >= 9
    assert by_name["usable_beat_count"] >= 8
    assert by_name["morphology_valid"] == 1
    assert by_name["rr_mean_ms"] == pytest.approx(1_000.0, abs=10.0)
    assert by_name["estimated_hr_bpm"] == pytest.approx(60.0, abs=1.0)
    assert np.isfinite(features).all()


def test_morphology_edge_and_failed_qrs_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cardiosentinel.features.morphology as morphology

    values = synthetic_periodic_ecg()
    monkeypatch.setattr(
        morphology,
        "_detect_r_peaks",
        lambda values, fs: np.asarray([10, len(values) - 10], dtype=np.int64),
    )
    features = morphology.extract_morphology_features(window(values))
    by_name = dict(zip(MORPHOLOGY_V1.names, features, strict=True))
    assert by_name["detected_r_peak_count"] == 2
    assert by_name["usable_beat_count"] == 0
    assert by_name["morphology_valid"] == 0
    assert np.isnan(by_name["post_r_80ms_delta_mv"])

    monkeypatch.setattr(
        morphology,
        "_detect_r_peaks",
        lambda values, fs: np.empty(0, dtype=np.int64),
    )
    failed = morphology.extract_morphology_features(window(values))
    assert failed[0] == 0
    assert failed[1] == 0
    assert failed[2] == 0
    assert np.isnan(failed[3:]).all()


def test_feature_extraction_cannot_observe_future_samples() -> None:
    values = synthetic_periodic_ecg()
    first = window(values)
    future_stream = np.concatenate((values, np.full(500, 99.0)))
    second = window(future_stream[: len(values)])
    assert np.array_equal(
        extract_signal_features(first), extract_signal_features(second)
    )
    assert np.array_equal(
        extract_morphology_features(first),
        extract_morphology_features(second),
        equal_nan=True,
    )


def test_feature_package_has_annotation_import_firewall() -> None:
    feature_root = Path(__file__).resolve().parents[2] / "src/cardiosentinel/features"
    forbidden_modules = {
        "cardiosentinel.data",
        "cardiosentinel.evaluation",
    }
    forbidden_names = {
        "STEvent",
        "WindowTarget",
        "peak_deviation_uv",
        "GRST",
        "LRST",
    }
    for path in feature_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not any(
            module == forbidden or module.startswith(f"{forbidden}.")
            for module in imported_modules
            for forbidden in forbidden_modules
        )
        assert names.isdisjoint(forbidden_names)


def test_predictive_feature_names_exclude_identity_and_annotations() -> None:
    forbidden_fragments = (
        "subject",
        "record",
        "stable_id",
        "channel",
        "lead",
        "partition",
        "target",
        "context",
        "annotation",
        "event_id",
        "marker_id",
        "expert",
    )
    assert all(
        fragment not in name
        for name in COMBINED_V1.names
        for fragment in forbidden_fragments
    )
