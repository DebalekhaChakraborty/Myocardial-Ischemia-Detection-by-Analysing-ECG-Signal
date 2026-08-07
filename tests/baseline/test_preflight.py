"""SYNTHETIC engineering preflight report tests."""

from pathlib import Path

import pytest

from cardiosentinel.baseline.preflight import baseline_preflight


def test_preflight_reports_frozen_counts_and_resource_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "cardiosentinel.baseline.preflight._project_materialization_seconds",
        lambda workers: {
            "projected_seconds": 123.0 / workers,
            "engineering_estimate_only": True,
        },
    )
    report = baseline_preflight(
        tmp_path / "missing-source",
        tmp_path / "features",
        Path("protocols/splits/ltstdb_v1.json"),
        workers=2,
    )
    assert report["expected_record_count"] == 86
    assert report["requested_workers"] == 2
    assert report["source_data"]["complete_record_count"] == 0
    assert report["blocks_execution"] is True
    assert report["blocking_reasons"] == ["Pinned LTSTDB source data is incomplete."]
    assert report["feature_cache"]["digest_valid_resumable_records"] == 0
    assert report["expected_primary_benchmark_counts"]["train"] == {
        "ischemic_positive": 93_613,
        "background_negative": 2_049_986,
    }
    assert (
        report["conservative_memory_estimates"]["selected_training_matrix"][
            "conservative_working_bytes"
        ]
        > 0
    )
    assert report["projected_full_materialization_runtime"]["projected_seconds"] == (
        61.5
    )
