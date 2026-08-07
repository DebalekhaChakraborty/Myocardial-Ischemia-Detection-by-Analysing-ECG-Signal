"""SYNTHETIC record-worker equivalence, resume, and failure tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.baseline.cache import read_feature_table, read_json
from cardiosentinel.baseline.materialize import (
    MaterializationJob,
    _materialization_results,
    materialize_features,
)
from cardiosentinel.data.models import DatasetRecord, ParsedAnnotations


def _synthetic_records(source: Path) -> tuple[DatasetRecord, ...]:
    import wfdb

    fs = 250
    sample_count = fs * 25
    samples = np.arange(sample_count)
    records = []
    for offset, record_id in enumerate(("s20001", "s20011")):
        signal = np.column_stack(
            (
                0.2 * np.sin(2 * np.pi * (1.0 + offset * 0.1) * samples / fs),
                0.15 * np.cos(2 * np.pi * 1.2 * samples / fs),
            )
        )
        wfdb.wrsamp(
            record_id,
            fs=fs,
            units=["mV", "mV"],
            sig_name=["I", "II"],
            p_signal=signal,
            fmt=["16", "16"],
            write_dir=str(source),
        )
        (source / f"{record_id}.stb").write_bytes(b"")
        subject_id = f"ltstdb:{record_id[:-1]}"
        records.append(
            DatasetRecord(
                dataset_id="ltstdb",
                dataset_version="1.0.0",
                record_id=record_id,
                subject_id=subject_id,
                sampling_frequency_hz=float(fs),
                signal_names=("I", "II"),
                lead_names=("I", "II"),
                signal_count=2,
                sample_count=sample_count,
                duration_seconds=sample_count / fs,
                source_path=str(source / record_id),
            )
        )
    return tuple(records)


def _job(source: Path, root: Path, record: DatasetRecord) -> MaterializationJob:
    return MaterializationJob(
        source=source,
        destination=root / "train" / f"{record.record_id}.npz",
        record=record,
        parsed=ParsedAnnotations((), (), ()),
        partition="train",
        metadata={"record_id": record.record_id},
        chunk_seconds=7.0,
    )


def test_workers_one_and_two_have_identical_scientific_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = _synthetic_records(source)
    sequential_root = tmp_path / "sequential"
    parallel_root = tmp_path / "parallel"
    sequential = list(
        _materialization_results(
            [_job(source, sequential_root, record) for record in records], 1
        )
    )
    parallel = list(
        _materialization_results(
            [_job(source, parallel_root, record) for record in records], 2
        )
    )
    sequential_quality = {
        job.record.record_id: result["morphology_quality"] for job, result in sequential
    }
    parallel_quality = {
        job.record.record_id: result["morphology_quality"] for job, result in parallel
    }
    assert sequential_quality == parallel_quality
    for record in records:
        first, _ = read_feature_table(
            sequential_root / "train" / f"{record.record_id}.npz"
        )
        second, _ = read_feature_table(
            parallel_root / "train" / f"{record.record_id}.npz"
        )
        assert np.array_equal(first.stable_ids, second.stable_ids)
        assert np.array_equal(first.target_families, second.target_families)
        assert np.array_equal(first.context_flags, second.context_flags)
        assert np.array_equal(first.features, second.features, equal_nan=True)


def test_parallel_materialization_resumes_digest_valid_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = _synthetic_records(source)
    parsed = tuple(ParsedAnnotations((), (), ()) for _ in records)
    split = {
        "records_by_subject": {
            record.subject_id: [record.record_id] for record in records
        },
        "partitions": {
            "train": {"subjects": [record.subject_id for record in records]},
            "validation": {"subjects": []},
            "test": {"subjects": []},
        },
    }
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.load_split_manifest", lambda _: split
    )
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.validate_split_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.inspect_dataset",
        lambda *args, **kwargs: (records, parsed),
    )
    root = tmp_path / "features"
    initial = materialize_features(source, root, Path("unused"), workers=2)
    initial_hashes = {
        entry["record_id"]: entry["cache_sha256"] for entry in initial["records"]
    }
    resumed = materialize_features(source, root, Path("unused"), workers=1)
    assert all(entry["resumed"] for entry in resumed["records"])
    assert {
        entry["record_id"]: entry["cache_sha256"] for entry in resumed["records"]
    } == initial_hashes
    assert resumed["feature_corpus_sha256"] == initial["feature_corpus_sha256"]


def test_failed_worker_is_identified_and_never_manifested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    good_records = _synthetic_records(source)
    failed = replace(
        good_records[1],
        sample_count=good_records[1].sample_count + 250,
        duration_seconds=good_records[1].duration_seconds + 1.0,
    )
    records = (good_records[0], failed)
    parsed = tuple(ParsedAnnotations((), (), ()) for _ in records)
    split = {
        "records_by_subject": {
            record.subject_id: [record.record_id] for record in records
        },
        "partitions": {
            "train": {"subjects": [record.subject_id for record in records]},
            "validation": {"subjects": []},
            "test": {"subjects": []},
        },
    }
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.load_split_manifest", lambda _: split
    )
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.validate_split_manifest",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.inspect_dataset",
        lambda *args, **kwargs: (records, parsed),
    )
    root = tmp_path / "features"
    with pytest.raises(RuntimeError, match=failed.record_id):
        materialize_features(source, root, Path("unused"), workers=2)
    manifest = read_json(root / "manifest.json")
    assert failed.record_id not in {
        entry["record_id"] for entry in manifest["records"]
    }
