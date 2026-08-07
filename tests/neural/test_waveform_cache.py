import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.metadata import B4MetadataIndex, B4WindowReference
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
)
from cardiosentinel.neural.waveform_cache import (
    CACHE_MANIFEST_NAME,
    CACHE_SCHEMA_VERSION,
    TRAINING_SELECTION_SHA256,
    B4CachedWaveformDataset,
    _load_or_create_progress,
    _manifest_candidate,
    audit_waveform_cache_equivalence,
    compute_waveform_cache_sha256,
    validate_waveform_cache,
)
from cardiosentinel.signal.models import WaveformSegment


def reference(
    record: str, partition: str, start: int, family: str = "background_negative"
) -> B4WindowReference:
    return B4WindowReference(
        stable_id=f"ltstdb:{record}:0:{start}:{start + 2500}",
        record_id=record,
        subject_id=record,
        channel_index=0,
        start_sample=start,
        end_sample=start + 2500,
        partition=partition,
        target_family=family,
        context_flags=(),
    )


def indexes() -> dict[str, B4MetadataIndex]:
    train = (
        reference("r-train", "train", 0, "ischemic_positive"),
        reference("r-train", "train", 2500),
    )
    validation = (reference("r-validation", "validation", 0),)
    return {
        "train": B4MetadataIndex("train", train, 1, 1, 1, "selection"),
        "validation": B4MetadataIndex("validation", validation, 0, 1, 1),
    }


def _identity() -> dict:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "training_selection_sha256": TRAINING_SELECTION_SHA256,
        "development_feature_integrity_sha256": "1" * 64,
        "development_source_integrity_sha256": "2" * 64,
        "processing_profile": "raw",
        "sampling_frequency_hz": 250.0,
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "samples_per_row": 2500,
        "dtype": "float32",
        "physical_unit": "mV",
    }


def cache_fixture(root: Path) -> tuple[dict, dict[str, B4MetadataIndex]]:
    index_map = indexes()
    partitions = {}
    for partition, index in index_map.items():
        waveforms = np.vstack(
            [
                np.full(2500, row + (10 if partition == "validation" else 0))
                for row in range(index.total_count)
            ]
        ).astype(np.float32)
        stable_ids = np.asarray([item.stable_id for item in index.references])
        waveform_file = f"{partition}_waveforms.npy"
        stable_id_file = f"{partition}_stable_ids.npy"
        np.save(root / waveform_file, waveforms, allow_pickle=False)
        np.save(root / stable_id_file, stable_ids, allow_pickle=False)
        partitions[partition] = {
            "row_count": index.total_count,
            "waveform_file": waveform_file,
            "waveform_sha256": sha256_file(root / waveform_file),
            "waveform_bytes": (root / waveform_file).stat().st_size,
            "stable_id_file": stable_id_file,
            "stable_id_sha256": sha256_file(root / stable_id_file),
            "stable_id_bytes": (root / stable_id_file).stat().st_size,
        }
    audit = {
        "selection_rule": "synthetic deterministic fixture",
        "audited_rows": 3,
        "audited_rows_by_partition": {"train": 2, "validation": 1},
        "records_represented": 2,
        "channel_indices_represented": [0],
        "record_channel_groups_represented": 2,
        "exact_mismatches": 0,
        "comparison": "np.array_equal",
        "result": "passed",
    }
    manifest = _manifest_candidate(
        _identity(),
        partitions,
        audit,
        {"git_sha": "a" * 40, "git_dirty": False},
        "synthetic fixture",
        {},
        1.0,
    )
    write_json_atomic(root / CACHE_MANIFEST_NAME, manifest)
    return manifest, index_map


def _patch_counts(monkeypatch) -> None:
    import cardiosentinel.neural.waveform_cache as cache

    monkeypatch.setitem(cache.EXPECTED_COUNTS, "train", {"total": 2})
    monkeypatch.setitem(cache.EXPECTED_COUNTS, "validation", {"total": 1})


def test_cached_loader_preserves_values_and_predictive_surface(
    tmp_path, monkeypatch
) -> None:
    _patch_counts(monkeypatch)
    _, index_map = cache_fixture(tmp_path)
    validated = validate_waveform_cache(tmp_path, index_map)
    dataset = B4CachedWaveformDataset(validated, index_map["train"])
    sample = dataset[1]

    assert sample._fields == ("waveform", "label")
    assert sample.waveform.shape == (1, 2500)
    assert sample.waveform.dtype == torch.float32
    assert np.array_equal(sample.waveform.numpy(), np.ones((1, 2500), np.float32))
    assert sample.label.item() == 0.0
    assert not hasattr(dataset, "features")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_sha256", "0" * 64, "identity"),
        ("split_sha256", "0" * 64, "identity"),
        ("training_selection_sha256", "0" * 64, "identity"),
    ],
)
def test_cache_rejects_wrong_frozen_identity(
    tmp_path, monkeypatch, field: str, value: str, message: str
) -> None:
    _patch_counts(monkeypatch)
    manifest, index_map = cache_fixture(tmp_path)
    manifest[field] = value
    write_json_atomic(tmp_path / CACHE_MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match=message):
        validate_waveform_cache(tmp_path, index_map)


@pytest.mark.parametrize("kind", ["waveform", "stable_id"])
def test_cache_rejects_corrupt_file(tmp_path, monkeypatch, kind: str) -> None:
    _patch_counts(monkeypatch)
    manifest, index_map = cache_fixture(tmp_path)
    path = tmp_path / manifest["partitions"]["train"][f"{kind}_file"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    manifest["partitions"]["train"][f"{kind}_bytes"] = path.stat().st_size
    manifest["waveform_cache_sha256"] = compute_waveform_cache_sha256(manifest)
    write_json_atomic(tmp_path / CACHE_MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_waveform_cache(tmp_path, index_map)


def test_cache_rejects_stable_id_misalignment(tmp_path, monkeypatch) -> None:
    _patch_counts(monkeypatch)
    manifest, index_map = cache_fixture(tmp_path)
    path = tmp_path / manifest["partitions"]["train"]["stable_id_file"]
    values = np.load(path, allow_pickle=False)[::-1]
    np.save(path, values, allow_pickle=False)
    item = manifest["partitions"]["train"]
    item["stable_id_sha256"] = sha256_file(path)
    item["stable_id_bytes"] = path.stat().st_size
    manifest["waveform_cache_sha256"] = compute_waveform_cache_sha256(manifest)
    write_json_atomic(tmp_path / CACHE_MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="stable-ID alignment"):
        validate_waveform_cache(tmp_path, index_map)


def test_cache_rejects_incomplete_or_test_partition(tmp_path, monkeypatch) -> None:
    _patch_counts(monkeypatch)
    with pytest.raises(ValueError, match="incomplete"):
        validate_waveform_cache(tmp_path, indexes())

    manifest, index_map = cache_fixture(tmp_path)
    manifest["partitions"]["test"] = copy.deepcopy(
        manifest["partitions"]["train"]
    )
    write_json_atomic(tmp_path / CACHE_MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="train and validation only"):
        validate_waveform_cache(tmp_path, index_map)


def test_cache_requires_nonversioned_location() -> None:
    with pytest.raises(ValueError, match="outside Git tracking"):
        validate_waveform_cache(REPOSITORY_ROOT / "tracked-cache", indexes())


def _segment_for(reference: B4WindowReference, value: float) -> WaveformSegment:
    return WaveformSegment(
        dataset_id="ltstdb",
        dataset_version="1.0.0",
        record_id=reference.record_id,
        subject_id=reference.subject_id,
        sampling_frequency_hz=250.0,
        start_sample=reference.start_sample,
        end_sample=reference.end_sample,
        start_seconds=reference.start_sample / 250.0,
        end_seconds=reference.end_sample / 250.0,
        signal_names=("ECG",),
        lead_names=("ECG",),
        physical_units=("mV",),
        source_physical_units=("mV",),
        values=np.full((2500, 1), value, dtype=np.float64),
        source="fixture",
        provenance={"requested_channels": (0,)},
    )


def test_exact_equivalence_audit_uses_array_equal(tmp_path, monkeypatch) -> None:
    _patch_counts(monkeypatch)
    _, index_map = cache_fixture(tmp_path)
    by_identity = {
        (item.record_id, item.start_sample): (
            row + (10 if partition == "validation" else 0)
        )
        for partition, index in index_map.items()
        for row, item in enumerate(index.references)
    }

    def reader(root, dataset, record, start, end, channels):
        del root, dataset, end, channels
        reference_item = next(
            item
            for index in index_map.values()
            for item in index.references
            if item.record_id == record and item.start_sample == start
        )
        return _segment_for(reference_item, by_identity[(record, start)])

    report = audit_waveform_cache_equivalence(
        tmp_path,
        tmp_path,
        index_map,
        _reader=reader,
        _source_verifier=lambda _: {},
    )
    assert report["audited_rows"] == 3
    assert report["exact_mismatches"] == 0
    assert report["comparison"] == "np.array_equal"


def test_partial_cache_with_wrong_provenance_is_rejected(tmp_path) -> None:
    identity = {"protocol_sha256": B4_PROTOCOL_SHA256}
    _load_or_create_progress(tmp_path, identity)
    with pytest.raises(ValueError, match="different provenance"):
        _load_or_create_progress(tmp_path, {"protocol_sha256": "0" * 64})
