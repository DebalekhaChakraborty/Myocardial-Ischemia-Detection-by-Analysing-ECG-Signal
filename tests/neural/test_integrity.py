import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.baseline.cache import (
    compute_feature_corpus_sha256,
    write_json_atomic,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import PRIMARY_ANNOTATION_DEFINITION
from cardiosentinel.features.schema import COMBINED_V1
from cardiosentinel.neural.integrity import (
    _validate_development_feature_integrity,
    _validate_development_source_integrity,
    source_record_sha256,
)
from cardiosentinel.neural.protocol import B4_SPLIT_SHA256


def _entry(record_id: str, partition: str) -> dict:
    return {
        "record_id": record_id,
        "subject_id": f"subject-{record_id}",
        "partition": partition,
        "cache_path": f"{partition}/{record_id}.npz",
        "status": "complete",
        "row_count": 1,
        "target_counts": {"background_negative": 1},
        "source_sha256": "0" * 64,
        "cache_sha256": "f" * 64,
    }


def _embedded(manifest: dict, entry: dict) -> dict:
    return {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "record_id": entry["record_id"],
        "subject_id": entry["subject_id"],
        "partition": entry["partition"],
        "source_sha256": entry["source_sha256"],
        "split_sha256": B4_SPLIT_SHA256,
        "feature_schema_sha256": COMBINED_V1.sha256,
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
        "row_count": 1,
        "target_counts": {"background_negative": 1},
    }


def feature_fixture(root: Path) -> tuple[dict, str]:
    entries = [
        _entry("r-train", "train"),
        _entry("r-validation", "validation"),
        _entry("r-test", "test"),
    ]
    manifest = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": B4_SPLIT_SHA256,
        "feature_schemas": {"combined_v1": COMBINED_V1.as_dict()},
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": PRIMARY_ANNOTATION_DEFINITION,
        "records": entries,
    }
    for entry in entries[:2]:
        path = root / entry["cache_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as destination:
            np.savez_compressed(
                destination,
                metadata_json=np.asarray(json.dumps(_embedded(manifest, entry))),
            )
        entry["cache_sha256"] = sha256_file(path)
    # The synthetic test cache intentionally does not exist.
    corpus_sha256 = compute_feature_corpus_sha256(manifest)
    manifest["feature_corpus_sha256"] = corpus_sha256
    write_json_atomic(root / "manifest.json", manifest)
    return manifest, corpus_sha256


def test_feature_integrity_hashes_development_and_never_opens_test(tmp_path) -> None:
    _, corpus_sha256 = feature_fixture(tmp_path)
    receipt = _validate_development_feature_integrity(
        tmp_path, expected_corpus_sha256=corpus_sha256
    )

    assert receipt["verification_result"] == "passed"
    assert receipt["verified_record_count"] == 2
    assert {item["partition"] for item in receipt["records"]} == {
        "train",
        "validation",
    }
    assert not (tmp_path / "test" / "r-test.npz").exists()


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_feature_integrity_rejects_corrupt_development_cache(
    tmp_path, partition: str
) -> None:
    _, corpus_sha256 = feature_fixture(tmp_path)
    path = next((tmp_path / partition).glob("*.npz"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        _validate_development_feature_integrity(
            tmp_path, expected_corpus_sha256=corpus_sha256
        )


def test_feature_integrity_rejects_noncanonical_manifest(tmp_path) -> None:
    manifest, corpus_sha256 = feature_fixture(tmp_path)
    manifest["records"][0]["row_count"] = 2
    write_json_atomic(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Canonical feature-corpus"):
        _validate_development_feature_integrity(
            tmp_path, expected_corpus_sha256=corpus_sha256
        )


def source_fixture(root: Path) -> tuple[dict, str]:
    root.mkdir(parents=True, exist_ok=True)
    official: dict[str, str] = {}
    records = []
    for record_id, partition in (
        ("r-train", "train"),
        ("r-validation", "validation"),
    ):
        file_digests = {}
        for suffix in ("hea", "dat", "stb"):
            filename = f"{record_id}.{suffix}"
            (root / filename).write_bytes(f"{filename}-fixture".encode())
            file_digests[filename] = sha256_file(root / filename)
            official[filename] = file_digests[filename]
        records.append(
            {
                "record_id": record_id,
                "partition": partition,
                "source_sha256": source_record_sha256(record_id, file_digests),
            }
        )
    for suffix in ("hea", "dat", "stb"):
        official[f"r-test.{suffix}"] = "a" * 64
    manifest = root / "SHA256SUMS.txt"
    manifest.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(official.items())),
        encoding="utf-8",
    )
    return {"records": records}, sha256_file(manifest)


def test_source_integrity_hashes_development_and_never_opens_test(tmp_path) -> None:
    feature_receipt, manifest_sha256 = source_fixture(tmp_path)
    receipt = _validate_development_source_integrity(
        tmp_path,
        feature_receipt,
        expected_manifest_sha256=manifest_sha256,
    )

    assert receipt["verification_result"] == "passed"
    assert receipt["verified_file_count"] == 6
    assert not (tmp_path / "r-test.dat").exists()


def test_source_integrity_rejects_changed_development_waveform(tmp_path) -> None:
    feature_receipt, manifest_sha256 = source_fixture(tmp_path)
    (tmp_path / "r-train.dat").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        _validate_development_source_integrity(
            tmp_path,
            feature_receipt,
            expected_manifest_sha256=manifest_sha256,
        )


def test_source_integrity_rejects_forbidden_receipt_partition(tmp_path) -> None:
    feature_receipt, manifest_sha256 = source_fixture(tmp_path)
    feature_receipt["records"].append(
        {"record_id": "r-test", "partition": "test", "source_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="forbidden partition"):
        _validate_development_source_integrity(
            tmp_path,
            feature_receipt,
            expected_manifest_sha256=manifest_sha256,
        )
