"""Offline tests for pinned-source integrity and receipt semantics."""

from pathlib import Path

import pytest

from cardiosentinel.baseline.materialize import materialize_features
from cardiosentinel.baseline.source import (
    OFFICIAL_MANIFEST_NAME,
    RECORDS_NAME,
    SOURCE_VERIFICATION_RECEIPT_NAME,
    _verify_source_against_manifest,
    acquisition_urls,
    parse_checksum_manifest,
    required_source_filenames,
)
from cardiosentinel.data.provenance import sha256_file


def _write_source_fixture(
    root: Path, records: tuple[str, ...], *, omit_checksum: str | None = None
) -> Path:
    root.mkdir()
    (root / RECORDS_NAME).write_text("\n".join(records) + "\n", encoding="utf-8")
    required = required_source_filenames(records)
    for filename in required:
        if filename != RECORDS_NAME:
            (root / filename).write_bytes(filename.encode("ascii"))
    lines = []
    for filename in required:
        if filename == omit_checksum:
            continue
        lines.append(f"{sha256_file(root / filename)} {filename}")
    manifest = root / OFFICIAL_MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _verify_fixture(root: Path, records: tuple[str, ...]) -> dict[str, object]:
    manifest = root / OFFICIAL_MANIFEST_NAME
    return _verify_source_against_manifest(
        root,
        records,
        official_manifest_sha256=sha256_file(manifest),
        write_receipt=True,
    )


def test_checksum_manifest_parser_accepts_standard_entries(tmp_path: Path) -> None:
    manifest = tmp_path / OFFICIAL_MANIFEST_NAME
    digest = "a" * 64
    manifest.write_text(f"{digest} *s20011.dat\n", encoding="utf-8")
    assert parse_checksum_manifest(manifest) == {"s20011.dat": digest}


def test_checksum_manifest_parser_rejects_malformed_entries(tmp_path: Path) -> None:
    manifest = tmp_path / OFFICIAL_MANIFEST_NAME
    manifest.write_text("not a checksum\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        parse_checksum_manifest(manifest)


def test_source_verification_rejects_absent_official_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Official checksum manifest"):
        _verify_source_against_manifest(
            tmp_path / "missing-source",
            ("s20001",),
            official_manifest_sha256="0" * 64,
            write_receipt=False,
        )


def test_all_required_source_files_verify_and_write_receipt(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records)
    receipt = _verify_fixture(root, records)
    assert receipt["verification_result"] == "passed"
    assert receipt["expected_record_count"] == 2
    assert receipt["verified_required_file_count"] == 7
    assert (root / SOURCE_VERIFICATION_RECEIPT_NAME).is_file()


def test_source_verification_rejects_missing_required_file(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records)
    (root / "s20001.dat").unlink()
    with pytest.raises(FileNotFoundError, match="absent"):
        _verify_fixture(root, records)


def test_source_verification_rejects_unpinned_local_manifest(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records)
    with pytest.raises(ValueError, match="pinned release"):
        _verify_source_against_manifest(
            root,
            records,
            official_manifest_sha256="0" * 64,
            write_receipt=False,
        )


def test_source_verification_rejects_missing_checksum_entry(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records, omit_checksum="s20001.stb")
    with pytest.raises(ValueError, match="lack official checksum"):
        _verify_fixture(root, records)


def test_source_verification_rejects_checksum_mismatch(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records)
    (root / "s20001.dat").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_fixture(root, records)


def test_source_verification_rejects_records_set_mismatch(tmp_path: Path) -> None:
    records = ("s20001", "s20011")
    root = tmp_path / "source"
    _write_source_fixture(root, records)
    (root / RECORDS_NAME).write_text("s20001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RECORDS set"):
        _verify_fixture(root, records)


def test_acquisition_urls_include_only_manifest_and_required_files() -> None:
    urls = acquisition_urls(("s20011", "s20001"))
    assert len(urls) == 8
    assert urls[0].endswith(OFFICIAL_MANIFEST_NAME)
    assert all(url.endswith(("RECORDS", ".hea", ".dat", ".stb")) for url in urls[1:])


def test_materialization_rejects_unverified_full_frozen_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    feature_root = tmp_path / "features"
    calls: list[set[str]] = []

    def reject_unverified(_: Path, record_ids: set[str]) -> dict[str, object]:
        calls.append(record_ids)
        raise ValueError("Pinned source is unverified.")

    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.verify_pinned_source", reject_unverified
    )
    monkeypatch.setattr(
        "cardiosentinel.baseline.materialize.inspect_dataset",
        lambda *args, **kwargs: pytest.fail("inspection must follow verification"),
    )
    with pytest.raises(ValueError, match="unverified"):
        materialize_features(
            source,
            feature_root,
            Path("protocols/splits/ltstdb_v1.json"),
        )
    assert len(calls) == 1
    assert len(calls[0]) == 86
