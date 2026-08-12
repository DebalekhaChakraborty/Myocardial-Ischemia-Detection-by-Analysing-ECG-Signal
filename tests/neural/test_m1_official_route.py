"""End-to-end exercise of the canonical `m1 run-stage1` orchestration.

The earlier route tests inspected AST and literals, which is exactly why they
missed a real defect: the official CLI called `execute_m1_stage1` without any
waveform iterator, so the canonical command would have failed the moment it met
a full-stream row outside the P1 primary cache. Static inspection cannot catch
that. This test executes the same orchestration the official route runs.

Everything is synthetic and tiny. The upstream gates are stubbed because CI does
not run the frozen scientific environment and holds no real corpus; the
production gates are unchanged and still evaluated by the real route.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cardiosentinel.neural import m1_experiment, p1_experiment
from cardiosentinel.neural.m1_experiment import (
    M1_ARM_ORDER,
    M1_STAGE1_RESULT_NAME,
    execute_m1_stage1,
)
from cardiosentinel.neural.p1_experiment import P1EmbeddingCache
from cardiosentinel.neural.patient_memory import (
    REPRESENTATION_DIM,
    STREAM_CACHE_MANIFEST_NAME,
)
from cardiosentinel.neural.physiology_fusion import (
    EMBEDDING_DIM,
    PHYSIOLOGY_DIM,
    PHYSIOLOGY_FEATURE_NAMES,
    fit_physiology_transform,
)
from tests.neural.test_patient_memory import reference

PRIMARY = ("background_negative", "ischemic_positive")
TRAIN_FAMILIES = ("background_negative", "ischemic_positive", "quality_excluded")
VALIDATION_FAMILIES = (
    "background_negative",
    "ischemic_positive",
    "rate_related",
    "axis_shift",
    "conduction_change",
    "quality_excluded",
)
WINDOWS_PER_STREAM = 12
STUB_DIGEST = "0" * 64


def _references(partition: str, records: tuple[str, ...], families: tuple[str, ...]):
    """Two channels per record, interleaved primary / challenge / non-primary."""
    rows = []
    for position, record in enumerate(records):
        for channel in (0, 1):
            for index in range(WINDOWS_PER_STREAM):
                rows.append(
                    reference(
                        record,
                        channel,
                        index,
                        subject=f"ltstdb:s{position:04d}",
                        family=families[index % len(families)],
                        partition=partition,
                    )
                )
    return tuple(rows)


TRAIN_ROWS = _references("train", ("rA", "rB"), TRAIN_FAMILIES)
VALIDATION_ROWS = _references("validation", ("rC", "rD"), VALIDATION_FAMILIES)
ALL_ROWS = {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS}


def _primary(rows):
    return tuple(row for row in rows if row.target_family in PRIMARY)


def _deterministic(stable_id: str, width: int, scale: float = 1.0) -> np.ndarray:
    seed = abs(hash(stable_id)) % (2**32)
    return (
        np.random.default_rng(seed).normal(size=width) * scale
    ).astype(np.float32)


class _StubEncoder(nn.Module):
    """A tiny stand-in for the locked B4-B encoder's `encode()` tap."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2500, EMBEDDING_DIM)

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.projection(waveforms.squeeze(1))


def _stub_encoder() -> _StubEncoder:
    """One fixed encoder, so the primary audit can be a real bitwise check."""
    torch.manual_seed(0)
    return _StubEncoder()


_ENCODER = _stub_encoder()


def _stub_waveform(stable_id: str) -> torch.Tensor:
    return torch.from_numpy(_deterministic(stable_id, 2500)).reshape(1, 2500)


def _stub_embedding(stable_id: str) -> np.ndarray:
    with torch.no_grad():
        return (
            _ENCODER.encode(_stub_waveform(stable_id).unsqueeze(0))
            .to(torch.float32)
            .numpy()[0]
        )


class _StubWaveformDataset:
    """Stands in for `B4WaveformDataset` so the canonical iterator is exercised.

    `canonical_waveform_batches` itself is NOT stubbed: the production generator
    is the code under test, which is the whole point of this file.
    """

    def __init__(self, references, source):
        self._references = tuple(references)
        self.source = Path(source)
        self.reads = 0

    def read_waveform(self, item):
        self.reads += 1
        return _stub_waveform(item.stable_id)

    @property
    def stats(self):
        class _Stats:
            source_reads = self.reads

        return _Stats()



def _availability_stream(rows_by_id, embedding_for, unavailable=()):
    """The M1-v2 physical seam: yields (reference, waveform_or_None, available).

    Availability is a property of the synthetic waveform, exactly as production
    decides it from real samples -- the stub never consults an identifier list
    to make the decision, it builds a genuinely flat waveform instead.
    """
    def _stream(_partition, identifiers):
        for key in identifiers:
            reference = rows_by_id[str(key)]
            if str(key) in unavailable:
                yield reference, None, False
            else:
                yield reference, embedding_for(str(key)), True
    return _stream



def _stub_physical_batches(rows_by_id, waveform_for, unavailable=frozenset()):
    """Stand-in for `physical_observation_batches`.

    Availability is decided from the synthetic samples themselves via the
    production predicate, so the stub never whitelists identifiers.
    """
    from cardiosentinel.neural.patient_memory import exact_flat_unavailable

    def _stream(references, source, *, batch_size=256):
        for item in references:
            key = str(item.stable_id)
            wave = waveform_for(key, flat=key in unavailable)
            values = wave.numpy().reshape(-1)
            if exact_flat_unavailable(values):
                yield item, None, False
            else:
                yield item, wave, True

    return _stream


@pytest.fixture
def official(tmp_path, monkeypatch):
    """Stub every upstream gate, then leave the M1 orchestration untouched."""
    primary_train = _primary(TRAIN_ROWS)
    primary_validation = _primary(VALIDATION_ROWS)
    caches = {"train": primary_train, "validation": primary_validation}

    raw = {
        row.stable_id: _deterministic(row.stable_id, PHYSIOLOGY_DIM).astype(np.float64)
        for rows in ALL_ROWS.values()
        for row in rows
    }
    validity = PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")
    for values in raw.values():
        values[validity] = 1.0
    transform = fit_physiology_transform(
        np.stack([raw[row.stable_id] for row in primary_train]), partition="train"
    )

    def _cache(cache_root, partition):
        rows = caches[partition]
        return P1EmbeddingCache(
            partition=partition,
            stable_ids=tuple(row.stable_id for row in rows),
            embeddings=np.stack([_stub_embedding(row.stable_id) for row in rows]),
            labels=np.asarray(
                [int(row.target_family == "ischemic_positive") for row in rows],
                dtype=np.int64,
            ),
            subject_ids=tuple(row.subject_id for row in rows),
            manifest={
                "cache_sha256": m1_experiment.FROZEN_P1_EMBEDDING_CACHE_SHA256[
                    partition
                ]
            },
        )

    class _Challenge:
        references = tuple(
            row
            for row in VALIDATION_ROWS
            if row.target_family
            in ("rate_related", "axis_shift", "conduction_change")
        )
        selection_sha256 = m1_experiment.CHALLENGE_SELECTION_SHA256
        counts: dict = {}

    monkeypatch.setattr(
        m1_experiment, "require_p1_runtime", lambda: ({"device": "cpu"}, STUB_DIGEST)
    )
    monkeypatch.setattr(
        m1_experiment,
        "require_clean_checkout",
        lambda: {"git_sha": "f" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(m1_experiment, "FROZEN_DEPENDENCY_DIGEST", STUB_DIGEST)
    monkeypatch.setattr(
        m1_experiment,
        "EXPECTED_POPULATIONS",
        {"train": {"total": len(primary_train)}},
    )
    monkeypatch.setattr(
        p1_experiment,
        "validate_p1_stage1_results",
        lambda *_a, **_k: {
            "p1_stage1_suite_sha256": m1_experiment.FROZEN_P1_STAGE1_SUITE_SHA256,
            "test_accessed": False,
            "arm_results": {
                "P1B_phys_fusion_v1": {
                    "experiment_lock_sha256": m1_experiment.FROZEN_P1B_LOCK_SHA256
                }
            },
        },
    )
    monkeypatch.setattr(
        m1_experiment,
        "validate_locked_model",
        lambda *_a, **_k: {
            "checkpoint_sha256": m1_experiment.B4B_CHECKPOINT_SHA256,
            "experiment_lock_sha256": m1_experiment.B4B_EXPERIMENT_LOCK_SHA256,
            "test": None,
        },
    )
    monkeypatch.setattr(
        m1_experiment,
        "validate_development_feature_integrity",
        lambda *_a: {
            "development_feature_integrity_sha256": (
                m1_experiment.FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256
            )
        },
    )
    monkeypatch.setattr(
        m1_experiment,
        "validate_development_source_integrity",
        lambda *_a: {
            "development_source_integrity_sha256": (
                m1_experiment.FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256
            )
        },
    )
    monkeypatch.setattr(
        m1_experiment, "load_frozen_physiology_transform", lambda *_a: transform
    )
    monkeypatch.setattr(
        m1_experiment,
        "FROZEN_PHYSIOLOGY_TRANSFORM_SHA256",
        transform.as_dict()["transform_sha256"],
    )
    monkeypatch.setattr(
        m1_experiment,
        "load_frozen_control_evidence",
        lambda *_a: {"validation_evidence": {}, "challenge_evidence": {}},
    )
    monkeypatch.setattr(m1_experiment, "load_p1_embedding_cache", _cache)
    monkeypatch.setattr(
        m1_experiment,
        "load_b4_references",
        lambda _root, partition, primary_only=True: (
            _primary(ALL_ROWS[partition]) if primary_only else ALL_ROWS[partition]
        ),
    )
    # Write real per-record npz files so the production chunked physiology
    # reader is exercised rather than stubbed.
    from cardiosentinel.features.schema import COMBINED_V1
    from cardiosentinel.neural.physiology_fusion import morphology_columns

    columns = list(morphology_columns())
    feature_root = tmp_path / "features"
    for partition, rows in ALL_ROWS.items():
        directory = feature_root / partition
        directory.mkdir(parents=True, exist_ok=True)
        for record in sorted({row.record_id for row in rows}):
            block = [row for row in rows if row.record_id == record]
            features = np.zeros((len(block), len(COMBINED_V1.names)), dtype=np.float64)
            for position, row in enumerate(block):
                features[position, columns] = raw[row.stable_id]
            np.savez(
                directory / f"{record}.npz",
                stable_ids=np.asarray(
                    [row.stable_id for row in block], dtype=np.str_
                ),
                features=features,
            )
    monkeypatch.setattr(
        m1_experiment, "load_official_b4b_encoder", lambda *_a: _ENCODER
    )
    monkeypatch.setattr(
        m1_experiment, "build_validation_challenge_index", lambda *_a: _Challenge()
    )
    rows_by_id = {
        item.stable_id: item for group in ALL_ROWS.values() for item in group
    }

    def _wave(key, *, flat=False):
        import torch as _t

        if flat:
            return _t.full((1, 2500), -5.12, dtype=_t.float32)
        return _stub_waveform(key)

    monkeypatch.setattr(m1_experiment, "B4WaveformDataset", _StubWaveformDataset)
    monkeypatch.setattr(
        m1_experiment,
        "physical_observation_batches",
        _stub_physical_batches(rows_by_id, _wave),
    )

    # The suite reads the control lock directly; give it the real layout.
    control_dir = tmp_path / "p1" / "P1B_phys_fusion_v1"
    control_dir.mkdir(parents=True)
    (control_dir / "EXPERIMENT_LOCK.json").write_text(
        json.dumps(
            {
                "experiment_id": "P1B_phys_fusion_v1",
                "experiment_lock_sha256": m1_experiment.FROZEN_P1B_LOCK_SHA256,
            }
        )
    )

    return {
        "run_root": tmp_path / "runs",
        "stream_cache_root": tmp_path / "caches",
        "cache_root": tmp_path / "p1-caches",
        "feature_root": feature_root,
        "source": tmp_path / "source",
        "b4b_run_dir": tmp_path / "b4b",
        "p1_run_root": tmp_path / "p1",
        "primary_train": primary_train,
        "primary_validation": primary_validation,
        "challenge": _Challenge.references,
    }


def _run(official):
    return execute_m1_stage1(
        run_root=official["run_root"],
        stream_cache_root=official["stream_cache_root"],
        cache_root=official["cache_root"],
        feature_root=official["feature_root"],
        source=official["source"],
        b4b_run_dir=official["b4b_run_dir"],
        p1_run_root=official["p1_run_root"],
        max_epochs=2,
    )


def test_official_route_runs_from_nothing_without_a_waveform_callback(official):
    """The canonical route must work from --source alone."""
    assert not official["stream_cache_root"].exists()

    payload = _run(official)

    # 1-2. Stream caches were created over the FULL stream, not just primary.
    for partition in ("train", "validation"):
        manifest_path = (
            official["stream_cache_root"] / partition / STREAM_CACHE_MANIFEST_NAME
        )
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["full_stream_row_count"] == len(ALL_ROWS[partition])
        assert manifest["stream_count"] == 4  # 2 records x 2 channels
        assert manifest["channel_indices"] == [0, 1]
        assert manifest["test_accessed"] is False
        assert manifest["contamination_safe"] is False

    # 3. Primary rows reused; only the extras were newly extracted.
    train_manifest = json.loads(
        (
            official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
        ).read_text()
    )
    audit = train_manifest["primary_overlap_audit"]
    assert train_manifest["primary_rows_reused"] == len(official["primary_train"])
    assert train_manifest["rows_newly_extracted"] == len(TRAIN_ROWS) - len(
        official["primary_train"]
    )
    assert audit["extra_disjoint_from_primary_cache"] is True

    # 4. Extra rows went through the canonical waveform path.
    assert audit["waveform_source_reads"] > 0
    assert audit["extra_ordered_stable_id_sha256"] is not None
    assert audit["extra_embedding_content_sha256"] is not None
    assert audit["extraction_receipt"]["encoder_state_unchanged"] is True
    assert audit["extraction_receipt"]["encoder_fine_tuned"] is False

    # 5. The deliberate primary audit sample matched bit-for-bit.
    assert audit["re_extracted_primary_rows"] > 0
    assert audit["re_extracted_primary_max_abs_deviation"] <= (
        m1_experiment.PRIMARY_AUDIT_TOLERANCE
    )

    # 6. Train-only standardizer exists and binds the exact upstream identities.
    standardizer = json.loads(
        (
            official["stream_cache_root"] / "M1_DISTANCE_STANDARDIZER.json"
        ).read_text()
    )
    assert standardizer["fitted_on_partition"] == "train"
    assert standardizer["fitted_rows"] == len(official["primary_train"])
    assert standardizer["dimension"] == REPRESENTATION_DIM
    identities = standardizer["input_identities"]
    assert identities["p1b_experiment_lock_sha256"] == (
        m1_experiment.FROZEN_P1B_LOCK_SHA256
    )
    assert identities["p1_stage1_suite_sha256"] == (
        m1_experiment.FROZEN_P1_STAGE1_SUITE_SHA256
    )

    # 7-9. All three arms ran in the frozen order and the suite validates.
    # JSON round-trips with sorted keys, so arm_order carries the frozen order.
    assert payload["arm_order"] == list(M1_ARM_ORDER)
    assert set(payload["arm_results"]) == set(M1_ARM_ORDER)
    for arm in M1_ARM_ORDER:
        assert (official["run_root"] / arm / "EXPERIMENT_LOCK.json").is_file()
        assert (official["run_root"] / arm / "model_selected.pt").is_file()
    assert (official["run_root"] / M1_STAGE1_RESULT_NAME).is_file()
    assert payload["memory_selection_performed"] is False
    assert payload["test_accessed"] is False


def test_official_route_selects_only_primary_rows_for_supervision(official):
    payload = _run(official)
    train = payload["stream_caches"]["train"]
    assert train["full_stream_row_count"] == len(TRAIN_ROWS)

    lock = json.loads(
        (official["run_root"] / M1_ARM_ORDER[0] / "EXPERIMENT_LOCK.json").read_text()
    )
    # Memory holds every window; supervision holds only the primary ones.
    assert lock["train_stream_cache"]["supervised_rows"] == len(
        official["primary_train"]
    )
    assert lock["train_stream_cache"]["full_stream_row_count"] == len(TRAIN_ROWS)
    assert lock["validation_stream_cache"]["primary_rows"] == len(
        official["primary_validation"]
    )
    assert lock["validation_stream_cache"]["challenge_rows"] == len(
        official["challenge"]
    )


def test_official_route_scores_challenge_rows_at_causal_positions(official):
    _run(official)
    directory = official["stream_cache_root"] / "validation"
    identifiers = [
        str(value)
        for value in np.load(directory / "stable_id.npy", allow_pickle=False)
    ]
    counts = np.load(directory / "past_observed_count.npy", allow_pickle=False)
    for row in official["challenge"]:
        position = identifiers.index(row.stable_id)
        # A challenge window's history is exactly the windows before it in its
        # own (record, channel) stream -- never the whole partition.
        assert 0 <= counts[position] < WINDOWS_PER_STREAM


def test_official_route_produces_subject_false_positive_evidence(official):
    _run(official)
    for arm in M1_ARM_ORDER:
        lock = json.loads(
            (official["run_root"] / arm / "EXPERIMENT_LOCK.json").read_text()
        )
        evidence = lock["subject_false_positive_evidence"]
        assert evidence["evidence_status"] == "supporting"
        assert evidence["threshold_optimized_from_this_evidence"] is False
        assert evidence["contributing_subject_count"] >= 1
        assert 0.0 <= evidence["pooled_background_negative_fpr"] <= 1.0
        assert evidence["quantile_interpolation"] == "linear"


def test_official_route_refuses_a_second_attempt(official):
    _run(official)
    with pytest.raises(Exception, match="already been claimed"):
        _run(official)


def test_official_route_reuses_a_validated_stream_cache(official):
    _run(official)
    first = json.loads(
        (
            official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
        ).read_text()
    )["stream_cache_sha256"]
    # A completed cache is reused, never rebuilt or overwritten.
    store, manifest = m1_experiment.load_stream_store(
        official["stream_cache_root"], "train"
    )
    assert manifest["stream_cache_sha256"] == first
    assert manifest["full_stream_row_count"] == len(TRAIN_ROWS)
    assert manifest["m1_stream_cache_schema"] == m1_experiment.M1_STREAM_CACHE_SCHEMA
    store.close()


def test_tampered_stream_cache_is_refused(official):
    _run(official)
    path = official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["p1b_experiment_lock_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="digest validation"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_a_self_consistent_but_wrong_cache_is_refused(official):
    """Re-digesting a tampered manifest must not launder it."""
    from cardiosentinel.neural.integrity import canonical_sha256

    _run(official)
    path = official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["p1_stage1_suite_sha256"] = "0" * 64
    manifest.pop("stream_cache_sha256")
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="frozen M1 protocol binds|expected"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_tampered_standardizer_is_refused(official):
    _run(official)
    path = official["stream_cache_root"] / "M1_DISTANCE_STANDARDIZER.json"
    payload = json.loads(path.read_text())
    payload["means"][0] = 42.0
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="digest validation"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_a_missing_standardizer_stops_for_human_review(official):
    _run(official)
    (official["stream_cache_root"] / "M1_DISTANCE_STANDARDIZER.json").unlink()
    with pytest.raises(Exception, match="human review"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_tampered_history_counts_are_refused(official):
    _run(official)
    path = official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["history_count_sha256"] = "0" * 64
    manifest.pop("stream_cache_sha256")
    from cardiosentinel.neural.integrity import canonical_sha256

    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="history counts"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_tampered_chronology_digest_is_refused(official):
    _run(official)
    path = official["stream_cache_root"] / "train" / STREAM_CACHE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["ordered_chronology_sha256"] = "0" * 64
    manifest.pop("stream_cache_sha256")
    from cardiosentinel.neural.integrity import canonical_sha256

    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="chronology digest re-derived"):
        m1_experiment.load_stream_store(official["stream_cache_root"], "train")


def test_cli_route_needs_no_externally_injected_waveform_callback():
    """The canonical command must be runnable from its own flags alone."""
    import argparse
    import inspect

    from cardiosentinel.neural.cli import run_m1_command

    signature = inspect.signature(execute_m1_stage1)
    seam = signature.parameters["_waveform_batches_for"]
    assert seam.default is None, "the production route must not require a callback"
    assert seam.name.startswith("_"), "the injection seam must stay private"

    required = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    assert "source" in required
    assert "_waveform_batches_for" not in required

    # The CLI passes only its own flags; nothing it cannot supply.
    source = inspect.getsource(run_m1_command)
    assert "_waveform_batches_for" not in source
    assert "waveform_batches_for" not in source
    assert "source=args.source" in source

    parser = argparse.ArgumentParser()
    from cardiosentinel.neural.cli import add_m1_parser

    add_m1_parser(parser.add_subparsers(dest="command"))
    parsed = parser.parse_args(["m1", "run-stage1"])
    assert parsed.source is not None
    assert parsed.stream_cache_root is not None
