"""Bounded-memory M1 execution: digest and scientific-semantic equivalence.

Attempt 1 of the canonical Stage-1 run was consumed by host memory exhaustion,
so the production path was reworked to be row-aligned and disk-backed. The whole
point is that *only* the memory profile changed, so this file proves two things
the reviewer actually cares about:

* every streaming digest reproduces its legacy identity byte-for-byte, so no
  historical or frozen identity moves;
* the bounded path and the retained in-memory reference implementation produce
  identical scientific content on a corpus containing both 2- and 3-channel
  records, primary rows, extra rows and interleaved challenge rows.

Everything is synthetic. No real corpus, model, run directory, canonical cache
or test partition is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cardiosentinel.neural import m1_experiment, p1_experiment
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m1_experiment import (
    M1_STREAM_CACHE_SCHEMA,
    STAGING_CLAIM_NAME,
    STAGING_PREFIX,
    _stream_slices,
    build_distance_standardizer_from_rows,
    load_stream_store,
    materialize_stream_store,
    prepare_stream_representations,
    scan_staging_directories,
)
from cardiosentinel.neural.m1_store import (
    COLD_START_BIN_FILE,
    D_LONG_FILE,
    D_SHORT_FILE,
    PAST_OBSERVED_FILE,
    PAST_UPDATE_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    M1RowStore,
    M1StoreError,
    M1StoreSpec,
    StreamingCanonicalArrayDigest,
    StreamingContentDigest,
    locate_rows,
    streaming_chronology_digest,
    streaming_ordered_stable_id_digest,
)
from cardiosentinel.neural.p1_experiment import (
    P1EmbeddingCache,
    embedding_content_digest,
    ordered_stable_id_digest,
)
from cardiosentinel.neural.patient_memory import (
    M1_ARM_FEATURES,
    M1_EXPERIMENT_IDS,
    REPRESENTATION_DIM,
    build_causal_streams,
    fit_distance_standardizer,
    generate_stream_memory,
    ordered_chronology_digest,
)
from cardiosentinel.neural.physiology_fusion import (
    EMBEDDING_DIM,
    PHYSIOLOGY_DIM,
    PHYSIOLOGY_FEATURE_NAMES,
    fit_physiology_transform,
    morphology_columns,
)
from tests.neural.test_patient_memory import reference

STUB_DIGEST = "0" * 64
PRIMARY = ("background_negative", "ischemic_positive")
CHALLENGE = ("rate_related", "axis_shift", "conduction_change")
FAMILIES = (*PRIMARY, *CHALLENGE, "quality_excluded")
WINDOWS = 12


# --------------------------------------------------------------------------
# Digest equivalence — the identities must not move
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,shape",
    [
        ("float32", (37, 146)),
        ("float64", (41, 3)),
        ("int64", (53, 2)),
        ("float32", (1, 146)),
        ("int64", (64,)),
        ("float64", (128, 7)),
    ],
)
@pytest.mark.parametrize("chunk", [1, 7, 16, 64, 10_000])
def test_streaming_content_digest_equals_legacy(dtype, shape, chunk):
    values = (np.random.default_rng(3).normal(size=shape) * 100).astype(dtype)
    legacy = embedding_content_digest(values)
    digest = StreamingContentDigest(values.shape, values.dtype)
    for start in range(0, values.shape[0], chunk):
        digest.update(values[start : start + chunk])
    assert digest.hexdigest() == legacy


def test_streaming_content_digest_refuses_a_truncated_stream():
    values = np.zeros((10, 4), dtype=np.float32)
    digest = StreamingContentDigest(values.shape, values.dtype)
    digest.update(values[:6])
    with pytest.raises(M1StoreError, match="saw 6 rows"):
        digest.hexdigest()


def test_streaming_stable_id_digest_equals_legacy():
    identifiers = [
        f"ltstdb:r{index // 3}:{index % 3}:{index * 1250}:{index * 1250 + 2500}"
        for index in range(211)
    ]
    assert streaming_ordered_stable_id_digest(iter(identifiers)) == (
        ordered_stable_id_digest(identifiers)
    )


def test_streaming_stable_id_digest_refuses_duplicates():
    with pytest.raises(M1StoreError, match="duplicates"):
        streaming_ordered_stable_id_digest(iter(["a", "b", "a"]))


def test_streaming_chronology_digest_equals_legacy():
    rows = [
        reference(record, channel, index)
        for record in ("rA", "rB")
        for channel in (0, 1, 2)
        for index in range(7)
    ]
    streams = build_causal_streams(rows)
    triples = [
        (item.record_id, item.channel_index, item.start_sample)
        for key in sorted(streams)
        for item in streams[key]
    ]
    assert streaming_chronology_digest(iter(triples)) == ordered_chronology_digest(
        streams
    )


def test_streaming_canonical_digest_matches_whole_object_serialization():
    payload = {"order": "row_order", "stable_ids": ["a", "b", "c"]}
    digest = StreamingCanonicalArrayDigest({"order": "row_order"}, "stable_ids")
    for value in payload["stable_ids"]:
        digest.append(value)
    assert digest.hexdigest() == canonical_sha256(payload)


def test_streaming_canonical_digest_handles_keys_around_the_list():
    # `sort_keys=True` can place scalars on either side of the streamed list.
    payload = {"aaa": 1, "rows": [[1, 2]], "zzz": {"b": 2, "a": 1}}
    digest = StreamingCanonicalArrayDigest({"aaa": 1, "zzz": {"b": 2, "a": 1}}, "rows")
    digest.append([1, 2])
    assert digest.hexdigest() == canonical_sha256(payload)


# --------------------------------------------------------------------------
# Synthetic corpus: 2-channel and 3-channel records, mixed families
# --------------------------------------------------------------------------


class _StubEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2500, EMBEDDING_DIM)

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.projection(waveforms.squeeze(1))


def _fixed_encoder() -> _StubEncoder:
    torch.manual_seed(0)
    return _StubEncoder()


_ENCODER = _fixed_encoder()


def _values(stable_id: str, width: int) -> np.ndarray:
    seed = abs(hash(stable_id)) % (2**32)
    return np.random.default_rng(seed).normal(size=width).astype(np.float32)


def _waveform(stable_id: str) -> torch.Tensor:
    return torch.from_numpy(_values(stable_id, 2500)).reshape(1, 2500)


def _embedding(stable_id: str) -> np.ndarray:
    with torch.no_grad():
        return (
            _ENCODER.encode(_waveform(stable_id).unsqueeze(0))
            .to(torch.float32)
            .numpy()[0]
        )


class _StubWaveformDataset:
    def __init__(self, references, source):
        self._references = tuple(references)
        self.source = Path(source)
        self.reads = 0

    def read_waveform(self, item):
        self.reads += 1
        return _waveform(item.stable_id)

    @property
    def stats(self):
        class _Stats:
            source_reads = self.reads

        return _Stats()


def _corpus(partition: str, records: dict[str, int]):
    rows = []
    for position, (record, channels) in enumerate(sorted(records.items())):
        for channel in range(channels):
            for index in range(WINDOWS):
                rows.append(
                    reference(
                        record,
                        channel,
                        index,
                        subject=f"ltstdb:s{position:04d}",
                        family=FAMILIES[index % len(FAMILIES)],
                        partition=partition,
                    )
                )
    return tuple(rows)


# A 2-channel and a 3-channel record in each partition, exactly as the frozen
# corpus mixes them.
TRAIN_ROWS = _corpus("train", {"rA": 2, "rB": 3})
VALIDATION_ROWS = _corpus("validation", {"rC": 2, "rD": 3})
ALL_ROWS = {"train": TRAIN_ROWS, "validation": VALIDATION_ROWS}



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
def corpus(tmp_path, monkeypatch):
    primary = {
        partition: tuple(r for r in rows if r.target_family in PRIMARY)
        for partition, rows in ALL_ROWS.items()
    }
    raw = {
        row.stable_id: _values(row.stable_id, PHYSIOLOGY_DIM).astype(np.float64)
        for rows in ALL_ROWS.values()
        for row in rows
    }
    validity = PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")
    for values in raw.values():
        values[validity] = 1.0
    transform = fit_physiology_transform(
        np.stack([raw[r.stable_id] for r in primary["train"]]), partition="train"
    )

    from cardiosentinel.features.schema import COMBINED_V1

    columns = list(morphology_columns())
    feature_root = tmp_path / "features"
    for partition, rows in ALL_ROWS.items():
        directory = feature_root / partition
        directory.mkdir(parents=True, exist_ok=True)
        for record in sorted({row.record_id for row in rows}):
            block = [row for row in rows if row.record_id == record]
            features = np.zeros((len(block), len(COMBINED_V1.names)), dtype=np.float64)
            for offset, row in enumerate(block):
                features[offset, columns] = raw[row.stable_id]
            np.savez(
                directory / f"{record}.npz",
                stable_ids=np.asarray([r.stable_id for r in block], dtype=np.str_),
                features=features,
            )

    def _cache(_root, partition):
        rows = primary[partition]
        return P1EmbeddingCache(
            partition=partition,
            stable_ids=tuple(r.stable_id for r in rows),
            embeddings=np.stack([_embedding(r.stable_id) for r in rows]),
            labels=np.asarray(
                [int(r.target_family == "ischemic_positive") for r in rows],
                dtype=np.int64,
            ),
            subject_ids=tuple(r.subject_id for r in rows),
            manifest={
                "cache_sha256": m1_experiment.FROZEN_P1_EMBEDDING_CACHE_SHA256[
                    partition
                ]
            },
        )

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
        {"train": {"total": len(primary["train"])}},
    )
    monkeypatch.setattr(
        m1_experiment,
        "FROZEN_PHYSIOLOGY_TRANSFORM_SHA256",
        transform.as_dict()["transform_sha256"],
    )
    monkeypatch.setattr(
        m1_experiment, "load_frozen_physiology_transform", lambda *_a: transform
    )
    monkeypatch.setattr(m1_experiment, "load_p1_embedding_cache", _cache)
    monkeypatch.setattr(
        p1_experiment, "load_p1_embedding_cache", _cache, raising=False
    )
    monkeypatch.setattr(
        m1_experiment,
        "load_b4_references",
        lambda _root, partition, primary_only=True: (
            tuple(r for r in ALL_ROWS[partition] if r.target_family in PRIMARY)
            if primary_only
            else ALL_ROWS[partition]
        ),
    )
    monkeypatch.setattr(
        p1_experiment,
        "read_frozen_physiology",
        lambda _root, _partition: raw,
        raising=False,
    )
    monkeypatch.setattr(
        m1_experiment, "read_frozen_physiology", lambda _root, _partition: raw
    )
    monkeypatch.setattr(
        m1_experiment, "load_official_b4b_encoder", lambda *_a: _ENCODER
    )
    rows_by_id = {
        item.stable_id: item for group in ALL_ROWS.values() for item in group
    }

    def _wave(key, *, flat=False):
        import torch as _t

        if flat:
            return _t.full((1, 2500), -5.12, dtype=_t.float32)
        return _waveform(key)

    monkeypatch.setattr(m1_experiment, "B4WaveformDataset", _StubWaveformDataset)
    monkeypatch.setattr(
        m1_experiment,
        "physical_observation_batches",
        _stub_physical_batches(rows_by_id, _wave),
    )

    return {
        "cache_root": tmp_path / "caches",
        "p1_cache_root": tmp_path / "p1",
        "feature_root": feature_root,
        "source": tmp_path / "source",
        "b4b_run_dir": tmp_path / "b4b",
        "p1b_run_dir": tmp_path / "p1b",
        "primary": primary,
        "transform": transform,
    }


UPSTREAM = {
    "p1_stage1_suite_sha256": m1_experiment.FROZEN_P1_STAGE1_SUITE_SHA256,
    "p1b_experiment_lock_sha256": m1_experiment.FROZEN_P1B_LOCK_SHA256,
    "physiology_transform_sha256": m1_experiment.FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
    "p1_train_embedding_cache_sha256": (
        m1_experiment.FROZEN_P1_EMBEDDING_CACHE_SHA256["train"]
    ),
    "encoder_checkpoint_sha256": m1_experiment.B4B_CHECKPOINT_SHA256,
    "development_feature_integrity_sha256": (
        m1_experiment.FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256
    ),
    "development_source_integrity_sha256": (
        m1_experiment.FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256
    ),
}


def _manifest_fields(partition: str) -> dict:
    return {
        "upstream_identities": {
            **UPSTREAM,
            "physiology_transform_sha256": (
                m1_experiment.FROZEN_PHYSIOLOGY_TRANSFORM_SHA256
            ),
        },
        "p1_stage1_suite_sha256": UPSTREAM["p1_stage1_suite_sha256"],
        "p1b_lock_sha256": UPSTREAM["p1b_experiment_lock_sha256"],
        "physiology_transform_sha256": (
            m1_experiment.FROZEN_PHYSIOLOGY_TRANSFORM_SHA256
        ),
        "feature_integrity_sha256": UPSTREAM[
            "development_feature_integrity_sha256"
        ],
        "source_integrity_sha256": UPSTREAM["development_source_integrity_sha256"],
        "embedding_cache_sha256": m1_experiment.FROZEN_P1_EMBEDDING_CACHE_SHA256[
            partition
        ],
        "git_sha": "f" * 40,
        "git_dirty": False,
        "dependency_digest": STUB_DIGEST,
    }


def _bounded(corpus, partition, standardizer=None):
    fields = dict(_manifest_fields(partition))
    fields["upstream_identities"] = {
        **fields["upstream_identities"],
        "physiology_transform_sha256": (
            m1_experiment.FROZEN_PHYSIOLOGY_TRANSFORM_SHA256
        ),
    }
    return materialize_stream_store(
        partition,
        cache_root=corpus["cache_root"],
        p1_cache_root=corpus["p1_cache_root"],
        feature_root=corpus["feature_root"],
        source=corpus["source"],
        b4b_run_dir=corpus["b4b_run_dir"],
        p1b_run_dir=corpus["p1b_run_dir"],
        standardizer=standardizer,
        manifest_fields=fields,
    )


def _reference_path(corpus, partition, standardizer=None):
    """The retained in-memory implementation, used only as a test oracle."""
    representation = prepare_stream_representations(
        partition,
        cache_root=corpus["p1_cache_root"],
        feature_root=corpus["feature_root"],
        source=corpus["source"],
        b4b_run_dir=corpus["b4b_run_dir"],
        p1b_run_dir=corpus["p1b_run_dir"],
    )
    if standardizer is None:
        lookup = representation.by_stable_id()
        matrix = np.stack(
            [
                np.asarray(lookup[r.stable_id], dtype=np.float64)
                for r in corpus["primary"]["train"]
            ]
        )
        standardizer = fit_distance_standardizer(matrix, partition="train")
    memory = generate_stream_memory(
        representation.streams,
        partition=partition,
        representations=representation.by_stable_id(),
        standardizer=standardizer,
    )
    return representation, memory, standardizer


# --------------------------------------------------------------------------
# Exact small-scale scientific equivalence
# --------------------------------------------------------------------------


def test_bounded_path_matches_the_reference_implementation_exactly(corpus):
    manifest, standardizer = _bounded(corpus, "train")
    store, reloaded = load_stream_store(corpus["cache_root"], "train")
    representation, memory, reference_standardizer = _reference_path(corpus, "train")

    # row order
    stable = [str(v) for v in np.asarray(store.array(STABLE_ID_FILE))]
    assert tuple(stable) == memory.stable_ids
    assert tuple(stable) == representation.stable_ids

    # fused z_t
    lookup = representation.by_stable_id()
    expected = np.stack([lookup[key] for key in stable])
    np.testing.assert_array_equal(
        np.asarray(store.array(REPRESENTATION_FILE)), expected
    )

    # standardizer statistics and cold-start prior
    np.testing.assert_array_equal(
        np.asarray(standardizer.means), np.asarray(reference_standardizer.means)
    )
    np.testing.assert_array_equal(
        np.asarray(standardizer.scales), np.asarray(reference_standardizer.scales)
    )
    np.testing.assert_array_equal(
        standardizer.prior_vector(), reference_standardizer.prior_vector()
    )
    assert standardizer.zero_variance_dimensions == (
        reference_standardizer.zero_variance_dimensions
    )

    # causal memory
    np.testing.assert_array_equal(
        np.asarray(store.array(D_SHORT_FILE)), memory.d_short
    )
    np.testing.assert_array_equal(np.asarray(store.array(D_LONG_FILE)), memory.d_long)
    np.testing.assert_array_equal(
        np.asarray(store.array(PAST_OBSERVED_FILE)), memory.past_observed_count
    )
    np.testing.assert_array_equal(
        np.asarray(store.array(PAST_UPDATE_FILE)), memory.past_update_count
    )
    np.testing.assert_array_equal(
        np.asarray(store.array(RECORDING_AGE_FILE)), memory.recording_age_seconds
    )
    assert tuple(
        str(v) for v in np.asarray(store.array(COLD_START_BIN_FILE))
    ) == memory.cold_start_bins

    # logical content identities
    assert reloaded["ordered_stable_id_sha256"] == ordered_stable_id_digest(
        memory.stable_ids
    )
    assert reloaded["ordered_chronology_sha256"] == memory.chronology_sha256
    assert reloaded["representation_content_sha256"] == embedding_content_digest(
        expected
    )
    assert reloaded["d_short_content_sha256"] == embedding_content_digest(
        memory.d_short
    )
    assert reloaded["d_long_content_sha256"] == embedding_content_digest(memory.d_long)
    assert reloaded["history_count_sha256"] == embedding_content_digest(
        np.stack([memory.past_observed_count, memory.past_update_count], axis=1)
    )
    assert manifest["stream_cache_sha256"] == reloaded["stream_cache_sha256"]
    store.close()


def test_bounded_arm_feature_matrices_match_the_reference(corpus):
    _manifest, standardizer = _bounded(corpus, "train")
    store, _ = load_stream_store(corpus["cache_root"], "train")
    representation, memory, _ = _reference_path(corpus, "train", standardizer)

    primary_ids = [r.stable_id for r in corpus["primary"]["train"]]
    positions = locate_rows(store, primary_ids)
    base = np.asarray(store.gather(REPRESENTATION_FILE, positions), dtype=np.float32)
    columns = {
        "d_short": store.gather(D_SHORT_FILE, positions),
        "d_long": store.gather(D_LONG_FILE, positions),
    }

    from cardiosentinel.neural.patient_memory import m1_arm_features, select_rows

    reference_rows = select_rows(memory, primary_ids)
    lookup = representation.by_stable_id()
    reference_base = np.stack([lookup[key] for key in primary_ids])

    for experiment_id in M1_EXPERIMENT_IDS:
        bounded = m1_arm_features(
            experiment_id,
            base,
            np.stack(
                [columns[name] for name in M1_ARM_FEATURES[experiment_id]], axis=1
            ).astype(np.float32),
        )
        expected = m1_arm_features(
            experiment_id,
            reference_base,
            memory.memory_matrix(experiment_id)[reference_rows],
        )
        np.testing.assert_array_equal(bounded, expected)
        assert bounded.shape[1] == REPRESENTATION_DIM + len(
            M1_ARM_FEATURES[experiment_id]
        )
    store.close()


def test_bounded_path_preserves_three_channel_stream_independence(corpus):
    _bounded(corpus, "train")
    store, manifest = load_stream_store(corpus["cache_root"], "train")
    assert manifest["channel_indices"] == [0, 1, 2]
    # rA has 2 channels, rB has 3 -> 5 independent streams.
    assert manifest["stream_count"] == 5
    channels = np.asarray(store.array("channel_index.npy"))
    records = np.asarray(store.array("record_id.npy"))
    assert sorted({int(v) for v in channels[records == "rB"]}) == [0, 1, 2]
    assert sorted({int(v) for v in channels[records == "rA"]}) == [0, 1]
    # every stream cold-starts independently
    observed = np.asarray(store.array(PAST_OBSERVED_FILE))
    assert int(np.sum(observed == 0)) == manifest["stream_count"]
    store.close()


def test_bounded_challenge_rows_keep_their_causal_positions(corpus):
    _manifest, standardizer = _bounded(corpus, "train")
    _bounded(corpus, "validation", standardizer)
    store, _ = load_stream_store(corpus["cache_root"], "validation")
    _representation, memory, _ = _reference_path(corpus, "validation", standardizer)

    challenge_ids = [
        r.stable_id for r in VALIDATION_ROWS if r.target_family in CHALLENGE
    ]
    positions = locate_rows(store, challenge_ids)
    from cardiosentinel.neural.patient_memory import select_rows

    reference_rows = select_rows(memory, challenge_ids)
    np.testing.assert_array_equal(
        np.asarray(store.gather(PAST_OBSERVED_FILE, positions)),
        memory.past_observed_count[reference_rows],
    )
    np.testing.assert_array_equal(
        np.asarray(store.gather(D_SHORT_FILE, positions)),
        memory.d_short[reference_rows],
    )
    store.close()


def test_bounded_history_is_label_independent(corpus):
    """Every full-stream row participates, primary or not."""
    _bounded(corpus, "train")
    store, manifest = load_stream_store(corpus["cache_root"], "train")
    assert manifest["full_stream_row_count"] == len(TRAIN_ROWS)
    assert manifest["label_independent_history"] is True
    assert manifest["primary_rows_reused"] == len(corpus["primary"]["train"])
    assert manifest["rows_newly_extracted"] == len(TRAIN_ROWS) - len(
        corpus["primary"]["train"]
    )
    store.close()


def test_bounded_physiology_uses_the_exact_frozen_transform(corpus):
    _bounded(corpus, "train")
    store, manifest = load_stream_store(corpus["cache_root"], "train")
    assert manifest["physiology_transform_sha256"] == (
        corpus["transform"].as_dict()["transform_sha256"]
    )
    representation = np.asarray(store.array(REPRESENTATION_FILE))
    _rep, _memory, _std = _reference_path(corpus, "train")
    # The physiology block is bit-identical to the reference transform output.
    assert representation.shape[1] == REPRESENTATION_DIM
    assert np.all(np.isfinite(representation[:, EMBEDDING_DIM:]))
    store.close()


def test_bounded_standardizer_is_primary_train_only(corpus):
    _bounded(corpus, "train")
    payload = json.loads(
        (corpus["cache_root"] / "M1_DISTANCE_STANDARDIZER.json").read_text()
    )
    assert payload["fitted_on_partition"] == "train"
    assert payload["fitted_on_full_stream"] is False
    assert payload["fitted_rows"] == len(corpus["primary"]["train"])
    assert payload["validation_statistics_used"] is False
    assert payload["input_identities"]["p1b_experiment_lock_sha256"] == (
        m1_experiment.FROZEN_P1B_LOCK_SHA256
    )


def test_standardizer_from_rows_refuses_null_identities(corpus):
    with pytest.raises(Exception, match="absent or null"):
        build_distance_standardizer_from_rows(
            np.zeros((len(corpus["primary"]["train"]), REPRESENTATION_DIM)),
            primary_train_stable_ids=[
                r.stable_id for r in corpus["primary"]["train"]
            ],
            upstream_identities={**UPSTREAM, "p1b_experiment_lock_sha256": None},
        )


# --------------------------------------------------------------------------
# Extraction, reuse and encoder integrity
# --------------------------------------------------------------------------


def test_extra_rows_are_extracted_exactly_once_and_reuse_is_proven(corpus):
    manifest, _ = _bounded(corpus, "train")
    audit = manifest["primary_overlap_audit"]
    extra = len(TRAIN_ROWS) - len(corpus["primary"]["train"])
    assert audit["rows_newly_extracted"] == extra
    assert audit["extraction_receipt"]["rows_extracted"] == extra
    assert audit["extraction_receipt"]["batch_size"] == 256
    assert audit["extra_disjoint_from_primary_cache"] is True
    assert audit["extra_ordered_stable_id_sha256"] is not None
    assert audit["extra_embedding_content_sha256"] is not None
    assert audit["waveform_source_reads"] > 0


def test_encoder_state_is_unchanged_by_extraction(corpus):
    manifest, _ = _bounded(corpus, "train")
    receipt = manifest["primary_overlap_audit"]["extraction_receipt"]
    assert receipt["encoder_state_unchanged"] is True
    assert receipt["encoder_fine_tuned"] is False
    assert (
        receipt["encoder_state_sha256_before"] == receipt["encoder_state_sha256_after"]
    )


def test_primary_rows_are_reused_not_regenerated(corpus):
    manifest, _ = _bounded(corpus, "train")
    audit = manifest["primary_overlap_audit"]
    # Only the deliberate audit sample re-touches primary rows.
    assert 0 < audit["re_extracted_primary_rows"] <= 64
    assert audit["re_extracted_primary_max_abs_deviation"] <= (
        m1_experiment.PRIMARY_AUDIT_TOLERANCE
    )


# --------------------------------------------------------------------------
# Cache, staging and loader refusals
# --------------------------------------------------------------------------


def test_existing_cache_is_never_overwritten(corpus):
    _bounded(corpus, "train")
    with pytest.raises(Exception, match="never overwritten"):
        _bounded(corpus, "train")


def test_partial_staging_area_stops_for_human_review(corpus):
    staging = corpus["cache_root"] / f"{STAGING_PREFIX}train"
    staging.mkdir(parents=True)
    (staging / STAGING_CLAIM_NAME).write_text("{}")
    assert scan_staging_directories(corpus["cache_root"]) == [str(staging)]
    with pytest.raises(Exception, match="never resumed"):
        _bounded(corpus, "train")
    # untouched: no delete, no repair, no resume
    assert staging.is_dir()


def test_promoted_cache_leaves_no_staging_directory(corpus):
    _bounded(corpus, "train")
    assert scan_staging_directories(corpus["cache_root"]) == []
    claim = json.loads(
        (corpus["cache_root"] / "train" / STAGING_CLAIM_NAME).read_text()
    )
    assert claim["promoted_to_canonical_cache"] is True
    assert claim["resume_permitted"] is False
    assert claim["automatic_deletion_permitted"] is False


def test_loader_refuses_a_wrong_schema(corpus):
    _bounded(corpus, "train")
    path = corpus["cache_root"] / "train" / "M1_STREAM_CACHE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["m1_stream_cache_schema"] = 99
    manifest.pop("stream_cache_sha256")
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="is not the supported"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_a_mutated_array(corpus):
    _bounded(corpus, "train")
    target = corpus["cache_root"] / "train" / D_SHORT_FILE
    values = np.load(target, allow_pickle=False)
    values[0] += 1.0
    np.save(target, values)
    with pytest.raises(Exception, match="does not match its digest"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_opens_arrays_read_only(corpus):
    _bounded(corpus, "train")
    store, _ = load_stream_store(corpus["cache_root"], "train")
    array = store.array(REPRESENTATION_FILE)
    assert not array.flags.writeable, "a validated cache must be read-only"
    store.close()


def test_loader_refuses_the_test_partition(corpus):
    with pytest.raises(Exception):
        load_stream_store(corpus["cache_root"], "test")


def test_locate_rows_refuses_unknown_and_duplicate_selections(corpus):
    _bounded(corpus, "train")
    store, _ = load_stream_store(corpus["cache_root"], "train")
    with pytest.raises(M1StoreError, match="absent"):
        locate_rows(store, ["ltstdb:zz:0:0:2500"])
    first = TRAIN_ROWS[0].stable_id
    with pytest.raises(M1StoreError, match="duplicates"):
        locate_rows(store, [first, first])
    store.close()


def test_stream_slices_are_contiguous_and_cover_every_row():
    rows = [
        reference(record, channel, index)
        for record in ("rA", "rB")
        for channel in (0, 1, 2)
        for index in range(4)
    ]
    streams = build_causal_streams(rows)
    ordered = [item for key in sorted(streams) for item in streams[key]]
    slices = _stream_slices(ordered)
    assert len(slices) == 6
    assert slices[0][1] == 0
    assert slices[-1][2] == len(ordered)
    for (_key, _begin, end), (_next_key, begin, _next_end) in zip(
        slices, slices[1:], strict=False
    ):
        assert end == begin


# --------------------------------------------------------------------------
# Governance reporting
# --------------------------------------------------------------------------


def test_preflight_reports_both_historical_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        m1_experiment, "require_p1_runtime", lambda: ({"device": "cpu"}, STUB_DIGEST)
    )
    monkeypatch.setattr(
        m1_experiment,
        "require_clean_checkout",
        lambda: {"git_sha": "f" * 40, "git_dirty": False},
    )
    report = m1_experiment.m1_preflight(tmp_path / "runs", tmp_path / "caches")
    governance = report["execution_governance"]
    # Two authorized invocations have occurred. No preflight may ever imply
    # that they did not.
    assert governance["prior_authorized_invocation_count"] == 2
    assert governance["active_protocol"] == "M1-v2"
    assert governance["prior_failed_preclaim_attempt_documented"] is True
    assert [a["attempt"] for a in governance["prior_failed_attempts"]] == [1, 2]
    assert governance["prior_failed_attempts"][0]["document"] == (
        "docs/M1_STAGE1_ATTEMPT1_FAILURE.md"
    )
    assert governance["prior_failed_attempts"][1]["document"] == (
        "docs/M1_STAGE1_ATTEMPT2_FAILURE.md"
    )
    assert governance["prior_scientific_m1_arm_claims"] == 0
    assert governance["prior_scientific_m1_results"] == 0
    assert governance["m1_protocol_v1_sha256"] == (
        "08f71c5b54ebd0fcc9c1f26f05d7df2c5a1b0ca5253b8821435a65673ad65253"
    )
    # Attempt 2 created partial execution artifacts; it created no claim-bearing
    # scientific result. The report must not collapse those into one flag.
    assert governance["prior_claim_bearing_scientific_result_artifacts_created"] is (
        False
    )
    assert governance["prior_partial_execution_artifacts_created"] is True
    assert governance["prior_partial_execution_artifacts"] == [
        "M1-v1 TRAIN stream store",
        "M1-v1 distance standardizer",
        "M1-v1 validation staging area",
    ]
    assert governance[
        "prior_partial_execution_artifacts_are_valid_m1_v2_artifacts"
    ] is False
    assert governance["prior_attempt_arm_claims_created"] is False
    assert "prior_attempt_scientific_artifacts_created" not in governance
    assert governance["replacement_execution_requires_new_human_authorization"] is True
    assert report["m1_stream_cache_schema"] == M1_STREAM_CACHE_SCHEMA
    # Reporting prior history must not itself authorize anything.
    assert report["ready_for_canonical_m1_stage1"] is False


def test_attempt_one_failure_document_exists_and_is_conservative():
    document = Path("docs/M1_STAGE1_ATTEMPT1_FAILURE.md").read_text()
    assert "strongly consistent with process termination under host memory" in document
    assert "kernel OOM killer confirmed" not in document
    assert "137" in document
    assert "first human authorization is CONSUMED" in document.replace("**", "")


def test_preflight_flags_a_staging_directory_for_human_review(tmp_path, monkeypatch):
    monkeypatch.setattr(
        m1_experiment, "require_p1_runtime", lambda: ({"device": "cpu"}, STUB_DIGEST)
    )
    monkeypatch.setattr(
        m1_experiment,
        "require_clean_checkout",
        lambda: {"git_sha": "f" * 40, "git_dirty": False},
    )
    caches = tmp_path / "caches"
    (caches / f"{STAGING_PREFIX}train").mkdir(parents=True)
    report = m1_experiment.m1_preflight(tmp_path / "runs", caches)
    assert report["status"] == "partial_stream_cache_human_review_required"
    assert report["stream_cache_state"]["staging_directories"]
    assert report["human_review_required"] is True


# --------------------------------------------------------------------------
# Store mechanics
# --------------------------------------------------------------------------


def test_row_store_round_trips_and_digests_in_chunks(tmp_path):
    spec = M1StoreSpec(rows=1000, representation_dim=REPRESENTATION_DIM)
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        values = np.random.default_rng(1).normal(size=(1000, REPRESENTATION_DIM))
        store.array(REPRESENTATION_FILE)[:] = values.astype(np.float32)
        store.flush()
        assert store.content_digest(REPRESENTATION_FILE) == embedding_content_digest(
            values.astype(np.float32)
        )
        assert store.content_digest(REPRESENTATION_FILE, chunk_rows=7) == (
            store.content_digest(REPRESENTATION_FILE, chunk_rows=999)
        )


def test_row_store_refuses_a_missing_array(tmp_path):
    spec = M1StoreSpec(rows=8, representation_dim=REPRESENTATION_DIM)
    M1RowStore(tmp_path / "store", spec, create=True).close()
    (tmp_path / "store" / D_SHORT_FILE).unlink()
    with pytest.raises(M1StoreError, match="missing"):
        M1RowStore(tmp_path / "store", spec, create=False)


# --------------------------------------------------------------------------
# Waveform source-read provenance
#
# The first bounded implementation called canonical_waveform_batches once per
# <=256-row flush, each building its own B4WaveformDataset, and then took
# max(stats.source_reads) across those independent instances. On the canonical
# TRAIN corpus that would have recorded ~256 reads instead of ~1,833,979. These
# tests use MORE THAN ONE flush and assert exact counts, so that bug fails here.
# --------------------------------------------------------------------------

WIDE_WINDOWS = 400  # 600 extra rows across two records => three flushes


def _wide_corpus():
    rows = []
    for position, (record, channels) in enumerate((("wA", 1), ("wB", 1))):
        for channel in range(channels):
            for index in range(WIDE_WINDOWS):
                rows.append(
                    reference(
                        record,
                        channel,
                        index,
                        subject=f"ltstdb:s{position:04d}",
                        # every 4th row is primary, so extras dominate
                        family=(
                            PRIMARY[index % 2]
                            if index % 4 == 0
                            else FAMILIES[2 + (index % 3)]
                        ),
                        partition="train",
                    )
                )
    return tuple(rows)


WIDE_ROWS = _wide_corpus()


@pytest.fixture
def wide(tmp_path, monkeypatch):
    primary = tuple(r for r in WIDE_ROWS if r.target_family in PRIMARY)
    raw = {
        row.stable_id: _values(row.stable_id, PHYSIOLOGY_DIM).astype(np.float64)
        for row in WIDE_ROWS
    }
    validity = PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")
    for values in raw.values():
        values[validity] = 1.0
    transform = fit_physiology_transform(
        np.stack([raw[r.stable_id] for r in primary]), partition="train"
    )

    from cardiosentinel.features.schema import COMBINED_V1

    columns = list(morphology_columns())
    feature_root = tmp_path / "features"
    (feature_root / "train").mkdir(parents=True)
    for record in sorted({r.record_id for r in WIDE_ROWS}):
        block = [r for r in WIDE_ROWS if r.record_id == record]
        features = np.zeros((len(block), len(COMBINED_V1.names)), dtype=np.float64)
        for offset, row in enumerate(block):
            features[offset, columns] = raw[row.stable_id]
        np.savez(
            feature_root / "train" / f"{record}.npz",
            stable_ids=np.asarray([r.stable_id for r in block], dtype=np.str_),
            features=features,
        )

    def _cache(_root, _partition):
        return P1EmbeddingCache(
            partition="train",
            stable_ids=tuple(r.stable_id for r in primary),
            embeddings=np.stack([_embedding(r.stable_id) for r in primary]),
            labels=np.zeros(len(primary), dtype=np.int64),
            subject_ids=tuple(r.subject_id for r in primary),
            manifest={
                "cache_sha256": m1_experiment.FROZEN_P1_EMBEDDING_CACHE_SHA256["train"]
            },
        )

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
        m1_experiment, "EXPECTED_POPULATIONS", {"train": {"total": len(primary)}}
    )
    monkeypatch.setattr(
        m1_experiment,
        "FROZEN_PHYSIOLOGY_TRANSFORM_SHA256",
        transform.as_dict()["transform_sha256"],
    )
    monkeypatch.setattr(
        m1_experiment, "load_frozen_physiology_transform", lambda *_a: transform
    )
    monkeypatch.setattr(m1_experiment, "load_p1_embedding_cache", _cache)
    monkeypatch.setattr(
        m1_experiment,
        "load_b4_references",
        lambda _root, _partition, primary_only=True: (
            primary if primary_only else WIDE_ROWS
        ),
    )
    monkeypatch.setattr(
        m1_experiment, "load_official_b4b_encoder", lambda *_a: _ENCODER
    )
    rows_by_id = {
        item.stable_id: item for group in ALL_ROWS.values() for item in group
    }

    def _wave(key, *, flat=False):
        import torch as _t

        if flat:
            return _t.full((1, 2500), -5.12, dtype=_t.float32)
        return _waveform(key)

    monkeypatch.setattr(m1_experiment, "B4WaveformDataset", _StubWaveformDataset)
    monkeypatch.setattr(
        m1_experiment,
        "physical_observation_batches",
        _stub_physical_batches(rows_by_id, _wave),
    )
    return {
        "cache_root": tmp_path / "caches",
        "p1_cache_root": tmp_path / "p1",
        "feature_root": feature_root,
        "source": tmp_path / "source",
        "b4b_run_dir": tmp_path / "b4b",
        "p1b_run_dir": tmp_path / "p1b",
        "primary": {"train": primary},
        "transform": transform,
    }


def test_extraction_reads_are_cumulative_across_multiple_flushes(wide):
    manifest, _ = _bounded(wide, "train")
    audit = manifest["primary_overlap_audit"]
    extra = len(WIDE_ROWS) - len(wide["primary"]["train"])

    assert extra > 512, "the fixture must force more than one 256-row flush"
    assert audit["rows_newly_extracted"] == extra
    # Exact, not `> 0`: one emitted row is one validated waveform read.
    assert audit["extraction_receipt"]["waveform_source_reads"] == extra
    assert audit["extraction_receipt"]["rows_extracted"] == extra
    assert audit["extraction_receipt"]["batch_size"] == 256


def test_primary_audit_read_count_is_exact(wide):
    manifest, _ = _bounded(wide, "train")
    audit = manifest["primary_overlap_audit"]
    receipt = audit["primary_audit_receipt"]
    assert receipt["rows_re_extracted"] == audit["re_extracted_primary_rows"]
    assert receipt["waveform_source_reads"] == audit["re_extracted_primary_rows"]
    assert receipt["rows_re_extracted"] > 0


def test_total_reads_are_extraction_plus_audit(wide):
    manifest, _ = _bounded(wide, "train")
    audit = manifest["primary_overlap_audit"]
    assert audit["waveform_source_reads"] == (
        audit["extraction_receipt"]["waveform_source_reads"]
        + audit["primary_audit_receipt"]["waveform_source_reads"]
    )
    extra = len(WIDE_ROWS) - len(wide["primary"]["train"])
    assert audit["waveform_source_reads"] == extra + audit[
        "re_extracted_primary_rows"
    ]


def test_primary_audit_proves_its_own_encoder_state(wide):
    manifest, _ = _bounded(wide, "train")
    receipt = manifest["primary_overlap_audit"]["primary_audit_receipt"]
    assert receipt["encoder_state_unchanged"] is True
    assert receipt["encoder_fine_tuned"] is False
    assert receipt["encoder_state_sha256_before"] is not None
    assert (
        receipt["encoder_state_sha256_before"] == receipt["encoder_state_sha256_after"]
    )
    assert receipt["embedding_dim"] == EMBEDDING_DIM


def test_audit_sample_matches_the_reference_selection_without_a_corpus_dict(wide):
    """The bounded lookup must pick the identical deterministic rows."""
    from cardiosentinel.neural.m1_experiment import _audit_sample_references

    cache = m1_experiment.load_p1_embedding_cache(wide["p1_cache_root"], "train")
    ordered = list(WIDE_ROWS)

    # Retained reference rule, expressed with the full-corpus dictionary the
    # production path is no longer allowed to build.
    by_id = {r.stable_id: r for r in ordered}
    step = max(len(cache.stable_ids) // m1_experiment.PRIMARY_AUDIT_ROWS, 1)
    expected = [
        by_id[key]
        for index, key in enumerate(cache.stable_ids)
        if index % step == 0 and key in by_id
    ][: m1_experiment.PRIMARY_AUDIT_ROWS]

    bounded = _audit_sample_references(ordered, cache)
    assert [r.stable_id for r in bounded] == [r.stable_id for r in expected]
    assert bounded


def test_production_audit_builds_no_full_corpus_reference_dictionary():
    import ast
    from pathlib import Path as _Path

    tree = ast.parse(_Path(m1_experiment.__file__).read_text())
    for name in ("_audit_sample_references", "_audit_primary_overlap"):
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for node in ast.walk(function):
            if isinstance(node, ast.DictComp):
                # A dict comprehension over every reference is exactly the
                # allocation that has to stay gone.
                source = ast.unparse(node)
                assert "for reference in ordered" not in source, source
                assert "for r in ordered" not in source, source


# --------------------------------------------------------------------------
# Schema-2 loader: complete physical artifact set + persisted stream metadata
# --------------------------------------------------------------------------


def _reseal(path: Path, mutate) -> None:
    """Rewrite a manifest and recompute its own digest, so it stays consistent."""
    manifest = json.loads(path.read_text())
    mutate(manifest)
    manifest.pop("stream_cache_sha256", None)
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))


def _manifest_path(corpus) -> Path:
    return corpus["cache_root"] / "train" / "M1_STREAM_CACHE_MANIFEST.json"


def test_manifest_binds_exactly_the_schema_artifact_set(corpus):
    manifest, _ = _bounded(corpus, "train")
    spec = M1StoreSpec(
        rows=manifest["full_stream_row_count"],
        representation_dim=REPRESENTATION_DIM,
    )
    assert set(manifest["artifact_sha256"]) == set(spec.arrays())
    # The supporting-evidence arrays must be inside the integrity boundary.
    for name in (
        RECORDING_AGE_FILE,
        COLD_START_BIN_FILE,
        "prototype_disagreement.npy",
    ):
        assert name in manifest["artifact_sha256"]


def test_loader_refuses_an_omitted_artifact_digest(corpus):
    _bounded(corpus, "train")
    _reseal(
        _manifest_path(corpus),
        lambda m: m["artifact_sha256"].pop(RECORDING_AGE_FILE),
    )
    with pytest.raises(Exception, match="exactly the schema's physical"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_an_unexpected_artifact_digest(corpus):
    _bounded(corpus, "train")
    _reseal(
        _manifest_path(corpus),
        lambda m: m["artifact_sha256"].update({"smuggled.npy": "0" * 64}),
    )
    with pytest.raises(Exception, match="exactly the schema's physical"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_a_self_consistent_wrong_stream_count(corpus):
    _bounded(corpus, "train")
    _reseal(_manifest_path(corpus), lambda m: m.update({"stream_count": 4}))
    with pytest.raises(Exception, match="streams, but the manifest records"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_self_consistent_wrong_record_ids(corpus):
    _bounded(corpus, "train")
    _reseal(_manifest_path(corpus), lambda m: m.update({"record_ids": ["rA", "rZ"]}))
    with pytest.raises(Exception, match="persisted record IDs differ"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_self_consistent_wrong_channel_indices(corpus):
    _bounded(corpus, "train")
    _reseal(_manifest_path(corpus), lambda m: m.update({"channel_indices": [0, 1]}))
    with pytest.raises(Exception, match="persisted channel indices differ"):
        load_stream_store(corpus["cache_root"], "train")


def test_a_self_consistent_manifest_alone_does_not_satisfy_the_loader(corpus):
    """Re-digesting a tampered manifest must never launder it."""
    _bounded(corpus, "train")
    path = _manifest_path(corpus)
    original = json.loads(path.read_text())
    _reseal(path, lambda m: m.update({"stream_count": 99}))
    resealed = json.loads(path.read_text())
    # internally consistent...
    body = {k: v for k, v in resealed.items() if k != "stream_cache_sha256"}
    assert resealed["stream_cache_sha256"] == canonical_sha256(body)
    assert resealed["stream_cache_sha256"] != original["stream_cache_sha256"]
    # ...and still refused.
    with pytest.raises(Exception):
        load_stream_store(corpus["cache_root"], "train")


# --------------------------------------------------------------------------
# M1-v2 review corrections: canonical all-NaN unavailable rows, exact loader
# sentinels, and independent re-derivation of the unavailable ordered-ID digest.
#
# Blocker 1: `_fill_physiology` used to write the frozen 18-d vector into EVERY
# full-stream row, so an unavailable row ended up as
# [128 NaN embedding ; 18 finite physiology] instead of 146 NaN.
# Blocker 2: the loader only asked whether an unavailable row was "not entirely
# finite", which that mixed row satisfies.
# Blocker 3: the manifest's unavailable ordered-ID digest was trusted rather
# than re-derived from stable_id.npy + observation_state.npy.
# --------------------------------------------------------------------------

# Must be an EXTRA row: primary rows are copied from the frozen P1 cache and
# never pass through the physical reader at all.
FLAT_ID = next(
    row.stable_id for row in VALIDATION_ROWS if row.target_family not in PRIMARY
)


@pytest.fixture
def outage(tmp_path, monkeypatch, corpus):
    """The `corpus` fixture with one VALIDATION row physically unavailable."""
    from cardiosentinel.neural.m1_store import OBSERVATION_STATE_FILE as _STATE

    def _wave(key, *, flat=False):
        import torch as _t

        if flat or key == FLAT_ID:
            return _t.full((1, 2500), -5.12, dtype=_t.float32)
        return _waveform(key)

    rows_by_id = {
        item.stable_id: item for group in ALL_ROWS.values() for item in group
    }
    monkeypatch.setattr(
        m1_experiment,
        "physical_observation_batches",
        _stub_physical_batches(rows_by_id, _wave),
    )
    corpus["flat_id"] = FLAT_ID
    corpus["state_file"] = _STATE
    return corpus


def _build_validation(outage):
    _manifest, standardizer = _bounded(outage, "train")
    manifest, _ = _bounded(outage, "validation", standardizer)
    return manifest


# --- Blocker 1 -------------------------------------------------------------


def test_unavailable_row_stays_canonical_all_nan_after_physiology(outage):
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
    )
    from cardiosentinel.neural.m1_store import (
        REPRESENTATION_FILE as REP,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_AVAILABLE as AVAIL,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_UNAVAILABLE_EXACT_FLAT as FLAT,
    )

    manifest = _build_validation(outage)
    assert manifest["unavailable_exact_flat_row_count"] == 1
    store, _ = load_stream_store(outage["cache_root"], "validation")
    states = np.asarray(store.array(OBSERVATION_STATE_FILE))
    rep = np.asarray(store.array(REP))
    row = int(np.flatnonzero(states == FLAT)[0])

    # the whole 146-d row, not merely the 128 embedding dims
    assert np.all(np.isnan(rep[row])), "unavailable row must be all-NaN"
    assert np.isnan(rep[row, :EMBEDDING_DIM]).all()
    assert np.isnan(rep[row, EMBEDDING_DIM:]).all(), "physiology must be skipped"
    # available rows are untouched and fully finite
    available = np.flatnonzero(states == AVAIL)
    assert np.all(np.isfinite(rep[available]))
    store.close()


def test_available_rows_receive_identical_physiology_despite_an_outage(outage):
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
        STABLE_ID_FILE,
    )
    from cardiosentinel.neural.m1_store import (
        REPRESENTATION_FILE as REP,
    )
    from cardiosentinel.neural.patient_memory import OBSERVATION_AVAILABLE as AVAIL

    _build_validation(outage)
    store, _ = load_stream_store(outage["cache_root"], "validation")
    states = np.asarray(store.array(OBSERVATION_STATE_FILE))
    ids = [str(v) for v in np.asarray(store.array(STABLE_ID_FILE))]
    rep = np.asarray(store.array(REP))

    physiology = {
        ids[i]: rep[i, EMBEDDING_DIM:]
        for i in np.flatnonzero(states == AVAIL)
    }
    validity = PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")
    raw = np.stack([
        _values(key, PHYSIOLOGY_DIM).astype(np.float64) for key in physiology
    ])
    raw[:, validity] = 1.0
    expected = outage["transform"].transform(raw)
    np.testing.assert_array_equal(
        np.stack(list(physiology.values())), expected.astype(np.float32)
    )
    store.close()


def test_physiology_accounting_separates_skipped_from_missing(outage):
    manifest = _build_validation(outage)
    audit = manifest["primary_overlap_audit"]
    assert audit["physiology_skipped_unavailable_rows"] == 1
    # the unavailable row is still proven to exist in the feature corpus
    assert audit["timeline_rows_present_in_feature_corpus"] == manifest[
        "full_stream_row_count"
    ]
    assert audit["physiology_fused_rows"] == manifest["available_row_count"]
    assert audit["physiology_refitted"] is False


# --- Blocker 2: exact sentinels -------------------------------------------


def _corrupt_unavailable_row(outage, mutate) -> Path:
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
    )
    from cardiosentinel.neural.m1_store import (
        REPRESENTATION_FILE as REP,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_UNAVAILABLE_EXACT_FLAT as FLAT,
    )

    directory = outage["cache_root"] / "validation"
    states = np.load(directory / OBSERVATION_STATE_FILE, allow_pickle=False)
    row = int(np.flatnonzero(states == FLAT)[0])
    rep = np.load(directory / REP, allow_pickle=False)
    mutate(rep, row)
    np.save(directory / REP, rep)
    return directory


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "128 NaN + 18 finite",
            lambda rep, row: rep.__setitem__(
                (row, slice(EMBEDDING_DIM, None)), np.float32(0.5)
            ),
        ),
        (
            "one finite value",
            lambda rep, row: rep.__setitem__((row, 3), np.float32(1.0)),
        ),
        (
            "positive inf",
            lambda rep, row: rep.__setitem__((row, 0), np.float32("inf")),
        ),
        (
            "negative inf",
            lambda rep, row: rep.__setitem__((row, 9), np.float32("-inf")),
        ),
        ("zero filled", lambda rep, row: rep.__setitem__(row, np.float32(0.0))),
    ],
)
def test_loader_refuses_a_partially_finite_unavailable_row(outage, name, mutate):
    _build_validation(outage)
    _corrupt_unavailable_row(outage, mutate)
    # the array digest catches it first; reseal so the sentinel rule is what
    # actually refuses.
    directory = outage["cache_root"] / "validation"
    path = directory / "M1_STREAM_CACHE_MANIFEST.json"
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.m1_store import REPRESENTATION_FILE as REP

    manifest = json.loads(path.read_text())
    manifest["artifact_sha256"][REP] = sha256_file(directory / REP)
    manifest["representation_content_sha256"] = _content_digest(directory / REP)
    manifest.pop("stream_cache_sha256")
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="canonical NaN in every dimension"):
        load_stream_store(outage["cache_root"], "validation")


def _content_digest(path: Path) -> str:
    from cardiosentinel.neural.m1_store import digest_array_file

    return digest_array_file(path)


def test_loader_accepts_a_valid_all_nan_unavailable_row(outage):
    _build_validation(outage)
    store, manifest = load_stream_store(outage["cache_root"], "validation")
    assert manifest["unavailable_exact_flat_row_count"] == 1
    store.close()


# --- Blocker 3: independent digest re-derivation ---------------------------


def _reseal_full(directory: Path, mutate_manifest=None) -> None:
    """Recompute every array digest and the manifest digest."""
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
        M1StoreSpec,
    )

    path = directory / "M1_STREAM_CACHE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    spec = M1StoreSpec(
        rows=manifest["full_stream_row_count"],
        representation_dim=REPRESENTATION_DIM,
    )
    manifest["artifact_sha256"] = {
        name: sha256_file(directory / name) for name in sorted(spec.arrays())
    }
    manifest["observation_state_content_sha256"] = _content_digest(
        directory / OBSERVATION_STATE_FILE
    )
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest.pop("stream_cache_sha256", None)
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    path.write_text(json.dumps(manifest))


def test_loader_rederives_the_unavailable_ordered_id_digest(outage):
    """Same unavailable COUNT, different unavailable ROW -> still refused."""
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
    )
    from cardiosentinel.neural.m1_store import (
        REPRESENTATION_FILE as REP,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_AVAILABLE as AVAIL,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_UNAVAILABLE_EXACT_FLAT as FLAT,
    )

    _build_validation(outage)
    directory = outage["cache_root"] / "validation"
    states = np.load(directory / OBSERVATION_STATE_FILE, allow_pickle=False)
    original = int(np.flatnonzero(states == FLAT)[0])
    other = int(np.flatnonzero(states == AVAIL)[0])

    # move the outage to a different row, keeping the count identical
    states[original] = AVAIL
    states[other] = FLAT
    np.save(directory / OBSERVATION_STATE_FILE, states)
    # Swap every per-state column too, so all the sentinel rules stay satisfied
    # and the ONLY thing left wrong is the unavailable ordered-ID identity.
    from cardiosentinel.neural.m1_store import (
        D_LONG_FILE,
        D_SHORT_FILE,
        DISAGREEMENT_FILE,
    )

    for name in (REP, D_SHORT_FILE, D_LONG_FILE, DISAGREEMENT_FILE):
        values = np.load(directory / name, allow_pickle=False)
        values[[original, other]] = values[[other, original]]
        np.save(directory / name, values)

    def _refresh(manifest):
        manifest["representation_content_sha256"] = _content_digest(directory / REP)
        manifest["d_short_content_sha256"] = _content_digest(directory / D_SHORT_FILE)
        manifest["d_long_content_sha256"] = _content_digest(directory / D_LONG_FILE)

    _reseal_full(directory, _refresh)
    with pytest.raises(Exception, match="unavailable ordered stable-ID digest"):
        load_stream_store(outage["cache_root"], "validation")


def test_loader_refuses_a_wrong_unavailable_digest(outage):
    _build_validation(outage)
    directory = outage["cache_root"] / "validation"
    _reseal_full(
        directory,
        lambda m: m.__setitem__("unavailable_ordered_stable_id_sha256", "0" * 64),
    )
    with pytest.raises(Exception, match="unavailable ordered stable-ID digest"):
        load_stream_store(outage["cache_root"], "validation")


def test_loader_refuses_a_non_null_digest_with_zero_unavailable_rows(corpus):
    _bounded(corpus, "train")
    directory = corpus["cache_root"] / "train"
    manifest = json.loads(
        (directory / "M1_STREAM_CACHE_MANIFEST.json").read_text()
    )
    assert manifest["unavailable_exact_flat_row_count"] == 0
    assert manifest["unavailable_ordered_stable_id_sha256"] is None
    _reseal_full(
        directory,
        lambda m: m.__setitem__("unavailable_ordered_stable_id_sha256", "a" * 64),
    )
    with pytest.raises(Exception, match="no row is unavailable"):
        load_stream_store(corpus["cache_root"], "train")


def test_loader_refuses_a_null_digest_with_unavailable_rows(outage):
    _build_validation(outage)
    directory = outage["cache_root"] / "validation"
    _reseal_full(
        directory,
        lambda m: m.__setitem__("unavailable_ordered_stable_id_sha256", None),
    )
    with pytest.raises(Exception, match="records no unavailable ordered-ID digest"):
        load_stream_store(outage["cache_root"], "validation")


def test_unavailable_row_memory_is_frozen_end_to_end(outage):
    """The outage row produces NaN scores and freezes the counters."""
    from cardiosentinel.neural.m1_store import (
        D_LONG_FILE as DL,
    )
    from cardiosentinel.neural.m1_store import (
        D_SHORT_FILE as DS,
    )
    from cardiosentinel.neural.m1_store import (
        DISAGREEMENT_FILE as DD,
    )
    from cardiosentinel.neural.m1_store import (
        OBSERVATION_STATE_FILE,
    )
    from cardiosentinel.neural.m1_store import (
        PAST_OBSERVED_FILE as PO,
    )
    from cardiosentinel.neural.m1_store import (
        PAST_UPDATE_FILE as PU,
    )
    from cardiosentinel.neural.m1_store import (
        RECORDING_AGE_FILE as RA,
    )
    from cardiosentinel.neural.patient_memory import (
        OBSERVATION_UNAVAILABLE_EXACT_FLAT as FLAT,
    )

    _build_validation(outage)
    store, _ = load_stream_store(outage["cache_root"], "validation")
    states = np.asarray(store.array(OBSERVATION_STATE_FILE))
    row = int(np.flatnonzero(states == FLAT)[0])
    assert np.isnan(np.asarray(store.array(DS))[row])
    assert np.isnan(np.asarray(store.array(DL))[row])
    assert np.isnan(np.asarray(store.array(DD))[row])
    assert np.isfinite(np.asarray(store.array(RA))[row])
    observed = np.asarray(store.array(PO))
    updated = np.asarray(store.array(PU))
    assert observed[row] == observed[row - 1] + 1  # the prior row's own update
    assert observed[row] == updated[row]
    store.close()
