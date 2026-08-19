"""Synthetic on-disk fixtures for the T2 canonical harness.

These build a **genuinely valid** M1 stream cache and a **genuinely valid**
LTSTDB-shaped feature corpus in a temporary directory: real manifest
self-digests, real per-array content digests, a real chronology digest, a real
distance standardizer and real per-record target-family arrays. Nothing here is
a stub or a monkeypatch.

That matters for two reasons.

First, the harness's byte-level validator is `m1_experiment.load_stream_store`,
the same route M2 and U1 use. A fixture that only *looked* like a stream cache
would force the T2 loader to accept a weaker check, and the mutation tests --
mutate `representation.npy` under an unchanged manifest, mutate `stable_id.npy`,
mutate `start_sample.npy` -- would prove nothing. Because the fixture is valid,
each mutation is a real refusal by the real validator.

Second, the assembled canonical route is driven against these fixtures through
the same orchestration function the public CLI uses. Test seams that inject past
a component hide defects in that component; there is no such seam here.

Subject and record **identities** are the real frozen TRAIN ones, read from
`protocols/splits/ltstdb_v1.json`. They are names, not data: using them means
`assign_internal_split` produces the frozen 48/8 partition and the frozen split
digest, so the split arithmetic under test is the real one. Every
representation, label and physiological value is synthetic.

The fixture is deliberately tiny. Materialising the frozen 2 208 431-row count
wrote 15 GB of pytest temp once already; the frozen-count gate is enforced only
on the canonical path (`root is None`), and a fixture timeline records
`frozen_row_count_enforced: False` so it can never be mistaken for corpus
evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural import m1_experiment as M1
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    COLD_START_BIN_FILE,
    D_LONG_FILE,
    D_SHORT_FILE,
    DISAGREEMENT_FILE,
    OBSERVATION_STATE_FILE,
    PAST_OBSERVED_FILE,
    PAST_UPDATE_FILE,
    RECORD_ID_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
    M1RowStore,
    M1StoreSpec,
)
from cardiosentinel.neural.patient_memory import (
    STANDARDIZER_NAME,
    STREAM_CACHE_MANIFEST_NAME,
    M1DistanceStandardizer,
)
from cardiosentinel.neural.t2_protocol import (
    T2_INPUT_DIM,
    T2_OBSERVATION_AVAILABLE,
    T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    T2_SPLIT_PATH,
    T2_WINDOW_LENGTH_SAMPLES,
)

WINDOW = T2_WINDOW_LENGTH_SAMPLES
STRIDE = 1250

PRIMARY_POSITIVE = "ischemic_positive"
PRIMARY_NEGATIVE = "background_negative"
CHALLENGE_RATE = "rate_related_confounder"
CHALLENGE_AXIS = "axis_shift_confounder"
CHALLENGE_CONDUCTION = "conduction_change_confounder"
OTHER_BOUNDARY = "boundary_ambiguous"


def frozen_train_subjects() -> tuple[str, ...]:
    """The 56 frozen TRAIN subject identities, from the frozen split manifest."""
    payload = json.loads(Path(T2_SPLIT_PATH).read_text())
    return tuple(str(value) for value in payload["partitions"]["train"]["subjects"])


def record_for_subject(subject_id: str) -> str:
    """`ltstdb:s2001` -> `s20011`, the documented LTSTDB record relationship."""
    return f"{subject_id.split(':', 1)[1]}1"


@dataclass(frozen=True)
class SyntheticStream:
    """One `(record_id, channel_index)` stream and its per-row target families."""

    record_id: str
    channel_index: int
    families: tuple[str, ...]
    unavailable: frozenset[int] = frozenset()
    # Lets a fixture place two spans of the same key at different times, which
    # is how the interleaved-stream refusal is exercised without forging
    # duplicate stable ids.
    start_offset: int = 0

    @property
    def row_count(self) -> int:
        return len(self.families)


def default_streams(
    subjects: tuple[str, ...], *, rows: int = 12
) -> tuple[SyntheticStream, ...]:
    """One stream per subject, with a repeating family pattern.

    The pattern deliberately interleaves challenge and other-non-primary rows
    between PRIMARY rows: that is what makes the causal-context semantics
    testable at all. A run of pure PRIMARY rows would prove nothing about
    whether a challenge `z_t` reaches a later PRIMARY loss.
    """
    pattern = (
        PRIMARY_NEGATIVE,
        PRIMARY_NEGATIVE,
        CHALLENGE_RATE,
        PRIMARY_POSITIVE,
        PRIMARY_NEGATIVE,
        OTHER_BOUNDARY,
        PRIMARY_NEGATIVE,
        CHALLENGE_AXIS,
        PRIMARY_POSITIVE,
        PRIMARY_NEGATIVE,
        CHALLENGE_CONDUCTION,
        PRIMARY_NEGATIVE,
    )
    built: list[SyntheticStream] = []
    for index, subject in enumerate(subjects):
        families = tuple(pattern[(index + step) % len(pattern)] for step in range(rows))
        built.append(
            SyntheticStream(
                record_id=record_for_subject(subject),
                channel_index=0,
                families=families,
            )
        )
    return tuple(built)


@dataclass(frozen=True)
class SyntheticEnvironment:
    """Paths to one complete synthetic TRAIN environment."""

    root: Path
    stream_cache_root: Path
    corpus_manifest: Path
    streams: tuple[SyntheticStream, ...]

    @property
    def partition_dir(self) -> Path:
        return self.stream_cache_root / "train"

    @property
    def row_count(self) -> int:
        return sum(stream.row_count for stream in self.streams)


def build_environment(
    root: Path,
    *,
    partition: str = "train",
    streams: tuple[SyntheticStream, ...] | None = None,
    subjects: tuple[str, ...] | None = None,
    rows: int = 12,
    stream_cache_sha256_override: str | None = None,
) -> SyntheticEnvironment:
    """Write a complete, self-consistent synthetic M1 store and feature corpus."""
    chosen = subjects if subjects is not None else frozen_train_subjects()
    built = streams if streams is not None else default_streams(chosen, rows=rows)
    stream_cache_root = Path(root) / "streams"
    corpus_root = Path(root) / "corpus"
    _write_stream_cache(
        stream_cache_root,
        partition,
        built,
        stream_cache_sha256_override=stream_cache_sha256_override,
    )
    _write_feature_corpus(corpus_root, partition, built)
    return SyntheticEnvironment(
        root=Path(root),
        stream_cache_root=stream_cache_root,
        corpus_manifest=corpus_root / "manifest.json",
        streams=built,
    )


# ---------------------------------------------------------------------------
# The M1 full stream memory cache
# ---------------------------------------------------------------------------


def _stream_rows(streams: tuple[SyntheticStream, ...]) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {
        "stable_id": [],
        "record_id": [],
        "channel_index": [],
        "start_sample": [],
        "cold_start_bin": [],
        "observation_state": [],
        "recording_age_seconds": [],
    }
    for stream in streams:
        for index in range(stream.row_count):
            start = (stream.start_offset + index) * STRIDE
            rows["stable_id"].append(
                f"ltstdb:{stream.record_id}:{stream.channel_index}:"
                f"{start}:{start + WINDOW}"
            )
            rows["record_id"].append(stream.record_id)
            rows["channel_index"].append(stream.channel_index)
            rows["start_sample"].append(start)
            elapsed = index * 5.0
            rows["cold_start_bin"].append(
                "0_5_minutes"
                if elapsed < 300
                else ("5_60_minutes" if elapsed < 3600 else "over_60_minutes")
            )
            rows["recording_age_seconds"].append(elapsed)
            rows["observation_state"].append(
                T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT
                if index in stream.unavailable
                else T2_OBSERVATION_AVAILABLE
            )
    return rows


def _write_stream_cache(
    stream_cache_root: Path,
    partition: str,
    streams: tuple[SyntheticStream, ...],
    *,
    stream_cache_sha256_override: str | None = None,
) -> None:
    directory = stream_cache_root / partition
    directory.mkdir(parents=True, exist_ok=True)
    rows = _stream_rows(streams)
    total = len(rows["stable_id"])
    spec = M1StoreSpec(rows=total, representation_dim=T2_INPUT_DIM)
    store = M1RowStore(directory, spec, create=True)

    generator = np.random.default_rng(2026)
    representation = generator.standard_normal((total, T2_INPUT_DIM)).astype(np.float32)
    states = np.asarray(rows["observation_state"], dtype=np.uint8)
    unavailable = states == T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT
    # The frozen physical-observation contract: an UNAVAILABLE row is canonical
    # NaN in EVERY dimension, and its memory features are NaN too.
    representation[unavailable, :] = np.nan
    store.array(REPRESENTATION_FILE)[:] = representation
    store.array(STABLE_ID_FILE)[:] = np.asarray(rows["stable_id"], dtype="<U64")
    store.array(RECORD_ID_FILE)[:] = np.asarray(rows["record_id"], dtype="<U64")
    store.array(CHANNEL_INDEX_FILE)[:] = np.asarray(
        rows["channel_index"], dtype="int64"
    )
    store.array(START_SAMPLE_FILE)[:] = np.asarray(rows["start_sample"], dtype="int64")
    store.array(COLD_START_BIN_FILE)[:] = np.asarray(
        rows["cold_start_bin"], dtype="<U32"
    )
    store.array(OBSERVATION_STATE_FILE)[:] = states
    store.array(RECORDING_AGE_FILE)[:] = np.asarray(
        rows["recording_age_seconds"], dtype="float64"
    )
    for name in (D_SHORT_FILE, D_LONG_FILE, DISAGREEMENT_FILE):
        values = generator.random(total)
        values[unavailable] = np.nan
        store.array(name)[:] = values
    for name in (PAST_OBSERVED_FILE, PAST_UPDATE_FILE):
        store.array(name)[:] = np.arange(total, dtype="int64")
    store.flush()

    record_ids = sorted({stream.record_id for stream in streams})
    channel_indices = sorted({int(stream.channel_index) for stream in streams})
    manifest = _stream_cache_manifest(
        store,
        partition=partition,
        stream_count=len(streams),
        record_ids=record_ids,
        channel_indices=channel_indices,
        unavailable=unavailable,
        standardizer_sha256=_write_standardizer(stream_cache_root),
    )
    if stream_cache_sha256_override is not None:
        manifest["stream_cache_sha256"] = stream_cache_sha256_override
    write_json_atomic(directory / STREAM_CACHE_MANIFEST_NAME, manifest)
    store.close()


def _write_standardizer(stream_cache_root: Path) -> str:
    stream_cache_root.mkdir(parents=True, exist_ok=True)
    payload = M1DistanceStandardizer(
        means=tuple(0.0 for _ in range(T2_INPUT_DIM)),
        scales=tuple(1.0 for _ in range(T2_INPUT_DIM)),
        prior=tuple(0.0 for _ in range(T2_INPUT_DIM)),
        zero_variance_dimensions=(),
        fitted_rows=1,
        fitted_population="synthetic_fixture",
        input_identities={"fixture": True},
    ).as_dict()
    write_json_atomic(stream_cache_root / STANDARDIZER_NAME, payload)
    return str(payload["standardizer_sha256"])


def _stream_cache_manifest(
    store: M1RowStore,
    *,
    partition: str,
    stream_count: int,
    record_ids: list[str],
    channel_indices: list[int],
    unavailable: np.ndarray,
    standardizer_sha256: str,
) -> dict[str, Any]:
    """Every frozen binding `load_stream_store` checks, with real digests.

    The frozen identity fields are read from `m1_experiment` rather than
    transcribed, so a fixture cannot drift away from the validator it is meant
    to satisfy.
    """
    unavailable_count = int(np.count_nonzero(unavailable))
    unavailable_digest = None
    if unavailable_count:
        identifiers = np.asarray(store.array(STABLE_ID_FILE))[unavailable]
        unavailable_digest = M1.streaming_ordered_stable_id_digest(
            str(value) for value in identifiers
        )
    manifest: dict[str, Any] = {
        "artifact_class": "m1_full_stream_memory_cache",
        "m1_stream_cache_schema": M1.M1_STREAM_CACHE_SCHEMA,
        "storage": "row_aligned_memmapped_npy_directory",
        "partition": partition,
        "m1_protocol_sha256": M1.M1_PROTOCOL_SHA256,
        "p1_protocol_sha256": M1.P1_PROTOCOL_SHA256,
        "p1_retention_decision_sha256": M1.P1_RETENTION_DECISION_SHA256,
        "p1_stage1_suite_sha256": M1.FROZEN_P1_STAGE1_SUITE_SHA256,
        "p1b_experiment_lock_sha256": M1.FROZEN_P1B_LOCK_SHA256,
        "encoder_checkpoint_sha256": M1.B4B_CHECKPOINT_SHA256,
        "encoder_experiment_lock_sha256": M1.B4B_EXPERIMENT_LOCK_SHA256,
        "embedding_tap": M1.EMBEDDING_TAP,
        "physiology_transform_sha256": M1.FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
        "physiology_schema_sha256": M1.MORPHOLOGY_SCHEMA_SHA256,
        "p1_embedding_cache_sha256": M1.FROZEN_P1_EMBEDDING_CACHE_SHA256[partition],
        "development_feature_integrity_sha256": (
            M1.FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256
        ),
        "development_source_integrity_sha256": (
            M1.FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256
        ),
        "b4_protocol_sha256": M1.B4_PROTOCOL_SHA256,
        "split_sha256": M1.B4_SPLIT_SHA256,
        "feature_corpus_sha256": M1.FEATURE_CORPUS_SHA256,
        "distance_standardizer_sha256": standardizer_sha256,
        "representation_dim": M1.REPRESENTATION_DIM,
        "full_stream_row_count": int(store.spec.rows),
        "stream_count": int(stream_count),
        "record_ids": list(record_ids),
        "channel_indices": list(channel_indices),
        "ordered_stable_id_sha256": store.stable_id_digest(),
        "ordered_chronology_sha256": store.chronology_digest(),
        "representation_content_sha256": store.content_digest(REPRESENTATION_FILE),
        "d_short_content_sha256": store.content_digest(D_SHORT_FILE),
        "d_long_content_sha256": store.content_digest(D_LONG_FILE),
        "history_count_sha256": store.paired_content_digest(
            PAST_OBSERVED_FILE, PAST_UPDATE_FILE
        ),
        "primary_rows_reused": 0,
        "rows_newly_extracted": int(store.spec.rows),
        "physical_observation_contract": M1.PHYSICAL_OBSERVATION_CONTRACT,
        "observation_state_enum": dict(M1.OBSERVATION_STATE_ENUM),
        "observation_state_version": M1.OBSERVATION_STATE_VERSION,
        "available_row_count": int(store.spec.rows) - unavailable_count,
        "unavailable_exact_flat_row_count": unavailable_count,
        "observation_state_content_sha256": store.content_digest(
            OBSERVATION_STATE_FILE
        ),
        "unavailable_ordered_stable_id_sha256": unavailable_digest,
        "m1_protocol_v1_sha256": M1.M1_PROTOCOL_V1_SHA256,
        "m1_physical_observation_decision_sha256": (
            M1.M1_PHYSICAL_OBSERVATION_DECISION_SHA256
        ),
        "m1_attempt2_census_sha256": M1.M1_ATTEMPT2_CENSUS_SHA256,
        "m1_attempt2_failure_sha256": M1.M1_ATTEMPT2_FAILURE_SHA256,
        "primary_overlap_audit": {
            "primary_rows_reused": 0,
            "rows_newly_extracted": int(store.spec.rows),
        },
        "label_independent_history": True,
        "update_policy": M1.UPDATE_POLICY,
        "contamination_safe": M1.CONTAMINATION_SAFE,
        "alpha_short": M1.ALPHA_SHORT,
        "alpha_long": M1.ALPHA_LONG,
        "memory_features": ["d_short", "d_long"],
        "git_sha": "0" * 40,
        "git_dirty": False,
        "environment_dependency_digest": M1.FROZEN_DEPENDENCY_DIGEST,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "artifact_sha256": store.artifact_digests(),
    }
    manifest["stream_cache_sha256"] = canonical_sha256(manifest)
    return manifest


# ---------------------------------------------------------------------------
# The LTSTDB-shaped feature corpus, i.e. the frozen target authority
# ---------------------------------------------------------------------------


def _write_feature_corpus(
    corpus_root: Path, partition: str, streams: tuple[SyntheticStream, ...]
) -> None:
    partition_dir = corpus_root / partition
    partition_dir.mkdir(parents=True, exist_ok=True)
    by_record: dict[str, list[SyntheticStream]] = {}
    for stream in streams:
        by_record.setdefault(stream.record_id, []).append(stream)

    records: list[dict[str, Any]] = []
    for record_id, record_streams in sorted(by_record.items()):
        cache_path = partition_dir / f"{record_id}.npz"
        row_count = _write_record_cache(cache_path, partition, record_streams)
        counts: dict[str, int] = {}
        for stream in record_streams:
            for family in stream.families:
                counts[family] = counts.get(family, 0) + 1
        records.append(
            {
                "record_id": record_id,
                "subject_id": f"ltstdb:{record_id[:-1]}",
                "partition": partition,
                "status": "complete",
                "cache_path": f"{partition}/{record_id}.npz",
                "cache_sha256": sha256_file(cache_path),
                "row_count": row_count,
                "target_counts": dict(sorted(counts.items())),
            }
        )
    write_json_atomic(
        corpus_root / "manifest.json",
        {
            "dataset": "ltstdb",
            "dataset_version": "1.0.0",
            "feature_corpus_sha256": M1.FEATURE_CORPUS_SHA256,
            "split_sha256": M1.B4_SPLIT_SHA256,
            "expected_split_sha256": M1.B4_SPLIT_SHA256,
            "window_seconds": 10.0,
            "stride_seconds": 5.0,
            "records": records,
        },
    )


def _write_record_cache(
    path: Path, partition: str, streams: list[SyntheticStream]
) -> int:
    stable_ids: list[str] = []
    record_ids: list[str] = []
    channels: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    partitions: list[str] = []
    families: list[str] = []
    for stream in streams:
        for index, family in enumerate(stream.families):
            start = (stream.start_offset + index) * STRIDE
            stable_ids.append(
                f"ltstdb:{stream.record_id}:{stream.channel_index}:"
                f"{start}:{start + WINDOW}"
            )
            record_ids.append(stream.record_id)
            channels.append(int(stream.channel_index))
            starts.append(start)
            ends.append(start + WINDOW)
            partitions.append(partition)
            families.append(family)
    np.savez_compressed(
        path,
        stable_ids=np.asarray(stable_ids, dtype=np.str_),
        record_ids=np.asarray(record_ids, dtype=np.str_),
        channel_indices=np.asarray(channels, dtype=np.int64),
        window_start_samples=np.asarray(starts, dtype=np.int64),
        window_end_samples=np.asarray(ends, dtype=np.int64),
        partitions=np.asarray(partitions, dtype=np.str_),
        target_families=np.asarray(families, dtype=np.str_),
    )
    return len(stable_ids)


def rewrite_record_cache(
    environment: SyntheticEnvironment,
    record_id: str,
    *,
    mutate: Any,
    partition: str = "train",
) -> None:
    """Rewrite one record cache through `mutate`, then re-bind its digest.

    The manifest digest is recomputed on purpose: a corrupted-target test that
    also broke the cache digest would be refused by the digest check and would
    never exercise the join at all.
    """
    manifest_path = environment.corpus_manifest
    manifest = json.loads(manifest_path.read_text())
    entry = next(
        item
        for item in manifest["records"]
        if item["record_id"] == record_id and item["partition"] == partition
    )
    cache_path = manifest_path.parent / entry["cache_path"]
    with np.load(cache_path, allow_pickle=False) as cached:
        columns = {name: np.asarray(cached[name]) for name in cached.files}
    mutate(columns)
    np.savez_compressed(cache_path, **columns)
    entry["cache_sha256"] = sha256_file(cache_path)
    entry["row_count"] = int(columns["stable_ids"].shape[0])
    write_json_atomic(manifest_path, manifest)


def mutate_array_file(path: Path, *, index: int = 0, delta: Any = 1) -> None:
    """Mutate one persisted `.npy` in place, leaving the manifest untouched.

    This is the whole point of the fixture being real: the manifest still
    declares the pre-mutation digests, so the loader must catch the difference
    from the bytes alone.
    """
    array = np.load(path, allow_pickle=False)
    flat = array.reshape(-1)
    if array.dtype.kind in {"U", "S"}:
        flat[index] = f"{flat[index]}-mutated"
    else:
        flat[index] = flat[index] + delta
    np.save(path, flat.reshape(array.shape))
