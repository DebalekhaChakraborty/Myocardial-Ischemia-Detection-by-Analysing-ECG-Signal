"""Lineage-complete, bounded-memory T2 full-timeline loader and role join.

The frozen M1 full stream memory cache is the **only** admissible `z_t` source.
The P1 embedding cache is refused by name and by digest: its TRAIN side is a 3:1
negatively-sampled *selection*, not a timeline, and training on it would destroy
temporal continuity on TRAIN while leaving VALIDATION intact.

Everything here is read-only and memory-mapped. The 2 208 431 x 146 TRAIN
representation is never copied into a second array, never converted into one
torch tensor, and never materialised beyond the active chunk. `M1RowStore` is
reused rather than reimplemented, so the bounded-read behaviour M1 already
proved is the behaviour T2 gets.

**The store is proven from its actual bytes, not from what its manifest says.**
Opening a timeline goes through `m1_experiment.load_stream_store`, the strongest
M1 validator in the repository and the one M2 and U1 already use: it re-derives
the manifest self-digest, every persisted array digest, the representation,
d_short and d_long content digests, the paired history digest, the ordered
stable-id digest and the ordered chronology digest from the persisted arrays,
and re-derives stream membership, the observation-state census and the
availability/finiteness contract. A `representation.npy`, `stable_id.npy` or
`start_sample.npy` mutated under an unchanged manifest is therefore refused.
T2 adds its own frozen-identity gates on top; it does not build a second,
weaker verifier.

The target family is a **masking and evaluation authority only**. It selects
which rows carry direct loss and which metrics they feed. It never enters the
trainable `z_t` vector -- there is no code path from a role to a model input.
`resolve_timeline_target_families` is the one canonical provider: it joins the
timeline to the frozen LTSTDB baseline feature corpus record by record, so no
raw annotation is re-read, no `.stb` is reinterpreted, no label is fabricated
from a context flag, and no 2.2-million-entry Python object graph is built.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np

from cardiosentinel.data.ltstdb import subject_id_for_record
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    COLD_START_BIN_FILE,
    DEFAULT_CHUNK_ROWS,
    OBSERVATION_STATE_FILE,
    RECORD_ID_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
    streaming_ordered_stable_id_digest,
)
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.t2_protocol import (
    ROLE_CHALLENGE_CONTEXT,
    ROLE_OTHER_NONPRIMARY_CONTEXT,
    ROLE_PRIMARY_DIRECT_LOSS,
    ROLE_UNAVAILABLE_NO_STATE_UPDATE,
    T2_CHALLENGE_CATEGORIES,
    T2_FEATURE_CORPUS_SHA256,
    T2_INPUT_DIM,
    T2_OBSERVATION_AVAILABLE,
    T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    T2_OTHER_NON_PRIMARY_CATEGORIES,
    T2_PRIMARY_CATEGORIES,
    T2_TRAIN_BACKGROUND_NEGATIVE,
    T2_TRAIN_CHALLENGE_ROW_COUNT,
    T2_TRAIN_FULL_STREAM_ROW_COUNT,
    T2_TRAIN_ISCHEMIC_POSITIVE,
    T2_TRAIN_OTHER_NON_PRIMARY_ROW_COUNT,
    T2_TRAIN_P1_EMBEDDING_CACHE_SHA256,
    T2_TRAIN_PRIMARY_ROW_COUNT,
    T2_TRAIN_REPRESENTATION_CONTENT_SHA256,
    T2_TRAIN_STREAM_CACHE_SHA256,
    T2_VALIDATION_BACKGROUND_NEGATIVE,
    T2_VALIDATION_CHALLENGE_ROW_COUNT,
    T2_VALIDATION_FULL_STREAM_ROW_COUNT,
    T2_VALIDATION_ISCHEMIC_POSITIVE,
    T2_VALIDATION_OTHER_NON_PRIMARY_ROW_COUNT,
    T2_VALIDATION_P1_EMBEDDING_CACHE_SHA256,
    T2_VALIDATION_PRIMARY_ROW_COUNT,
    T2_VALIDATION_REPRESENTATION_CONTENT_SHA256,
    T2_VALIDATION_STREAM_CACHE_SHA256,
    T2_WINDOW_LENGTH_SAMPLES,
)

STREAM_CACHE_ROOT: Final = (
    REPOSITORY_ROOT / "cardiosentinel-features" / ("m1-stream-memory-v2")
)
STREAM_CACHE_MANIFEST: Final = "M1_STREAM_CACHE_MANIFEST.json"
CORPUS_MANIFEST: Final = (
    REPOSITORY_ROOT / "cardiosentinel-features" / "ltstdb-baseline-v1" / "manifest.json"
)

# The one artifact that must never become the timeline source.
FORBIDDEN_TIMELINE_SOURCES: Final = (
    "p1-b4b-embeddings-v1",
    "p1_embeddings.npz",
)
FORBIDDEN_TIMELINE_DIGESTS: Final = (
    T2_TRAIN_P1_EMBEDDING_CACHE_SHA256,
    T2_VALIDATION_P1_EMBEDDING_CACHE_SHA256,
)

SEALED_PARTITION: Final = "test"
PERMITTED_PARTITIONS: Final = ("train", "validation")

EXPECTED_ROWS: Final = {
    "train": T2_TRAIN_FULL_STREAM_ROW_COUNT,
    "validation": T2_VALIDATION_FULL_STREAM_ROW_COUNT,
}
EXPECTED_STREAM_CACHE_SHA256: Final = {
    "train": T2_TRAIN_STREAM_CACHE_SHA256,
    "validation": T2_VALIDATION_STREAM_CACHE_SHA256,
}
EXPECTED_REPRESENTATION_SHA256: Final = {
    "train": T2_TRAIN_REPRESENTATION_CONTENT_SHA256,
    "validation": T2_VALIDATION_REPRESENTATION_CONTENT_SHA256,
}

_ROLE_BY_CATEGORY: Final = {
    **{name: ROLE_PRIMARY_DIRECT_LOSS for name in T2_PRIMARY_CATEGORIES},
    **{name: ROLE_CHALLENGE_CONTEXT for name in T2_CHALLENGE_CATEGORIES},
    **{name: ROLE_OTHER_NONPRIMARY_CONTEXT for name in T2_OTHER_NON_PRIMARY_CATEGORIES},
}

# ---------------------------------------------------------------------------
# Coded families and roles
#
# A `<U40` role string per row would be 88 MB for TRAIN, and a `<U32` family
# string 283 MB. Both are held for the whole timeline, so both are coded into
# `uint8` instead: 2.2 MB each, with the code tables frozen here and a parity
# test proving the coded roles agree with `assign_row_roles` exactly.
# ---------------------------------------------------------------------------

T2_TARGET_FAMILY_ORDER: Final = (
    *T2_PRIMARY_CATEGORIES,
    *T2_CHALLENGE_CATEGORIES,
    *T2_OTHER_NON_PRIMARY_CATEGORIES,
)
FAMILY_CODE: Final = {name: index for index, name in enumerate(T2_TARGET_FAMILY_ORDER)}
FAMILY_NAME: Final = tuple(T2_TARGET_FAMILY_ORDER)
FAMILY_CODE_UNRESOLVED: Final = 255

ROLE_CODE_PRIMARY: Final = 0
ROLE_CODE_CHALLENGE: Final = 1
ROLE_CODE_OTHER: Final = 2
ROLE_CODE_UNAVAILABLE: Final = 3
ROLE_NAME_BY_CODE: Final = (
    ROLE_PRIMARY_DIRECT_LOSS,
    ROLE_CHALLENGE_CONTEXT,
    ROLE_OTHER_NONPRIMARY_CONTEXT,
    ROLE_UNAVAILABLE_NO_STATE_UPDATE,
)

_ROLE_CODE_BY_FAMILY_CODE: Final = np.asarray(
    [
        {
            ROLE_PRIMARY_DIRECT_LOSS: ROLE_CODE_PRIMARY,
            ROLE_CHALLENGE_CONTEXT: ROLE_CODE_CHALLENGE,
            ROLE_OTHER_NONPRIMARY_CONTEXT: ROLE_CODE_OTHER,
        }[_ROLE_BY_CATEGORY[name]]
        for name in T2_TARGET_FAMILY_ORDER
    ],
    dtype=np.uint8,
)

# Only `ischemic_positive` is a positive label. Everything else that carries a
# direct loss is `background_negative`; nothing else carries one at all.
POSITIVE_FAMILY_CODE: Final = FAMILY_CODE["ischemic_positive"]
NEGATIVE_FAMILY_CODE: Final = FAMILY_CODE["background_negative"]

EXPECTED_FAMILY_CENSUS: Final = {
    "train": {
        "row_count": T2_TRAIN_FULL_STREAM_ROW_COUNT,
        "primary_row_count": T2_TRAIN_PRIMARY_ROW_COUNT,
        "ischemic_positive": T2_TRAIN_ISCHEMIC_POSITIVE,
        "background_negative": T2_TRAIN_BACKGROUND_NEGATIVE,
        "challenge_row_count": T2_TRAIN_CHALLENGE_ROW_COUNT,
        "other_non_primary_row_count": T2_TRAIN_OTHER_NON_PRIMARY_ROW_COUNT,
    },
    "validation": {
        "row_count": T2_VALIDATION_FULL_STREAM_ROW_COUNT,
        "primary_row_count": T2_VALIDATION_PRIMARY_ROW_COUNT,
        "ischemic_positive": T2_VALIDATION_ISCHEMIC_POSITIVE,
        "background_negative": T2_VALIDATION_BACKGROUND_NEGATIVE,
        "challenge_row_count": T2_VALIDATION_CHALLENGE_ROW_COUNT,
        "other_non_primary_row_count": T2_VALIDATION_OTHER_NON_PRIMARY_ROW_COUNT,
    },
}

# The metadata members the canonical target join reads. `features` is
# deliberately absent: the 40-dimensional baseline feature matrix is not a T2
# input and is never loaded, so an `.npz` member read stays small.
TARGET_JOIN_MEMBERS: Final = (
    "stable_ids",
    "record_ids",
    "channel_indices",
    "window_start_samples",
    "window_end_samples",
    "partitions",
    "target_families",
)


class T2TimelineError(RuntimeError):
    """Raised when the T2 timeline cannot be proven lineage-complete."""


def refuse_sealed_partition(partition: str) -> str:
    """Refuse TEST before any path is resolved."""
    name = str(partition)
    if name == SEALED_PARTITION:
        raise T2TimelineError(
            "The B4 sealed TEST partition is refused by name. T2 has no TEST "
            "route, no TEST option and no TEST artifact."
        )
    if name not in PERMITTED_PARTITIONS:
        raise T2TimelineError(
            f"{name!r} is not a T2 partition; {list(PERMITTED_PARTITIONS)} are."
        )
    return name


def refuse_forbidden_source(path: Path | str) -> None:
    """Refuse the 3:1 P1 embedding selection as a timeline source."""
    text = str(path)
    for marker in FORBIDDEN_TIMELINE_SOURCES:
        if marker in text:
            raise T2TimelineError(
                f"{text!r} is the P1 embedding cache. Its TRAIN side is a 3:1 "
                "negatively-sampled selection, not a timeline, and it carries "
                "neither the physiology block nor any ordering key. The M1 full "
                "stream memory cache is the only admissible T2 z_t source."
            )


def require_frozen_row_count(partition: str, rows: int) -> int:
    """The canonical timeline is exactly as long as the frozen corpus says.

    A full replay is thinned by nothing at all, so a short timeline is refused
    rather than trimmed to fit.
    """
    expected = EXPECTED_ROWS[refuse_sealed_partition(partition)]
    if int(rows) != expected:
        raise T2TimelineError(
            f"The {partition} timeline carries {rows} rows against the frozen "
            f"{expected}. A full replay is thinned by nothing at all."
        )
    return int(rows)


def require_frozen_stream_identity(
    partition: str, manifest: dict[str, Any]
) -> dict[str, str]:
    """A canonical timeline is the PROMOTED store, not merely a valid one.

    A store can be perfectly self-consistent -- every array digest matching,
    every ordering digest re-derivable -- and still be a different store. This
    is where the canonical route refuses that: the stream-cache identity and the
    representation content identity must be the exact frozen ones.
    """
    name = refuse_sealed_partition(partition)
    expected_cache = EXPECTED_STREAM_CACHE_SHA256[name]
    observed_cache = manifest.get("stream_cache_sha256")
    if observed_cache != expected_cache:
        raise T2TimelineError(
            f"The {name} stream cache digests to {observed_cache!r}, not the "
            f"frozen {expected_cache!r}. A self-consistent store that is not "
            "the promoted store is refused."
        )
    expected_representation = EXPECTED_REPRESENTATION_SHA256[name]
    observed_representation = manifest.get("representation_content_sha256")
    if observed_representation != expected_representation:
        raise T2TimelineError(
            f"The {name} representation content digests to "
            f"{observed_representation!r}, not the frozen "
            f"{expected_representation!r}."
        )
    return {
        "stream_cache_sha256": str(observed_cache),
        "representation_content_sha256": str(observed_representation),
    }


def require_not_forbidden_digest(digest: str) -> str:
    if digest in FORBIDDEN_TIMELINE_DIGESTS:
        raise T2TimelineError(
            f"Digest {digest} identifies the P1 embedding cache, which may not "
            "be the T2 timeline source."
        )
    return digest


@dataclass(frozen=True, slots=True)
class T2Stream:
    """One `(record_id, channel_index)` stream as a contiguous row span."""

    record_id: str
    channel_index: int
    subject_id: str
    start_index: int
    stop_index: int

    @property
    def row_count(self) -> int:
        return self.stop_index - self.start_index

    @property
    def key(self) -> tuple[str, int]:
        return (self.record_id, int(self.channel_index))


class T2Timeline:
    """A read-only, memory-mapped view of one partition's full timeline.

    Nothing here loads the corpus. The representation stays an `mmap_mode="r"`
    array and callers take bounded slices; `rows()` never returns more than one
    chunk's worth of float data.

    Opening one is not cheap and is not meant to be: `m1_experiment
    .load_stream_store` re-derives every persisted identity from the bytes on
    disk before this object exists. That is the point -- a mutated
    `representation.npy`, `stable_id.npy` or `start_sample.npy` under an
    untouched manifest never becomes a T2 timeline.
    """

    def __init__(self, partition: str, *, root: Path | None = None) -> None:
        # Imported here rather than at module scope: `m1_experiment` is a heavy
        # module and importing it eagerly would drag the whole M1 execution
        # surface into every T2 protocol import.
        from cardiosentinel.neural.m1_experiment import load_stream_store

        self.partition = refuse_sealed_partition(partition)
        # The canonical route passes no root, and only that path enforces the
        # frozen row count. A synthetic fixture supplies its own root and is
        # therefore never mistaken for the corpus: the identity it reports says
        # so explicitly.
        self.canonical_source = root is None
        directory = Path(root) if root is not None else STREAM_CACHE_ROOT
        refuse_forbidden_source(directory)
        self.cache_root = directory
        self.directory = directory / self.partition
        manifest_path = self.directory / STREAM_CACHE_MANIFEST
        if not manifest_path.is_file():
            raise T2TimelineError(f"No M1 stream cache manifest at {manifest_path}.")

        # THE byte-level proof. Everything below it validates T2's own frozen
        # bindings on a store whose persisted arrays have already been
        # re-digested, re-ordered and re-censused from disk.
        store, manifest = load_stream_store(directory, self.partition)
        self.store = store
        self.manifest: dict[str, Any] = dict(manifest)
        self.byte_level_validation = "m1_experiment.load_stream_store"
        self._validate_manifest()
        self.row_count = int(self.manifest["full_stream_row_count"])
        self._streams: tuple[T2Stream, ...] | None = None

    # -- validation --------------------------------------------------------

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("partition") != self.partition:
            raise T2TimelineError(
                f"The stream cache records partition "
                f"{manifest.get('partition')!r}, not {self.partition!r}."
            )
        if manifest.get("artifact_class") != "m1_full_stream_memory_cache":
            raise T2TimelineError(
                "T2 consumes the M1 full stream memory cache, not "
                f"{manifest.get('artifact_class')!r}."
            )
        if manifest.get("representation_dim") != T2_INPUT_DIM:
            raise T2TimelineError(
                f"The stream cache carries representation_dim "
                f"{manifest.get('representation_dim')}, not the frozen "
                f"{T2_INPUT_DIM}."
            )
        # The frozen-identity gates apply to the CANONICAL path only, exactly
        # like the frozen row count. `load_stream_store` has already proved
        # this store self-consistent from its bytes; what these add is that the
        # self-consistent store is the *promoted* one. A synthetic fixture is
        # necessarily a different, equally self-consistent store, and says so
        # in its identity via `canonical_source: false`.
        if self.canonical_source:
            require_frozen_row_count(
                self.partition, int(manifest.get("full_stream_row_count", -1))
            )
            require_frozen_stream_identity(self.partition, manifest)
        require_not_forbidden_digest(str(manifest.get("stream_cache_sha256")))
        if manifest.get("test_accessed") is not False:
            raise T2TimelineError("The stream cache records TEST access.")
        if manifest.get("sealed_test_state") != "unopened":
            raise T2TimelineError("The stream cache does not record TEST unopened.")

    # -- streams -----------------------------------------------------------

    def streams(self) -> tuple[T2Stream, ...]:
        """Contiguous `(record_id, channel_index)` spans, chronologically ordered.

        The cache is already grouped and ordered; this proves it rather than
        assuming it, and refuses a non-chronological or re-interleaved stream.
        """
        if self._streams is not None:
            return self._streams
        records = np.asarray(self.store.array(RECORD_ID_FILE))
        channels = np.asarray(self.store.array(CHANNEL_INDEX_FILE))
        starts = np.asarray(self.store.array(START_SAMPLE_FILE))

        boundaries = [0]
        changed = (records[1:] != records[:-1]) | (channels[1:] != channels[:-1])
        boundaries.extend((np.nonzero(changed)[0] + 1).tolist())
        boundaries.append(len(records))

        seen: set[tuple[str, int]] = set()
        built: list[T2Stream] = []
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            record = str(records[begin])
            channel = int(channels[begin])
            key = (record, channel)
            if key in seen:
                raise T2TimelineError(
                    f"Stream {key} appears in more than one span; the timeline is "
                    "interleaved and no longer one causal stream per key."
                )
            seen.add(key)
            span = starts[begin:end]
            if span.size > 1 and not bool(np.all(span[1:] > span[:-1])):
                raise T2TimelineError(
                    f"Stream {key} is not strictly ordered by start_sample."
                )
            built.append(
                T2Stream(
                    record_id=record,
                    channel_index=channel,
                    subject_id=subject_id_for_record(record),
                    start_index=int(begin),
                    stop_index=int(end),
                )
            )
        self._streams = tuple(built)
        return self._streams

    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({stream.subject_id for stream in self.streams()}))

    def streams_for_subjects(
        self, subjects: Iterator[str] | set[str]
    ) -> tuple[T2Stream, ...]:
        wanted = set(subjects)
        return tuple(stream for stream in self.streams() if stream.subject_id in wanted)

    # -- bounded reads -----------------------------------------------------

    def representation(self, start: int, stop: int) -> np.ndarray:
        """A bounded read-only slice. Never the whole corpus."""
        array = self.store.array(REPRESENTATION_FILE)
        return np.asarray(array[start:stop])

    def column(self, name: str, start: int, stop: int) -> np.ndarray:
        return np.asarray(self.store.array(name)[start:stop])

    def observation_state(self, start: int, stop: int) -> np.ndarray:
        return self.column(OBSERVATION_STATE_FILE, start, stop)

    def stable_ids(self, start: int, stop: int) -> np.ndarray:
        return self.column(STABLE_ID_FILE, start, stop)

    def cold_start_bins(self, start: int, stop: int) -> np.ndarray:
        return self.column(COLD_START_BIN_FILE, start, stop)

    def identity(self) -> dict[str, Any]:
        return {
            "identity_class": "t2_timeline_identity",
            "partition": self.partition,
            "artifact_class": self.manifest["artifact_class"],
            "row_count": self.row_count,
            "stream_count": len(self.streams()),
            "subject_count": len(self.subjects()),
            "representation_dim": T2_INPUT_DIM,
            "stream_cache_sha256": self.manifest["stream_cache_sha256"],
            "representation_content_sha256": (
                self.manifest["representation_content_sha256"]
            ),
            "ordered_stable_id_sha256": self.manifest["ordered_stable_id_sha256"],
            "ordered_chronology_sha256": self.manifest["ordered_chronology_sha256"],
            "artifact_sha256": dict(self.manifest["artifact_sha256"]),
            "byte_level_validation_route": self.byte_level_validation,
            "persisted_bytes_revalidated": True,
            "canonical_source": self.canonical_source,
            "frozen_row_count_enforced": self.canonical_source,
            "p1_embedding_cache_used_as_timeline": False,
            "negative_sampling_applied": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> T2Timeline:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Target / role join
# ---------------------------------------------------------------------------


def load_corpus_target_families(
    partition: str, *, manifest_path: Path | None = None
) -> dict[str, str]:
    """Per-record target-family counts from the frozen corpus authority.

    Returned as `{record_id: partition}` for membership proof only; the per-row
    family join is performed by `assign_row_roles` against a caller-supplied
    per-row family array, so this module never fabricates a label.
    """
    name = refuse_sealed_partition(partition)
    path = Path(manifest_path) if manifest_path is not None else CORPUS_MANIFEST
    if not path.is_file():
        raise T2TimelineError(f"No frozen corpus manifest at {path}.")
    manifest = json.loads(path.read_text())
    return {
        str(record["record_id"]): str(record["partition"])
        for record in manifest["records"]
        if record["partition"] == name
    }


def role_for_category(category: str) -> str:
    """Map one frozen corpus target family to its T2 row role."""
    role = _ROLE_BY_CATEGORY.get(str(category))
    if role is None:
        raise T2TimelineError(
            f"Unknown target family {category!r}; the frozen corpus families are "
            f"{sorted(_ROLE_BY_CATEGORY)}."
        )
    return role


# ---------------------------------------------------------------------------
# The canonical frozen target-family provider
# ---------------------------------------------------------------------------


def _read_corpus_record_metadata(path: Path) -> dict[str, np.ndarray]:
    """Read one record cache's metadata members. Never its feature matrix.

    `np.load` on an `.npz` is lazy, so naming the members keeps the read to the
    identity and target-family columns: the 40-dimensional baseline feature
    matrix is not a T2 input and is never decompressed.
    """
    with np.load(path, allow_pickle=False) as cached:
        missing = [name for name in TARGET_JOIN_MEMBERS if name not in cached.files]
        if missing:
            raise T2TimelineError(
                f"The frozen record cache {path.name} is missing {missing}; it "
                "cannot be the persisted target-family authority."
            )
        return {name: np.asarray(cached[name]) for name in TARGET_JOIN_MEMBERS}


def _corpus_entries(partition: str, manifest: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for record in manifest["records"]:
        if record["partition"] != partition or record.get("status") != "complete":
            continue
        record_id = str(record["record_id"])
        if record_id in entries:
            raise T2TimelineError(
                f"The frozen corpus manifest lists {record_id} twice in "
                f"{partition}; the target authority is ambiguous."
            )
        entries[record_id] = record
    if not entries:
        raise T2TimelineError(
            f"The frozen corpus manifest lists no completed {partition} record."
        )
    return entries


def resolve_timeline_target_families(
    timeline: T2Timeline,
    *,
    manifest_path: Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve every timeline row to exactly one persisted frozen target family.

    This is the one canonical provider. It reads **only** persisted corpus
    metadata: no raw ECG annotation is re-read, no `.stb` is reinterpreted and
    no label is inferred from a context flag. The authority is the frozen
    `cardiosentinel-features/ltstdb-baseline-v1` corpus, whose per-record cache
    digest is re-verified from disk before its arrays are believed.

    The join is record-wise and therefore bounded: one record's identity and
    target columns are held at a time (the widest LTSTDB record is ~33 000
    rows), never a 2.2-million-entry mapping over the whole corpus. The result
    is a `uint8` code array, not 2.2 million Python strings.

    Refused, in order of how quietly each would otherwise pass:

    * a timeline row whose stable id is absent from its record's cache;
    * a corpus row consumed twice, or a duplicate stable id inside one cache;
    * a corpus row never consumed -- an extra row silently ignored;
    * a corpus row whose record, channel, window start or window end disagrees
      with the timeline row it was matched to;
    * a corpus row from another partition;
    * a target family outside the frozen set.

    Returns `(family_codes, lineage)`.
    """
    partition = refuse_sealed_partition(timeline.partition)
    path = Path(manifest_path) if manifest_path is not None else CORPUS_MANIFEST
    if not path.is_file():
        raise T2TimelineError(f"No frozen corpus manifest at {path}.")
    refuse_forbidden_source(path)
    manifest = json.loads(path.read_text())
    corpus_sha256 = str(manifest.get("feature_corpus_sha256"))
    if timeline.canonical_source and corpus_sha256 != T2_FEATURE_CORPUS_SHA256:
        raise T2TimelineError(
            f"The target authority digests to {corpus_sha256!r}, not the frozen "
            f"{T2_FEATURE_CORPUS_SHA256!r}. A canonical T2 run binds one corpus."
        )
    if timeline.manifest.get("feature_corpus_sha256") != corpus_sha256:
        raise T2TimelineError(
            "The M1 stream cache was built against feature corpus "
            f"{timeline.manifest.get('feature_corpus_sha256')!r}, but the "
            f"target authority offered is {corpus_sha256!r}. The timeline and "
            "its labels must come from one corpus."
        )
    entries = _corpus_entries(partition, manifest)

    root = path.parent
    codes = np.full(timeline.row_count, FAMILY_CODE_UNRESOLVED, dtype=np.uint8)
    record_digests: dict[str, str] = {}
    consumed_rows = 0

    for record_id, streams in _streams_by_record(timeline).items():
        entry = entries.get(record_id)
        if entry is None:
            raise T2TimelineError(
                f"Timeline record {record_id!r} has no completed {partition} "
                "entry in the frozen corpus manifest; its rows have no "
                "persisted target family and none is invented."
            )
        cache_path = (root / str(entry["cache_path"])).resolve()
        try:
            cache_path.relative_to(root.resolve())
        except ValueError as error:
            raise T2TimelineError(
                f"The corpus cache path for {record_id} escapes the corpus root."
            ) from error
        digest = sha256_file(cache_path)
        if digest != str(entry["cache_sha256"]):
            raise T2TimelineError(
                f"The frozen corpus cache for {record_id} digests to {digest}, "
                f"not the manifest's {entry['cache_sha256']}."
            )
        record_digests[record_id] = digest

        columns = _read_corpus_record_metadata(cache_path)
        rows_here = int(columns["stable_ids"].shape[0])
        if rows_here != int(entry["row_count"]):
            raise T2TimelineError(
                f"The {record_id} cache carries {rows_here} rows against the "
                f"manifest's {entry['row_count']}."
            )
        offending = sorted(set(columns["partitions"].tolist()) - {partition})
        if offending:
            raise T2TimelineError(
                f"The {record_id} cache carries rows from partition {offending}; "
                f"a {partition} join accepts {partition} rows only."
            )
        lookup: dict[str, int] = {}
        for position, value in enumerate(columns["stable_ids"].tolist()):
            key = str(value)
            if key in lookup:
                raise T2TimelineError(
                    f"The {record_id} cache repeats stable id {key!r}; a row "
                    "cannot resolve to two persisted target families."
                )
            lookup[key] = position
        claimed = np.zeros(rows_here, dtype=bool)

        for stream in streams:
            consumed_rows += _resolve_stream_families(
                timeline=timeline,
                stream=stream,
                columns=columns,
                lookup=lookup,
                claimed=claimed,
                codes=codes,
                chunk_rows=chunk_rows,
            )
        unconsumed = int(np.count_nonzero(~claimed))
        if unconsumed:
            raise T2TimelineError(
                f"{unconsumed} of the {record_id} cache's {rows_here} persisted "
                "rows were never consumed by the timeline. A silently ignored "
                "target row means the replay is not the frozen population."
            )
        del lookup, columns, claimed

    unresolved = int(np.count_nonzero(codes == FAMILY_CODE_UNRESOLVED))
    if unresolved:
        raise T2TimelineError(
            f"{unresolved} timeline rows resolved to no persisted target family."
        )
    if consumed_rows != timeline.row_count:
        raise T2TimelineError(
            f"The join consumed {consumed_rows} rows against a timeline of "
            f"{timeline.row_count}."
        )

    census = {
        name: int(np.count_nonzero(codes == FAMILY_CODE[name]))
        for name in T2_TARGET_FAMILY_ORDER
    }
    lineage = {
        "identity_class": "t2_target_authority_identity",
        "authority": "ltstdb_baseline_v1_feature_corpus",
        "partition": partition,
        "feature_corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": sha256_file(path),
        "record_cache_sha256": dict(sorted(record_digests.items())),
        "record_count": len(record_digests),
        "row_count": int(timeline.row_count),
        "primary_row_count": int(
            census["ischemic_positive"] + census["background_negative"]
        ),
        "ischemic_positive": census["ischemic_positive"],
        "background_negative": census["background_negative"],
        "challenge_row_count": int(
            sum(census[name] for name in T2_CHALLENGE_CATEGORIES)
        ),
        "other_non_primary_row_count": int(
            sum(census[name] for name in T2_OTHER_NON_PRIMARY_CATEGORIES)
        ),
        "target_family_counts": census,
        "ordered_stable_id_sha256": timeline.manifest["ordered_stable_id_sha256"],
        "ordered_chronology_sha256": timeline.manifest["ordered_chronology_sha256"],
        "stream_cache_sha256": timeline.manifest["stream_cache_sha256"],
        "representation_content_sha256": (
            timeline.manifest["representation_content_sha256"]
        ),
        "join_key": "stable_id",
        "join_is_record_wise_bounded": True,
        "raw_annotations_reread": False,
        "stb_reinterpreted": False,
        "labels_derived_from_context_flags": False,
        "every_row_resolved_exactly_once": True,
        "target_family_is_model_input": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    if timeline.canonical_source:
        require_frozen_family_census(partition, lineage)
    return codes, lineage


def _streams_by_record(timeline: T2Timeline) -> dict[str, tuple[T2Stream, ...]]:
    """Group the timeline's streams by record, preserving timeline order."""
    grouped: dict[str, list[T2Stream]] = {}
    for stream in timeline.streams():
        grouped.setdefault(stream.record_id, []).append(stream)
    return {record: tuple(items) for record, items in grouped.items()}


def _resolve_stream_families(
    *,
    timeline: T2Timeline,
    stream: T2Stream,
    columns: dict[str, np.ndarray],
    lookup: dict[str, int],
    claimed: np.ndarray,
    codes: np.ndarray,
    chunk_rows: int,
) -> int:
    """Resolve one stream's rows against one record's persisted target columns."""
    families = columns["target_families"]
    corpus_records = columns["record_ids"]
    corpus_channels = columns["channel_indices"]
    corpus_starts = columns["window_start_samples"]
    corpus_ends = columns["window_end_samples"]

    resolved = 0
    for begin in range(stream.start_index, stream.stop_index, chunk_rows):
        end = min(begin + chunk_rows, stream.stop_index)
        stable_ids = timeline.stable_ids(begin, end)
        starts = timeline.column(START_SAMPLE_FILE, begin, end)
        for offset, raw_identifier in enumerate(stable_ids.tolist()):
            identifier = str(raw_identifier)
            position = lookup.get(identifier)
            if position is None:
                raise T2TimelineError(
                    f"Timeline row {begin + offset} carries stable id "
                    f"{identifier!r}, which the frozen {stream.record_id} cache "
                    "does not contain. No target family is invented for it."
                )
            if claimed[position]:
                raise T2TimelineError(
                    f"Persisted target row {identifier!r} was claimed twice; a "
                    "timeline row must resolve to exactly one target family."
                )
            row_start = int(starts[offset])
            if str(corpus_records[position]) != stream.record_id:
                raise T2TimelineError(
                    f"Target row {identifier!r} names record "
                    f"{corpus_records[position]!r}, not {stream.record_id!r}."
                )
            if int(corpus_channels[position]) != int(stream.channel_index):
                raise T2TimelineError(
                    f"Target row {identifier!r} names channel "
                    f"{int(corpus_channels[position])}, not "
                    f"{int(stream.channel_index)}."
                )
            if int(corpus_starts[position]) != row_start:
                raise T2TimelineError(
                    f"Target row {identifier!r} starts at "
                    f"{int(corpus_starts[position])}, but the timeline row "
                    f"starts at {row_start}."
                )
            expected_end = row_start + T2_WINDOW_LENGTH_SAMPLES
            if int(corpus_ends[position]) != expected_end:
                raise T2TimelineError(
                    f"Target row {identifier!r} ends at "
                    f"{int(corpus_ends[position])}; a "
                    f"{T2_WINDOW_LENGTH_SAMPLES}-sample window starting at "
                    f"{row_start} ends at {expected_end}."
                )
            family = str(families[position])
            code = FAMILY_CODE.get(family)
            if code is None:
                raise T2TimelineError(
                    f"Target row {identifier!r} carries family {family!r}; the "
                    f"frozen families are {list(T2_TARGET_FAMILY_ORDER)}."
                )
            codes[begin + offset] = code
            claimed[position] = True
            resolved += 1
    return resolved


def require_frozen_family_census(partition: str, lineage: dict[str, Any]) -> None:
    """A canonical join reproduces the frozen census exactly, field by field."""
    expected = EXPECTED_FAMILY_CENSUS[refuse_sealed_partition(partition)]
    for field_, value in expected.items():
        observed = int(lineage[field_])
        if observed != int(value):
            raise T2TimelineError(
                f"The canonical {partition} target join produced {field_}="
                f"{observed} against the frozen {value}. The population is not "
                "the frozen one and nothing proceeds."
            )


def ordered_stable_id_digest_for_rows(
    timeline: T2Timeline,
    positions: Iterable[int] | None = None,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> str:
    """Order-sensitive stable-id digest over the timeline, streamed.

    With `positions` omitted this re-derives the whole-timeline digest from the
    persisted `stable_id.npy`, which is exactly the identity the M1 manifest
    binds -- so the population a T2 artifact claims is provably the one M1
    promoted, not merely one of the same length.
    """

    def identifiers() -> Iterator[str]:
        if positions is not None:
            column = timeline.store.array(STABLE_ID_FILE)
            for index in positions:
                yield str(column[int(index)])
            return
        for begin in range(0, timeline.row_count, chunk_rows):
            end = min(begin + chunk_rows, timeline.row_count)
            yield from (str(value) for value in timeline.stable_ids(begin, end))

    return streaming_ordered_stable_id_digest(identifiers())


def assign_row_roles(
    *, categories: np.ndarray, observation_state: np.ndarray
) -> np.ndarray:
    """Assign one role per row. Availability dominates the target family.

    A physically unavailable observation is `UNAVAILABLE_NO_STATE_UPDATE`
    whatever its label says: there is no observation to consume, so there is
    nothing for the model to see and nothing to score.
    """
    if categories.shape != observation_state.shape:
        raise T2TimelineError(
            f"Category and observation-state arrays disagree: "
            f"{categories.shape} vs {observation_state.shape}."
        )
    roles = np.empty(categories.shape, dtype="<U40")
    for index, (category, state) in enumerate(
        zip(categories.tolist(), observation_state.tolist(), strict=True)
    ):
        if int(state) == T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT:
            roles[index] = ROLE_UNAVAILABLE_NO_STATE_UPDATE
            continue
        if int(state) != T2_OBSERVATION_AVAILABLE:
            raise T2TimelineError(
                f"Observation state {state} is neither AVAILABLE nor "
                "UNAVAILABLE_EXACT_FLAT; an uninitialised row cannot be replayed."
            )
        roles[index] = role_for_category(str(category))
    return roles


def direct_loss_mask(roles: np.ndarray) -> np.ndarray:
    """Only PRIMARY rows carry direct loss. Nothing else ever does."""
    return roles == ROLE_PRIMARY_DIRECT_LOSS


def context_mask(roles: np.ndarray) -> np.ndarray:
    """Rows the model consumes as causal context: every available role."""
    return roles != ROLE_UNAVAILABLE_NO_STATE_UPDATE


# ---------------------------------------------------------------------------
# The coded equivalents. Identical semantics, `uint8` instead of `<U40`.
# ---------------------------------------------------------------------------


def role_codes_for_families(
    family_codes: np.ndarray, observation_state: np.ndarray | None = None
) -> np.ndarray:
    """Coded row roles. Availability dominates the target family, as always.

    With `observation_state` omitted every row is treated as available, which is
    what the whole-timeline census wants: the family census counts persisted
    labels, and the availability mask is applied where rows are actually
    consumed.
    """
    codes = np.asarray(family_codes, dtype=np.uint8)
    if int(codes.max(initial=0)) >= len(FAMILY_NAME):
        raise T2TimelineError("A family code falls outside the frozen family table.")
    roles = _ROLE_CODE_BY_FAMILY_CODE[codes]
    if observation_state is None:
        return roles
    states = np.asarray(observation_state)
    if states.shape != codes.shape:
        raise T2TimelineError(
            f"Family and observation-state arrays disagree: "
            f"{codes.shape} vs {states.shape}."
        )
    unavailable = states == T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT
    available = states == T2_OBSERVATION_AVAILABLE
    if int(np.count_nonzero(~(available | unavailable))):
        raise T2TimelineError(
            "An observation state is neither AVAILABLE nor "
            "UNAVAILABLE_EXACT_FLAT; an uninitialised row cannot be replayed."
        )
    roles = roles.copy()
    roles[unavailable] = ROLE_CODE_UNAVAILABLE
    return roles


def role_names_for_codes(role_codes: np.ndarray) -> np.ndarray:
    """Expand coded roles back to the frozen role strings."""
    table = np.asarray(ROLE_NAME_BY_CODE, dtype="<U40")
    return table[np.asarray(role_codes, dtype=np.uint8)]


def primary_labels_for_families(family_codes: np.ndarray) -> np.ndarray:
    """The binary PRIMARY target: `ischemic_positive` is 1, everything else 0.

    Reading a label for a non-PRIMARY row is meaningless, so callers mask first;
    this returns 0 there rather than a sentinel precisely so that an unmasked
    use cannot silently manufacture a positive.
    """
    return (np.asarray(family_codes, dtype=np.uint8) == POSITIVE_FAMILY_CODE).astype(
        np.int64
    )
