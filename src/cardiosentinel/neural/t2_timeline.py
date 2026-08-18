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

The target family is a **masking and evaluation authority only**. It selects
which rows carry direct loss and which metrics they feed. It never enters the
trainable `z_t` vector -- there is no code path from a role to a model input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

import numpy as np

from cardiosentinel.data.ltstdb import subject_id_for_record
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    COLD_START_BIN_FILE,
    OBSERVATION_STATE_FILE,
    RECORD_ID_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
    M1RowStore,
    M1StoreSpec,
)
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.t2_protocol import (
    ROLE_CHALLENGE_CONTEXT,
    ROLE_OTHER_NONPRIMARY_CONTEXT,
    ROLE_PRIMARY_DIRECT_LOSS,
    ROLE_UNAVAILABLE_NO_STATE_UPDATE,
    T2_CHALLENGE_CATEGORIES,
    T2_INPUT_DIM,
    T2_OBSERVATION_AVAILABLE,
    T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    T2_OTHER_NON_PRIMARY_CATEGORIES,
    T2_PRIMARY_CATEGORIES,
    T2_TRAIN_FULL_STREAM_ROW_COUNT,
    T2_TRAIN_P1_EMBEDDING_CACHE_SHA256,
    T2_TRAIN_REPRESENTATION_CONTENT_SHA256,
    T2_TRAIN_STREAM_CACHE_SHA256,
    T2_VALIDATION_FULL_STREAM_ROW_COUNT,
    T2_VALIDATION_P1_EMBEDDING_CACHE_SHA256,
    T2_VALIDATION_REPRESENTATION_CONTENT_SHA256,
    T2_VALIDATION_STREAM_CACHE_SHA256,
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
    """

    def __init__(self, partition: str, *, root: Path | None = None) -> None:
        self.partition = refuse_sealed_partition(partition)
        # The canonical route passes no root, and only that path enforces the
        # frozen row count. A synthetic fixture supplies its own root and is
        # therefore never mistaken for the corpus: the identity it reports says
        # so explicitly.
        self.canonical_source = root is None
        directory = Path(root) if root is not None else STREAM_CACHE_ROOT
        refuse_forbidden_source(directory)
        self.directory = directory / self.partition
        manifest_path = self.directory / STREAM_CACHE_MANIFEST
        if not manifest_path.is_file():
            raise T2TimelineError(f"No M1 stream cache manifest at {manifest_path}.")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        self._validate_manifest()
        rows = int(self.manifest["full_stream_row_count"])
        self.store = M1RowStore(
            self.directory,
            M1StoreSpec(rows=rows, representation_dim=T2_INPUT_DIM),
            create=False,
        )
        self.row_count = rows
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
        if self.canonical_source:
            require_frozen_row_count(
                self.partition, int(manifest.get("full_stream_row_count", -1))
            )
        expected_cache = EXPECTED_STREAM_CACHE_SHA256[self.partition]
        if manifest.get("stream_cache_sha256") != expected_cache:
            raise T2TimelineError(
                f"The {self.partition} stream cache digests to "
                f"{manifest.get('stream_cache_sha256')!r}, not the frozen "
                f"{expected_cache!r}."
            )
        expected_representation = EXPECTED_REPRESENTATION_SHA256[self.partition]
        if manifest.get("representation_content_sha256") != expected_representation:
            raise T2TimelineError(
                f"The {self.partition} representation content digests to "
                f"{manifest.get('representation_content_sha256')!r}, not the "
                f"frozen {expected_representation!r}."
            )
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
