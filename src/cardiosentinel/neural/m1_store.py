"""Bounded-memory primitives for the canonical M1 execution path.

Attempt 1 of the canonical M1 Stage-1 run was terminated after 6h41m with exit
137 and no traceback, in a way strongly consistent with host memory exhaustion.
The cause was not the science: the implementation materialized the entire
full-stream representation in memory before persisting anything — one ndarray
object per newly extracted row, a whole-corpus physiology mapping, millions of
reference objects and the stacked matrices, all live at once.

This module supplies the pieces that make peak memory a function of chunk size
rather than corpus size:

* `StreamingContentDigest` and the canonical-JSON streaming digests reproduce
  the EXISTING digest identities byte-for-byte, so no historical identity moves.
* `M1RowStore` is a row-aligned, disk-backed, immutable store built on
  `numpy.lib.format.open_memmap`. Rows are written once, in canonical order,
  and read back memory-mapped.

Nothing here changes M1 science. The frozen protocol fixes the representation,
the memory contract and the evidence, not the storage format.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.lib.format import open_memmap

# Implementation identity of the on-disk layout. This is an ENGINEERING
# version, deliberately separate from the frozen M1 scientific protocol SHA:
# the protocol does not fix a serialization format.
M1_STREAM_CACHE_SCHEMA: Final = 2

# Row chunk used by every bounded operation. Sized so one chunk of the widest
# array (float32 [chunk, 146]) stays in the low tens of MB.
DEFAULT_CHUNK_ROWS: Final = 8_192

STABLE_ID_DTYPE: Final = "<U64"
RECORD_ID_DTYPE: Final = "<U64"


class M1StoreError(RuntimeError):
    """Raised when a bounded-memory store operation cannot proceed safely."""


# --------------------------------------------------------------------------
# Streaming digests — must reproduce the existing identities exactly
# --------------------------------------------------------------------------


class StreamingContentDigest:
    """Chunked equivalent of `embedding_content_digest`.

    The existing identity is

        sha256( repr((shape, str(dtype))) + contiguous row-major bytes )

    so a streaming version only has to know the final shape and dtype up front
    and then feed C-contiguous row blocks in order. Byte-for-byte equality with
    the legacy function is asserted by tests over float32/float64/int64 and
    non-trivial chunk boundaries; if it ever diverged, every M1 cache identity
    would silently change.
    """

    __slots__ = ("_hasher", "_dtype", "_shape", "_rows_seen")

    def __init__(self, shape: tuple[int, ...], dtype: Any) -> None:
        self._shape = tuple(int(v) for v in shape)
        self._dtype = np.dtype(dtype)
        self._hasher = hashlib.sha256()
        self._hasher.update(repr((self._shape, str(self._dtype))).encode("utf-8"))
        self._rows_seen = 0

    def update(self, block: np.ndarray) -> None:
        chunk = np.ascontiguousarray(block, dtype=self._dtype)
        if chunk.shape[1:] != self._shape[1:]:
            raise M1StoreError(
                f"Streaming digest chunk has trailing shape {chunk.shape[1:]}, "
                f"expected {self._shape[1:]}."
            )
        self._hasher.update(chunk.tobytes())
        self._rows_seen += int(chunk.shape[0]) if chunk.ndim else 1

    def hexdigest(self) -> str:
        if self._rows_seen != self._shape[0]:
            raise M1StoreError(
                f"Streaming digest saw {self._rows_seen} rows, expected "
                f"{self._shape[0]}. A truncated stream must never produce a "
                "digest."
            )
        return self._hasher.hexdigest()


def digest_array_file(path: Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> str:
    """Content digest of a persisted `.npy` array without loading it whole."""
    array = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    digest = StreamingContentDigest(array.shape, array.dtype)
    for start in range(0, array.shape[0], chunk_rows):
        digest.update(np.asarray(array[start : start + chunk_rows]))
    return digest.hexdigest()


class StreamingCanonicalArrayDigest:
    """Incremental `canonical_sha256({key: ..., list_key: [...]})`.

    `canonical_sha256` is `json.dumps(payload, sort_keys=True,
    separators=(",", ":"))` hashed as UTF-8. For a payload whose only large
    member is one list, the serialization is fully predictable, so the list can
    be emitted element by element instead of being built in full. Tests pin the
    streaming result against the legacy whole-object call.
    """

    __slots__ = ("_hasher", "_list_key", "_first", "_closed", "_tail")

    def __init__(self, scalars: dict[str, Any], list_key: str) -> None:
        if list_key in scalars:
            raise M1StoreError("The streamed list key must not also be a scalar.")
        ordered = sorted([*scalars.keys(), list_key])
        self._hasher = hashlib.sha256()
        self._list_key = list_key

        head_parts: list[str] = ["{"]
        tail_parts: list[str] = []
        seen_list = False
        for index, key in enumerate(ordered):
            separator = "," if index else ""
            if key == list_key:
                head_parts.append(f'{separator}{json.dumps(key)}:[')
                seen_list = True
                continue
            fragment = (
                f"{separator}{json.dumps(key)}:"
                f"{json.dumps(scalars[key], sort_keys=True, separators=(',', ':'))}"
            )
            (tail_parts if seen_list else head_parts).append(fragment)
        if not seen_list:
            raise M1StoreError("The streamed list key was never emitted.")

        self._hasher.update("".join(head_parts).encode("utf-8"))
        self._tail = "]" + "".join(tail_parts) + "}"
        self._first = True
        self._closed = False

    def append(self, element: Any) -> None:
        if self._closed:
            raise M1StoreError("Cannot append to a closed streaming digest.")
        prefix = "" if self._first else ","
        self._first = False
        self._hasher.update(
            (
                prefix + json.dumps(element, sort_keys=True, separators=(",", ":"))
            ).encode("utf-8")
        )

    def hexdigest(self) -> str:
        if not self._closed:
            self._hasher.update(self._tail.encode("utf-8"))
            self._closed = True
        return self._hasher.hexdigest()


def streaming_ordered_stable_id_digest(
    identifiers: Iterator[str] | Sequence[str],
) -> str:
    """Order-sensitive stable-ID digest without materializing the whole list.

    Duplicate detection still requires a set of identifiers, which is bounded by
    the row count but is one small Python string per row rather than one ndarray
    object per row. That is the deliberate trade: identity checking cannot be
    weakened, so this stays O(N) in cheap objects and O(1) in arrays.
    """
    digest = StreamingCanonicalArrayDigest({"order": "row_order"}, "stable_ids")
    seen: set[str] = set()
    for value in identifiers:
        key = str(value)
        if key in seen:
            raise M1StoreError("Embedding stable IDs contain duplicates.")
        seen.add(key)
        digest.append(key)
    return digest.hexdigest()


def streaming_chronology_digest(rows: Iterator[Sequence[Any]]) -> str:
    """Order-sensitive `(record_id, channel_index, start_sample)` digest."""
    digest = StreamingCanonicalArrayDigest(
        {"order": "stream_then_start_sample"}, "rows"
    )
    for record_id, channel_index, start_sample in rows:
        digest.append([str(record_id), int(channel_index), int(start_sample)])
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Row-aligned disk-backed store
# --------------------------------------------------------------------------

REPRESENTATION_FILE: Final = "representation.npy"
STABLE_ID_FILE: Final = "stable_id.npy"
RECORD_ID_FILE: Final = "record_id.npy"
CHANNEL_INDEX_FILE: Final = "channel_index.npy"
START_SAMPLE_FILE: Final = "start_sample.npy"
D_SHORT_FILE: Final = "d_short.npy"
D_LONG_FILE: Final = "d_long.npy"
PAST_OBSERVED_FILE: Final = "past_observed_count.npy"
PAST_UPDATE_FILE: Final = "past_update_count.npy"
DISAGREEMENT_FILE: Final = "prototype_disagreement.npy"
RECORDING_AGE_FILE: Final = "recording_age_seconds.npy"
COLD_START_BIN_FILE: Final = "cold_start_bin.npy"

IDENTITY_FILES: Final = (
    STABLE_ID_FILE,
    RECORD_ID_FILE,
    CHANNEL_INDEX_FILE,
    START_SAMPLE_FILE,
)
MEMORY_FILES: Final = (
    D_SHORT_FILE,
    D_LONG_FILE,
    PAST_OBSERVED_FILE,
    PAST_UPDATE_FILE,
    DISAGREEMENT_FILE,
    RECORDING_AGE_FILE,
    COLD_START_BIN_FILE,
)


@dataclass(frozen=True, slots=True)
class M1StoreSpec:
    """The row-aligned layout of one partition's development store."""

    rows: int
    representation_dim: int

    def arrays(self) -> dict[str, tuple[tuple[int, ...], str]]:
        return {
            REPRESENTATION_FILE: ((self.rows, self.representation_dim), "float32"),
            STABLE_ID_FILE: ((self.rows,), STABLE_ID_DTYPE),
            RECORD_ID_FILE: ((self.rows,), RECORD_ID_DTYPE),
            CHANNEL_INDEX_FILE: ((self.rows,), "int64"),
            START_SAMPLE_FILE: ((self.rows,), "int64"),
            D_SHORT_FILE: ((self.rows,), "float64"),
            D_LONG_FILE: ((self.rows,), "float64"),
            PAST_OBSERVED_FILE: ((self.rows,), "int64"),
            PAST_UPDATE_FILE: ((self.rows,), "int64"),
            DISAGREEMENT_FILE: ((self.rows,), "float64"),
            RECORDING_AGE_FILE: ((self.rows,), "float64"),
            COLD_START_BIN_FILE: ((self.rows,), "<U32"),
        }


class M1RowStore:
    """A disk-backed, row-aligned store for one development partition.

    Every array is opened through `open_memmap`, so writing a row touches only
    the corresponding page: the process never holds the corpus. Reads use
    `mmap_mode="r"`, so validation and memory generation also stay bounded.

    The store is write-once by convention — the canonical route creates it in a
    staging directory and promotes it — and it is never mutated after the
    manifest is written.
    """

    def __init__(self, directory: Path, spec: M1StoreSpec, *, create: bool) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self._arrays: dict[str, np.ndarray] = {}
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            for name, (shape, dtype) in spec.arrays().items():
                self._arrays[name] = open_memmap(
                    self.directory / name, mode="w+", dtype=np.dtype(dtype), shape=shape
                )
        else:
            for name, (shape, dtype) in spec.arrays().items():
                path = self.directory / name
                if not path.is_file():
                    raise M1StoreError(f"M1 store is missing {name}.")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise M1StoreError(
                        f"M1 store array {name} has shape {array.shape}/"
                        f"{array.dtype}, expected {shape}/{dtype}."
                    )
                self._arrays[name] = array

    def array(self, name: str) -> np.ndarray:
        if name not in self._arrays:
            raise M1StoreError(f"Unknown M1 store array {name!r}.")
        return self._arrays[name]

    def flush(self) -> None:
        for array in self._arrays.values():
            if isinstance(array, np.memmap):
                array.flush()

    def close(self) -> None:
        """Drop references so the mappings can be released promptly."""
        self.flush()
        self._arrays.clear()

    def __enter__(self) -> M1RowStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- bounded reads ----------------------------------------------------

    def chunks(
        self, name: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
    ) -> Iterator[np.ndarray]:
        array = self.array(name)
        for start in range(0, array.shape[0], chunk_rows):
            yield np.asarray(array[start : start + chunk_rows])

    def content_digest(
        self, name: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
    ) -> str:
        array = self.array(name)
        digest = StreamingContentDigest(array.shape, array.dtype)
        for start in range(0, array.shape[0], chunk_rows):
            digest.update(np.asarray(array[start : start + chunk_rows]))
        return digest.hexdigest()

    def paired_content_digest(
        self, first: str, second: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
    ) -> str:
        """Digest of two columns stacked as `[N, 2]`, without stacking them all."""
        left = self.array(first)
        right = self.array(second)
        if left.shape != right.shape:
            raise M1StoreError("Paired digest columns are not row-aligned.")
        digest = StreamingContentDigest(
            (left.shape[0], 2), np.result_type(left.dtype, right.dtype)
        )
        for start in range(0, left.shape[0], chunk_rows):
            block = np.stack(
                [
                    np.asarray(left[start : start + chunk_rows]),
                    np.asarray(right[start : start + chunk_rows]),
                ],
                axis=1,
            )
            digest.update(block)
        return digest.hexdigest()

    def stable_id_digest(self, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> str:
        def identifiers() -> Iterator[str]:
            array = self.array(STABLE_ID_FILE)
            for start in range(0, array.shape[0], chunk_rows):
                block = np.asarray(array[start : start + chunk_rows])
                yield from (str(value) for value in block)

        return streaming_ordered_stable_id_digest(identifiers())

    def chronology_digest(self, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> str:
        def triples() -> Iterator[tuple[Any, Any, Any]]:
            records = self.array(RECORD_ID_FILE)
            channels = self.array(CHANNEL_INDEX_FILE)
            starts = self.array(START_SAMPLE_FILE)
            for begin in range(0, records.shape[0], chunk_rows):
                end = begin + chunk_rows
                yield from zip(
                    np.asarray(records[begin:end]),
                    np.asarray(channels[begin:end]),
                    np.asarray(starts[begin:end]),
                    strict=True,
                )

        return streaming_chronology_digest(triples())

    def artifact_digests(self) -> dict[str, str]:
        """SHA-256 of every physical file, streamed from disk."""
        from cardiosentinel.data.provenance import sha256_file

        return {
            name: sha256_file(self.directory / name)
            for name in sorted(self.spec.arrays())
        }

    def gather(self, name: str, positions: np.ndarray) -> np.ndarray:
        """Materialize only the selected rows of one column.

        This is how the bounded path builds the supervised training and evidence
        matrices: the primary TRAIN, primary VALIDATION and challenge
        populations are small and bounded, so they may become ordinary in-memory
        arrays, while the full stream never does.
        """
        array = self.array(name)
        index = np.asarray(positions, dtype=np.int64)
        if index.size and (int(index.min()) < 0 or int(index.max()) >= array.shape[0]):
            raise M1StoreError("Row selection falls outside the M1 store.")
        return np.asarray(array[index])


def locate_rows(store: M1RowStore, wanted: Sequence[str]) -> np.ndarray:
    """Positions of an ordered ID subset, found by one bounded scan.

    A full `stable_id -> row` dictionary over the whole stream is exactly the
    kind of per-row Python object that exhausted memory in Attempt 1. Only the
    wanted identifiers are held, so this is bounded by the selected population
    rather than by the corpus.
    """
    requested = [str(value) for value in wanted]
    lookup = {key: position for position, key in enumerate(requested)}
    if len(lookup) != len(requested):
        raise M1StoreError("Requested row selection contains duplicates.")

    found = np.full(len(requested), -1, dtype=np.int64)
    array = store.array(STABLE_ID_FILE)
    for start in range(0, array.shape[0], DEFAULT_CHUNK_ROWS):
        block = np.asarray(array[start : start + DEFAULT_CHUNK_ROWS])
        for offset, value in enumerate(block):
            position = lookup.get(str(value))
            if position is not None:
                found[position] = start + offset
    missing = int(np.sum(found < 0))
    if missing:
        raise M1StoreError(
            f"{missing} requested rows are absent from the M1 full-stream store."
        )
    return found
