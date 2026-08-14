"""Partition-aware COMBINED_V1 feature join for scientific timeline assembly.

`m2_gate_derivation.join_sqi_and_morphology` was written for the frozen M2
**TRAIN** gate derivation and selects its record cache paths through
`_train_record_cache_paths`, which filters the COMBINED_V1 manifest to
`partition == "train"`. That is correct for the task it was written for, and it
is not changed here: the frozen TRAIN receipt must keep meaning what it meant.

The canonical DEVELOPMENT route replays VALIDATION, where that TRAIN-only set
can never equal the stream cache's record list. M2 development attempt #1
consumed both arm claims and then failed on exactly that mismatch, before any
row was scored -- see
`docs/M2_DEVELOPMENT_ATTEMPT1_FAILURE_AND_RECOVERY_DECISION_V1.md`.

So scientific timeline assembly gets its own helper that takes the partition
explicitly and names it in every refusal. The join itself is otherwise
identical: alignment is by frozen stable identity, and an unmatched, missing or
extra row is fatal rather than inner-joined away.

The partition is always supplied by the caller from an authorized constant. It
is never derived from a label, a score, a prediction or any observed result.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.baseline.cache import require_nonversioned_path
from cardiosentinel.neural.m1_store import RECORD_ID_FILE, STABLE_ID_FILE
from cardiosentinel.neural.m2_gate_derivation import (
    COMBINED_NEEDED_COLUMNS,
    FEATURE_MANIFEST_NAME,
    _combined_column_indices,
)

FORBIDDEN_JOIN_PARTITIONS: Final = ("test",)


class M2FeatureJoinError(RuntimeError):
    """Raised when the partition-aware COMBINED_V1 join cannot proceed."""


def require_join_partition(partition: str) -> str:
    """Only a development partition may ever be joined here.

    TEST is refused before the feature manifest is opened, so no sealed-test
    cache path can be resolved by this route.
    """
    evaluated = str(partition).strip().lower()
    if evaluated in FORBIDDEN_JOIN_PARTITIONS:
        raise M2FeatureJoinError(
            f"The COMBINED_V1 join hard-rejects the {evaluated!r} partition; "
            "the B4 sealed test remains unopened."
        )
    if evaluated not in ("train", "validation"):
        raise M2FeatureJoinError(
            f"Unknown development partition {evaluated!r}; the COMBINED_V1 join "
            "accepts 'train' or 'validation' only."
        )
    return evaluated


def combined_record_cache_paths_for_partition(
    feature_root: Path, partition: str
) -> dict[str, Path]:
    """The COMBINED_V1 per-record cache paths for ONE named partition.

    Rejects duplicate record ids and any cache path that escapes the feature
    root. Deliberately separate from the TRAIN-only helper the frozen gate
    derivation uses, so neither can be mistaken for the other.
    """
    from cardiosentinel.baseline.cache import read_json

    evaluated = require_join_partition(partition)
    root = require_nonversioned_path(Path(feature_root), "COMBINED_V1 feature root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)

    paths: dict[str, Path] = {}
    for entry in manifest.get("records", ()):
        if entry.get("partition") != evaluated or entry.get("status") != "complete":
            continue
        record_id = str(entry["record_id"])
        cache_path = (root / str(entry["cache_path"])).resolve()
        cache_path.relative_to(root)
        if record_id in paths:
            raise M2FeatureJoinError(
                f"The COMBINED_V1 manifest lists record {record_id!r} more than "
                f"once for partition {evaluated!r}; the join would be ambiguous."
            )
        paths[record_id] = cache_path
    return paths


def _duplicates(values: list[str]) -> list[str]:
    """The repeated identities, in one pass.

    `[v for v in values if values.count(v) > 1]` is quadratic, and this runs
    only on the corrupted-record error path -- exactly where a large record
    would make the run spend minutes formatting a message instead of failing
    promptly.
    """
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _require_exact_stable_id_correspondence(
    record_id: str,
    partition: str,
    npz_ids: np.ndarray,
    npz_features: np.ndarray,
    stream_ids: np.ndarray,
) -> None:
    """EXACT set equality between a record's feature NPZ and its stream rows.

    Requiring only that every stream row has a feature match would silently
    accept a feature cache holding EXTRA rows -- a corpus that is not the one
    the stream cache was built from, quietly reduced to a subset at join time.
    A join that can drop rows can change an evaluated population without
    saying so, so both directions are fatal here.

    Order is not asserted: the join realigns by stable identity, and the frozen
    upstream contract fixes the STREAM order (the causal chronology the store
    persists), not the order rows happen to sit in a feature cache.
    """
    if npz_ids.shape[0] != npz_features.shape[0]:
        raise M2FeatureJoinError(
            f"COMBINED_V1 record {record_id} holds {npz_ids.shape[0]} stable IDs "
            f"but {npz_features.shape[0]} feature rows; the cache is not "
            "row-aligned with itself."
        )
    npz_list = npz_ids.tolist()
    npz_set = set(npz_list)
    if len(npz_set) != len(npz_list):
        raise M2FeatureJoinError(
            f"COMBINED_V1 record {record_id} has duplicate stable IDs "
            f"(beginning {_duplicates(npz_list)[:3]}); the join would be "
            "ambiguous."
        )
    stream_list = stream_ids.tolist()
    stream_set = set(stream_list)
    if len(stream_set) != len(stream_list):
        raise M2FeatureJoinError(
            f"The {partition.upper()} stream cache has duplicate stable IDs for "
            f"record {record_id} (beginning {_duplicates(stream_list)[:3]}); the "
            "evaluated population would be ambiguous."
        )
    missing = sorted(stream_set - npz_set)
    extra = sorted(npz_set - stream_set)
    if missing or extra:
        raise M2FeatureJoinError(
            f"COMBINED_V1 record {record_id} does not correspond exactly to its "
            f"{partition.upper()} stream rows: {len(missing)} stream rows have "
            f"no feature match (beginning {missing[:3]}) and {len(extra)} "
            f"feature rows are absent from the stream cache (beginning "
            f"{extra[:3]}). No row is dropped to make the join succeed."
        )


def require_all_rows_written(
    written: np.ndarray, partition: str, stable_ids: np.ndarray
) -> None:
    """Prove every row was STRUCTURALLY assigned by the join.

    Separate from the feature values on purpose. The previous implementation
    initialised the destination with NaN and treated any remaining NaN as an
    unwritten row -- but NaN is also the legitimate representation of an
    upstream source null, so a valid corpus raised a structural error and
    consumed M2 development recovery1.
    """
    unwritten = int(np.count_nonzero(~written))
    if not unwritten:
        return
    missing = np.flatnonzero(~written)
    raise M2FeatureJoinError(
        f"The COMBINED_V1 {partition} join structurally assigned only "
        f"{int(np.count_nonzero(written))} of {written.shape[0]} rows; "
        f"{unwritten} were never written, beginning at stream positions "
        f"{missing[:3].tolist()} "
        f"({[str(v) for v in stable_ids[missing[:3]]]}). No row is dropped to "
        "make the join succeed."
    )


def join_sqi_and_morphology_for_partition(
    store, manifest: dict[str, Any], feature_root: Path, partition: str
) -> dict[str, np.ndarray]:
    """Row-align the frozen SQI + morphology columns for ONE named partition.

    Requires exact record-set equality between the COMBINED_V1 corpus and the
    M1 stream-cache manifest **for the same partition**, then joins strictly by
    frozen stable identity. A missing, unmatched or extra row is fatal: this
    never inner-joins a disagreement away, because doing so would silently
    change the evaluated population.
    """
    evaluated = require_join_partition(partition)
    manifest_partition = str(manifest.get("partition", "")).strip().lower()
    if manifest_partition and manifest_partition != evaluated:
        raise M2FeatureJoinError(
            f"The stream-cache manifest is partition {manifest_partition!r} but "
            f"the join was asked for {evaluated!r}. The two must be the same "
            "partition; this is the exact mismatch that consumed M2 development "
            "attempt #1."
        )

    record_ids = np.asarray(store.array(RECORD_ID_FILE))
    stable_ids = np.asarray(store.array(STABLE_ID_FILE))
    rows = record_ids.shape[0]
    column_indices = _combined_column_indices()
    columns = {
        name: np.full(rows, np.nan, dtype=np.float64)
        for name in COMBINED_NEEDED_COLUMNS
    }
    # STRUCTURAL assignment is tracked separately from feature VALUES. Using
    # `isnan(output)` as proof that a row was never written conflates the two,
    # because NaN is also the legitimate representation of an upstream source
    # null -- a spectral ratio the frozen signal contract permits to be
    # uncomputable. That conflation consumed M2 development recovery1; see
    # `docs/M2_DEVELOPMENT_RECOVERY1_FAILURE_AND_RECOVERY2_DECISION_V1.md`.
    written = np.zeros(rows, dtype=bool)

    record_paths = combined_record_cache_paths_for_partition(feature_root, evaluated)
    expected = {str(value) for value in manifest["record_ids"]}
    observed = set(record_paths)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise M2FeatureJoinError(
            f"The COMBINED_V1 feature corpus's {evaluated.upper()} record set "
            f"does not match the M1 stream cache's {evaluated.upper()} record "
            f"list: {len(missing)} absent from the corpus (beginning "
            f"{missing[:3]}) and {len(extra)} present that the stream cache does "
            f"not list (beginning {extra[:3]})."
        )

    boundary = np.empty(rows, dtype=bool)
    boundary[0] = True
    boundary[1:] = record_ids[1:] != record_ids[:-1]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], rows]

    for start, end in zip(starts, ends, strict=True):
        record_id = str(record_ids[start])
        with np.load(record_paths[record_id], allow_pickle=False) as cached:
            npz_ids = np.asarray(cached["stable_ids"], dtype=np.str_)
            npz_features = np.asarray(cached["features"], dtype=np.float64)
        block_ids = stable_ids[start:end]
        _require_exact_stable_id_correspondence(
            record_id, evaluated, npz_ids, npz_features, block_ids
        )
        lookup = {value: index for index, value in enumerate(npz_ids)}
        positions = np.asarray([lookup[sid] for sid in block_ids], dtype=np.int64)
        for name, column in column_indices.items():
            # Source values are carried through EXACTLY. A legitimate null
            # survives as NaN; it is never replaced by zero, a TRAIN median, a
            # bound or an infinity, and its row is never dropped. The M2 policy
            # owns what such a value means -- for an AVAILABLE row a non-finite
            # G3 feature already fails G3, and an unavailable row already makes
            # G2-G6 not applicable.
            columns[name][start:end] = npz_features[positions, column]
        written[start:end] = True

    require_all_rows_written(written, evaluated, stable_ids)
    return columns
