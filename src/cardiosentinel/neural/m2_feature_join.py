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
        lookup = {value: index for index, value in enumerate(npz_ids)}
        block_ids = stable_ids[start:end]
        try:
            positions = np.asarray([lookup[sid] for sid in block_ids], dtype=np.int64)
        except KeyError as error:
            raise M2FeatureJoinError(
                f"A {evaluated.upper()} stream row for record {record_id} has no "
                f"COMBINED_V1 match: {error}."
            ) from error
        for name, column in column_indices.items():
            columns[name][start:end] = npz_features[positions, column]

    for name, values in columns.items():
        if np.any(np.isnan(values)):
            raise M2FeatureJoinError(
                f"COMBINED_V1 {evaluated} join left unmatched rows for {name!r}."
            )
    return columns
