"""Typed evidence stores and member-restricted upstream readers for T1.

Two things live here.

**The T1 stores.** The label-blind input evidence store and the cross-fitted
OOF state evidence store, both persisted as compact typed arrays with a
digest-bound JSON manifest beside them. Neither may carry a label, a target
family, an episode identity, a challenge identity or any TEST field, and
``require_evidence_column_permitted`` refuses one structurally rather than
trusting the caller.

**The member-restricted upstream readers.** The existing convenience readers
materialise every column named in a manifest entry. For the T2 outer row
identity that includes ``label``, ``target_family`` and ``primary_mask``, so
either would silently pull evaluation annotation into memory during a step that
is supposed to be label-blind. The readers here name what they materialise, and
refuse a forbidden member by name.

Nothing in this module runs the state machine, selects a policy or computes a
metric. It moves arrays and proves their identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.t1_execution_spec import (
    T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS,
    T1_EXPECTED_SCORED_ROWS,
    T1_EXPECTED_UNAVAILABLE_ROWS,
    T1_INPUT_EVIDENCE_COLUMNS,
    T1_M2_COLUMNS_USED,
    T1_OOF_STATE_EVIDENCE_COLUMNS,
    T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND,
    T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND,
    T1_TIMELINE_ROW_COUNT,
    require_evidence_column_permitted,
    require_label_blind_member,
)

INPUT_EVIDENCE_MANIFEST_NAME: Final = "T1_INPUT_EVIDENCE.json"
INPUT_EVIDENCE_ARRAY_NAME: Final = "t1_input_evidence.npz"
OOF_STATE_MANIFEST_NAME: Final = "T1_OOF_STATE_EVIDENCE.json"
OOF_STATE_ARRAY_NAME: Final = "t1_oof_state_evidence.npz"

INPUT_EVIDENCE_SCHEMA: Final = "t1_v1_label_blind_input_evidence/1"
OOF_STATE_SCHEMA: Final = "t1_v1_oof_state_evidence/1"

# Absence is stored as a NaN sentinel, never as a plausible score. The
# `score_present` mask is authoritative; the sentinel exists so the arrays stay
# rectangular, and every reader is expected to consult the mask first.
ABSENT: Final = float("nan")

_DTYPES: Final = {
    "stable_id": np.str_,
    "record_id": np.str_,
    "channel_index": np.int32,
    "start_sample": np.int64,
    "subject_id": np.str_,
    "score_present": np.bool_,
    "m2g_detector_score": np.float64,
    "detector_decision_d_t": np.bool_,
    "oof_calibrated_probability_p_t": np.float64,
    "decision_error_uncertainty_u_t": np.float64,
    "s4d_temporal_evidence_s_t": np.float64,
    "elapsed_stream_seconds": np.float64,
    "fold_index": np.int32,
    "selected_policy_id": np.str_,
    "p_watch": np.float64,
    "s_watch": np.float64,
    "p_event": np.float64,
    "s_event": np.float64,
    "emitted_state": np.str_,
    "state_elapsed_seconds": np.float64,
    "transition_from": np.str_,
    "transition_to": np.str_,
    "transition_occurred": np.bool_,
}


class T1EvidenceStoreError(RuntimeError):
    """Raised when an evidence store is malformed, misaligned or contaminated."""


def _require_columns(
    arrays: dict[str, np.ndarray], expected: Sequence[str], label: str
) -> int:
    missing = sorted(set(expected) - set(arrays))
    if missing:
        raise T1EvidenceStoreError(f"{label} is missing columns {missing}.")
    extra = sorted(set(arrays) - set(expected))
    if extra:
        raise T1EvidenceStoreError(
            f"{label} carries unexpected columns {extra}. An evidence store holds "
            "exactly its frozen schema; an extra column is either a contaminant or "
            "a schema change that was never reviewed."
        )
    for column in expected:
        require_evidence_column_permitted(column)
    counts = {len(np.asarray(arrays[column])) for column in expected}
    if len(counts) != 1:
        raise T1EvidenceStoreError(f"{label} columns disagree on row count: {counts}.")
    return counts.pop()


def _typed(column: str, values: Any) -> np.ndarray:
    dtype = _DTYPES.get(column)
    if dtype is None:
        raise T1EvidenceStoreError(f"No frozen dtype for column {column!r}.")
    return np.asarray(values, dtype=dtype)


def _write_store(
    directory: Path,
    *,
    arrays: dict[str, np.ndarray],
    columns: Sequence[str],
    array_name: str,
    manifest_name: str,
    schema: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row_count = _require_columns(arrays, columns, manifest_name)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / array_name
    if array_path.exists():
        raise T1EvidenceStoreError(
            f"{array_path} already exists. Promoted evidence is immutable and is "
            "never overwritten."
        )
    typed = {column: _typed(column, arrays[column]) for column in columns}
    np.savez(array_path, **typed)
    manifest = {
        "schema": schema,
        "array_file": array_name,
        "array_sha256": sha256_file(array_path),
        "columns": list(columns),
        "row_count": int(row_count),
        "contains_label": False,
        "contains_target_family": False,
        "contains_challenge_identity": False,
        "test_accessed": False,
        **extra,
    }
    manifest["content_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )
    write_json_atomic(directory / manifest_name, manifest)
    return manifest


def write_input_evidence(
    directory: Path, arrays: dict[str, np.ndarray], *, lineage: dict[str, Any]
) -> dict[str, Any]:
    """Persist the one label-blind input timeline."""
    return _write_store(
        directory,
        arrays=arrays,
        columns=T1_INPUT_EVIDENCE_COLUMNS,
        array_name=INPUT_EVIDENCE_ARRAY_NAME,
        manifest_name=INPUT_EVIDENCE_MANIFEST_NAME,
        schema=INPUT_EVIDENCE_SCHEMA,
        extra={"label_blind": True, "upstream_lineage": dict(lineage)},
    )


def write_oof_state_evidence(
    directory: Path, arrays: dict[str, np.ndarray], *, fold_selection_sha256: str
) -> dict[str, Any]:
    """Persist the cross-fitted held-out state evidence, one trace per subject."""
    return _write_store(
        directory,
        arrays=arrays,
        columns=T1_OOF_STATE_EVIDENCE_COLUMNS,
        array_name=OOF_STATE_ARRAY_NAME,
        manifest_name=OOF_STATE_MANIFEST_NAME,
        schema=OOF_STATE_SCHEMA,
        extra={
            "cross_fitted": True,
            "fold_selection_sha256": fold_selection_sha256,
            "is_unseen_generalization": False,
        },
    )


def read_store(
    directory: Path, manifest_name: str, *, columns: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Read a T1 store back, digest-checked, materialising only what is asked for."""
    directory = Path(directory)
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        raise T1EvidenceStoreError(f"Evidence manifest is missing at {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = directory / manifest["array_file"]
    digest = sha256_file(array_path)
    if digest != manifest["array_sha256"]:
        raise T1EvidenceStoreError(
            f"{array_path} digests to {digest}, not the promoted "
            f"{manifest['array_sha256']}. Promoted evidence changed on disk."
        )
    wanted = tuple(columns) if columns is not None else tuple(manifest["columns"])
    unknown = sorted(set(wanted) - set(manifest["columns"]))
    if unknown:
        raise T1EvidenceStoreError(f"Store has no columns {unknown}.")
    with np.load(array_path, allow_pickle=False) as payload:
        return {column: np.asarray(payload[column]) for column in wanted}


# ---------------------------------------------------------------------------
# Member-restricted upstream readers
# ---------------------------------------------------------------------------


def read_m2g_row_evidence(
    path: Path, *, columns: Sequence[str] = T1_M2_COLUMNS_USED
) -> dict[str, np.ndarray]:
    """Read the retained M2-G replay, naming the columns that are materialised.

    ``update_admitted`` is an M2 gate outcome and a forbidden T1 transition
    input, so it is not in the default column set and asking for it is refused.
    """
    requested = tuple(columns)
    forbidden = sorted(set(requested) - set(T1_M2_COLUMNS_USED))
    if forbidden:
        raise T1EvidenceStoreError(
            f"M2-G columns {forbidden} are not T1 inputs. `update_admitted` in "
            "particular is an M2 gate outcome, which the protocol forbids the "
            "transition function."
        )
    with np.load(Path(path), allow_pickle=False) as payload:
        available = set(payload.files)
        missing = sorted(set(requested) - available)
        if missing:
            raise T1EvidenceStoreError(f"M2-G row evidence lacks columns {missing}.")
        return {column: np.asarray(payload[column]) for column in requested}


def read_t2_identity_members(
    path: Path, *, members: Sequence[str] = T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND
) -> dict[str, np.ndarray]:
    """Read named members of the T2 outer row identity, and only those.

    The archive also holds ``label``, ``target_family`` and ``primary_mask``.
    They stay closed here: a runtime transition that depended on evaluation
    annotation would not be deployable, because that annotation does not exist
    on a live stream. This is why the whole-archive readers cannot be used at
    this stage.
    """
    requested = tuple(members)
    for member in requested:
        require_label_blind_member(member)
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(requested) - set(payload.files))
        if missing:
            raise T1EvidenceStoreError(f"T2 row identity lacks members {missing}.")
        return {member: np.asarray(payload[member]) for member in requested}


def read_t2_selected_scores(path: Path) -> dict[str, np.ndarray]:
    """Read the retained arm's continuous score and its presence mask.

    ``predicted_positive`` is deliberately not read: it is the arm's binary
    decision at a T2 reporting threshold, and that threshold is T2 experiment
    evidence, never a T1 policy value.
    """
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted({"score", "score_present"} - set(payload.files))
        if missing:
            raise T1EvidenceStoreError(f"T2 score group lacks members {missing}.")
        return {
            "score": np.asarray(payload["score"]),
            "score_present": np.asarray(payload["score_present"]),
        }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def require_stable_id_alignment(
    m2_stable_ids: np.ndarray, t2_stable_ids: np.ndarray
) -> int:
    """Exact equality, in order. Not a set comparison, not a reindex."""
    if len(m2_stable_ids) != len(t2_stable_ids):
        raise T1EvidenceStoreError(
            f"M2-G has {len(m2_stable_ids)} rows and T2 has {len(t2_stable_ids)}. "
            "The two sources must describe the same timeline."
        )
    if not np.array_equal(np.asarray(m2_stable_ids), np.asarray(t2_stable_ids)):
        first = int(np.argmax(np.asarray(m2_stable_ids) != np.asarray(t2_stable_ids)))
        raise T1EvidenceStoreError(
            "M2-G and T2 stable ids diverge, first at position "
            f"{first}: {m2_stable_ids[first]!r} vs {t2_stable_ids[first]!r}. This is "
            "a hard stop; the timelines are not reconciled, re-indexed or repaired."
        )
    return len(m2_stable_ids)


def require_availability_alignment(
    m2_scored: np.ndarray, t2_present: np.ndarray
) -> dict[str, int]:
    """M2 `scored` and T2 `score_present` must agree row for row."""
    scored = np.asarray(m2_scored).astype(bool)
    present = np.asarray(t2_present).astype(bool)
    if scored.shape != present.shape:
        raise T1EvidenceStoreError("Availability masks have different shapes.")
    disagreements = int(np.count_nonzero(scored != present))
    if disagreements:
        raise T1EvidenceStoreError(
            f"M2 `scored` and T2 `score_present` disagree on {disagreements} rows. "
            "A row is either evidence in both sources or in neither."
        )
    available = int(np.count_nonzero(scored))
    unavailable = int(scored.size - available)
    return {
        "row_count": int(scored.size),
        "scored": available,
        "unavailable": unavailable,
    }


def require_expected_census(census: dict[str, int]) -> dict[str, int]:
    """The frozen row census, which closes exactly or stops the run."""
    expected = {
        "row_count": T1_TIMELINE_ROW_COUNT,
        "scored": T1_EXPECTED_SCORED_ROWS,
        "unavailable": T1_EXPECTED_UNAVAILABLE_ROWS,
    }
    if census != expected:
        raise T1EvidenceStoreError(
            f"Row census {census} is not the frozen {expected}. The timeline is not "
            "the one the protocol was frozen against."
        )
    return census


def forbidden_members() -> tuple[str, ...]:
    """Named for tests and for the lineage record; never read."""
    return tuple(T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND)


def forbidden_evidence_columns() -> tuple[str, ...]:
    return tuple(T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS)
