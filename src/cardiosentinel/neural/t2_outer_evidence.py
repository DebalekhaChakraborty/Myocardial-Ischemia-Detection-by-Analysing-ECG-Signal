"""Per-row outer-VALIDATION T2 evidence: typed arrays with a hashed manifest.

Following `m2_evidence_store` and `u1_evidence_store` exactly, and for the same
reason: 492,904 timeline rows of temporal evidence for each of two arms belong
in a typed binary store with a digest, not inflated into a JSON document. The
claim-bearing outer result binds this store by SHA-256 rather than restating it.

**Why this store has to exist at all.** T1 consumes the SELECTED T2 arm's
per-row temporal evidence. If the one-shot outer VALIDATION left only aggregate
metrics behind, T1 could only get those rows by re-running outer VALIDATION --
and there is exactly one outer attempt, ever. Persisting row-aligned evidence
here is what makes T1 possible without a second exposure of VALIDATION.

**Both arms are persisted, side by side.** The selection is recorded, not
applied: storing only the winner would make the decision unauditable after the
fact, exactly as it would have for U1's two calibrator families.

**What the score is, stated exactly.** It is
`sigmoid(current-window T2 logit)` -- an **uncalibrated temporal model score**.
It is not a calibrated probability, not a confidence and not an uncertainty.
The only retained calibrated probability in this programme comes from U1, and
`T2_CALIBRATION_OF_T2_AUTHORISED` is False.

**Unavailable rows carry no score.** A physically unavailable exact-flat
position advances the timeline and leaves the model state untouched, but the
model never sees it, so no scientific score exists for it. That is encoded by
an explicit `score_present` mask. NaN appears in the score column only as a
storage sentinel beside `score_present=False`, and
`require_scores_present` refuses to hand back a masked position as if it were a
model output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.t2_protocol import T2_ARMS, require_arm

T2_OUTER_EVIDENCE_SCHEMA: Final = "t2_v1_outer_validation_evidence_store/1"
T2_OUTER_STORE_MANIFEST_NAME: Final = "T2_OUTER_ROW_EVIDENCE.json"
T2_OUTER_IDENTITY_NAME: Final = "t2_outer_row_identity.npz"

T2_SCORE_SEMANTICS: Final = "uncalibrated_temporal_model_score"
T2_SCORE_DEFINITION: Final = "sigmoid(current_window_t2_logit)"

# Row identity: enough to bind every score to its physical position in the
# frozen stream, and nothing that is a model input.
T2_OUTER_IDENTITY_COLUMNS: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "subject_id",
    "target_family",
    "cold_start_bin",
    "observation_state",
    "score_present",
    "primary_mask",
    "label",
)

T2_OUTER_SCORE_COLUMNS: Final = ("score", "score_present", "predicted_positive")


def outer_score_file(arm: str) -> str:
    require_arm(arm)
    short = "gru" if arm.startswith("causal_gru") else "s4d"
    return f"t2_outer_scores_{short}.npz"


class T2OuterEvidenceError(RuntimeError):
    """Raised when per-row outer T2 evidence cannot be persisted or read."""


@dataclass(frozen=True, slots=True)
class T2OuterRowGroup:
    """One row group as row-aligned typed arrays."""

    name: str
    columns: tuple[str, ...]
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        missing = [column for column in self.columns if column not in self.arrays]
        extra = [column for column in self.arrays if column not in self.columns]
        if missing or extra:
            raise T2OuterEvidenceError(
                f"{self.name} row group schema mismatch; missing {missing}, "
                f"unexpected {extra}."
            )
        lengths = {int(values.shape[0]) for values in self.arrays.values()}
        if len(lengths) != 1:
            raise T2OuterEvidenceError(f"{self.name} columns are not row-aligned.")

    @property
    def row_count(self) -> int:
        return int(next(iter(self.arrays.values())).shape[0])


_IDENTITY_DTYPES: Final = {
    "stable_id": np.str_,
    "record_id": np.str_,
    "subject_id": np.str_,
    "target_family": np.str_,
    "cold_start_bin": np.str_,
    "channel_index": np.int64,
    "start_sample": np.int64,
    "observation_state": np.uint8,
    "label": np.int64,
    "score_present": np.bool_,
    "primary_mask": np.bool_,
}
_SCORE_DTYPES: Final = {
    "score": np.float64,
    "score_present": np.bool_,
    "predicted_positive": np.bool_,
}


def _as_group(
    name: str,
    columns: Sequence[str],
    data: dict[str, Any],
    dtypes: dict[str, Any],
) -> T2OuterRowGroup:
    arrays = {
        column: np.asarray(data[column], dtype=dtypes[column]) for column in columns
    }
    return T2OuterRowGroup(name=name, columns=tuple(columns), arrays=arrays)


def _require_score_masking(
    scores: np.ndarray, present: np.ndarray, *, label: str
) -> None:
    """A masked position may hold only the NaN sentinel; a present one may not.

    Both halves matter. A finite value hiding behind `score_present=False`
    would be a real model output that downstream code was told to ignore; a NaN
    with `score_present=True` would be a non-output that downstream code was
    told to treat as a score.
    """
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(present, dtype=bool)
    if values.shape != mask.shape:
        raise T2OuterEvidenceError(f"{label}: scores and mask are not row-aligned.")
    if int(np.count_nonzero(~np.isnan(values[~mask]))):
        raise T2OuterEvidenceError(
            f"{label}: a row with score_present=false carries a finite value. An "
            "unavailable observation produces no score and none is invented."
        )
    if int(np.count_nonzero(~np.isfinite(values[mask]))):
        raise T2OuterEvidenceError(
            f"{label}: a row with score_present=true carries a non-finite value. "
            "NaN is a storage sentinel for absence and is never a model score."
        )


def write_t2_outer_evidence_store(
    root: Path,
    *,
    identity: dict[str, Any],
    arm_scores: dict[str, dict[str, Any]],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Persist row identity plus both arms' scores, bound by a self-digest."""
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)

    identity_group = _as_group(
        "row_identity", T2_OUTER_IDENTITY_COLUMNS, identity, _IDENTITY_DTYPES
    )
    identifiers = identity_group.arrays["stable_id"]
    if len(set(identifiers.tolist())) != identity_group.row_count:
        raise T2OuterEvidenceError(
            "The outer evidence store carries duplicate stable IDs; the frozen "
            "ordering would be ambiguous and rows are never deduplicated."
        )
    missing_arms = [arm for arm in T2_ARMS if arm not in arm_scores]
    if missing_arms:
        raise T2OuterEvidenceError(
            f"The outer evidence store must carry both frozen arms; missing "
            f"{missing_arms}. A single-arm store cannot support the selection "
            "it is meant to make auditable."
        )

    row_groups: dict[str, Any] = {}
    identity_path = directory / T2_OUTER_IDENTITY_NAME
    if identity_path.exists():
        raise T2OuterEvidenceError(
            f"{identity_path} already exists; outer evidence is never overwritten."
        )
    with identity_path.open("wb") as handle:
        np.savez(handle, **identity_group.arrays)
    row_groups["row_identity"] = _group_entry(
        identity_group, T2_OUTER_IDENTITY_NAME, identity_path
    )

    present = np.asarray(identity_group.arrays["score_present"], dtype=bool)
    for arm in T2_ARMS:
        group = _as_group(arm, T2_OUTER_SCORE_COLUMNS, arm_scores[arm], _SCORE_DTYPES)
        if group.row_count != identity_group.row_count:
            raise T2OuterEvidenceError(
                f"The {arm} score group carries {group.row_count} rows against "
                f"{identity_group.row_count} identity rows."
            )
        if not np.array_equal(
            np.asarray(group.arrays["score_present"], dtype=bool), present
        ):
            raise T2OuterEvidenceError(
                f"The {arm} score-present mask disagrees with the row identity's. "
                "Availability is a property of the physical observation, not of "
                "the arm that scored it."
            )
        _require_score_masking(
            group.arrays["score"], group.arrays["score_present"], label=arm
        )
        filename = outer_score_file(arm)
        path = directory / filename
        if path.exists():
            raise T2OuterEvidenceError(
                f"{path} already exists; outer evidence is never overwritten."
            )
        with path.open("wb") as handle:
            np.savez(handle, **group.arrays)
        row_groups[arm] = _group_entry(group, filename, path)

    available = int(np.count_nonzero(present))
    # `primary_mask` is the PRIMARY TARGET population, defined by the label
    # authority alone. The scored PRIMARY population is its intersection with
    # `score_present`, and the difference is stated rather than absorbed.
    primary = np.asarray(identity_group.arrays["primary_mask"], dtype=bool)
    manifest: dict[str, Any] = {
        "schema": T2_OUTER_EVIDENCE_SCHEMA,
        "row_groups": row_groups,
        "arms": list(T2_ARMS),
        "arms_persisted": list(T2_ARMS),
        "row_count": identity_group.row_count,
        "scored_available_row_count": available,
        "unavailable_no_score_row_count": identity_group.row_count - available,
        "primary_target_row_count": int(primary.sum()),
        "primary_scored_available_row_count": int((primary & present).sum()),
        "primary_unavailable_no_score_count": int((primary & ~present).sum()),
        "score_semantics": T2_SCORE_SEMANTICS,
        "score_definition": T2_SCORE_DEFINITION,
        "score_is_calibrated_probability": False,
        "score_is_confidence": False,
        "score_is_uncertainty": False,
        "score_dtype": "float64",
        "nan_is_storage_sentinel_for_absence": True,
        "nan_is_ever_a_model_score": False,
        "full_timeline_ordering": "stream_then_start_sample",
        "lossy_conversion_applied": False,
        "test_rows_present": False,
        **{key: value for key, value in lineage.items()},
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    (directory / T2_OUTER_STORE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _group_entry(group: T2OuterRowGroup, filename: str, path: Path) -> dict[str, Any]:
    return {
        "file": filename,
        "row_count": group.row_count,
        "columns": list(group.columns),
        "dtypes": {
            name: str(values.dtype) for name, values in sorted(group.arrays.items())
        },
        "sha256": sha256_file(path),
    }


REQUIRED_LINEAGE_FIELDS: Final = (
    "validation_stream_cache_sha256",
    "validation_representation_content_sha256",
    "ordered_stable_id_sha256",
    "ordered_chronology_sha256",
    "target_authority_identity",
    "checkpoint_sha256",
    "checkpoint_lock_sha256",
    "internal_dev_thresholds",
)


def validate_t2_outer_evidence_store(
    manifest: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Re-verify the manifest self-digest, its schema, its lineage and its bytes."""
    recorded = manifest.get("content_sha256")
    body = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise T2OuterEvidenceError(
            "The T2 outer evidence manifest failed digest validation."
        )
    if manifest.get("schema") != T2_OUTER_EVIDENCE_SCHEMA:
        raise T2OuterEvidenceError(
            f"Unknown T2 outer evidence schema {manifest.get('schema')!r}."
        )
    if list(manifest.get("arms_persisted") or []) != list(T2_ARMS):
        raise T2OuterEvidenceError(
            "The outer evidence store must persist both frozen arms."
        )
    missing = [name for name in REQUIRED_LINEAGE_FIELDS if name not in manifest]
    if missing:
        raise T2OuterEvidenceError(f"The outer evidence store is missing {missing}.")
    for flag in (
        "score_is_calibrated_probability",
        "score_is_confidence",
        "score_is_uncertainty",
        "nan_is_ever_a_model_score",
        "lossy_conversion_applied",
        "test_rows_present",
    ):
        if manifest.get(flag) is not False:
            raise T2OuterEvidenceError(
                f"The T2 outer evidence store must record {flag}=false."
            )
    if manifest.get("score_semantics") != T2_SCORE_SEMANTICS:
        raise T2OuterEvidenceError(
            "The persisted score must be named an uncalibrated temporal model "
            "score; it is not a calibrated probability, a confidence or an "
            "uncertainty."
        )
    require_outer_row_accounting(manifest)

    groups = dict(manifest["row_groups"])
    expected = {"row_identity", *T2_ARMS}
    if set(groups) != expected:
        raise T2OuterEvidenceError(
            f"The outer evidence store binds {sorted(groups)}, expected "
            f"{sorted(expected)}."
        )
    if root is not None:
        base = Path(root)
        for key, entry in groups.items():
            path = base / entry["file"]
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise T2OuterEvidenceError(
                    f"Persisted outer row group {key} does not match its digest."
                )
    return dict(manifest)


def require_outer_row_accounting(manifest: dict[str, Any]) -> dict[str, int]:
    """Rows are accounted for exactly. Nothing is silently dropped.

    The frozen target population and the population that actually carries a
    score are not the same thing whenever a physical observation is missing, so
    the store states both and their difference, and refuses a set that does not
    add up.
    """
    total = int(manifest["row_count"])
    scored = int(manifest["scored_available_row_count"])
    unscored = int(manifest["unavailable_no_score_row_count"])
    if scored + unscored != total:
        raise T2OuterEvidenceError(
            f"Full-timeline accounting does not close: {scored} scored + "
            f"{unscored} unavailable != {total} rows."
        )
    primary_total = int(manifest["primary_target_row_count"])
    primary_scored = int(manifest["primary_scored_available_row_count"])
    primary_unscored = int(manifest["primary_unavailable_no_score_count"])
    if primary_scored + primary_unscored != primary_total:
        raise T2OuterEvidenceError(
            f"PRIMARY accounting does not close: {primary_scored} scored + "
            f"{primary_unscored} unavailable != {primary_total} target rows."
        )
    return {
        "row_count": total,
        "scored_available_row_count": scored,
        "unavailable_no_score_row_count": unscored,
        "primary_target_row_count": primary_total,
        "primary_scored_available_row_count": primary_scored,
        "primary_unavailable_no_score_count": primary_unscored,
    }


def read_t2_outer_row_group(
    root: Path, manifest: dict[str, Any], group: str
) -> T2OuterRowGroup:
    """Read one row group back at full precision, digest-checked first."""
    validate_t2_outer_evidence_store(manifest, root=Path(root))
    if group not in manifest["row_groups"]:
        raise T2OuterEvidenceError(f"No outer row group {group!r} in the store.")
    entry = manifest["row_groups"][group]
    columns = tuple(entry["columns"])
    with np.load(Path(root) / entry["file"], allow_pickle=False) as payload:
        arrays = {column: np.asarray(payload[column]) for column in columns}
    return T2OuterRowGroup(name=group, columns=columns, arrays=arrays)


def selected_arm_scores(
    root: Path, manifest: dict[str, Any], arm: str
) -> tuple[T2OuterRowGroup, T2OuterRowGroup]:
    """The identity group and one arm's score group, for downstream T1.

    This is the whole point of the store: T1 can take the arm outer VALIDATION
    selected and read its per-row temporal evidence, bound to `stable_id`,
    without re-running the one-shot outer attempt.
    """
    require_arm(arm)
    identity = read_t2_outer_row_group(root, manifest, "row_identity")
    scores = read_t2_outer_row_group(root, manifest, arm)
    if scores.row_count != identity.row_count:
        raise T2OuterEvidenceError(
            f"The {arm} score group and the row identity disagree on row count."
        )
    return identity, scores


def row_index_by_stable_id(identity: T2OuterRowGroup) -> dict[str, int]:
    """Position of each row by its frozen stable id, for a bound lookup."""
    return {
        str(value): position
        for position, value in enumerate(identity.arrays["stable_id"].tolist())
    }


def require_scores_present(scores: T2OuterRowGroup, positions: Sequence[int]) -> None:
    """Refuse to hand back a masked position as if it were a model output."""
    present = np.asarray(scores.arrays["score_present"], dtype=bool)
    index = np.asarray(list(positions), dtype=np.int64)
    if index.size and int(np.count_nonzero(~present[index])):
        raise T2OuterEvidenceError(
            "A requested row carries no T2 score: the observation was physically "
            "unavailable. No score is invented for it and no denominator is "
            "adjusted silently."
        )
