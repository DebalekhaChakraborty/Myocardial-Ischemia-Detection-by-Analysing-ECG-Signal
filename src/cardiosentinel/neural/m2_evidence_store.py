"""Bounded-memory M2 evidence accumulation, disk-backed where it must be.

The M1 host-memory incident established that retaining corpus-scale Python row
objects is unsafe. The danger is not numeric data -- 473,897 float64 scores are
under 4 MB -- it is the forest of `M2RowEvidence` dataclasses, each with a
nested `M2GateDecision` and its per-feature dict, held for two arms at once.

So the canonical route never accumulates rows as objects. Replay proceeds one
`(record_id, channel_index)` stream at a time; each stream's bounded batch of
evidence objects is folded into

* `M2AdmissionAccumulator` -- integer counters that reproduce
  `summarize_admission` exactly, and
* `M2ScoreTable` -- compact typed arrays of identity and score,

and is then released. The one genuinely large artifact, the full causal
prototype trajectory (473,897 x 146 float64 per arm), is written to disk per
stream and read back only for the stream being evaluated.

**Precision is preserved end to end.** Scores, times and prototypes are stored
as float64 and read back as float64, so
`sqrt(mean((mu_long(t) - mu_ref) ** 2))` is reproduced exactly. There is no
downcast, no quantization and no compression-with-loss anywhere on this path.

**Label-blind.** The store holds identities, scores, gate outcomes and
prototypes. It never holds a label, a subject identifier, a target family or an
annotation, and nothing here decides which prototypes to keep: the whole
trajectory is persisted first, and stress annotations select points from it only
afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m2_evidence import (
    CONDITION_DENOMINATORS,
    CONDITION_REPORT_NAMES,
    PrototypeTrajectory,
)
from cardiosentinel.neural.m2_policy import M2RowEvidence, require_m2_arm

EVIDENCE_STORE_SCHEMA: Final = "m2_v1_evidence_store/1"
TRAJECTORY_DIR: Final = "prototype_trajectories"
ROW_EVIDENCE_NAME: Final = "row_evidence.npz"
STORE_MANIFEST_NAME: Final = "M2_EVIDENCE_STORE.json"

ROW_EVIDENCE_COLUMNS: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "available_time",
    "score",
    "scored",
    "update_admitted",
)


class M2EvidenceStoreError(RuntimeError):
    """Raised when bounded evidence accumulation cannot proceed with integrity."""


# --------------------------------------------------------------------------
# Streaming admission accounting
# --------------------------------------------------------------------------


@dataclass(slots=True)
class M2AdmissionAccumulator:
    """Integer counters folded stream by stream.

    Reproduces `summarize_admission` exactly, including its tri-state
    applicability semantics: a condition that was not applicable is excluded
    from both numerator and denominator, never counted as a failure.
    """

    arm: str | None = None
    rows: int = 0
    available_rows: int = 0
    scored_rows: int = 0
    admitted_rows: int = 0
    defined_time_since: int = 0
    undefined_time_since: int = 0
    evaluated: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(CONDITION_REPORT_NAMES, 0)
    )
    failed: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(CONDITION_REPORT_NAMES, 0)
    )

    def add_stream(self, evidence: Sequence[M2RowEvidence]) -> None:
        """Fold one bounded stream, then let its objects go."""
        for row in evidence:
            arm = require_m2_arm(row.arm)
            if self.arm is None:
                self.arm = arm
            elif arm != self.arm:
                raise M2EvidenceStoreError(
                    f"Admission evidence mixes arms {sorted({self.arm, arm})}."
                )
            self.rows += 1
            if row.decision.g1_available:
                self.available_rows += 1
            if row.decision.score is not None:
                self.scored_rows += 1
            if row.update_admitted:
                self.admitted_rows += 1
            if row.time_since_last_admitted_update is None:
                self.undefined_time_since += 1
            else:
                self.defined_time_since += 1
            results = row.decision.condition_results()
            for key, condition in CONDITION_REPORT_NAMES.items():
                outcome = results[condition]
                if outcome is None:
                    continue
                self.evaluated[key] += 1
                if outcome is False:
                    self.failed[key] += 1

    def summary(self) -> dict[str, Any]:
        """The `summarize_admission` payload, assembled from the counters."""
        if self.rows == 0 or self.arm is None:
            raise M2EvidenceStoreError(
                "Admission evidence is empty; nothing to summarize."
            )
        refusals = {
            key: {
                "condition": condition,
                "applicable": self.evaluated[key] > 0,
                "failed_count": self.failed[key],
                "evaluated_count": self.evaluated[key],
                "not_applicable_count": self.rows - self.evaluated[key],
                "fraction": (
                    (self.failed[key] / self.evaluated[key])
                    if self.evaluated[key]
                    else None
                ),
                "denominator": CONDITION_DENOMINATORS[key],
            }
            for key, condition in CONDITION_REPORT_NAMES.items()
        }
        admitted_fraction = self.admitted_rows / self.rows
        return {
            "arm": self.arm,
            "rows": self.rows,
            "available_rows": self.available_rows,
            "scored_rows": self.scored_rows,
            "update_admitted_count": self.admitted_rows,
            "update_admission_fraction": admitted_fraction,
            "update_admission_denominator": "all_timeline_rows",
            "freeze_fraction": 1.0 - admitted_fraction,
            "freeze_denominator": "all_timeline_rows",
            "refusals": refusals,
            "refusal_semantics": (
                "each entry counts rows where that condition was applicable, "
                "evaluated and failed, over the rows where it was applicable "
                "and evaluated; a row failing several applicable conditions is "
                "counted in each, so these overlap and do not sum to the freeze "
                "fraction. A condition that was not applicable -- because an "
                "upstream physical prerequisite gave it no input, or because it "
                "is not operative for this arm -- is excluded from both "
                "numerator and denominator rather than counted as a failure."
            ),
            "time_since_last_admitted_update": {
                "defined_rows": self.defined_time_since,
                "undefined_rows": self.undefined_time_since,
                "undefined_semantics": (
                    "no admitted update has occurred yet in this stream; the "
                    "quantity is undefined and is never imputed as zero"
                ),
            },
            "accumulated": "stream_by_stream",
            "corpus_scale_row_objects_retained": False,
        }


# --------------------------------------------------------------------------
# Compact identity-keyed scores
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2ScoreTable:
    """Identity and score as compact typed arrays, never as row objects."""

    arm: str
    stable_ids: np.ndarray
    scores: np.ndarray
    scored: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.stable_ids.shape[0],
            self.scores.shape[0],
            self.scored.shape[0],
        }
        if len(shapes) != 1:
            raise M2EvidenceStoreError("Score table columns are not row-aligned.")
        if len(set(self.stable_ids.tolist())) != self.stable_ids.shape[0]:
            raise M2EvidenceStoreError("The score table has duplicate stable IDs.")

    @property
    def row_count(self) -> int:
        return int(self.stable_ids.shape[0])

    def scores_for(self, wanted: Sequence[str]) -> np.ndarray:
        """Scores for an exact identity subset, in the order requested.

        Every refusal the row-object join made is preserved: an unknown
        identity, a duplicate request and an unscored row are all fatal. No
        prediction is invented and no denominator is altered automatically.
        """
        requested = np.asarray([str(value) for value in wanted], dtype=np.str_)
        if len(set(requested.tolist())) != requested.shape[0]:
            raise M2EvidenceStoreError(
                "The requested evaluation population has duplicate stable IDs."
            )
        order = np.argsort(self.stable_ids, kind="stable")
        sorted_ids = self.stable_ids[order]
        position = np.searchsorted(sorted_ids, requested)
        position = np.clip(position, 0, sorted_ids.shape[0] - 1)
        found = sorted_ids[position] == requested
        if not bool(np.all(found)):
            missing = requested[~found]
            raise M2EvidenceStoreError(
                f"{int(missing.shape[0])} requested rows have no replay "
                f"evidence, beginning {sorted(missing.tolist())[:3]}. A metric "
                "row with no causal replay history is never dropped silently."
            )
        indices = order[position]
        if not bool(np.all(self.scored[indices])):
            unscored = requested[~self.scored[indices]]
            raise M2EvidenceStoreError(
                f"Score-bearing M2 population contains {int(unscored.shape[0])} "
                f"rows with no prediction, beginning "
                f"{sorted(unscored.tolist())[:3]}. STOP FOR HUMAN REVIEW. The "
                "row is never dropped from a metric, no prediction is invented "
                "and no denominator is altered automatically."
            )
        return np.asarray(self.scores[indices], dtype=np.float64)


@dataclass(slots=True)
class _ScoreAccumulator:
    """Per-stream lists of small arrays, concatenated once at the end."""

    stable_ids: list[np.ndarray] = field(default_factory=list)
    record_ids: list[np.ndarray] = field(default_factory=list)
    channels: list[np.ndarray] = field(default_factory=list)
    starts: list[np.ndarray] = field(default_factory=list)
    times: list[np.ndarray] = field(default_factory=list)
    scores: list[np.ndarray] = field(default_factory=list)
    scored: list[np.ndarray] = field(default_factory=list)
    admitted: list[np.ndarray] = field(default_factory=list)


def _stream_slug(key: tuple[str, int]) -> str:
    record_id, channel = key
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(record_id))
    return f"{safe}__ch{int(channel)}"


# --------------------------------------------------------------------------
# The disk-backed store
# --------------------------------------------------------------------------


@dataclass(slots=True)
class M2EvidenceStore:
    """One arm's label-blind evidence, accumulated stream by stream."""

    root: Path
    arm: str
    _scores: _ScoreAccumulator = field(default_factory=_ScoreAccumulator)
    _stream_keys: list[tuple[str, int]] = field(default_factory=list)
    _admission: M2AdmissionAccumulator = field(default_factory=M2AdmissionAccumulator)
    _finalized: bool = False

    def __post_init__(self) -> None:
        self.arm = require_m2_arm(self.arm)
        self.root = Path(self.root)
        (self.root / TRAJECTORY_DIR).mkdir(parents=True, exist_ok=True)

    @property
    def admission(self) -> M2AdmissionAccumulator:
        return self._admission

    def add_stream(
        self,
        key: tuple[str, int],
        evidence: Sequence[M2RowEvidence],
        trajectory: Sequence[tuple[float, np.ndarray]] | None = None,
    ) -> None:
        """Fold one stream's bounded batch and persist its trajectory.

        The caller may release `evidence` and `trajectory` immediately after:
        nothing here retains a reference to either.
        """
        if self._finalized:
            raise M2EvidenceStoreError("The evidence store is already finalized.")
        if key in self._stream_keys:
            raise M2EvidenceStoreError(f"Stream {key} was already accumulated.")
        if not evidence:
            raise M2EvidenceStoreError(f"Stream {key} produced no evidence rows.")

        self._admission.add_stream(evidence)
        self._stream_keys.append(key)

        acc = self._scores
        acc.record_ids.append(
            np.asarray([row.record_id for row in evidence], dtype=np.str_)
        )
        acc.channels.append(
            np.asarray([int(row.channel_index) for row in evidence], dtype=np.int64)
        )
        acc.starts.append(
            np.asarray([int(row.start_sample) for row in evidence], dtype=np.int64)
        )
        acc.times.append(
            np.asarray(
                [float(row.available_time) for row in evidence], dtype=np.float64
            )
        )
        acc.scores.append(
            np.asarray(
                [
                    np.nan if row.decision.score is None else float(row.decision.score)
                    for row in evidence
                ],
                dtype=np.float64,
            )
        )
        acc.scored.append(
            np.asarray(
                [row.decision.score is not None for row in evidence], dtype=np.bool_
            )
        )
        acc.admitted.append(
            np.asarray([bool(row.update_admitted) for row in evidence], dtype=np.bool_)
        )
        from cardiosentinel.neural.m2_evaluation import stable_id_for_key

        acc.stable_ids.append(
            np.asarray(
                [
                    stable_id_for_key(
                        (row.record_id, int(row.channel_index), int(row.start_sample))
                    )
                    for row in evidence
                ],
                dtype=np.str_,
            )
        )

        if trajectory is not None:
            self._write_trajectory(key, trajectory)

    def _write_trajectory(
        self, key: tuple[str, int], trajectory: Sequence[tuple[float, np.ndarray]]
    ) -> None:
        if not trajectory:
            raise M2EvidenceStoreError(
                f"Stream {key} produced an empty prototype trajectory; the "
                "causal history is never persisted as absent."
            )
        times = np.asarray([float(item[0]) for item in trajectory], dtype=np.float64)
        prototypes = np.stack(
            [np.asarray(item[1], dtype=np.float64) for item in trajectory]
        )
        path = self.root / TRAJECTORY_DIR / f"{_stream_slug(key)}.npz"
        # Uncompressed: lossless either way, and this path is read back per
        # stream during drift evaluation.
        with path.open("wb") as handle:
            np.savez(handle, times=times, prototypes=prototypes)

    def load_trajectory(self, key: tuple[str, int]) -> PrototypeTrajectory:
        """Read ONE stream's trajectory back at full float64 precision."""
        path = self.root / TRAJECTORY_DIR / f"{_stream_slug(key)}.npz"
        if not path.is_file():
            raise M2EvidenceStoreError(
                f"No persisted prototype trajectory for stream {key}."
            )
        with np.load(path, allow_pickle=False) as payload:
            times = np.asarray(payload["times"], dtype=np.float64)
            prototypes = np.asarray(payload["prototypes"], dtype=np.float64)
        return PrototypeTrajectory(times=times, prototypes=prototypes)

    def score_table(self) -> M2ScoreTable:
        """The compact identity/score table for the whole replayed population."""
        if not self._scores.stable_ids:
            raise M2EvidenceStoreError("No stream evidence was accumulated.")
        return M2ScoreTable(
            arm=self.arm,
            stable_ids=np.concatenate(self._scores.stable_ids),
            scores=np.concatenate(self._scores.scores),
            scored=np.concatenate(self._scores.scored),
        )

    def finalize(self) -> dict[str, Any]:
        """Persist the row arrays and bind the store's schema and content digest."""
        if self._finalized:
            raise M2EvidenceStoreError("The evidence store is already finalized.")
        acc = self._scores
        if not acc.stable_ids:
            raise M2EvidenceStoreError("No stream evidence was accumulated.")
        arrays = {
            "stable_id": np.concatenate(acc.stable_ids),
            "record_id": np.concatenate(acc.record_ids),
            "channel_index": np.concatenate(acc.channels),
            "start_sample": np.concatenate(acc.starts),
            "available_time": np.concatenate(acc.times),
            "score": np.concatenate(acc.scores),
            "scored": np.concatenate(acc.scored),
            "update_admitted": np.concatenate(acc.admitted),
        }
        if tuple(sorted(arrays)) != tuple(sorted(ROW_EVIDENCE_COLUMNS)):
            raise M2EvidenceStoreError("The row-evidence schema is incomplete.")
        rows_path = self.root / ROW_EVIDENCE_NAME
        with rows_path.open("wb") as handle:
            np.savez(handle, **arrays)

        trajectory_digests = {
            _stream_slug(key): sha256_file(
                self.root / TRAJECTORY_DIR / f"{_stream_slug(key)}.npz"
            )
            for key in self._stream_keys
            if (self.root / TRAJECTORY_DIR / f"{_stream_slug(key)}.npz").is_file()
        }
        manifest: dict[str, Any] = {
            "schema": EVIDENCE_STORE_SCHEMA,
            "arm": self.arm,
            "row_count": int(arrays["stable_id"].shape[0]),
            "stream_count": len(self._stream_keys),
            "ordered_stream_keys": [
                [str(record), int(channel)] for record, channel in self._stream_keys
            ],
            "row_evidence_columns": list(ROW_EVIDENCE_COLUMNS),
            "row_evidence_dtypes": {
                name: str(values.dtype) for name, values in sorted(arrays.items())
            },
            "row_evidence_sha256": sha256_file(rows_path),
            "prototype_trajectory_count": len(trajectory_digests),
            "prototype_trajectory_sha256": dict(sorted(trajectory_digests.items())),
            "score_dtype": "float64",
            "prototype_dtype": "float64",
            "lossy_conversion_applied": False,
            "labels_present": False,
            "subject_identifiers_present": False,
            "annotations_present": False,
            "trajectory_points_selected_by_annotation": False,
            "corpus_scale_row_objects_retained": False,
        }
        manifest["content_sha256"] = canonical_sha256(manifest)
        (self.root / STORE_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        self._finalized = True
        return manifest


def validate_evidence_store_manifest(
    manifest: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Re-verify a store's schema, its self-digest and its file digests."""
    recorded = manifest.get("content_sha256")
    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if recorded is None or recorded != canonical_sha256(body):
        raise M2EvidenceStoreError(
            "The evidence store manifest failed digest validation."
        )
    if manifest.get("schema") != EVIDENCE_STORE_SCHEMA:
        raise M2EvidenceStoreError(
            f"Unknown evidence store schema {manifest.get('schema')!r}."
        )
    for flag in (
        "lossy_conversion_applied",
        "labels_present",
        "subject_identifiers_present",
        "annotations_present",
        "trajectory_points_selected_by_annotation",
        "corpus_scale_row_objects_retained",
    ):
        if manifest.get(flag) is not False:
            raise M2EvidenceStoreError(f"The evidence store must record {flag}=false.")
    if manifest.get("score_dtype") != "float64":
        raise M2EvidenceStoreError("Scores must be stored at full float64 precision.")
    if manifest.get("prototype_dtype") != "float64":
        raise M2EvidenceStoreError(
            "Prototypes must be stored at full float64 precision so the frozen "
            "drift statistic is reproduced exactly."
        )
    if root is not None:
        base = Path(root)
        if sha256_file(base / ROW_EVIDENCE_NAME) != manifest["row_evidence_sha256"]:
            raise M2EvidenceStoreError("The persisted row evidence does not match.")
        for slug, digest in manifest["prototype_trajectory_sha256"].items():
            path = base / TRAJECTORY_DIR / f"{slug}.npz"
            if not path.is_file() or sha256_file(path) != digest:
                raise M2EvidenceStoreError(
                    f"Persisted prototype trajectory {slug} does not match its digest."
                )
    return dict(manifest)


def stream_keys_from(manifest: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """The ordered `(record_id, channel_index)` keys a store accumulated."""
    return tuple(
        (str(record), int(channel))
        for record, channel in manifest.get("ordered_stream_keys", ())
    )


def concatenated_stable_ids(root: Path) -> Iterable[str]:
    """Stream the store's stable IDs back without materialising row objects."""
    with np.load(Path(root) / ROW_EVIDENCE_NAME, allow_pickle=False) as payload:
        for value in payload["stable_id"]:
            yield str(value)
