"""Future outer-VALIDATION evaluator and descriptive metrics -- execution refused.

Every entry point that would touch VALIDATION calls
`require_outer_validation_authorized()` **first**, before path resolution,
before the representation memmap and before any label read. The activation
constant lives in `t2_persistence`, is `False`, and has no setter, flag or
environment variable. This module exists so the semantics can be reviewed before
scientific exposure, not so they can be run.

The metric functions below are pure and are exercised synthetically. They
compute nothing about the real corpus in this change set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np

from cardiosentinel.baseline.metrics import (
    binary_metrics,
    subject_bootstrap_confidence_intervals,
    subject_macro_metrics,
)
from cardiosentinel.neural.t2_models import seed_everything
from cardiosentinel.neural.t2_persistence import (
    ARM_SELECTION_PENDING,
    require_outer_validation_authorized,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARMS,
    T2_BOOTSTRAP_CLAIM,
    T2_BOOTSTRAP_REPLICATES,
    T2_BOOTSTRAP_SEED,
    T2_CHALLENGE_FAMILIES,
    T2_COLD_START_STRATA,
    T2_POOLED_METRICS,
    T2_SUBJECT_MACRO_METRICS,
    T2_VALIDATION_PRIMARY_ROW_COUNT,
    T2_WINDOW_STRIDE_SECONDS,
    select_t2_arm,
    validate_t2_protocol_document,
)

OUTER_VALIDATION_RESULT_CLASS: Final = "t2_v1_outer_validation_result"


class T2EvaluationError(RuntimeError):
    """Raised when T2 evaluation semantics are violated."""


# ---------------------------------------------------------------------------
# The disabled canonical evaluator
# ---------------------------------------------------------------------------


def execute_canonical_outer_validation(
    expected_git_sha: str | None = None,
    *,
    validation_root: Any = None,
    corpus_manifest: Any = None,
) -> dict[str, Any]:
    """The one-shot outer-VALIDATION route. Refuses while unauthorized.

    The activation gate is the first statement: no argument is inspected, no
    path is resolved and no VALIDATION array is opened before it fires. Only
    once the activation state is `True` does it reach the worker -- which then
    proves the authorized merged commit, a clean tree and an unconsumed outer
    attempt, and claims, all before any VALIDATION per-row artifact exists.

    `expected_git_sha` is not decorative and is not ignored after activation:
    it is the same human-authorization mechanism the TRAIN route uses, and a
    wrong one stops the attempt before the VALIDATION loader is invoked.
    """
    require_outer_validation_authorized()
    return _outer_validation_worker(  # pragma: no cover - gate is False
        expected_git_sha,
        validation_root=validation_root,
        corpus_manifest=corpus_manifest,
    )


def open_validation_timeline(*_args: Any, **kwargs: Any) -> Any:
    """Would open the VALIDATION timeline. Refuses before touching the store."""
    require_outer_validation_authorized()
    return _open_validation_timeline(  # pragma: no cover - gate is False
        kwargs.get("root")
    )


def load_validation_labels(*_args: Any, **kwargs: Any) -> Any:
    """Would read VALIDATION labels. Refuses first."""
    require_outer_validation_authorized()
    timeline = _open_validation_timeline(  # pragma: no cover - gate is False
        kwargs.get("root")
    )
    return _load_validation_targets(  # pragma: no cover - gate is False
        timeline, manifest_path=kwargs.get("corpus_manifest")
    )


OUTER_VALIDATION_ENTRY_POINTS: Final = (
    execute_canonical_outer_validation,
    open_validation_timeline,
    load_validation_labels,
)


# ---------------------------------------------------------------------------
# The execution body, written now so it can be reviewed before exposure
#
# Everything below the gate. A future activation change set flips
# `T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED` and updates the authorized merged
# SHA; it does not get to invent evaluation logic once the TRAIN numbers are
# known. These functions are private, are never called by the public entry
# points while the gate is False, and are exercised only against synthetic
# fixtures.
# ---------------------------------------------------------------------------


def _open_validation_timeline(root: Any = None) -> Any:
    """Open the VALIDATION timeline through the one byte-level verified route."""
    from cardiosentinel.neural.t2_timeline import T2Timeline

    return T2Timeline("validation", root=root)


def _load_validation_targets(timeline: Any, *, manifest_path: Any = None) -> Any:
    """Resolve VALIDATION rows to their persisted frozen target families."""
    from cardiosentinel.neural.t2_timeline import resolve_timeline_target_families

    return resolve_timeline_target_families(timeline, manifest_path=manifest_path)


def _outer_validation_preflight(
    expected_git_sha: str | None,
    *,
    run_root: Any = None,
    training_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Everything provable before the outer claim, in the frozen order.

    The order is the point. Activation has already refused at the public entry
    point; here the authorized merged commit is proved, the tree is proved
    clean, and the one outer attempt is proved unconsumed -- all of it small
    immutable identity material. No VALIDATION path is resolved, no VALIDATION
    array is opened and no VALIDATION label is read by any of it.
    """
    from cardiosentinel.neural import t2_persistence as persistence
    from cardiosentinel.neural.t2_development_run import require_expected_git_sha

    git_sha = require_expected_git_sha(expected_git_sha)
    protocol_sha = validate_t2_protocol_document()
    execution_spec_sha = persistence.validate_t2_execution_spec()
    root = persistence.T2_RUN_ROOT if run_root is None else Path(run_root)
    attempt = training_attempt_id or persistence.T2_TRAINING_ATTEMPT_ID
    unclaimed = persistence.require_unclaimed_outer_attempt(
        root, persistence.T2_OUTER_VALIDATION_ATTEMPT_ID
    )
    # The TRAIN attempt is verified from its bytes BEFORE the outer claim: an
    # outer attempt bound to a mutated training result would be consumed for
    # nothing.
    training = persistence.validate_canonical_t2_attempt(root, attempt)
    return {
        "preflight_class": "t2_outer_validation_preflight",
        "experiment_identity": persistence.T2_EXPERIMENT_IDENTITY,
        "attempt_id": persistence.T2_OUTER_VALIDATION_ATTEMPT_ID,
        "training_attempt_id": attempt,
        "authorized_git_sha": git_sha,
        "t2_protocol_sha256": protocol_sha,
        "t2_execution_spec_sha256": execution_spec_sha,
        "training_attempt_verification": training,
        "claim_state": unclaimed,
        "validation_accessed": False,
        "validation_path_resolved": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


@dataclass(slots=True)
class _OuterExposure:
    """What the outer attempt has actually touched, for an honest receipt."""

    stage: str = "claim_outer_attempt"
    validation_accessed: bool = False
    target_authority_opened: bool = False
    current_arm: str | None = None
    arms_scored: list[str] = field(default_factory=list)
    checkpoints_loaded: list[str] = field(default_factory=list)
    primary_outcomes_exposed: bool = False
    challenge_outcomes_exposed: bool = False
    arm_selection_exposed: bool = False
    row_evidence_promoted: bool = False

    def begin_arm(self, arm: str) -> None:
        self.current_arm = arm

    def complete_arm(self, arm: str) -> None:
        self.arms_scored.append(arm)
        self.current_arm = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "validation_accessed": self.validation_accessed,
            "target_authority_opened": self.target_authority_opened,
            "current_arm": self.current_arm,
            "arms_scored": list(self.arms_scored),
            "checkpoints_loaded": list(self.checkpoints_loaded),
            "primary_outcomes_exposed": self.primary_outcomes_exposed,
            "challenge_outcomes_exposed": self.challenge_outcomes_exposed,
            "arm_selection_exposed": self.arm_selection_exposed,
            "row_evidence_promoted": self.row_evidence_promoted,
            "test_accessed": False,
        }


def _outer_validation_worker(
    expected_git_sha: str | None = None,
    *,
    run_root: Any = None,
    training_attempt_id: str | None = None,
    validation_root: Any = None,
    corpus_manifest: Any = None,
) -> dict[str, Any]:
    """The one-shot outer-VALIDATION attempt, claim-bearing end to end.

    Order, and why each step sits where it does:

    1. preflight -- authorized Git SHA, clean tree, unconsumed outer attempt,
       and the TRAIN attempt verified from its bytes. None of it opens
       VALIDATION;
    2. runtime **START**, then claim `t2-v1-outer-validation`. The claim is the
       gate: only after it may a real VALIDATION per-row artifact be opened, so
       a corrupted input discovered afterwards consumes the attempt and is
       recorded honestly rather than found by an untraced pre-run scan;
    3. open the VALIDATION timeline through the same byte-level verified loader
       TRAIN used, and resolve its target authority the same way;
    4. per arm in frozen order: **pre-checkpoint-load** runtime observation,
       load the checkpoint by digest onto the canonical execution device, and
       run **one** complete causal pass over the whole timeline;
    5. every downstream quantity -- PRIMARY, challenge, cold-start, subject
       macro, bootstrap and the stream-aware descriptors -- is read out of that
       one pass. There is no second challenge replay and no second temporal
       replay;
    6. persist row-aligned per-row evidence for BOTH arms, so T1 can consume the
       selected arm without ever re-running this attempt;
    7. delegate selection to `t2_protocol.select_t2_arm`, promote, lock,
       observe **COMPLETION**.
    """
    from cardiosentinel.neural import t2_persistence as persistence
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        RuntimeIntegrityRecord,
    )

    checks = _outer_validation_preflight(
        expected_git_sha, run_root=run_root, training_attempt_id=training_attempt_id
    )
    root = persistence.T2_RUN_ROOT if run_root is None else Path(run_root)
    runtime = RuntimeIntegrityRecord()
    persistence.observe_t2_runtime_stage(
        runtime,
        point=EnforcementPoint.START,
        detail=persistence.STAGE_OUTER_START,
    )
    claimed = persistence.claim_t2_outer_directory(
        root, persistence.T2_OUTER_VALIDATION_ATTEMPT_ID, runtime=runtime
    )
    exposure = _OuterExposure()
    try:
        return _outer_after_claim(
            checks=checks,
            root=root,
            claimed=claimed,
            runtime=runtime,
            exposure=exposure,
            validation_root=validation_root,
            corpus_manifest=corpus_manifest,
        )
    except BaseException as error:
        # ANY post-claim failure consumes the one-shot outer attempt. Completed
        # evidence stays exactly where it is as forensic material; there is no
        # retry, no alternate attempt name and no selective arm rerun.
        persistence.record_t2_outer_attempt_failure(
            root,
            claimed,
            exception=error,
            stage=exposure.stage,
            arm=exposure.current_arm,
            exposure=exposure.as_dict(),
            runtime=runtime,
        )
        raise


def _outer_after_claim(
    *,
    checks: dict[str, Any],
    root: Any,
    claimed: Any,
    runtime: Any,
    exposure: _OuterExposure,
    validation_root: Any,
    corpus_manifest: Any,
) -> dict[str, Any]:
    """Everything the outer claim authorises, in the frozen order."""
    from cardiosentinel.neural import t2_persistence as persistence
    from cardiosentinel.neural.m1_store import COLD_START_BIN_FILE, START_SAMPLE_FILE
    from cardiosentinel.neural.runtime_sentinel import EnforcementPoint
    from cardiosentinel.neural.t2_outer_evidence import (
        T2_OUTER_STORE_MANIFEST_NAME,
        write_t2_outer_evidence_store,
    )
    from cardiosentinel.neural.t2_timeline import FAMILY_NAME
    from cardiosentinel.neural.t2_training import (
        T2TimelineReader,
        restore_model_state,
        score_streams,
    )

    training = checks["training_attempt_verification"]
    training_dir = persistence.t2_run_directory(root, checks["training_attempt_id"])
    training_lock = json.loads(
        (training_dir / persistence.EXPERIMENT_LOCK_NAME).read_text()
    )
    thresholds = dict(training_lock["internal_dev_thresholds"])

    execution_device = persistence.canonical_execution_device()
    seed_everything()
    persistence.require_deterministic_execution(execution_device)
    observed_runtime = persistence.runtime_provenance(execution_device)

    exposure.stage = "open_validation_timeline"
    timeline = _open_validation_timeline(validation_root)
    exposure.validation_accessed = True
    try:
        exposure.stage = "resolve_target_authority"
        family_codes, target_identity = _load_validation_targets(
            timeline, manifest_path=corpus_manifest
        )
        exposure.target_authority_opened = True
        reader = T2TimelineReader(timeline, family_codes)
        streams = timeline.streams()
        families = np.asarray(FAMILY_NAME, dtype="<U32")

        row_identity, common = _outer_row_identity(timeline, family_codes, families)
        per_arm: dict[str, Any] = {}
        arm_scores: dict[str, Any] = {}
        pooled_auprc_by_arm: dict[str, float] = {}
        macro_auprc_by_arm: dict[str, float] = {}
        bootstrap: dict[str, Any] = {}
        descriptors: dict[str, Any] = {}
        runtimes: list[dict[str, Any]] = []

        for arm in T2_ARMS:
            exposure.stage = f"score_arm:{arm}"
            exposure.begin_arm(arm)
            persistence.observe_t2_runtime_stage(
                runtime,
                point=EnforcementPoint.PRE_PROMOTION,
                detail=persistence.stage_pre_checkpoint_load(arm),
            )
            checkpoint_lock = persistence.read_checkpoint_lock(training_dir, arm)
            state = persistence.load_checkpoint(
                training_dir / persistence.CHECKPOINT_NAME[arm],
                expected_sha256=checkpoint_lock["checkpoint_sha256"],
            )
            exposure.checkpoints_loaded.append(arm)
            model = restore_model_state(arm, state, device=execution_device)
            arm_runtime = {
                **observed_runtime,
                "model_parameter_device": persistence.model_parameter_device(model),
            }
            persistence.require_execution_device_agreement(
                arm_runtime, arm_runtime["model_parameter_device"]
            )
            if runtimes:
                persistence.require_single_runtime(runtimes[0], arm_runtime)
            runtimes.append(arm_runtime)
            threshold = float(thresholds[arm]["threshold"])

            # ONE pass. Everything below is read out of it.
            scored = score_streams(model, reader, streams, device=execution_device)
            evidence = _outer_arm_evidence(
                timeline=timeline,
                streams=streams,
                scored=scored,
                common=common,
                threshold=threshold,
                families=families,
                family_codes=family_codes,
                cold_start=np.asarray(timeline.store.array(COLD_START_BIN_FILE)),
                start_samples=np.asarray(timeline.store.array(START_SAMPLE_FILE)),
            )
            exposure.primary_outcomes_exposed = True
            exposure.challenge_outcomes_exposed = True

            per_arm[arm] = {
                "architecture": arm,
                "checkpoint_sha256": checkpoint_lock["checkpoint_sha256"],
                "checkpoint_lock_sha256": checkpoint_lock["checkpoint_lock_sha256"],
                "internal_dev_threshold": threshold,
                "threshold_altered_by_outer_validation": False,
                "single_causal_pass": True,
                "same_pass_supplies_primary_and_challenge": True,
                "same_pass_supplies_temporal_descriptors": True,
                "second_challenge_replay_performed": False,
                "second_temporal_replay_performed": False,
                "runtime": arm_runtime,
                **evidence["aggregate"],
            }
            arm_scores[arm] = evidence["row_scores"]
            bootstrap[arm] = evidence["bootstrap"]
            descriptors[arm] = evidence["descriptors"]
            pooled_auprc_by_arm[arm] = float(evidence["aggregate"]["pooled"]["auprc"])
            macro_auprc_by_arm[arm] = float(
                evidence["aggregate"]["subject_macro"]["auprc"]["value"]
            )
            exposure.complete_arm(arm)

        exposure.stage = "promote_row_evidence"
        persistence.observe_t2_runtime_stage(
            runtime,
            point=EnforcementPoint.PRE_PROMOTION,
            detail=persistence.OUTER_EVIDENCE_DIRNAME,
        )
        evidence_root = claimed.run_dir / persistence.OUTER_EVIDENCE_DIRNAME
        store_manifest = write_t2_outer_evidence_store(
            evidence_root,
            identity=row_identity,
            arm_scores=arm_scores,
            lineage={
                "validation_stream_cache_sha256": (
                    timeline.manifest["stream_cache_sha256"]
                ),
                "validation_representation_content_sha256": (
                    timeline.manifest["representation_content_sha256"]
                ),
                "ordered_stable_id_sha256": (
                    timeline.manifest["ordered_stable_id_sha256"]
                ),
                "ordered_chronology_sha256": (
                    timeline.manifest["ordered_chronology_sha256"]
                ),
                "target_authority_identity": target_identity,
                "checkpoint_sha256": dict(training_lock["checkpoint_sha256"]),
                "checkpoint_lock_sha256": dict(training_lock["checkpoint_lock_sha256"]),
                "internal_dev_thresholds": thresholds,
            },
        )
        exposure.row_evidence_promoted = True
        claimed.promoted[
            f"{persistence.OUTER_EVIDENCE_DIRNAME}/{T2_OUTER_STORE_MANIFEST_NAME}"
        ] = _file_sha256(evidence_root / T2_OUTER_STORE_MANIFEST_NAME)

        exposure.stage = "select_arm"
        decision = select_from_outer_validation(
            pooled_auprc=pooled_auprc_by_arm,
            subject_macro_auprc=macro_auprc_by_arm,
            parameter_counts={
                arm: int(training_lock["trainable_parameters"][arm]) for arm in T2_ARMS
            },
        )
        exposure.arm_selection_exposed = True

        exposure.stage = "promote_outer_result"
        row_accounting = {
            **common["accounting"],
            "metrics_use_scored_rows_only": True,
            "score_invented_for_unavailable_row": False,
        }
        result = {
            "artifact_class": persistence.OUTER_RESULT_CLASS,
            "attempt_id": claimed.attempt_id,
            "experiment_identity": persistence.T2_EXPERIMENT_IDENTITY,
            "git_sha": checks["authorized_git_sha"],
            "git_dirty": False,
            "authorized_git_sha": checks["authorized_git_sha"],
            "t2_protocol_sha256": checks["t2_protocol_sha256"],
            "t2_execution_spec_sha256": checks["t2_execution_spec_sha256"],
            "training_attempt_id": checks["training_attempt_id"],
            "training_result_sha256": training["result_sha256"],
            "training_experiment_lock_sha256": training["experiment_lock_sha256"],
            "training_attempt_verification": training,
            "checkpoint_sha256": dict(training_lock["checkpoint_sha256"]),
            "checkpoint_lock_sha256": dict(training_lock["checkpoint_lock_sha256"]),
            "checkpoint_lock_self_sha256": dict(
                training_lock["checkpoint_lock_self_sha256"]
            ),
            "internal_dev_thresholds": thresholds,
            "validation_stream_cache_sha256": (
                timeline.manifest["stream_cache_sha256"]
            ),
            "validation_timeline_identity": timeline.identity(),
            "target_authority_identity": target_identity,
            "row_accounting": row_accounting,
            "primary_population_identity": {
                "row_count": int(target_identity["primary_row_count"]),
                "ischemic_positive": int(target_identity["ischemic_positive"]),
                "background_negative": int(target_identity["background_negative"]),
                "primary_target_row_count": row_accounting["primary_target_row_count"],
                "primary_scored_available_row_count": (
                    row_accounting["primary_scored_available_row_count"]
                ),
                "primary_unavailable_no_score_count": (
                    row_accounting["primary_unavailable_no_score_count"]
                ),
            },
            "challenge_population_identity": {
                "row_count": int(target_identity["challenge_row_count"]),
                "merged_into_primary": False,
                **dict(CHALLENGE_CAUSAL_SEMANTICS),
            },
            "unavailable_row_census": {
                "unavailable_no_score_row_count": (
                    row_accounting["unavailable_no_score_row_count"]
                ),
                "state_carried_across_unavailable": True,
                "score_present_for_unavailable": False,
            },
            "per_arm_evidence": per_arm,
            "row_evidence_store": {
                "directory": persistence.OUTER_EVIDENCE_DIRNAME,
                "manifest": T2_OUTER_STORE_MANIFEST_NAME,
                "content_sha256": store_manifest["content_sha256"],
                "arms_persisted": list(store_manifest["arms_persisted"]),
                "row_count": store_manifest["row_count"],
                "score_semantics": store_manifest["score_semantics"],
                "supports_t1_without_rerunning_outer_validation": True,
            },
            "subject_bootstrap": bootstrap,
            "temporal_descriptors": descriptors,
            "selection_decision": decision,
            "selected_arm": decision["selected_arm"],
            "runtime": runtimes[0],
            "execution_device": str(execution_device),
            "latency_used_in_selection": False,
            "challenge_used_in_selection": False,
            "attempts_permitted": 1,
            "automatic_retry_performed": False,
            "validation_accessed": True,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        }
        validate_outer_validation_result(result)
        promoted = persistence.finalize_and_promote_t2_outer_result(
            claimed,
            result=result,
            provenance={
                "authorized_git_sha": checks["authorized_git_sha"],
                "training_attempt_id": checks["training_attempt_id"],
                "training_result_sha256": training["result_sha256"],
                "training_experiment_lock_sha256": (training["experiment_lock_sha256"]),
                "checkpoint_sha256": dict(training_lock["checkpoint_sha256"]),
                "checkpoint_lock_sha256": dict(training_lock["checkpoint_lock_sha256"]),
                "checkpoint_lock_self_sha256": dict(
                    training_lock["checkpoint_lock_self_sha256"]
                ),
                "internal_dev_thresholds": thresholds,
                "validation_timeline_identity": timeline.identity(),
                "target_authority_identity": target_identity,
                "row_accounting": row_accounting,
                "row_evidence_store_sha256": store_manifest["content_sha256"],
                "selected_arm": decision["selected_arm"],
                "selection_decision": decision,
                "runtime": runtimes[0],
            },
            runtime=runtime,
        )
        return {
            "report_class": "t2_canonical_outer_validation_completion",
            "attempt_id": claimed.attempt_id,
            "run_dir": str(claimed.run_dir),
            "status": promoted["status"]["status"],
            "selected_arm": decision["selected_arm"],
            "row_accounting": row_accounting,
            "row_evidence_store_sha256": store_manifest["content_sha256"],
            "experiment_lock_sha256": promoted["lock"]["experiment_lock_sha256"],
            "result": result,
        }
    finally:
        timeline.close()


def _file_sha256(path: Any) -> str:
    from cardiosentinel.data.provenance import sha256_file

    return sha256_file(Path(path))


def _outer_row_identity(
    timeline: Any, family_codes: np.ndarray, families: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full-timeline row identity plus the population accounting it implies.

    Built once and shared by both arms: availability is a property of the
    physical observation, not of the arm that scored it.
    """
    from cardiosentinel.data.ltstdb import subject_id_for_record
    from cardiosentinel.neural.m1_store import (
        CHANNEL_INDEX_FILE,
        COLD_START_BIN_FILE,
        OBSERVATION_STATE_FILE,
        RECORD_ID_FILE,
        STABLE_ID_FILE,
        START_SAMPLE_FILE,
    )
    from cardiosentinel.neural.t2_protocol import T2_OBSERVATION_AVAILABLE
    from cardiosentinel.neural.t2_timeline import (
        ROLE_CODE_PRIMARY,
        primary_labels_for_families,
        role_codes_for_families,
    )

    states = np.asarray(timeline.store.array(OBSERVATION_STATE_FILE))
    records = np.asarray(timeline.store.array(RECORD_ID_FILE))
    present = states == T2_OBSERVATION_AVAILABLE
    # `primary_mask` is the PRIMARY **target** population: it is defined by the
    # frozen label authority alone and does not depend on whether the physical
    # observation survived. Conflating it with the role mask -- which demotes an
    # unavailable row to UNAVAILABLE -- is exactly the silent equation of target
    # and scored populations that the accounting exists to prevent. The scored
    # PRIMARY population is derived as `primary_mask & score_present`.
    primary = np.isin(
        family_codes,
        np.asarray(
            [
                index
                for index, name in enumerate(families.tolist())
                if name in {"ischemic_positive", "background_negative"}
            ],
            dtype=np.uint8,
        ),
    )
    # The role mask is still derived, because it is what proves the demotion
    # happens at all: an unavailable row is never PRIMARY_DIRECT_LOSS.
    roles = role_codes_for_families(family_codes, states)
    if int(np.count_nonzero((roles == ROLE_CODE_PRIMARY) & ~present)):
        raise T2EvaluationError(  # pragma: no cover - contradiction by construction
            "An unavailable row retained the PRIMARY direct-loss role."
        )
    identity = {
        "stable_id": np.asarray(timeline.store.array(STABLE_ID_FILE)),
        "record_id": records,
        "channel_index": np.asarray(timeline.store.array(CHANNEL_INDEX_FILE)),
        "start_sample": np.asarray(timeline.store.array(START_SAMPLE_FILE)),
        "subject_id": np.asarray(
            [subject_id_for_record(str(value)) for value in records], dtype=np.str_
        ),
        "target_family": families[family_codes],
        "cold_start_bin": np.asarray(timeline.store.array(COLD_START_BIN_FILE)),
        "observation_state": states,
        "score_present": present,
        "primary_mask": primary,
        "label": primary_labels_for_families(family_codes),
    }
    total = int(states.shape[0])
    scored = int(np.count_nonzero(present))
    accounting = {
        "row_count": total,
        "scored_available_row_count": scored,
        "unavailable_no_score_row_count": total - scored,
        "primary_target_row_count": int(primary.sum()),
        "primary_scored_available_row_count": int((primary & present).sum()),
        "primary_unavailable_no_score_count": int((primary & ~present).sum()),
    }
    _require_row_accounting_closes(accounting)
    common = {
        "score_present": present,
        "primary_mask": primary,
        "labels": identity["label"],
        "subjects": identity["subject_id"],
        "accounting": accounting,
    }
    return identity, common


def _outer_arm_evidence(
    *,
    timeline: Any,
    streams: Any,
    scored: Any,
    common: dict[str, Any],
    threshold: float,
    families: np.ndarray,
    family_codes: np.ndarray,
    cold_start: np.ndarray,
    start_samples: np.ndarray,
) -> dict[str, Any]:
    """Everything one arm's single causal pass yields, scattered back to rows.

    The pass returns compacted scored rows; they are scattered onto the full
    timeline here so the descriptors can see the real sequence -- including the
    non-PRIMARY predictions between PRIMARY ones and the gaps where no
    prediction exists.
    """
    present = np.asarray(common["score_present"], dtype=bool)
    primary = np.asarray(common["primary_mask"], dtype=bool)
    labels = np.asarray(common["labels"], dtype=np.int64)
    subjects = np.asarray(common["subjects"])
    total = int(present.shape[0])

    row_scores = np.full(total, np.nan, dtype=np.float64)
    positions = np.asarray(scored.positions, dtype=np.int64)
    row_scores[positions] = np.asarray(scored.scores, dtype=np.float64)
    if int(np.count_nonzero(np.isnan(row_scores[present]))):
        raise T2EvaluationError(
            "An available row received no score from the causal pass; the pass "
            "and the availability mask disagree."
        )
    if int(np.count_nonzero(~np.isnan(row_scores[~present]))):
        raise T2EvaluationError(
            "An unavailable row received a score. No score is invented for a "
            "physically missing observation."
        )
    predicted = np.zeros(total, dtype=bool)
    predicted[present] = row_scores[present] >= threshold

    scored_primary = primary & present
    challenge_codes = np.asarray(
        [
            index
            for index, name in enumerate(families.tolist())
            if name in set(T2_CHALLENGE_FAMILIES_RAW)
        ],
        dtype=np.uint8,
    )
    challenge = np.isin(family_codes, challenge_codes) & present

    aggregate = {
        "scored_row_count": int(np.count_nonzero(present)),
        "primary_scored_row_count": int(scored_primary.sum()),
        "challenge_scored_row_count": int(challenge.sum()),
        "unavailable_rows_scored": 0,
        "pooled": pooled_evidence(
            labels[scored_primary].tolist(),
            row_scores[scored_primary].tolist(),
            threshold,
        ),
        "subject_macro": subject_macro_evidence(
            subjects[scored_primary].tolist(),
            labels[scored_primary].tolist(),
            row_scores[scored_primary].tolist(),
            threshold,
        ),
        "cold_start": cold_start_strata_evidence(
            cold_start[scored_primary].tolist(),
            labels[scored_primary].tolist(),
            row_scores[scored_primary].tolist(),
            threshold,
        ),
        "challenge": challenge_family_evidence(
            _challenge_family_labels(families[family_codes][challenge]),
            labels[challenge].tolist(),
            row_scores[challenge].tolist(),
            threshold,
        ),
    }
    bootstrap = subject_bootstrap_evidence(
        subjects[scored_primary].tolist(),
        labels[scored_primary].tolist(),
        row_scores[scored_primary].tolist(),
        threshold,
    )
    descriptor_streams = [
        T2DescriptorStream(
            record_id=stream.record_id,
            channel_index=int(stream.channel_index),
            predictions=predicted[stream.start_index : stream.stop_index],
            score_present=present[stream.start_index : stream.stop_index],
            primary_mask=primary[stream.start_index : stream.stop_index],
            labels=labels[stream.start_index : stream.stop_index],
        )
        for stream in streams
    ]
    return {
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "descriptors": temporal_descriptors(descriptor_streams),
        "row_scores": {
            "score": row_scores,
            "score_present": present,
            "predicted_positive": predicted,
        },
    }


# The frozen corpus family names behind the three challenge reporting families.
T2_CHALLENGE_FAMILIES_RAW: Final = (
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
)
_CHALLENGE_REPORTING_NAME: Final = dict(
    zip(T2_CHALLENGE_FAMILIES_RAW, T2_CHALLENGE_FAMILIES, strict=True)
)


def _challenge_family_labels(raw_families: np.ndarray) -> list[str]:
    """Map persisted corpus family names onto the frozen reporting names."""
    return [_CHALLENGE_REPORTING_NAME[str(value)] for value in raw_families]


# ---------------------------------------------------------------------------
# The future result schema, validated synthetically
# ---------------------------------------------------------------------------

REQUIRED_OUTER_VALIDATION_FIELDS: Final = (
    "artifact_class",
    "training_experiment_lock_sha256",
    "checkpoint_sha256",
    "internal_dev_thresholds",
    "t2_protocol_sha256",
    "t2_execution_spec_sha256",
    "validation_stream_cache_sha256",
    "primary_population_identity",
    "challenge_population_identity",
    "per_arm_evidence",
    "subject_bootstrap",
    "temporal_descriptors",
    "selection_decision",
    "selected_arm",
    "test_accessed",
    "sealed_test_state",
)


def validate_outer_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate the future result's shape. No real values exist yet."""
    if result.get("artifact_class") != OUTER_VALIDATION_RESULT_CLASS:
        raise T2EvaluationError(
            f"Unknown outer-validation class {result.get('artifact_class')!r}."
        )
    missing = [name for name in REQUIRED_OUTER_VALIDATION_FIELDS if name not in result]
    if missing:
        raise T2EvaluationError(f"The outer-validation result is missing {missing}.")
    for arm in T2_ARMS:
        if arm not in result["per_arm_evidence"]:
            raise T2EvaluationError(f"No outer-validation evidence for {arm}.")
    if result.get("test_accessed") is not False:
        raise T2EvaluationError("An outer-validation result records TEST access.")
    if result.get("sealed_test_state") != "unopened":
        raise T2EvaluationError("The B4 sealed test must remain unopened.")
    primary = result["primary_population_identity"]
    if int(primary.get("row_count", -1)) != T2_VALIDATION_PRIMARY_ROW_COUNT:
        raise T2EvaluationError(
            f"The PRIMARY population must be {T2_VALIDATION_PRIMARY_ROW_COUNT} "
            f"rows; got {primary.get('row_count')}."
        )
    if "row_accounting" in result:
        _require_row_accounting_closes(result["row_accounting"])
        target = int(primary.get("primary_target_row_count", -1))
        if target != int(result["row_accounting"]["primary_target_row_count"]):
            raise T2EvaluationError(
                "The PRIMARY population identity and the row accounting disagree "
                "about the target population."
            )
        if target != T2_VALIDATION_PRIMARY_ROW_COUNT:
            raise T2EvaluationError(
                f"The PRIMARY target population must be "
                f"{T2_VALIDATION_PRIMARY_ROW_COUNT} rows; got {target}."
            )
    return result


def _require_row_accounting_closes(accounting: dict[str, Any]) -> None:
    """Scored plus unavailable equals the target. Nothing is silently dropped."""
    scored = int(accounting["scored_available_row_count"])
    unscored = int(accounting["unavailable_no_score_row_count"])
    if scored + unscored != int(accounting["row_count"]):
        raise T2EvaluationError(
            f"Full-timeline accounting does not close: {scored} + {unscored} != "
            f"{accounting['row_count']}."
        )
    primary_scored = int(accounting["primary_scored_available_row_count"])
    primary_unscored = int(accounting["primary_unavailable_no_score_count"])
    if primary_scored + primary_unscored != int(accounting["primary_target_row_count"]):
        raise T2EvaluationError(
            f"PRIMARY accounting does not close: {primary_scored} + "
            f"{primary_unscored} != {accounting['primary_target_row_count']}."
        )


def select_from_outer_validation(
    *,
    pooled_auprc: dict[str, float],
    subject_macro_auprc: dict[str, float],
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    """Delegate to the frozen protocol rule; never reimplement it here."""
    return select_t2_arm(
        pooled_auprc=pooled_auprc,
        subject_macro_auprc=subject_macro_auprc,
        parameter_counts=parameter_counts,
    )


def training_selection_status() -> dict[str, Any]:
    """After TRAIN-only execution both arms remain candidates."""
    return {
        "arm_selection_status": ARM_SELECTION_PENDING,
        "selected_arm": None,
        "candidates": list(T2_ARMS),
        "selection_requires": "one_shot_outer_validation",
    }


# ---------------------------------------------------------------------------
# Pooled / subject-macro / bootstrap evidence
# ---------------------------------------------------------------------------


def pooled_evidence(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    """The frozen pooled metric set at the arm's internal-dev threshold."""
    metrics = binary_metrics(labels, scores, threshold)
    return {name: metrics[name] for name in T2_POOLED_METRICS}


def _as_arrays(
    subjects: Sequence[str], labels: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The baseline helpers take `(labels, scores, subjects)` as numpy arrays."""
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(subjects, dtype=np.str_),
    )


def subject_macro_evidence(
    subjects: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Subject-macro metrics; the subject is the inferential unit, never a window."""
    label_array, score_array, subject_array = _as_arrays(subjects, labels, scores)
    macro = subject_macro_metrics(label_array, score_array, subject_array, threshold)
    return {name: macro.get(name) for name in T2_SUBJECT_MACRO_METRICS}


def subject_bootstrap_evidence(
    subjects: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """1000 subject resamples, seed 2026. Windows are never bootstrapped."""
    label_array, score_array, subject_array = _as_arrays(subjects, labels, scores)
    intervals = subject_bootstrap_confidence_intervals(
        label_array,
        score_array,
        subject_array,
        threshold,
        replicates=T2_BOOTSTRAP_REPLICATES,
        seed=T2_BOOTSTRAP_SEED,
    )
    return {
        "evidence_class": "t2_subject_bootstrap",
        "replicates": T2_BOOTSTRAP_REPLICATES,
        "seed": T2_BOOTSTRAP_SEED,
        "unit": "subject",
        "window_bootstrap_performed": False,
        "model_refitted_per_replicate": False,
        "claim_scope": T2_BOOTSTRAP_CLAIM,
        "intervals": intervals,
    }


# ---------------------------------------------------------------------------
# Temporal descriptive evidence -- descriptive only, never a selection input
# ---------------------------------------------------------------------------


def _runs(predictions: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous positive runs as `(start, stop)` half-open index pairs."""
    flags = np.asarray(predictions).astype(bool)
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


@dataclass(frozen=True, slots=True)
class T2DescriptorStream:
    """One physical `(record_id, channel_index)` stream's descriptor inputs.

    `predictions` and `score_present` are **full-timeline** and row-aligned:
    one entry per raw timeline position, in chronological order, including
    non-PRIMARY positions and physically unavailable ones. `score_present` is
    False exactly where no model output exists.
    """

    record_id: str
    channel_index: int
    predictions: np.ndarray
    score_present: np.ndarray
    primary_mask: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            np.asarray(self.predictions).shape,
            np.asarray(self.score_present).shape,
            np.asarray(self.primary_mask).shape,
            np.asarray(self.labels).shape,
        }
        if len(shapes) != 1:
            raise T2EvaluationError(
                f"Descriptor stream {self.record_id}:{self.channel_index} is not "
                f"row-aligned: {sorted(str(shape) for shape in shapes)}."
            )

    @property
    def row_count(self) -> int:
        return int(np.asarray(self.predictions).shape[0])


def _segments(stream: T2DescriptorStream) -> list[np.ndarray]:
    """Contiguous runs of SCORED positions within one stream.

    An unavailable position carries the model state unchanged but produces no
    prediction, so it is a continuity break for descriptive purposes: nothing
    was observed across it, and asserting a prediction run through it would be
    asserting evidence that does not exist. This says nothing about the model
    state, which is genuinely carried.
    """
    present = np.asarray(stream.score_present).astype(bool)
    if present.size == 0:
        return []
    padded = np.concatenate(([False], present, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges == 1)[0]
    stops = np.nonzero(edges == -1)[0]
    flags = np.asarray(stream.predictions).astype(bool)
    return [flags[start:stop] for start, stop in zip(starts, stops, strict=True)]


def temporal_descriptors(
    streams: Sequence[T2DescriptorStream],
    *,
    stride_seconds: float = T2_WINDOW_STRIDE_SECONDS,
) -> dict[str, Any]:
    """The frozen descriptive temporal statistics, derived stream by stream.

    Descriptive only: these can never choose a checkpoint, choose an arm or
    alter a threshold, and nothing in this module lets them.

    **Why the input is a list of streams and not one pooled vector.** A pooled
    PRIMARY-only vector is not a timeline. Concatenating it would splice
    separate records and channels end to end, so a positive run could be
    asserted across a stream boundary between two different patients; and
    filtering to PRIMARY first would delete the intervening challenge and
    other-non-primary predictions, stitching two runs into one that never
    happened. Both produce longer, calmer-looking runs than the model actually
    produced. Segmentation here is therefore by `(record_id, channel_index)`
    first and by scored-position continuity second.

    **Which rows count.** Runs and transitions use the frozen-threshold
    predictions on every AVAILABLE SCORED row -- PRIMARY, CHALLENGE and OTHER
    non-primary alike -- because all three are real outputs of the same causal
    pass. The row role is a mask over the outputs, never a model input.

    **The transition denominator is physical time.** Not
    `scored_rows * stride`, which would silently shrink the denominator by
    every unavailable and every non-primary position and inflate the rate.
    Exposure is `full_timeline_row_count * stride` per stream, summed.

    **What `prediction_persistence_around_labelled_ischemic_intervals` is.** It
    is the fraction of labelled-positive PRIMARY *windows* that were predicted
    positive -- a window-level quantity conditional on the labelled-positive
    population, computed independently of run segmentation. It is **not** an
    episode onset/offset persistence measurement: it never groups windows into
    episodes at all. Formal episode reasoning is T1's.
    """
    stride = float(stride_seconds)
    keys = [(item.record_id, int(item.channel_index)) for item in streams]
    if len(set(keys)) != len(keys):
        raise T2EvaluationError(
            "A descriptor stream key appears twice; the timeline would be "
            "interleaved and runs could cross a stream boundary."
        )

    run_lengths: list[int] = []
    transitions = 0
    exposure_seconds = 0.0
    labelled_positive = 0
    labelled_positive_predicted = 0

    for stream in streams:
        # Physical exposure counts EVERY raw timeline position, including the
        # non-primary and unavailable ones the patient was still monitored
        # through.
        exposure_seconds += stream.row_count * stride
        primary = np.asarray(stream.primary_mask).astype(bool)
        present = np.asarray(stream.score_present).astype(bool)
        flags = np.asarray(stream.predictions).astype(bool)
        truth = np.asarray(stream.labels).astype(bool)
        scored_primary = primary & present
        labelled = scored_primary & truth
        labelled_positive += int(labelled.sum())
        labelled_positive_predicted += int((labelled & flags).sum())

        for segment in _segments(stream):
            if segment.size == 0:
                continue
            padded = np.concatenate(([False], segment, [False]))
            edges = np.diff(padded.astype(np.int8))
            starts = np.nonzero(edges == 1)[0]
            stops = np.nonzero(edges == -1)[0]
            run_lengths.extend(
                (stops - starts).tolist()  # type: ignore[arg-type]
            )
            # Transitions are counted WITHIN a contiguous scored segment only:
            # never across a stream boundary and never across a gap where no
            # prediction was observed.
            transitions += int(np.count_nonzero(np.diff(segment.astype(np.int8)) != 0))

    durations = sorted(float(length) * stride for length in run_lengths)
    isolated = sum(1 for length in run_lengths if length == 1)
    exposure_hours = exposure_seconds / 3600.0
    return {
        "evidence_class": "t2_temporal_descriptors",
        "is_selection_input": False,
        "may_alter_threshold": False,
        "stream_count": len(streams),
        "physical_exposure_seconds": exposure_seconds,
        "positive_prediction_run_count": len(run_lengths),
        "median_positive_run_duration_seconds": (
            None if not durations else float(np.median(durations))
        ),
        "isolated_single_window_positive_fraction": (
            None if not run_lengths else isolated / len(run_lengths)
        ),
        "transition_count": transitions,
        "transition_count_per_hour": (
            None if exposure_hours == 0 else transitions / exposure_hours
        ),
        "labelled_positive_window_count": labelled_positive,
        "labelled_positive_window_prediction_fraction": (
            None
            if labelled_positive == 0
            else labelled_positive_predicted / labelled_positive
        ),
        "prediction_persistence_around_labelled_ischemic_intervals": (
            None
            if labelled_positive == 0
            else labelled_positive_predicted / labelled_positive
        ),
        "prediction_persistence_definition": (
            "fraction_of_labelled_positive_windows_predicted_positive"
        ),
        "prediction_persistence_unit": "window",
        "prediction_persistence_conditioning_population": ("labelled_positive_windows"),
        "prediction_persistence_is_episode_onset_offset_measurement": False,
        "prediction_persistence_derived_from_run_segmentation": False,
        "runs_cross_stream_boundaries": False,
        "primary_only_sequence_used_for_runs": False,
        "unavailable_gap_stitches_runs": False,
        "run_segmentation_key": "record_id_channel_index",
        "transition_denominator": "full_physical_timeline_exposure",
        "episode_grouping_performed": False,
        "formal_episode_reasoning_belongs_to": "t1",
    }


# ---------------------------------------------------------------------------
# Cold start and challenge reporting mechanics
# ---------------------------------------------------------------------------


def cold_start_strata_evidence(
    bins: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """Inherited strata, reported. No warmup threshold and no repair."""
    bins_array = np.asarray(bins)
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    strata: dict[str, Any] = {}
    for stratum in T2_COLD_START_STRATA:
        selected = bins_array == stratum
        count = int(selected.sum())
        if count == 0:
            strata[stratum] = {"row_count": 0, "metrics": None}
            continue
        strata[stratum] = {
            "row_count": count,
            "metrics": binary_metrics(
                labels_array[selected].tolist(),
                scores_array[selected].tolist(),
                threshold,
            ),
        }
    return {
        "evidence_class": "t2_cold_start_evidence",
        "warmup_threshold_applied": False,
        "cold_start_repair_applied": False,
        "alternative_state_initialization": False,
        "strata": strata,
    }


# The precise causal-context semantics. A broad `trained_on: false` would be
# FALSE: an AVAILABLE challenge `z_t` is label-blind causal context and can
# influence a later PRIMARY training loss through the carried state. What is
# true is narrower, and each clause below is separately true.
CHALLENGE_CAUSAL_SEMANTICS: Final = {
    "direct_training_loss_received": False,
    "challenge_identity_model_input": False,
    "challenge_label_model_input": False,
    "may_be_label_blind_causal_context": True,
    "checkpoint_selection_input": False,
    "arm_selection_input": False,
}


def challenge_family_evidence(
    families: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """False-positive behaviour per challenge family at the frozen threshold.

    Note what this does **not** say. There is deliberately no `trained_on`
    field: a challenge row receives no direct loss, but the model does consume
    its representation as causal context, so its `z_t` can move a later PRIMARY
    row's loss through the carried state. `CHALLENGE_CAUSAL_SEMANTICS` states
    the six things that are actually true instead of one thing that is not.
    """
    families_array = np.asarray(families)
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    subsets: dict[str, Any] = {}
    for family in T2_CHALLENGE_FAMILIES:
        selected = families_array == family
        count = int(selected.sum())
        predicted = scores_array[selected] >= threshold
        subsets[family] = {
            "row_count": count,
            "false_positive_count": int(predicted.sum()),
            "false_positive_rate": (None if count == 0 else float(predicted.mean())),
            "evidence_level": (
                "exploratory_descriptive"
                if family == "conduction_change"
                else "quantitative_secondary"
            ),
            "is_selection_input": False,
            "merged_into_primary": False,
            **dict(CHALLENGE_CAUSAL_SEMANTICS),
        }
        if int(labels_array[selected].sum()) and family != "conduction_change":
            subsets[family]["label_positive_present"] = True
    return {
        "evidence_class": "t2_challenge_evidence",
        "is_selection_input": False,
        "merged_into_primary": False,
        **dict(CHALLENGE_CAUSAL_SEMANTICS),
        "subsets": subsets,
    }
