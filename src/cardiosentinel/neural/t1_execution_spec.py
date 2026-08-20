"""Structural binder for the frozen T1-v1 canonical development execution spec.

The science is already frozen in ``t1_protocol``. This module holds the
**non-scientific execution mechanics**: the canonical identity, the future CLI
surface, the stage order, the artifact plan, the label-firewall ordering, the
runtime-integrity requirements and the refusals that enforce them.

**Structural binder only.** Standard library throughout. It loads no run
artifact, opens no M2/U1/T2 evidence, computes no score, calculates no
threshold, executes ``next_state`` on no real data, reads no VALIDATION, reads
no TEST and mutates no scientific state. Nothing here can touch scientific
state, and the accompanying tests prove that structurally rather than trusting
it.

**This module does not execute anything.** The canonical development harness
(``t1_development_run``) belongs to a later, separately reviewed PR.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]

T1_EXECUTION_SPEC_NAME: Final = "T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1"
T1_EXECUTION_SPEC_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{T1_EXECUTION_SPEC_NAME}.md"
)
T1_EXECUTION_SPEC_SHA256: Final = (
    "11b6a9aff2f1d928a9f33516db2ea764cf0553a949cd79c14562bafe34f090bf"
)

# The protocol this specification serves. Bound so the two cannot drift apart.
T1_PROTOCOL_DOCUMENT_SHA256: Final = (
    "ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4"
)
T1_SPECIFICATION_STARTING_GIT_SHA: Final = "9fa7e88ee648a71e397516ccce210dcf0f06c409"

# ---------------------------------------------------------------------------
# Canonical execution identity (spec §1)
# ---------------------------------------------------------------------------
T1_EXPERIMENT_IDENTITY: Final = "T1_state_machine_v1"
T1_DEVELOPMENT_ATTEMPT_ID: Final = "t1-v1-development"
T1_RUN_ROOT_RELATIVE: Final = Path("cardiosentinel-runs") / "phase9-t1-development-v1"

T1_ATTEMPT_NAME_IS_DETERMINISTIC: Final = True
T1_ATTEMPT_NAME_CARRIES_TIMESTAMP: Final = False
T1_ATTEMPT_NAME_CARRIES_UUID: Final = False
T1_ATTEMPT_NAME_CARRIES_RANDOM_SUFFIX: Final = False
T1_AUTOMATIC_RETRY_PERMITTED: Final = False
T1_RECOVERY_IDENTITY_PREDECLARED: Final = False
T1_ALTERNATE_RUN_ROOT_PERMITTED: Final = False
T1_CLAIM_IS_THE_RUN_DIRECTORY: Final = True
T1_CLAIM_CONSUMED_ONCE_CREATED: Final = True

# ---------------------------------------------------------------------------
# Future public CLI contract (spec §2)
# ---------------------------------------------------------------------------
T1_FUTURE_CLI_MODULE: Final = "cardiosentinel.neural.t1_development_run"
T1_CANONICAL_EXECUTION_FLAG: Final = "--execute-canonical-development"
T1_EXPECTED_GIT_SHA_FLAG: Final = "--expected-git-sha"
T1_FUTURE_CLI_OPTIONS: Final = (
    T1_CANONICAL_EXECUTION_FLAG,
    T1_EXPECTED_GIT_SHA_FLAG,
)

T1_FORBIDDEN_CLI_OPTIONS: Final = (
    "--q-watch",
    "--q-event",
    "--profile",
    "--threshold",
    "--p-watch",
    "--s-watch",
    "--p-event",
    "--s-event",
    "--subject",
    "--fold",
    "--retry",
    "--force",
    "--seed",
    "--bootstrap",
    "--test",
    "--router",
)

# The future harness merge SHA is not known when this specification is written,
# so no execution SHA is frozen. The specification-PR SHA is explicitly not it.
T1_FUTURE_EXECUTION_GIT_SHA_KNOWN: Final = False
T1_GIT_SHA_IS_THE_AUTHORIZATION_MECHANISM: Final = True

# ---------------------------------------------------------------------------
# Canonical interpreter and runtime integrity (spec §3)
# ---------------------------------------------------------------------------
T1_CANONICAL_INTERPRETER: Final = "/home/AI_POC/venvs/tactics/bin/python"
T1_EXPECTED_DEPENDENCY_DIGEST: Final = (
    "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
)
T1_RUNTIME_SENTINEL_DOCUMENT_SHA256: Final = (
    "cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075"
)
T1_PACKAGE_INSTALL_PERMITTED: Final = False
T1_AUTOMATIC_ENVIRONMENT_REPAIR_PERMITTED: Final = False
T1_ALTERNATE_INTERPRETER_PERMITTED: Final = False

T1_RUNTIME_ENFORCEMENT_POINTS: Final = (
    "start",
    "pre_label_blind_input_promotion",
    "pre_fold_selection_promotion",
    "pre_held_out_evidence_promotion",
    "pre_oof_result_promotion",
    "pre_final_configuration_promotion",
    "pre_experiment_lock_promotion",
    "completion",
)
T1_RUNTIME_ENFORCEMENT_MAY_BE_WEAKENED: Final = False

# ---------------------------------------------------------------------------
# Upstream verification (spec §5)
# ---------------------------------------------------------------------------
T1_REQUIRED_UPSTREAM_VALIDATORS: Final = (
    "validate_retained_m2_arm",
    "validate_u1_retention_decision",
    "validate_retained_u1_calibration",
    "validate_retained_t2_arm",
)
T1_PARALLEL_WEAKER_VERIFIER_PERMITTED: Final = False

T1_REQUIRED_M2_RETAINED_ARM: Final = "M2-G"
T1_REQUIRED_U1_FAMILY: Final = "platt_logistic_on_recovered_logit"
T1_REQUIRED_T2_RETAINED_ARM: Final = "causal_s4d_longitudinal_v1"
T1_REQUIRED_T2_SCORE_SEMANTICS: Final = "uncalibrated_temporal_model_score"

T1_U1_REFIT_PERMITTED: Final = False
T1_U1_DEPLOYMENT_CALIBRATOR_PERMITTED_FOR_DEVELOPMENT: Final = False
T1_M2_REPLAY_PERMITTED: Final = False
T1_T2_REPLAY_PERMITTED: Final = False

# The exact already-fitted arithmetic the harness reuses. Naming it here is what
# makes "no refit" checkable rather than merely stated.
T1_U1_APPLY_CALLABLE: Final = "U1Calibrator.apply_to_scores"
T1_U1_FOLD_MANIFEST_NAME: Final = "U1_FOLD_MANIFEST.json"
T1_U1_FOLD_COUNT: Final = 12
T1_U1_FIT_SUBJECTS_PER_FOLD: Final = 11
T1_U1_FORBIDDEN_FITTING_CALLABLES: Final = (
    "fit_calibrator",
    "select_calibrator_family",
    "scipy.optimize",
    "minimize",
)

# ---------------------------------------------------------------------------
# The label-blind full-timeline assembly (spec §6)
# ---------------------------------------------------------------------------
T1_TIMELINE_ROW_COUNT: Final = 492_904
T1_EXPECTED_SCORED_ROWS: Final = 492_898
T1_EXPECTED_UNAVAILABLE_ROWS: Final = 6

T1_M2_ROW_EVIDENCE_NAME: Final = "row_evidence.npz"
T1_M2_COLUMNS_USED: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "score",
    "scored",
)
T1_M2_COLUMNS_NOT_TRANSITION_FEATURES: Final = ("update_admitted", "available_time")

T1_T2_IDENTITY_NAME: Final = "t2_outer_row_identity.npz"
T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "subject_id",
    "score_present",
    "observation_state",
)
T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND: Final = (
    "target_family",
    "label",
    "primary_mask",
    "cold_start_bin",
)

# The existing convenience readers materialise EVERY column named in the
# manifest entry, so either would silently pull `label` and `target_family` into
# memory during a step that is supposed to be label-blind.
T1_T2_READERS_FORBIDDEN_FOR_LABEL_BLIND_ASSEMBLY: Final = (
    "read_t2_outer_row_group",
    "selected_arm_scores",
)
T1_WHOLE_NPZ_MATERIALISATION_PERMITTED: Final = False
T1_MEMBER_RESTRICTED_READER_REQUIRED: Final = True

# ---------------------------------------------------------------------------
# Subject identity (spec §7) and calibration arithmetic (spec §8)
# ---------------------------------------------------------------------------
T1_SUBJECT_AUTHORITY_CALLABLE: Final = "subject_id_for_record"
T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE: Final = False
T1_SUBJECT_IDENTITY_DERIVED_FROM_LABEL: Final = False
T1_SUBJECT_IDENTITY_CROSS_CHECK_MUST_AGREE_EXACTLY: Final = True

T1_DETECTOR_THRESHOLD: Final = 0.7554003000259399
T1_U1_CLAMP_DELTA: Final = 1e-7
T1_CALIBRATION_IS_A_FIT: Final = False
T1_CALIBRATION_CONTRACT: Final = (
    "apply_frozen_held_out_subject_platt_calibrator_to_every_scored_m2g_row"
)

# ---------------------------------------------------------------------------
# Availability alignment (spec §9) and elapsed time (spec §11)
# ---------------------------------------------------------------------------
T1_STABLE_ID_EQUALITY_REQUIRED: Final = True
T1_AVAILABILITY_MASK_EQUALITY_REQUIRED: Final = True
T1_AVAILABILITY_MISMATCH_IS_HARD_STOP: Final = True
T1_SYNTHETIC_SCORE_PERMITTED: Final = False

T1_SAMPLING_FREQUENCY_HZ: Final = 250
T1_ELAPSED_STREAM_SECONDS_SOURCE: Final = "physical_sample_coordinates"
T1_ELAPSED_FROM_ROW_ORDINAL_PERMITTED: Final = False
T1_EMITTED_STATE_CONVENTION: Final = "state_after_processing_current_row"
T1_STATE_DURATION_FIELD: Final = "state_elapsed_seconds"
T1_STATE_ELAPSED_CREATES_NEW_TRANSITION_CONDITION: Final = False

# ---------------------------------------------------------------------------
# Evidence store schemas (spec §10, §18)
# ---------------------------------------------------------------------------
T1_INPUT_EVIDENCE_COLUMNS: Final = (
    "stable_id",
    "record_id",
    "channel_index",
    "start_sample",
    "subject_id",
    "score_present",
    "m2g_detector_score",
    "detector_decision_d_t",
    "oof_calibrated_probability_p_t",
    "decision_error_uncertainty_u_t",
    "s4d_temporal_evidence_s_t",
    "elapsed_stream_seconds",
)
T1_OOF_STATE_EVIDENCE_COLUMNS: Final = T1_INPUT_EVIDENCE_COLUMNS + (
    "fold_index",
    "selected_policy_id",
    "p_watch",
    "s_watch",
    "p_event",
    "s_event",
    "emitted_state",
    "state_elapsed_seconds",
    "transition_from",
    "transition_to",
    "transition_occurred",
)
T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS: Final = (
    "label",
    "target_family",
    "episode_identity",
    "challenge_identity",
    "primary_mask",
    "test_field",
)
T1_EVIDENCE_PERSISTS_TYPED_ARRAYS: Final = True

# ---------------------------------------------------------------------------
# The fold-scoped label firewall (spec §12, §16, §17)
# ---------------------------------------------------------------------------
T1_FOLD_COUNT: Final = 12
T1_CANDIDATE_POLICIES_PER_FOLD: Final = 12
T1_FOLD_RETRY_PERMITTED: Final = False

T1_GLOBAL_LABEL_TABLE_PERMITTED: Final = False
T1_FOLD_SCOPED_TARGET_AUTHORITY_REQUIRED: Final = True
T1_T2_IDENTITY_NPZ_AS_LABEL_TABLE_PERMITTED: Final = False

T1_FOLD_SELECTION_ARTIFACT_BINDINGS: Final = (
    "fold_index",
    "held_out_subject",
    "fit_subjects",
    "fit_label_authority_identity",
    "candidate_policy_identities",
    "generated_thresholds_per_candidate",
    "fit_candidate_selection_metrics",
    "selected_policy",
    "selected_thresholds",
    "selection_path_tie_break_stage",
    "t1_protocol_sha256",
    "t1_execution_spec_sha256",
    "input_evidence_store_sha256",
    "test_accessed",
)
T1_FOLD_SELECTION_REREAD_AND_DIGEST_VERIFIED: Final = True
T1_HELD_OUT_BARRIER_IS_STRUCTURAL: Final = True
T1_HELD_OUT_ACCESS_FLAG: Final = "held_out_label_access_authorized_for_this_fold"

T1_REJECTED_POLICIES_RUN_ON_HELD_OUT: Final = False
T1_HELD_OUT_POLICY_RUNS_PER_FOLD: Final = 1

# ---------------------------------------------------------------------------
# Selection, exposure, bootstrap, challenge, final configuration
# (spec §15, §20, §21, §22, §23)
# ---------------------------------------------------------------------------
T1_EPISODE_F1_DEFINITION: Final = "two_tp_over_two_tp_plus_fp_plus_fn"
T1_UNDEFINED_METRIC_BECOMES_ZERO: Final = False
T1_UNDEFINED_COMPARISON_REQUIRES_HUMAN_REVIEW: Final = True

T1_STRIDE_SECONDS: Final = 5.0
T1_EXPOSURE_INCLUDES_UNAVAILABLE_POSITIONS: Final = True
T1_EXPOSURE_IS_PRIMARY_ONLY: Final = False
T1_FALSE_ONSET_NUMERATOR: Final = "unmatched_predicted_event_runs"

T1_BOOTSTRAP_REPLICATES: Final = 1000
T1_BOOTSTRAP_SEED: Final = 2026
T1_BOOTSTRAP_UNIT: Final = "subject"
T1_BOOTSTRAP_RESELECTS_POLICY: Final = False
T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY: Final = True

T1_CHALLENGE_JOIN_AFTER_STATE_TRACE: Final = True
T1_CHALLENGE_IS_SELECTION_INPUT: Final = False
T1_CHALLENGE_IS_TRANSITION_INPUT: Final = False
T1_CHALLENGE_IS_THRESHOLD_GENERATION_INPUT: Final = False

T1_FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE: Final = False
T1_FINAL_CONFIGURATION_OVERWRITES_OOF_RESULT: Final = False
T1_CATEGORICAL_STATE_AUPRC_REPORTED: Final = False

# ---------------------------------------------------------------------------
# Artifact plan (spec §24) and failure semantics (spec §25)
# ---------------------------------------------------------------------------
T1_PLANNED_ARTIFACTS: Final = (
    "T1_RUN_STATUS.json",
    "T1_PREFLIGHT.json",
    "T1_INPUT_LINEAGE.json",
    "T1_INPUT_EVIDENCE.json",
    "T1_FOLD_SELECTIONS.json",
    "T1_OOF_STATE_EVIDENCE.json",
    "T1_OOF_RESULT.json",
    "T1_SUBJECT_EVIDENCE.json",
    "T1_BOOTSTRAP.json",
    "T1_CHALLENGE_EVIDENCE.json",
    "T1_FINAL_CONFIGURATION.json",
    "T1_RESULT.json",
    "T1_EXPERIMENT_LOCK.json",
)
T1_ARTIFACTS_CREATED_BY_THIS_SPECIFICATION: Final = ()
T1_PROMOTED_EVIDENCE_MAY_BE_OVERWRITTEN: Final = False
T1_ATOMIC_WRITE_REQUIRED: Final = True

T1_FAILURE_RECEIPT_FIELDS: Final = (
    "stage",
    "current_fold",
    "label_blind_input_opened",
    "fit_labels_opened_for_folds",
    "fold_selections_promoted",
    "held_out_labels_opened_for_folds",
    "held_out_traces_completed",
    "oof_evidence_promoted",
    "final_configuration_started",
    "final_configuration_completed",
    "test_state",
    "runtime_integrity_state",
    "exception_type",
    "exception_message",
)
T1_PRE_CLAIM_REFUSAL_CONSUMES_ATTEMPT: Final = False
T1_POST_CLAIM_FAILURE_CONSUMES_ATTEMPT: Final = True
T1_FAILED_ATTEMPT_MAY_BE_DELETED_OR_REWRITTEN: Final = False

# ---------------------------------------------------------------------------
# Firewalls (spec §26)
# ---------------------------------------------------------------------------
T1_TEST_ACCESSED: Final = False
T1_SEALED_TEST_STATE: Final = "unopened"
T1_TEST_CLI_OPTION_EXPOSED: Final = False
T1_ROUTING_DEFINED: Final = False
T1_LLM_PARTICIPATES_IN_EXECUTION: Final = False
T1_SCIENTIFIC_EXECUTION_PERFORMED_BY_THIS_MODULE: Final = False

# ---------------------------------------------------------------------------
# The frozen stage order (spec §28)
# ---------------------------------------------------------------------------
STAGE_START: Final = "start"
STAGE_VERIFY_GIT: Final = "verify_expected_git_sha"
STAGE_VALIDATE_PROTOCOL: Final = "validate_t1_protocol_document"
STAGE_VALIDATE_SPEC: Final = "validate_t1_execution_spec_document"
STAGE_VALIDATE_M2: Final = "validate_m2_retention_decision"
STAGE_VALIDATE_U1: Final = "validate_u1_retention_decision"
STAGE_VALIDATE_T2: Final = "validate_t2_retention_decision"
STAGE_PROVE_TEST_UNOPENED: Final = "prove_test_unopened"
STAGE_PROVE_ATTEMPT_ABSENT: Final = "prove_canonical_attempt_absent"
STAGE_CLAIM: Final = "claim_run_directory"
STAGE_VERIFY_UPSTREAM: Final = "verify_upstream_chain_after_claim"
STAGE_ASSEMBLE_LABEL_BLIND: Final = "assemble_label_blind_full_timeline"
STAGE_PROMOTE_INPUT_EVIDENCE: Final = "promote_label_blind_input_evidence"
STAGE_FOLD_OPEN_FIT_LABELS: Final = "fold_open_fit_subject_labels"
STAGE_FOLD_GENERATE_THRESHOLDS: Final = "fold_generate_candidate_thresholds"
STAGE_FOLD_RUN_CANDIDATES: Final = "fold_run_twelve_candidate_policies"
STAGE_FOLD_SELECT: Final = "fold_select_one_policy"
STAGE_FOLD_PROMOTE_SELECTION: Final = "fold_promote_selection_artifact"
STAGE_FOLD_AUTHORIZE_HELD_OUT: Final = "fold_authorize_held_out_label_access"
STAGE_FOLD_OPEN_HELD_OUT_LABELS: Final = "fold_open_held_out_labels"
STAGE_FOLD_RUN_SELECTED: Final = "fold_run_selected_policy_on_held_out"
STAGE_FOLD_PROMOTE_HELD_OUT: Final = "fold_promote_held_out_evidence"
STAGE_OOF_STATE_EVIDENCE: Final = "promote_oof_state_evidence"
STAGE_OOF_RESULT: Final = "promote_oof_result"
STAGE_BOOTSTRAP: Final = "subject_evidence_and_bootstrap"
STAGE_CHALLENGE: Final = "challenge_reporting_join"
STAGE_FINAL_CONFIGURATION: Final = "final_all_validation_configuration"
STAGE_EXPERIMENT_LOCK: Final = "promote_experiment_lock"
STAGE_COMPLETION: Final = "completion"

T1_STAGE_ORDER: Final = (
    STAGE_START,
    STAGE_VERIFY_GIT,
    STAGE_VALIDATE_PROTOCOL,
    STAGE_VALIDATE_SPEC,
    STAGE_VALIDATE_M2,
    STAGE_VALIDATE_U1,
    STAGE_VALIDATE_T2,
    STAGE_PROVE_TEST_UNOPENED,
    STAGE_PROVE_ATTEMPT_ABSENT,
    STAGE_CLAIM,
    STAGE_VERIFY_UPSTREAM,
    STAGE_ASSEMBLE_LABEL_BLIND,
    STAGE_PROMOTE_INPUT_EVIDENCE,
    STAGE_FOLD_OPEN_FIT_LABELS,
    STAGE_FOLD_GENERATE_THRESHOLDS,
    STAGE_FOLD_RUN_CANDIDATES,
    STAGE_FOLD_SELECT,
    STAGE_FOLD_PROMOTE_SELECTION,
    STAGE_FOLD_AUTHORIZE_HELD_OUT,
    STAGE_FOLD_OPEN_HELD_OUT_LABELS,
    STAGE_FOLD_RUN_SELECTED,
    STAGE_FOLD_PROMOTE_HELD_OUT,
    STAGE_OOF_STATE_EVIDENCE,
    STAGE_OOF_RESULT,
    STAGE_BOOTSTRAP,
    STAGE_CHALLENGE,
    STAGE_FINAL_CONFIGURATION,
    STAGE_EXPERIMENT_LOCK,
    STAGE_COMPLETION,
)

T1_PER_ROW_ACCESS_STAGES: Final = (
    STAGE_ASSEMBLE_LABEL_BLIND,
    STAGE_PROMOTE_INPUT_EVIDENCE,
    STAGE_FOLD_OPEN_FIT_LABELS,
    STAGE_FOLD_GENERATE_THRESHOLDS,
    STAGE_FOLD_RUN_CANDIDATES,
    STAGE_FOLD_SELECT,
    STAGE_FOLD_PROMOTE_SELECTION,
    STAGE_FOLD_OPEN_HELD_OUT_LABELS,
    STAGE_FOLD_RUN_SELECTED,
    STAGE_FOLD_PROMOTE_HELD_OUT,
)


class T1ExecutionSpecError(RuntimeError):
    """Raised when an execution-specification rule is violated."""


def _sha256_file(path: Path) -> str:
    """The same streaming digest the other protocol/spec modules use.

    Binder modules stay standard-library only, so this mirrors
    ``t1_protocol._sha256_file`` rather than importing the persistence helper.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_t1_execution_spec_document(path: Path | None = None) -> str:
    """Verify the frozen execution specification byte-for-byte.

    The path is resolved at CALL time, not bound as a default argument. A
    default bound at definition time can never be reached by monkeypatching the
    module constant, which has caused real confusion in this repository before.
    """
    document = Path(path) if path is not None else T1_EXECUTION_SPEC_PATH
    if not document.is_file():
        raise T1ExecutionSpecError(f"T1 execution spec is missing at {document}.")
    digest = _sha256_file(document)
    if digest != T1_EXECUTION_SPEC_SHA256:
        raise T1ExecutionSpecError(
            f"T1 execution spec digest {digest} differs from the frozen "
            f"{T1_EXECUTION_SPEC_SHA256}. The specification is immutable."
        )
    return digest


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def require_cli_option_permitted(option: str) -> str:
    """The future CLI carries no scientific knob. This is where that is enforced."""
    if option in T1_FORBIDDEN_CLI_OPTIONS:
        raise T1ExecutionSpecError(
            f"{option!r} may not exist on the T1 development CLI. A scientific "
            "choice reachable from a command line is a scientific choice a human "
            "can make after seeing results, which is exactly what the prospective "
            "design exists to prevent."
        )
    if option not in T1_FUTURE_CLI_OPTIONS:
        raise T1ExecutionSpecError(
            f"{option!r} is not one of the two frozen CLI options "
            f"{T1_FUTURE_CLI_OPTIONS}."
        )
    return option


def require_label_blind_member(member: str) -> str:
    """Refuse any identity member that may not be materialised label-blind."""
    if member in T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND:
        raise T1ExecutionSpecError(
            f"{member!r} may not be materialised during label-blind assembly. "
            "A runtime transition that depended on evaluation annotation would "
            "not be deployable, because that annotation does not exist on a live "
            "stream."
        )
    if member not in T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND:
        raise T1ExecutionSpecError(
            f"{member!r} is not one of the permitted label-blind identity members "
            f"{T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND}."
        )
    return member


def require_member_restricted_reader(reader_name: str) -> str:
    """Refuse the convenience readers that materialise every manifest column."""
    if reader_name in T1_T2_READERS_FORBIDDEN_FOR_LABEL_BLIND_ASSEMBLY:
        raise T1ExecutionSpecError(
            f"{reader_name!r} materialises every column named in the manifest "
            "entry, which for the T2 row identity includes 'label' and "
            "'target_family'. Label-blind assembly requires a member-restricted "
            "reader that names what it materialises."
        )
    return reader_name


def require_no_refit(callable_name: str) -> str:
    """Applying a frozen calibrator is arithmetic; fitting one is a new decision."""
    if callable_name in T1_U1_FORBIDDEN_FITTING_CALLABLES:
        raise T1ExecutionSpecError(
            f"{callable_name!r} would fit or reselect a calibrator. T1 reuses the "
            f"already-fitted parameters through {T1_U1_APPLY_CALLABLE}; a refit "
            "would replace frozen U1 evidence with new evidence created after the "
            "fact."
        )
    return callable_name


def require_evidence_column_permitted(column: str) -> str:
    if column in T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS:
        raise T1ExecutionSpecError(
            f"{column!r} may not appear in a T1 evidence store. The downstream "
            "routing layer must be able to consume this store without inheriting "
            "an evaluation-label dependency."
        )
    return column


def require_stage_known(stage: str) -> str:
    if stage not in T1_STAGE_ORDER:
        raise T1ExecutionSpecError(f"{stage!r} is not a frozen T1 execution stage.")
    return stage


def stage_index(stage: str) -> int:
    return T1_STAGE_ORDER.index(require_stage_known(stage))


def require_stage_precedes(earlier: str, later: str) -> None:
    """Prove one stage is frozen ahead of another, by index rather than by prose."""
    if stage_index(earlier) >= stage_index(later):
        raise T1ExecutionSpecError(
            f"Stage {earlier!r} must precede {later!r} in the frozen order."
        )


def require_claim_before_per_row_access(stage: str) -> str:
    """No per-row scientific evidence may be opened before the claim exists."""
    require_stage_known(stage)
    if stage in T1_PER_ROW_ACCESS_STAGES and stage_index(stage) < stage_index(
        STAGE_CLAIM
    ):
        raise T1ExecutionSpecError(  # pragma: no cover - frozen order forbids it
            f"Stage {stage!r} opens per-row evidence before the claim. The run "
            "directory is the scientific claim; reading the timeline and then "
            "declining to claim would be an unrecorded look at the data."
        )
    return stage


def require_held_out_access_authorized(fold_state: dict[str, Any]) -> dict[str, Any]:
    """The structural half of the fold-scoped label firewall.

    A fold may open its held-out subject's labels only after its selection
    artifact has been promoted AND re-read with a verified digest.
    """
    if not fold_state.get("selection_promoted"):
        raise T1ExecutionSpecError(
            "Held-out labels may not be opened before this fold's selection "
            "artifact has been promoted. Selecting a policy after seeing held-out "
            "truth is not cross-fitting."
        )
    if not fold_state.get("selection_digest_verified"):
        raise T1ExecutionSpecError(
            "The promoted fold-selection artifact must be re-read and its digest "
            "verified before held-out labels are opened. An artifact that was "
            "written but never read back is not proof that it was written."
        )
    if not fold_state.get(T1_HELD_OUT_ACCESS_FLAG):
        raise T1ExecutionSpecError(
            f"{T1_HELD_OUT_ACCESS_FLAG} is not set for this fold."
        )
    return fold_state


def require_single_held_out_policy_run(run_count: int) -> int:
    if run_count != T1_HELD_OUT_POLICY_RUNS_PER_FOLD:
        raise T1ExecutionSpecError(
            f"Exactly {T1_HELD_OUT_POLICY_RUNS_PER_FOLD} policy runs the held-out "
            f"subject, not {run_count}. Running the rejected candidates there "
            "would turn the held-out fold into a second selection set."
        )
    return run_count


def require_no_test_access(partition: str) -> str:
    if str(partition).strip().lower() == "test":
        raise T1ExecutionSpecError(
            "TEST is sealed. The T1 development package resolves no TEST path, "
            "reads no TEST metadata and computes no TEST metric."
        )
    return partition


def require_defined_metric(name: str, value: float | None) -> float:
    """An undefined metric stays undefined; it never becomes a silent zero."""
    if value is None:
        raise T1ExecutionSpecError(
            f"{name} is undefined for this population. It is preserved as "
            "undefined and requires human review rather than being converted to "
            "zero, which would read as a real measurement of zero."
        )
    return float(value)


def specification_identity() -> dict[str, Any]:
    """Everything a future artifact binds to name this specification."""
    return {
        "execution_spec_name": T1_EXECUTION_SPEC_NAME,
        "execution_spec_sha256": T1_EXECUTION_SPEC_SHA256,
        "protocol_document_sha256": T1_PROTOCOL_DOCUMENT_SHA256,
        "experiment_identity": T1_EXPERIMENT_IDENTITY,
        "attempt_id": T1_DEVELOPMENT_ATTEMPT_ID,
        "run_root_relative": str(T1_RUN_ROOT_RELATIVE),
        "specification_starting_git_sha": T1_SPECIFICATION_STARTING_GIT_SHA,
        "test_accessed": T1_TEST_ACCESSED,
        "sealed_test_state": T1_SEALED_TEST_STATE,
        "routing_defined": T1_ROUTING_DEFINED,
        "scientific_execution_performed": (
            T1_SCIENTIFIC_EXECUTION_PERFORMED_BY_THIS_MODULE
        ),
    }
