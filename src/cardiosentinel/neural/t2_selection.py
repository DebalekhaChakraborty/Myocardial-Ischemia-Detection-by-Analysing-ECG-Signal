"""Immutable binding of the human T2 longitudinal retention decision.

This module records a *decision*, not a computation. It validates the completed
one-shot outer-VALIDATION attempt through the existing canonical verifier,
proves that the arm a human retained is the arm the canonical selector chose,
and refuses anything else. It deliberately contains no model construction, no
inference, no checkpoint scoring, no metric recomputation, no threshold fitting,
no TRAIN replay, no VALIDATION replay and no selection logic: the T2 execution
machinery is untouched, and importing or calling this module can never alter an
artifact or open a partition.

**The retained object is the continuous score**, not the binary trace. T2's
scientific role is longitudinal temporal evidence for the current window, so
what is retained for downstream T1 development is
`uncalibrated_temporal_model_score = sigmoid(current_window_t2_logit)`. The
frozen S4D threshold is bound here as immutable *experiment and reporting*
evidence, and the constants below record, in executable form, that it is not T1
policy and cannot select a T1 state.

The comparator arm is bound too. `causal_gru_longitudinal_v1` remains immutable
comparator/ablation evidence, so a later phase can prove it is not silently
substituting the GRU for the retained S4D -- or quietly dropping it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.t2_outer_evidence import (
    T2_OUTER_STORE_MANIFEST_NAME,
    T2_SCORE_DEFINITION,
    T2_SCORE_SEMANTICS,
)
from cardiosentinel.neural.t2_persistence import (
    OUTER_LOCK_NAME,
    OUTER_RESULT_NAME,
    T2_EXECUTION_SPEC_SHA256,
    T2_OUTER_VALIDATION_ATTEMPT_ID,
    T2_TRAIN_ARTIFACT_REVIEW_SHA256,
    validate_canonical_t2_outer_validation_attempt,
    validate_t2_execution_spec,
    validate_t2_train_artifact_review_document,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_PROTOCOL_SHA256,
    validate_t2_protocol_document,
)

T2_RETENTION_DECISION_NAME: Final = "T2_LONGITUDINAL_TEMPORAL_RETENTION_DECISION_V1"
T2_RETENTION_DECISION_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{T2_RETENTION_DECISION_NAME}.md"
)
T2_RETENTION_DECISION_SHA256: Final = (
    "4846921135b0ac83ceb40a0db063c2e4a3b2520971f279abe4f0c517c4f7dd20"
)

# ---------------------------------------------------------------------------
# The retained arm and its immutable comparator
# ---------------------------------------------------------------------------
T2_RETAINED_ARM: Final = T2_ARM_S4D
T2_COMPARATOR_ARM: Final = T2_ARM_GRU

T2_RETAINED_CHECKPOINT_SHA256: Final = (
    "63ccfbe00c209f94124610f1a22b25d84a2ad2b7e941ecaa3f0c8e9684a6722e"
)
T2_RETAINED_CHECKPOINT_LOCK_SHA256: Final = (
    "a9807515736abfeb9bcc34a3d98a8bdc766b1bae73a53eb3c2a8acc38259f8c7"
)
T2_RETAINED_CHECKPOINT_LOCK_SELF_SHA256: Final = (
    "a51ad25ed6cbef266b282954097e623b9666f82e0b25e84f7ced175cf46f5139"
)
T2_RETAINED_INTERNAL_DEV_THRESHOLD: Final = 0.8972153067588806

T2_COMPARATOR_CHECKPOINT_SHA256: Final = (
    "027048c5b3fedb13d1c695f2550b352ff81d447fc2f4dc4bbbb617dd420fa82b"
)
T2_COMPARATOR_CHECKPOINT_LOCK_SHA256: Final = (
    "61c5091125060c90ff52b51a3c8c3f0673688845787f2449578fe2057d1274ad"
)
T2_COMPARATOR_CHECKPOINT_LOCK_SELF_SHA256: Final = (
    "fab35e12016b8a2d10dd3ba29eca4d9c2df05af83fd40e85d653f996183bd9a5"
)
T2_COMPARATOR_INTERNAL_DEV_THRESHOLD: Final = 0.8328019380569458

# ---------------------------------------------------------------------------
# The one-shot outer attempt the decision is bound to
# ---------------------------------------------------------------------------
T2_OUTER_AUTHORIZED_GIT_SHA: Final = "b0f189a57bea8bd28884e7e40be50136fd6e2927"
T2_OUTER_RESULT_SHA256: Final = (
    "c58ed40dac753157b00ce6c70eb52fe903ecee72a5ef84e40932c1a80e259dbf"
)
T2_OUTER_EXPERIMENT_LOCK_FILE_SHA256: Final = (
    "54a0ca54736097ceb326dc968b3e58f8036f4867394d27df90df3e5da184a68c"
)
T2_OUTER_EXPERIMENT_LOCK_SELF_SHA256: Final = (
    "f90b93afc6ba94d76441eb789de924c1256c76d03c8dd8c4eea22014e4c65d9c"
)
T2_ROW_EVIDENCE_MANIFEST_SHA256: Final = (
    "c76453b8970a06c6beb3c280ab6e0518fa4cf81fcb304f6f9aa9c569d2634949"
)
T2_ROW_EVIDENCE_CONTENT_SHA256: Final = (
    "2240ca683fbcb790609c47f4a82af85250abb281fbbb9751dc74607a4eb591ca"
)
T2_VALIDATION_ROW_COUNT: Final = 492_904

T2_TRAINING_RESULT_SHA256: Final = (
    "ff9258f95631405b6705811d638d754400a067be4c1a43bb9d52021bb246adb8"
)
T2_TRAINING_EXPERIMENT_LOCK_SELF_SHA256: Final = (
    "d8de03554931fe65a6f1c1242d80c1c95f1a6a26f93b8013cff5bc221a92202f"
)

# ---------------------------------------------------------------------------
# What the canonical selector decided, recorded rather than recomputed
# ---------------------------------------------------------------------------
T2_SELECTION_BASIS: Final = "pooled_primary_validation_auprc"
T2_SELECTION_STAGE: Final = "stage_1_pooled_primary_validation_auprc"
T2_SELECTION_TIE_TOLERANCE: Final = 0.002
T2_COMPARATOR_POOLED_AUPRC: Final = 0.29486969381230116
T2_RETAINED_POOLED_AUPRC: Final = 0.388084635785268
T2_POOLED_AUPRC_DIFFERENCE: Final = 0.09321494197296681

T2_SELECTION_USED_TRAIN_EVIDENCE: Final = False
T2_SELECTION_USED_CHALLENGE_EVIDENCE: Final = False
T2_SELECTION_USED_LATENCY: Final = False
T2_SELECTION_WEIGHTED_SCORE_USED: Final = False
T2_RETENTION_STATISTICAL_SIGNIFICANCE_CLAIM: Final = False

# ---------------------------------------------------------------------------
# Governance the decision freezes
# ---------------------------------------------------------------------------
T2_RETAINED_THRESHOLD_IS_T1_POLICY: Final = False
T2_RETAINED_THRESHOLD_MAY_SELECT_T1_STATE: Final = False
T2_RERUN_PERMITTED: Final = False
T2_EXTENDED_TRAINING_PERMITTED: Final = False
T2_OUTER_VALIDATION_IS_DEVELOPMENT_EVIDENCE: Final = True
T2_OUTER_VALIDATION_IS_UNSEEN_GENERALIZATION: Final = False
T2_SEALED_TEST_STATE: Final = "unopened"


class T2SelectionError(RuntimeError):
    """Raised when the retained T2 arm cannot be proven."""


def validate_t2_retention_decision(
    path: Path = T2_RETENTION_DECISION_PATH,
) -> str:
    """Verify the frozen retention decision document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise T2SelectionError(f"T2 retention decision is missing at {document}.")
    digest = sha256_file(document)
    if digest != T2_RETENTION_DECISION_SHA256:
        raise T2SelectionError(
            f"T2 retention decision digest {digest} differs from the frozen "
            f"{T2_RETENTION_DECISION_SHA256}. The decision is immutable."
        )
    return digest


def _require_identity(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise T2SelectionError(
            f"{label} is {observed!r}, but the retention decision binds "
            f"{expected!r}. The retained T2 evidence is immutable."
        )


def validate_retained_t2_arm(run_root: Path) -> dict[str, Any]:
    """Prove the retained T2 arm against the frozen one-shot outer evidence.

    Read-only. The canonical verifier does the byte-level work -- outer result,
    outer lock, row-evidence manifest and arrays, the referenced TRAIN attempt,
    both checkpoints and both checkpoint locks -- and that proof is reused here
    rather than re-derived beside it.

    What this adds is the retention layer the verifier has no opinion about: the
    arm a human retained must be the arm the canonical selector chose, the
    comparator must still be present, and the bound identities must be exactly
    the ones the decision document names. Selection itself is never recomputed.
    """
    root = Path(run_root)
    outer = validate_canonical_t2_outer_validation_attempt(
        root, T2_OUTER_VALIDATION_ATTEMPT_ID
    )
    if outer.get("verified") is not True:
        raise T2SelectionError(
            "The canonical outer-VALIDATION attempt did not verify; the "
            "retained arm cannot rest on it."
        )

    # The canonical selector's decision, and the human decision, must be one
    # arm. A retention that disagreed with the frozen rule would be a second,
    # informal selection procedure.
    canonical_arm = outer.get("selected_arm")
    _require_identity("The canonical selected arm", canonical_arm, T2_RETAINED_ARM)
    if canonical_arm == T2_COMPARATOR_ARM:
        raise T2SelectionError(
            "The canonical attempt selected the comparator arm; the retention "
            "decision cannot silently switch T2 to it."
        )

    _require_identity(
        "The outer result digest", outer.get("result_sha256"), T2_OUTER_RESULT_SHA256
    )
    _require_identity(
        "The outer experiment-lock self-digest",
        outer.get("experiment_lock_sha256"),
        T2_OUTER_EXPERIMENT_LOCK_SELF_SHA256,
    )
    _require_identity(
        "The row-evidence store digest",
        outer.get("row_evidence_store_sha256"),
        T2_ROW_EVIDENCE_CONTENT_SHA256,
    )

    checkpoints = dict(outer.get("checkpoint_sha256") or {})
    locks = dict(outer.get("checkpoint_lock_sha256") or {})
    for arm, checkpoint, lock in (
        (
            T2_RETAINED_ARM,
            T2_RETAINED_CHECKPOINT_SHA256,
            T2_RETAINED_CHECKPOINT_LOCK_SHA256,
        ),
        (
            T2_COMPARATOR_ARM,
            T2_COMPARATOR_CHECKPOINT_SHA256,
            T2_COMPARATOR_CHECKPOINT_LOCK_SHA256,
        ),
    ):
        if arm not in checkpoints or arm not in locks:
            raise T2SelectionError(
                f"Arm {arm!r} is absent from the immutable outer evidence. Both "
                "arms are retained evidence: one as the T2 component, one as the "
                "comparator."
            )
        _require_identity(f"The {arm} checkpoint digest", checkpoints[arm], checkpoint)
        _require_identity(f"The {arm} checkpoint-lock digest", locks[arm], lock)

    training = dict(outer.get("training_attempt_verification") or {})
    _require_identity(
        "The referenced TRAIN result digest",
        training.get("result_sha256"),
        T2_TRAINING_RESULT_SHA256,
    )
    _require_identity(
        "The referenced TRAIN experiment-lock self-digest",
        training.get("experiment_lock_sha256"),
        T2_TRAINING_EXPERIMENT_LOCK_SELF_SHA256,
    )

    attempt = root / T2_OUTER_VALIDATION_ATTEMPT_ID
    _require_identity(
        "The outer experiment-lock file digest",
        sha256_file(attempt / OUTER_LOCK_NAME),
        T2_OUTER_EXPERIMENT_LOCK_FILE_SHA256,
    )
    manifest_path = attempt / "row_evidence" / T2_OUTER_STORE_MANIFEST_NAME
    _require_identity(
        "The row-evidence manifest digest",
        sha256_file(manifest_path),
        T2_ROW_EVIDENCE_MANIFEST_SHA256,
    )

    result = json.loads((attempt / OUTER_RESULT_NAME).read_text())

    # A checkpoint lock has two distinct digests: the bytes of the lock file,
    # already bound above through the canonical verifier, and the lock's own
    # self-digest computed over its content. Binding only the first would leave
    # the second free to drift, so the promoted result's record of it is
    # compared directly rather than left as a decorative constant.
    lock_self = dict(result.get("checkpoint_lock_self_sha256") or {})
    for arm, expected_self in (
        (T2_RETAINED_ARM, T2_RETAINED_CHECKPOINT_LOCK_SELF_SHA256),
        (T2_COMPARATOR_ARM, T2_COMPARATOR_CHECKPOINT_LOCK_SELF_SHA256),
    ):
        if arm not in lock_self:
            raise T2SelectionError(
                f"The outer result records no checkpoint-lock self-digest for {arm!r}."
            )
        _require_identity(
            f"The {arm} checkpoint-lock self-digest", lock_self[arm], expected_self
        )

    # The frozen governing documents are verified through their own existing
    # validators, so this module never re-derives a digest rule of its own.
    protocol_sha = validate_t2_protocol_document()
    execution_spec_sha = validate_t2_execution_spec()
    train_review_sha = validate_t2_train_artifact_review_document()
    _require_identity(
        "The outer result's protocol identity",
        result.get("t2_protocol_sha256"),
        protocol_sha,
    )
    _require_identity(
        "The outer result's execution-spec identity",
        result.get("t2_execution_spec_sha256"),
        execution_spec_sha,
    )
    _require_identity(
        "The bound TRAIN-artifact review identity",
        train_review_sha,
        T2_TRAIN_ARTIFACT_REVIEW_SHA256,
    )

    manifest = json.loads(manifest_path.read_text())
    persisted = list(manifest.get("arms_persisted") or [])
    for arm in (T2_RETAINED_ARM, T2_COMPARATOR_ARM):
        if arm not in persisted:
            raise T2SelectionError(
                f"The row-evidence store does not persist {arm!r}; T1 would have "
                "to rerun T2 to obtain it."
            )
    _require_identity(
        "The row-evidence score semantics",
        manifest.get("score_semantics"),
        T2_SCORE_SEMANTICS,
    )
    for claim in (
        "score_is_calibrated_probability",
        "score_is_confidence",
        "score_is_uncertainty",
    ):
        if manifest.get(claim) is not False:
            raise T2SelectionError(
                f"The row-evidence store records {claim}=True. The retained T2 "
                "object is an uncalibrated temporal model score."
            )

    if result.get("test_accessed") is not False:
        raise T2SelectionError("The outer result records TEST access.")
    if result.get("sealed_test_state") != T2_SEALED_TEST_STATE:
        raise T2SelectionError("The outer result does not record TEST as unopened.")
    _require_identity(
        "The outer authorized commit",
        result.get("authorized_git_sha"),
        T2_OUTER_AUTHORIZED_GIT_SHA,
    )

    return {
        "decision_class": "t2_longitudinal_temporal_retention_decision",
        "retention_decision_sha256": validate_t2_retention_decision(),
        "retained_arm": T2_RETAINED_ARM,
        "comparator_arm": T2_COMPARATOR_ARM,
        "retained": {T2_ARM_GRU: False, T2_ARM_S4D: True},
        "score_semantics": T2_SCORE_SEMANTICS,
        "score_definition": T2_SCORE_DEFINITION,
        "row_evidence_store_sha256": T2_ROW_EVIDENCE_CONTENT_SHA256,
        "validation_row_count": T2_VALIDATION_ROW_COUNT,
        "supports_t1_without_rerunning_outer_validation": True,
        "threshold_is_t1_policy": T2_RETAINED_THRESHOLD_IS_T1_POLICY,
        "threshold_may_select_t1_state": T2_RETAINED_THRESHOLD_MAY_SELECT_T1_STATE,
        "retained_internal_dev_threshold": T2_RETAINED_INTERNAL_DEV_THRESHOLD,
        "comparator_internal_dev_threshold": T2_COMPARATOR_INTERNAL_DEV_THRESHOLD,
        "selection_basis": T2_SELECTION_BASIS,
        "selection_stage": T2_SELECTION_STAGE,
        "pooled_auprc_difference": T2_POOLED_AUPRC_DIFFERENCE,
        "selection_used_train_evidence": T2_SELECTION_USED_TRAIN_EVIDENCE,
        "selection_used_challenge_evidence": T2_SELECTION_USED_CHALLENGE_EVIDENCE,
        "selection_used_latency": T2_SELECTION_USED_LATENCY,
        "weighted_score_used": T2_SELECTION_WEIGHTED_SCORE_USED,
        "statistical_significance_claim": (T2_RETENTION_STATISTICAL_SIGNIFICANCE_CLAIM),
        "outer_validation_is_development_evidence": (
            T2_OUTER_VALIDATION_IS_DEVELOPMENT_EVIDENCE
        ),
        "outer_validation_is_unseen_generalization": (
            T2_OUTER_VALIDATION_IS_UNSEEN_GENERALIZATION
        ),
        "t2_rerun_permitted": T2_RERUN_PERMITTED,
        "t2_extended_training_permitted": T2_EXTENDED_TRAINING_PERMITTED,
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "t2_train_artifact_review_sha256": T2_TRAIN_ARTIFACT_REVIEW_SHA256,
        "protocol_document_sha256": protocol_sha,
        "execution_spec_document_sha256": execution_spec_sha,
        "train_artifact_review_document_sha256": train_review_sha,
        "checkpoint_lock_self_sha256": {
            T2_RETAINED_ARM: T2_RETAINED_CHECKPOINT_LOCK_SELF_SHA256,
            T2_COMPARATOR_ARM: T2_COMPARATOR_CHECKPOINT_LOCK_SELF_SHA256,
        },
        "outer_authorized_git_sha": T2_OUTER_AUTHORIZED_GIT_SHA,
        "test_accessed": False,
        "sealed_test_state": T2_SEALED_TEST_STATE,
    }
