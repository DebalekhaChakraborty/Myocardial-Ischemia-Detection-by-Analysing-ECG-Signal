"""Immutable binding of the human M2 update-policy retention decision.

This module records a *decision*, not a computation. It validates the frozen
recovery2 suite, the retained arm's promoted result and lock, and the control
arm's frozen evidence, and refuses anything else. It deliberately contains no
replay, no scoring, no gate evaluation and no mutation: the M2 execution
machinery is untouched, and reading this module can never alter an artifact.

The control arm is bound too. `M2-0` remains immutable ablation evidence, so a
later phase can prove it is not silently reusing the naive update policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.m2_gate import M2_GATE_RECEIPT_SHA256, M2_PROTOCOL_SHA256
from cardiosentinel.neural.m2_persistence import (
    ARM_RESULT_NAME,
    EXPERIMENT_LOCK_NAME,
    SUITE_RESULT_NAME,
    arm_experiment_id,
    suite_directory,
)
from cardiosentinel.neural.m2_policy import M2_ARM_GATED, M2_ARM_NAIVE
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

M2_RETENTION_DECISION_NAME: Final = "M2_UPDATE_POLICY_RETENTION_DECISION_V1"
M2_RETENTION_DECISION_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{M2_RETENTION_DECISION_NAME}.md"
)
M2_RETENTION_DECISION_SHA256: Final = (
    "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
)

M2_RETAINED_SUITE_ID: Final = "m2-v1-development-two-arm-recovery2"
M2_SUITE_SHA256: Final = (
    "8a6b0a1c64da72fc0f4573c742bef491b01b4eb8179f0759da3c537a01939a02"
)

M2_RETAINED_ARM: Final = M2_ARM_GATED
M2_RETAINED_ARM_RESULT_SHA256: Final = (
    "a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75"
)
M2_RETAINED_LOCK_SHA256: Final = (
    "5ac07d9f1ea3859e046c84fb91f22cee1bb20ef4857837b1c82fcd944dbf0fe8"
)

# Frozen control/ablation evidence: measured, reported, and NOT retained.
M2_CONTROL_ARM: Final = M2_ARM_NAIVE
M2_CONTROL_ARM_RESULT_SHA256: Final = (
    "37a6e9d4c01b823e407addbb897d14f6f54835a347a6557a745890585395644c"
)
M2_CONTROL_LOCK_SHA256: Final = (
    "8f7109494efc243613046dd57bcdece80491cf6897c867e224235eb8480c1461"
)

M2_SELECTION_BASIS: Final = "development_evidence_only"
M2_SELECTION_TEST_ACCESSED: Final = False
M2_SELECTION_WEIGHTED_SCORE_USED: Final = False
M2_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM: Final = False
M2_RERUN_PERMITTED: Final = False


class M2SelectionError(RuntimeError):
    """Raised when the retained M2 update policy cannot be proven."""


def validate_m2_retention_decision(
    path: Path = M2_RETENTION_DECISION_PATH,
) -> str:
    """Verify the frozen retention decision document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise M2SelectionError(f"M2 retention decision is missing at {document}.")
    digest = sha256_file(document)
    if digest != M2_RETENTION_DECISION_SHA256:
        raise M2SelectionError(
            f"M2 retention decision digest {digest} differs from the frozen "
            f"{M2_RETENTION_DECISION_SHA256}. The decision is immutable."
        )
    return digest


def _require_arm_artifacts(
    run_root: Path, arm: str, *, result_sha256: str, lock_sha256: str
) -> None:
    """Prove one arm's promoted result and lock against the frozen digests."""
    run_dir = Path(run_root) / arm_experiment_id(M2_RETAINED_SUITE_ID, arm)
    result_path = run_dir / ARM_RESULT_NAME
    lock_path = run_dir / EXPERIMENT_LOCK_NAME
    for path in (result_path, lock_path):
        if not path.is_file():
            raise M2SelectionError(f"Arm {arm} is missing {path.name} at {path}.")

    observed_result = sha256_file(result_path)
    if observed_result != result_sha256:
        raise M2SelectionError(
            f"Arm {arm} result digests to {observed_result}, but the retention "
            f"decision binds {result_sha256}. The evidence is immutable."
        )

    lock = json.loads(lock_path.read_text())
    if lock.get("experiment_lock_sha256") != lock_sha256:
        raise M2SelectionError(
            f"Arm {arm} lock differs from the frozen retention identity."
        )
    if lock.get("arm") != arm:
        raise M2SelectionError(f"Arm {arm} lock records arm {lock.get('arm')!r}.")
    if lock.get("test_accessed") is not False:
        raise M2SelectionError(f"Arm {arm} lock records test access.")


def validate_retained_m2_arm(run_root: Path) -> dict[str, Any]:
    """Prove the retained update policy against the frozen suite and locks.

    Read-only: no artifact is written, and the control arm is checked only to
    confirm it still carries its frozen ablation identity.
    """
    root = Path(run_root)
    suite_path = suite_directory(root, M2_RETAINED_SUITE_ID) / SUITE_RESULT_NAME
    if not suite_path.is_file():
        raise M2SelectionError(f"No M2 suite result at {suite_path}.")
    suite = json.loads(suite_path.read_text())

    if suite.get("m2_suite_sha256") != M2_SUITE_SHA256:
        raise M2SelectionError(
            "The M2 suite is not the one the retention decision binds."
        )
    if suite.get("suite_id") != M2_RETAINED_SUITE_ID:
        raise M2SelectionError(
            f"The suite records id {suite.get('suite_id')!r}, not the retained "
            f"{M2_RETAINED_SUITE_ID!r}."
        )
    if suite.get("automatic_arm_preference_applied") is not False:
        raise M2SelectionError(
            "The canonical suite must continue to record that it applied no "
            "automatic arm preference; the decision lives in the governance "
            "document."
        )
    if suite.get("memory_selection_performed") is not False:
        raise M2SelectionError(
            "The canonical suite must continue to record that it performed no "
            "selection."
        )
    if suite.get("memory_selected") is not None:
        raise M2SelectionError("The canonical suite records a selected arm.")
    if suite.get("test_accessed") is not False:
        raise M2SelectionError("The M2 suite records test access.")
    if suite.get("sealed_test_state") != "unopened":
        raise M2SelectionError("The M2 suite does not record TEST as unopened.")

    _require_arm_artifacts(
        root,
        M2_RETAINED_ARM,
        result_sha256=M2_RETAINED_ARM_RESULT_SHA256,
        lock_sha256=M2_RETAINED_LOCK_SHA256,
    )
    _require_arm_artifacts(
        root,
        M2_CONTROL_ARM,
        result_sha256=M2_CONTROL_ARM_RESULT_SHA256,
        lock_sha256=M2_CONTROL_LOCK_SHA256,
    )

    return {
        "retention_decision_sha256": validate_m2_retention_decision(),
        "m2_suite_sha256": M2_SUITE_SHA256,
        "m2_protocol_sha256": M2_PROTOCOL_SHA256,
        "m2_gate_receipt_sha256": M2_GATE_RECEIPT_SHA256,
        "retained_suite_id": M2_RETAINED_SUITE_ID,
        "retained_arm": M2_RETAINED_ARM,
        "retained_arm_result_sha256": M2_RETAINED_ARM_RESULT_SHA256,
        "retained_lock_sha256": M2_RETAINED_LOCK_SHA256,
        "control_arm": M2_CONTROL_ARM,
        "control_arm_result_sha256": M2_CONTROL_ARM_RESULT_SHA256,
        "control_lock_sha256": M2_CONTROL_LOCK_SHA256,
        "retained": {
            M2_ARM_NAIVE: False,
            M2_ARM_GATED: True,
        },
        "selection_basis": M2_SELECTION_BASIS,
        "test_accessed": M2_SELECTION_TEST_ACCESSED,
        "weighted_score_used": M2_SELECTION_WEIGHTED_SCORE_USED,
        "statistical_significance_claim": (M2_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM),
        "m2_rerun_permitted": M2_RERUN_PERMITTED,
    }
