"""Immutable binding of the human M1 memory-architecture retention decision.

This module records a *decision*, not a computation. It validates the frozen
M1-v2 Stage-1 suite and the retained arm lock and refuses anything else. It
deliberately contains no training, no scoring and no mutation: the M1 execution
machinery is untouched, and reading this module can never alter an artifact.

The non-retained arms are bound too. They remain immutable ablation evidence,
so a later phase can prove it is not silently reusing one of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.patient_memory import (
    M1_PROTOCOL_SHA256,
    M1D_EXPERIMENT_ID,
    M1L_EXPERIMENT_ID,
    M1S_EXPERIMENT_ID,
    REPOSITORY_ROOT,
)

M1_RETENTION_DECISION_NAME: Final = "M1_MEMORY_RETENTION_DECISION_V1"
M1_RETENTION_DECISION_PATH: Final = (
    REPOSITORY_ROOT / "docs" / f"{M1_RETENTION_DECISION_NAME}.md"
)
M1_RETENTION_DECISION_SHA256: Final = (
    "45b29cd83ecfc60b43639be5569075a9cf561650f58a9812ade3051467f11b51"
)

M1_STAGE1_SUITE_SHA256: Final = (
    "be36f0743dad649756626a981c3dd05ec6f54dc9c01150e70bb3caeb407bac0e"
)

M1_RETAINED_EXPERIMENT_ID: Final = M1L_EXPERIMENT_ID
M1_RETAINED_LOCK_SHA256: Final = (
    "a2636855e14bdd54ff3b0a17f238579d097366bb64761e723003b6d6a13c75a5"
)
M1_RETAINED_CHECKPOINT_SHA256: Final = (
    "a26b6a18db8c005a051054417156068174a166062a5498f32fd48e473ad58510"
)

# Frozen ablation evidence: measured, reported, and NOT selected for M2.
M1_ABLATION_LOCK_SHA256: Final = {
    M1S_EXPERIMENT_ID: (
        "e9fd43f7920686c8f14cdf3da7ca2e2a5e6553289c638263e9c57e54be593a65"
    ),
    M1D_EXPERIMENT_ID: (
        "2d08ffbbbb3fcd962f3abec99d7b2f97823b6ccaafb85fa681dc05363af1a3c1"
    ),
}

M1_SELECTION_BASIS: Final = "development_evidence_only"
M1_SELECTION_TEST_ACCESSED: Final = False
M1_SELECTION_WEIGHTED_SCORE_USED: Final = False
M1_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM: Final = False
M1_RERUN_PERMITTED: Final = False


class M1SelectionError(RuntimeError):
    """Raised when the retained M1 identity cannot be proven."""


def validate_m1_retention_decision(
    path: Path = M1_RETENTION_DECISION_PATH,
) -> str:
    """Verify the frozen retention decision document byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise M1SelectionError(f"M1 retention decision is missing at {document}.")
    digest = sha256_file(document)
    if digest != M1_RETENTION_DECISION_SHA256:
        raise M1SelectionError(
            f"M1 retention decision digest {digest} differs from the frozen "
            f"{M1_RETENTION_DECISION_SHA256}. The decision is immutable."
        )
    return digest


def validate_retained_m1_arm(run_root: Path) -> dict[str, Any]:
    """Prove the retained arm against the frozen suite and lock.

    Read-only: no artifact is written, and the non-retained arms are checked
    only to confirm they still carry their frozen ablation identities.
    """
    root = Path(run_root)
    suite_path = root / "M1_STAGE1_RESULTS.json"
    if not suite_path.is_file():
        raise M1SelectionError(f"No M1 Stage-1 result at {suite_path}.")
    suite = json.loads(suite_path.read_text())
    if suite.get("m1_stage1_suite_sha256") != M1_STAGE1_SUITE_SHA256:
        raise M1SelectionError(
            "The M1 Stage-1 suite is not the one the retention decision binds."
        )
    if suite.get("memory_selection_performed") is not False:
        raise M1SelectionError(
            "The canonical suite must continue to record that it performed no "
            "selection; the decision lives in the governance document."
        )
    if suite.get("test_accessed") is not False:
        raise M1SelectionError("The M1 Stage-1 suite records test access.")

    lock_path = root / M1_RETAINED_EXPERIMENT_ID / "EXPERIMENT_LOCK.json"
    if not lock_path.is_file():
        raise M1SelectionError(f"No retained arm lock at {lock_path}.")
    lock = json.loads(lock_path.read_text())
    if lock.get("experiment_lock_sha256") != M1_RETAINED_LOCK_SHA256:
        raise M1SelectionError(
            "The retained arm lock differs from the frozen retention identity."
        )
    if lock.get("m1_protocol_sha256") != M1_PROTOCOL_SHA256:
        raise M1SelectionError("The retained arm does not bind the M1-v2 protocol.")
    if lock.get("test") is not None or lock.get("test_accessed") is not False:
        raise M1SelectionError("The retained arm lock records test evidence.")

    for arm, expected in M1_ABLATION_LOCK_SHA256.items():
        ablation = json.loads((root / arm / "EXPERIMENT_LOCK.json").read_text())
        if ablation.get("experiment_lock_sha256") != expected:
            raise M1SelectionError(
                f"Frozen ablation arm {arm} no longer carries its recorded lock."
            )

    return {
        "retention_decision_sha256": validate_m1_retention_decision(),
        "m1_stage1_suite_sha256": M1_STAGE1_SUITE_SHA256,
        "m1_protocol_sha256": M1_PROTOCOL_SHA256,
        "retained_experiment_id": M1_RETAINED_EXPERIMENT_ID,
        "retained_lock_sha256": M1_RETAINED_LOCK_SHA256,
        "retained_checkpoint_sha256": M1_RETAINED_CHECKPOINT_SHA256,
        "ablation_lock_sha256": dict(M1_ABLATION_LOCK_SHA256),
        "retained": {
            M1S_EXPERIMENT_ID: False,
            M1L_EXPERIMENT_ID: True,
            M1D_EXPERIMENT_ID: False,
        },
        "selection_basis": M1_SELECTION_BASIS,
        "test_accessed": M1_SELECTION_TEST_ACCESSED,
        "weighted_score_used": M1_SELECTION_WEIGHTED_SCORE_USED,
        "statistical_significance_claim": (
            M1_SELECTION_STATISTICAL_SIGNIFICANCE_CLAIM
        ),
        "m1_rerun_permitted": M1_RERUN_PERMITTED,
    }
