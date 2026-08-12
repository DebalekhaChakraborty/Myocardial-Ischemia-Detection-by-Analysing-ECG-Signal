"""Read-only binding of the FROZEN M2-v1 contamination-safe gate constants.

Every constant here was derived once, prospectively, from TRAIN data only and
is recorded in `docs/M2_GATE_DERIVATION_RECEIPT_V1.json`. This module records
those decisions and validates their documents. It contains no replay, no
training, no scoring and no mutation: M2 scientific implementation is a separate,
not-yet-authorized step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

M2_PROTOCOL_NAME: Final = "M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1"
M2_PROTOCOL_PATH: Final = REPOSITORY_ROOT / "docs" / f"{M2_PROTOCOL_NAME}.md"
M2_PROTOCOL_SHA256: Final = (
    "9d0a635e5b954d1334a78ac327d0190e41a62de7abf570b79446a5110ff53436"
)

M2_GATE_RECEIPT_PATH: Final = (
    REPOSITORY_ROOT / "docs" / "M2_GATE_DERIVATION_RECEIPT_V1.json"
)
M2_GATE_RECEIPT_SHA256: Final = (
    "3befd05dc7e9c51ddfed99078d3020375fd610b328d19e64fc7ee3cc745f398e"
)

M2_RETAINED_EXPERIMENT_ID: Final = "M1L_long_memory_v2"

# --- G3: waveform SQI, TRAIN-only Q99 upper bounds ------------------------
# Order is part of the frozen identity. Amplitude/rhythm features are
# deliberately excluded: they vary legitimately with patient physiology, and G3
# screens artifact/noise rather than selecting a physiological phenotype.
G3_SQI_COLUMNS: Final = (
    "flatline_fraction",
    "repeated_value_fraction",
    "derivative_outlier_fraction",
    "high_frequency_power_ratio",
    "powerline_ratio_50hz",
    "powerline_ratio_60hz",
)
G3_EXCLUDED_COLUMNS: Final = (
    "robust_amplitude_range_mv",
    "robust_derivative_scale_mv_per_s",
)
G3_FINITE_PRECONDITION_COLUMN: Final = "finite_sample_fraction"
G3_QUANTILE: Final = 0.99
G3_QUANTILE_METHOD: Final = "linear"
G3_UPPER_BOUNDS: Final = {
    "flatline_fraction": 0.4853941576630652,
    "repeated_value_fraction": 0.4853941576630652,
    "derivative_outlier_fraction": 0.12404961984793918,
    "high_frequency_power_ratio": 0.026922298961394597,
    "powerline_ratio_50hz": 0.0017282393761769012,
    "powerline_ratio_60hz": 0.0012844103306429878,
}
# flatline_fraction and repeated_value_fraction are bitwise identical in the
# frozen corpus, so these six columns impose five independent constraints.
G3_INDEPENDENT_CONSTRAINTS: Final = 5

# --- G4: deterministic normal-evidence margin -----------------------------
G4_DERIVATION_QUANTILE: Final = 0.50
G4_QUANTILE_METHOD: Final = "linear"
NORMAL_EVIDENCE_THRESHOLD: Final = 0.0002997174742631614
M1L_CLASSIFICATION_THRESHOLD: Final = 0.7554003000259399
G4_SCORE_SEMANTICS: Final = (
    "uncalibrated model score; not a probability, confidence, uncertainty or "
    "conformal score"
)

# --- G5: memory-update safety refractory ----------------------------------
REFRACTORY_DURATION_SECONDS: Final = 60.0
REFRACTORY_IS_RE_ARMABLE: Final = True
REFRACTORY_COUNTED_IN_UPDATES: Final = False
REFRACTORY_SEMANTICS: Final = (
    "memory-update safety refractory; NOT NORMAL/WATCH/EVENT/RECOVERY, not "
    "episode reasoning, not clinical persistence logic"
)

# --- G6: morphology computability admission -------------------------------
G6_MORPHOLOGY_INCLUDED: Final = True
G6_COLUMN: Final = "morphology_valid"
G6_NAME: Final = "morphology computability admission"
G6_ARMS_REFRACTORY: Final = False

# --- scope ----------------------------------------------------------------
M2_CORE_ARMS: Final = ("M2-0", "M2-G")
M2_ROLLBACK_IN_CORE: Final = False
M2_VALIDATION_ACCESSED_IN_DERIVATION: Final = False
M2_TEST_ACCESSED: Final = False


class M2GateError(RuntimeError):
    """Raised when a frozen M2 gate identity cannot be proven."""


def validate_m2_protocol(path: Path = M2_PROTOCOL_PATH) -> str:
    """Verify the frozen M2-v1 protocol byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise M2GateError(f"Frozen M2 protocol is missing at {document}.")
    digest = sha256_file(document)
    if digest != M2_PROTOCOL_SHA256:
        raise M2GateError(
            f"M2 protocol digest {digest} differs from the frozen "
            f"{M2_PROTOCOL_SHA256}. The protocol is immutable."
        )
    return digest


def validate_m2_gate_receipt(path: Path = M2_GATE_RECEIPT_PATH) -> str:
    """Verify the TRAIN-only derivation receipt byte-for-byte."""
    document = Path(path)
    if not document.is_file():
        raise M2GateError(f"M2 gate derivation receipt is missing at {document}.")
    digest = sha256_file(document)
    if digest != M2_GATE_RECEIPT_SHA256:
        raise M2GateError(
            f"M2 gate receipt digest {digest} differs from the frozen "
            f"{M2_GATE_RECEIPT_SHA256}."
        )
    return digest


def m2_gate_identity() -> dict[str, Any]:
    """The complete frozen gate, with its documents proven."""
    if NORMAL_EVIDENCE_THRESHOLD >= M1L_CLASSIFICATION_THRESHOLD:
        raise M2GateError(
            "The normal-evidence margin must be strictly below the M1L "
            "classification threshold; they are different decisions."
        )
    if set(G3_UPPER_BOUNDS) != set(G3_SQI_COLUMNS):
        raise M2GateError("G3 bounds do not match the frozen column set.")
    return {
        "m2_protocol_sha256": validate_m2_protocol(),
        "m2_gate_receipt_sha256": validate_m2_gate_receipt(),
        "retained_experiment_id": M2_RETAINED_EXPERIMENT_ID,
        "g3_columns": list(G3_SQI_COLUMNS),
        "g3_upper_bounds": dict(G3_UPPER_BOUNDS),
        "g3_independent_constraints": G3_INDEPENDENT_CONSTRAINTS,
        "g4_normal_evidence_threshold": NORMAL_EVIDENCE_THRESHOLD,
        "g4_score_semantics": G4_SCORE_SEMANTICS,
        "g5_refractory_seconds": REFRACTORY_DURATION_SECONDS,
        "g5_semantics": REFRACTORY_SEMANTICS,
        "g6_morphology_included": G6_MORPHOLOGY_INCLUDED,
        "core_arms": list(M2_CORE_ARMS),
        "rollback_in_core": M2_ROLLBACK_IN_CORE,
        "validation_accessed_in_derivation": M2_VALIDATION_ACCESSED_IN_DERIVATION,
        "test_accessed": M2_TEST_ACCESSED,
    }
