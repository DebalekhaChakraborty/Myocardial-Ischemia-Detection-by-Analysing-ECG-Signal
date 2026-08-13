"""The canonical frozen retained-M1L scorer adapter for M2 replay.

`m2_policy.replay_stream` takes an injected
`(raw_146d_representation, pre_update_d_long) -> score` callable. This module
binds the ONE real implementation of that callable: the frozen retained M1L
head, loaded read-only from its canonical run directory.

This is an adapter, not a second interpretation of M1L. The head is
constructed by `build_deterministic_m1_head` and its weights loaded from the
frozen `model_selected.pt`; the input is assembled by the frozen
`m1_arm_features`; the forward pass matches the canonical numerical path
established during M2 derivation verification. No fitting, no retraining, no
optimizer, no threshold selection happens here.

**Scorer input is exactly `[raw frozen 146-d z_t ; pre-update d_long(t)]`**, as
inherited from the retained M1L experiment -- 147 dimensions, in that order.
The standardized representation is NOT substituted (M1L was trained on the raw
fused vector; standardization belongs to the memory distance space only), the
post-update `d_long` is never used, and no G3/G4/G5/G6 quantity, patient
identity, subject identity, annotation state, uncertainty or temporal state is
appended.

**Two thresholds that must never be interchanged.** They are deliberately kept
in separate constants with separate names and separate meanings:

* `M1L_CLASSIFICATION_THRESHOLD` (`0.7554003000259399`) -- the frozen retained
  M1L operating point, used for classification/evaluation semantics only.
* `NORMAL_EVIDENCE_THRESHOLD` (`0.0002997174742631614`) -- the frozen M2 G4
  margin, used ONLY to decide whether patient memory may update.

Choosing an operating point for classification and deciding what is safe to
learn as normal are different problems. `assert_thresholds_are_distinct()`
exists so an accidental interchange fails loudly rather than silently changing
what the system learns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.m1_experiment import validate_m1_lock
from cardiosentinel.neural.m2_gate_derivation import M1L_INTRA_OP_THREADS
from cardiosentinel.neural.patient_memory import (
    M1L_EXPERIMENT_ID,
    REPRESENTATION_DIM,
    build_deterministic_m1_head,
    m1_arm_features,
    m1_input_dim,
)
from cardiosentinel.neural.physiology_fusion import P1_BATCH_SIZE

SELECTED_MODEL_NAME: Final = "model_selected.pt"

# The frozen retained identities this adapter refuses to run without.
RETAINED_M1L_LOCK_SHA256: Final = (
    "a2636855e14bdd54ff3b0a17f238579d097366bb64761e723003b6d6a13c75a5"
)
RETAINED_M1L_CHECKPOINT_SHA256: Final = (
    "a26b6a18db8c005a051054417156068174a166062a5498f32fd48e473ad58510"
)
FROZEN_P1B_LOCK_SHA256: Final = (
    "796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0"
)
FROZEN_B4B_CHECKPOINT_SHA256: Final = (
    "b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9"
)

# The frozen M1L operating point. Classification/evaluation semantics ONLY.
M1L_CLASSIFICATION_THRESHOLD: Final = 0.7554003000259399
# The frozen M2 G4 margin. Memory-update admission ONLY. Bound from m2_gate so
# there is exactly one definition of it in the codebase.
NORMAL_EVIDENCE_THRESHOLD: Final = GATE.NORMAL_EVIDENCE_THRESHOLD

M2_SCORER_INPUT_CONTRACT: Final = (
    "[raw frozen 146-d z_t ; pre-update d_long(t)] = 147 dimensions, in that "
    "order, exactly as inherited from the retained M1L experiment. The "
    "standardized representation is not substituted, post-update d_long is "
    "never used, and no gate quantity, identity, annotation, uncertainty or "
    "temporal state is appended."
)


class M2ScorerError(RuntimeError):
    """Raised when the canonical M1L scorer cannot be bound with full integrity."""


def assert_thresholds_are_distinct() -> None:
    """Refuse any build in which the two thresholds have become interchangeable."""
    if M1L_CLASSIFICATION_THRESHOLD == NORMAL_EVIDENCE_THRESHOLD:
        raise M2ScorerError(
            "The M1L classification threshold and the M2 normal-evidence "
            "margin have become equal. They answer different questions and "
            "must never be interchanged."
        )
    if not NORMAL_EVIDENCE_THRESHOLD < M1L_CLASSIFICATION_THRESHOLD:
        raise M2ScorerError(
            "The M2 normal-evidence margin must be strictly below the M1L "
            "classification threshold; admitting normal memory is deliberately "
            "the stricter of the two decisions."
        )
    if NORMAL_EVIDENCE_THRESHOLD != GATE.NORMAL_EVIDENCE_THRESHOLD:
        raise M2ScorerError("The M2 update threshold is not the frozen gate constant.")


def validate_retained_m1l_identity(run_dir: Path) -> dict[str, Any]:
    """Prove every frozen identity the scorer depends on, before loading it."""
    assert_thresholds_are_distinct()
    lock = validate_m1_lock(Path(run_dir))

    if lock["experiment_id"] != M1L_EXPERIMENT_ID:
        raise M2ScorerError(
            f"M2 binds the retained arm {M1L_EXPERIMENT_ID}, not "
            f"{lock['experiment_id']!r}."
        )
    if GATE.M2_RETAINED_EXPERIMENT_ID != M1L_EXPERIMENT_ID:
        raise M2ScorerError("The frozen M2 gate does not retain M1L.")
    for label, observed, expected in (
        ("retained M1L lock", lock["experiment_lock_sha256"], RETAINED_M1L_LOCK_SHA256),
        (
            "retained M1L checkpoint",
            lock["artifact_sha256"][SELECTED_MODEL_NAME],
            RETAINED_M1L_CHECKPOINT_SHA256,
        ),
        (
            "P1-B global control lock",
            lock["global_control_lock_sha256"],
            FROZEN_P1B_LOCK_SHA256,
        ),
        (
            "B4-B encoder checkpoint",
            lock["encoder_checkpoint_sha256"],
            FROZEN_B4B_CHECKPOINT_SHA256,
        ),
    ):
        if observed != expected:
            raise M2ScorerError(
                f"{label} is {observed!r}, expected the frozen {expected!r}."
            )
    if lock["representation_dim"] != REPRESENTATION_DIM:
        raise M2ScorerError(
            f"The retained lock declares representation_dim "
            f"{lock['representation_dim']}, expected {REPRESENTATION_DIM}."
        )
    if list(lock["memory_features"]) != ["d_long"]:
        raise M2ScorerError(
            f"The retained arm's memory features are {lock['memory_features']}, "
            "expected exactly ['d_long']."
        )
    if lock["head"]["input_dim"] != m1_input_dim(M1L_EXPERIMENT_ID):
        raise M2ScorerError("The retained head input dimension is not the frozen 147.")
    if float(lock["threshold"]) != M1L_CLASSIFICATION_THRESHOLD:
        raise M2ScorerError(
            f"The retained lock's classification threshold {lock['threshold']!r} "
            f"differs from the frozen {M1L_CLASSIFICATION_THRESHOLD!r}."
        )
    if lock["test_accessed"] is not False or lock["test_metrics"] is not None:
        raise M2ScorerError("The retained M1L lock records test access.")
    standardizer = dict(lock.get("distance_standardizer") or {})
    if standardizer.get("fitted_on_partition") != "train":
        raise M2ScorerError("The bound distance standardizer is not TRAIN-only.")
    return lock


class FrozenM1LScorer:
    """The canonical `(raw z_t, pre-update d_long) -> score` callable.

    Holds the frozen head read-only. There is no optimizer, no parameter is
    ever written, and the module is left in eval mode with gradients disabled.
    """

    __slots__ = ("_head", "_lock", "_state_digest")

    def __init__(self, run_dir: Path) -> None:
        lock = validate_retained_m1l_identity(run_dir)
        head = build_deterministic_m1_head(M1L_EXPERIMENT_ID)
        state = torch.load(
            Path(run_dir) / SELECTED_MODEL_NAME, map_location="cpu", weights_only=True
        )
        head.load_state_dict(state)
        head.eval()
        head.requires_grad_(False)
        self._head = head
        self._lock = lock
        self._state_digest = _state_dict_digest(head)

    @property
    def lock(self) -> dict[str, Any]:
        return dict(self._lock)

    @property
    def classification_threshold(self) -> float:
        """The frozen M1L operating point. NEVER the memory-update threshold."""
        return M1L_CLASSIFICATION_THRESHOLD

    def identity(self) -> dict[str, Any]:
        return {
            "retained_experiment_id": M1L_EXPERIMENT_ID,
            "retained_lock_sha256": RETAINED_M1L_LOCK_SHA256,
            "retained_checkpoint_sha256": RETAINED_M1L_CHECKPOINT_SHA256,
            "p1b_lock_sha256": FROZEN_P1B_LOCK_SHA256,
            "b4b_checkpoint_sha256": FROZEN_B4B_CHECKPOINT_SHA256,
            "input_dim": m1_input_dim(M1L_EXPERIMENT_ID),
            "input_contract": M2_SCORER_INPUT_CONTRACT,
            "classification_threshold": M1L_CLASSIFICATION_THRESHOLD,
            "classification_threshold_used_for_memory_admission": False,
            "memory_admission_threshold": NORMAL_EVIDENCE_THRESHOLD,
            "retrained": False,
            "fitted": False,
            "optimizer_constructed": False,
            "threshold_selected_here": False,
            "head_state_sha256": self._state_digest,
            "intra_op_threads": M1L_INTRA_OP_THREADS,
        }

    def assert_unmutated(self) -> None:
        """Prove the frozen weights did not move during scoring."""
        if _state_dict_digest(self._head) != self._state_digest:
            raise M2ScorerError(
                "The frozen retained M1L head changed during scoring. No M2 "
                "path may write to it."
            )

    def score_batch(
        self, representations: np.ndarray, d_long: np.ndarray
    ) -> np.ndarray:
        """Score a bounded block of `[raw z_t ; pre-update d_long]` rows."""
        base = np.asarray(representations, dtype=np.float64)
        memory = np.asarray(d_long, dtype=np.float64).reshape(-1, 1)
        features = m1_arm_features(M1L_EXPERIMENT_ID, base, memory)
        if features.shape[1] != m1_input_dim(M1L_EXPERIMENT_ID):
            raise M2ScorerError(
                f"Scorer input width {features.shape[1]} is not the frozen "
                f"{m1_input_dim(M1L_EXPERIMENT_ID)}."
            )
        outputs: list[np.ndarray] = []
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(M1L_INTRA_OP_THREADS)
        try:
            with torch.no_grad():
                for start in range(0, features.shape[0], P1_BATCH_SIZE):
                    chunk = torch.from_numpy(features[start : start + P1_BATCH_SIZE])
                    outputs.append(
                        torch.sigmoid(self._head(chunk)).to(torch.float64).numpy()
                    )
        finally:
            torch.set_num_threads(previous_threads)
        scores = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise M2ScorerError(
                "The frozen retained M1L head produced a non-finite score."
            )
        self.assert_unmutated()
        return scores

    def __call__(self, representation: np.ndarray, d_long: float) -> float:
        """The `M2Scorer` protocol: one row, raw z_t plus pre-update d_long."""
        vector = np.asarray(representation, dtype=np.float64)
        if vector.shape != (REPRESENTATION_DIM,):
            raise M2ScorerError(
                f"The M1L scorer expects a raw [{REPRESENTATION_DIM}] "
                f"representation, received {vector.shape}."
            )
        return float(self.score_batch(vector[None, :], np.asarray([d_long]))[0])


def _state_dict_digest(head: torch.nn.Module) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        array = tensor.detach().cpu().numpy()
        hasher.update(repr((array.shape, str(array.dtype))).encode("utf-8"))
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def load_frozen_m1l_scorer(run_root: Path) -> FrozenM1LScorer:
    """Bind the canonical scorer from the frozen M1 Stage-1 run root."""
    return FrozenM1LScorer(Path(run_root) / M1L_EXPERIMENT_ID)
