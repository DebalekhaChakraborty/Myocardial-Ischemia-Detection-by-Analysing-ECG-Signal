"""The ONE canonical M2-v1 DEVELOPMENT invocation.

This module is the only public route to a claim-bearing M2 development result.
It always runs both frozen arms, in the frozen order `M2-0` then `M2-G`, and it
never selects between them: no third arm exists, no single-arm scientific route
is exposed, and no automatic preference is applied anywhere.

**Nothing runs on import**, and `__main__` dispatch sits at the very end of the
file, after every helper the run needs is defined.

**One suite, two independent attempts.** A shared `suite_id` names the run;
each arm gets its own immutable claim directory `<suite_id>__M2-0` and
`<suite_id>__M2-G`. A single shared experiment id would make M2-0 claim the
directory and M2-G collide with it. The ids are deterministic — never random,
never timestamped, never auto-renamed on collision.

**Execution order** (frozen):

1. PRE-CLAIM READINESS — Git SHA and clean checkout, frozen runtime, M2
   protocol and gate-receipt digests, the stress-eligibility decision digest,
   the retained M1L lock/checkpoint, the P1-B lock, the B4-B checkpoint, the
   frozen TRAIN-only distance standardizer, the label firewall, the TEST
   firewall, and the pair-claim absence check. **Nothing that requires
   VALIDATION is opened here**; the scorer and standardizer are readied because
   neither does. Any failure means no arm claim, no VALIDATION access, no retry.
2. START / CLAIM — an independent `RuntimeIntegrityRecord` per arm, a
   successful START for BOTH, then M2-0 claimed and M2-G claimed. Only after
   BOTH claims succeed may VALIDATION be opened.
3. DEVELOPMENT SOURCE INTEGRITY — the raw `.hea`/`.dat`/`.stb` the stress
   selection later reads are proven against the official pinned manifest and
   the frozen feature-corpus identity, using the repository's existing
   verifiers. TEST files are never hashed.
4. FULL LABEL-BLIND REPLAY — the validation input is loaded exactly once and
   the canonical full replay identity is proven. Both arms replay the identical
   frozen rows with the identical frozen scorer, each keeping its own stream
   state. No annotation is loaded until both trajectories are complete, so no
   M2-0 result can alter M2-G's replay.
5. POST-REPLAY — only then: primary membership, challenge membership,
   source-defined stress intervals, identity-keyed joins, frozen evidence.
6. PERSIST — per arm, then one aggregating two-arm suite with its own
   PRE_PROMOTION observation.

**Bounded memory.** Replay proceeds stream by stream and per-row evidence is
folded into compact typed arrays and integer counters. Prototype trajectories
live in a disk-backed store and drift evaluation loads **one stream at a time**.
Scores, times and prototypes are float64 throughout, so
`sqrt(mean((mu_long(t) - mu_ref) ** 2))` is reproduced exactly.

**Execution history is never asserted by a source constant.** Whether a
canonical run has happened is read from the claim directories, run-status
files, experiment locks and suite result — see `canonical_execution_history()`.
Source code cannot rewrite itself, so a hard-coded "no run yet" boolean could
only ever become a lie.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.m2_execution import (
    CANONICAL_DEVELOPMENT_PARTITION,
    M2ExecutionError,
    assert_label_firewall,
    require_canonical_development_partition,
)
from cardiosentinel.neural.m2_policy import M2_ARMS

EXECUTION_FLAG: Final = "--execute-canonical-development"
EXPECTED_GIT_SHA_FLAG: Final = "--expected-git-sha"

CANONICAL_ARM_ORDER: Final = ("M2-0", "M2-G")
"""The frozen two-arm order. Both always run; neither is ever selected."""

ORIGINAL_SUITE_ID: Final = "m2-v1-development-two-arm"
"""Attempt #1. CONSUMED and FAILED before scoring; permanently preserved and
never re-run, re-cleaned or reused. See
`docs/M2_DEVELOPMENT_ATTEMPT1_FAILURE_AND_RECOVERY_DECISION_V1.md`."""

RECOVERY1_SUITE_ID: Final = "m2-v1-development-two-arm-recovery1"
"""Recovery1. ALSO consumed and failed before scoring, on a distinct defect:
the feature join treated a legitimate source null as an unwritten row. Preserved
permanently and never reused."""

CANONICAL_SUITE_ID: Final = "m2-v1-development-two-arm-recovery2"
"""The ONE permitted recovery2 identity, frozen prospectively before any further
development access. The earlier recovery decision prohibited an *implicit*
recovery2; this one was separately and explicitly authorized after human review
of a second, distinct pre-scoring execution defect. There is no recovery3,
attempt4, timestamp or random suffix: if this suite is ever claimed and fails,
execution STOPS FOR HUMAN REVIEW."""

RECOVERY_DECISION_DOCUMENT: Final = (
    "docs/M2_DEVELOPMENT_ATTEMPT1_FAILURE_AND_RECOVERY_DECISION_V1.md"
)
RECOVERY_DECISION_SHA256: Final = (
    "e9d55d7a047e9610c6e156afc9e1a98aafbca86a3131c02a8e56624da7ad57d6"
)
ATTEMPT1_REASON_CLASS: Final = "pre_scoring_partition_alignment_execution_defect"
RECOVERY1_REASON_CLASS: Final = "pre_scoring_source_null_join_sentinel_defect"

RECOVERY2_DECISION_DOCUMENT: Final = (
    "docs/M2_DEVELOPMENT_RECOVERY1_FAILURE_AND_RECOVERY2_DECISION_V1.md"
)
RECOVERY2_DECISION_SHA256: Final = (
    "93e53d3c8281d922823d48b73712a2a1ede1c5b0f5bc9f41694af563e1a2fca4"
)

# Recovery1's frozen forensic identity, captured read-only from its preserved
# artifacts before any code changed.
RECOVERY1_EXECUTION_GIT_SHA: Final = "d77fbdc37415c43728dbe3173ce58a85cfe2e71d"
RECOVERY1_STATUS_SHA256: Final = {
    "M2-0": "642cc8376c87826a5d7fdbd5d0730ca44b20f3429c26ea44c58974b45244d054",
    "M2-G": "8ba15ca25b70c7686b2e39fe3e073607511835ff42fa19b5ee4d9138f4a0170d",
}
RECOVERY1_FAILURE_RECEIPT_SHA256: Final = (
    "5b05873d48f1355292113a07d6025258e071cb9b13a35caaff1a10132cbb0408"
)
RECOVERY1_FAILURE_RECEIPT_FILE_SHA256: Final = (
    "7773c6135a22e7ba64699511e1db1e92c8aac1ec9b90727d7805f540d5156446"
)
RECOVERY1_FAILED_STAGE: Final = "full_label_blind_replay_both_arms"
RECOVERY1_EXCEPTION_TYPE: Final = "M2FeatureJoinError"
RECOVERY1_EXCEPTION_SUBSTRING: Final = "left unmatched rows for"

# --------------------------------------------------------------------------
# Frozen forensic identity of attempt #1
# --------------------------------------------------------------------------
#
# A claim directory alone proves nothing: it shows that SOMETHING was claimed,
# not that THIS attempt failed, that it failed before scoring, that no metric
# was produced, or that TEST stayed shut. Recovery therefore verifies the exact
# preserved artifacts against these frozen digests before it may run.

ORIGINAL_EXECUTION_GIT_SHA: Final = "3c1ba4ce87ade6a2d17386b3a9d2b579ded442e7"

ORIGINAL_STATUS_SHA256: Final = {
    "M2-0": "3699e656ee5ab6c6d3fba90dd7dd726cbb06233d478e53b81327f394e9f6365d",
    "M2-G": "7908130758cfffa171fe47f3958ee4ef7961bfe3d352486e1f2558862251a751",
}

ORIGINAL_FAILURE_RECEIPT_SHA256: Final = (
    "31345512e42578c1bac8ad611689501a49c40e5d7ab2d70f6c41508d9c4492eb"
)
ORIGINAL_FAILURE_RECEIPT_FILE_SHA256: Final = (
    "8c3a0734dcd2b3dd695e7ad3ff88933a48e80ec48adc3a42b759932cc07cb278"
)

ORIGINAL_FAILED_STAGE: Final = "full_label_blind_replay_both_arms"
ORIGINAL_EXCEPTION_TYPE: Final = "M2GateDerivationError"
ORIGINAL_EXCEPTION_SUBSTRING: Final = (
    "TRAIN record set does not match the M1 stream cache"
)


def recovery_lineage() -> dict[str, Any]:
    """The lineage every recovery2 artifact must bind. It conceals neither.

    Both prior attempts are named with their own reason class and their own
    exposure. Recovery1's conservative receipt value and the human forensic
    control-flow determination are carried as SEPARATE facts: the receipt said
    `indeterminate` because the tracker could not prove otherwise, while the
    traceback shows the failure occurred inside stream assembly before any
    stream reached replay. Neither replaces the other.
    """
    return {
        "recovery2_decision_document": RECOVERY2_DECISION_DOCUMENT,
        "recovery2_decision_sha256": RECOVERY2_DECISION_SHA256,
        "recovery_from_original_suite_id": ORIGINAL_SUITE_ID,
        "recovery1_suite_id": RECOVERY1_SUITE_ID,
        "recovery2_suite_id": CANONICAL_SUITE_ID,
        "attempt1_reason_class": ATTEMPT1_REASON_CLASS,
        "recovery1_reason_class": RECOVERY1_REASON_CLASS,
        "attempt1_scoring_started": False,
        "attempt1_metrics_computed": False,
        "attempt1_test_accessed": False,
        "recovery1_receipt_scoring_started": "indeterminate",
        "recovery1_human_forensic_scorer_invocation_observed": False,
        "recovery1_replay_completed": False,
        "recovery1_metrics_computed": False,
        "recovery1_test_accessed": False,
    }


PLANNED_EXECUTION_ORDER: Final = (
    "pre_claim_artifact_readiness",
    "start_and_claim_both_arms",
    "development_source_integrity",
    "full_label_blind_replay_both_arms",
    "post_replay_population_construction",
    "post_replay_frozen_evidence",
    "persist_and_promote_per_arm",
    "two_arm_suite_without_selection",
)

ROUTE_IMPLEMENTS_TWO_ARM_SUITE: Final = True
"""A static fact about THIS implementation -- never evidence about run history.
Whether a canonical run occurred is read from the filesystem by
`canonical_execution_history()`, because a source constant cannot update
itself."""


class M2DevelopmentRunError(RuntimeError):
    """Raised when the canonical development route refuses to proceed."""


# --------------------------------------------------------------------------
# Canonical roots -- the repository's existing conventions, not new ones
# --------------------------------------------------------------------------


def canonical_roots() -> dict[str, Path]:
    """The deterministic canonical inputs and outputs of the M2 development run.

    Every input root is the repository's already-frozen convention. The two
    outputs are dedicated to M2 development so the run cannot land in another
    experiment's tree.
    """
    from cardiosentinel.neural.m2_gate_derivation import (
        DEFAULT_FEATURE_ROOT,
        DEFAULT_M1_RUN_ROOT,
        DEFAULT_P1_CACHE_ROOT,
        DEFAULT_STREAM_CACHE_ROOT,
    )
    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    return {
        "source_root": REPOSITORY_ROOT / "cardiosentinel-data" / "ltstdb" / "1.0.0",
        "feature_root": Path(DEFAULT_FEATURE_ROOT),
        "stream_cache_root": Path(DEFAULT_STREAM_CACHE_ROOT),
        # The frozen P1 embedding cache. NOT the M1 stream-memory root: they
        # are different artifacts and the primary population lives only here.
        "p1_cache_root": Path(DEFAULT_P1_CACHE_ROOT),
        "m1_run_root": Path(DEFAULT_M1_RUN_ROOT),
        "run_root": (
            REPOSITORY_ROOT / "cardiosentinel-runs" / "phase6-m2-development-v1"
        ),
    }


def _assert_frozen_arm_order() -> None:
    if tuple(CANONICAL_ARM_ORDER) != tuple(M2_ARMS):
        raise M2DevelopmentRunError(
            f"The canonical arm order {CANONICAL_ARM_ORDER} disagrees with the "
            f"frozen arms {M2_ARMS}."
        )


# --------------------------------------------------------------------------
# Execution history comes from the filesystem, never from a source constant
# --------------------------------------------------------------------------


STATE_UNCLAIMED: Final = "unclaimed"
STATE_CLAIMED: Final = "claimed"
STATE_COMPLETE: Final = "complete"
STATE_FAILED: Final = "failed"
STATE_CONSUMED_FAILED_PRE_SCORING: Final = "consumed_failed_pre_scoring"
STATE_CONSUMED_FAILED_STREAM_ASSEMBLY: Final = (
    "consumed_failed_pre_scoring_stream_assembly"
)


def _suite_state(root: Path, suite_id: str) -> dict[str, Any]:
    """One suite's observed state, read from its artifacts."""
    import json

    from cardiosentinel.neural.m2_persistence import (
        ARM_RESULT_NAME,
        ATTEMPT_FAILURE_RECEIPT_NAME,
        EXPERIMENT_LOCK_NAME,
        RUN_STATUS_NAME,
        SUITE_RESULT_NAME,
        arm_experiment_id,
        failure_review_directory,
        suite_directory,
    )

    arms: dict[str, Any] = {}
    for arm in CANONICAL_ARM_ORDER:
        run_dir = root / arm_experiment_id(suite_id, arm)
        status = None
        status_path = run_dir / RUN_STATUS_NAME
        if status_path.is_file():
            status = json.loads(status_path.read_text()).get("status")
        arms[arm] = {
            "claimed": run_dir.exists(),
            "status": status,
            "result_promoted": (run_dir / ARM_RESULT_NAME).is_file(),
            "lock_promoted": (run_dir / EXPERIMENT_LOCK_NAME).is_file(),
        }
    suite_promoted = (suite_directory(root, suite_id) / SUITE_RESULT_NAME).is_file()
    receipt = (
        failure_review_directory(root, suite_id) / ATTEMPT_FAILURE_RECEIPT_NAME
    ).is_file()
    claimed = any(entry["claimed"] for entry in arms.values())
    complete = suite_promoted and all(
        entry["result_promoted"] and entry["lock_promoted"] for entry in arms.values()
    )
    failed = (
        claimed
        and not complete
        and (
            receipt
            or any(
                entry["status"] == "FAILED_OR_INTERRUPTED" for entry in arms.values()
            )
        )
    )
    if complete:
        state = STATE_COMPLETE
    elif failed:
        state = STATE_FAILED
    elif claimed:
        state = STATE_CLAIMED
    else:
        state = STATE_UNCLAIMED
    return {
        "suite_id": suite_id,
        "state": state,
        "arms": arms,
        "suite_result_promoted": suite_promoted,
        "failure_receipt_present": receipt,
        "any_attempt_claimed": claimed,
    }


def canonical_execution_history(
    run_root: Path | None = None, suite_id: str = CANONICAL_SUITE_ID
) -> dict[str, Any]:
    """What the canonical artifacts say about both attempts.

    Read from the filesystem, never from a source constant: code cannot rewrite
    itself, so a hard-coded claim about run history could only become a lie.
    """
    from cardiosentinel.neural.m2_persistence import (
        M2PersistenceError,
        validate_recovery1_failure_lineage,
    )

    root = Path(run_root or canonical_roots()["run_root"])
    original = _suite_state(root, ORIGINAL_SUITE_ID)
    # `consumed_failed_pre_scoring` is a SCIENTIFIC claim about what attempt #1
    # did and did not touch, so it is granted only when the preserved artifacts
    # prove it against the frozen digests. A claim directory on its own proves
    # that something was claimed and nothing more.
    original["lineage_verified"] = False
    if original["any_attempt_claimed"]:
        from cardiosentinel.neural.m2_persistence import (
            M2PersistenceError,
            validate_original_attempt1_failure_lineage,
        )

        try:
            lineage = validate_original_attempt1_failure_lineage(root)
        except M2PersistenceError as error:
            original["state"] = STATE_CLAIMED
            original["lineage_error"] = str(error)
        else:
            original["state"] = STATE_CONSUMED_FAILED_PRE_SCORING
            original["lineage_verified"] = True
            original["scoring_started"] = lineage["scoring_started"]
            original["metrics_computed"] = lineage["metrics_computed"]
            original["test_accessed"] = lineage["test_accessed"]
    # Recovery1 is a SECOND consumed pre-scoring failure with its own frozen
    # identity, and is likewise proven rather than inferred.
    recovery1 = _suite_state(root, RECOVERY1_SUITE_ID)
    recovery1["lineage_verified"] = False
    if recovery1["any_attempt_claimed"]:
        try:
            proof = validate_recovery1_failure_lineage(root)
        except M2PersistenceError as error:
            recovery1["state"] = STATE_CLAIMED
            recovery1["lineage_error"] = str(error)
        else:
            recovery1["state"] = STATE_CONSUMED_FAILED_STREAM_ASSEMBLY
            recovery1["lineage_verified"] = True
            recovery1["receipt_scoring_started"] = proof[
                "recovery1_receipt_scoring_started"
            ]
            recovery1["human_forensic_scorer_invocation_observed"] = proof[
                "recovery1_human_forensic_scorer_invocation_observed"
            ]
            recovery1["replay_completed"] = proof["recovery1_replay_completed"]
            recovery1["metrics_computed"] = proof["recovery1_metrics_computed"]
            recovery1["test_accessed"] = proof["recovery1_test_accessed"]

    recovery = _suite_state(root, suite_id)
    return {
        "run_root": str(root),
        "suite_id": suite_id,
        "original_attempt": original,
        "recovery1_attempt": recovery1,
        "recovery_attempt": recovery,
        # Back-compatible view of the suite this runner would execute.
        "arms": recovery["arms"],
        "suite_result_promoted": recovery["suite_result_promoted"],
        "any_attempt_claimed": recovery["any_attempt_claimed"],
        "evidence_source": "canonical claim directories, run status, locks, suite",
    }


# --------------------------------------------------------------------------
# Pre-claim readiness: everything provable WITHOUT opening VALIDATION
# --------------------------------------------------------------------------


def require_expected_git_sha(expected_git_sha: str | None) -> str:
    """HEAD must equal the human-authorized SHA, on a clean checkout.

    Checked BEFORE any data access, so a run against an unreviewed tree stops
    without consuming an attempt or opening a partition.
    """
    from cardiosentinel.data.provenance import git_provenance
    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    if not expected_git_sha:
        raise M2DevelopmentRunError(
            f"{EXPECTED_GIT_SHA_FLAG} is required. The canonical development run "
            "executes only against the exact human-reviewed master SHA, which "
            "is known only after the activation PR is merged and verified."
        )
    expected = str(expected_git_sha).strip().lower()
    provenance = git_provenance(REPOSITORY_ROOT)
    actual = str(provenance["git_sha"]).lower()
    if provenance["git_dirty"]:
        raise M2DevelopmentRunError(
            "Canonical M2 development evidence requires a clean Git checkout; "
            "the working tree is dirty. No data was opened."
        )
    if actual != expected:
        raise M2DevelopmentRunError(
            f"HEAD is {actual}, but the human authorization names {expected}. "
            "Execution stops BEFORE any data access; nothing was opened and no "
            "attempt was consumed."
        )
    return actual


def _assert_test_firewall() -> None:
    for forbidden in ("test", "TEST", " test "):
        try:
            require_canonical_development_partition(forbidden)
        except M2ExecutionError:
            continue
        raise M2DevelopmentRunError(  # pragma: no cover - firewall would be broken
            f"The partition firewall accepted {forbidden!r}; refusing to run."
        )


def preflight(*, expected_git_sha: str | None) -> dict[str, Any]:
    """Identity checks that need no local scientific artifact at all.

    Kept separate from `pre_claim_readiness` so the plan-only invocation can
    prove the Git authorization, the runtime, the frozen protocol digests and
    both firewalls without requiring the corpus to be present.
    """
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural import m2_gate as GATE
    from cardiosentinel.neural.m2_stress_intervals import (
        DECISION_DOCUMENT,
        DECISION_SHA256,
    )
    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        require_runtime_identity,
    )

    _assert_frozen_arm_order()
    git_sha = require_expected_git_sha(expected_git_sha)
    start = require_runtime_identity(
        EnforcementPoint.START, detail="m2_development_preflight"
    )
    protocol_sha = GATE.validate_m2_protocol()
    receipt_sha = GATE.validate_m2_gate_receipt()
    decision_sha = sha256_file(REPOSITORY_ROOT / DECISION_DOCUMENT)
    if decision_sha != DECISION_SHA256:
        raise M2DevelopmentRunError(
            f"{DECISION_DOCUMENT} digests to {decision_sha}, not the frozen "
            f"{DECISION_SHA256}. The stress-eligibility decision and the "
            "implementation have drifted apart."
        )
    recovery_sha = sha256_file(REPOSITORY_ROOT / RECOVERY_DECISION_DOCUMENT)
    if recovery_sha != RECOVERY_DECISION_SHA256:
        raise M2DevelopmentRunError(
            f"{RECOVERY_DECISION_DOCUMENT} digests to {recovery_sha}, not the "
            f"frozen {RECOVERY_DECISION_SHA256}. The recovery decision and the "
            "implementation have drifted apart."
        )
    firewall = assert_label_firewall()
    partition = require_canonical_development_partition(CANONICAL_DEVELOPMENT_PARTITION)
    _assert_test_firewall()
    return {
        "preflight_class": "m2_v1_canonical_development_preflight",
        "git_sha": git_sha,
        "git_dirty": False,
        "partition": partition,
        "arms": list(CANONICAL_ARM_ORDER),
        "arm_selection_performed": False,
        "runtime_identity": start.as_dict(),
        "m2_protocol_sha256": protocol_sha,
        "m2_gate_receipt_sha256": receipt_sha,
        "stress_eligibility_decision_sha256": decision_sha,
        "recovery_lineage": recovery_lineage(),
        "label_firewall": firewall,
        "data_opened": False,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


def require_recovery_preconditions(run_root: Path, suite_id: str) -> dict[str, Any]:
    """The recovery may run only from a PROVEN prior state.

    Attempt #1 is verified from its preserved artifacts against the frozen
    digests -- never inferred from a directory existing -- and the recovery
    suite must still be unclaimed. Nothing here repairs, cleans or re-runs
    anything, and no VALIDATION is opened.
    """
    from cardiosentinel.neural.m2_persistence import (
        validate_original_attempt1_failure_lineage,
        validate_recovery1_failure_lineage,
    )

    lineage = validate_original_attempt1_failure_lineage(run_root)
    recovery1_lineage = validate_recovery1_failure_lineage(run_root)
    history = canonical_execution_history(run_root, suite_id)
    original = history["original_attempt"]
    if original["state"] != STATE_CONSUMED_FAILED_PRE_SCORING:
        raise M2DevelopmentRunError(
            f"The original attempt {ORIGINAL_SUITE_ID!r} is in state "
            f"{original['state']!r}; the recovery route runs only after a "
            "verified consumed pre-scoring failure."
        )
    if history["recovery1_attempt"]["state"] != STATE_CONSUMED_FAILED_STREAM_ASSEMBLY:
        raise M2DevelopmentRunError(
            f"Recovery1 {RECOVERY1_SUITE_ID!r} is in state "
            f"{history['recovery1_attempt']['state']!r}; recovery2 runs only "
            "after a verified consumed recovery1 stream-assembly failure."
        )
    recovery = history["recovery_attempt"]
    if recovery["state"] != STATE_UNCLAIMED:
        raise M2DevelopmentRunError(
            f"The recovery suite {suite_id!r} is already {recovery['state']!r}. "
            "The attempt is consumed; nothing is deleted, renamed, reseeded or "
            "retried, and no further recovery is implicitly authorized."
        )
    history["original_attempt_lineage"] = lineage
    history["recovery1_attempt_lineage"] = recovery1_lineage
    return history


def pre_claim_readiness(
    *,
    expected_git_sha: str | None,
    roots: dict[str, Path],
    suite_id: str,
    loaders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete readiness gate. Nothing here requires VALIDATION.

    The scorer and the TRAIN-only distance standardizer are readied here
    deliberately: both are frozen non-VALIDATION artifacts, and discovering a
    missing or altered checkpoint AFTER claiming two canonical attempts would
    consume them for nothing.
    """
    from cardiosentinel.neural.m2_persistence import require_unclaimed_suite
    from cardiosentinel.neural.m2_scorer import (
        FROZEN_B4B_CHECKPOINT_SHA256,
        FROZEN_P1B_LOCK_SHA256,
        RETAINED_M1L_CHECKPOINT_SHA256,
        RETAINED_M1L_LOCK_SHA256,
    )

    inject = loaders or {}
    readiness = preflight(expected_git_sha=expected_git_sha)
    readiness["execution_history"] = require_recovery_preconditions(
        roots["run_root"], suite_id
    )

    if "load_frozen_m1l_scorer" in inject:
        load_scorer = inject["load_frozen_m1l_scorer"]
    else:
        from cardiosentinel.neural.m2_scorer import load_frozen_m1l_scorer

        load_scorer = load_frozen_m1l_scorer
    if "load_distance_standardizer" in inject:
        load_standardizer = inject["load_distance_standardizer"]
    else:
        from cardiosentinel.neural.m2_execution import load_distance_standardizer

        load_standardizer = load_distance_standardizer

    scorer = load_scorer(roots["m1_run_root"])
    scorer_identity = scorer.identity()
    for name, expected in (
        ("retained_lock_sha256", RETAINED_M1L_LOCK_SHA256),
        ("retained_checkpoint_sha256", RETAINED_M1L_CHECKPOINT_SHA256),
        ("p1b_lock_sha256", FROZEN_P1B_LOCK_SHA256),
        ("b4b_checkpoint_sha256", FROZEN_B4B_CHECKPOINT_SHA256),
    ):
        if scorer_identity.get(name) != expected:
            raise M2DevelopmentRunError(
                f"Scorer {name} is {scorer_identity.get(name)!r}, expected the "
                f"frozen {expected!r}. No arm is claimed and no VALIDATION is "
                "opened."
            )
    # The distance standardizer is TRAIN-only and lives at the cache ROOT, not
    # inside a partition directory, so readying it opens no VALIDATION data.
    standardizer = load_standardizer(roots["stream_cache_root"])
    claim_check = require_unclaimed_suite(roots["run_root"], suite_id)

    readiness.update(
        {
            "readiness_class": "m2_v1_pre_claim_artifact_readiness",
            "scorer_identity": scorer_identity,
            "distance_standardizer_ready": True,
            "pair_claim_check": claim_check,
            "validation_opened_during_readiness": False,
        }
    )
    return {"readiness": readiness, "scorer": scorer, "standardizer": standardizer}


# --------------------------------------------------------------------------
# Post-replay annotation assembly
# --------------------------------------------------------------------------


def build_primary_annotations(primary: Any, *, stream_cache_root: Path) -> Any:
    """PRIMARY annotations: frozen labels/subjects, plus label-free age bins.

    Labels and subjects come from the frozen P1 validation population; the
    cold-start bin comes from the M1 stream cache's persisted
    `recording_age_seconds` binning, which is label-free by construction.
    Nothing here consults an M2 score.
    """
    from cardiosentinel.neural.m1_experiment import load_stream_store
    from cardiosentinel.neural.m1_store import COLD_START_BIN_FILE, STABLE_ID_FILE
    from cardiosentinel.neural.m2_evaluation import M2PrimaryAnnotationTable

    store, _manifest = load_stream_store(
        Path(stream_cache_root), CANONICAL_DEVELOPMENT_PARTITION
    )
    try:
        cache_ids = np.asarray(store.array(STABLE_ID_FILE))
        bins = np.asarray(store.array(COLD_START_BIN_FILE))
    finally:
        store.close()

    index = {str(value): position for position, value in enumerate(cache_ids)}
    missing = [sid for sid in primary.stable_ids if sid not in index]
    if missing:
        raise M2DevelopmentRunError(
            f"{len(missing)} frozen primary rows are absent from the stream "
            f"cache, beginning {missing[:3]}. A metric row with no causal "
            "replay history is never evaluated."
        )
    positions = [index[sid] for sid in primary.stable_ids]
    return M2PrimaryAnnotationTable(
        stable_ids=np.asarray(primary.stable_ids, dtype=np.str_),
        labels=np.asarray(primary.labels, dtype=np.int64),
        subject_ids=np.asarray(primary.subject_ids, dtype=np.str_),
        cold_start_bins=np.asarray([str(bins[p]) for p in positions], dtype=np.str_),
    )


def build_challenge_annotations(challenge: Any) -> Any:
    """CHALLENGE annotations, straight from the frozen selection. No labels."""
    from cardiosentinel.neural.m2_evaluation import M2ChallengeAnnotationTable

    return M2ChallengeAnnotationTable(
        stable_ids=np.asarray(challenge.stable_ids, dtype=np.str_),
        target_families=np.asarray(challenge.target_families, dtype=np.str_),
        subject_ids=np.asarray(challenge.subject_ids, dtype=np.str_),
    )


def verify_development_source(
    *, source_root: Path, feature_root: Path
) -> dict[str, Any]:
    """Prove the raw `.hea`/`.dat`/`.stb` are the official frozen source.

    The stress selection reads raw LTSTDB `.stb`, so an arbitrary local
    directory is never trusted. This reuses the repository's existing
    development verifiers, which bind the files to the official pinned manifest,
    the frozen per-record source digests and the frozen feature-corpus identity,
    over the train/validation development partitions only. TEST files are never
    hashed.
    """
    from cardiosentinel.neural.integrity import (
        validate_development_feature_integrity,
        validate_development_source_integrity,
    )

    feature_receipt = validate_development_feature_integrity(Path(feature_root))
    source_receipt = validate_development_source_integrity(
        Path(source_root), feature_receipt
    )
    return {
        "identity_class": "m2_v1_development_source_integrity",
        "feature_receipt": feature_receipt,
        "source_receipt": source_receipt,
        "annotation_set": "stb",
        "test_partition_hashed": False,
        "verified_before_stress_selection": True,
    }


def parsed_validation_annotations(*, source_root: Path, feature_root: Path):
    """Frozen `.stb` annotations for the VALIDATION records, read post-replay.

    Reads the primary annotation set only; alternative ST definitions are never
    mixed. Called strictly after both arms' trajectories are complete, so no
    annotation can reach a score, a gate decision or a memory update.
    """
    from cardiosentinel.data.ltstdb import read_annotations, read_record
    from cardiosentinel.neural.metadata import load_b4_references

    references = load_b4_references(
        Path(feature_root), CANONICAL_DEVELOPMENT_PARTITION, primary_only=False
    )
    for record_id in sorted({item.record_id for item in references}):
        record = read_record(Path(source_root), record_id)
        yield read_annotations(Path(source_root), record)


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def replay_both_arms(
    *,
    stores: dict[str, Any],
    standardizer: Any,
    scorer: Any,
    stream_cache_root: Path,
    feature_root: Path,
    stream_source: Any | None = None,
) -> None:
    """One pass over the validation input; both arms replayed per stream.

    The input is loaded exactly once and both arms consume the identical frozen
    rows in the frozen order. The arms share nothing but that input and the
    frozen scorer: each keeps its own `M2StreamState`, so no M2-0 result can
    alter M2-G's replay. Each stream's rows and trajectories are released before
    the next stream is read.
    """
    from cardiosentinel.neural.m2_policy import replay_stream

    if stream_source is None:
        from cardiosentinel.neural.m2_execution import iter_timeline_streams

        stream_source = iter_timeline_streams(
            CANONICAL_DEVELOPMENT_PARTITION,
            stream_cache_root=stream_cache_root,
            feature_root=feature_root,
        )

    for key, rows in stream_source:
        for arm in CANONICAL_ARM_ORDER:
            collected: list[tuple[float, Any]] = []

            def observer(_index, available_time, mu_long, _sink=collected):
                _sink.append((float(available_time), np.asarray(mu_long)))

            evidence = replay_stream(
                rows,
                arm=arm,
                standardizer=standardizer,
                scorer=scorer,
                prototype_observer=observer,
            )
            stores[arm].add_stream(key, evidence, collected)
            del evidence, collected
        del rows


# --------------------------------------------------------------------------
# The canonical orchestration
# --------------------------------------------------------------------------


def require_canonical_suite_id(suite_id: str) -> str:
    """Production canonical execution is ALWAYS the one canonical suite.

    A caller-chosen suite name would let a second "canonical" suite be executed
    under a different directory after the first was consumed, which is exactly
    the automatic-rerun the claim model forbids. The check runs before claim
    checking, before any filesystem creation and before any VALIDATION access.
    """
    if str(suite_id) != CANONICAL_SUITE_ID:
        raise M2DevelopmentRunError(
            f"The canonical M2 development suite is {CANONICAL_SUITE_ID!r}; "
            f"{suite_id!r} is refused. A second suite name would let a consumed "
            "canonical attempt be re-run under another directory. Nothing was "
            "claimed, created or opened."
        )
    return CANONICAL_SUITE_ID


def execute_canonical_development(
    *,
    expected_git_sha: str | None,
    execute: bool = False,
    _roots: dict[str, Path] | None = None,
    _loaders: dict[str, Any] | None = None,
    _suite_id: str | None = None,
) -> dict[str, Any]:
    """The canonical two-arm development run.

    Without `execute=True` this performs the identity preflight and returns the
    plan; it opens no partition and consumes no attempt.

    There is deliberately **no public `suite_id` parameter**: production always
    runs the one `CANONICAL_SUITE_ID`. `_roots`, `_loaders` and `_suite_id` are
    private TEST-ONLY seams, absent from the CLI and from the public scientific
    contract; `_suite_id` still cannot bypass a consumed canonical attempt,
    because the pair preflight refuses any suite whose paths already exist.
    """
    if not execute:
        plan = preflight(expected_git_sha=expected_git_sha)
        plan["planned_execution_order"] = list(PLANNED_EXECUTION_ORDER)
        plan["executed"] = False
        return plan
    suite_id = (
        require_canonical_suite_id(CANONICAL_SUITE_ID)
        if _suite_id is None
        else str(_suite_id)
    )
    roots = dict(_roots or canonical_roots())
    # Deterministic post-claim failure accounting. Attempt #1 left both status
    # files reading STARTED because the exception escaped outside a promotion
    # gate; once any claim exists an uncaught exception must leave evidence.
    tracker = _AttemptTracker(run_root=roots["run_root"], suite_id=suite_id)
    try:
        return _run(
            expected_git_sha=expected_git_sha,
            suite_id=suite_id,
            roots=roots,
            loaders=dict(_loaders or {}),
            production=_suite_id is None,
            tracker=tracker,
        )
    except BaseException as error:
        tracker.record_failure(error)
        raise


@dataclass(slots=True)
class _AttemptTracker:
    """The REAL execution state, so a failure reports real scientific exposure.

    Attempt #1's `scoring_started=false` is a frozen determination about that
    attempt; it is not a template for future receipts. A failure after the
    scorer has been invoked must say so, and a fact that an abrupt exception
    leaves genuinely unknowable is recorded as `indeterminate` rather than as a
    flattering `false`.

    It records nothing until an arm claim exists: a failure BEFORE any claim
    consumes no attempt and leaves no directory to annotate.
    """

    run_root: Path
    suite_id: str
    stage: str = "pre_claim_artifact_readiness"
    claimed_arms: list[str] = field(default_factory=list)
    validation_opened: bool = False
    scoring_started: bool = False
    replay_completed: bool = False
    post_replay_evaluation_started: bool = False
    metrics_completed: dict[str, bool] = field(default_factory=dict)
    runtimes: dict[str, Any] = field(default_factory=dict)
    # What the RUN believed it promoted. Kept as an observation only: the
    # failure receipt re-reads the actual artifacts, because the finalizer
    # permits a window where the arm result is promoted and verified but the
    # lock promotion then fails -- a tracker set only on full success would
    # report `false` for a file that demonstrably exists.
    arm_result_promoted: dict[str, bool] = field(default_factory=dict)
    experiment_lock_promoted: dict[str, bool] = field(default_factory=dict)
    suite_result_promoted: bool = False

    def tracking_scorer(self, scorer: Any) -> Any:
        """A transparent wrapper that flags scoring, changing no numeric.

        It sets `scoring_started` immediately before delegating and returns the
        underlying scorer's output unchanged -- no rounding, no casting, no
        re-derivation. The frozen M1L weights, features and scores are
        untouched.
        """
        tracker = self

        class _TrackingScorer:
            def __call__(self, *args, **kwargs):
                tracker.scoring_started = True
                return scorer(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(scorer, name)

        return _TrackingScorer()

    def exposure(self) -> dict[str, Any]:
        """What can be asserted, and what genuinely cannot."""
        from cardiosentinel.neural.m2_persistence import INDETERMINATE

        # Once replay has begun, "did the scorer run?" is only knowable as far
        # as the wrapper observed it: an exception mid-stream may or may not
        # have followed a completed score.
        scoring = self.scoring_started
        if not scoring and self.validation_opened and not self.replay_completed:
            scoring = INDETERMINATE
        metrics_any = any(self.metrics_completed.values())
        metrics = metrics_any
        if not metrics_any and self.post_replay_evaluation_started:
            metrics = INDETERMINATE
        return {
            "scoring_started": scoring,
            "replay_completed": self.replay_completed,
            "post_replay_evaluation_started": self.post_replay_evaluation_started,
            "metrics_computed_or_completed": metrics,
            "metrics_completed_per_arm": dict(self.metrics_completed),
        }

    def record_failure(self, error: BaseException) -> dict[str, Any] | None:
        from cardiosentinel.neural.m2_persistence import record_attempt_failure

        if not self.claimed_arms:
            return None
        return record_attempt_failure(
            self.run_root,
            self.suite_id,
            exception=error,
            stage=self.stage,
            claimed_arms=list(self.claimed_arms),
            validation_opened=self.validation_opened,
            exposure=self.exposure(),
            runtime_records=dict(self.runtimes),
            # Passed as the tracker's OBSERVATION; the receipt's canonical
            # promotion values come from the filesystem.
            promotion_state={
                "arm_result_promoted": {
                    arm: bool(self.arm_result_promoted.get(arm, False))
                    for arm in CANONICAL_ARM_ORDER
                },
                "experiment_lock_promoted": {
                    arm: bool(self.experiment_lock_promoted.get(arm, False))
                    for arm in CANONICAL_ARM_ORDER
                },
                "suite_result_promoted": self.suite_result_promoted,
            },
            decision_sha256=RECOVERY_DECISION_SHA256,
        )


def _run(
    *,
    expected_git_sha: str | None,
    suite_id: str,
    roots: dict[str, Path],
    loaders: dict[str, Any],
    production: bool = True,
    tracker: Any | None = None,
) -> dict[str, Any]:
    """The frozen execution order, end to end."""
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.m2_evaluation import (
        build_challenge_bundle,
        build_primary_bundle,
        cold_start_stratified_evidence,
        false_alarm_evidence,
        policy_evidence,
        streaming_contamination_evidence,
        window_evidence,
    )
    from cardiosentinel.neural.m2_evidence_store import (
        M2EvidenceStore,
        validate_evidence_store_manifest,
    )
    from cardiosentinel.neural.m2_execution import (
        m2_execution_identity,
        streaming_input_identity,
    )
    from cardiosentinel.neural.m2_persistence import (
        ARM_RESULT_NAME,
        EXPERIMENT_LOCK_NAME,
        arm_experiment_id,
        build_suite_body,
        claim_evidence_workspace,
        claim_run_directory,
        finalize_and_promote_arm_result,
        finalize_and_promote_suite_result,
        read_json_result,
    )
    from cardiosentinel.neural.m2_scorer import M1L_CLASSIFICATION_THRESHOLD
    from cardiosentinel.neural.m2_stress_intervals import (
        build_stress_selection_from_parsed,
    )
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        RuntimeIntegrityRecord,
        require_runtime_identity,
    )

    def _use(name: str, default):
        return loaders.get(name, default)

    # -- 1. PRE-CLAIM READINESS. Nothing requiring VALIDATION is opened. -----
    readied = pre_claim_readiness(
        expected_git_sha=expected_git_sha,
        roots=roots,
        suite_id=suite_id,
        loaders=loaders,
    )
    readiness = readied["readiness"]
    scorer = readied["scorer"]
    standardizer = readied["standardizer"]
    track = (
        tracker
        if tracker is not None
        else _AttemptTracker(run_root=roots["run_root"], suite_id=suite_id)
    )
    track.stage = "start_and_claim_both_arms"

    # -- 2. START + CLAIM both arms BEFORE any development data is opened. ---
    runtimes = {arm: RuntimeIntegrityRecord() for arm in CANONICAL_ARM_ORDER}
    for arm in CANONICAL_ARM_ORDER:
        require_runtime_identity(
            EnforcementPoint.START, record=runtimes[arm], detail=f"m2_replay:{arm}"
        )
    track.runtimes = runtimes
    claims = {}
    for arm in CANONICAL_ARM_ORDER:
        claims[arm] = claim_run_directory(
            roots["run_root"],
            arm_experiment_id(suite_id, arm),
            arm,
            runtime=runtimes[arm],
        )
        track.claimed_arms.append(arm)
    workspace = claim_evidence_workspace(roots["run_root"], suite_id)
    track.stage = "development_source_integrity"

    # -- 3. DEVELOPMENT SOURCE INTEGRITY, before any raw annotation is read. -
    verify_source = _use("verify_development_source", verify_development_source)
    source_identity = verify_source(
        source_root=roots["source_root"], feature_root=roots["feature_root"]
    )

    # -- 4. FULL LABEL-BLIND REPLAY. --------------------------------------
    replay_authority = _use(
        "canonical_replay_population", _default_canonical_replay_population
    )
    track.stage = "full_label_blind_replay_both_arms"
    track.validation_opened = True
    replay_population, replay_stable_ids, manifest = replay_authority(
        roots["stream_cache_root"]
    )
    stores = {
        arm: M2EvidenceStore(root=workspace / arm, arm=arm)
        for arm in CANONICAL_ARM_ORDER
    }
    replay_both_arms(
        stores=stores,
        standardizer=standardizer,
        # Transparent wrapper: flags exposure, returns the frozen scorer's
        # output byte-for-byte unchanged.
        scorer=track.tracking_scorer(scorer),
        stream_cache_root=roots["stream_cache_root"],
        feature_root=roots["feature_root"],
        stream_source=loaders.get("stream_source"),
    )
    track.replay_completed = True
    store_manifests = {}
    for arm in CANONICAL_ARM_ORDER:
        store_manifests[arm] = validate_evidence_store_manifest(
            stores[arm].finalize(), root=workspace / arm
        )

    track.stage = "post_replay_population_construction"
    track.post_replay_evaluation_started = True
    # -- 5. POST-REPLAY. No annotation was loaded until here. ---------------
    primary = _use("primary_evaluation_population", _default_primary_population)(
        roots["p1_cache_root"]
    )
    challenge = _use("challenge_evaluation_population", _default_challenge_population)(
        roots["feature_root"]
    )
    from cardiosentinel.neural.m2_populations import prove_population_containment

    containment = prove_population_containment(
        replay_population=replay_population,
        replay_stable_ids=replay_stable_ids,
        primary=primary,
        challenge=challenge,
    )
    del replay_stable_ids

    primary_annotations = _use("build_primary_annotations", build_primary_annotations)(
        primary, stream_cache_root=roots["stream_cache_root"]
    )
    challenge_annotations = build_challenge_annotations(challenge)
    stress = build_stress_selection_from_parsed(
        _use("parsed_validation_annotations", parsed_validation_annotations)(
            source_root=roots["source_root"], feature_root=roots["feature_root"]
        )
    )
    stress_identity = dict(stress.identity())
    stress_identity["development_source_identity"] = source_identity

    track.stage = "post_replay_frozen_evidence"
    results: dict[str, dict[str, Any]] = {}
    lock_digests: dict[str, str] = {}
    population_identities: dict[str, Any] = {}
    for arm in CANONICAL_ARM_ORDER:
        table = stores[arm].score_table()
        primary_bundle = build_primary_bundle(
            arm, table, primary_annotations, primary_population=primary
        )
        challenge_bundle = build_challenge_bundle(
            arm, table, challenge_annotations, challenge_population=challenge
        )
        result = {
            "artifact_class": "m2_v1_canonical_arm_result",
            "arm": arm,
            "scientific_computation_completed": True,
            "label_blind_replay_completed": True,
            "m1l_classification_threshold": M1L_CLASSIFICATION_THRESHOLD,
            "threshold_selected_here": False,
            "classifier_retrained": False,
            "memory_selection_performed": False,
            "memory_selected": None,
            "rollback": False,
            "partition_accessed": CANONICAL_DEVELOPMENT_PARTITION,
            "validation_accessed": True,
            "test_accessed": False,
            "sealed_test_state": "unopened",
            "suite_id": suite_id,
            "experiment_id": arm_experiment_id(suite_id, arm),
            "replay_population_identity": replay_population.identity(),
            "primary_evaluation_population_identity": (
                primary_bundle.population_identity()
            ),
            "challenge_evaluation_population_identity": (
                challenge_bundle.population_identity()
            ),
            "stress_interval_selection_identity": stress_identity,
            # Top-level provenance for the whole arm, and identical to the copy
            # the stress selection carries -- the persistence layer requires
            # they agree exactly.
            "development_source_identity": source_identity,
            **recovery_lineage(),
            "population_containment_proof": containment,
            "pre_claim_readiness": readiness,
            "evidence_store_identity": store_manifests[arm],
            "policy_evidence": policy_evidence(
                stores[arm].admission, replay_population=replay_population
            ),
            "window_evidence": window_evidence(primary_bundle),
            "false_alarm_evidence": false_alarm_evidence(
                primary_bundle=primary_bundle, challenge_bundle=challenge_bundle
            ),
            "cold_start_evidence": cold_start_stratified_evidence(primary_bundle),
            # One trajectory resident at a time; the store reads exactly the
            # stress-bearing stream being evaluated and releases it.
            "contamination_evidence": streaming_contamination_evidence(
                stress_intervals=list(stress.evaluation_intervals()),
                load_trajectory=stores[arm].load_trajectory,
                replay_population=replay_population,
                stress_selection_identity=stress_identity,
            ),
        }
        if not population_identities:
            population_identities = {
                "replay_population_identity": result["replay_population_identity"],
                "primary_evaluation_population_identity": result[
                    "primary_evaluation_population_identity"
                ],
                "challenge_evaluation_population_identity": result[
                    "challenge_evaluation_population_identity"
                ],
                "stress_interval_selection_identity": stress_identity,
            }
        execution_identity = m2_execution_identity(
            streaming_input_identity(manifest),
            scorer,
            runtimes[arm],
            validation_accessed=True,
        )
        # Every prespecified evidence section for this arm now exists.
        track.metrics_completed[arm] = True
        track.stage = f"persist_and_promote_per_arm:{arm}"
        results[arm] = finalize_and_promote_arm_result(
            claims[arm],
            result=result,
            execution_identity=execution_identity,
            runtime=runtimes[arm],
        )
        track.arm_result_promoted[arm] = True
        track.experiment_lock_promoted[arm] = True
        lock_digests[arm] = read_json_result(
            claims[arm].run_dir / EXPERIMENT_LOCK_NAME
        )["experiment_lock_sha256"]
        del primary_bundle, challenge_bundle, table

    # -- 6. ONE aggregating suite. No new metric, no preference. -------------
    # The suite's own runtime record is created and STARTed first, so its
    # PRE_PROMOTION observation can be embedded before the payload is signed.
    track.stage = "two_arm_suite_without_selection"
    suite_runtime = RuntimeIntegrityRecord()
    require_runtime_identity(
        EnforcementPoint.START, record=suite_runtime, detail="m2_suite"
    )
    track.runtimes["suite"] = suite_runtime
    suite_body = build_suite_body(
        suite_id=suite_id,
        arm_results={
            arm: {
                "experiment_id": arm_experiment_id(suite_id, arm),
                "run_dir": str(claims[arm].run_dir),
                # Hashed from the PROMOTED bytes, never restated from memory.
                "arm_result_sha256": sha256_file(claims[arm].run_dir / ARM_RESULT_NAME),
            }
            for arm in CANONICAL_ARM_ORDER
        },
        arm_lock_sha256=lock_digests,
        population_identities=population_identities,
        development_source_identity=source_identity,
        recovery_lineage=recovery_lineage(),
        git_sha=readiness["git_sha"],
    )
    suite = finalize_and_promote_suite_result(
        roots["run_root"],
        suite_id,
        suite_body=suite_body,
        runtime=suite_runtime,
        expected_suite_id=CANONICAL_SUITE_ID if production else None,
    )
    track.suite_result_promoted = True
    return {
        "preflight_class": readiness["preflight_class"],
        "partition": readiness["partition"],
        "arms": list(CANONICAL_ARM_ORDER),
        "planned_execution_order": list(PLANNED_EXECUTION_ORDER),
        "executed": True,
        "suite_id": suite_id,
        "suite": suite,
        "arm_experiment_ids": {
            arm: arm_experiment_id(suite_id, arm) for arm in CANONICAL_ARM_ORDER
        },
        "memory_selection_performed": False,
        "memory_selected": None,
    }


def _default_canonical_replay_population(stream_cache_root: Path):
    from cardiosentinel.neural.m2_execution import canonical_replay_population

    return canonical_replay_population(
        CANONICAL_DEVELOPMENT_PARTITION, stream_cache_root=Path(stream_cache_root)
    )


def _default_primary_population(p1_cache_root: Path):
    from cardiosentinel.neural.m2_populations import primary_evaluation_population

    return primary_evaluation_population(Path(p1_cache_root))


def _default_challenge_population(feature_root: Path):
    from cardiosentinel.neural.m2_populations import challenge_evaluation_population

    return challenge_evaluation_population(Path(feature_root))


# --------------------------------------------------------------------------
# CLI: only the deliberate execution controls
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Every root and identity is deterministic, so nothing else.

    There is deliberately no partition, arm, threshold, retry, seed or
    alternative data-source option: each would be a way to run something other
    than the one reviewed canonical experiment.
    """
    parser = argparse.ArgumentParser(
        prog="m2_development_run",
        description=(
            "Canonical M2-v1 DEVELOPMENT run (VALIDATION only, both arms, no "
            "selection). Requires explicit execution consent and the exact "
            "human-authorized Git SHA."
        ),
    )
    parser.add_argument(
        EXECUTION_FLAG,
        action="store_true",
        help="Explicit consent to run the canonical two-arm development experiment.",
    )
    parser.add_argument(
        EXPECTED_GIT_SHA_FLAG,
        required=True,
        help="The human-reviewed master SHA this authorization names. HEAD must match.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Never runs on import."""
    args = build_parser().parse_args(argv)
    result = execute_canonical_development(
        expected_git_sha=getattr(args, "expected_git_sha"),
        execute=getattr(args, "execute_canonical_development"),
    )
    print(result["preflight_class"], result["partition"], result["arms"])
    return 0


# `__main__` dispatch is LAST on purpose: every helper the canonical run needs
# is defined above it, so module execution can never enter the run with an
# undefined runtime helper.
if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())
