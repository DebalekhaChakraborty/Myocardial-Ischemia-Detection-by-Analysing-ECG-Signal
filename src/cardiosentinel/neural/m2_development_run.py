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

CANONICAL_SUITE_ID: Final = "m2-v1-development-two-arm"
"""The one canonical M2 development suite identity. Deterministic, not a seed."""

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


def canonical_execution_history(
    run_root: Path | None = None, suite_id: str = CANONICAL_SUITE_ID
) -> dict[str, Any]:
    """What the canonical artifacts say about whether a run has happened."""
    import json

    from cardiosentinel.neural.m2_persistence import (
        ARM_RESULT_NAME,
        EXPERIMENT_LOCK_NAME,
        RUN_STATUS_NAME,
        SUITE_RESULT_NAME,
        arm_experiment_id,
        suite_directory,
    )

    root = Path(run_root or canonical_roots()["run_root"])
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
    suite_path = suite_directory(root, suite_id) / SUITE_RESULT_NAME
    return {
        "run_root": str(root),
        "suite_id": suite_id,
        "arms": arms,
        "suite_result_promoted": suite_path.is_file(),
        "any_attempt_claimed": any(entry["claimed"] for entry in arms.values()),
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
        "label_firewall": firewall,
        "data_opened": False,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


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
    for field, expected in (
        ("retained_lock_sha256", RETAINED_M1L_LOCK_SHA256),
        ("retained_checkpoint_sha256", RETAINED_M1L_CHECKPOINT_SHA256),
        ("p1b_lock_sha256", FROZEN_P1B_LOCK_SHA256),
        ("b4b_checkpoint_sha256", FROZEN_B4B_CHECKPOINT_SHA256),
    ):
        if scorer_identity.get(field) != expected:
            raise M2DevelopmentRunError(
                f"Scorer {field} is {scorer_identity.get(field)!r}, expected the "
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


def execute_canonical_development(
    *,
    expected_git_sha: str | None,
    execute: bool = False,
    suite_id: str = CANONICAL_SUITE_ID,
    _roots: dict[str, Path] | None = None,
    _loaders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The canonical two-arm development run.

    Without `execute=True` this performs the identity preflight and returns the
    plan; it opens no partition and consumes no attempt.

    `_roots` and `_loaders` are private TEST-ONLY dependency injection seams.
    They are absent from the CLI and from the public scientific contract: a real
    run always uses `canonical_roots()` and the real loaders.
    """
    if not execute:
        plan = preflight(expected_git_sha=expected_git_sha)
        plan["planned_execution_order"] = list(PLANNED_EXECUTION_ORDER)
        plan["executed"] = False
        return plan
    return _run(
        expected_git_sha=expected_git_sha,
        suite_id=suite_id,
        roots=dict(_roots or canonical_roots()),
        loaders=dict(_loaders or {}),
    )


def _run(
    *,
    expected_git_sha: str | None,
    suite_id: str,
    roots: dict[str, Path],
    loaders: dict[str, Any],
) -> dict[str, Any]:
    """The frozen execution order, end to end."""
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
        EXPERIMENT_LOCK_NAME,
        arm_experiment_id,
        build_suite_result,
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

    # -- 2. START + CLAIM both arms BEFORE any development data is opened. ---
    runtimes = {arm: RuntimeIntegrityRecord() for arm in CANONICAL_ARM_ORDER}
    for arm in CANONICAL_ARM_ORDER:
        require_runtime_identity(
            EnforcementPoint.START, record=runtimes[arm], detail=f"m2_replay:{arm}"
        )
    claims = {}
    for arm in CANONICAL_ARM_ORDER:
        claims[arm] = claim_run_directory(
            roots["run_root"],
            arm_experiment_id(suite_id, arm),
            arm,
            runtime=runtimes[arm],
        )
    workspace = claim_evidence_workspace(roots["run_root"], suite_id)

    # -- 3. DEVELOPMENT SOURCE INTEGRITY, before any raw annotation is read. -
    verify_source = _use("verify_development_source", verify_development_source)
    source_identity = verify_source(
        source_root=roots["source_root"], feature_root=roots["feature_root"]
    )

    # -- 4. FULL LABEL-BLIND REPLAY. --------------------------------------
    replay_authority = _use(
        "canonical_replay_population", _default_canonical_replay_population
    )
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
        scorer=scorer,
        stream_cache_root=roots["stream_cache_root"],
        feature_root=roots["feature_root"],
        stream_source=loaders.get("stream_source"),
    )
    store_manifests = {}
    for arm in CANONICAL_ARM_ORDER:
        store_manifests[arm] = validate_evidence_store_manifest(
            stores[arm].finalize(), root=workspace / arm
        )

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
            "development_source_identity": source_identity,
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
        results[arm] = finalize_and_promote_arm_result(
            claims[arm],
            result=result,
            execution_identity=execution_identity,
            runtime=runtimes[arm],
        )
        lock_digests[arm] = read_json_result(
            claims[arm].run_dir / EXPERIMENT_LOCK_NAME
        )["experiment_lock_sha256"]
        del primary_bundle, challenge_bundle, table

    # -- 6. ONE aggregating suite. No new metric, no preference. -------------
    suite = build_suite_result(
        suite_id=suite_id,
        arm_results={
            arm: {
                "experiment_id": arm_experiment_id(suite_id, arm),
                "run_dir": str(claims[arm].run_dir),
                "arm_result_sha256": results[arm]["artifact_sha256"][
                    "M2_ARM_RESULT.json"
                ]
                if "artifact_sha256" in results[arm]
                else None,
            }
            for arm in CANONICAL_ARM_ORDER
        },
        arm_lock_sha256=lock_digests,
        population_identities=population_identities,
        development_source_identity=source_identity,
        git_sha=readiness["git_sha"],
    )
    suite_runtime = RuntimeIntegrityRecord()
    require_runtime_identity(
        EnforcementPoint.START, record=suite_runtime, detail="m2_suite"
    )
    finalize_and_promote_suite_result(
        roots["run_root"],
        suite_id,
        suite=suite,
        runtime=suite_runtime,
        arm_run_dirs={arm: claims[arm].run_dir for arm in CANONICAL_ARM_ORDER},
    )
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
