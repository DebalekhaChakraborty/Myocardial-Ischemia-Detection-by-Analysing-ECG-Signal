"""The ONE canonical M2-v1 DEVELOPMENT invocation. Not yet executed.

This module is the only public route to a claim-bearing M2 development result.
It always runs both frozen arms, in the frozen order `M2-0` then `M2-G`, and it
never selects between them: no third arm exists, no single-arm scientific route
is exposed, and no automatic preference is applied anywhere.

**Nothing runs on import.** Execution requires BOTH `--execute-canonical-development`
and `--expected-git-sha <HUMAN_REVIEWED_MASTER_SHA>`, and HEAD must equal that
SHA on a clean checkout before any data is opened. The expected SHA is
deliberately not defaulted: the scientific run happens only after this
activation PR is merged and the resulting master SHA has been human-verified,
so hard-coding today's branch SHA would authorize the wrong tree.

**Partition.** Hard-fixed to VALIDATION via
`require_canonical_development_partition`. TEST is rejected before metadata,
stream-cache, source-path, waveform or annotation access, no B4 sealed-test
utility is imported, and no `TEST_ATTEMPT` is ever created. There is no CLI
option that selects a partition.

**Execution order** (frozen; see the module's `PLANNED_EXECUTION_ORDER`):

1. PRE-CLAIM -- runtime identity, clean checkout, expected Git SHA, protocol and
   receipt identities, scorer and locks, TEST firewall, and the absence of an
   existing canonical claim. Nothing scientific is opened yet.
2. START/CLAIM -- an independent `RuntimeIntegrityRecord` per arm; a successful
   START recorded for BOTH arms; both canonical attempts claimed. A failed claim
   stops for human review WITHOUT opening validation.
3. FULL LABEL-BLIND REPLAY -- load the validation input exactly once, prove the
   canonical full replay identity, and replay both arms over the identical
   frozen population with the identical frozen scorer. No annotation is loaded
   until both trajectories are complete.
4. POST-REPLAY -- only then: primary membership, challenge membership,
   source-defined stress intervals, identity joins, frozen evidence.

No result from M2-0 may alter the execution of M2-G: the arms share the frozen
input and scorer and nothing else, and evaluation happens after both replays.

**Bounded memory is mandatory** (§16). The validation corpus is replayed stream
by stream, one `(record_id, channel_index)` at a time per arm, and per-row
evidence is accumulated into compact typed arrays in a disk-backed store rather
than an expanding forest of Python row objects. The M1 host-memory incident
established that corpus-scale Python object retention is unsafe. Scores, gate
evidence and prototype trajectories are stored at full float64 precision so the
frozen drift statistic is reproduced exactly, with no lossy conversion, no
whole-corpus duplicate representation matrix and no two-arm whole-corpus
duplication.
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

PLANNED_EXECUTION_ORDER: Final = (
    "pre_claim_identity_checks",
    "start_and_claim_both_arms",
    "full_label_blind_replay_both_arms",
    "post_replay_population_construction",
    "post_replay_frozen_evidence",
    "persist_and_promote_per_arm",
    "two_arm_suite_without_selection",
)

NO_SCIENTIFIC_EXECUTION_YET: Final = True
"""No canonical M2 development execution has occurred. Flipped only by a real
authorized run, never by the activation work that built this route."""


class M2DevelopmentRunError(RuntimeError):
    """Raised when the canonical development route refuses to proceed."""


def _assert_frozen_arm_order() -> None:
    if tuple(CANONICAL_ARM_ORDER) != tuple(M2_ARMS):
        raise M2DevelopmentRunError(
            f"The canonical arm order {CANONICAL_ARM_ORDER} disagrees with the "
            f"frozen arms {M2_ARMS}."
        )


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


def preflight(*, expected_git_sha: str | None) -> dict[str, Any]:
    """Every check that must pass before any scientific input is opened.

    Deliberately opens nothing: it verifies the runtime identity, the Git
    authorization, the label firewall, the frozen arm order and the partition
    firewall, and returns a record of what it proved.
    """
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        require_runtime_identity,
    )

    _assert_frozen_arm_order()
    git_sha = require_expected_git_sha(expected_git_sha)
    start = require_runtime_identity(
        EnforcementPoint.START, detail="m2_development_preflight"
    )
    firewall = assert_label_firewall()
    partition = require_canonical_development_partition(CANONICAL_DEVELOPMENT_PARTITION)
    for forbidden in ("test",):
        try:
            require_canonical_development_partition(forbidden)
        except M2ExecutionError:
            pass
        else:  # pragma: no cover - the firewall would have to be broken
            raise M2DevelopmentRunError(
                f"The partition firewall accepted {forbidden!r}; refusing to run."
            )
    return {
        "preflight_class": "m2_v1_canonical_development_preflight",
        "git_sha": git_sha,
        "git_dirty": False,
        "partition": partition,
        "arms": list(CANONICAL_ARM_ORDER),
        "arm_selection_performed": False,
        "runtime_identity": start.as_dict(),
        "label_firewall": firewall,
        "data_opened": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


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


def replay_both_arms(
    *,
    stores: dict[str, Any],
    standardizer: Any,
    scorer: Any,
    stream_cache_root: Path,
    feature_root: Path,
) -> None:
    """One pass over the validation input; both arms replayed per stream.

    The input is loaded exactly once and both arms consume the identical frozen
    rows in the frozen order. The arms share nothing but that input and the
    frozen scorer: each keeps its own `M2StreamState`, so no M2-0 result can
    alter M2-G's replay. Each stream's rows and trajectories are released before
    the next stream is read.
    """
    from cardiosentinel.neural.m2_execution import iter_timeline_streams
    from cardiosentinel.neural.m2_policy import replay_stream

    for key, rows in iter_timeline_streams(
        CANONICAL_DEVELOPMENT_PARTITION,
        stream_cache_root=stream_cache_root,
        feature_root=feature_root,
    ):
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


def execute_canonical_development(
    *,
    expected_git_sha: str | None,
    execute: bool = False,
    run_root: Path | None = None,
    experiment_id: str | None = None,
    stream_cache_root: Path | None = None,
    feature_root: Path | None = None,
    m1_run_root: Path | None = None,
    p1_cache_root: Path | None = None,
    evidence_root: Path | None = None,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """The canonical two-arm development run.

    Without `execute=True` this performs the preflight and returns the plan; it
    opens no partition and consumes no attempt. With `execute=True` it runs the
    frozen order end to end: claim both arms BEFORE the shared validation input
    is read, replay both arms label-blind, then -- and only then -- construct
    the metric populations, join by identity and compute the frozen evidence.
    """
    plan = preflight(expected_git_sha=expected_git_sha)
    plan["planned_execution_order"] = list(PLANNED_EXECUTION_ORDER)
    plan["executed"] = False
    if not execute:
        return plan
    return _run(
        plan=plan,
        run_root=run_root,
        experiment_id=experiment_id,
        stream_cache_root=stream_cache_root,
        feature_root=feature_root,
        m1_run_root=m1_run_root,
        p1_cache_root=p1_cache_root,
        evidence_root=evidence_root,
        dataset_root=dataset_root,
    )


def _run(
    *,
    plan: dict[str, Any],
    run_root: Path | None,
    experiment_id: str | None,
    stream_cache_root: Path | None,
    feature_root: Path | None,
    m1_run_root: Path | None,
    p1_cache_root: Path | None,
    evidence_root: Path | None,
    dataset_root: Path | None,
) -> dict[str, Any]:
    """The frozen execution order. Never invoked by the activation task."""
    from cardiosentinel.neural.m2_evaluation import (
        build_challenge_bundle,
        build_primary_bundle,
        cold_start_stratified_evidence,
        contamination_evidence,
        false_alarm_evidence,
        policy_evidence,
        window_evidence,
    )
    from cardiosentinel.neural.m2_evidence_store import M2EvidenceStore
    from cardiosentinel.neural.m2_execution import (
        canonical_replay_population,
        load_distance_standardizer,
        m2_execution_identity,
        streaming_input_identity,
    )
    from cardiosentinel.neural.m2_gate_derivation import (
        DEFAULT_FEATURE_ROOT,
        DEFAULT_M1_RUN_ROOT,
        DEFAULT_STREAM_CACHE_ROOT,
    )
    from cardiosentinel.neural.m2_persistence import (
        claim_run_directory,
        finalize_and_promote_arm_result,
    )
    from cardiosentinel.neural.m2_populations import (
        challenge_evaluation_population,
        primary_evaluation_population,
        prove_population_containment,
    )
    from cardiosentinel.neural.m2_scorer import (
        M1L_CLASSIFICATION_THRESHOLD,
        load_frozen_m1l_scorer,
    )
    from cardiosentinel.neural.m2_stress_intervals import (
        build_stress_selection_from_parsed,
    )
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        RuntimeIntegrityRecord,
        require_runtime_identity,
    )

    stream_cache_root = Path(stream_cache_root or DEFAULT_STREAM_CACHE_ROOT)
    feature_root = Path(feature_root or DEFAULT_FEATURE_ROOT)
    m1_run_root = Path(m1_run_root or DEFAULT_M1_RUN_ROOT)
    if (
        run_root is None
        or experiment_id is None
        or evidence_root is None
        or dataset_root is None
    ):
        raise M2DevelopmentRunError(
            "A canonical development execution requires an explicit run root, "
            "experiment id, evidence root and dataset root."
        )

    # START + CLAIM both arms before any development data is opened.
    runtimes = {arm: RuntimeIntegrityRecord() for arm in CANONICAL_ARM_ORDER}
    for arm in CANONICAL_ARM_ORDER:
        require_runtime_identity(
            EnforcementPoint.START, record=runtimes[arm], detail=f"m2_replay:{arm}"
        )
    claims = {
        arm: claim_run_directory(
            Path(run_root), experiment_id, arm, runtime=runtimes[arm]
        )
        for arm in CANONICAL_ARM_ORDER
    }

    # FULL LABEL-BLIND REPLAY.
    replay_population, replay_stable_ids, manifest = canonical_replay_population(
        CANONICAL_DEVELOPMENT_PARTITION, stream_cache_root=stream_cache_root
    )
    scorer = load_frozen_m1l_scorer(m1_run_root)
    standardizer = load_distance_standardizer(stream_cache_root)
    stores = {
        arm: M2EvidenceStore(root=Path(evidence_root) / arm, arm=arm)
        for arm in CANONICAL_ARM_ORDER
    }
    replay_both_arms(
        stores=stores,
        standardizer=standardizer,
        scorer=scorer,
        stream_cache_root=stream_cache_root,
        feature_root=feature_root,
    )
    store_manifests = {arm: stores[arm].finalize() for arm in CANONICAL_ARM_ORDER}

    # POST-REPLAY. No annotation was loaded until here.
    primary = primary_evaluation_population(Path(p1_cache_root or stream_cache_root))
    challenge = challenge_evaluation_population(feature_root)
    containment = prove_population_containment(
        replay_population=replay_population,
        replay_stable_ids=replay_stable_ids,
        primary=primary,
        challenge=challenge,
    )
    del replay_stable_ids

    primary_annotations = build_primary_annotations(
        primary, stream_cache_root=stream_cache_root
    )
    challenge_annotations = build_challenge_annotations(challenge)
    stress = build_stress_selection_from_parsed(
        parsed_validation_annotations(
            dataset_root=Path(dataset_root), feature_root=feature_root
        )
    )
    stress_identity = stress.identity()

    results: dict[str, dict[str, Any]] = {}
    for arm in CANONICAL_ARM_ORDER:
        table = stores[arm].score_table()
        primary_bundle = build_primary_bundle(
            arm, table, primary_annotations, primary_population=primary
        )
        challenge_bundle = build_challenge_bundle(
            arm, table, challenge_annotations, challenge_population=challenge
        )
        trajectories = {
            key: stores[arm].load_trajectory(key)
            for key in {i.stream_key for i in stress.intervals}
        }
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
            "replay_population_identity": replay_population.identity(),
            "primary_evaluation_population_identity": (
                primary_bundle.population_identity()
            ),
            "challenge_evaluation_population_identity": (
                challenge_bundle.population_identity()
            ),
            "stress_interval_selection_identity": stress_identity,
            "population_containment_proof": containment,
            "evidence_store_identity": store_manifests[arm],
            "policy_evidence": policy_evidence(
                stores[arm].admission, replay_population=replay_population
            ),
            "window_evidence": window_evidence(primary_bundle),
            "false_alarm_evidence": false_alarm_evidence(
                primary_bundle=primary_bundle, challenge_bundle=challenge_bundle
            ),
            "cold_start_evidence": cold_start_stratified_evidence(primary_bundle),
            "contamination_evidence": contamination_evidence(
                trajectories,
                stress_intervals=list(stress.evaluation_intervals()),
                replay_population=replay_population,
                stress_selection_identity=stress_identity,
            ),
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
        del primary_bundle, challenge_bundle, trajectories, table

    plan["executed"] = True
    plan["arm_results"] = {arm: results[arm]["experiment_id"] for arm in results}
    plan["memory_selected"] = None
    plan["memory_selection_performed"] = False
    return plan


def build_parser() -> argparse.ArgumentParser:
    """The CLI. It exposes no partition option, by design."""
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


if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())


def parsed_validation_annotations(*, dataset_root: Path, feature_root: Path):
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
        record = read_record(Path(dataset_root), record_id)
        yield read_annotations(Path(dataset_root), record)
