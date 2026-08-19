"""The one canonical TRAIN-only T2 route.

There is exactly one public scientific command, and it trains **both** frozen
candidates in one attempt:

    /home/AI_POC/venvs/tactics/bin/python \\
      -m cardiosentinel.neural.t2_development_run \\
      --execute-canonical-training \\
      --expected-git-sha <HUMAN_AUTHORIZED_MERGED_SHA>

There is deliberately no `--arm`, `--epoch`, `--lr`, `--batch-size`, `--tbptt`,
`--seed`, `--device`, `--threshold`, `--retry`, `--force`, `--validation` or
`--test`. Every one of those would be a scientific choice the frozen protocol
has already made, or a firewall bypass.

**There is no activation switch for training.** The human authorization
mechanism is the exact merged Git SHA supplied through `--expected-git-sha`,
matching the reviewed one-shot pattern P1, M1, M2 and U1 already use. The route
is complete; it simply remains unexecuted until a human runs it against a merged
commit.

`--execute-canonical-outer-validation` exists so the route can be reviewed, and
it refuses: the activation state is `False`, and the refusal fires before any
VALIDATION path, array or label is touched.

**The choreography, and why it is in this order.** Preflight proves Git, the
protocol bytes, the execution-spec bytes and that the claim is unconsumed --
all of it small immutable identity material. Only then is the runtime observed
at START and the attempt claimed. The real TRAIN store and the real target
authority are opened **after** the claim, so a corrupted input discovered there
consumes the attempt and is recorded honestly as an input-lineage failure
rather than being quietly discovered by a pre-run scan that leaves no trace.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cardiosentinel.data.provenance import git_provenance
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityRecord,
)
from cardiosentinel.neural.t2_models import model_identity, seed_everything
from cardiosentinel.neural.t2_persistence import (
    ARM_RESULT_NAME,
    ARM_SELECTION_PENDING,
    CHECKPOINT_LOCK_NAME,
    CHECKPOINT_NAME,
    INTERNAL_SPLIT_NAME,
    POPULATION_NAME,
    PREFLIGHT_NAME,
    RESULT_CLASS,
    STAGE_TRAINING_START,
    T2_EXECUTION_SPEC_SHA256,
    T2_EXPERIMENT_IDENTITY,
    T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED,
    T2_RUN_ROOT,
    T2_TRAINING_ATTEMPT_ID,
    T2ActivationError,
    T2PersistenceError,
    canonical_execution_device,
    claim_t2_run_directory,
    finalize_and_promote_t2_result,
    model_parameter_device,
    observe_t2_runtime_stage,
    promote_checkpoint,
    promote_component,
    record_t2_attempt_failure,
    require_deterministic_execution,
    require_execution_device_agreement,
    require_outer_validation_authorized,
    require_single_runtime,
    require_unclaimed_t2_attempt,
    runtime_provenance,
    stage_pre_model_construction,
    validate_t2_execution_spec,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARMS,
    T2_INTERNAL_DEV_SUBJECTS,
    T2_INTERNAL_SPLIT_SHA256,
    T2_PROTOCOL_SHA256,
    T2_SPLIT_PATH,
    assign_internal_split,
    require_capacity_envelope,
    require_full_chronological_population,
    validate_internal_split,
    validate_t2_protocol_document,
)
from cardiosentinel.neural.t2_timeline import (
    CORPUS_MANIFEST,
    T2Timeline,
    ordered_stable_id_digest_for_rows,
    resolve_timeline_target_families,
)
from cardiosentinel.neural.t2_training import (
    T2TimelineReader,
    fit_class_weight_evidence,
    restore_model_state,
    train_arm,
)

FORBIDDEN_OPTIONS: Final = (
    "--arm",
    "--epoch",
    "--epochs",
    "--lr",
    "--learning-rate",
    "--batch-size",
    "--tbptt",
    "--seed",
    "--device",
    "--threshold",
    "--retry",
    "--force",
    "--validation",
    "--test",
)

TRAIN_PARTITION: Final = "train"


class T2RunError(RuntimeError):
    """Raised when the canonical T2 route cannot proceed."""


def require_expected_git_sha(expected_git_sha: str | None) -> str:
    """A canonical run names the merged commit it believes it is executing."""
    if not expected_git_sha:
        raise T2RunError(
            "--expected-git-sha is required: a canonical T2 run must name the "
            "human-authorized merged commit it believes it is executing."
        )
    git = git_provenance(REPOSITORY_ROOT)
    if git["git_dirty"]:
        raise T2RunError(
            "The working tree is dirty. Canonical T2 evidence requires a clean "
            "checkout, matching the existing P1/M1/M2/U1 convention."
        )
    if git["git_sha"] != expected_git_sha:
        raise T2RunError(
            f"The checkout is at {git['git_sha']}, but the run expects "
            f"{expected_git_sha}. Nothing is executed."
        )
    return str(git["git_sha"])


def preflight(expected_git_sha: str | None) -> dict[str, Any]:
    """Everything provable before the claim, and before any timeline access."""
    git_sha = require_expected_git_sha(expected_git_sha)
    protocol_sha = validate_t2_protocol_document()
    execution_spec_sha = validate_t2_execution_spec()
    unclaimed = require_unclaimed_t2_attempt(T2_RUN_ROOT, T2_TRAINING_ATTEMPT_ID)
    return {
        "preflight_class": "t2_training_preflight",
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "attempt_id": T2_TRAINING_ATTEMPT_ID,
        "git_sha": git_sha,
        "t2_protocol_sha256": protocol_sha,
        "t2_execution_spec_sha256": execution_spec_sha,
        "arms": list(T2_ARMS),
        "claim_state": unclaimed,
        "outer_validation_execution_authorized": (
            T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED
        ),
        "partition_accessed": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "per_row_train_input_opened_before_claim": False,
    }


# ---------------------------------------------------------------------------
# The private execution seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T2TrainingSources:
    """Where the canonical body reads from. Private; never a CLI option.

    The public route constructs exactly one of these, with every field at its
    canonical default. Tests construct one pointing at a synthetic on-disk
    timeline, a synthetic target authority and a temporary run root -- and then
    drive the *same* orchestration function, so a defect in the assembly itself
    cannot hide behind passing unit tests of its parts.

    Note what is NOT here: epochs, learning rate, TBPTT length, seed, device,
    threshold rule, split rule. Those are frozen science and are read from the
    protocol, not injected.
    """

    run_root: Path = T2_RUN_ROOT
    attempt_id: str = T2_TRAINING_ATTEMPT_ID
    stream_cache_root: Path | None = None
    corpus_manifest: Path = CORPUS_MANIFEST
    split_manifest: Path = T2_SPLIT_PATH
    canonical: bool = True

    def open_timeline(self) -> T2Timeline:
        return T2Timeline(TRAIN_PARTITION, root=self.stream_cache_root)


@dataclass(slots=True)
class _Exposure:
    """What the attempt has actually touched, for an honest failure receipt."""

    stage: str = "claim_canonical_attempt"
    train_store_opened: bool = False
    target_authority_opened: bool = False
    optimizer_stepped: bool = False
    internal_dev_scored: bool = False
    threshold_derived: bool = False
    execution_device: str | None = None
    # The arm currently executing, which is NOT `arms_completed[-1]`: if the
    # GRU completes and the S4D then fails, the last completed arm is the GRU
    # and naming it in the receipt would attribute the failure to the arm that
    # worked. This is set before an arm's scientific execution begins and
    # cleared only after that arm completes.
    current_arm: str | None = None
    arms_completed: list[str] = field(default_factory=list)

    def begin_arm(self, arm: str) -> None:
        self.current_arm = arm

    def complete_arm(self, arm: str) -> None:
        self.arms_completed.append(arm)
        self.current_arm = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "train_store_opened": self.train_store_opened,
            "target_authority_opened": self.target_authority_opened,
            "optimizer_stepped": self.optimizer_stepped,
            "internal_dev_scored": self.internal_dev_scored,
            "threshold_derived": self.threshold_derived,
            "execution_device": self.execution_device,
            "current_arm": self.current_arm,
            "arms_completed": list(self.arms_completed),
            "arm_selection_performed": False,
            "outer_validation_accessed": False,
            "test_accessed": False,
        }


def _partition_subjects(split_manifest: Path) -> dict[str, tuple[str, ...]]:
    import json

    payload = json.loads(Path(split_manifest).read_text())
    return {
        name: tuple(str(value) for value in block["subjects"])
        for name, block in payload["partitions"].items()
    }


def execute_canonical_training(expected_git_sha: str | None) -> dict[str, Any]:
    """The one TRAIN-only canonical route, complete and executable.

    Preflight proves the commit, the frozen protocol, the frozen execution spec
    and that the attempt is unconsumed. Then the runtime is observed at START,
    the claim directory is created, and everything the claim authorises runs to
    completion: real store validation, the real target join, the deterministic
    48/8 split, the FIT-only class weight, both arms in frozen order, one
    retained checkpoint and one frozen internal-dev threshold each, the
    canonical result, and the canonical lock.

    No arm is selected. No retry is ever performed.
    """
    checks = preflight(expected_git_sha)
    return _execute_training_attempt(checks, T2TrainingSources())


def _execute_training_attempt(
    checks: dict[str, Any], sources: T2TrainingSources
) -> dict[str, Any]:
    """Claim the attempt, then run the frozen body under one failure receipt."""
    runtime = RuntimeIntegrityRecord()
    observe_t2_runtime_stage(
        runtime, point=EnforcementPoint.START, detail=STAGE_TRAINING_START
    )
    claimed = claim_t2_run_directory(
        sources.run_root, sources.attempt_id, runtime=runtime
    )
    exposure = _Exposure()
    try:
        return _run_after_claim(
            checks=checks,
            sources=sources,
            claimed=claimed,
            runtime=runtime,
            exposure=exposure,
        )
    except BaseException as error:
        # One arm failing fails the WHOLE attempt. Completed evidence is left
        # exactly where it is as forensic material, the receipt is additive and
        # lives outside the claim, and nothing is rerun, reseeded or renamed.
        record_t2_attempt_failure(
            sources.run_root,
            claimed,
            exception=error,
            stage=exposure.stage,
            arm=exposure.current_arm,
            exposure=exposure.as_dict(),
            runtime=runtime,
        )
        raise


def _run_after_claim(
    *,
    checks: dict[str, Any],
    sources: T2TrainingSources,
    claimed: Any,
    runtime: RuntimeIntegrityRecord,
    exposure: _Exposure,
) -> dict[str, Any]:
    """Everything the claim authorises, in the frozen order."""
    exposure.stage = "validate_train_store"
    timeline = sources.open_timeline()
    exposure.train_store_opened = True
    try:
        return _train_both_arms(
            checks=checks,
            sources=sources,
            claimed=claimed,
            runtime=runtime,
            exposure=exposure,
            timeline=timeline,
        )
    finally:
        timeline.close()


def _train_both_arms(
    *,
    checks: dict[str, Any],
    sources: T2TrainingSources,
    claimed: Any,
    runtime: RuntimeIntegrityRecord,
    exposure: _Exposure,
    timeline: T2Timeline,
) -> dict[str, Any]:
    exposure.stage = "resolve_target_authority"
    family_codes, target_identity = resolve_timeline_target_families(
        timeline, manifest_path=sources.corpus_manifest
    )
    exposure.target_authority_opened = True
    reader = T2TimelineReader(timeline, family_codes)

    exposure.stage = "construct_internal_split"
    subjects_by_partition = _partition_subjects(sources.split_manifest)
    split = assign_internal_split(timeline.subjects())
    validate_internal_split(
        split,
        validation_subjects=subjects_by_partition["validation"],
        test_subjects=subjects_by_partition["test"],
    )
    if split["split_sha256"] != T2_INTERNAL_SPLIT_SHA256:
        raise T2RunError(
            f"The internal split digests to {split['split_sha256']}, not the "
            f"frozen {T2_INTERNAL_SPLIT_SHA256}. The 48/8 partition is frozen "
            "and is never re-derived differently."
        )
    if tuple(split["internal_dev_subjects"]) != T2_INTERNAL_DEV_SUBJECTS:
        raise T2RunError(
            "The internal-dev subject set is not the frozen one; nothing proceeds."
        )
    fit_subjects = tuple(split["fit_subjects"])
    internal_dev_subjects = tuple(split["internal_dev_subjects"])
    fit_streams = timeline.streams_for_subjects(set(fit_subjects))
    internal_dev_streams = timeline.streams_for_subjects(set(internal_dev_subjects))
    if not fit_streams or not internal_dev_streams:
        raise T2RunError("The internal split produced an empty stream selection.")
    require_full_chronological_population(
        offered_row_count=sum(stream.row_count for stream in timeline.streams()),
        full_stream_row_count=timeline.row_count,
    )

    # Determinism is established ONCE, before the first runtime reading. Doing
    # it lazily inside the first arm's construction would make arm A observe
    # `deterministic_algorithms: False` and arm B observe `True`, and the
    # same-runtime check would then correctly refuse a comparison that was
    # never actually mixed. Each arm still reseeds from the same fresh origin.
    exposure.stage = "establish_deterministic_runtime"
    determinism = seed_everything()
    # ONE device, selected once, before any arm exists. There is no flag and
    # no override; if determinism cannot be satisfied on it, execution STOPS
    # rather than falling back to the CPU behind the provenance's back.
    execution_device = canonical_execution_device()
    determinism.update(require_deterministic_execution(execution_device))
    exposure.execution_device = str(execution_device)

    exposure.stage = "derive_fit_class_weight"
    class_weight = fit_class_weight_evidence(
        reader,
        fit_streams,
        fit_subjects=fit_subjects,
        internal_dev_subjects=internal_dev_subjects,
    )

    population = {
        "artifact_class": "t2_training_population",
        "partition": TRAIN_PARTITION,
        "train_timeline_identity": timeline.identity(),
        "target_authority_identity": target_identity,
        # Re-derived from the persisted stable_id.npy, not copied from the
        # manifest: this is what proves the replayed population is the one M1
        # promoted, and not merely one of the same length.
        "rederived_ordered_stable_id_sha256": ordered_stable_id_digest_for_rows(
            timeline
        ),
        "population_identity_proves": [
            "same_rows",
            "same_ordering",
            "same_physical_timeline",
            "same_category_authority",
        ],
        "row_count": timeline.row_count,
        "stream_count": len(timeline.streams()),
        "subject_count": len(timeline.subjects()),
        "fit_subjects": list(fit_subjects),
        "internal_dev_subjects": list(internal_dev_subjects),
        "fit_stream_count": len(fit_streams),
        "internal_dev_stream_count": len(internal_dev_streams),
        "fit_availability": reader.availability_census(fit_streams),
        "internal_dev_availability": reader.availability_census(internal_dev_streams),
        "class_weight": class_weight,
        "deterministic_runtime": determinism,
        "execution_device": str(execution_device),
        "internal_dev_contributes_optimizer_gradient": False,
        "negative_sampling_applied": False,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    if (
        population["rederived_ordered_stable_id_sha256"]
        != (timeline.manifest["ordered_stable_id_sha256"])
    ):
        raise T2RunError(
            "The ordered stable-id digest re-derived from the persisted store "
            "does not match the M1 manifest. The replayed population is not the "
            "promoted one."
        )

    exposure.stage = "promote_preflight_and_population"
    promote_component(claimed, PREFLIGHT_NAME, dict(checks), runtime=runtime)
    promote_component(claimed, INTERNAL_SPLIT_NAME, dict(split), runtime=runtime)
    promote_component(claimed, POPULATION_NAME, population, runtime=runtime)

    arm_results: dict[str, Any] = {}
    checkpoint_sha256: dict[str, str] = {}
    checkpoint_lock_sha256: dict[str, str] = {}
    checkpoint_lock_self_sha256: dict[str, str] = {}
    thresholds: dict[str, Any] = {}
    parameter_counts: dict[str, int] = {}
    runtimes: list[dict[str, Any]] = []

    for arm in T2_ARMS:
        exposure.stage = f"train_arm:{arm}"
        exposure.begin_arm(arm)
        observed_runtime = runtime_provenance(execution_device)
        if runtimes:
            require_single_runtime(runtimes[0], observed_runtime)
        runtimes.append(observed_runtime)
        device_proof: dict[str, Any] = {}

        def _before_model_construction(name: str) -> None:
            observe_t2_runtime_stage(
                runtime,
                point=EnforcementPoint.PRE_PROMOTION,
                detail=stage_pre_model_construction(name),
            )

        def _on_model_constructed(
            _name: str, model: Any, _runtime: dict[str, Any] = observed_runtime
        ) -> None:
            device_proof.update(
                require_execution_device_agreement(
                    _runtime, model_parameter_device(model)
                )
            )

        trained = train_arm(
            arm,
            reader,
            fit_streams=fit_streams,
            internal_dev_streams=internal_dev_streams,
            internal_dev_subjects=internal_dev_subjects,
            pos_weight=class_weight["positive_class_weight"],
            device=execution_device,
            before_model_construction=_before_model_construction,
            on_model_constructed=_on_model_constructed,
        )
        require_execution_device_agreement(
            observed_runtime, trained["model_parameter_device"]
        )
        exposure.optimizer_stepped = True
        exposure.internal_dev_scored = True
        exposure.threshold_derived = True

        exposure.stage = f"promote_checkpoint:{arm}"
        # The identity is read off a model rebuilt from the retained state, not
        # off the live trainer: what is attested is what will be promoted.
        identity = model_identity(
            restore_model_state(arm, trained["state_dict"], device=execution_device)
        )
        parameter_counts[arm] = int(identity["trainable_parameters"])
        checkpoint_lock = promote_checkpoint(
            claimed,
            arm,
            trained["state_dict"],
            identity={
                "model_identity": identity,
                "best_epoch": trained["best_epoch"],
                "best_internal_dev_pooled_auprc": (
                    trained["best_internal_dev_pooled_auprc"]
                ),
                "internal_dev_score_sha256": trained["internal_dev_score_sha256"],
                "internal_dev_threshold": trained["internal_dev_threshold"][
                    "threshold"
                ],
                "runtime": observed_runtime,
                "execution_device_proof": dict(device_proof),
            },
            runtime=runtime,
        )
        checkpoint_sha256[arm] = checkpoint_lock["checkpoint_sha256"]
        checkpoint_lock_self_sha256[arm] = checkpoint_lock["checkpoint_lock_sha256"]
        checkpoint_lock_sha256[arm] = _file_sha256(
            claimed.run_dir / CHECKPOINT_LOCK_NAME[arm]
        )
        thresholds[arm] = trained["internal_dev_threshold"]

        arm_payload = {
            "artifact_class": "t2_arm_training_result",
            "arm": arm,
            "model_identity": identity,
            "checkpoint_file": CHECKPOINT_NAME[arm],
            "checkpoint_sha256": checkpoint_sha256[arm],
            "checkpoint_lock_sha256": checkpoint_lock_self_sha256[arm],
            "best_epoch": trained["best_epoch"],
            "best_internal_dev_pooled_auprc": (
                trained["best_internal_dev_pooled_auprc"]
            ),
            "internal_dev_score_sha256": trained["internal_dev_score_sha256"],
            "internal_dev_primary_row_count": (
                trained["internal_dev_primary_row_count"]
            ),
            "epochs": trained["epochs"],
            "epochs_completed": trained["epochs_completed"],
            "early_stopped": trained["early_stopped"],
            "checkpoint_selection": trained["checkpoint_selection"],
            "internal_dev_threshold": trained["internal_dev_threshold"],
            "threshold_derived_from_best_checkpoint": True,
            "threshold_derived_during_epoch_selection": False,
            "threshold_derived_from_outer_validation": False,
            "positive_class_weight": trained["positive_class_weight"],
            "runtime": observed_runtime,
            "execution_device": trained["execution_device"],
            "model_parameter_device": trained["model_parameter_device"],
            "execution_device_proof": dict(device_proof),
            "arm_selection_status": ARM_SELECTION_PENDING,
            "arm_selected": None,
            "outer_validation_accessed": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        }
        promote_component(claimed, ARM_RESULT_NAME[arm], arm_payload, runtime=runtime)
        arm_results[arm] = arm_payload
        exposure.complete_arm(arm)

    exposure.stage = "promote_canonical_result"
    require_capacity_envelope(parameter_counts)
    result = {
        "artifact_class": RESULT_CLASS,
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "git_sha": checks["git_sha"],
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "internal_split_sha256": split["split_sha256"],
        "train_timeline_identity": timeline.identity(),
        "target_authority_identity": target_identity,
        "fit_subjects": list(fit_subjects),
        "internal_dev_subjects": list(internal_dev_subjects),
        "fit_positive_count": class_weight["fit_positive_count"],
        "fit_negative_count": class_weight["fit_negative_count"],
        "positive_class_weight": class_weight["positive_class_weight"],
        "component_sha256": dict(claimed.promoted),
        "checkpoint_sha256": dict(checkpoint_sha256),
        "checkpoint_lock_sha256": dict(checkpoint_lock_sha256),
        "internal_dev_thresholds": dict(thresholds),
        "trainable_parameters": dict(parameter_counts),
        "execution_device": str(execution_device),
        "runtime": dict(runtimes[0]),
        "arms": list(T2_ARMS),
        "arm_selection_status": ARM_SELECTION_PENDING,
        "arm_selected": None,
        "arm_compared_on_train_evidence": False,
        "outer_validation_accessed": False,
        "automatic_retry_performed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    provenance = {
        "train_timeline_identity": timeline.identity(),
        "target_authority_identity": target_identity,
        "fit_subjects": list(fit_subjects),
        "internal_dev_subjects": list(internal_dev_subjects),
        "fit_positive_count": class_weight["fit_positive_count"],
        "fit_negative_count": class_weight["fit_negative_count"],
        "positive_class_weight": class_weight["positive_class_weight"],
        "checkpoint_sha256": dict(checkpoint_sha256),
        "checkpoint_lock_sha256": dict(checkpoint_lock_sha256),
        "checkpoint_lock_self_sha256": dict(checkpoint_lock_self_sha256),
        "internal_dev_thresholds": dict(thresholds),
        "execution_device": str(execution_device),
        "runtime": dict(runtimes[0]),
    }
    promoted = finalize_and_promote_t2_result(
        claimed, result=result, provenance=provenance, runtime=runtime
    )
    return {
        "report_class": "t2_canonical_training_completion",
        "execution_device": str(execution_device),
        "attempt_id": claimed.attempt_id,
        "experiment_identity": T2_EXPERIMENT_IDENTITY,
        "run_dir": str(claimed.run_dir),
        "status": promoted["status"]["status"],
        "git_sha": checks["git_sha"],
        "t2_protocol_sha256": T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": T2_EXECUTION_SPEC_SHA256,
        "internal_split_sha256": split["split_sha256"],
        "fit_subject_count": len(fit_subjects),
        "internal_dev_subject_count": len(internal_dev_subjects),
        "fit_positive_count": class_weight["fit_positive_count"],
        "fit_negative_count": class_weight["fit_negative_count"],
        "positive_class_weight": class_weight["positive_class_weight"],
        "arm_results": {
            arm: {
                "best_epoch": payload["best_epoch"],
                "epochs_completed": payload["epochs_completed"],
                "early_stopped": payload["early_stopped"],
                "internal_dev_threshold": payload["internal_dev_threshold"][
                    "threshold"
                ],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
            for arm, payload in arm_results.items()
        },
        "experiment_lock_sha256": promoted["lock"]["experiment_lock_sha256"],
        "artifact_sha256": dict(promoted["lock"]["artifact_sha256"]),
        "checkpoint_sha256": dict(checkpoint_sha256),
        "checkpoint_lock_sha256": dict(checkpoint_lock_sha256),
        "runtime_enforcement_stages": list(
            promoted["lock"]["runtime_enforcement_stages"]
        ),
        "arm_selection_status": ARM_SELECTION_PENDING,
        "arm_selected": None,
        "human_review_required": True,
        "automatic_retry_performed": False,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


def _file_sha256(path: Path) -> str:
    from cardiosentinel.data.provenance import sha256_file

    return sha256_file(Path(path))


def execute_canonical_outer_validation(expected_git_sha: str | None) -> dict[str, Any]:
    """Refuses before any VALIDATION path, array or label is touched.

    After activation this forwards the human-authorized merged commit through
    to the outer worker, which proves it -- and a clean tree, and an unconsumed
    outer attempt -- before claiming and before opening anything.
    """
    require_outer_validation_authorized()
    from cardiosentinel.neural.t2_evaluation import (  # pragma: no cover
        execute_canonical_outer_validation as run_outer_validation,
    )

    return run_outer_validation(  # pragma: no cover - unreachable while gated
        expected_git_sha
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardiosentinel.neural.t2_development_run",
        description=(
            "The canonical TRAIN-only T2 route. Trains both frozen candidates; "
            "exposes no scientific option."
        ),
    )
    parser.add_argument(
        "--execute-canonical-training",
        action="store_true",
        help="Run the one canonical TRAIN-only attempt for both frozen arms.",
    )
    parser.add_argument(
        "--execute-canonical-outer-validation",
        action="store_true",
        help=(
            "Reserved. Refuses: outer VALIDATION execution is not authorized by "
            "the frozen activation state."
        ),
    )
    parser.add_argument(
        "--expected-git-sha",
        default=None,
        help="The human-authorized merged commit this run believes it executes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.execute_canonical_outer_validation:
            execute_canonical_outer_validation(args.expected_git_sha)
            return 0
        if not args.execute_canonical_training:
            parser.error("--execute-canonical-training is required.")
        report = execute_canonical_training(args.expected_git_sha)
        print(report["experiment_lock_sha256"])
        return 0
    except T2ActivationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    except (T2RunError, T2PersistenceError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2


__all__ = [
    "FORBIDDEN_OPTIONS",
    "T2RunError",
    "T2TrainingSources",
    "build_parser",
    "execute_canonical_outer_validation",
    "execute_canonical_training",
    "main",
    "preflight",
    "require_expected_git_sha",
]


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
