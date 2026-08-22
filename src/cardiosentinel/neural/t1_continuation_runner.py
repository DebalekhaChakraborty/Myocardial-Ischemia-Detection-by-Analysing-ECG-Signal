"""The continuation execution engine. Built, gated, and unable to run.

This is the executor for the one measurement continuation the recovery amendment
authorizes. It exists so that the continuation *can* be run once a human decides
to run it, and it refuses at its first stage until then:
`T1_CONTINUATION_AUTHORIZED` is `False`, and arming it is a separate governance
act, not a flag this module reads from an argument.

**Stage order, and why it is this order.**

```
 1 require_continuation_authorized      permission, before anything is resolved
 2 require_continuation_identity        the name, before a path is built
 3 validate the frozen amendment        the rule, before the evidence it governs
 4 verify_predecessor                   all 20 digests, or refuse entirely
 5 prove_negative_capability            what this graph cannot do
 6 consume_oof_state_trace              read the persisted trace; never regenerate
 7 require_trace_matches_selections     Layer 3: the trace is the promoted decision
 8 claim the continuation run root      first write of the whole run
 9 per fold: labels -> measure -> promote
10 assemble and promote the six artifacts
11 attestation, then the experiment lock
```

Permission first, evidence last. A refused run resolves no artifact, opens no
label and creates no directory -- every check above stage 8 is read-only, so an
unauthorized invocation touches nothing at all.

Labels are opened **after** the predecessor verifies and per fold, never in bulk.
The barrier the canonical run enforced at §16 is re-proved here rather than
assumed: each fold's promoted selection digest is re-verified immediately before
that fold's labels are opened.

**No automatic retry, ever.** A post-claim failure of the continuation consumes
it. The amendment authorizes one, and no second identity is predeclared, so this
module has no retry path, no resume path and no `--force`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural.t1_continuation_attestation import (
    build_continuation_attestation,
    continuation_provenance,
)
from cardiosentinel.neural.t1_continuation_gate import (
    ContinuationCounters,
    instrumented_protocol_entry_points,
    prove_negative_capability,
)
from cardiosentinel.neural.t1_continuation_labels import (
    held_out_labels_for_fold,
    require_labels_cover_trace,
)
from cardiosentinel.neural.t1_continuation_measurement import (
    consume_oof_state_trace,
    measure_fold,
    require_trace_matches_selections,
)
from cardiosentinel.neural.t1_continuation_persistence import (
    build_continuation_held_out_evidence,
    promote_continuation_held_out_evaluation,
)
from cardiosentinel.neural.t1_continuation_predecessor import verify_predecessor
from cardiosentinel.neural.t1_continuation_results import (
    CONTINUATION_RESULT_ARTIFACTS,
    build_bootstrap,
    build_challenge_evidence,
    build_experiment_lock,
    build_final_configuration,
    build_oof_result,
    build_subject_evidence,
)
from cardiosentinel.neural.t1_continuation_spec import (
    CONSUMED_ATTEMPT_DIR,
    CONTINUATION_ATTEMPT_ID,
    CONTINUATION_ATTESTATION_NAME,
    CONTINUATION_RUN_ROOT,
    CONTINUATION_RUN_ROOT_RELATIVE,
    PREDECESSOR_FOLD_SELECTIONS,
    require_continuation_authorized,
    require_continuation_identity,
)

#: The modules the negative capability gate proves, including this one.
CONTINUATION_PROVEN_MODULES: Final = (
    "cardiosentinel.neural.t1_continuation_spec",
    "cardiosentinel.neural.t1_continuation_predecessor",
    "cardiosentinel.neural.t1_continuation_gate",
    "cardiosentinel.neural.t1_continuation_measurement",
    "cardiosentinel.neural.t1_continuation_attestation",
    "cardiosentinel.neural.t1_continuation_persistence",
    "cardiosentinel.neural.t1_continuation_results",
    "cardiosentinel.neural.t1_continuation_runner",
)

STAGE_AUTHORIZE: Final = "require_continuation_authorization"
STAGE_IDENTITY: Final = "require_continuation_identity"
STAGE_VERIFY_PREDECESSOR: Final = "verify_predecessor"
STAGE_PROVE_CAPABILITY: Final = "prove_negative_capability"
STAGE_CONSUME_TRACE: Final = "consume_persisted_oof_state_trace"
STAGE_MATCH_SELECTIONS: Final = "require_trace_matches_selections"
STAGE_CLAIM: Final = "claim_continuation_run_root"
STAGE_FOLDS: Final = "measure_folds_against_held_out_labels"
STAGE_RESULTS: Final = "assemble_run_level_artifacts"
STAGE_ATTEST: Final = "promote_execution_attestation"
STAGE_LOCK: Final = "promote_experiment_lock"

CONTINUATION_STAGE_ORDER: Final = (
    STAGE_AUTHORIZE,
    STAGE_IDENTITY,
    STAGE_VERIFY_PREDECESSOR,
    STAGE_PROVE_CAPABILITY,
    STAGE_CONSUME_TRACE,
    STAGE_MATCH_SELECTIONS,
    STAGE_CLAIM,
    STAGE_FOLDS,
    STAGE_RESULTS,
    STAGE_ATTEST,
    STAGE_LOCK,
)

#: Stages that write. Everything before the claim is read-only, which is what
#: makes an unauthorized or refused invocation leave no trace of having happened.
WRITING_STAGES: Final = (
    STAGE_CLAIM,
    STAGE_FOLDS,
    STAGE_RESULTS,
    STAGE_ATTEST,
    STAGE_LOCK,
)


class T1ContinuationRunError(RuntimeError):
    """Raised when the continuation cannot proceed. Never retried automatically."""


@dataclass
class ContinuationRunRecord:
    """What happened, in order. Carried so a failure can be described exactly."""

    entered: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    counters: ContinuationCounters = field(default_factory=ContinuationCounters)
    promoted: dict[str, str] = field(default_factory=dict)

    def enter(self, stage: str) -> None:
        if stage in self.entered:
            raise T1ContinuationRunError(
                f"Stage {stage!r} was already entered. A continuation stage runs "
                "once; re-entering one is a retry by another name."
            )
        self.entered.append(stage)

    def complete(self, stage: str) -> None:
        self.completed.append(stage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages_entered": list(self.entered),
            "stages_completed": list(self.completed),
            "stage_count": len(CONTINUATION_STAGE_ORDER),
            "counters": self.counters.as_dict(),
            "promoted_artifact_digests": dict(sorted(self.promoted.items())),
        }


def _authorized_git_sha() -> str:
    """The commit the continuation runs at, from git provenance."""
    from cardiosentinel.data.provenance import git_provenance

    provenance = git_provenance()
    if provenance.get("git_dirty"):
        raise T1ContinuationRunError(
            "The working tree is dirty. A continuation whose code cannot be "
            "named by one commit cannot be reproduced from its own receipt."
        )
    return str(provenance["git_sha"])


def preflight(corpus_root: Path | None = None) -> dict[str, Any]:
    """Every read-only gate, in order, without claiming anything.

    Runs stages 1-7. Safe to call at any time: it writes nothing and creates no
    directory. While `T1_CONTINUATION_AUTHORIZED` is False it refuses at stage 1
    having touched nothing, which is the state this module ships in.
    """
    record = ContinuationRunRecord()

    record.enter(STAGE_AUTHORIZE)
    require_continuation_authorized()
    record.complete(STAGE_AUTHORIZE)

    record.enter(STAGE_IDENTITY)
    require_continuation_identity(
        CONTINUATION_ATTEMPT_ID, CONTINUATION_RUN_ROOT_RELATIVE
    )
    record.complete(STAGE_IDENTITY)

    record.enter(STAGE_VERIFY_PREDECESSOR)
    verification = verify_predecessor(CONSUMED_ATTEMPT_DIR)
    record.complete(STAGE_VERIFY_PREDECESSOR)

    record.enter(STAGE_PROVE_CAPABILITY)
    gate_proof = prove_negative_capability(
        CONTINUATION_PROVEN_MODULES,
        record.counters,
        require_clean_interpreter=True,
    )
    record.complete(STAGE_PROVE_CAPABILITY)

    record.enter(STAGE_CONSUME_TRACE)
    trace = consume_oof_state_trace(CONSUMED_ATTEMPT_DIR, record.counters)
    record.complete(STAGE_CONSUME_TRACE)

    record.enter(STAGE_MATCH_SELECTIONS)
    verified = require_trace_matches_selections(trace, CONSUMED_ATTEMPT_DIR)
    record.complete(STAGE_MATCH_SELECTIONS)

    return {
        "record": record,
        "verification": verification,
        "gate_proof": gate_proof,
        "trace": trace,
        "verified_folds": verified,
        "corpus_root": corpus_root,
    }


def _claim(attempt_dir: Path) -> Path:
    """Create the continuation attempt directory, exactly once.

    The first write of the entire run. An existing directory is a refusal, not
    an overwrite: the amendment authorizes one continuation, and a second run
    into the same root would be a retry.
    """
    if attempt_dir.exists():
        raise T1ContinuationRunError(
            f"The continuation attempt at {attempt_dir} is already claimed. It "
            "is not overwritten, resumed or retried: the amendment authorizes "
            "one continuation and predeclares no successor."
        )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt_dir


def _promote_json(attempt_dir: Path, name: str, payload: Mapping[str, Any]) -> str:
    path = attempt_dir / name
    if path.exists():
        raise T1ContinuationRunError(
            f"{name} is already promoted; it is not rewritten."
        )
    write_json_atomic(path, dict(payload))
    return sha256_file(path)


def execute_continuation(corpus_root: Path) -> dict[str, Any]:
    """Run the authorized measurement continuation, once.

    **This function cannot run today.** It refuses inside `preflight` at stage 1,
    before any path is resolved, because `T1_CONTINUATION_AUTHORIZED` is False.
    It is written and tested so that the continuation is ready when a human
    decides, not so that it is ready to happen by accident.

    The protocol's forbidden entry points are instrumented for the whole run, so
    a call into `next_state`, a threshold generator or a policy selector records
    itself and stops the run rather than quietly producing a number.
    """
    prepared = preflight(corpus_root)
    record: ContinuationRunRecord = prepared["record"]
    verification = prepared["verification"]
    trace = prepared["trace"]
    verified = prepared["verified_folds"]
    provenance = continuation_provenance(verification)
    git_sha = _authorized_git_sha()

    attempt_dir = CONTINUATION_RUN_ROOT / CONTINUATION_ATTEMPT_ID

    with instrumented_protocol_entry_points(record.counters):
        record.enter(STAGE_CLAIM)
        _claim(attempt_dir)
        record.complete(STAGE_CLAIM)

        record.enter(STAGE_FOLDS)
        target_source = _target_source(corpus_root)
        measurements: dict[int, dict[str, Any]] = {}
        for fold_index in sorted(PREDECESSOR_FOLD_SELECTIONS):
            _reverify_fold_selection(fold_index)
            labels = held_out_labels_for_fold(target_source, fold_index)
            mask = trace.fold_mask(fold_index)
            require_labels_cover_trace(
                labels,
                [str(v) for v in trace.columns["stable_id"][mask]],
                fold_index,
            )
            measurement = measure_fold(
                trace, fold_index, labels, verified, record.counters
            )
            measurements[fold_index] = measurement.as_dict()
            evidence = build_continuation_held_out_evidence(
                measurements[fold_index],
                authorized_git_sha=git_sha,
                fold_selection_sha256=PREDECESSOR_FOLD_SELECTIONS[fold_index][2],
                provenance=provenance,
            )
            promote_continuation_held_out_evaluation(attempt_dir, fold_index, evidence)
        record.complete(STAGE_FOLDS)

        record.enter(STAGE_RESULTS)
        oof_result = build_oof_result(measurements, provenance=provenance)
        subject_evidence = build_subject_evidence(measurements, provenance=provenance)
        bootstrap = build_bootstrap(subject_evidence, provenance=provenance)
        challenge = build_challenge_evidence(subject_evidence, provenance=provenance)
        final_configuration = build_final_configuration(
            provenance=provenance,
            upstream_identities=verification.as_dict(),
        )
        for name, payload in (
            ("T1_OOF_RESULT.json", oof_result),
            ("T1_SUBJECT_EVIDENCE.json", subject_evidence),
            ("T1_BOOTSTRAP.json", bootstrap),
            ("T1_CHALLENGE_EVIDENCE.json", challenge),
            ("T1_FINAL_CONFIGURATION.json", final_configuration),
        ):
            record.promoted[name] = _promote_json(attempt_dir, name, payload)
        record.complete(STAGE_RESULTS)

        record.enter(STAGE_ATTEST)
        attestation = build_continuation_attestation(
            record.counters,
            verification,
            gate_proof=prepared["gate_proof"],
            folds_measured=sorted(measurements),
        )
        record.promoted[CONTINUATION_ATTESTATION_NAME] = _promote_json(
            attempt_dir, CONTINUATION_ATTESTATION_NAME, attestation
        )
        record.complete(STAGE_ATTEST)

        record.enter(STAGE_LOCK)
        lock = build_experiment_lock(
            provenance=provenance,
            attestation=attestation,
            promoted_digests=record.promoted,
        )
        record.promoted["T1_EXPERIMENT_LOCK.json"] = _promote_json(
            attempt_dir, "T1_EXPERIMENT_LOCK.json", lock
        )
        record.complete(STAGE_LOCK)

    record.counters.require_all_zero()
    return {
        "attempt_dir": attempt_dir,
        "record": record.as_dict(),
        "attestation": attestation,
        "experiment_lock": lock,
    }


def _target_source(corpus_root: Path) -> Any:
    from cardiosentinel.neural.t1_continuation_labels import continuation_target_source

    return continuation_target_source(corpus_root)


def _reverify_fold_selection(fold_index: int) -> str:
    """Re-prove the §16 barrier immediately before this fold's labels are opened.

    The predecessor verifier already checked every selection digest. This checks
    one of them again, at the moment it matters, because §9 item 2 asks for the
    barrier to be re-proved rather than assumed -- a digest verified minutes ago
    and a digest verified now are different assurances.
    """
    _subject, _policy, expected = PREDECESSOR_FOLD_SELECTIONS[fold_index]
    path = (
        CONSUMED_ATTEMPT_DIR
        / "fold_selections"
        / (f"T1_FOLD_{fold_index:02d}_SELECTION.json")
    )
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise T1ContinuationRunError(
            f"Fold {fold_index} selection digests {observed}, not the promoted "
            f"{expected}. The §16 barrier is re-proved before every fold's "
            "labels are opened, and it did not hold."
        )
    return observed


def continuation_runner_capability() -> dict[str, Any]:
    """What this layer provides, as data a receipt can carry. Executes nothing."""
    return {
        "runner": "T1ContinuationExecutor",
        "stage_order": list(CONTINUATION_STAGE_ORDER),
        "writing_stages": list(WRITING_STAGES),
        "read_only_before_claim": True,
        "artifacts": list(CONTINUATION_RESULT_ARTIFACTS),
        "proven_modules": list(CONTINUATION_PROVEN_MODULES),
        "automatic_retry_permitted": False,
        "resume_supported": False,
        "test_accessed": False,
    }
