"""The continuation execution attestation, and the provenance every artifact carries.

Amendment §13.7 requires the continuation to promote an attestation carrying the
four zero counters, the consumed trace's digests, and the flags that record what
was *not* done here. Its final sentence is the reason this module exists:

    The attestation travels **with the evidence**, not in a test log, for the
    same reason every other artifact records `test_accessed: false`: a claim
    that lives only where the evidence does not is a claim a future reader
    cannot check.

So the attestation is an artifact, not an assertion, and building it is refused
unless every value it would carry is already true. A non-zero counter, a failed
digest comparison or a missing field is a refusal rather than a warning.

**This module builds and validates. It does not write.** Promotion goes through
the existing `t1_persistence` path, which owns atomic write, re-read and digest
verification. Splitting it that way keeps the artifact contract reviewable
without giving this module the ability to create a run directory.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence

from cardiosentinel.neural.t1_continuation_gate import ContinuationCounters
from cardiosentinel.neural.t1_continuation_predecessor import PredecessorVerification
from cardiosentinel.neural.t1_continuation_spec import (
    CONTINUATION_ATTESTATION_CLASS,
    CONTINUATION_ATTESTATION_FIXED_VALUES,
    CONTINUATION_ATTESTATION_REQUIRED_FIELDS,
    CONTINUATION_STATE_TRACE_SOURCE,
    CONTINUATION_ZERO_COUNTERS,
    FORBIDDEN_CONTINUATION_FIELDS,
    PREDECESSOR_OOF_ARRAY_SHA256,
    PREDECESSOR_OOF_CONTENT_SHA256,
    continuation_identity,
)
from cardiosentinel.neural.t1_recovery_amendment import (
    CONTINUED_ATTEMPT_ID,
    RECOVERY_AMENDMENT_NAME,
    RECOVERY_AMENDMENT_SHA256,
)

ATTESTATION_SCHEMA: Final = "t1_v1_continuation_execution_attestation/1"


class T1ContinuationAttestationError(RuntimeError):
    """Raised when the attestation would carry an untrue or incomplete claim."""


def continuation_provenance(
    verification: PredecessorVerification,
) -> dict[str, Any]:
    """The `continues` and `consumed_evidence` blocks (amendment §8).

    `continues` names what this run is a continuation *of*; `consumed_evidence`
    names every artifact it read and the digest it read them at. Together they
    let a reader reconstruct the input set without trusting this run's summary
    of it.
    """
    return {
        "continues": {
            "predecessor_run": CONTINUED_ATTEMPT_ID,
            "predecessor_run_root": str(verification.attempt_dir.name),
            "predecessor_digest": PREDECESSOR_OOF_CONTENT_SHA256,
            "predecessor_state_trace_array_sha256": PREDECESSOR_OOF_ARRAY_SHA256,
            "governing_amendment": RECOVERY_AMENDMENT_NAME,
            "governing_amendment_sha256": RECOVERY_AMENDMENT_SHA256,
        },
        "consumed_evidence": [
            dict(entry) for entry in verification.consumed_evidence()
        ],
    }


def build_continuation_attestation(
    counters: ContinuationCounters,
    verification: PredecessorVerification,
    *,
    gate_proof: Mapping[str, Any],
    folds_measured: Sequence[int],
) -> dict[str, Any]:
    """Assemble the §13.7 attestation, refusing anything it could not claim truly.

    The counters are re-read here rather than trusted from an earlier check: the
    whole value of a runtime counter is that it is read at completion, and a
    counter validated once at the start would prove only that nothing had run
    yet.
    """
    zeroed = counters.require_all_zero()

    if not verification.verified:  # pragma: no cover - constructor guarantees it
        raise T1ContinuationAttestationError("Predecessor was not verified.")

    attestation: dict[str, Any] = {
        "artifact_class": CONTINUATION_ATTESTATION_CLASS,
        "schema": ATTESTATION_SCHEMA,
        **zeroed,
        "state_trace_source": CONTINUATION_STATE_TRACE_SOURCE,
        "state_trace_content_sha256": PREDECESSOR_OOF_CONTENT_SHA256,
        "state_trace_array_sha256": PREDECESSOR_OOF_ARRAY_SHA256,
        "selection_performed_here": False,
        "thresholds_generated_here": False,
        "state_transitions_regenerated": False,
        "predecessor_digests_verified": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "folds_measured": sorted(int(index) for index in folds_measured),
        "negative_capability_gate": {
            "gate": gate_proof.get("gate"),
            "modules_proven": list(gate_proof.get("modules_proven", ())),
            "layer_1_structural": bool(gate_proof.get("layer_1_structural")),
            "layer_2_runtime": bool(gate_proof.get("layer_2_runtime")),
            "layer_3_evidence": True,
        },
        **continuation_identity(),
        **continuation_provenance(verification),
        "predecessor_verification": verification.as_dict(),
    }
    return validate_continuation_attestation(attestation)


def validate_continuation_attestation(attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an attestation that is incomplete, untrue, or says too much.

    Three separate failures, all refusals:

    * a **missing** required field -- the artifact cannot answer §13.7;
    * a **wrong** fixed value -- it answers, but not truthfully;
    * a **forbidden** field such as `policy_runs` -- it reports a quantity that
      cannot exist, because no policy was run here. §13.6 Layer 3 names that
      absence as evidence, so carrying the key at all would contradict the
      claim even if its value were zero.
    """
    missing = [
        field
        for field in CONTINUATION_ATTESTATION_REQUIRED_FIELDS
        if field not in attestation
    ]
    if missing:
        raise T1ContinuationAttestationError(
            f"The continuation attestation is missing {missing}. Amendment "
            "§13.7 makes a missing attestation a refusal: evidence that cannot "
            "answer the specification is worse than evidence that is absent."
        )

    wrong = {
        field: (attestation[field], expected)
        for field, expected in CONTINUATION_ATTESTATION_FIXED_VALUES.items()
        if attestation.get(field) != expected
    }
    if wrong:
        raise T1ContinuationAttestationError(
            "The continuation attestation would carry values the amendment "
            f"fixes differently: {wrong}."
        )

    present = [field for field in FORBIDDEN_CONTINUATION_FIELDS if field in attestation]
    if present:
        raise T1ContinuationAttestationError(
            f"The continuation attestation carries {present}. No policy was run "
            "here, so that counter has no meaning; §13.6 Layer 3 requires its "
            "absence, not a zero."
        )

    for counter in CONTINUATION_ZERO_COUNTERS:
        value = attestation[counter]
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            raise T1ContinuationAttestationError(
                f"Counter {counter} is {value!r}; §13.6 requires integer zero."
            )
    return dict(attestation)
