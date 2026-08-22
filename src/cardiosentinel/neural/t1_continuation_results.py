"""The six run-level artifacts of the continuation, stages 24 through 29.

These are the artifacts the consumed attempt never wrote. Each is assembled from
per-fold measurements the continuation produced by joining the **persisted** OOF
state trace to held-out labels; nothing here re-runs a policy, re-derives a
threshold or re-enters the state machine.

**Why these are not `t1_assembly`'s functions.** `t1_assembly` already assembles
stages 24-29 and it binds no forbidden name, so reusing it is tempting. It
imports `t1_development_run`, though, which means a continuation that called
into it would reach the state machine and the threshold generator through a path
the Layer 1 import proof does not inspect -- it checks what the continuation's
own modules bind, not what their callees do. Rather than widen the gate to a
transitive check one change before the single authorized run, the two helpers
this layer actually needs are re-implemented here, and
`test_t1_continuation_results` proves them **exactly** equal to the frozen
originals. Same resolution as `contiguous_runs` in the measurement layer, for
the same reason.

**The confusion vocabulary.** Two names exist for one quantity and both are
correct in their own layer: `tp/fp/tn/fn` is what a counter is called next to
the loop that increments it, `true_positive/...` is what a reported margin is
called in an evidence document. `t1_composition._pooled_confusion` is where they
meet on the canonical path; that module is forbidden here, so the translation is
written out again below and a test asserts it equals the canonical mapping. A
producer and a consumer disagreeing about exactly this pair is what consumed the
canonical attempt at stage 24, so a missing count is a refusal, never a zero.
"""

from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence

import numpy as np

from cardiosentinel.neural.t1_continuation_spec import (
    CONTINUATION_ATTEMPT_ID,
    CONTINUATION_RUN_CLASS,
    PREDECESSOR_FOLD_SELECTIONS,
    PREDECESSOR_OOF_ARRAY_SHA256,
    PREDECESSOR_OOF_CONTENT_SHA256,
    continuation_identity,
)
from cardiosentinel.neural.t1_recovery_amendment import (
    RECOVERY_AMENDMENT_NAME,
    RECOVERY_AMENDMENT_SHA256,
)

#: The frozen §21 bootstrap design. Restated rather than imported, because it
#: lives in `t1_development_run`; a test asserts every value matches.
T1_BOOTSTRAP_REPLICATES: Final = 1000
T1_BOOTSTRAP_SEED: Final = 2026
T1_BOOTSTRAP_UNIT: Final = "subject"
T1_BOOTSTRAP_RESELECTS_POLICY: Final = False
T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY: Final = True
BOOTSTRAP_SUBJECT_STATISTIC: Final = "episode_f1"

#: The evaluator's counter names mapped to the names an evidence document reads.
#: Equal to `t1_composition.PRIMARY_CONFUSION_KEYS`, asserted by test.
PRIMARY_CONFUSION_KEYS: Final = {
    "tp": "true_positive",
    "fp": "false_positive",
    "tn": "true_negative",
    "fn": "false_negative",
}

EPISODE_EVIDENCE_KEYS: Final = (
    "reference_episodes",
    "predicted_event_runs",
    "matched_episodes",
    "unmatched_predicted_runs",
)

OOF_RESULT_CLASS: Final = "t1_v1_continuation_oof_result"
SUBJECT_EVIDENCE_CLASS: Final = "t1_v1_continuation_subject_evidence"
BOOTSTRAP_CLASS: Final = "t1_v1_continuation_subject_bootstrap"
CHALLENGE_CLASS: Final = "t1_v1_continuation_challenge_evidence"
FINAL_CONFIGURATION_CLASS: Final = "t1_v1_continuation_final_configuration"
EXPERIMENT_LOCK_CLASS: Final = "t1_v1_continuation_experiment_lock"

#: Artifact file names, in the order the continuation promotes them.
CONTINUATION_RESULT_ARTIFACTS: Final = (
    "T1_OOF_RESULT.json",
    "T1_SUBJECT_EVIDENCE.json",
    "T1_BOOTSTRAP.json",
    "T1_CHALLENGE_EVIDENCE.json",
    "T1_FINAL_CONFIGURATION.json",
    "T1_EXPERIMENT_LOCK.json",
)


class T1ContinuationResultError(RuntimeError):
    """Raised when a run-level artifact cannot be assembled truthfully."""


# ---------------------------------------------------------------------------
# Frozen helpers, re-implemented and proven equal
# ---------------------------------------------------------------------------


def subject_bootstrap_indices(subject_count: int) -> np.ndarray:
    """1000 replicates, seed 2026, subjects resampled with multiplicity.

    Byte-for-byte the frozen design in `t1_development_run`, which the
    continuation may not import. Equivalence is asserted by test, not assumed.
    """
    generator = np.random.default_rng(T1_BOOTSTRAP_SEED)
    return generator.integers(
        0, subject_count, size=(T1_BOOTSTRAP_REPLICATES, subject_count)
    )


def window_mcc(
    predicted_positive: np.ndarray, actual_positive: np.ndarray
) -> float | None:
    """Matthews correlation. Undefined when any margin is empty, never zero."""
    predicted = np.asarray(predicted_positive).astype(bool)
    actual = np.asarray(actual_positive).astype(bool)
    tp = int(np.count_nonzero(predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return None
    return (tp * tn - fp * fn) / denominator


def _confusion_arrays(confusion: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Put the counts in the shape the MCC helper takes. Computes no statistic."""
    counts = {key: int(confusion.get(key, 0)) for key in ("tp", "fp", "tn", "fn")}
    predicted = [True] * (counts["tp"] + counts["fp"]) + [False] * (
        counts["tn"] + counts["fn"]
    )
    actual = (
        [True] * counts["tp"]
        + [False] * counts["fp"]
        + [False] * counts["tn"]
        + [True] * counts["fn"]
    )
    return np.asarray(predicted, dtype=bool), np.asarray(actual, dtype=bool)


def _median(values: Sequence[float]) -> float | None:
    """Undefined for an empty sample, never zero.

    A subject whose episodes were all missed has no onset latency at all;
    reporting zero would read as an instant detection.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _episode_f1(episodes: Mapping[str, int]) -> float | None:
    matched = int(episodes["matched_episodes"])
    reference = int(episodes["reference_episodes"])
    predicted = int(episodes["predicted_event_runs"])
    denominator = predicted + reference
    if denominator == 0:
        return None
    return 2.0 * matched / denominator


def translate_confusion(confusion: Mapping[str, int]) -> dict[str, int]:
    """`tp/fp/tn/fn` -> `true_positive/...`. A missing count is a refusal.

    A silently defaulted margin is an evidence value that nothing produced, and
    the difference between "no true positives" and "the producer stopped
    supplying true positives" is the difference this function exists to keep.
    """
    missing = [key for key in PRIMARY_CONFUSION_KEYS if key not in confusion]
    if missing:
        raise T1ContinuationResultError(
            f"PRIMARY confusion is missing {missing}. A missing count is "
            "refused rather than defaulted to zero: that substitution is the "
            "defect that consumed the canonical attempt at stage 24."
        )
    return {
        long_name: int(confusion[short])
        for short, long_name in PRIMARY_CONFUSION_KEYS.items()
    }


# ---------------------------------------------------------------------------
# §19 OOF result
# ---------------------------------------------------------------------------


def _pool(measurements: Mapping[int, Mapping[str, Any]], block: str, keys) -> dict:
    totals = dict.fromkeys(keys, 0)
    for fold_index, measurement in sorted(measurements.items()):
        section = measurement.get(block)
        if not isinstance(section, Mapping):
            raise T1ContinuationResultError(
                f"Fold {fold_index} measurement lacks {block!r}."
            )
        for key in totals:
            if key not in section:
                raise T1ContinuationResultError(
                    f"Fold {fold_index} {block} is missing {key!r}."
                )
            totals[key] += int(section[key])
    return totals


def build_oof_result(
    measurements: Mapping[int, Mapping[str, Any]], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """§19: the pooled held-out result the consumed attempt died before writing."""
    if not measurements:
        raise T1ContinuationResultError("No fold measurements to pool.")
    pooled_confusion = _pool(
        measurements, "primary_confusion", ("tp", "fp", "tn", "fn")
    )
    pooled_episodes = _pool(measurements, "episode_evidence", EPISODE_EVIDENCE_KEYS)
    latencies = [
        float(value)
        for _, measurement in sorted(measurements.items())
        for value in measurement["onset_latency_seconds"]
    ]
    predicted, actual = _confusion_arrays(pooled_confusion)

    return {
        "artifact_class": OOF_RESULT_CLASS,
        "fold_count": len(measurements),
        "primary_confusion": translate_confusion(pooled_confusion),
        "episode_evidence": dict(pooled_episodes),
        "pooled_episode_f1": _episode_f1(pooled_episodes),
        "pooled_primary_window_mcc": window_mcc(predicted, actual),
        "onset_latency_seconds_median": _median(latencies),
        "matched_episode_count": int(pooled_episodes["matched_episodes"]),
        "fold_summaries": [
            {
                "fold_index": int(m["fold_index"]),
                "held_out_subject": m["held_out_subject"],
                "selected_policy_id": m["selected_policy_id"],
                "primary_confusion": translate_confusion(m["primary_confusion"]),
                "episode_evidence": dict(m["episode_evidence"]),
                "episode_f1": _episode_f1(m["episode_evidence"]),
            }
            for _, m in sorted(measurements.items())
        ],
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }


# ---------------------------------------------------------------------------
# §21 subject evidence and bootstrap
# ---------------------------------------------------------------------------


def build_subject_evidence(
    measurements: Mapping[int, Mapping[str, Any]], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-subject evidence, each arranged from that subject's held-out fold."""
    subjects: dict[str, dict[str, Any]] = {}
    for fold_index, measurement in sorted(measurements.items()):
        subject = measurement["held_out_subject"]
        if subject in subjects:
            raise T1ContinuationResultError(
                f"{subject!r} appears in more than one fold; the subject-fold "
                "bijection the consumed attempt promoted is violated."
            )
        confusion = measurement["primary_confusion"]
        episodes = measurement["episode_evidence"]
        predicted, actual = _confusion_arrays(confusion)
        subjects[subject] = {
            "fold_index": int(fold_index),
            "selected_policy_id": measurement["selected_policy_id"],
            "primary_confusion": translate_confusion(confusion),
            "episode_evidence": dict(episodes),
            BOOTSTRAP_SUBJECT_STATISTIC: _episode_f1(episodes),
            "primary_window_mcc": window_mcc(predicted, actual),
            "onset_latency_seconds_median": _median(
                measurement["onset_latency_seconds"]
            ),
        }
    return {
        "artifact_class": SUBJECT_EVIDENCE_CLASS,
        "subject_count": len(subjects),
        "subjects": dict(sorted(subjects.items())),
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }


def subject_statistic(subject_evidence: Mapping[str, Any]) -> dict[str, float]:
    """The one float per subject the frozen bootstrap resamples.

    An undefined episode F1 is carried as NaN rather than zero: the bootstrap
    preserves undefined replicates, and a zero would be indistinguishable from
    a real measurement of zero.
    """
    return {
        subject: float("nan")
        if evidence[BOOTSTRAP_SUBJECT_STATISTIC] is None
        else float(evidence[BOOTSTRAP_SUBJECT_STATISTIC])
        for subject, evidence in sorted(subject_evidence["subjects"].items())
    }


def build_bootstrap(
    subject_evidence: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """§21: 1000 replicates, seed 2026, subjects resampled with multiplicity.

    A subject drawn twice contributes its already-frozen measurement twice. No
    fold is rerun, no policy is re-derived, and nothing is reselected inside the
    bootstrap. Undefined replicates are preserved as undefined.
    """
    statistic = subject_statistic(subject_evidence)
    subjects = sorted(statistic)
    if not subjects:
        raise T1ContinuationResultError("Bootstrap has no subjects to resample.")
    indices = subject_bootstrap_indices(len(subjects))
    if indices.shape != (T1_BOOTSTRAP_REPLICATES, len(subjects)):
        raise T1ContinuationResultError(
            f"Bootstrap indices are {indices.shape}, not "
            f"({T1_BOOTSTRAP_REPLICATES}, {len(subjects)})."
        )
    values = np.asarray([statistic[name] for name in subjects])
    replicates = [
        None if np.isnan(values[row]).any() else float(np.mean(values[row]))
        for row in indices
    ]
    defined = [value for value in replicates if value is not None]
    return {
        "artifact_class": BOOTSTRAP_CLASS,
        "replicates": T1_BOOTSTRAP_REPLICATES,
        "seed": T1_BOOTSTRAP_SEED,
        "unit": T1_BOOTSTRAP_UNIT,
        "statistic": BOOTSTRAP_SUBJECT_STATISTIC,
        "policy_reselected_inside_bootstrap": T1_BOOTSTRAP_RESELECTS_POLICY,
        "resampled_with_multiplicity": T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY,
        "subject_order": list(subjects),
        "defined_replicates": len(defined),
        "undefined_replicates": len(replicates) - len(defined),
        "percentile_2_5": float(np.percentile(defined, 2.5)) if defined else None,
        "percentile_97_5": float(np.percentile(defined, 97.5)) if defined else None,
        "claim_scope": (
            "between_subject_variation_conditional_on_the_cross_fitted_t1_"
            "development_procedure"
        ),
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }


# ---------------------------------------------------------------------------
# §22 challenge, §23 final configuration, §24 lock
# ---------------------------------------------------------------------------


def build_challenge_evidence(
    subject_evidence: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    challenge_strata: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """§22: the challenge reporting join, over subjects only.

    The challenge strata are *reported*, never selected on. When no stratum
    membership is supplied the artifact says so explicitly rather than inventing
    an empty subgroup: an absent join and an empty join are different facts, and
    only one of them is evidence.
    """
    subjects = subject_evidence["subjects"]
    strata = dict(challenge_strata or {})
    reported: dict[str, Any] = {}
    for name, members in sorted(strata.items()):
        present = [s for s in members if s in subjects]
        reported[name] = {
            "subject_count": len(present),
            "subjects": sorted(present),
            "episode_f1_values": [
                subjects[s][BOOTSTRAP_SUBJECT_STATISTIC] for s in sorted(present)
            ],
        }
    return {
        "artifact_class": CHALLENGE_CLASS,
        "join_performed": bool(strata),
        "strata_reported": sorted(reported),
        "strata": reported,
        "selection_performed_on_challenge_evidence": False,
        "note": (
            "Challenge strata are reported, never selected on. Absent strata are "
            "recorded as absent rather than as empty subgroups."
        ),
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }


def build_final_configuration(
    *, provenance: Mapping[str, Any], upstream_identities: Mapping[str, Any]
) -> dict[str, Any]:
    """§23: the all-VALIDATION configuration, read from the promoted selections.

    Every threshold and policy is read verbatim from the twelve promoted fold
    selections bound by amendment §1.4. Nothing is generated or selected here.
    """
    folds = {
        str(index): {
            "held_out_subject": subject,
            "selected_policy_id": policy_id,
            "fold_selection_sha256": digest,
        }
        for index, (subject, policy_id, digest) in sorted(
            PREDECESSOR_FOLD_SELECTIONS.items()
        )
    }
    policies = sorted({entry["selected_policy_id"] for entry in folds.values()})
    return {
        "artifact_class": FINAL_CONFIGURATION_CLASS,
        "fold_count": len(folds),
        "folds": folds,
        "selected_policies": policies,
        "thresholds_source": "promoted_fold_selection_artifacts",
        "thresholds_generated_here": False,
        "selection_performed_here": False,
        "upstream_identities": dict(sorted(upstream_identities.items())),
        "state_trace_array_sha256": PREDECESSOR_OOF_ARRAY_SHA256,
        "state_trace_content_sha256": PREDECESSOR_OOF_CONTENT_SHA256,
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }


def build_experiment_lock(
    *,
    provenance: Mapping[str, Any],
    attestation: Mapping[str, Any],
    promoted_digests: Mapping[str, str],
) -> dict[str, Any]:
    """The continuation's provenance closure.

    Carries the digest of every artifact this continuation promoted, so the
    whole set is checkable from one place, plus the attestation's four zero
    counters and the amendment under which the run was permitted.
    """
    for counter in (
        "state_machine_invocations",
        "threshold_generation_calls",
        "policy_selection_calls",
        "fold_evaluations",
    ):
        if attestation.get(counter) != 0:
            raise T1ContinuationResultError(
                f"Cannot lock a continuation whose {counter} is "
                f"{attestation.get(counter)!r}."
            )
    return {
        "artifact_class": EXPERIMENT_LOCK_CLASS,
        "run_class": CONTINUATION_RUN_CLASS,
        "attempt_id": CONTINUATION_ATTEMPT_ID,
        "governing_amendment": RECOVERY_AMENDMENT_NAME,
        "governing_amendment_sha256": RECOVERY_AMENDMENT_SHA256,
        "promoted_artifact_digests": dict(sorted(promoted_digests.items())),
        "promoted_artifact_count": len(promoted_digests),
        "execution_attestation": dict(attestation),
        "attempts_authorized": 1,
        "automatic_retry_permitted": False,
        **continuation_identity(),
        **{k: provenance[k] for k in ("continues", "consumed_evidence")},
    }
