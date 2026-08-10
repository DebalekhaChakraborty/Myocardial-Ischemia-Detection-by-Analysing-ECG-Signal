"""Official B4-A/B4-B/B4-C validation challenge evidence, predictions only.

This module produces the development challenge evidence required by
`docs/B4_VALIDATION_CHALLENGE_PROTOCOL_V1.md` before the architecture-selection
gate. It is deliberately a *reader*: every quantity is derived from an immutable
experiment lock plus that experiment's immutable locked validation predictions.

No model is constructed or loaded for inference, no waveform or dataset is read,
and there is no route to the sealed test. The checkpoint is touched only to
confirm the identity already bound by the lock, never to run it.

The metric definitions are not restated here. `challenge_metrics` and
`positive_context_analysis` are the already-frozen production implementations
shared with the B0-B3 baselines, and the evidence levels come from the frozen
`CHALLENGE_EVIDENCE_POLICIES` table. This module supplies execution procedure and
integrity checking, not new statistics.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.baseline.cache import read_json, write_json_atomic
from cardiosentinel.baseline.metrics import (
    challenge_metrics,
    positive_context_analysis,
)
from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.evaluation.protocol import (
    CHALLENGE_EVIDENCE_POLICIES,
    POSITIVE_CONTEXT_EVIDENCE_LEVEL,
    POSITIVE_CONTEXT_FLAGS,
    challenge_evidence_policy,
)
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    REPOSITORY_ROOT,
    protocol_sha256,
)
from cardiosentinel.neural.resource_benchmark import (
    ARCHITECTURE_PROTOCOL_SHA256,
    OFFICIAL_ORDER,
    validate_locked_model,
)

VALIDATION_CHALLENGE_PROTOCOL_NAME: Final = "B4_VALIDATION_CHALLENGE_PROTOCOL_V1"
VALIDATION_CHALLENGE_PROTOCOL_PATH: Final = (
    REPOSITORY_ROOT / "docs" / "B4_VALIDATION_CHALLENGE_PROTOCOL_V1.md"
)
VALIDATION_CHALLENGE_PROTOCOL_SHA256: Final = (
    "5478c46fe5013d1d893f1f134d35af68ec3007105c542a2115718c0431858692"
)

SUITE_DIR_NAME: Final = "B4_architecture_validation_challenge_v1"
SUITE_ATTEMPT_NAME: Final = "VALIDATION_CHALLENGE_ATTEMPT.json"
SUITE_RESULTS_NAME: Final = "VALIDATION_CHALLENGE_RESULTS.json"
VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.npz"

SUITE_STATUS_STARTED: Final = "STARTED"
SUITE_STATUS_COMPLETE: Final = "COMPLETE"

EVALUATED_PARTITION: Final = "validation"
FORBIDDEN_PARTITIONS: Final = frozenset({"test"})
THRESHOLD_SOURCE: Final = "locked_experiment_lock.validation_threshold"
CHALLENGE_NAMES: Final = ("rate_related", "axis_shift", "conduction_change")

# Every column the frozen prediction artifact must provide. A missing or extra
# column is a schema mismatch, not something to work around.
REQUIRED_PREDICTION_COLUMNS: Final = (
    "stable_id",
    "subject_id",
    "record_id",
    "channel_index",
    "target_family",
    "context_flags",
    "label",
    "score",
)

METRIC_DEFINITIONS: Final = {
    "challenge_metrics": (
        "cardiosentinel.baseline.metrics.challenge_metrics "
        "(frozen production implementation shared with B0-B3)"
    ),
    "positive_context": (
        "cardiosentinel.baseline.metrics.positive_context_analysis "
        "(frozen production implementation shared with B0-B3)"
    ),
    "evidence_policy": (
        "cardiosentinel.evaluation.protocol.CHALLENGE_EVIDENCE_POLICIES"
    ),
}


class ValidationChallengeError(RuntimeError):
    """Raised when challenge evidence cannot be produced with full integrity."""


def validate_validation_challenge_protocol(
    path: Path = VALIDATION_CHALLENGE_PROTOCOL_PATH,
) -> str:
    """Fail if the frozen challenge procedure bytes have changed."""
    digest = protocol_sha256(path)
    if digest != VALIDATION_CHALLENGE_PROTOCOL_SHA256:
        raise ValidationChallengeError(
            "B4_VALIDATION_CHALLENGE_PROTOCOL_V1.md differs from its frozen "
            "SHA-256."
        )
    return digest


def require_evaluated_partition(partition: str) -> str:
    """Accept only the development validation partition; refuse the test one.

    The sealed test has no route through this module. This guard exists so that
    a caller passing `"test"` fails loudly rather than being quietly ignored.
    """
    if partition in FORBIDDEN_PARTITIONS:
        raise ValidationChallengeError(
            "The validation challenge evaluator must never access the "
            f"{partition!r} partition."
        )
    if partition != EVALUATED_PARTITION:
        raise ValidationChallengeError(
            f"Unsupported challenge partition {partition!r}; "
            f"only {EVALUATED_PARTITION!r} is evaluated."
        )
    return partition


def _historical_binding(lock: dict[str, Any]) -> dict[str, Any]:
    """Record which provenance fields the lock does and does not carry.

    B4-A predates several binding fields. Rather than fabricate them, the suite
    reports exactly what was available so a reviewer can see the difference.
    """
    optional = (
        "environment_dependency_digest",
        "candidate_architecture",
        "architecture_protocol_sha256",
    )
    required = (
        "split_sha256",
        "feature_corpus_sha256",
        "training_selection_sha256",
        "development_feature_integrity_sha256",
        "validation_predictions_sha256",
    )
    missing_required = [field for field in required if lock.get(field) is None]
    if missing_required:
        raise ValidationChallengeError(
            "The experiment lock is missing mandatory provenance fields: "
            f"{sorted(missing_required)}."
        )
    return {
        "required_fields_present": list(required),
        "optional_fields_present": [
            field for field in optional if lock.get(field) is not None
        ],
        "optional_fields_absent": [
            field for field in optional if lock.get(field) is None
        ],
    }


def load_locked_validation_predictions(
    run_dir: Path, lock: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Load the immutable validation predictions bound to this experiment lock.

    Nothing here is repaired, coerced, imputed or dropped. Any disagreement
    between the artifact and the lock is a refusal.
    """
    path = Path(run_dir) / VALIDATION_PREDICTIONS_NAME
    if not path.is_file():
        raise ValidationChallengeError(
            f"No {VALIDATION_PREDICTIONS_NAME} in {run_dir}."
        )
    expected = lock.get("validation_predictions_sha256")
    if not expected:
        raise ValidationChallengeError(
            "The experiment lock does not bind a validation prediction digest."
        )
    if sha256_file(path) != expected:
        raise ValidationChallengeError(
            "The validation prediction artifact does not match the digest bound "
            "by the experiment lock."
        )

    with np.load(path, allow_pickle=False) as archive:
        present = set(archive.files)
        missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in present]
        if missing:
            raise ValidationChallengeError(
                f"The validation prediction artifact lacks columns {sorted(missing)}."
            )
        columns = {name: archive[name] for name in REQUIRED_PREDICTION_COLUMNS}

    lengths = {name: int(value.shape[0]) for name, value in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValidationChallengeError(
            f"The validation prediction columns are misaligned: {lengths}."
        )
    rows = next(iter(lengths.values()))
    if rows == 0:
        raise ValidationChallengeError("The validation prediction artifact is empty.")

    scores = np.asarray(columns["score"], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValidationChallengeError(
            "The validation prediction artifact contains a non-finite score."
        )
    labels = np.asarray(columns["label"], dtype=np.int64)
    if not np.all(np.isin(labels, (0, 1))):
        raise ValidationChallengeError(
            "Validation labels must be exactly 0 or 1."
        )
    stable_ids = np.asarray(columns["stable_id"], dtype=np.str_)
    if len(set(stable_ids.tolist())) != rows:
        raise ValidationChallengeError(
            "The validation prediction artifact contains duplicate stable IDs."
        )

    expected_rows = lock.get("validation_rows") or {}
    if expected_rows.get("total") is not None and expected_rows["total"] != rows:
        raise ValidationChallengeError(
            f"The prediction artifact has {rows} rows but the lock records "
            f"{expected_rows['total']}."
        )
    subjects = np.asarray(columns["subject_id"], dtype=np.str_)
    subject_count = len(set(subjects.tolist()))
    if (
        expected_rows.get("subjects") is not None
        and expected_rows["subjects"] != subject_count
    ):
        raise ValidationChallengeError(
            f"The prediction artifact has {subject_count} subjects but the lock "
            f"records {expected_rows['subjects']}."
        )

    columns["score"] = scores
    columns["label"] = labels
    columns["stable_id"] = stable_ids
    columns["subject_id"] = subjects
    columns["target_family"] = np.asarray(columns["target_family"], dtype=np.str_)
    columns["context_flags"] = np.asarray(columns["context_flags"], dtype=np.str_)
    return columns


def _challenge_evidence(
    columns: dict[str, np.ndarray], threshold: float
) -> dict[str, Any]:
    """Apply the frozen challenge implementation and label each stratum."""
    frozen = challenge_metrics(
        columns["target_family"],
        columns["score"],
        columns["subject_id"],
        threshold,
    )
    evidence: dict[str, Any] = {}
    for challenge in CHALLENGE_NAMES:
        policy = challenge_evidence_policy(challenge)
        measured = frozen[challenge]
        evidence[challenge] = {
            "target_family": policy.target_family,
            "evidence_status": policy.evidence_level,
            "is_headline_metric": policy.is_headline_metric,
            # §11: a fraction is never reported without its denominator, and an
            # empty stratum stays null rather than becoming a fabricated 0.0.
            "denominator_negative_window_count": measured["challenge_window_count"],
            "false_positive_count": measured["false_positive_count"],
            "false_positive_fraction": measured["false_positive_fraction"],
            "supporting_subject_count": measured["contributing_subject_count"],
            "bootstrap_permitted": measured["bootstrap_permitted"],
            "frozen_metric": measured,
        }
    return evidence


def evaluate_candidate_validation_challenge(
    run_dir: Path, *, official_model: str
) -> dict[str, Any]:
    """Produce one candidate's development challenge evidence from predictions.

    This never constructs or runs a model. `validate_locked_model` hashes the
    locked checkpoint purely to confirm the identity the lock already binds.
    """
    if official_model not in OFFICIAL_ORDER:
        raise ValidationChallengeError(f"Unknown official model {official_model!r}.")
    protocol_digest = validate_validation_challenge_protocol()
    partition = require_evaluated_partition(EVALUATED_PARTITION)

    lock = validate_locked_model(Path(run_dir), official_model=official_model)
    binding = _historical_binding(lock)

    threshold = lock.get("validation_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        raise ValidationChallengeError(
            "The experiment lock does not bind a finite validation threshold."
        )
    threshold = float(threshold)

    columns = load_locked_validation_predictions(Path(run_dir), lock)
    subjects = columns["subject_id"]

    payload: dict[str, Any] = {
        "official_model": official_model,
        "experiment_id": lock["experiment_id"],
        "architecture": lock.get("candidate_architecture")
        or lock.get("model", {}).get("architecture"),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "validation_prediction_sha256": lock["validation_predictions_sha256"],
        "locked_validation_threshold": threshold,
        "threshold_source": THRESHOLD_SOURCE,
        "threshold_selected_by_evaluator": False,
        "partition": partition,
        "split_sha256": lock["split_sha256"],
        "feature_corpus_sha256": lock["feature_corpus_sha256"],
        "training_selection_sha256": lock["training_selection_sha256"],
        "development_feature_integrity_sha256": lock[
            "development_feature_integrity_sha256"
        ],
        "provenance_binding": binding,
        "validation_window_count": int(columns["score"].shape[0]),
        "validation_subject_count": len(set(subjects.tolist())),
        "validation_positive_count": int(np.sum(columns["label"] == 1)),
        "validation_negative_count": int(np.sum(columns["label"] == 0)),
        "challenges": _challenge_evidence(columns, threshold),
        "positive_context": {
            "evidence_status": POSITIVE_CONTEXT_EVIDENCE_LEVEL,
            "context_flags": list(POSITIVE_CONTEXT_FLAGS),
            "strata": positive_context_analysis(
                columns["label"],
                columns["score"],
                subjects,
                columns["context_flags"],
                threshold,
            ),
        },
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "validation_challenge_protocol_sha256": protocol_digest,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "dataset_accessed": False,
        "test_accessed": False,
        "model_inference_performed": False,
        "waveform_accessed": False,
    }
    payload["challenge_result_sha256"] = canonical_sha256(payload)
    return payload


def _claim_suite_attempt(path: Path, payload: dict[str, Any]) -> str:
    """Create the official attempt with an atomic O_EXCL creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValidationChallengeError(
            "An official B4 validation challenge attempt already exists at "
            f"{path}. The official suite is one-shot: automatic rerun, "
            "candidate-specific retry and attempt replacement are prohibited "
            "and require documented human review."
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(body)
    return canonical_sha256(payload)


def run_official_validation_challenge_suite(
    run_directories: dict[str, Path],
    run_root: Path,
    *,
    command: str = "cardiosentinel b4 validation-challenge",
) -> dict[str, Any]:
    """Evaluate B4-A, B4-B and B4-C in one exclusive official invocation.

    Exactly three locked run directories are required, evaluated in the frozen
    order. No model may be omitted, no fourth model added, and no model
    evaluated officially on its own.

    There is deliberately no evaluator, backend or metric-function parameter:
    official evidence can only ever come from the frozen production challenge
    implementation applied to locked predictions.
    """
    if set(run_directories) != set(OFFICIAL_ORDER):
        raise ValidationChallengeError(
            "The official validation challenge suite requires exactly "
            f"{list(OFFICIAL_ORDER)}; received {sorted(run_directories)}."
        )
    protocol_digest = validate_validation_challenge_protocol()
    started = time.monotonic()

    suite_dir = Path(run_root) / SUITE_DIR_NAME
    attempt_path = suite_dir / SUITE_ATTEMPT_NAME
    attempt = {
        "suite": SUITE_DIR_NAME,
        "attempt_sequence": 1,
        "attempt_status": SUITE_STATUS_STARTED,
        "repeat_attempt_permitted": False,
        "selective_candidate_retry_permitted": False,
        "candidate_order": list(OFFICIAL_ORDER),
        "command": command,
        "validation_challenge_protocol_sha256": protocol_digest,
        "created_at_utc_audit_only": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    attempt_sha256 = _claim_suite_attempt(attempt_path, attempt)

    results: dict[str, Any] = {}
    for name in OFFICIAL_ORDER:
        result = evaluate_candidate_validation_challenge(
            Path(run_directories[name]), official_model=name
        )
        recorded = result.get("challenge_result_sha256")
        body = {k: v for k, v in result.items() if k != "challenge_result_sha256"}
        if recorded != canonical_sha256(body):
            raise ValidationChallengeError(
                f"{name} challenge evidence failed digest revalidation."
            )
        if result["dataset_accessed"] or result["test_accessed"]:
            raise ValidationChallengeError(
                f"{name} challenge evidence claims dataset or test access."
            )
        if result["model_inference_performed"]:
            raise ValidationChallengeError(
                f"{name} challenge evidence claims model inference."
            )
        results[name] = result

    suite: dict[str, Any] = {
        "suite": SUITE_DIR_NAME,
        "command": command,
        "attempt_sequence": 1,
        "suite_attempt_sha256": attempt_sha256,
        "candidate_order": list(OFFICIAL_ORDER),
        "partition": EVALUATED_PARTITION,
        "validation_challenge_protocol_sha256": protocol_digest,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "architecture_protocol_sha256": ARCHITECTURE_PROTOCOL_SHA256,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "challenge_evidence_policies": {
            challenge: {
                "target_family": policy.target_family,
                "evidence_level": policy.evidence_level,
                "is_headline_metric": policy.is_headline_metric,
                "supports_inferential_bootstrap": (
                    policy.supports_inferential_bootstrap
                ),
            }
            for challenge, policy in CHALLENGE_EVIDENCE_POLICIES.items()
        },
        "threshold_source": THRESHOLD_SOURCE,
        "experiment_lock_sha256": {
            name: results[name]["experiment_lock_sha256"] for name in OFFICIAL_ORDER
        },
        "validation_prediction_sha256": {
            name: results[name]["validation_prediction_sha256"]
            for name in OFFICIAL_ORDER
        },
        "locked_validation_threshold": {
            name: results[name]["locked_validation_threshold"]
            for name in OFFICIAL_ORDER
        },
        "challenge_result_sha256": {
            name: results[name]["challenge_result_sha256"] for name in OFFICIAL_ORDER
        },
        "candidate_results": results,
        "dataset_accessed": False,
        "test_accessed": False,
        "model_inference_performed": False,
        "architecture_selection_performed": False,
        "suite_duration_seconds": time.monotonic() - started,
    }
    suite["validation_challenge_suite_sha256"] = canonical_sha256(suite)

    write_json_atomic(suite_dir / SUITE_RESULTS_NAME, suite)
    write_json_atomic(
        attempt_path,
        {
            **attempt,
            "attempt_status": SUITE_STATUS_COMPLETE,
            "validation_challenge_suite_sha256": suite[
                "validation_challenge_suite_sha256"
            ],
        },
    )
    return suite


def read_official_validation_challenge_results(run_root: Path) -> dict[str, Any]:
    """Read an existing official result and re-derive its combined digest."""
    path = Path(run_root) / SUITE_DIR_NAME / SUITE_RESULTS_NAME
    if not path.is_file():
        raise ValidationChallengeError(f"No official challenge result at {path}.")
    suite = read_json(path)
    recorded = suite.get("validation_challenge_suite_sha256")
    body = {
        k: v for k, v in suite.items() if k != "validation_challenge_suite_sha256"
    }
    if recorded is None or recorded != canonical_sha256(body):
        raise ValidationChallengeError(
            "The official challenge result failed digest revalidation."
        )
    return suite


__all__ = [
    "SUITE_ATTEMPT_NAME",
    "SUITE_DIR_NAME",
    "SUITE_RESULTS_NAME",
    "VALIDATION_CHALLENGE_PROTOCOL_SHA256",
    "ValidationChallengeError",
    "evaluate_candidate_validation_challenge",
    "load_locked_validation_predictions",
    "read_official_validation_challenge_results",
    "require_evaluated_partition",
    "run_official_validation_challenge_suite",
    "validate_validation_challenge_protocol",
]
