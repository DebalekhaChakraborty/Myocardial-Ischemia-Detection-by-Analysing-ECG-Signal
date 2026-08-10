"""Official B4-A/B4-B/B4-C validation challenge evidence.

The canonical B4 `validation_predictions.npz` artifacts contain **primary**
validation rows only (`ischemic_positive` and `background_negative`): the
development index is built by `build_validation_index`, which calls
`load_b4_references` with its `primary_only=True` default. Challenge confounder
rows therefore never reach those files, and negative challenge evidence cannot
be derived from them.

This module instead rebuilds the frozen validation challenge population from
validation metadata and scores it with each already-locked model:

    frozen validation metadata
        -> exact rate / axis / conduction challenge references
        -> validated raw physical-mV waveform reads
        -> already locked B4-A / B4-B / B4-C model (inference only)
        -> that candidate's already locked validation threshold
        -> rate / axis / conduction FPR evidence

Nothing is trained, no threshold is selected, no locked artifact is modified and
there is no route to the sealed test. Positive-context descriptives continue to
come from the locked primary predictions, where ischemic-positive rows and their
context flags already live; that evidence is kept explicitly separate from the
negative challenge inference.

The statistics are not restated here. `challenge_metrics` and
`positive_context_analysis` are the frozen production implementations shared
with the B0-B3 baselines, and the evidence levels come from the frozen
`CHALLENGE_EVIDENCE_POLICIES` table.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from cardiosentinel.baseline.cache import (
    read_json,
    require_nonversioned_path,
    write_json_atomic,
)
from cardiosentinel.baseline.metrics import (
    challenge_metrics,
    positive_context_analysis,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.evaluation.protocol import (
    CHALLENGE_EVIDENCE_POLICIES,
    POSITIVE_CONTEXT_EVIDENCE_LEVEL,
    POSITIVE_CONTEXT_FLAGS,
    challenge_evidence_policy,
)
from cardiosentinel.neural.data import B4WaveformDataset
from cardiosentinel.neural.integrity import (
    canonical_sha256,
    validate_development_feature_integrity,
    validate_development_source_integrity,
)
from cardiosentinel.neural.metadata import B4WindowReference, load_b4_references
from cardiosentinel.neural.protocol import (
    B4_PROTOCOL_SHA256,
    B4_SPLIT_SHA256,
    DATASET,
    DATASET_VERSION,
    FEATURE_CORPUS_SHA256,
    REPOSITORY_ROOT,
    SAMPLING_FREQUENCY_HZ,
    WINDOW_SAMPLES,
    protocol_sha256,
    require_development_partition,
)
from cardiosentinel.neural.resource_benchmark import (
    ARCHITECTURE_PROTOCOL_SHA256,
    OFFICIAL_ORDER,
    load_locked_model,
    validate_locked_model,
)

VALIDATION_CHALLENGE_PROTOCOL_NAME: Final = "B4_VALIDATION_CHALLENGE_PROTOCOL_V1"
VALIDATION_CHALLENGE_PROTOCOL_PATH: Final = (
    REPOSITORY_ROOT / "docs" / "B4_VALIDATION_CHALLENGE_PROTOCOL_V1.md"
)
VALIDATION_CHALLENGE_PROTOCOL_SHA256: Final = (
    "4ab7e2e6adcf1e4d4e88a4ca5114515abc917aef6c1927ee52aaf79b16a81be1"
)

SUITE_DIR_NAME: Final = "B4_architecture_validation_challenge_v1"
SUITE_ATTEMPT_NAME: Final = "VALIDATION_CHALLENGE_ATTEMPT.json"
SUITE_RESULTS_NAME: Final = "VALIDATION_CHALLENGE_RESULTS.json"
VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.npz"

SUITE_STATUS_STARTED: Final = "STARTED"
SUITE_STATUS_COMPLETE: Final = "COMPLETE"
SUITE_STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

EVALUATED_PARTITION: Final = "validation"
FORBIDDEN_PARTITIONS: Final = frozenset({"test"})
THRESHOLD_SOURCE: Final = "locked_experiment_lock.validation_threshold"
INFERENCE_BATCH_SIZE: Final = 256

CHALLENGE_FAMILIES: Final = (
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
)
CHALLENGE_NAMES: Final = ("rate_related", "axis_shift", "conduction_change")

# Frozen validation challenge population. The window counts come from Benchmark
# V1; the subject counts are the denominators already recorded in the frozen
# B0-B3 `challenge_metrics_validation.json` evidence, not a fresh derivation.
CHALLENGE_EXPECTED_COUNTS: Final = {
    "rate_related_confounder": {"windows": 4973, "subjects": 4},
    "axis_shift_confounder": {"windows": 3000, "subjects": 8},
    "conduction_change_confounder": {"windows": 164, "subjects": 1},
}
CHALLENGE_SELECTION_SHA256: Final = (
    "49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a"
)
CHALLENGE_TOTAL_WINDOWS: Final = 8137

# Primary validation population bound by every candidate lock.
PRIMARY_VALIDATION_COUNTS: Final = {
    "total": 473_897,
    "positive": 21_628,
    "negative": 452_269,
}
PRIMARY_FAMILY_SET: Final = frozenset(("ischemic_positive", "background_negative"))

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
    """Accept only the development validation partition; refuse the test one."""
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
    return require_development_partition(partition)


@dataclass(frozen=True, slots=True)
class ValidationChallengeIndex:
    """The frozen validation challenge population and its provenance."""

    references: tuple[B4WindowReference, ...]
    selection_sha256: str
    counts: dict[str, dict[str, int]]

    @property
    def target_families(self) -> np.ndarray:
        return np.asarray(
            [item.target_family for item in self.references], dtype=np.str_
        )

    @property
    def subject_ids(self) -> np.ndarray:
        return np.asarray(
            [item.subject_id for item in self.references], dtype=np.str_
        )


def challenge_selection_digest(references: tuple[B4WindowReference, ...]) -> str:
    """Canonically digest the sorted stable IDs of a challenge selection."""
    identifiers = sorted(item.stable_id for item in references)
    if len(set(identifiers)) != len(identifiers):
        raise ValidationChallengeError(
            "The validation challenge selection contains duplicate stable IDs."
        )
    return canonical_sha256(identifiers)


def build_validation_challenge_index(feature_root: Path) -> ValidationChallengeIndex:
    """Rebuild and verify the frozen validation challenge population.

    This is provenance work over validation metadata. No waveform is read here,
    no model is involved, and the test partition is never named.
    """
    require_evaluated_partition(EVALUATED_PARTITION)
    references = tuple(
        item
        for item in load_b4_references(
            Path(feature_root), EVALUATED_PARTITION, primary_only=False
        )
        if item.target_family in CHALLENGE_FAMILIES
    )
    counts: dict[str, dict[str, int]] = {}
    for family in CHALLENGE_FAMILIES:
        rows = [item for item in references if item.target_family == family]
        observed = {
            "windows": len(rows),
            "subjects": len({item.subject_id for item in rows}),
        }
        expected = CHALLENGE_EXPECTED_COUNTS[family]
        if observed != expected:
            raise ValidationChallengeError(
                f"Frozen challenge population mismatch for {family}: "
                f"expected {expected}, observed {observed}."
            )
        counts[family] = observed
    if len(references) != CHALLENGE_TOTAL_WINDOWS:
        raise ValidationChallengeError(
            f"Expected {CHALLENGE_TOTAL_WINDOWS} challenge windows, "
            f"observed {len(references)}."
        )
    digest = challenge_selection_digest(references)
    if digest != CHALLENGE_SELECTION_SHA256:
        raise ValidationChallengeError(
            "The validation challenge selection digest differs from the frozen "
            "identity."
        )
    return ValidationChallengeIndex(
        references=references, selection_sha256=digest, counts=counts
    )


def _model_state_digest(model: torch.nn.Module) -> str:
    """Digest every parameter and buffer so inference cannot mutate the model."""
    with torch.no_grad():
        items = [
            (name, np.asarray(tensor.detach().cpu().numpy()).tobytes().hex())
            for name, tensor in sorted(model.state_dict().items())
        ]
    return canonical_sha256(items)


def score_challenge_windows(
    run_dir: Path,
    lock: dict[str, Any],
    index: ValidationChallengeIndex,
    source: Path,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score the frozen challenge windows with the locked model, inference only.

    The model is loaded from `model_selected.pt` alone, put in eval mode with
    gradients disabled, and its full state is digested before and after so a
    silent mutation is impossible. No optimizer, training checkpoint, backward
    pass or threshold search exists on this path.
    """
    model = load_locked_model(Path(run_dir), lock)
    model.eval()
    model.requires_grad_(False)
    state_before = _model_state_digest(model)

    reader = B4WaveformDataset(index.references, Path(source))
    scores = np.empty(len(index.references), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(index.references), batch_size):
            chunk = index.references[start : start + batch_size]
            waveforms = torch.stack(
                [reader.read_waveform(reference) for reference in chunk]
            )
            if waveforms.shape[1:] != (1, WINDOW_SAMPLES):
                raise ValidationChallengeError(
                    "Challenge waveform batch violates the frozen input contract."
                )
            if not torch.isfinite(waveforms).all():
                raise ValidationChallengeError(
                    "A challenge waveform contains a non-finite sample."
                )
            logits = model(waveforms).reshape(-1)
            scores[start : start + len(chunk)] = (
                torch.sigmoid(logits).to(torch.float64).numpy()
            )

    state_after = _model_state_digest(model)
    if state_before != state_after:
        raise ValidationChallengeError(
            "The locked model state changed during challenge inference."
        )
    if not np.all(np.isfinite(scores)):
        raise ValidationChallengeError(
            "Challenge inference produced a non-finite score."
        )

    receipt = {
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": True,
        "inference_mode": "torch.no_grad + eval + requires_grad_(False)",
        "gradients_enabled": False,
        "optimizer_used": False,
        "training_checkpoint_used": False,
        "batch_size": batch_size,
        "windows_scored": int(scores.shape[0]),
        "waveform_contract": {
            "channels": 1,
            "samples": WINDOW_SAMPLES,
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "physical_unit": "mV",
            "dtype": "float32",
            "processing_profile": "raw",
        },
        "waveform_reads": reader.stats.source_reads,
    }
    return scores, receipt


def _require_frozen_provenance(lock: dict[str, Any]) -> dict[str, Any]:
    """Require frozen provenance values, not merely non-null fields."""
    exact = {
        "split_sha256": B4_SPLIT_SHA256,
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
    }
    for field, expected in exact.items():
        observed = lock.get(field)
        if observed != expected:
            raise ValidationChallengeError(
                f"The experiment lock binds {field}={observed!r}, expected "
                f"{expected!r}."
            )
    required = (
        "training_selection_sha256",
        "development_feature_integrity_sha256",
        "validation_predictions_sha256",
    )
    missing = [field for field in required if lock.get(field) is None]
    if missing:
        raise ValidationChallengeError(
            f"The experiment lock is missing mandatory provenance: {sorted(missing)}."
        )
    optional = (
        "environment_dependency_digest",
        "candidate_architecture",
        "architecture_protocol_sha256",
    )
    return {
        "required_fields_present": list(exact) + list(required),
        "optional_fields_present": [f for f in optional if lock.get(f) is not None],
        "optional_fields_absent": [f for f in optional if lock.get(f) is None],
    }


def load_primary_validation_predictions(
    run_dir: Path, lock: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Load the locked PRIMARY validation predictions for positive context.

    These carry primary rows only. Nothing here is repaired or coerced before it
    has been validated; a disagreement with the lock is a refusal.
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

    # Validate before any dtype coercion: a malformed artifact is refused.
    labels = columns["label"]
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValidationChallengeError(
            "Primary validation labels must already be an integer 0/1 column."
        )
    if not np.all(np.isin(labels, (0, 1))):
        raise ValidationChallengeError("Primary validation labels must be 0 or 1.")
    scores = columns["score"]
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValidationChallengeError("Validation scores must be a floating column.")
    if not np.all(np.isfinite(scores)):
        raise ValidationChallengeError(
            "The validation prediction artifact contains a non-finite score."
        )
    families = set(np.unique(columns["target_family"]).tolist())
    if not families <= PRIMARY_FAMILY_SET:
        raise ValidationChallengeError(
            "The locked validation predictions must contain primary families "
            f"only; observed {sorted(families)}."
        )
    if len(set(columns["stable_id"].tolist())) != rows:
        raise ValidationChallengeError(
            "The validation prediction artifact contains duplicate stable IDs."
        )

    expected_rows = lock.get("validation_rows") or {}
    if expected_rows.get("partition") not in (None, EVALUATED_PARTITION):
        raise ValidationChallengeError(
            "The lock's validation_rows does not describe the validation partition."
        )
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    for field, observed in (
        ("total", rows),
        ("positive", positives),
        ("negative", negatives),
    ):
        bound = expected_rows.get(field)
        if bound is not None and bound != observed:
            raise ValidationChallengeError(
                f"Primary validation {field} is {observed} but the lock records "
                f"{bound}."
            )
    return columns


def evaluate_candidate_validation_challenge(
    run_dir: Path,
    *,
    official_model: str,
    feature_root: Path,
    source: Path,
    challenge_index: ValidationChallengeIndex | None = None,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> dict[str, Any]:
    """Produce one candidate's validation challenge evidence."""
    if official_model not in OFFICIAL_ORDER:
        raise ValidationChallengeError(f"Unknown official model {official_model!r}.")
    protocol_digest = validate_validation_challenge_protocol()
    partition = require_evaluated_partition(EVALUATED_PARTITION)

    lock = validate_locked_model(Path(run_dir), official_model=official_model)
    binding = _require_frozen_provenance(lock)

    threshold = lock.get("validation_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        raise ValidationChallengeError(
            "The experiment lock does not bind a finite validation threshold."
        )
    threshold = float(threshold)

    index = challenge_index or build_validation_challenge_index(Path(feature_root))
    feature_receipt = validate_development_feature_integrity(Path(feature_root))
    source_receipt = validate_development_source_integrity(
        Path(source), feature_receipt
    )
    scores, inference = score_challenge_windows(
        Path(run_dir), lock, index, Path(source), batch_size=batch_size
    )

    frozen = challenge_metrics(
        index.target_families, scores, index.subject_ids, threshold
    )
    challenges: dict[str, Any] = {}
    for name in CHALLENGE_NAMES:
        policy = challenge_evidence_policy(name)
        measured = frozen[name]
        challenges[name] = {
            "target_family": policy.target_family,
            "evidence_status": policy.evidence_level,
            "is_headline_metric": policy.is_headline_metric,
            "challenge_window_count": measured["challenge_window_count"],
            "false_positive_count": measured["false_positive_count"],
            "false_positive_fraction": measured["false_positive_fraction"],
            "supporting_subject_count": measured["contributing_subject_count"],
            "bootstrap_permitted": measured["bootstrap_permitted"],
            "frozen_metric": measured,
        }

    primary = load_primary_validation_predictions(Path(run_dir), lock)
    payload: dict[str, Any] = {
        "official_model": official_model,
        "experiment_id": lock["experiment_id"],
        "architecture": lock.get("candidate_architecture")
        or lock.get("model", {}).get("architecture"),
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "locked_inference_model": lock["locked_inference_model"],
        "locked_validation_threshold": threshold,
        "threshold_source": THRESHOLD_SOURCE,
        "threshold_selected_by_evaluator": False,
        "partition": partition,
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": lock["split_sha256"],
        "feature_corpus_sha256": lock["feature_corpus_sha256"],
        "training_selection_sha256": lock["training_selection_sha256"],
        "development_feature_integrity_sha256": lock[
            "development_feature_integrity_sha256"
        ],
        "development_source_integrity_sha256": source_receipt[
            "development_source_integrity_sha256"
        ],
        "provenance_binding": binding,
        "challenge_selection_sha256": index.selection_sha256,
        "challenge_population": index.counts,
        "challenge_window_total": len(index.references),
        "challenges": challenges,
        "challenge_evidence_source": "locked_model_inference_on_frozen_challenge_rows",
        "inference_receipt": inference,
        "positive_context": {
            "evidence_status": POSITIVE_CONTEXT_EVIDENCE_LEVEL,
            "evidence_source": "locked_primary_validation_predictions",
            "context_flags": list(POSITIVE_CONTEXT_FLAGS),
            "validation_prediction_sha256": lock["validation_predictions_sha256"],
            "primary_window_count": int(primary["score"].shape[0]),
            "strata": positive_context_analysis(
                np.asarray(primary["label"], dtype=np.int64),
                np.asarray(primary["score"], dtype=np.float64),
                np.asarray(primary["subject_id"], dtype=np.str_),
                np.asarray(primary["context_flags"], dtype=np.str_),
                threshold,
            ),
        },
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "validation_challenge_protocol_sha256": protocol_digest,
        "b4_protocol_sha256": B4_PROTOCOL_SHA256,
        "dataset_accessed": True,
        "waveform_accessed": True,
        "test_accessed": False,
        "model_inference_performed": True,
        "training_performed": False,
        "threshold_search_performed": False,
    }
    payload["challenge_result_sha256"] = canonical_sha256(payload)
    return payload


def _write_attempt(path: Path, payload: dict[str, Any]) -> None:
    """Persist the attempt receipt durably without ever releasing the claim."""
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        handle.flush()
        os.fsync(handle.fileno())
    return canonical_sha256(payload)


def run_official_validation_challenge_suite(
    run_directories: dict[str, Path],
    run_root: Path,
    feature_root: Path,
    source: Path,
    *,
    command: str = "cardiosentinel b4 validation-challenge",
) -> dict[str, Any]:
    """Evaluate B4-A, B4-B and B4-C in one exclusive official invocation.

    Exactly three locked run directories are required, evaluated in the frozen
    order. There is deliberately no evaluator, backend, metric or retry
    parameter: official evidence can only come from the frozen production
    challenge implementation applied to locked-model inference over the frozen
    challenge rows.
    """
    if set(run_directories) != set(OFFICIAL_ORDER):
        raise ValidationChallengeError(
            "The official validation challenge suite requires exactly "
            f"{list(OFFICIAL_ORDER)}; received {sorted(run_directories)}."
        )
    protocol_digest = validate_validation_challenge_protocol()
    provenance = git_provenance(REPOSITORY_ROOT)
    if provenance["git_dirty"]:
        raise ValidationChallengeError(
            "Official challenge evidence requires a clean Git checkout."
        )
    started = time.monotonic()

    suite_dir = require_nonversioned_path(
        Path(run_root) / SUITE_DIR_NAME,
        "official B4 validation challenge evidence",
    )
    results_path = suite_dir / SUITE_RESULTS_NAME
    if results_path.exists():
        raise ValidationChallengeError(
            f"An official challenge result already exists at {results_path}."
        )
    attempt_path = suite_dir / SUITE_ATTEMPT_NAME
    attempt = {
        "suite": SUITE_DIR_NAME,
        "attempt_sequence": 1,
        "attempt_status": SUITE_STATUS_STARTED,
        "repeat_attempt_permitted": False,
        "selective_candidate_retry_permitted": False,
        "candidate_order": list(OFFICIAL_ORDER),
        "command": command,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "validation_challenge_protocol_sha256": protocol_digest,
        "created_at_utc_audit_only": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    attempt_sha256 = _claim_suite_attempt(attempt_path, attempt)

    try:
        index = build_validation_challenge_index(Path(feature_root))
        results: dict[str, Any] = {}
        for name in OFFICIAL_ORDER:
            result = evaluate_candidate_validation_challenge(
                Path(run_directories[name]),
                official_model=name,
                feature_root=Path(feature_root),
                source=Path(source),
                challenge_index=index,
            )
            recorded = result.get("challenge_result_sha256")
            body = {
                k: v for k, v in result.items() if k != "challenge_result_sha256"
            }
            if recorded != canonical_sha256(body):
                raise ValidationChallengeError(
                    f"{name} challenge evidence failed digest revalidation."
                )
            if result["test_accessed"] or result["training_performed"]:
                raise ValidationChallengeError(
                    f"{name} challenge evidence claims test access or training."
                )
            if result["challenge_selection_sha256"] != index.selection_sha256:
                raise ValidationChallengeError(
                    f"{name} was scored on a different challenge selection."
                )
            results[name] = result

        suite: dict[str, Any] = {
            "suite": SUITE_DIR_NAME,
            "command": command,
            "attempt_sequence": 1,
            "suite_attempt_sha256": attempt_sha256,
            "candidate_order": list(OFFICIAL_ORDER),
            "partition": EVALUATED_PARTITION,
            "git_sha": provenance["git_sha"],
            "git_dirty": provenance["git_dirty"],
            "validation_challenge_protocol_sha256": protocol_digest,
            "b4_protocol_sha256": B4_PROTOCOL_SHA256,
            "architecture_protocol_sha256": ARCHITECTURE_PROTOCOL_SHA256,
            "metric_definitions": dict(METRIC_DEFINITIONS),
            "metric_implementation_sha256": _metric_implementation_digest(),
            "challenge_evidence_policies": {
                name: {
                    "target_family": policy.target_family,
                    "evidence_level": policy.evidence_level,
                    "is_headline_metric": policy.is_headline_metric,
                    "supports_inferential_bootstrap": (
                        policy.supports_inferential_bootstrap
                    ),
                }
                for name, policy in CHALLENGE_EVIDENCE_POLICIES.items()
            },
            "challenge_selection_sha256": index.selection_sha256,
            "challenge_population": index.counts,
            "threshold_source": THRESHOLD_SOURCE,
            "experiment_lock_sha256": {
                n: results[n]["experiment_lock_sha256"] for n in OFFICIAL_ORDER
            },
            "checkpoint_sha256": {
                n: results[n]["checkpoint_sha256"] for n in OFFICIAL_ORDER
            },
            "locked_validation_threshold": {
                n: results[n]["locked_validation_threshold"] for n in OFFICIAL_ORDER
            },
            "challenge_result_sha256": {
                n: results[n]["challenge_result_sha256"] for n in OFFICIAL_ORDER
            },
            "candidate_results": results,
            "dataset_accessed": True,
            "test_accessed": False,
            "training_performed": False,
            "architecture_selection_performed": False,
            "suite_duration_seconds": time.monotonic() - started,
        }
        suite["validation_challenge_suite_sha256"] = canonical_sha256(suite)
        write_json_atomic(results_path, suite)
        _write_attempt(
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
    except BaseException as error:
        # The claim is never released. It is rewritten in place so a reviewer
        # can see exactly how the one official attempt was consumed.
        _write_attempt(
            attempt_path,
            {
                **attempt,
                "attempt_status": SUITE_STATUS_FAILED,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=20),
                "human_review_required": True,
                "repeat_attempt_permitted": False,
                "selective_candidate_retry_permitted": False,
                "automatic_retry_performed": False,
            },
        )
        raise


def _metric_implementation_digest() -> str:
    """Bind the actual metric source, not merely the function names."""
    import inspect

    sources = [
        inspect.getsource(challenge_metrics),
        inspect.getsource(positive_context_analysis),
        repr(sorted(CHALLENGE_EVIDENCE_POLICIES.items())),
    ]
    return canonical_sha256(sources)


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
    "CHALLENGE_EXPECTED_COUNTS",
    "CHALLENGE_SELECTION_SHA256",
    "SUITE_ATTEMPT_NAME",
    "SUITE_DIR_NAME",
    "SUITE_RESULTS_NAME",
    "VALIDATION_CHALLENGE_PROTOCOL_SHA256",
    "ValidationChallengeError",
    "ValidationChallengeIndex",
    "build_validation_challenge_index",
    "challenge_selection_digest",
    "evaluate_candidate_validation_challenge",
    "load_primary_validation_predictions",
    "read_official_validation_challenge_results",
    "require_evaluated_partition",
    "run_official_validation_challenge_suite",
    "score_challenge_windows",
    "validate_validation_challenge_protocol",
]
