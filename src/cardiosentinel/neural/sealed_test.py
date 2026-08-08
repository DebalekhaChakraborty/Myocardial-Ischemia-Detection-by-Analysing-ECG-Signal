"""One-shot sealed-test evaluator for the locked B4 development experiment.

This module performs the single predeclared B4 test evaluation. It is separate
from every development module by construction:

* Development types cannot hold a test row. `B4WindowReference` rejects the test
  partition in its own validator, so sealed-test rows use the distinct
  `SealedTestWindowReference` type defined here.
* No test-resolving function can run without a `SealedTestAccess` token, and the
  only way to obtain that token is `open_sealed_test_attempt`, which returns it
  exclusively after `TEST_ATTEMPT.json` has been written and fsynced to durable
  storage. Receipt-before-access is therefore structural, not merely ordered.
* The checkpoint and the decision threshold come only from the immutable
  development lock. This module never selects, tunes, or recomputes a threshold,
  never constructs an optimizer, and never calls backward.

There is exactly one attempt. No override, force, retry, or reset exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from cardiosentinel.baseline.cache import (
    FEATURE_MANIFEST_NAME,
    read_json,
    require_nonversioned_path,
)
from cardiosentinel.baseline.metrics import (
    binary_metrics,
    challenge_bootstrap_confidence_intervals,
    challenge_metrics,
    positive_context_analysis,
    subject_bootstrap_confidence_intervals,
    subject_macro_metrics,
)
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.evaluation.protocol import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
from cardiosentinel.neural.data import B4WaveformDataset
from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.experiment import (
    EXPERIMENT_ID,
    PROGRAM_IDENTITY,
    input_contract,
    resolve_run_dir,
    validate_experiment_lock,
)
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.metadata import _manifest_identity, _metadata_arrays
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.protocol import (
    DATASET,
    DATASET_VERSION,
    PRIMARY_FAMILIES,
    REPOSITORY_ROOT,
    WINDOW_SAMPLES,
    validate_frozen_protocol,
)
from cardiosentinel.neural.provenance import runtime_environment
from cardiosentinel.signal.io import read_local_segment

SEALED_TEST_PARTITION = "test"
TEST_ATTEMPT_NAME = "TEST_ATTEMPT.json"
TEST_METRICS_NAME = "TEST_METRICS.json"
TEST_PREDICTIONS_NAME = "TEST_PREDICTIONS.npz"
TEST_AUDIT_NAME = "TEST_AUDIT.json"

ATTEMPT_STARTED = "STARTED"
ATTEMPT_COMPLETE = "COMPLETE"
ATTEMPT_FAILED = "FAILED_OR_INTERRUPTED"
ATTEMPT_SEQUENCE = 1

DEFAULT_COMMAND = "cardiosentinel b4 evaluate-locked-test"
INFERENCE_BATCH_SIZE = 256

# Frozen Benchmark V1 primary test population. These are historical protocol
# facts recorded before this evaluator existed; they are verified, never chosen.
SEALED_TEST_COUNTS = {
    "positive": 20_899,
    "negative": 432_905,
    "total": 453_804,
    "subjects": 12,
}
CHALLENGE_FAMILIES = (
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
)


class SealedTestAttemptError(RuntimeError):
    """Raised when the one-shot attempt contract forbids proceeding."""


@dataclass(frozen=True, slots=True)
class SealedTestAccess:
    """Capability proving the durable one-shot attempt receipt already exists.

    Only `open_sealed_test_attempt` constructs this. Every function that can
    resolve, open, or read sealed-test data demands one.
    """

    run_dir: Path
    receipt_path: Path
    receipt_sha256: str
    experiment_lock_sha256: str
    checkpoint_sha256: str
    locked_threshold: float


@dataclass(frozen=True, slots=True)
class SealedTestWindowReference:
    """Identity and target metadata for one sealed-test window.

    Deliberately distinct from `B4WindowReference`, which cannot represent a
    test row at all. Field names match so the frozen lossless segment validator
    applies unchanged.
    """

    stable_id: str
    record_id: str
    subject_id: str
    channel_index: int
    start_sample: int
    end_sample: int
    partition: str
    target_family: str
    context_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.partition != SEALED_TEST_PARTITION:
            raise ValueError("A sealed-test reference must hold the test partition.")
        if self.channel_index < 0 or self.start_sample < 0:
            raise ValueError("Sealed-test window identity contains a negative index.")
        if self.end_sample - self.start_sample != WINDOW_SAMPLES:
            raise ValueError("Sealed-test window must contain exactly 2500 samples.")
        expected = (
            f"{DATASET}:{self.record_id}:{self.channel_index}:"
            f"{self.start_sample}:{self.end_sample}"
        )
        if self.stable_id != expected:
            raise ValueError("Sealed-test stable ID does not match window identity.")

    @property
    def is_primary(self) -> bool:
        return self.target_family in PRIMARY_FAMILIES

    @property
    def binary_label(self) -> int:
        if not self.is_primary:
            raise ValueError("Only primary sealed-test rows carry a binary label.")
        return int(self.target_family == "ischemic_positive")


def _require_access(access: SealedTestAccess) -> SealedTestAccess:
    """Refuse any sealed-test operation without a durable attempt receipt."""
    if not isinstance(access, SealedTestAccess):
        raise SealedTestAttemptError(
            "Sealed-test access requires a durable attempt receipt token."
        )
    if not access.receipt_path.is_file():
        raise SealedTestAttemptError(
            "The sealed-test attempt receipt is no longer present on disk."
        )
    return access


def write_json_durable(path: Path, payload: dict[str, Any]) -> str:
    """Write JSON atomically and fsync both file and directory; return its hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash every parameter and buffer so weight mutation is detectable."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _execution_payload(
    command: str,
    source: Path,
    feature_root: Path,
    run_root: Path,
    requested_device: str | None,
    resolved_device: str,
    workers: int,
) -> dict[str, Any]:
    paths = {
        "source": str(Path(source).expanduser().resolve()),
        "feature_root": str(Path(feature_root).expanduser().resolve()),
        "run_root": str(Path(run_root).expanduser().resolve()),
    }
    rendered = [PROGRAM_IDENTITY, *command.split()[1:]]
    for name in ("source", "feature_root", "run_root"):
        rendered += [f"--{name.replace('_', '-')}", paths[name]]
    if requested_device is not None:
        rendered += ["--device", requested_device]
    rendered += ["--workers", str(workers)]
    return {
        "experiment_id": EXPERIMENT_ID,
        "program": PROGRAM_IDENTITY,
        "command": command,
        **paths,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "workers": workers,
        "shell_command": " ".join(rendered),
    }


def open_sealed_test_attempt(
    source: Path,
    feature_root: Path,
    run_root: Path,
    *,
    command: str = DEFAULT_COMMAND,
    requested_device: str | None = None,
    workers: int = 0,
) -> tuple[SealedTestAccess, dict[str, Any]]:
    """Validate the lock, then durably record attempt #1 before any test access.

    Every check here reads development artifacts only. If any check fails, no
    receipt is written and the sealed test remains unopened.
    """
    protocol_sha256 = validate_frozen_protocol()
    provenance = git_provenance(REPOSITORY_ROOT)
    if provenance["git_dirty"]:
        raise SealedTestAttemptError(
            "The sealed-test evaluation requires a clean evaluator checkout."
        )
    run_dir = resolve_run_dir(run_root)
    lock = validate_experiment_lock(run_dir)
    if lock["experiment_id"] != EXPERIMENT_ID:
        raise SealedTestAttemptError("The development lock has the wrong experiment.")
    if lock["status"] != "locked_for_one_shot_test":
        raise SealedTestAttemptError("The development lock is not sealed for test.")
    if lock["test"] is not None:
        raise SealedTestAttemptError("The development lock already records a test.")
    if lock["git_dirty"] is not False:
        raise SealedTestAttemptError("The development lock is not from a clean tree.")

    checkpoint = run_dir / str(lock["locked_inference_model"])
    if not checkpoint.is_file():
        raise SealedTestAttemptError("The locked inference checkpoint is absent.")
    observed = sha256_file(checkpoint)
    if observed != lock["checkpoint_sha256"]:
        raise SealedTestAttemptError("The locked checkpoint SHA-256 does not match.")

    threshold = lock["validation_threshold"]
    if not isinstance(threshold, float) or not np.isfinite(threshold):
        raise SealedTestAttemptError("The lock has no finite validation threshold.")

    receipt_path = run_dir / TEST_ATTEMPT_NAME
    if receipt_path.exists():
        existing = read_json(receipt_path) if receipt_path.is_file() else {}
        raise SealedTestAttemptError(
            "A B4 sealed-test attempt already exists "
            f"(status={existing.get('attempt_status')}, "
            f"sequence={existing.get('attempt_sequence')}). There is exactly one "
            "predeclared attempt; it cannot be repeated, reset, or overridden, "
            "and any further evaluation requires documented human review."
        )

    determinism = initialize_determinism(requested_device=requested_device)
    environment = runtime_environment(determinism.device, workers)
    execution = _execution_payload(
        command, source, feature_root, run_root, requested_device,
        determinism.device, workers,
    )
    receipt = {
        "experiment_id": EXPERIMENT_ID,
        "attempt_sequence": ATTEMPT_SEQUENCE,
        "attempt_status": ATTEMPT_STARTED,
        "repeat_attempt_permitted": False,
        "experiment_lock_sha256": lock["experiment_lock_sha256"],
        "locked_checkpoint_sha256": lock["checkpoint_sha256"],
        "locked_validation_threshold": threshold,
        "threshold_selection_rule": lock["threshold_selection_rule"],
        "development_git_sha": lock["git_sha"],
        "evaluator_git_sha": provenance["git_sha"],
        "evaluator_git_dirty": provenance["git_dirty"],
        "protocol_sha256": protocol_sha256,
        "split_sha256": lock["split_sha256"],
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "input_contract": input_contract(),
        "environment": environment,
        "execution": execution,
        "created_at_utc_audit_only": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "test_data_access_began": False,
        "test": None,
    }
    # The receipt reaches durable storage here. Nothing above this line has
    # resolved or opened a single sealed-test artifact.
    receipt_sha256 = write_json_durable(receipt_path, receipt)
    access = SealedTestAccess(
        run_dir=run_dir,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
        experiment_lock_sha256=lock["experiment_lock_sha256"],
        checkpoint_sha256=lock["checkpoint_sha256"],
        locked_threshold=threshold,
    )
    return access, lock


def _update_attempt(
    access: SealedTestAccess, **fields: Any
) -> dict[str, Any]:
    """Amend the attempt in place; the fact that attempt #1 occurred persists."""
    receipt = read_json(access.receipt_path)
    receipt.update(fields)
    receipt["attempt_sequence"] = ATTEMPT_SEQUENCE
    receipt["repeat_attempt_permitted"] = False
    write_json_durable(access.receipt_path, receipt)
    return receipt


def load_sealed_test_references(
    access: SealedTestAccess, feature_root: Path
) -> tuple[SealedTestWindowReference, ...]:
    """Read sealed-test identity metadata. Requires the durable attempt receipt."""
    _require_access(access)
    root = require_nonversioned_path(feature_root, "B4 sealed-test metadata root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    _manifest_identity(manifest)
    entries = tuple(
        sorted(
            (
                entry
                for entry in manifest.get("records", ())
                if entry.get("partition") == SEALED_TEST_PARTITION
                and entry.get("status") == "complete"
            ),
            key=lambda entry: str(entry["record_id"]),
        )
    )
    if not entries:
        raise ValueError("The sealed-test corpus has no complete records.")
    references: list[SealedTestWindowReference] = []
    for entry in entries:
        cache_path = (root / str(entry["cache_path"])).resolve()
        try:
            cache_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Sealed-test cache path escapes its root.") from error
        arrays = _metadata_arrays(cache_path)
        if len({array.size for array in arrays}) != 1:
            raise ValueError("Sealed-test metadata arrays are not row-aligned.")
        for values in zip(*arrays, strict=True):
            (
                stable_id, record_id, subject_id, channel_index,
                start, end, partition, family, context,
            ) = values
            if str(partition) != SEALED_TEST_PARTITION:
                raise ValueError("Sealed-test metadata row has the wrong partition.")
            references.append(
                SealedTestWindowReference(
                    stable_id=str(stable_id),
                    record_id=str(record_id),
                    subject_id=str(subject_id),
                    channel_index=int(channel_index),
                    start_sample=int(start),
                    end_sample=int(end),
                    partition=SEALED_TEST_PARTITION,
                    target_family=str(family),
                    context_flags=tuple(
                        item for item in str(context).split("|") if item
                    ),
                )
            )
    identifiers = [item.stable_id for item in references]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Sealed-test metadata contains duplicate stable IDs.")
    # Record-aware ordering keeps canonical source reads sequential per channel.
    return tuple(
        sorted(
            references,
            key=lambda item: (item.record_id, item.channel_index, item.start_sample),
        )
    )


def verify_primary_population(
    references: tuple[SealedTestWindowReference, ...]
) -> dict[str, int]:
    """Confirm the observed primary population equals the frozen V1 counts."""
    primary = [item for item in references if item.is_primary]
    observed = {
        "positive": sum(item.binary_label == 1 for item in primary),
        "negative": sum(item.binary_label == 0 for item in primary),
        "total": len(primary),
        "subjects": len({item.subject_id for item in primary}),
    }
    if observed != SEALED_TEST_COUNTS:
        raise ValueError(
            f"Sealed-test primary population differs from Benchmark V1: {observed}"
        )
    return observed


def load_locked_model(
    access: SealedTestAccess, run_dir: Path, lock: dict[str, Any], device: str
) -> torch.nn.Module:
    """Load only the locked weights; no optimizer state is read or created."""
    _require_access(access)
    checkpoint = run_dir / str(lock["locked_inference_model"])
    if sha256_file(checkpoint) != access.checkpoint_sha256:
        raise SealedTestAttemptError("The locked checkpoint changed before inference.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "optimizer" in state:
        raise SealedTestAttemptError(
            "The locked inference artifact must not carry optimizer state."
        )
    model = B4CompactCNN()
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    model.requires_grad_(False)
    return model


def _read_waveform(
    source: Path, reference: SealedTestWindowReference
) -> NDArray[np.float32]:
    """Read one canonical mV window and apply the frozen lossless validation."""
    segment = read_local_segment(
        source,
        DATASET,
        reference.record_id,
        reference.start_sample,
        reference.end_sample,
        (reference.channel_index,),
    )
    # Reuse the development validator verbatim so the sealed-test input contract
    # cannot drift from the validated development path.
    B4WaveformDataset._validate_segment(reference, segment)
    return np.asarray(segment.values[:, 0], dtype=np.float32)


def score_sealed_test(
    access: SealedTestAccess,
    source: Path,
    references: tuple[SealedTestWindowReference, ...],
    model: torch.nn.Module,
    device: str,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    _reader=None,
) -> NDArray[np.float64]:
    """Score every supplied row exactly once under no_grad; weights never change."""
    _require_access(access)
    if model.training:
        raise SealedTestAttemptError("Sealed-test inference requires eval mode.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise SealedTestAttemptError("Sealed-test inference requires no gradients.")
    reader = _reader or _read_waveform
    torch_device = torch.device(device)
    scores = np.empty(len(references), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(references), batch_size):
            chunk = references[start : start + batch_size]
            waveforms = np.stack([reader(source, item) for item in chunk])
            batch = torch.from_numpy(waveforms).reshape(
                len(chunk), 1, WINDOW_SAMPLES
            )
            logits = model(batch.to(device=torch_device, dtype=torch.float32))
            scores[start : start + len(chunk)] = (
                torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            )
    if not np.isfinite(scores).all():
        raise ValueError("Sealed-test scoring produced a non-finite score.")
    return scores


def _arrays(
    references: tuple[SealedTestWindowReference, ...]
) -> dict[str, NDArray[Any]]:
    return {
        "stable_id": np.asarray([i.stable_id for i in references], dtype=np.str_),
        "subject_id": np.asarray([i.subject_id for i in references], dtype=np.str_),
        "record_id": np.asarray([i.record_id for i in references], dtype=np.str_),
        "channel_index": np.asarray(
            [i.channel_index for i in references], dtype=np.int64
        ),
        "target_family": np.asarray(
            [i.target_family for i in references], dtype=np.str_
        ),
        "context_flags": np.asarray(
            ["|".join(i.context_flags) for i in references], dtype=np.str_
        ),
    }


def build_test_evidence(
    references: tuple[SealedTestWindowReference, ...],
    scores: NDArray[np.float64],
    threshold: float,
) -> dict[str, Any]:
    """Compute frozen primary, macro, bootstrap, challenge and context evidence.

    The threshold is supplied by the caller from the immutable lock. No
    threshold is selected, searched, or optimized anywhere in this module.
    """
    columns = _arrays(references)
    primary_mask = np.asarray([item.is_primary for item in references])
    primary = [item for item in references if item.is_primary]
    labels = np.asarray([item.binary_label for item in primary], dtype=np.int64)
    primary_scores = scores[primary_mask]
    primary_subjects = columns["subject_id"][primary_mask]
    primary_contexts = columns["context_flags"][primary_mask]

    pooled = binary_metrics(labels, primary_scores, threshold)
    macro = subject_macro_metrics(labels, primary_scores, primary_subjects, threshold)
    bootstrap = subject_bootstrap_confidence_intervals(
        labels,
        primary_scores,
        primary_subjects,
        threshold,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    challenge = challenge_metrics(
        columns["target_family"], scores, columns["subject_id"], threshold
    )
    challenge_bootstrap = challenge_bootstrap_confidence_intervals(
        columns["target_family"],
        scores,
        columns["subject_id"],
        threshold,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    context = positive_context_analysis(
        labels, primary_scores, primary_subjects, primary_contexts, threshold
    )
    return {
        "partition": SEALED_TEST_PARTITION,
        "evidence_class": "sealed_one_shot_test_result",
        "sampled": False,
        "threshold": threshold,
        "threshold_source": "immutable_development_experiment_lock",
        "threshold_selected_on_test": False,
        "primary_population": {
            "row_count": int(labels.size),
            "positive_count": int(pooled["positive_count"]),
            "negative_count": int(pooled["negative_count"]),
            "positive_prevalence": pooled["positive_prevalence"],
            "subject_count": int(np.unique(primary_subjects).size),
        },
        "scored_row_count": int(scores.size),
        "pooled": pooled,
        "subject_macro": macro,
        "subject_bootstrap": bootstrap,
        "challenge": challenge,
        "challenge_bootstrap": challenge_bootstrap,
        "positive_context": context,
        "score_semantics": (
            "uncalibrated sigmoid model score; not calibrated probability"
        ),
    }


def write_test_predictions(
    access: SealedTestAccess,
    path: Path,
    references: tuple[SealedTestWindowReference, ...],
    scores: NDArray[np.float64],
) -> str:
    """Persist identity, label and score only; never a waveform or feature row."""
    _require_access(access)
    columns = _arrays(references)
    labels = np.asarray(
        [item.binary_label if item.is_primary else -1 for item in references],
        dtype=np.int64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **columns, label=labels, score=scores)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def evaluate_locked_test(
    source: Path,
    feature_root: Path,
    run_root: Path,
    *,
    command: str = DEFAULT_COMMAND,
    requested_device: str | None = None,
    workers: int = 0,
    _reader=None,
) -> dict[str, Any]:
    """Perform the single predeclared B4 sealed-test evaluation.

    The attempt receipt is written durably before any sealed-test artifact is
    resolved. There is no force, retry, reset, threshold, checkpoint or seed
    option: the checkpoint and threshold come only from the development lock.
    """
    started = time.monotonic()
    access, lock = open_sealed_test_attempt(
        source,
        feature_root,
        run_root,
        command=command,
        requested_device=requested_device,
        workers=workers,
    )
    run_dir = access.run_dir
    device = read_json(access.receipt_path)["execution"]["resolved_device"]
    test_access_began = False
    try:
        references = load_sealed_test_references(access, feature_root)
        test_access_began = True
        _update_attempt(access, test_data_access_began=True)
        primary_counts = verify_primary_population(references)

        model = load_locked_model(access, run_dir, lock, device)
        model_sha_before = model_state_sha256(model)
        scores = score_sealed_test(
            access, source, references, model, device, _reader=_reader
        )
        model_sha_after = model_state_sha256(model)
        if model_sha_before != model_sha_after:
            raise SealedTestAttemptError(
                "The locked B4 weights changed during sealed-test inference."
            )

        evidence = build_test_evidence(references, scores, access.locked_threshold)
        metrics_sha256 = write_json_durable(run_dir / TEST_METRICS_NAME, evidence)
        predictions_sha256 = write_test_predictions(
            access, run_dir / TEST_PREDICTIONS_NAME, references, scores
        )
        duration = time.monotonic() - started
        audit = {
            "experiment_id": EXPERIMENT_ID,
            "attempt_status": ATTEMPT_COMPLETE,
            "attempt_sequence": ATTEMPT_SEQUENCE,
            "repeat_attempt_permitted": False,
            "experiment_lock_sha256": access.experiment_lock_sha256,
            "test_attempt_sha256": access.receipt_sha256,
            "development_git_sha": lock["git_sha"],
            "evaluator_git_sha": git_provenance(REPOSITORY_ROOT)["git_sha"],
            "evaluator_git_dirty": False,
            "checkpoint_sha256": access.checkpoint_sha256,
            "locked_validation_threshold": access.locked_threshold,
            "threshold_source": "immutable_development_experiment_lock",
            "split_sha256": lock["split_sha256"],
            "dataset": DATASET,
            "dataset_version": DATASET_VERSION,
            "input_contract": input_contract(),
            "waveform_retrieval": "record-aware direct canonical source reads",
            "external_test_waveform_cache": None,
            "test_primary_counts": primary_counts,
            "test_challenge_counts": {
                family: int(
                    np.sum(_arrays(references)["target_family"] == family)
                )
                for family in CHALLENGE_FAMILIES
            },
            "scored_row_count": int(scores.size),
            "environment": runtime_environment(device, workers),
            "execution": read_json(access.receipt_path)["execution"],
            "predictions_sha256": predictions_sha256,
            "metrics_sha256": metrics_sha256,
            "model_state_sha256_before_inference": model_sha_before,
            "model_state_sha256_after_inference": model_sha_after,
            "model_weights_unchanged": True,
            "optimizer_constructed": False,
            "backward_invoked": False,
            "threshold_selection_performed": False,
            "duration_seconds": duration,
        }
        audit["test_audit_sha256"] = canonical_sha256(audit)
        audit_sha256 = write_json_durable(run_dir / TEST_AUDIT_NAME, audit)
        _update_attempt(
            access,
            attempt_status=ATTEMPT_COMPLETE,
            test_data_access_began=True,
            test_audit_sha256=audit_sha256,
            test_metrics_sha256=metrics_sha256,
            test_predictions_sha256=predictions_sha256,
            completed_at_utc_audit_only=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        )
        return {
            "attempt_status": ATTEMPT_COMPLETE,
            "experiment_id": EXPERIMENT_ID,
            "run_dir": str(run_dir),
            "threshold": access.locked_threshold,
            "test_evidence": evidence,
            "test_audit_sha256": audit_sha256,
            "repeat_attempt_permitted": False,
        }
    except BaseException as error:
        try:
            _update_attempt(
                access,
                attempt_status=ATTEMPT_FAILED,
                test_data_access_began=test_access_began,
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(limit=20),
                human_review_required=True,
                repeat_attempt_permitted=False,
            )
        except OSError:  # pragma: no cover - receipt already proves the attempt
            pass
        raise
