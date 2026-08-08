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
    compute_feature_corpus_sha256,
    read_cache_metadata,
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
from cardiosentinel.baseline.source import (
    OFFICIAL_MANIFEST_NAME,
    OFFICIAL_MANIFEST_SHA256,
    parse_checksum_manifest,
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
from cardiosentinel.neural.integrity import (
    SOURCE_SUFFIXES,
    _expected_embedded_metadata,
    _validate_manifest_identity,
    canonical_sha256,
    source_record_sha256,
)
from cardiosentinel.neural.metadata import _manifest_identity, _metadata_arrays
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.protocol import (
    DATASET,
    DATASET_VERSION,
    FEATURE_CORPUS_SHA256,
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
    """Capability proving the durable one-shot attempt claim already exists.

    Only `open_sealed_test_attempt` constructs this. Every function that can
    resolve, open, or read sealed-test data demands one.

    `initial_attempt_receipt_sha256` hashes the exact STARTED bytes written when
    attempt #1 was exclusively claimed. The receipt is amended later, so this
    digest deliberately does not describe the final receipt and must never be
    read as doing so.
    """

    run_dir: Path
    receipt_path: Path
    initial_attempt_receipt_sha256: str
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


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _describe_existing_attempt(path: Path) -> str:
    """Summarize a prior claim without letting corruption mask the refusal."""
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"status={existing.get('attempt_status')}, "
            f"sequence={existing.get('attempt_sequence')}"
        )
    except (OSError, ValueError):
        # An unreadable or truncated claim still consumed the one attempt.
        return "status=unreadable_or_corrupt"


def claim_attempt_exclusively(path: Path, payload: dict[str, Any]) -> str:
    """Create the one-shot attempt claim with an atomic O_EXCL creation.

    `os.open(O_CREAT | O_EXCL)` is atomic on POSIX, so exactly one process can
    ever create this path. A check-then-write sequence could not provide that
    guarantee: two processes could both observe an absent file and both write.

    A partially written or later-corrupted claim is never removed or reused. The
    attempt is consumed the instant the path exists, which is the conservative
    reading of the one-shot contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise SealedTestAttemptError(
            "A B4 sealed-test attempt already exists "
            f"({_describe_existing_attempt(path)}). There is exactly one "
            "predeclared attempt; it cannot be repeated, reset, or overridden, "
            "and any further evaluation requires documented human review."
        ) from error
    # From here the claim exists on disk. It is never unlinked on failure.
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)
    return sha256_file(path)


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
    _fsync_directory(path.parent)
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
    # The claim is created atomically and durably here. Nothing above this line
    # has resolved or opened a single sealed-test artifact, and only a process
    # that wins the exclusive creation ever receives a capability token.
    initial_receipt_sha256 = claim_attempt_exclusively(receipt_path, receipt)
    access = SealedTestAccess(
        run_dir=run_dir,
        receipt_path=receipt_path,
        initial_attempt_receipt_sha256=initial_receipt_sha256,
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


def _safe_cache_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Sealed-test cache path escapes its root.") from error
    return path


def _sealed_test_entries(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    entries = tuple(
        sorted(
            (
                entry
                for entry in manifest.get("records", ())
                if entry.get("partition") == SEALED_TEST_PARTITION
            ),
            key=lambda entry: str(entry.get("record_id")),
        )
    )
    record_ids = [entry.get("record_id") for entry in entries]
    if not entries or len(set(record_ids)) != len(record_ids):
        raise ValueError("Sealed-test feature records are absent or duplicated.")
    if any(entry.get("status") != "complete" for entry in entries):
        raise ValueError("The sealed-test feature corpus has an incomplete record.")
    return entries


def validate_sealed_test_feature_integrity(
    access: SealedTestAccess, feature_root: Path
) -> dict[str, Any]:
    """Rehash the current sealed-test caches and verify their embedded metadata.

    This is deliberately impossible before the durable attempt claim: it demands
    the capability token, so no sealed-test byte is hashed until attempt #1 has
    been consumed.
    """
    _require_access(access)
    root = require_nonversioned_path(feature_root, "B4 sealed-test feature root")
    manifest = read_json(root / FEATURE_MANIFEST_NAME)
    # Reuse the frozen development identity check verbatim: it is partition
    # agnostic and already binds dataset, split, geometry, schema and corpus.
    _validate_manifest_identity(manifest, FEATURE_CORPUS_SHA256)
    if compute_feature_corpus_sha256(manifest) != FEATURE_CORPUS_SHA256:
        raise ValueError("Sealed-test canonical feature-corpus SHA-256 differs.")

    verified: list[dict[str, Any]] = []
    for entry in _sealed_test_entries(manifest):
        cache_path = _safe_cache_path(root, str(entry["cache_path"]))
        if not cache_path.is_file():
            raise ValueError(
                f"Sealed-test feature cache is absent: {entry['record_id']}"
            )
        actual = sha256_file(cache_path)
        if actual != entry.get("cache_sha256"):
            raise ValueError(
                f"Sealed-test feature cache SHA-256 mismatch: {entry['record_id']}"
            )
        # Metadata only. The numeric `features` member is never requested.
        metadata = read_cache_metadata(cache_path)
        expected = _expected_embedded_metadata(manifest, entry)
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Sealed-test feature metadata mismatch: {entry['record_id']}"
            )
        verified.append(
            {
                "record_id": entry["record_id"],
                "subject_id": entry["subject_id"],
                "partition": entry["partition"],
                "row_count": entry["row_count"],
                "cache_sha256": actual,
                "source_sha256": entry["source_sha256"],
            }
        )

    scientific = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "split_sha256": manifest["split_sha256"],
        "feature_corpus_sha256": FEATURE_CORPUS_SHA256,
        "partition": SEALED_TEST_PARTITION,
        "records": verified,
    }
    return {
        **scientific,
        "sealed_test_feature_integrity_sha256": canonical_sha256(scientific),
        "verified_test_record_count": len(verified),
        "verified_test_cache_count": len(verified),
        "local_feature_root": str(root),
        "verification_result": "passed",
    }


def validate_sealed_test_source_integrity(
    access: SealedTestAccess,
    source: Path,
    feature_integrity_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Hash the current sealed-test source bytes against the pinned manifest.

    Requires the capability token, so no test waveform byte is read before the
    durable attempt claim exists.
    """
    _require_access(access)
    root = require_nonversioned_path(source, "B4 sealed-test waveform source")
    manifest_path = root / OFFICIAL_MANIFEST_NAME
    if sha256_file(manifest_path) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("Official LTSTDB source manifest digest is not pinned.")
    official = parse_checksum_manifest(manifest_path)

    verified: list[dict[str, Any]] = []
    for entry in feature_integrity_receipt.get("records", ()):
        if entry.get("partition") != SEALED_TEST_PARTITION:
            raise ValueError("Sealed-test source receipt saw a foreign partition.")
        record_id = str(entry["record_id"])
        if Path(record_id).name != record_id:
            raise ValueError("Sealed-test source record ID is unsafe.")
        digests: dict[str, str] = {}
        for suffix in SOURCE_SUFFIXES:
            filename = f"{record_id}.{suffix}"
            if filename not in official:
                raise ValueError(f"Official source entry is absent: {filename}")
            actual = sha256_file(root / filename)
            if actual != official[filename]:
                raise ValueError(f"Sealed-test source SHA-256 mismatch: {filename}")
            digests[filename] = actual
        record_digest = source_record_sha256(record_id, digests)
        if record_digest != entry.get("source_sha256"):
            raise ValueError(
                f"Sealed-test source record digest mismatch: {record_id}"
            )
        verified.append(
            {
                "record_id": record_id,
                "partition": SEALED_TEST_PARTITION,
                "files": digests,
                "source_sha256": record_digest,
            }
        )

    scientific = {
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "official_manifest_sha256": OFFICIAL_MANIFEST_SHA256,
        "partition": SEALED_TEST_PARTITION,
        "records": verified,
    }
    return {
        **scientific,
        "sealed_test_source_integrity_sha256": canonical_sha256(scientific),
        "verified_test_record_count": len(verified),
        "verified_test_source_file_count": len(verified) * len(SOURCE_SUFFIXES),
        "local_source_root": str(root),
        "verification_result": "passed",
    }


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
        cache_path = _safe_cache_path(root, str(entry["cache_path"]))
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
        # Integrity of the sealed-test bytes is proven before any row is read
        # and long before any score exists. Both gates require the capability.
        test_access_began = True
        _update_attempt(access, test_data_access_began=True)
        feature_receipt = validate_sealed_test_feature_integrity(
            access, feature_root
        )
        source_receipt = validate_sealed_test_source_integrity(
            access, source, feature_receipt
        )

        references = load_sealed_test_references(access, feature_root)
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
            "initial_attempt_receipt_sha256": (
                access.initial_attempt_receipt_sha256
            ),
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
            "sealed_test_feature_integrity_sha256": feature_receipt[
                "sealed_test_feature_integrity_sha256"
            ],
            "sealed_test_source_integrity_sha256": source_receipt[
                "sealed_test_source_integrity_sha256"
            ],
            "canonical_feature_corpus_sha256": FEATURE_CORPUS_SHA256,
            "official_source_manifest_sha256": OFFICIAL_MANIFEST_SHA256,
            "verified_test_record_count": feature_receipt[
                "verified_test_record_count"
            ],
            "verified_test_cache_count": feature_receipt["verified_test_cache_count"],
            "verified_test_source_file_count": source_receipt[
                "verified_test_source_file_count"
            ],
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
            sealed_test_feature_integrity_sha256=feature_receipt[
                "sealed_test_feature_integrity_sha256"
            ],
            sealed_test_source_integrity_sha256=source_receipt[
                "sealed_test_source_integrity_sha256"
            ],
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
