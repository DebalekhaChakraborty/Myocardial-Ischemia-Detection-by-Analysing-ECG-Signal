"""Run scaffold for the T1 episode-state harness.

A run directory is a claim. Claiming one records what code, what config, what
environment and what input produced the artifacts inside it, so a reviewer can
tell two runs apart without trusting either one's summary.

What every run captures:

* the Git commit and whether the checkout was dirty,
* the canonical digest of the config body,
* runtime metadata -- interpreter, platform, and the resolved dependency set
  with its digest,
* a content digest of the input window stream, so an identical-looking rerun
  over different inputs is detectable,
* the frozen T1 protocol document digest.

**This scaffold does not produce protocol evidence.** It writes
``protocol_evidence: false`` for every ``harness_verification`` run, and it
refuses outright to claim a run directory in the canonical T1 development
namespace, which requires a separately authorized execution specification that
does not exist. The distinction is enforced here rather than left to a naming
convention.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from cardiosentinel.baseline.cache import write_json_atomic
from cardiosentinel.data.provenance import git_provenance, sha256_file
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.provenance import dependency_environment
from cardiosentinel.neural.t1_config import (
    RUN_CLASS_CANONICAL,
    T1_EXECUTION_SPECIFICATION_AUTHORIZED,
    T1EpisodeConfig,
)
from cardiosentinel.neural.t1_engine import (
    T1RunOutputs,
    resolve_thresholds,
    run_t1_episode_state_machine,
)
from cardiosentinel.neural.t1_execution_spec import (
    T1_DEVELOPMENT_ATTEMPT_ID,
    T1_EXECUTION_SPEC_NAME,
    T1_EXECUTION_SPEC_SHA256,
    T1_EXPERIMENT_IDENTITY,
    T1_RUN_ROOT_RELATIVE,
)
from cardiosentinel.neural.t1_protocol import (
    T1_PROTOCOL_NAME,
    T1_PROTOCOL_SHA256,
    T1_ROUTING_DEFINED,
    T1_SEALED_TEST_STATE,
    T1_TEST_ACCESSED,
    T1Thresholds,
)
from cardiosentinel.neural.t1_stream import T1WindowEvidence

STATUS_STARTED: Final = "STARTED"
STATUS_COMPLETE: Final = "COMPLETE"
STATUS_FAILED: Final = "FAILED_OR_INTERRUPTED"

RUN_STATUS_NAME: Final = "T1_RUN_STATUS.json"
RUN_MANIFEST_NAME: Final = "T1_RUN_MANIFEST.json"
RESULT_NAME: Final = "T1_RUN_RESULT.json"
STATE_TRACE_NAME: Final = "T1_STATE_TRACE.json"
EPISODES_NAME: Final = "T1_EPISODES.json"
TRANSITIONS_NAME: Final = "T1_TRANSITIONS.json"
ALERTS_NAME: Final = "T1_ALERTS.json"
RECOVERY_NAME: Final = "T1_RECOVERY_SPANS.json"
FAILURE_RECEIPT_NAME: Final = "T1_FAILURE_RECEIPT.json"

OUTPUT_NAMES: Final = (
    STATE_TRACE_NAME,
    EPISODES_NAME,
    TRANSITIONS_NAME,
    ALERTS_NAME,
    RECOVERY_NAME,
)

# The canonical namespace, DERIVED from the frozen execution specification
# rather than copied. A local copy of an identity is a copy that can drift out
# of agreement with the document that owns it, and the drift is silent.
CANONICAL_RUN_ROOT: Final = REPOSITORY_ROOT / T1_RUN_ROOT_RELATIVE
CANONICAL_RESERVED_PREFIXES: Final = (
    T1_DEVELOPMENT_ATTEMPT_ID,
    Path(T1_RUN_ROOT_RELATIVE).name,
)

RUN_CLASS_FIELD: Final = "run_class"
MANIFEST_CLASS: Final = "t1_episode_harness_run_manifest"
RESULT_CLASS: Final = "t1_episode_harness_run_result"


class T1ExecutionError(RuntimeError):
    """Raised when a run cannot be claimed, described or completed honestly."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def window_stream_digest(windows: Sequence[T1WindowEvidence]) -> dict[str, Any]:
    """Content digest of the exact input stream, in the exact order supplied.

    Order is part of the identity: the same windows in a different order are a
    different causal stream and must not share a digest.
    """
    payload = [
        {
            "window_id": window.window_id,
            "subject_id": window.subject_id,
            "record_id": window.record_id,
            "channel_index": int(window.channel_index),
            "start_sample": int(window.start_sample),
            "model_score": window.model_score,
            "calibrated_probability": window.calibrated_probability,
            "temporal_evidence": window.temporal_evidence,
            "calibrated_uncertainty": window.calibrated_uncertainty,
            "signal_quality": window.signal_quality,
            "context_flags": list(window.context_flags),
        }
        for window in windows
    ]
    return {
        "input_artifact_sha256": canonical_sha256(payload),
        "input_window_count": len(payload),
        "input_order_is_part_of_identity": True,
    }


def runtime_metadata() -> dict[str, Any]:
    """Interpreter, platform and the resolved dependency set with its digest."""
    environment = dependency_environment()
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "installed_package_count": environment["installed_package_count"],
        "installed_packages_sha256": environment["installed_packages_sha256"],
        "key_dependencies": environment["key_dependencies"],
    }


def t1_run_directory(run_root: Path, attempt_id: str) -> Path:
    return Path(run_root) / str(attempt_id)


def absolute_run_path(path: Path | str) -> Path:
    """Normalise a run path without depending on the process working directory.

    A relative path is resolved against the repository root, never against the
    current directory: the shell working directory in this environment does not
    stay where it was put, and a guard that silently reinterprets its own
    argument guards nothing.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return Path(os.path.normpath(candidate)).resolve()


def canonical_namespace_would_be_materialised(run_root: Path, attempt_id: str) -> bool:
    """Would claiming this run create the canonical run root, at any depth?

    ``claim_run_directory`` calls ``mkdir(parents=True)`` on the run
    directory's parent and then creates the directory itself, so every ancestor
    of the target comes into existence. The canonical root is therefore
    materialised exactly when it is the target or an ancestor of it -- which
    covers all three ways a caller can reach it: naming it as the run root,
    naming a directory inside it, and naming its parent with the canonical
    directory as the attempt id.
    """
    target = absolute_run_path(Path(run_root) / str(attempt_id))
    canonical = absolute_run_path(CANONICAL_RUN_ROOT)
    return target == canonical or canonical in target.parents


def require_non_canonical_attempt(config: T1EpisodeConfig) -> None:
    """Refuse to claim anything that would look like canonical T1 evidence.

    Both halves matter. The attempt id is checked because a canonical *name*
    beside synthetic output is misleading evidence; the run path is checked
    because the canonical run directory is itself claim-bearing, and creating
    it consumes the one canonical attempt whatever the directory is called.
    """
    if (
        config.run_class == RUN_CLASS_CANONICAL
        and not T1_EXECUTION_SPECIFICATION_AUTHORIZED
    ):
        raise T1ExecutionError(
            f"A canonical T1 development run is not available. The frozen "
            f"execution specification {T1_EXECUTION_SPEC_NAME} exists and is "
            "merged, but the canonical development harness it specifies has not "
            "been implemented, and scientific execution is not authorized. Use "
            "the harness verification run class."
        )
    if canonical_namespace_would_be_materialised(config.run_root, config.attempt_id):
        raise T1ExecutionError(
            f"Claiming {config.attempt_id!r} under {config.run_root} would create "
            f"the canonical T1 run root {CANONICAL_RUN_ROOT}. That directory is "
            "itself claim-bearing: a canonical run directory existing in any "
            "state consumes the one canonical attempt, and consuming it with "
            "synthetic output would be unrecoverable. Choose a run root outside "
            "the canonical namespace."
        )
    lowered = str(config.attempt_id).lower()
    for reserved in CANONICAL_RESERVED_PREFIXES:
        if lowered.startswith(reserved.lower()):
            raise T1ExecutionError(
                f"Attempt id {config.attempt_id!r} is reserved for the canonical T1 "
                f"development attempt ({reserved!r}...). A harness verification run "
                "may not occupy that name: canonical evidence must never be "
                "confusable with synthetic output."
            )


def require_unclaimed(run_root: Path, attempt_id: str) -> Path:
    """The directory is the claim, so its absence is what is proved here."""
    run_dir = t1_run_directory(run_root, attempt_id)
    if run_dir.exists():
        raise T1ExecutionError(
            f"Run directory {run_dir} already exists. It is not deleted, reset, "
            "renamed or reseeded, and no alternate name is chosen automatically. "
            "Pick a new attempt_id explicitly, or review the existing run."
        )
    return run_dir


@dataclass(frozen=True, slots=True)
class T1ClaimedRun:
    """A claimed run directory and the identity written into it."""

    run_dir: Path
    attempt_id: str
    started_at: str
    manifest: dict[str, Any]


def build_run_manifest(
    config: T1EpisodeConfig,
    *,
    windows: Sequence[T1WindowEvidence],
    started_at: str,
    thresholds: T1Thresholds | None,
) -> dict[str, Any]:
    """Everything needed to tell this run apart from any other run."""
    git = git_provenance(REPOSITORY_ROOT)
    manifest: dict[str, Any] = {
        "artifact_class": MANIFEST_CLASS,
        "attempt_id": config.attempt_id,
        RUN_CLASS_FIELD: config.run_class,
        "protocol_evidence": config.protocol_evidence,
        "started_at": started_at,
        "protocol": {
            "name": T1_PROTOCOL_NAME,
            "document_sha256": T1_PROTOCOL_SHA256,
        },
        "execution_specification": {
            "name": T1_EXECUTION_SPEC_NAME,
            "document_sha256": T1_EXECUTION_SPEC_SHA256,
            "canonical_experiment_identity": T1_EXPERIMENT_IDENTITY,
            "canonical_attempt_id": T1_DEVELOPMENT_ATTEMPT_ID,
            "canonical_run_root": str(T1_RUN_ROOT_RELATIVE),
            "canonical_namespace_claimed_by_this_run": False,
        },
        "git": {
            "git_sha": git["git_sha"],
            "git_dirty": git["git_dirty"],
        },
        "config": {
            "config_sha256": config.config_sha256,
            "source_path": str(config.source_path) if config.source_path else None,
            "threshold_source": config.threshold_source,
            "q_watch": config.q_watch,
            "q_event": config.q_event,
            "persistence_profile": config.profile.name,
            "detector_threshold": config.detector_threshold,
            "cold_start_seconds": config.cold_start_seconds,
            "refractory_seconds": config.refractory_seconds,
            "refractory_applies_to": "alert_emission_only",
        },
        "input": window_stream_digest(windows),
        "runtime": runtime_metadata(),
        "firewalls": {
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
            "routing_defined": T1_ROUTING_DEFINED,
            "thresholds_optimized": False,
            "tuned_against_test_data": False,
            "performance_claimed": False,
        },
    }
    if config.source_path is not None and Path(config.source_path).is_file():
        manifest["config"]["source_sha256"] = sha256_file(Path(config.source_path))
    if thresholds is not None:
        manifest["thresholds_used"] = thresholds._asdict()
    return manifest


def claim_run_directory(
    config: T1EpisodeConfig,
    *,
    windows: Sequence[T1WindowEvidence],
    thresholds: T1Thresholds | None = None,
) -> T1ClaimedRun:
    """Create the run directory and write its status and manifest."""
    require_non_canonical_attempt(config)
    run_dir = require_unclaimed(config.run_root, config.attempt_id)
    started_at = _now()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise T1ExecutionError(
            f"Run directory {run_dir} was claimed concurrently; nothing is "
            "overwritten and nothing is retried."
        ) from error
    manifest = build_run_manifest(
        config, windows=windows, started_at=started_at, thresholds=thresholds
    )
    write_json_atomic(
        run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": config.attempt_id,
            RUN_CLASS_FIELD: config.run_class,
            "protocol_evidence": config.protocol_evidence,
            "status": STATUS_STARTED,
            "started_at": started_at,
            "updated_at": started_at,
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
        },
    )
    write_json_atomic(run_dir / RUN_MANIFEST_NAME, manifest)
    return T1ClaimedRun(
        run_dir=run_dir,
        attempt_id=config.attempt_id,
        started_at=started_at,
        manifest=manifest,
    )


def write_outputs(claimed: T1ClaimedRun, outputs: T1RunOutputs) -> dict[str, str]:
    """Promote the five outputs and return their digests."""
    payload = outputs.as_json_payload()
    files = {
        STATE_TRACE_NAME: {"state_trace": payload["state_trace"]},
        EPISODES_NAME: {"episodes": payload["episodes"]},
        TRANSITIONS_NAME: {"transitions": payload["transitions"]},
        ALERTS_NAME: {"alerts": payload["alerts"]},
        RECOVERY_NAME: {"recovery_spans": payload["recovery_spans"]},
    }
    digests: dict[str, str] = {}
    for name, body in files.items():
        path = claimed.run_dir / name
        write_json_atomic(path, body)
        digests[name] = sha256_file(path)
    return digests


def complete_run(
    claimed: T1ClaimedRun, outputs: T1RunOutputs, *, thresholds: T1Thresholds
) -> dict[str, Any]:
    """Write the outputs, the result and the terminal status."""
    digests = write_outputs(claimed, outputs)
    finished_at = _now()
    result = {
        "artifact_class": RESULT_CLASS,
        "attempt_id": claimed.attempt_id,
        RUN_CLASS_FIELD: claimed.manifest[RUN_CLASS_FIELD],
        "protocol_evidence": claimed.manifest["protocol_evidence"],
        "started_at": claimed.started_at,
        "finished_at": finished_at,
        "manifest_sha256": canonical_sha256(claimed.manifest),
        "output_sha256": digests,
        "thresholds_used": thresholds._asdict(),
        "summary": outputs.summary(),
        "claims": {
            "performance_claimed": False,
            "thresholds_optimized": False,
            "tuned_against_test_data": False,
            "evidence_class": (
                "protocol_evidence"
                if claimed.manifest["protocol_evidence"]
                else "harness_verification_only_not_scientific_evidence"
            ),
        },
    }
    write_json_atomic(claimed.run_dir / RESULT_NAME, result)
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            RUN_CLASS_FIELD: claimed.manifest[RUN_CLASS_FIELD],
            "protocol_evidence": claimed.manifest["protocol_evidence"],
            "status": STATUS_COMPLETE,
            "started_at": claimed.started_at,
            "updated_at": finished_at,
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
        },
    )
    return result


def write_failure_receipt(claimed: T1ClaimedRun, error: BaseException) -> None:
    """Additive receipt. Nothing is deleted, repaired or retried."""
    write_json_atomic(
        claimed.run_dir / FAILURE_RECEIPT_NAME,
        {
            "attempt_id": claimed.attempt_id,
            "failed_at": _now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "automatic_retry_permitted": False,
        },
    )
    write_json_atomic(
        claimed.run_dir / RUN_STATUS_NAME,
        {
            "attempt_id": claimed.attempt_id,
            RUN_CLASS_FIELD: claimed.manifest[RUN_CLASS_FIELD],
            "protocol_evidence": claimed.manifest["protocol_evidence"],
            "status": STATUS_FAILED,
            "started_at": claimed.started_at,
            "updated_at": _now(),
            "test_accessed": T1_TEST_ACCESSED,
            "sealed_test_state": T1_SEALED_TEST_STATE,
        },
    )


def execute_t1_run(
    windows: Iterable[T1WindowEvidence],
    config: T1EpisodeConfig,
    *,
    thresholds: T1Thresholds | None = None,
) -> dict[str, Any]:
    """Claim a directory, run the state machine, promote the outputs.

    The window stream is materialised once so its digest describes exactly what
    was executed. Causality is preserved inside the engine, which still pulls
    one window at a time and never looks ahead.
    """
    materialised = list(windows)
    active = thresholds if thresholds is not None else resolve_thresholds(config)
    claimed = claim_run_directory(config, windows=materialised, thresholds=active)
    try:
        outputs = run_t1_episode_state_machine(materialised, config, thresholds=active)
    except BaseException as error:
        write_failure_receipt(claimed, error)
        raise
    return complete_run(claimed, outputs, thresholds=active)


def read_run_artifact(run_dir: Path, name: str) -> dict[str, Any]:
    """Read one promoted artifact back, for verification."""
    path = Path(run_dir) / name
    if not path.is_file():
        raise T1ExecutionError(f"Run artifact {name} is missing at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))
