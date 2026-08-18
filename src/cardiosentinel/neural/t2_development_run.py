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

`--execute-canonical-outer-validation` exists so the route can be reviewed, and
it refuses: the activation state is `False`, and the refusal fires before any
VALIDATION path, array or label is touched.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Final

from cardiosentinel.data.provenance import git_provenance
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT
from cardiosentinel.neural.t2_persistence import (
    T2_EXPERIMENT_IDENTITY,
    T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED,
    T2_RUN_ROOT,
    T2_TRAINING_ATTEMPT_ID,
    T2ActivationError,
    T2PersistenceError,
    require_outer_validation_authorized,
    require_unclaimed_t2_attempt,
    validate_t2_execution_spec,
)
from cardiosentinel.neural.t2_protocol import (
    T2_ARMS,
    validate_t2_protocol_document,
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
    }


def execute_canonical_training(expected_git_sha: str | None) -> dict[str, Any]:
    """The one TRAIN-only canonical route.

    The training body is intentionally not wired to real data in this change
    set: implementing the science and executing it are separate authorizations.
    Preflight runs, proves the claim is unconsumed, and stops before claiming.
    """
    checks = preflight(expected_git_sha)
    raise T2RunError(
        "T2 canonical training is implemented but not authorized to execute in "
        "this change set. Preflight passed at git "
        f"{checks['git_sha']}, protocol {checks['t2_protocol_sha256'][:12]}..., "
        f"execution spec {checks['t2_execution_spec_sha256'][:12]}...; the "
        "attempt directory was NOT claimed and no timeline row was opened. A "
        "separate human authorization executes the one canonical training run."
    )


def execute_canonical_outer_validation(_expected_git_sha: str | None) -> dict[str, Any]:
    """Refuses before any VALIDATION path, array or label is touched."""
    require_outer_validation_authorized()
    raise T2RunError(  # pragma: no cover - unreachable while unauthorized
        "Outer VALIDATION was authorized but no execution body exists yet."
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
        execute_canonical_training(args.expected_git_sha)
        return 0
    except T2ActivationError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    except (T2RunError, T2PersistenceError) as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
