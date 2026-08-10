"""CLI registration for B4 development-only engineering commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path("cardiosentinel-data/ltstdb/1.0.0")
DEFAULT_FEATURE_ROOT = Path("cardiosentinel-features/ltstdb-baseline-v1")
DEFAULT_WAVEFORM_CACHE_ROOT = Path("cardiosentinel-features/b4-waveform-v1")
DEFAULT_RUN_ROOT = Path("cardiosentinel-runs/phase3b2-b4-v1")
DEFAULT_CANDIDATE_RUN_ROOT = Path("cardiosentinel-runs/phase3b2-architecture-v1")
DEFAULT_B4A_RUN_DIR = DEFAULT_RUN_ROOT / "B4_raw_compact_cnn_v1"
DEFAULT_B4B_RUN_DIR = DEFAULT_CANDIDATE_RUN_ROOT / "B4B_cnn_transformer_v1"
DEFAULT_B4C_RUN_DIR = DEFAULT_CANDIDATE_RUN_ROOT / "B4C_cnn_ssm_v1"


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)


def _cache_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_WAVEFORM_CACHE_ROOT
    )


def _run_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)


def add_b4_parser(subparsers: argparse._SubParsersAction) -> None:
    b4 = subparsers.add_parser(
        "b4", help="Run train/validation-only B4 engineering checks."
    )
    commands = b4.add_subparsers(dest="b4_command", required=True)
    preflight = commands.add_parser(
        "preflight", help="Validate frozen B4 development readiness."
    )
    _common_paths(preflight)
    preflight.add_argument("--workers", type=int, default=0)
    preflight.add_argument("--device", choices=("cpu", "cuda"))
    preflight.add_argument("--allow-dirty", action="store_true")

    smoke = commands.add_parser(
        "smoke", help="Run a tiny non-scientific train/validation smoke check."
    )
    _common_paths(smoke)
    smoke.add_argument("--train-windows", type=int, default=2)
    smoke.add_argument("--validation-windows", type=int, default=2)
    smoke.add_argument("--device", choices=("cpu", "cuda"))

    benchmark = commands.add_parser(
        "benchmark-io", help="Measure deterministic train/validation waveform I/O."
    )
    _common_paths(benchmark)
    benchmark.add_argument("--train-windows", type=int, default=32)
    benchmark.add_argument("--validation-windows", type=int, default=32)
    benchmark.add_argument("--batch-size", type=int, default=256)
    benchmark.add_argument("--cache-windows", type=int, default=0)

    verify = commands.add_parser(
        "verify-development",
        help="Hash current train/validation feature and waveform sources.",
    )
    _common_paths(verify)

    materialize = commands.add_parser(
        "cache-materialize",
        help="Build the resumable lossless train/validation waveform cache.",
    )
    _common_paths(materialize)
    _cache_root(materialize)
    materialize.add_argument("--allow-dirty", action="store_true")

    audit = commands.add_parser(
        "cache-audit",
        help="Validate cache hashes, alignment, and exact source equivalence.",
    )
    _common_paths(audit)
    _cache_root(audit)

    cache_benchmark = commands.add_parser(
        "benchmark-cache",
        help="Compare representative direct and mmap development I/O.",
    )
    _common_paths(cache_benchmark)
    _cache_root(cache_benchmark)
    cache_benchmark.add_argument("--train-windows", type=int, default=1024)
    cache_benchmark.add_argument("--validation-windows", type=int, default=1024)

    compute = commands.add_parser(
        "benchmark-compute",
        help="Time disposable B4 forward/backward development compute.",
    )
    compute.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    _cache_root(compute)
    compute.add_argument("--batches", type=int, default=32)
    compute.add_argument("--device", choices=("cpu", "cuda"))

    run_preflight = commands.add_parser(
        "run-preflight",
        help=(
            "Validate canonical B4 train/validation run readiness. "
            "Does not access the test partition."
        ),
    )
    _common_paths(run_preflight)
    _cache_root(run_preflight)
    _run_root(run_preflight)
    run_preflight.add_argument("--device", choices=("cpu", "cuda"))
    run_preflight.add_argument("--workers", type=int, default=0)
    run_preflight.add_argument("--allow-dirty", action="store_true")

    run_train = commands.add_parser(
        "run-train-validation",
        help=(
            "Runs the single canonical B4 train/validation experiment. "
            "Does not access the test partition."
        ),
    )
    _common_paths(run_train)
    _cache_root(run_train)
    _run_root(run_train)
    run_train.add_argument("--device", choices=("cpu", "cuda"))
    run_train.add_argument("--workers", type=int, default=0)
    run_train.add_argument(
        "--no-validation-predictions",
        action="store_true",
        help="Skip the development validation prediction artifact.",
    )

    # Exactly one sealed-test command. It deliberately exposes no --threshold,
    # --checkpoint, --force, --retry, --overwrite or --seed option: the
    # checkpoint and threshold always come from the immutable development lock.
    sealed_test = commands.add_parser(
        "evaluate-locked-test",
        help=(
            "Performs the single predeclared B4 test evaluation from the "
            "immutable development lock. Writes the attempt receipt before "
            "test access and refuses repeat attempts."
        ),
    )
    sealed_test.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    sealed_test.add_argument(
        "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
    )
    _run_root(sealed_test)
    sealed_test.add_argument("--device", choices=("cpu", "cuda"))
    sealed_test.add_argument("--workers", type=int, default=0)

    # Official A/B/C validation challenge evidence. There is deliberately no
    # --candidate flag: the suite is all three or nothing, and no threshold,
    # metric or retry override exists.
    challenge = commands.add_parser(
        "validation-challenge",
        help=(
            "Runs the one official B4-A/B4-B/B4-C validation challenge evidence "
            "suite: locked-model inference over the frozen validation challenge "
            "rows. Trains nothing and does not access the test partition."
        ),
    )
    challenge.add_argument(
        "--run-root", type=Path, default=DEFAULT_CANDIDATE_RUN_ROOT
    )
    challenge.add_argument("--b4a-run", type=Path, default=DEFAULT_B4A_RUN_DIR)
    challenge.add_argument("--b4b-run", type=Path, default=DEFAULT_B4B_RUN_DIR)
    challenge.add_argument("--b4c-run", type=Path, default=DEFAULT_B4C_RUN_DIR)
    challenge.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    challenge.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)

    # Candidate architecture runners. Only b4b and b4c are selectable, and no
    # model, optimizer, threshold, seed or epoch override is exposed.
    candidate = commands.add_parser(
        "candidate",
        help="Run the canonical B4-B/B4-C development experiments.",
    )
    candidate_commands = candidate.add_subparsers(
        dest="candidate_command", required=True
    )
    for name, description in (
        (
            "run-preflight",
            "Validate canonical B4-B/B4-C run readiness. "
            "Does not access the test partition.",
        ),
        (
            "run-train-validation",
            "Runs one canonical B4-B/B4-C development train/validation "
            "experiment. Does not access the test partition.",
        ),
    ):
        parser = candidate_commands.add_parser(name, help=description)
        parser.add_argument("--candidate", choices=("b4b", "b4c"), required=True)
        parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        parser.add_argument(
            "--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT
        )
        _cache_root(parser)
        parser.add_argument(
            "--run-root", type=Path, default=DEFAULT_CANDIDATE_RUN_ROOT
        )
        parser.add_argument("--workers", type=int, default=0)


def run_b4_command(args: argparse.Namespace) -> int:
    from cardiosentinel.neural.engineering import (
        b4_preflight,
        b4_smoke,
        benchmark_io,
    )

    if args.b4_command == "preflight":
        report = b4_preflight(
            args.source,
            args.feature_root,
            workers=args.workers,
            require_clean=not args.allow_dirty,
            requested_device=args.device,
        )
    elif args.b4_command == "smoke":
        report = b4_smoke(
            args.source,
            args.feature_root,
            train_windows=args.train_windows,
            validation_windows=args.validation_windows,
            requested_device=args.device,
        )
    elif args.b4_command == "benchmark-io":
        report = benchmark_io(
            args.source,
            args.feature_root,
            train_windows=args.train_windows,
            validation_windows=args.validation_windows,
            batch_size=args.batch_size,
            cache_windows=args.cache_windows,
        )
    elif args.b4_command == "verify-development":
        from cardiosentinel.neural.integrity import (
            validate_development_feature_integrity,
            validate_development_source_integrity,
        )

        feature_receipt = validate_development_feature_integrity(args.feature_root)
        source_receipt = validate_development_source_integrity(
            args.source, feature_receipt
        )
        report = {
            "development_feature_integrity": feature_receipt,
            "development_source_integrity": source_receipt,
        }
    elif args.b4_command == "cache-materialize":
        from cardiosentinel.neural.waveform_cache import (
            materialize_development_waveform_cache,
        )

        report = materialize_development_waveform_cache(
            args.source,
            args.feature_root,
            args.cache_root,
            require_clean=not args.allow_dirty,
        )
    elif args.b4_command == "cache-audit":
        from cardiosentinel.neural.waveform_cache import (
            audit_waveform_cache_equivalence,
            build_development_indexes,
            validate_waveform_cache,
        )

        indexes = build_development_indexes(args.feature_root)
        cache = validate_waveform_cache(args.cache_root, indexes)
        report = {
            "waveform_cache_sha256": cache.manifest["waveform_cache_sha256"],
            "equivalence_audit": audit_waveform_cache_equivalence(
                args.source, args.cache_root, indexes
            ),
        }
    elif args.b4_command == "run-preflight":
        from cardiosentinel.neural.experiment import b4_scientific_preflight

        report = b4_scientific_preflight(
            args.source,
            args.feature_root,
            args.cache_root,
            args.run_root,
            requested_device=args.device,
            require_clean=not args.allow_dirty,
            workers=args.workers,
        )
    elif args.b4_command == "run-train-validation":
        from cardiosentinel.neural.experiment import (
            DEFAULT_COMMAND,
            run_b4_train_validation,
        )

        # The canonical scientific run has no --allow-dirty option by design:
        # a clean checkout is mandatory and cannot be relaxed from the CLI.
        report = run_b4_train_validation(
            args.source,
            args.feature_root,
            args.cache_root,
            args.run_root,
            command=DEFAULT_COMMAND,
            requested_device=args.device,
            workers=args.workers,
            save_validation_predictions=not args.no_validation_predictions,
        )
    elif args.b4_command == "candidate":
        from cardiosentinel.neural.candidate_experiment import (
            DEFAULT_COMMAND as CANDIDATE_COMMAND,
        )
        from cardiosentinel.neural.candidate_experiment import (
            PREFLIGHT_COMMAND as CANDIDATE_PREFLIGHT_COMMAND,
        )
        from cardiosentinel.neural.candidate_experiment import (
            candidate_scientific_preflight,
            run_candidate_train_validation,
        )

        if args.candidate_command == "run-preflight":
            report = candidate_scientific_preflight(
                args.candidate,
                args.source,
                args.feature_root,
                args.cache_root,
                args.run_root,
                workers=args.workers,
            )
            report["command"] = CANDIDATE_PREFLIGHT_COMMAND
        else:
            report = run_candidate_train_validation(
                args.candidate,
                args.source,
                args.feature_root,
                args.cache_root,
                args.run_root,
                command=CANDIDATE_COMMAND,
                workers=args.workers,
            )
    elif args.b4_command == "validation-challenge":
        from cardiosentinel.neural.validation_challenge import (
            run_official_validation_challenge_suite,
        )

        report = run_official_validation_challenge_suite(
            {
                "B4-A": args.b4a_run,
                "B4-B": args.b4b_run,
                "B4-C": args.b4c_run,
            },
            args.run_root,
            args.feature_root,
            args.source,
            command="cardiosentinel b4 validation-challenge",
        )
    elif args.b4_command == "evaluate-locked-test":
        from cardiosentinel.neural.sealed_test import (
            DEFAULT_COMMAND as SEALED_TEST_COMMAND,
        )
        from cardiosentinel.neural.sealed_test import evaluate_locked_test

        report = evaluate_locked_test(
            args.source,
            args.feature_root,
            args.run_root,
            command=SEALED_TEST_COMMAND,
            requested_device=args.device,
            workers=args.workers,
        )
    elif args.b4_command == "benchmark-cache":
        from cardiosentinel.neural.engineering import (
            benchmark_direct_and_cached_io,
        )

        report = benchmark_direct_and_cached_io(
            args.source,
            args.feature_root,
            args.cache_root,
            train_windows=args.train_windows,
            validation_windows=args.validation_windows,
        )
    else:
        from cardiosentinel.neural.engineering import benchmark_compute_only

        report = benchmark_compute_only(
            args.feature_root,
            args.cache_root,
            batches=args.batches,
            requested_device=args.device,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


DEFAULT_P1_RUN_ROOT = Path("cardiosentinel-runs/phase4-p1-physiology-v1")
DEFAULT_P1_CACHE_ROOT = Path("cardiosentinel-features/p1-b4b-embeddings-v1")


def add_p1_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the P1 development commands.

    Only two official routes exist: a read-only preflight, and the Stage P1-1
    suite that runs BOTH arms. There is deliberately no single-arm route and no
    force/retry/overwrite option.
    """
    p1 = subparsers.add_parser(
        "p1", help="Phase 4 P1 physiology-fusion development commands."
    )
    commands = p1.add_subparsers(dest="p1_command", required=True)
    for name, description in (
        (
            "preflight",
            "Read-only P1 Stage-1 readiness report. Creates no artifact and "
            "does not access the test partition.",
        ),
        (
            "run-stage1",
            "Runs the one canonical P1-A vs P1-B Stage-1 ablation. Both arms "
            "are mandatory. Does not access the test partition.",
        ),
    ):
        parser = commands.add_parser(name, help=description)
        parser.add_argument("--run-root", type=Path, default=DEFAULT_P1_RUN_ROOT)
        parser.add_argument("--cache-root", type=Path, default=DEFAULT_P1_CACHE_ROOT)
        parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)


def run_p1_command(args: argparse.Namespace) -> int:
    from cardiosentinel.neural.p1_experiment import p1_preflight

    if args.p1_command == "preflight":
        report = p1_preflight(args.run_root, args.cache_root)
    else:
        raise SystemExit(
            "cardiosentinel p1 run-stage1 requires the reviewed canonical "
            "embedding caches and physiology transform; it is invoked from the "
            "authorized scientific runner, not ad hoc."
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
