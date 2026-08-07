"""CLI registration for B4 development-only engineering commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path("cardiosentinel-data/ltstdb/1.0.0")
DEFAULT_FEATURE_ROOT = Path("cardiosentinel-features/ltstdb-baseline-v1")
DEFAULT_WAVEFORM_CACHE_ROOT = Path("cardiosentinel-features/b4-waveform-v1")


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)


def _cache_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_WAVEFORM_CACHE_ROOT
    )


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
