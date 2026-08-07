"""CLI registration for B4 development-only engineering commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path("cardiosentinel-data/ltstdb/1.0.0")
DEFAULT_FEATURE_ROOT = Path("cardiosentinel-features/ltstdb-baseline-v1")


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)


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
    else:
        report = benchmark_io(
            args.source,
            args.feature_root,
            train_windows=args.train_windows,
            validation_windows=args.validation_windows,
            batch_size=args.batch_size,
            cache_windows=args.cache_windows,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
