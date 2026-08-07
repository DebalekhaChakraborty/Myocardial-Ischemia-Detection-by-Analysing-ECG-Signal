"""Explicit command-line stages for external baseline experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from cardiosentinel.baseline.cache import require_external_path
from cardiosentinel.models.baselines import BASELINE_NAMES

PINNED_DOWNLOAD_URL = "https://physionet.org/files/ltstdb/1.0.0/"


def add_baseline_parser(subparsers: argparse._SubParsersAction) -> None:
    baseline = subparsers.add_parser(
        "baseline", help="Materialize and execute frozen classical baselines."
    )
    commands = baseline.add_subparsers(dest="baseline_command", required=True)
    acquire = commands.add_parser(
        "acquire", help="Plan or explicitly execute pinned LTSTDB waveform acquisition."
    )
    acquire.add_argument("--destination", required=True, type=Path)
    acquire.add_argument(
        "--execute",
        action="store_true",
        help="Run resumable wget after printing destination and available disk.",
    )
    materialize = commands.add_parser(
        "materialize", help="Build resumable per-record combined_v1 feature caches."
    )
    materialize.add_argument("--source", required=True, type=Path)
    materialize.add_argument("--feature-root", required=True, type=Path)
    materialize.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
    materialize.add_argument("--records", help="Optional comma-separated record IDs.")
    materialize.add_argument("--chunk-seconds", type=float, default=300.0)
    materialize.add_argument("--force", action="store_true")
    smoke = commands.add_parser(
        "smoke-remote",
        help="Validate bounded waveform features on one record per partition.",
    )
    smoke.add_argument("--output-root", required=True, type=Path)
    smoke.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
    smoke.add_argument("--duration-seconds", type=float, default=60.0)
    smoke.add_argument("--force", action="store_true")
    fit = commands.add_parser(
        "fit", help="Fit train, score validation, and freeze an experiment lock."
    )
    fit.add_argument("--feature-root", required=True, type=Path)
    fit.add_argument("--run-root", required=True, type=Path)
    fit.add_argument("--experiment-id", required=True)
    fit.add_argument("--baseline", required=True, choices=BASELINE_NAMES)
    fit.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
    evaluate = commands.add_parser(
        "evaluate-test", help="Evaluate a validated frozen lock on sealed test rows."
    )
    evaluate.add_argument("--feature-root", required=True, type=Path)
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )


def _acquire(destination: Path, execute: bool) -> int:
    destination = require_external_path(destination, "Waveform destination")
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    command = [
        "wget",
        "--recursive",
        "--no-parent",
        "--continue",
        "--no-host-directories",
        "--cut-dirs=3",
        f"--directory-prefix={destination}",
        "--reject=index.html*",
        PINNED_DOWNLOAD_URL,
    ]
    print("Dataset: Long-Term ST Database v1.0.0, including waveform files")
    print(f"Destination: {destination}")
    print(f"Available disk bytes: {free_bytes}")
    print(f"Resumable command: {' '.join(command)}")
    if not execute:
        print("Status: plan only; add --execute to start acquisition")
        return 0
    print("Status: starting explicit acquisition", flush=True)
    subprocess.run(command, check=True)
    return 0


def run_baseline_command(args: argparse.Namespace) -> int:
    if args.baseline_command == "acquire":
        return _acquire(args.destination, args.execute)
    if args.baseline_command == "materialize":
        from cardiosentinel.baseline.materialize import materialize_features

        records = None if args.records is None else tuple(args.records.split(","))
        manifest = materialize_features(
            args.source,
            args.feature_root,
            args.split,
            records=records,
            chunk_seconds=args.chunk_seconds,
            force=args.force,
            command="cardiosentinel baseline materialize",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.baseline_command == "smoke-remote":
        from cardiosentinel.baseline.smoke import run_remote_smoke

        report = run_remote_smoke(
            args.output_root,
            args.split,
            duration_seconds=args.duration_seconds,
            force=args.force,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.baseline_command == "fit":
        from cardiosentinel.baseline.workflow import fit_and_lock

        summary = fit_and_lock(
            args.feature_root,
            args.run_root,
            args.experiment_id,
            args.baseline,
            split_path=args.split,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    from cardiosentinel.baseline.workflow import evaluate_test

    summary = evaluate_test(args.feature_root, args.run_dir, split_path=args.split)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
