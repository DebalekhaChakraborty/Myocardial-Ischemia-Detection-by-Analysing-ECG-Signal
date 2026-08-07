"""Explicit command-line stages for external baseline experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from cardiosentinel.baseline.cache import require_nonversioned_path
from cardiosentinel.baseline.source import (
    acquisition_urls,
    verify_pinned_source,
)
from cardiosentinel.evaluation.protocol import LTSTDB_V1_SPLIT_SHA256
from cardiosentinel.evaluation.splits import (
    load_split_manifest,
    validate_split_manifest,
)
from cardiosentinel.models.baselines import BASELINE_NAMES


def _expected_record_ids(split: dict[str, Any]) -> set[str]:
    return {
        record
        for records in split["records_by_subject"].values()
        for record in records
    }


def _load_frozen_split(split_path: Path) -> tuple[dict[str, Any], set[str]]:
    split = load_split_manifest(split_path)
    validate_split_manifest(
        split,
        expected_hash=LTSTDB_V1_SPLIT_SHA256,
        expected_subject_count=80,
        expected_record_count=86,
    )
    return split, _expected_record_ids(split)


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
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
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
    materialize.add_argument("--workers", type=int, default=1)
    materialize.add_argument("--force", action="store_true")
    verify = commands.add_parser(
        "verify-source",
        help=(
            "Verify required LTSTDB files against PhysioNet's pinned SHA-256 "
            "manifest."
        ),
    )
    verify.add_argument("--source", required=True, type=Path)
    verify.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
    preflight = commands.add_parser(
        "preflight", help="Report full-run data, storage, memory, and time readiness."
    )
    preflight.add_argument("--source", required=True, type=Path)
    preflight.add_argument("--feature-root", required=True, type=Path)
    preflight.add_argument("--workers", type=int, default=1)
    preflight.add_argument(
        "--split",
        type=Path,
        default=Path("protocols/splits/ltstdb_v1.json"),
    )
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


def _acquire(destination: Path, split_path: Path, execute: bool) -> int:
    destination = require_nonversioned_path(destination, "Waveform destination")
    _, record_ids = _load_frozen_split(split_path)
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    urls = acquisition_urls(record_ids)
    command = [
        "wget",
        "--continue",
        f"--directory-prefix={destination}",
        *urls,
    ]
    print("Dataset: Long-Term ST Database v1.0.0, required Benchmark V1 files only")
    print(f"Destination: {destination}")
    print(f"Available disk bytes: {free_bytes}")
    print(f"Pinned files to acquire: {len(urls)}")
    print("Acquisition is resumable with wget --continue.")
    if not execute:
        print("Status: plan only; add --execute to start acquisition")
        return 0
    print("Status: starting explicit acquisition", flush=True)
    subprocess.run(command, check=True)
    return 0


def run_baseline_command(args: argparse.Namespace) -> int:
    if args.baseline_command == "acquire":
        return _acquire(args.destination, args.split, args.execute)
    if args.baseline_command == "verify-source":
        _, record_ids = _load_frozen_split(args.split)
        receipt = verify_pinned_source(args.source, record_ids)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.baseline_command == "materialize":
        from cardiosentinel.baseline.materialize import materialize_features

        records = None if args.records is None else tuple(args.records.split(","))
        manifest = materialize_features(
            args.source,
            args.feature_root,
            args.split,
            records=records,
            chunk_seconds=args.chunk_seconds,
            workers=args.workers,
            force=args.force,
            command="cardiosentinel baseline materialize",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.baseline_command == "preflight":
        from cardiosentinel.baseline.preflight import baseline_preflight

        report = baseline_preflight(
            args.source, args.feature_root, args.split, workers=args.workers
        )
        print(json.dumps(report, indent=2, sort_keys=True))
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
