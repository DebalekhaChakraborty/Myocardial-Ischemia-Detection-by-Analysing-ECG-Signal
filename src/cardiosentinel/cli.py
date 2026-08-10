"""Minimal factual command-line interface for CardioSentinel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cardiosentinel import __version__
from cardiosentinel.baseline.cli import add_baseline_parser, run_baseline_command
from cardiosentinel.config import DEFAULT_CONFIG_PATH, load_config
from cardiosentinel.data.manifest import (
    build_manifest,
    download_metadata,
    inspect_dataset,
    validate_local_dataset,
    write_manifest,
)
from cardiosentinel.data.remote import probe_remote, validate_remote_dataset
from cardiosentinel.neural.cli import (
    add_b4_parser,
    add_m1_parser,
    add_p1_parser,
    run_b4_command,
    run_m1_command,
    run_p1_command,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without invoking research workflows."""
    parser = argparse.ArgumentParser(
        prog="cardiosentinel",
        description="CardioSentinel research-software utilities.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a validated configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="Print factual project metadata.")
    data_parser = subparsers.add_parser(
        "data", help="Inspect and validate local WFDB datasets."
    )
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    for command in ("inspect", "validate"):
        command_parser = data_commands.add_parser(command)
        command_parser.add_argument("dataset", choices=("edb", "ltstdb"))
        command_parser.add_argument("--source", required=True, type=Path)
        command_parser.add_argument("--annotation-set")
    download_parser = data_commands.add_parser(
        "download-metadata", help="Download headers and annotations, never waveforms."
    )
    download_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    download_parser.add_argument("--destination", required=True, type=Path)
    download_parser.add_argument("--annotation-set")
    manifest_parser = data_commands.add_parser("manifest")
    manifest_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    manifest_parser.add_argument("--source", required=True, type=Path)
    manifest_parser.add_argument("--output", required=True, type=Path)
    manifest_parser.add_argument("--annotation-set")
    probe_parser = data_commands.add_parser(
        "probe-remote", help="Inspect one remote header and annotation stream."
    )
    probe_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    probe_parser.add_argument("--record", required=True)
    probe_parser.add_argument("--annotation-set")
    remote_validate_parser = data_commands.add_parser(
        "validate-remote",
        help="Validate remote headers and annotations without waveforms.",
    )
    remote_validate_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    remote_validate_parser.add_argument("--annotation-set")
    signal_parser = subparsers.add_parser(
        "signal", help="Inspect physical ECG and audit causal preprocessing."
    )
    signal_commands = signal_parser.add_subparsers(dest="signal_command", required=True)
    waveform_probe = signal_commands.add_parser(
        "probe-remote", help="Read one bounded remote physical waveform interval."
    )
    waveform_probe.add_argument("dataset", choices=("edb", "ltstdb"))
    waveform_probe.add_argument("--record", required=True)
    waveform_probe.add_argument("--start-seconds", type=float, required=True)
    waveform_probe.add_argument("--duration-seconds", type=float, required=True)
    waveform_probe.add_argument(
        "--channels",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help="Optional comma-separated zero-based channel indices.",
    )
    audit_parser = signal_commands.add_parser(
        "filter-audit", help="Print causal filter coefficients and response as JSON."
    )
    audit_parser.add_argument("--sampling-frequency-hz", type=float, required=True)
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Generate and validate frozen benchmark metadata."
    )
    benchmark_commands = benchmark_parser.add_subparsers(
        dest="benchmark_command", required=True
    )
    summarize_parser = benchmark_commands.add_parser(
        "summarize", help="Aggregate deterministic window targets without training."
    )
    summarize_parser.add_argument("--dataset", choices=("ltstdb", "edb"), required=True)
    summarize_parser.add_argument("--annotation-set")
    summarize_parser.add_argument("--source", type=Path)
    summarize_parser.add_argument("--split", type=Path)
    summarize_parser.add_argument("--output", type=Path)
    summarize_parser.add_argument(
        "--edb-cohort",
        choices=("full", "overlap_clean"),
        default="full",
        help="Explicit EDB secondary cohort; ignored for LTSTDB.",
    )
    generate_parser = benchmark_commands.add_parser(
        "generate-split", help="Generate the pre-model LTSTDB V1 subject split."
    )
    generate_parser.add_argument("--source", type=Path)
    generate_parser.add_argument("--output", type=Path, required=True)
    validate_parser = benchmark_commands.add_parser(
        "validate-split", help="Validate split integrity and its canonical hash."
    )
    validate_parser.add_argument("--split", type=Path, required=True)
    info_parser = benchmark_commands.add_parser(
        "split-info", help="Print frozen split counts and identity."
    )
    info_parser.add_argument("--split", type=Path, required=True)
    add_baseline_parser(subparsers)
    add_b4_parser(subparsers)
    add_p1_parser(subparsers)
    add_m1_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded research utility and return its process status."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        config = load_config(args.config)
        print(f"Project: {config.project.name}")
        print(f"Package version: {__version__}")
        print("Status: Research software only; not a medical device")
        print(f"Active configuration profile: {config.project.profile}")
        return 0

    if args.command == "data":
        if args.data_command == "download-metadata":
            downloaded_to = download_metadata(
                args.dataset, args.destination, args.annotation_set
            )
            print(f"Downloaded {args.dataset} metadata to {downloaded_to}")
            return 0
        if args.data_command == "probe-remote":
            print(
                json.dumps(
                    probe_remote(args.dataset, args.record, args.annotation_set),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.data_command == "validate-remote":
            report = validate_remote_dataset(args.dataset, args.annotation_set)
            print(f"Validation: {'passed' if report.is_valid else 'failed'}")
            print(json.dumps(report.summary, sort_keys=True))
            for warning in report.warnings:
                print(f"Warning: {warning}")
            report.raise_for_errors()
            return 0
        if args.data_command == "inspect":
            records, parsed = inspect_dataset(
                args.dataset, args.source, args.annotation_set
            )
            print(f"Records: {len(records)}")
            print(f"Subjects: {len({record.subject_id for record in records})}")
            print(f"Events: {sum(len(item.events) for item in parsed)}")
            return 0
        if args.data_command == "validate":
            report = validate_local_dataset(
                args.dataset, args.source, args.annotation_set
            )
            print(f"Validation: {'passed' if report.is_valid else 'failed'}")
            print(report.summary)
            report.raise_for_errors()
            return 0
        manifest = build_manifest(
            args.dataset,
            args.source,
            "cardiosentinel data manifest",
            args.annotation_set,
        )
        write_manifest(manifest, args.output)
        print(f"Manifest: {args.output}")
        return 0

    if args.command == "signal":
        if args.signal_command == "probe-remote":
            from cardiosentinel.signal.io import (
                read_remote_seconds,
                waveform_summary,
            )

            segment = read_remote_seconds(
                args.dataset,
                args.record,
                args.start_seconds,
                args.duration_seconds,
                args.channels,
            )
            print(json.dumps(waveform_summary(segment), indent=2, sort_keys=True))
            return 0
        from cardiosentinel.signal.preprocessing import filter_audit

        config = load_config(args.config)
        print(
            json.dumps(
                filter_audit(config.preprocessing, args.sampling_frequency_hz),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "benchmark":
        from cardiosentinel.data.provenance import git_provenance
        from cardiosentinel.evaluation.benchmark import (
            REPOSITORY_ROOT,
            load_benchmark_metadata,
            summarize_from_sources,
        )
        from cardiosentinel.evaluation.protocol import (
            DEFAULT_SEED,
            LTSTDB_V1_GENERATOR_CODE_SHA256,
            LTSTDB_V1_SPLIT_SHA256,
        )
        from cardiosentinel.evaluation.splits import (
            generate_split_manifest,
            load_split_manifest,
            validate_split_manifest,
            write_json,
        )

        if args.benchmark_command == "generate-split":
            records, parsed = load_benchmark_metadata("ltstdb", "stb", args.source)
            generation_provenance = git_provenance(REPOSITORY_ROOT)
            manifest = generate_split_manifest(
                records,
                parsed,
                str(generation_provenance["git_sha"]),
                bool(generation_provenance["git_dirty"]),
                seed=DEFAULT_SEED,
            )
            write_json(args.output, manifest)
            print(f"Split: {args.output}")
            print(f"SHA-256: {manifest['split_sha256']}")
            return 0
        if args.benchmark_command in {"validate-split", "split-info"}:
            manifest = load_split_manifest(args.split)
            validate_split_manifest(
                manifest,
                expected_hash=LTSTDB_V1_SPLIT_SHA256,
                expected_generator_code_hash=LTSTDB_V1_GENERATOR_CODE_SHA256,
                expected_subject_count=80,
                expected_record_count=86,
            )
            if args.benchmark_command == "validate-split":
                print("Validation: passed")
                print(f"SHA-256: {manifest['split_sha256']}")
            else:
                payload = {
                    "dataset": manifest["dataset"],
                    "annotation_definition": manifest["annotation_definition"],
                    "sealed_test_partition": manifest["sealed_test_partition"],
                    "split_sha256": manifest["split_sha256"],
                    "partition_summaries": manifest["partition_summaries"],
                }
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        split_path = args.split
        if args.dataset == "ltstdb" and split_path is None:
            from cardiosentinel.evaluation.benchmark import DEFAULT_SPLIT_PATH

            split_path = DEFAULT_SPLIT_PATH
        command = "cardiosentinel benchmark summarize"
        summary = summarize_from_sources(
            args.dataset,
            args.annotation_set,
            split_path,
            args.source,
            command,
            args.edb_cohort,
        )
        if args.output:
            write_json(args.output, summary)
            print(f"Summary: {args.output}")
        else:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "baseline":
        return run_baseline_command(args)

    if args.command == "b4":
        return run_b4_command(args)
    if args.command == "p1":
        return run_p1_command(args)
    if args.command == "m1":
        return run_m1_command(args)

    parser.print_help()
    return 0
