"""Minimal factual command-line interface for CardioSentinel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cardiosentinel import __version__
from cardiosentinel.config import DEFAULT_CONFIG_PATH, load_config
from cardiosentinel.data.manifest import (
    build_manifest,
    download_dataset,
    inspect_dataset,
    validate_local_dataset,
    write_manifest,
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
    download_parser = data_commands.add_parser("download")
    download_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    download_parser.add_argument("--destination", required=True, type=Path)
    manifest_parser = data_commands.add_parser("manifest")
    manifest_parser.add_argument("dataset", choices=("edb", "ltstdb"))
    manifest_parser.add_argument("--source", required=True, type=Path)
    manifest_parser.add_argument("--output", required=True, type=Path)
    manifest_parser.add_argument("--annotation-set")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a metadata-only command and return its process status."""
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
        if args.data_command == "download":
            downloaded_to = download_dataset(args.dataset, args.destination)
            print(f"Downloaded {args.dataset} to {downloaded_to}")
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

    parser.print_help()
    return 0
