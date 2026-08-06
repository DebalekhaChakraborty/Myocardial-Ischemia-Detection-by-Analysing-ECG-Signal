"""Minimal factual command-line interface for CardioSentinel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cardiosentinel import __version__
from cardiosentinel.config import DEFAULT_CONFIG_PATH, load_config


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

    parser.print_help()
    return 0

