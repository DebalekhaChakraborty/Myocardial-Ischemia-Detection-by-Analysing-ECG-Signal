"""Dataset source identity, file hashing, and run provenance helpers."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from cardiosentinel import __version__


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_file_digests(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    """Hash selected files using paths relative to the dataset root."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def verify_sha256_manifest(
    checksums_path: Path, root: Path, paths: Iterable[Path] | None = None
) -> dict[str, str]:
    """Verify a selected download, or a complete release when paths is omitted."""
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, filename = line.partition("  ")
        if not separator:
            digest, separator, filename = line.partition(" ")
        filename = filename.lstrip("*").strip()
        if len(digest) != 64 or not filename:
            raise ValueError(f"Malformed checksum entry in {checksums_path}: {line!r}")
        expected[filename] = digest.lower()

    selected = sorted(expected) if paths is None else sorted(
        path.relative_to(root).as_posix() for path in paths
    )
    missing_entries = [filename for filename in selected if filename not in expected]
    if missing_entries:
        raise ValueError(
            f"No checksum entry for downloaded files: {sorted(missing_entries)}"
        )
    missing = [filename for filename in selected if not (root / filename).is_file()]
    if missing:
        raise ValueError(
            f"Missing files required by checksum manifest: {sorted(missing)}"
        )
    actual = {filename: sha256_file(root / filename) for filename in selected}
    mismatches = [
        filename for filename, digest in actual.items() if digest != expected[filename]
    ]
    if mismatches:
        raise ValueError(f"SHA-256 mismatch for: {sorted(mismatches)}")
    return actual


def git_provenance(repository_root: Path) -> dict[str, object]:
    """Capture revision state without failing outside a Git checkout."""

    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_sha": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "cardiosentinel_version": __version__,
    }
