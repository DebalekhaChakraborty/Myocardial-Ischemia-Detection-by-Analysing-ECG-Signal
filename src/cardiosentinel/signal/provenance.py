"""Runtime and physical-signal provenance for Phase 2 processing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import wfdb

from cardiosentinel import __version__
from cardiosentinel.signal.config import FilterProfile
from cardiosentinel.signal.models import WaveformSegment
from cardiosentinel.signal.preprocessing import PROCESSING_SCHEMA_VERSION


def runtime_provenance(repository_root: Path | None = None) -> dict[str, object]:
    root = repository_root or Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "cardiosentinel_version": __version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "wfdb_version": wfdb.__version__,
    }


def processing_provenance(
    segment: WaveformSegment,
    profile: FilterProfile,
    repository_root: Path | None = None,
) -> dict[str, object]:
    return {
        **runtime_provenance(repository_root),
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "dataset": {"id": segment.dataset_id, "version": segment.dataset_version},
        "record": segment.record_id,
        "subject": segment.subject_id,
        "sample_interval": [segment.start_sample, segment.end_sample],
        "sampling_frequency_hz": segment.sampling_frequency_hz,
        "source_physical_units": list(segment.source_physical_units),
        "canonical_physical_units": list(segment.physical_units),
        "filter_configuration": profile.as_dict(),
    }
