"""Large-N engineering stress: the production path must stay bounded.

Attempt 1 failed because peak memory scaled with the corpus. These tests grow N
and assert the RSS curve stays flat rather than tracking row count. They are
engineering measurements, not science: the encoder is fake, the data is
synthetic, and the scratch root is explicitly non-canonical.

The large case is opt-in via `M1_STRESS_ROWS` so ordinary CI stays fast.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cardiosentinel.neural.m1_store import (
    DEFAULT_CHUNK_ROWS,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    M1RowStore,
    M1StoreSpec,
    locate_rows,
    streaming_ordered_stable_id_digest,
)
from cardiosentinel.neural.patient_memory import REPRESENTATION_DIM

STRESS_ROWS = int(os.environ.get("M1_STRESS_ROWS", "0"))


def resident_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def _fill(store: M1RowStore, rows: int, *, batch: int = 256) -> None:
    """Write embeddings batch-by-batch, exactly as the production path does."""
    representation = store.array(REPRESENTATION_FILE)
    stable = store.array(STABLE_ID_FILE)
    generator = np.random.default_rng(0)
    for start in range(0, rows, batch):
        end = min(start + batch, rows)
        block = generator.normal(size=(end - start, REPRESENTATION_DIM)).astype(
            np.float32
        )
        representation[start:end] = block
        stable[start:end] = np.asarray(
            [f"ltstdb:r{i // 512}:{i % 3}:{i * 1250}:{i * 1250 + 2500}"
             for i in range(start, end)],
            dtype=stable.dtype,
        )
        del block
    store.flush()


@pytest.mark.parametrize("rows", [20_000])
def test_write_path_memory_is_flat_in_n(tmp_path, rows):
    baseline = resident_mb()
    spec = M1StoreSpec(rows=rows, representation_dim=REPRESENTATION_DIM)
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        _fill(store, rows)
        peak = resident_mb()
        # One chunk of float32[8192, 146] is ~4.8 MB; the whole corpus would be
        # ~11 MB at this N, so a generous ceiling still fails a path that
        # accumulates per-row Python objects.
        assert peak - baseline < 600, f"write path grew {peak - baseline:.0f} MB"


def test_digest_path_does_not_load_the_corpus(tmp_path):
    rows = 20_000
    spec = M1StoreSpec(rows=rows, representation_dim=REPRESENTATION_DIM)
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        _fill(store, rows)
    store = M1RowStore(tmp_path / "store", spec, create=False)
    baseline = resident_mb()
    digest = store.content_digest(REPRESENTATION_FILE, chunk_rows=DEFAULT_CHUNK_ROWS)
    after = resident_mb()
    assert len(digest) == 64
    assert after - baseline < 400, f"digest grew {after - baseline:.0f} MB"
    store.close()


def test_streaming_digest_never_materializes_the_sequence():
    """A generator of 200k ids must not require a 200k-element list."""
    baseline = resident_mb()
    digest = streaming_ordered_stable_id_digest(
        f"ltstdb:r{i // 512}:{i % 3}:{i * 1250}:{i * 1250 + 2500}"
        for i in range(200_000)
    )
    assert len(digest) == 64
    # The duplicate-detection set is unavoidable and dominates; it is small
    # Python strings rather than one ndarray object per row.
    assert resident_mb() - baseline < 500


def test_targeted_selection_is_bounded_by_the_selection(tmp_path):
    rows = 20_000
    spec = M1StoreSpec(rows=rows, representation_dim=REPRESENTATION_DIM)
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        _fill(store, rows)
    store = M1RowStore(tmp_path / "store", spec, create=False)
    wanted = [
        str(v) for v in np.asarray(store.array(STABLE_ID_FILE))[:1000]
    ]
    baseline = resident_mb()
    positions = locate_rows(store, wanted)
    gathered = store.gather(REPRESENTATION_FILE, positions)
    assert gathered.shape == (1000, REPRESENTATION_DIM)
    assert resident_mb() - baseline < 300
    store.close()


@pytest.mark.skipif(
    STRESS_ROWS < 500_000,
    reason="set M1_STRESS_ROWS>=500000 to run the large-N engineering stress",
)
def test_large_n_stress_curve(tmp_path):
    baseline = resident_mb()
    spec = M1StoreSpec(rows=STRESS_ROWS, representation_dim=REPRESENTATION_DIM)
    marks: dict[int, float] = {}
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        representation = store.array(REPRESENTATION_FILE)
        generator = np.random.default_rng(0)
        for start in range(0, STRESS_ROWS, 256):
            end = min(start + 256, STRESS_ROWS)
            representation[start:end] = generator.normal(
                size=(end - start, REPRESENTATION_DIM)
            ).astype(np.float32)
            if end in (100_000, 250_000, 500_000):
                marks[end] = resident_mb()
        store.flush()
    peak = resident_mb()
    growth = peak - baseline
    # Linear per-row accumulation at 500k rows would be many GB.
    assert growth < 2_000, f"stress growth {growth:.0f} MB at {STRESS_ROWS} rows"
    assert marks, "no intermediate marks were recorded"
