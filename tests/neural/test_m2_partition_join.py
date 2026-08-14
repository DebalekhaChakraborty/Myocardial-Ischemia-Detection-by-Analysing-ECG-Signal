"""The partition-aware COMBINED_V1 join, and the REAL `iter_timeline_streams`.

M2 development attempt #1 consumed both arm claims and then failed here, before
a single row was scored: `join_sqi_and_morphology` resolves its record cache
paths through a TRAIN-only helper, and the canonical route replays VALIDATION.

The assembled end-to-end test could not catch it, because it injected
`stream_source` and so replaced exactly the component that was broken. These
tests therefore drive the real `iter_timeline_streams("validation")` against a
synthetic on-disk validation stream cache and COMBINED_V1 corpus.

Synthetic fixtures only: no real VALIDATION stream cache, no real feature NPZ,
no real `.stb`, no P1 labels, no challenge selection, no scoring, no metric.
TEST is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_execution as X
from cardiosentinel.neural import m2_feature_join as J
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    COLD_START_BIN_FILE,
    D_LONG_FILE,
    D_SHORT_FILE,
    DISAGREEMENT_FILE,
    OBSERVATION_STATE_FILE,
    PAST_OBSERVED_FILE,
    PAST_UPDATE_FILE,
    RECORD_ID_FILE,
    RECORDING_AGE_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
)
from cardiosentinel.neural.m2_gate_derivation import (
    COMBINED_NEEDED_COLUMNS,
    FEATURE_MANIFEST_NAME,
    _combined_column_indices,
)
from cardiosentinel.neural.patient_memory import (
    OBSERVATION_AVAILABLE,
    REPRESENTATION_DIM,
)
from cardiosentinel.neural.protocol import DATASET, WINDOW_SAMPLES

# Two validation records, three streams, four rows each.
VALIDATION_STREAMS = (("v00001", 0), ("v00001", 1), ("v00002", 0))
TRAIN_RECORDS = ("t00009",)
ROWS = 4


def _stable_id(record_id, channel, index):
    start = index * 1250
    return f"{DATASET}:{record_id}:{channel}:{start}:{start + WINDOW_SAMPLES}"


def _rows_in_causal_order():
    """Rows grouped by record, then chronological -- the store's own order."""
    ordered = []
    for record_id, channel in VALIDATION_STREAMS:
        for index in range(ROWS):
            ordered.append((record_id, channel, index))
    return ordered


class _FakeStore:
    """A minimal stand-in exposing only `array()`, as the real store does."""

    def __init__(self, arrays):
        self._arrays = arrays

    def array(self, name):
        return self._arrays[name]

    def close(self):
        return None


def _store_arrays():
    ordered = _rows_in_causal_order()
    return {
        RECORD_ID_FILE: np.asarray([r for r, _c, _i in ordered], dtype=np.str_),
        CHANNEL_INDEX_FILE: np.asarray([c for _r, c, _i in ordered], dtype=np.int64),
        START_SAMPLE_FILE: np.asarray(
            [i * 1250 for _r, _c, i in ordered], dtype=np.int64
        ),
        STABLE_ID_FILE: np.asarray(
            [_stable_id(r, c, i) for r, c, i in ordered], dtype=np.str_
        ),
        OBSERVATION_STATE_FILE: np.full(
            len(ordered), OBSERVATION_AVAILABLE, dtype=np.uint8
        ),
        REPRESENTATION_FILE: np.tile(
            np.arange(REPRESENTATION_DIM, dtype=np.float32) / 100.0, (len(ordered), 1)
        ),
        D_SHORT_FILE: np.zeros(len(ordered), dtype=np.float64),
        D_LONG_FILE: np.zeros(len(ordered), dtype=np.float64),
        PAST_OBSERVED_FILE: np.zeros(len(ordered), dtype=np.int64),
        PAST_UPDATE_FILE: np.zeros(len(ordered), dtype=np.int64),
        DISAGREEMENT_FILE: np.zeros(len(ordered), dtype=np.float64),
        RECORDING_AGE_FILE: np.zeros(len(ordered), dtype=np.float64),
        COLD_START_BIN_FILE: np.full(len(ordered), "over_60_minutes", dtype="<U32"),
    }


def _manifest(partition="validation"):
    records = sorted({record for record, _channel in VALIDATION_STREAMS})
    return {
        "partition": partition,
        "record_ids": records,
        "full_stream_row_count": len(_rows_in_causal_order()),
        "ordered_stable_id_sha256": "1" * 64,
        "stream_cache_sha256": "2" * 64,
        "ordered_chronology_sha256": "3" * 64,
        "split_sha256": "4" * 64,
        "feature_corpus_sha256": "5" * 64,
        "representation_dim": REPRESENTATION_DIM,
        "distance_standardizer_sha256": "6" * 64,
    }


def _feature_value(record_id, channel, index, column):
    """A deterministic, distinguishable value per (row, column)."""
    seed = (hash((record_id, channel, index)) % 97) / 1000.0
    return round(0.5 + seed + column / 1000.0, 6)


def _write_feature_corpus(root: Path, *, validation=True, train=True):
    """A synthetic COMBINED_V1 corpus with both partitions represented."""
    root.mkdir(parents=True, exist_ok=True)
    width = max(_combined_column_indices().values()) + 1
    records = []

    if validation:
        for record_id in sorted({r for r, _c in VALIDATION_STREAMS}):
            rows = [
                (record_id, channel, index)
                for rec, channel in VALIDATION_STREAMS
                if rec == record_id
                for index in range(ROWS)
            ]
            ids = np.asarray([_stable_id(*row) for row in rows], dtype=np.str_)
            features = np.zeros((len(rows), width), dtype=np.float64)
            for position, row in enumerate(rows):
                for column in range(width):
                    features[position, column] = _feature_value(*row, column)
            path = root / f"{record_id}.npz"
            np.savez(path, stable_ids=ids, features=features)
            records.append(
                {
                    "record_id": record_id,
                    "partition": "validation",
                    "status": "complete",
                    "cache_path": path.name,
                }
            )

    if train:
        for record_id in TRAIN_RECORDS:
            ids = np.asarray([f"{DATASET}:{record_id}:0:0:2500"], dtype=np.str_)
            path = root / f"{record_id}.npz"
            np.savez(path, stable_ids=ids, features=np.zeros((1, width)))
            records.append(
                {
                    "record_id": record_id,
                    "partition": "train",
                    "status": "complete",
                    "cache_path": path.name,
                }
            )

    (root / FEATURE_MANIFEST_NAME).write_text(json.dumps({"records": records}))
    return root


# --------------------------------------------------------------------------
# §7 -- the partition-aware helper selects the right partition
# --------------------------------------------------------------------------


def test_validation_record_paths_are_selected(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    paths = J.combined_record_cache_paths_for_partition(root, "validation")
    assert sorted(paths) == ["v00001", "v00002"]
    for value in paths.values():
        assert value.is_file()


def test_train_records_are_not_substituted_for_validation(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    validation = J.combined_record_cache_paths_for_partition(root, "validation")
    train = J.combined_record_cache_paths_for_partition(root, "train")
    assert set(validation).isdisjoint(train)
    assert sorted(train) == list(TRAIN_RECORDS)
    assert TRAIN_RECORDS[0] not in validation


def test_incomplete_entries_are_ignored(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    manifest = json.loads((root / FEATURE_MANIFEST_NAME).read_text())
    manifest["records"].append(
        {
            "record_id": "v99999",
            "partition": "validation",
            "status": "pending",
            "cache_path": "v00001.npz",
        }
    )
    (root / FEATURE_MANIFEST_NAME).write_text(json.dumps(manifest))
    assert "v99999" not in J.combined_record_cache_paths_for_partition(
        root, "validation"
    )


def test_duplicate_record_ids_are_refused(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    manifest = json.loads((root / FEATURE_MANIFEST_NAME).read_text())
    manifest["records"].append(
        {
            "record_id": "v00001",
            "partition": "validation",
            "status": "complete",
            "cache_path": "v00002.npz",
        }
    )
    (root / FEATURE_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(J.M2FeatureJoinError, match="more than once"):
        J.combined_record_cache_paths_for_partition(root, "validation")


def test_the_join_hard_rejects_test(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    with pytest.raises(J.M2FeatureJoinError, match="sealed test"):
        J.combined_record_cache_paths_for_partition(root, "test")
    with pytest.raises(J.M2FeatureJoinError, match="sealed test"):
        J.require_join_partition("TEST")


# --------------------------------------------------------------------------
# §7 -- the exact defect: TRAIN-only paths + a VALIDATION manifest
# --------------------------------------------------------------------------


def test_train_only_corpus_with_a_validation_manifest_fails_by_partition(tmp_path):
    """The attempt #1 shape, now with a partition-specific error."""
    root = _write_feature_corpus(tmp_path / "features", validation=False, train=True)
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError) as caught:
        J.join_sqi_and_morphology_for_partition(store, _manifest(), root, "validation")
    message = str(caught.value)
    assert "VALIDATION record set" in message
    assert "TRAIN record set" not in message
    assert "v00001" in message


def test_a_partition_mismatch_between_manifest_and_request_is_refused(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError, match="must be the same"):
        J.join_sqi_and_morphology_for_partition(
            store, _manifest(partition="train"), root, "validation"
        )


def test_a_missing_stable_id_is_fatal_not_inner_joined(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    # Drop one row from one record's NPZ.
    path = root / "v00002.npz"
    with np.load(path, allow_pickle=False) as cached:
        ids = np.asarray(cached["stable_ids"])[1:]
        features = np.asarray(cached["features"])[1:]
    np.savez(path, stable_ids=ids, features=features)
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError, match="no feature match"):
        J.join_sqi_and_morphology_for_partition(store, _manifest(), root, "validation")


def test_the_validation_join_aligns_by_stable_identity(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    store = _FakeStore(_store_arrays())
    columns = J.join_sqi_and_morphology_for_partition(
        store, _manifest(), root, "validation"
    )
    assert sorted(columns) == sorted(COMBINED_NEEDED_COLUMNS)
    indices = _combined_column_indices()
    ordered = _rows_in_causal_order()
    for position, row in enumerate(ordered):
        for name, column in indices.items():
            assert columns[name][position] == pytest.approx(
                _feature_value(*row, column)
            )


# --------------------------------------------------------------------------
# §7/§8 -- the REAL iter_timeline_streams, not an injected substitute
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_validation_cache(tmp_path, monkeypatch):
    """A synthetic on-disk validation stream cache + COMBINED_V1 corpus."""
    feature_root = _write_feature_corpus(tmp_path / "features")
    arrays = _store_arrays()
    manifest = _manifest()

    def fake_load_stream_store(cache_root, partition, **kwargs):
        assert str(partition) == "validation"
        return _FakeStore(arrays), manifest

    monkeypatch.setattr(X, "load_stream_store", fake_load_stream_store)
    return {"feature_root": feature_root, "manifest": manifest, "arrays": arrays}


def test_the_real_iterator_yields_validation_streams(synthetic_validation_cache):
    """The regression that was missing: the real iterator, on VALIDATION."""
    streams = list(
        X.iter_timeline_streams(
            "validation",
            stream_cache_root=Path("unused"),
            feature_root=synthetic_validation_cache["feature_root"],
        )
    )
    assert [key for key, _rows in streams] == sorted(VALIDATION_STREAMS)
    for key, rows in streams:
        assert len(rows) == ROWS
        # Chronological within the stream.
        assert [row.start_sample for row in rows] == sorted(
            row.start_sample for row in rows
        )
        for row in rows:
            assert (row.record_id, row.channel_index) == key
            assert row.representation is not None
            assert set(row.sqi) == set(G.G3_SQI_COLUMNS)


def test_the_real_iterator_carries_the_expected_feature_values(
    synthetic_validation_cache,
):
    indices = _combined_column_indices()
    for key, rows in X.iter_timeline_streams(
        "validation",
        stream_cache_root=Path("unused"),
        feature_root=synthetic_validation_cache["feature_root"],
    ):
        record_id, channel = key
        for index, row in enumerate(rows):
            expected_ffs = _feature_value(
                record_id, channel, index, indices["finite_sample_fraction"]
            )
            assert row.finite_sample_fraction == pytest.approx(expected_ffs)
            assert row.morphology_valid == pytest.approx(
                _feature_value(record_id, channel, index, indices["morphology_valid"])
            )
            for name in G.G3_SQI_COLUMNS:
                assert row.sqi[name] == pytest.approx(
                    _feature_value(record_id, channel, index, indices[name])
                )


def test_the_real_iterator_fails_on_a_train_only_corpus(tmp_path, monkeypatch):
    """If a TRAIN-only join is ever reintroduced, this test fails."""
    feature_root = _write_feature_corpus(
        tmp_path / "features", validation=False, train=True
    )
    arrays = _store_arrays()
    manifest = _manifest()
    monkeypatch.setattr(
        X, "load_stream_store", lambda *a, **k: (_FakeStore(arrays), manifest)
    )
    with pytest.raises(J.M2FeatureJoinError, match="VALIDATION record set"):
        list(
            X.iter_timeline_streams(
                "validation",
                stream_cache_root=Path("unused"),
                feature_root=feature_root,
            )
        )


def test_the_canonical_iterator_does_not_use_the_train_only_helper():
    import inspect

    source = inspect.getsource(X.iter_timeline_streams)
    assert "join_sqi_and_morphology_for_partition" in source
    assert "join_sqi_and_morphology(store" not in source
    assert "_train_record_cache_paths" not in source


# --------------------------------------------------------------------------
# §9 -- the frozen TRAIN gate derivation is untouched
# --------------------------------------------------------------------------


def test_the_train_only_helper_still_means_train():
    """The frozen TRAIN derivation keeps its semantics and its helper."""
    import inspect

    from cardiosentinel.neural import m2_gate_derivation as GD

    source = inspect.getsource(GD._train_record_cache_paths)
    assert 'entry.get("partition") == "train"' in source
    # The frozen derivation still uses it, unchanged.
    assert "_train_record_cache_paths(" in inspect.getsource(GD.join_sqi_and_morphology)
    assert "TRAIN record set" in inspect.getsource(GD.join_sqi_and_morphology)


def test_the_frozen_m2_protocol_and_receipt_are_unchanged():
    assert G.validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert G.validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256
    assert (
        G.M2_PROTOCOL_SHA256
        == "a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c"
    )
    assert (
        G.M2_GATE_RECEIPT_SHA256
        == "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )


def test_the_frozen_gate_thresholds_are_unchanged():
    from cardiosentinel.neural import m2_scorer as SC

    assert SC.M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert SC.NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614
    assert G.M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"


# --------------------------------------------------------------------------
# §8 -- the assembled route drives the REAL iterator
# --------------------------------------------------------------------------


def test_the_assembled_route_uses_the_real_iterator(tmp_path, monkeypatch):
    """§8 -- an end-to-end run whose stream assembly is NOT injected.

    Attempt #1 was invisible to the previous end-to-end test because that test
    injected `stream_source`. Here the expensive scientific components stay
    synthetic, but `replay_both_arms` falls through to the real
    `iter_timeline_streams("validation")` against a synthetic corpus.
    """
    from cardiosentinel.neural import m2_development_run as R

    feature_root = _write_feature_corpus(tmp_path / "features")
    arrays = _store_arrays()
    manifest = _manifest()
    monkeypatch.setattr(
        X, "load_stream_store", lambda *a, **k: (_FakeStore(arrays), manifest)
    )

    seen: list[tuple[str, int]] = []

    class _Store:
        def __init__(self, **kwargs):
            self.arm = kwargs.get("arm")

        def add_stream(self, key, evidence, trajectory=None):
            seen.append(key)

    class _Scorer:
        def __call__(self, representation, d_long):
            return 0.1

    from cardiosentinel.neural.patient_memory import M1DistanceStandardizer

    standardizer = M1DistanceStandardizer(
        means=tuple([0.0] * REPRESENTATION_DIM),
        scales=tuple([1.0] * REPRESENTATION_DIM),
        prior=tuple([0.0] * REPRESENTATION_DIM),
        zero_variance_dimensions=(),
        fitted_rows=1,
        fitted_population="train",
        input_identities={"partition": "train"},
    )

    R.replay_both_arms(
        stores={arm: _Store(arm=arm) for arm in R.CANONICAL_ARM_ORDER},
        standardizer=standardizer,
        scorer=_Scorer(),
        stream_cache_root=Path("unused"),
        feature_root=feature_root,
        # NOTE: stream_source deliberately NOT injected.
    )
    # Every stream, once per arm, in frozen key order.
    assert seen == [key for key in sorted(VALIDATION_STREAMS) for _arm in range(2)]


def test_replay_falls_through_to_the_real_iterator_by_default():
    """If someone makes `stream_source` mandatory again, this fails."""
    import inspect

    from cardiosentinel.neural import m2_development_run as R

    signature = inspect.signature(R.replay_both_arms)
    assert signature.parameters["stream_source"].default is None
    source = inspect.getsource(R.replay_both_arms)
    assert "iter_timeline_streams" in source


# --------------------------------------------------------------------------
# §7 -- EXACT stable-ID correspondence, both directions
# --------------------------------------------------------------------------


def _corrupt_record_npz(root: Path, record_id: str, mutate):
    path = root / f"{record_id}.npz"
    with np.load(path, allow_pickle=False) as cached:
        ids = np.asarray(cached["stable_ids"])
        features = np.asarray(cached["features"])
    ids, features = mutate(ids, features)
    np.savez(path, stable_ids=ids, features=features)


def test_an_extra_feature_stable_id_is_refused(tmp_path):
    """A corpus holding rows the stream cache does not list is not that corpus."""
    root = _write_feature_corpus(tmp_path / "features")
    _corrupt_record_npz(
        root,
        "v00002",
        lambda ids, features: (
            np.concatenate(
                [ids, np.asarray(["ltstdb:v00002:9:0:2500"], dtype=ids.dtype)]
            ),
            np.vstack([features, features[:1]]),
        ),
    )
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError, match="absent from the stream cache"):
        J.join_sqi_and_morphology_for_partition(store, _manifest(), root, "validation")


def test_a_duplicate_feature_stable_id_is_refused(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    _corrupt_record_npz(
        root,
        "v00002",
        lambda ids, features: (
            np.concatenate([ids, ids[:1]]),
            np.vstack([features, features[:1]]),
        ),
    )
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError, match="duplicate stable IDs"):
        J.join_sqi_and_morphology_for_partition(store, _manifest(), root, "validation")


def test_a_duplicate_stream_stable_id_is_refused(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    arrays = _store_arrays()
    ids = arrays[STABLE_ID_FILE].copy()
    ids[1] = ids[0]  # two rows of the same stream now share an identity
    arrays[STABLE_ID_FILE] = ids
    with pytest.raises(J.M2FeatureJoinError, match="stream cache has duplicate"):
        J.join_sqi_and_morphology_for_partition(
            _FakeStore(arrays), _manifest(), root, "validation"
        )


def test_a_feature_row_count_mismatch_is_refused(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    _corrupt_record_npz(root, "v00002", lambda ids, features: (ids, features[:-1]))
    store = _FakeStore(_store_arrays())
    with pytest.raises(J.M2FeatureJoinError, match="not row-aligned with itself"):
        J.join_sqi_and_morphology_for_partition(store, _manifest(), root, "validation")


def test_feature_rows_in_a_different_order_still_align(tmp_path):
    """Order is not asserted: the join realigns by stable identity."""
    root = _write_feature_corpus(tmp_path / "features")
    baseline = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )
    _corrupt_record_npz(
        root,
        "v00001",
        lambda ids, features: (ids[::-1].copy(), features[::-1].copy()),
    )
    permuted = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )
    for name, values in baseline.items():
        assert np.array_equal(values, permuted[name]), name


# --------------------------------------------------------------------------
# §8 -- a REAL on-disk M1RowStore, through the real memmap read path
# --------------------------------------------------------------------------


def _write_real_store(directory: Path):
    """Create a genuine `M1RowStore` on disk with the existing writer.

    Uses the production `M1StoreSpec`/`M1RowStore(create=True)` writer and then
    reopens through the real `create=False` memmap loader. No production
    behaviour is invented, and `load_stream_store`'s frozen manifest identities
    are deliberately not faked: this covers the store layer beneath the join,
    which is where the join actually reads from.
    """
    from cardiosentinel.neural.m1_store import M1RowStore, M1StoreSpec

    arrays = _store_arrays()
    rows = arrays[RECORD_ID_FILE].shape[0]
    spec = M1StoreSpec(rows=rows, representation_dim=REPRESENTATION_DIM)
    store = M1RowStore(directory, spec, create=True)
    for name, values in arrays.items():
        store.array(name)[:] = values
    store.close()
    return M1RowStore(directory, spec, create=False)


def test_the_join_reads_a_real_on_disk_store(tmp_path):
    """§8 -- the same join, driven through the real memmap store."""
    root = _write_feature_corpus(tmp_path / "features")
    store = _write_real_store(tmp_path / "store")
    try:
        columns = J.join_sqi_and_morphology_for_partition(
            store, _manifest(), root, "validation"
        )
    finally:
        store.close()

    indices = _combined_column_indices()
    for position, row in enumerate(_rows_in_causal_order()):
        for name, column in indices.items():
            assert columns[name][position] == pytest.approx(
                _feature_value(*row, column)
            )


def test_a_real_on_disk_store_still_refuses_a_train_only_corpus(tmp_path):
    root = _write_feature_corpus(tmp_path / "features", validation=False, train=True)
    store = _write_real_store(tmp_path / "store")
    try:
        with pytest.raises(J.M2FeatureJoinError, match="VALIDATION record set"):
            J.join_sqi_and_morphology_for_partition(
                store, _manifest(), root, "validation"
            )
    finally:
        store.close()
