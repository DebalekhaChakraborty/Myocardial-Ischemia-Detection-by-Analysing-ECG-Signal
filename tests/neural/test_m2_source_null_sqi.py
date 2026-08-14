"""Source-null SQI: structural missingness is not a null feature value.

M2 development recovery1 consumed both arm claims and then failed because the
partition-aware join used `isnan(output)` as proof that a row had never been
assigned. NaN is also the legitimate representation of an upstream source null
-- the frozen signal contract permits a spectral ratio to be uncomputable -- so
a valid corpus raised a structural-integrity error. See
`docs/M2_DEVELOPMENT_RECOVERY1_FAILURE_AND_RECOVERY2_DECISION_V1.md`.

These tests prove the two are now distinguished, that a source null survives the
join bit-for-bit, and that the SCIENTIFIC meaning of such a value is decided
where it always was -- by the existing frozen `evaluate_gate` / `evaluate_g3`,
unchanged and with no NaN-specific branch.

Synthetic fixtures only. No real VALIDATION corpus, no `.stb`, no scoring, no
metric, no TEST.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from cardiosentinel.neural import m2_feature_join as J
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural.m1_store import STABLE_ID_FILE
from cardiosentinel.neural.m2_gate_derivation import _combined_column_indices
from cardiosentinel.neural.patient_memory import (
    OBSERVATION_AVAILABLE,
    OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    REPRESENTATION_DIM,
)
from tests.neural.test_m2_partition_join import (
    _FakeStore,
    _manifest,
    _rows_in_causal_order,
    _stable_id,
    _store_arrays,
    _write_feature_corpus,
)

NULL_COLUMN = "high_frequency_power_ratio"


def _null_one_value(root, record_id, position, column_name):
    """Make ONE source feature value a legitimate null, as the corpus may."""
    path = root / f"{record_id}.npz"
    with np.load(path, allow_pickle=False) as cached:
        ids = np.asarray(cached["stable_ids"])
        features = np.asarray(cached["features"])
    features[position, _combined_column_indices()[column_name]] = np.nan
    np.savez(path, stable_ids=ids, features=features)
    return str(ids[position])


# --------------------------------------------------------------------------
# §17.1-3 -- a source null is not structural missingness
# --------------------------------------------------------------------------


def test_a_source_null_no_longer_looks_like_an_unwritten_row(tmp_path):
    """§17.1 -- the exact recovery1 failure, now a successful join."""
    root = _write_feature_corpus(tmp_path / "features")
    _null_one_value(root, "v00002", 1, NULL_COLUMN)
    columns = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )
    assert set(columns) >= {NULL_COLUMN}
    assert np.count_nonzero(np.isnan(columns[NULL_COLUMN])) == 1


def test_the_source_null_survives_the_join_exactly(tmp_path):
    """§17.2/§17.10 -- carried through as NaN, never imputed."""
    root = _write_feature_corpus(tmp_path / "features")
    null_id = _null_one_value(root, "v00002", 1, NULL_COLUMN)

    arrays = _store_arrays()
    columns = J.join_sqi_and_morphology_for_partition(
        _FakeStore(arrays), _manifest(), root, "validation"
    )
    position = int(np.flatnonzero(np.asarray(arrays[STABLE_ID_FILE]) == null_id)[0])
    value = columns[NULL_COLUMN][position]
    assert np.isnan(value)
    # Not zero, not a bound, not an infinity, not a median.
    assert value != 0.0
    assert not np.isinf(value)
    for bound in G.G3_UPPER_BOUNDS.values():
        assert not (value == bound)


def test_no_row_is_dropped_and_neighbours_are_bit_identical(tmp_path):
    """§17.3 -- the null costs its neighbours nothing."""
    root = _write_feature_corpus(tmp_path / "features")
    baseline = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )
    null_id = _null_one_value(root, "v00002", 1, NULL_COLUMN)
    nulled = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )

    total = len(_rows_in_causal_order())
    for name, values in nulled.items():
        assert values.shape[0] == total, name
    arrays = _store_arrays()
    position = int(np.flatnonzero(np.asarray(arrays[STABLE_ID_FILE]) == null_id)[0])
    for name, values in nulled.items():
        for index in range(total):
            if name == NULL_COLUMN and index == position:
                assert np.isnan(values[index])
                continue
            # Every other value is bit-identical to the un-nulled join.
            assert values[index].hex() == baseline[name][index].hex(), (name, index)


def test_every_other_column_is_unaffected_by_one_null(tmp_path):
    root = _write_feature_corpus(tmp_path / "features")
    _null_one_value(root, "v00001", 0, NULL_COLUMN)
    columns = J.join_sqi_and_morphology_for_partition(
        _FakeStore(_store_arrays()), _manifest(), root, "validation"
    )
    for name, values in columns.items():
        if name == NULL_COLUMN:
            continue
        assert not np.any(np.isnan(values)), name


# --------------------------------------------------------------------------
# §17.4-6 -- structural integrity checks are all intact
# --------------------------------------------------------------------------


def test_a_genuinely_unwritten_row_is_still_fatal():
    """§17.4 -- the mask catches what the NaN sentinel was meant to catch."""
    stable_ids = np.asarray(
        [_stable_id("v00001", 0, index) for index in range(4)], dtype=np.str_
    )
    written = np.ones(4, dtype=bool)
    written[2] = False
    with pytest.raises(J.M2FeatureJoinError, match="never written"):
        J.require_all_rows_written(written, "validation", stable_ids)


def test_the_mask_accepts_a_fully_written_join():
    stable_ids = np.asarray(
        [_stable_id("v00001", 0, index) for index in range(4)], dtype=np.str_
    )
    assert (
        J.require_all_rows_written(np.ones(4, dtype=bool), "validation", stable_ids)
        is None
    )


def test_the_unwritten_refusal_names_positions_and_stable_ids():
    stable_ids = np.asarray(
        [_stable_id("v00001", 0, index) for index in range(4)], dtype=np.str_
    )
    written = np.ones(4, dtype=bool)
    written[1] = False
    with pytest.raises(J.M2FeatureJoinError) as caught:
        J.require_all_rows_written(written, "validation", stable_ids)
    message = str(caught.value)
    assert "[1]" in message
    assert str(stable_ids[1]) in message
    assert "3 of 4" in message


def test_the_join_delegates_to_the_assignment_mask():
    source = inspect.getsource(J.join_sqi_and_morphology_for_partition)
    assert "written[start:end] = True" in source
    assert "require_all_rows_written(written, evaluated, stable_ids)" in source


def test_the_conflated_nan_sentinel_is_gone():
    """The message that consumed recovery1 no longer exists anywhere."""
    join_source = inspect.getsource(J.join_sqi_and_morphology_for_partition)
    mask_source = inspect.getsource(J.require_all_rows_written)
    assert "structurally assigned only" in mask_source
    assert "never written" in mask_source
    # The old conflated sentinel and its message are both gone.
    assert "left unmatched rows for" not in join_source
    assert "left unmatched rows for" not in mask_source
    assert "np.any(np.isnan" not in join_source


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("extra", "absent from the stream cache"),
        ("duplicate", "duplicate stable IDs"),
        ("truncate", "not row-aligned with itself"),
    ],
)
def test_identity_checks_remain_intact_with_a_source_null(tmp_path, mutate, match):
    """§17.5/§17.6 -- a null does not weaken any structural check."""
    root = _write_feature_corpus(tmp_path / "features")
    _null_one_value(root, "v00002", 0, NULL_COLUMN)
    path = root / "v00002.npz"
    with np.load(path, allow_pickle=False) as cached:
        ids = np.asarray(cached["stable_ids"])
        features = np.asarray(cached["features"])
    if mutate == "extra":
        ids = np.concatenate(
            [ids, np.asarray(["ltstdb:v00002:9:0:2500"], dtype=ids.dtype)]
        )
        features = np.vstack([features, features[:1]])
    elif mutate == "duplicate":
        ids = np.concatenate([ids, ids[:1]])
        features = np.vstack([features, features[:1]])
    else:
        features = features[:-1]
    np.savez(path, stable_ids=ids, features=features)

    with pytest.raises(J.M2FeatureJoinError, match=match):
        J.join_sqi_and_morphology_for_partition(
            _FakeStore(_store_arrays()), _manifest(), root, "validation"
        )


def test_a_missing_record_cache_is_still_fatal(tmp_path):
    import json

    from cardiosentinel.neural.m2_gate_derivation import FEATURE_MANIFEST_NAME

    root = _write_feature_corpus(tmp_path / "features")
    (root / "v00002.npz").unlink()
    manifest = json.loads((root / FEATURE_MANIFEST_NAME).read_text())
    manifest["records"] = [
        entry for entry in manifest["records"] if entry["record_id"] != "v00002"
    ]
    (root / FEATURE_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(J.M2FeatureJoinError, match="record set"):
        J.join_sqi_and_morphology_for_partition(
            _FakeStore(_store_arrays()), _manifest(), root, "validation"
        )


# --------------------------------------------------------------------------
# §7/§17.7-9 -- the SCIENTIFIC meaning lives in the existing frozen policy
# --------------------------------------------------------------------------


def _gate(observation_state, *, arm, sqi_null=True, representation=True, score=0.0):
    """Call the REAL frozen gate. No wrapper, no NaN-specific path."""
    sqi = dict.fromkeys(G.G3_SQI_COLUMNS, 0.0)
    if sqi_null:
        sqi[NULL_COLUMN] = float("nan")
    available = observation_state == OBSERVATION_AVAILABLE
    return P.evaluate_gate(
        arm=arm,
        observation_state=observation_state,
        representation=(
            np.zeros(REPRESENTATION_DIM, dtype=np.float64)
            if representation and available
            else None
        ),
        finite_sample_fraction=1.0 if available else None,
        sqi=sqi if available else None,
        morphology_valid=1.0 if available else None,
        score=score if available else None,
        available_time=10.0,
        refractory_until_before=float("-inf") if arm == "M2-G" else None,
    )


def test_case_a_unavailable_exact_flat_with_a_null_is_g1_only():
    """§7 CASE A / §17.7 -- G1 fails; G2-G6 are NOT APPLICABLE."""
    decision = _gate(OBSERVATION_UNAVAILABLE_EXACT_FLAT, arm="M2-G")
    assert decision.g1_available is False
    results = decision.condition_results()
    for condition in ("G2", "G3", "G4", "G5", "G6"):
        assert results[condition] is None, condition
    assert decision.admitted is False
    # A physically unavailable row is NOT counted as a G3 refusal.
    assert decision.g3_sqi_admissible is None


def test_case_b_available_with_a_null_g3_feature_fails_g3():
    """§7 CASE B / §17.8 -- non-finite already fails the frozen G3 rule."""
    decision = _gate(OBSERVATION_AVAILABLE, arm="M2-G")
    assert decision.g1_available is True
    assert not decision.g3_feature_results[NULL_COLUMN]
    assert decision.g3_sqi_admissible is False
    assert decision.admitted is False
    for name, passed in decision.g3_feature_results.items():
        if name != NULL_COLUMN:
            assert bool(passed) is True, name


def test_an_available_row_without_the_null_still_passes_g3():
    """The null is what fails it -- nothing else changed."""
    decision = _gate(OBSERVATION_AVAILABLE, arm="M2-G", sqi_null=False)
    assert decision.g3_sqi_admissible is True
    assert all(decision.g3_feature_results.values())


def test_the_frozen_g3_rule_is_finite_and_bounded_with_no_nan_branch():
    """§17.11 -- no NaN-specific branch, and no threshold changed."""
    source = inspect.getsource(P.evaluate_g3)
    assert "np.isfinite(value) and value <= GATE.G3_UPPER_BOUNDS[column]" in source
    for banned in ("nan_policy", "nan_to_num", "fillna", "impute", "median"):
        assert banned not in source, banned
    assert G.G3_UPPER_BOUNDS[NULL_COLUMN] > 0


def test_case_c_m2_0_ignores_g3_exactly_as_before():
    """§7 CASE C / §17.9 -- the naive control does not operate G3-G6."""
    decision = _gate(OBSERVATION_AVAILABLE, arm="M2-0")
    assert decision.g1_available is True
    results = decision.condition_results()
    for condition in ("G3", "G4", "G5", "G6"):
        assert results[condition] is None, condition
    # An available, finite row still admits under the inherited naive policy.
    assert decision.admitted is True


def test_m2_0_admits_identically_with_and_without_the_null():
    """The naive control's behaviour is untouched by an M2-G-only quantity."""
    with_null = _gate(OBSERVATION_AVAILABLE, arm="M2-0", sqi_null=True)
    without = _gate(OBSERVATION_AVAILABLE, arm="M2-0", sqi_null=False)
    assert with_null.admitted == without.admitted is True
    assert with_null.condition_results() == without.condition_results()


def test_m2_0_still_refuses_an_unavailable_row():
    decision = _gate(OBSERVATION_UNAVAILABLE_EXACT_FLAT, arm="M2-0")
    assert decision.g1_available is False
    assert decision.admitted is False


# --------------------------------------------------------------------------
# §17.12-13 -- nothing frozen moved
# --------------------------------------------------------------------------


def test_the_frozen_m2_protocol_and_receipt_digests_are_unchanged():
    assert (
        G.validate_m2_protocol()
        == "a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c"
    )
    assert (
        G.validate_m2_gate_receipt()
        == "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )


def test_no_threshold_or_bound_changed():
    from cardiosentinel.neural import m2_scorer as SC

    assert SC.M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert SC.NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614
    assert set(G.G3_SQI_COLUMNS) == {
        "flatline_fraction",
        "repeated_value_fraction",
        "derivative_outlier_fraction",
        "high_frequency_power_ratio",
        "powerline_ratio_50hz",
        "powerline_ratio_60hz",
    }
    assert P.M2_ARMS == ("M2-0", "M2-G")


def test_the_join_performs_no_imputation():
    """§17.10 -- structurally: the join writes only the source value."""
    import ast

    tree = ast.parse(inspect.getsource(J.join_sqi_and_morphology_for_partition))
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for banned in ("nan_to_num", "fillna", "median", "nanmean", "nanmedian", "where"):
        assert banned not in called, banned
    source = inspect.getsource(J.join_sqi_and_morphology_for_partition)
    assert "columns[name][start:end] = npz_features[positions, column]" in source


def test_this_module_opens_no_real_development_data():
    import ast
    from pathlib import Path

    forbidden = {
        "load_stream_store",
        "load_p1_embedding_cache",
        "build_validation_challenge_index",
        "iter_timeline_streams",
        "read_annotations",
        "read_record",
        "canonical_roots",
        "execute_canonical_development",
    }
    tree = ast.parse(Path(__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (called & forbidden), sorted(called & forbidden)
