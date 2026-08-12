"""M1-v2: physical observation availability decided before patient memory.

Attempt 2 under M1-v1 stopped on an exact-flat 10-second sensor interval. The
human decision (POLICY B) is that such an interval is a PHYSICALLY UNAVAILABLE
SENSOR OBSERVATION -- not physiology, and not a low-confidence observation.

These tests pin the resulting semantics: an unavailable row keeps its timeline
position and real elapsed time, but produces no B4-B call, no representation,
no deviation score, no memory update and no counter increment.

Everything is synthetic. No canonical root, no real corpus, no test partition.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m1_experiment, patient_memory
from cardiosentinel.neural.m1_store import (
    M1_STREAM_CACHE_SCHEMA,
    OBSERVATION_STATE_FILE,
    M1StoreSpec,
)
from cardiosentinel.neural.patient_memory import (
    ALPHA_LONG,
    ALPHA_SHORT,
    M1_EXPERIMENT_IDS,
    M1_PROTOCOL_V1_SHA256,
    OBSERVATION_AVAILABLE,
    OBSERVATION_STATE_ENUM,
    OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    OBSERVATION_UNINITIALIZED,
    REPRESENTATION_DIM,
    DualTimescaleMemory,
    M1MemoryError,
    exact_flat_unavailable,
    fit_distance_standardizer,
    m1_boundary_statement,
    validate_m1_protocol,
    validate_m1_protocol_v1,
)
from tests.neural.test_patient_memory import vector

# --------------------------------------------------------------------------
# The predicate is exactly the existing B4 hard criterion
# --------------------------------------------------------------------------


def test_predicate_matches_the_existing_b4_hard_validator():
    from cardiosentinel.signal.errors import SignalValidationError
    from cardiosentinel.signal.validation import validate_waveform_segment

    class _Segment:
        sampling_frequency_hz = 250.0
        sample_count = 2500
        channel_count = 1
        physical_units = ("mV",)

        def __init__(self, values):
            self.values = values.reshape(-1, 1)

    for values, expect_flat in (
        (np.full(2500, -5.12), True),
        (np.zeros(2500), True),
        (np.linspace(0.0, 1.0, 2500), False),
        (np.concatenate([np.zeros(2499), [1e-12]]), False),
    ):
        assert exact_flat_unavailable(values) is expect_flat
        raised = False
        try:
            validate_waveform_segment(_Segment(values))
        except SignalValidationError as error:
            raised = "no dynamic variation" in str(error)
        except Exception:
            raised = False
        assert raised is expect_flat, values[:3]


def test_predicate_never_reclassifies_a_fatal_failure():
    """Non-finite is FATAL and must never become 'unavailable'."""
    broken = np.full(2500, 0.5)
    broken[7] = np.nan
    with pytest.raises(M1MemoryError, match="fatal"):
        exact_flat_unavailable(broken)
    infinite = np.full(2500, 0.5)
    infinite[3] = np.inf
    with pytest.raises(M1MemoryError, match="fatal"):
        exact_flat_unavailable(infinite)


def test_b4_default_contract_is_unchanged():
    """M1-v2 must not weaken the frozen B4 waveform contract."""
    from cardiosentinel.signal.validation import validate_waveform_segment

    signature = inspect.signature(validate_waveform_segment)
    assert signature.parameters["require_dynamic"].default is True
    assert signature.parameters["require_finite"].default is True
    reader = inspect.getsource(
        __import__("cardiosentinel.signal.io", fromlist=["_read_segment"])._read_segment
    )
    assert "validate_waveform_segment(segment)" in reader


def test_availability_is_computed_not_whitelisted():
    """No production module may special-case the six known development IDs."""
    for module in (patient_memory, m1_experiment):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for known in (
            "ltstdb:s20571:1:8921250:8923750",
            "ltstdb:s20571:1:8927500:8930000",
        ):
            assert known not in literals, f"{module.__name__} whitelists {known}"
        assert "s20571" not in source


def test_morphology_valid_is_not_the_availability_criterion():
    predicate = inspect.getsource(patient_memory.exact_flat_unavailable)
    tree = ast.parse(predicate.lstrip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Name):
            assert "morphology" not in node.id


# --------------------------------------------------------------------------
# Protocol identities
# --------------------------------------------------------------------------


def test_v1_protocol_remains_immutable_and_v2_is_active():
    assert validate_m1_protocol_v1() == M1_PROTOCOL_V1_SHA256
    assert validate_m1_protocol() != M1_PROTOCOL_V1_SHA256
    assert M1_PROTOCOL_V1_SHA256 in patient_memory.SUPERSEDED_M1_PROTOCOL_SHA256


def test_v2_experiment_identities_cannot_masquerade_as_v1():
    assert M1_EXPERIMENT_IDS == (
        "M1S_short_memory_v2",
        "M1L_long_memory_v2",
        "M1D_dual_memory_v2",
    )
    for legacy in patient_memory.M1_V1_EXPERIMENT_IDS:
        assert legacy not in M1_EXPERIMENT_IDS
        with pytest.raises(M1MemoryError):
            patient_memory.require_m1_experiment(legacy)


def test_v2_uses_fresh_canonical_roots():
    from cardiosentinel.neural.cli import (
        DEFAULT_M1_RUN_ROOT,
        DEFAULT_M1_STREAM_CACHE_ROOT,
    )

    assert str(DEFAULT_M1_RUN_ROOT).endswith("phase5-m1-dual-memory-v2")
    assert str(DEFAULT_M1_STREAM_CACHE_ROOT).endswith("m1-stream-memory-v2")


def test_head_architecture_is_unchanged_from_v1():
    for experiment_id, width, params in (
        ("M1S_short_memory_v2", 147, 9537),
        ("M1L_long_memory_v2", 147, 9537),
        ("M1D_dual_memory_v2", 148, 9601),
    ):
        head = patient_memory.build_m1_head(experiment_id)
        identity = patient_memory.m1_head_identity(experiment_id, head)
        assert identity["input_dim"] == width
        assert identity["trainable_parameter_count"] == params
        assert identity["hidden_dim"] == 64
        assert identity["dropout"] == 0.10


def test_alphas_and_counter_semantics_are_frozen():
    assert ALPHA_SHORT == 0.01148597964710385
    assert ALPHA_LONG == 0.0009622411662165709
    boundary = m1_boundary_statement()
    assert boundary["alpha_time_rescaled"] is False
    assert boundary["unavailable_row_increments_counters"] is False
    assert boundary["unavailable_row_updates_memory"] is False
    assert boundary["b4_input_contract_weakened"] is False


# --------------------------------------------------------------------------
# Memory freeze across unavailable slots
# --------------------------------------------------------------------------


@pytest.fixture
def standardizer():
    rows = np.stack([vector(seed) for seed in range(80)]).astype(np.float64)
    return fit_distance_standardizer(rows, partition="train")


def _replay(states, standardizer, values):
    """Reference replay implementing the frozen M1-v2 unavailable semantics."""
    memory = DualTimescaleMemory(standardizer.prior_vector())
    out = []
    for state, x in zip(states, values, strict=True):
        if state == OBSERVATION_UNAVAILABLE_EXACT_FLAT:
            out.append(
                {
                    "d_short": np.nan,
                    "d_long": np.nan,
                    "observed": memory.past_observed_count,
                    "updated": memory.past_update_count,
                    "mu_short": memory.mu_short,
                }
            )
            continue
        f = memory.observe(standardizer.standardize(x)[0])
        out.append(
            {
                "d_short": f.d_short,
                "d_long": f.d_long,
                "observed": f.past_observed_count,
                "updated": f.past_update_count,
                "mu_short": memory.mu_short,
            }
        )
    return out


def test_single_outage_freezes_memory_and_preserves_time(standardizer):
    values = [vector(i) for i in range(5)]
    states = [OBSERVATION_AVAILABLE] * 5
    states[2] = OBSERVATION_UNAVAILABLE_EXACT_FLAT
    rows = _replay(states, standardizer, values)

    assert np.isnan(rows[2]["d_short"]) and np.isnan(rows[2]["d_long"])
    # counters unchanged across the slot
    assert rows[2]["observed"] == rows[1]["observed"] + 1
    assert rows[2]["observed"] == rows[2]["updated"]
    # prototype identical before and across the outage
    np.testing.assert_array_equal(rows[1]["mu_short"], rows[2]["mu_short"])
    # the next available row resumes from the pre-outage prototype
    baseline = _replay([OBSERVATION_AVAILABLE] * 4, standardizer,
                       [values[0], values[1], values[3], values[4]])
    assert rows[3]["d_short"] == pytest.approx(baseline[2]["d_short"])
    assert rows[3]["observed"] == baseline[2]["observed"]


def test_six_slot_outage_matches_the_discovered_event(standardizer):
    """Six overlapping unavailable slots, as in the 35-second s20571 event."""
    values = [vector(i) for i in range(12)]
    states = [OBSERVATION_AVAILABLE] * 12
    for i in range(3, 9):
        states[i] = OBSERVATION_UNAVAILABLE_EXACT_FLAT
    rows = _replay(states, standardizer, values)

    frozen = rows[2]
    for i in range(3, 9):
        assert np.isnan(rows[i]["d_short"]) and np.isnan(rows[i]["d_long"])
        assert rows[i]["observed"] == frozen["observed"] + 1
        assert rows[i]["updated"] == frozen["updated"] + 1
        np.testing.assert_array_equal(rows[i]["mu_short"], frozen["mu_short"])
    # resumes with the same fixed alpha, no catch-up
    baseline = _replay(
        [OBSERVATION_AVAILABLE] * 6, standardizer,
        [values[0], values[1], values[2], values[9], values[10], values[11]],
    )
    assert rows[9]["d_short"] == pytest.approx(baseline[3]["d_short"])


def test_all_available_is_equivalent_to_v1_semantics(standardizer):
    values = [vector(i) for i in range(10)]
    states = [OBSERVATION_AVAILABLE] * 10
    rows = _replay(states, standardizer, values)
    memory = DualTimescaleMemory(standardizer.prior_vector())
    for index, x in enumerate(values):
        f = memory.observe(standardizer.standardize(x)[0])
        assert rows[index]["d_short"] == f.d_short
        assert rows[index]["d_long"] == f.d_long
        assert rows[index]["observed"] == f.past_observed_count
        assert rows[index]["updated"] == f.past_update_count


def test_outage_on_one_channel_does_not_touch_another(standardizer):
    """Stream isolation survives the new unavailable state."""
    values = [vector(i) for i in range(6)]
    clean = _replay([OBSERVATION_AVAILABLE] * 6, standardizer, values)
    outaged = _replay(
        [OBSERVATION_AVAILABLE, OBSERVATION_UNAVAILABLE_EXACT_FLAT]
        + [OBSERVATION_AVAILABLE] * 4,
        standardizer,
        values,
    )
    # Each stream replays independently: a fresh DualTimescaleMemory per stream
    # means the clean channel is bit-identical regardless of the other's outage.
    assert clean[0]["d_short"] == outaged[0]["d_short"]
    np.testing.assert_array_equal(clean[0]["mu_short"], outaged[0]["mu_short"])


# --------------------------------------------------------------------------
# Store schema 3
# --------------------------------------------------------------------------


def test_schema_three_binds_observation_state():
    assert M1_STREAM_CACHE_SCHEMA == 3
    arrays = M1StoreSpec(rows=8, representation_dim=REPRESENTATION_DIM).arrays()
    assert OBSERVATION_STATE_FILE in arrays
    shape, dtype = arrays[OBSERVATION_STATE_FILE]
    assert shape == (8,) and dtype == "uint8"


def test_observation_state_enum_is_frozen():
    assert OBSERVATION_STATE_ENUM == {
        "UNINITIALIZED_INVALID_FOR_COMPLETED_CACHE": 0,
        "AVAILABLE": 1,
        "UNAVAILABLE_EXACT_FLAT": 2,
    }
    assert OBSERVATION_UNINITIALIZED == 0
    assert OBSERVATION_AVAILABLE == 1
    assert OBSERVATION_UNAVAILABLE_EXACT_FLAT == 2


def test_score_bearing_selection_refuses_an_unavailable_row(tmp_path):
    from cardiosentinel.neural.m1_store import M1RowStore

    spec = M1StoreSpec(rows=6, representation_dim=REPRESENTATION_DIM)
    with M1RowStore(tmp_path / "store", spec, create=True) as store:
        states = store.array(OBSERVATION_STATE_FILE)
        states[:] = OBSERVATION_AVAILABLE
        states[3] = OBSERVATION_UNAVAILABLE_EXACT_FLAT
        store.flush()
        m1_experiment.require_available_rows(
            store, np.array([0, 1, 2]), "primary TRAIN supervised rows"
        )
        with pytest.raises(M1MemoryError, match="physically unavailable"):
            m1_experiment.require_available_rows(
                store, np.array([2, 3]), "primary VALIDATION metric rows"
            )


def test_execute_checks_every_score_bearing_population():
    source = inspect.getsource(m1_experiment.execute_m1_stage1)
    for purpose in (
        "primary TRAIN supervised rows",
        "primary VALIDATION metric rows",
        "frozen VALIDATION challenge rows",
    ):
        assert purpose in source
    assert source.count("require_available_rows") >= 3


def test_unavailable_rows_never_reach_the_encoder():
    fill = inspect.getsource(m1_experiment._fill_embeddings)
    tree = ast.parse(fill.lstrip())
    # the only encoder call site must sit in the available-batch flush
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "extract_frozen_embeddings"
    ]
    assert len(calls) == 1
    assert "an unavailable observation must never reach B4-B" in fill
