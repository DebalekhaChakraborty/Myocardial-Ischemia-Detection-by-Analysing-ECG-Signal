"""The frozen M2-v1 gate: identities, semantics, and no execution path."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural.m2_gate import (
    m2_gate_identity,
    validate_m2_gate_receipt,
    validate_m2_protocol,
)


def test_frozen_documents_validate():
    assert validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256
    assert m2_gate_identity()["m2_protocol_sha256"] == G.M2_PROTOCOL_SHA256


def test_g3_column_identity_and_order():
    assert G.G3_SQI_COLUMNS == (
        "flatline_fraction",
        "repeated_value_fraction",
        "derivative_outlier_fraction",
        "high_frequency_power_ratio",
        "powerline_ratio_50hz",
        "powerline_ratio_60hz",
    )
    assert set(G.G3_UPPER_BOUNDS) == set(G.G3_SQI_COLUMNS)
    # amplitude/rhythm features vary with physiology and are excluded on purpose
    for excluded in G.G3_EXCLUDED_COLUMNS:
        assert excluded not in G.G3_SQI_COLUMNS
    assert G.G3_FINITE_PRECONDITION_COLUMN == "finite_sample_fraction"


def test_g3_quantile_rule_is_q99_linear():
    assert G.G3_QUANTILE == 0.99
    assert G.G3_QUANTILE_METHOD == "linear"
    receipt = json.loads(Path("docs/M2_GATE_DERIVATION_RECEIPT_V1.json").read_text())
    g3 = receipt["g3_sqi"]
    assert g3["quantile_rule"] == "numpy.quantile(values, 0.99, method='linear')"
    assert g3["population"]["partition"] == "train"
    assert g3["population"]["label_filtering"] is False
    assert g3["population"]["validation_rows_used"] is False
    assert g3["population"]["test_rows_used"] is False
    for column, bound in G.G3_UPPER_BOUNDS.items():
        assert g3["frozen_upper_bounds_q99"][column] == pytest.approx(bound, abs=0.0)


def test_g3_records_its_redundant_columns():
    """Six declared columns, five independent constraints -- stated, not hidden."""
    assert G.G3_INDEPENDENT_CONSTRAINTS == 5
    assert G.G3_UPPER_BOUNDS["flatline_fraction"] == (
        G.G3_UPPER_BOUNDS["repeated_value_fraction"]
    )


def test_g4_rule_is_train_background_negative_median():
    receipt = json.loads(Path("docs/M2_GATE_DERIVATION_RECEIPT_V1.json").read_text())
    g4 = receipt["g4_normal_evidence"]
    assert g4["derivation_rule"] == (
        "numpy.quantile(M1L_score_on_PRIMARY_TRAIN_background_negative, 0.50, "
        "method='linear')"
    )
    assert G.G4_DERIVATION_QUANTILE == 0.50
    assert G.G4_QUANTILE_METHOD == "linear"
    assert g4["population"]["scope"] == "primary_train_background_negative_only"
    assert g4["population"]["model_retrained"] is False
    assert g4["population"]["new_memory_replay"] is False
    assert g4["normal_evidence_threshold"] == G.NORMAL_EVIDENCE_THRESHOLD


def test_normal_evidence_threshold_is_not_the_classification_threshold():
    assert G.NORMAL_EVIDENCE_THRESHOLD != G.M1L_CLASSIFICATION_THRESHOLD
    assert G.NORMAL_EVIDENCE_THRESHOLD < G.M1L_CLASSIFICATION_THRESHOLD


def test_score_is_never_called_a_probability():
    assert "not a probability" in G.G4_SCORE_SEMANTICS
    text = Path("docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md").read_text()
    assert "DETERMINISTIC NORMAL-EVIDENCE MARGIN" in text
    # the phrase may appear only where the protocol forbids its use
    for line in text.splitlines():
        if "low uncertainty gate" in line:
            assert "must not use the phrase" in line, line


def test_g6_morphology_is_included_as_computability_only():
    assert G.G6_MORPHOLOGY_INCLUDED is True
    assert G.G6_COLUMN == "morphology_valid"
    assert G.G6_NAME == "morphology computability admission"
    # morphology failure must not by itself start a refractory
    assert G.G6_ARMS_REFRACTORY is False


def test_refractory_is_sixty_seconds_of_real_time_and_re_armable():
    assert G.REFRACTORY_DURATION_SECONDS == 60.0
    assert G.REFRACTORY_IS_RE_ARMABLE is True
    assert G.REFRACTORY_COUNTED_IN_UPDATES is False
    assert "NOT NORMAL/WATCH/EVENT/RECOVERY" in G.REFRACTORY_SEMANTICS
    receipt = json.loads(Path("docs/M2_GATE_DERIVATION_RECEIPT_V1.json").read_text())
    g5 = receipt["g5_refractory"]
    assert g5["implemented_as_update_count"] is False
    assert "(start_sample+2500)/250.0" in g5["timing"].replace(" ", "")


def test_rollback_is_excluded_from_the_claim_bearing_core():
    assert G.M2_CORE_ARMS == ("M2-0", "M2-G")
    assert G.M2_ROLLBACK_IN_CORE is False
    text = Path("docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md").read_text()
    assert "There is no M2-GR claim-bearing arm" in text
    assert "MECHANISM EVIDENCE\nONLY" in text or "MECHANISM EVIDENCE" in text


def test_no_validation_or_test_was_used_in_any_derivation():
    assert G.M2_VALIDATION_ACCESSED_IN_DERIVATION is False
    assert G.M2_TEST_ACCESSED is False
    receipt = json.loads(Path("docs/M2_GATE_DERIVATION_RECEIPT_V1.json").read_text())
    assert receipt["validation_accessed"] is False
    assert receipt["test_accessed"] is False
    assert receipt["train_only_sanity"]["validation_accessed"] is False
    assert receipt["train_only_sanity"]["test_accessed"] is False


def test_gate_does_not_collapse_adaptation():
    """The exit rule forbids a trivial never-update policy."""
    receipt = json.loads(Path("docs/M2_GATE_DERIVATION_RECEIPT_V1.json").read_text())
    sanity = receipt["train_only_sanity"]
    assert sanity["final_m2g_update_fraction_after_causal_refractory"] > 0.0
    assert sanity["per_stream_update_fraction"]["median"] > 0.0


def test_binding_module_has_no_execution_or_mutation_path():
    tree = ast.parse(Path(G.__file__).read_text())
    forbidden = {"write_json_atomic", "save", "unlink", "rmtree", "rename", "open"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name
    imported = {
        alias.name
        for imp in ast.walk(tree)
        if isinstance(imp, ast.ImportFrom)
        for alias in imp.names
    }
    for banned in ("train_m1_arm", "execute_m1_stage1", "materialize_stream_store"):
        assert banned not in imported
    assert "sealed_test" not in Path(G.__file__).read_text()


def test_retained_arm_is_unchanged_from_the_m1_decision():
    from cardiosentinel.neural.m1_selection import M1_RETAINED_EXPERIMENT_ID

    assert G.M2_RETAINED_EXPERIMENT_ID == M1_RETAINED_EXPERIMENT_ID


def test_protocol_is_frozen_not_proposed():
    text = Path("docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md").read_text()
    assert "STATUS: FROZEN SCIENTIFIC PROTOCOL" in text
    assert "PROPOSED — HUMAN REVIEW REQUIRED" not in text
    assert "**OPEN**" not in text


def test_quantile_rules_reproduce_on_a_known_vector():
    """The declared rules are the numpy ones, not a paraphrase."""
    values = np.arange(101, dtype=np.float64)
    assert np.quantile(values, G.G3_QUANTILE, method=G.G3_QUANTILE_METHOD) == 99.0
    assert np.quantile(values, G.G4_DERIVATION_QUANTILE,
                       method=G.G4_QUANTILE_METHOD) == 50.0
