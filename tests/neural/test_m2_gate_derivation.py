"""Canonical-runtime reproduction of the M2-v1 TRAIN-only receipt.

Provenance checks run unconditionally. Full numerical-reproduction checks
that need the local TRAIN caches (not committed to git -- see
`.gitignore`'s `/cardiosentinel-features/` and `/cardiosentinel-runs/`) are
skipped when that data is absent, matching this repo's existing convention
for local-data-dependent tests (see `test_m1_memory_scaling.py`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.data.provenance import sha256_file
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural import m2_gate_derivation as D
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

RECEIPT = json.loads(
    (REPOSITORY_ROOT / "docs" / "M2_GATE_DERIVATION_RECEIPT_V1.json").read_text()
)
SUPERSEDED_RECEIPT_SHA256 = (
    "3befd05dc7e9c51ddfed99078d3020375fd610b328d19e64fc7ee3cc745f398e"
)
SUPERSEDED_DEPENDENCY_DIGEST = (
    "78e838d2d41a0239f16dbfbaabdddc7efeaffac391ca13a8bbf1475c080cdc25"
)
CANONICAL_DEPENDENCY_DIGEST = (
    "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
)

# Recorded independently of the receipt/module, as of the pre-canonicalization
# freeze, so a future accidental edit to either is still caught.
PRE_CANONICAL_G3_BOUNDS = {
    "flatline_fraction": 0.4853941576630652,
    "repeated_value_fraction": 0.4853941576630652,
    "derivative_outlier_fraction": 0.12404961984793918,
    "high_frequency_power_ratio": 0.026922298961394597,
    "powerline_ratio_50hz": 0.0017282393761769012,
    "powerline_ratio_60hz": 0.0012844103306429878,
}
PRE_CANONICAL_NORMAL_EVIDENCE_THRESHOLD = 0.0002997174742631614


def test_canonical_receipt_dependency_digest_is_canonical():
    assert RECEIPT["environment"]["dependency_digest"] == CANONICAL_DEPENDENCY_DIGEST
    assert (
        G.M2_GATE_RECEIPT_SHA256
        == "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )


def test_canonical_receipt_records_the_superseded_identity():
    canon = RECEIPT["canonicalization"]
    assert canon["superseded_receipt_sha256"] == SUPERSEDED_RECEIPT_SHA256
    assert canon["superseded_receipt_dependency_digest"] == SUPERSEDED_DEPENDENCY_DIGEST
    assert canon["canonical_dependency_digest"] == CANONICAL_DEPENDENCY_DIGEST
    assert canon["scientific_values_changed"] is False
    assert canon["validation_accessed"] is False
    assert canon["test_accessed"] is False
    assert canon["reproduction_field_level_checks_failed"] == 0
    assert canon["reproduction_field_level_checks_performed"] > 0


def test_pre_canonical_scientific_constants_are_unchanged():
    for column, bound in PRE_CANONICAL_G3_BOUNDS.items():
        assert RECEIPT["g3_sqi"]["frozen_upper_bounds_q99"][column] == bound
        assert G.G3_UPPER_BOUNDS[column] == bound
    assert (
        RECEIPT["g4_normal_evidence"]["normal_evidence_threshold"]
        == PRE_CANONICAL_NORMAL_EVIDENCE_THRESHOLD
    )
    assert G.NORMAL_EVIDENCE_THRESHOLD == PRE_CANONICAL_NORMAL_EVIDENCE_THRESHOLD


def test_protocol_and_receipt_sha_bindings_validate():
    assert G.validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert G.validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256


def test_retained_arm_and_core_arms_unchanged():
    assert G.M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    assert G.M2_CORE_ARMS == ("M2-0", "M2-G")
    assert G.M2_ROLLBACK_IN_CORE is False


def test_no_validation_or_test_access_recorded():
    assert RECEIPT["validation_accessed"] is False
    assert RECEIPT["test_accessed"] is False
    assert RECEIPT["train_only_sanity"]["validation_accessed"] is False
    assert RECEIPT["train_only_sanity"]["test_accessed"] is False


def test_binding_module_still_has_no_execution_or_mutation_path():
    tree = ast.parse(Path(G.__file__).read_text())
    forbidden = {"write_json_atomic", "save", "unlink", "rmtree", "rename", "open"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name


def test_runtime_sentinel_remains_design_only_and_untouched():
    path = REPOSITORY_ROOT / "docs" / "RUNTIME_INTEGRITY_SENTINEL_V1.md"
    text = path.read_text()
    assert "STATUS: DESIGN ONLY" in text
    assert "NOT IMPLEMENTED" in text
    assert sha256_file(path) == (
        "cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075"
    )


def test_derivation_verifier_uses_no_tolerance_comparison():
    """`compare_to_frozen` must use strict equality only -- never a tolerance."""
    text = Path(D.__file__).read_text()
    assert "isclose" not in text
    assert "rtol" not in text
    assert "atol" not in text
    tree = ast.parse(text)
    forbidden_writes = {"write_json_atomic", "unlink", "rmtree"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden_writes, name


LOCAL_DATA_AVAILABLE = (
    D.DEFAULT_STREAM_CACHE_ROOT.exists() and D.DEFAULT_M1_RUN_ROOT.exists()
)
LOCAL_DATA_SKIP_REASON = (
    "requires the local M1-v2 stream cache and M1L checkpoint "
    "(cardiosentinel-features/, cardiosentinel-runs/ -- gitignored, not "
    "available in CI)"
)


@pytest.mark.skipif(not LOCAL_DATA_AVAILABLE, reason=LOCAL_DATA_SKIP_REASON)
def test_canonical_reproduction_is_bit_exact_and_deterministic():
    report = D.run_derivation()
    assert report["identity_bindings"]["all_bound"], report["identity_bindings"][
        "mismatches"
    ]
    assert report["comparison"]["reproduced"], report["comparison"]["mismatches"]
    computed_g4 = report["computed"]["g4_normal_evidence"]
    assert (
        computed_g4["normal_evidence_threshold"]
        == PRE_CANONICAL_NORMAL_EVIDENCE_THRESHOLD
    )
    assert (
        computed_g4["descriptive_distribution"]["min"]
        == RECEIPT["g4_normal_evidence"]["descriptive_distribution"]["min"]
    )
    second = D.run_derivation()
    assert second["computed"] == report["computed"]


@pytest.mark.skipif(not LOCAL_DATA_AVAILABLE, reason=LOCAL_DATA_SKIP_REASON)
def test_arithmetic_path_fractions_match_receipt_exactly():
    report = D.run_derivation()
    computed = report["computed"]
    assert (
        computed["g3_sqi"]["combined_train_rejection_fraction"]
        == RECEIPT["g3_sqi"]["combined_train_rejection_fraction"]
    )
    for key in ("sqi", "normal_evidence", "morphology", "refractory"):
        assert (
            computed["train_only_sanity"]["refusal_fractions"][key]
            == RECEIPT["train_only_sanity"]["refusal_fractions"][key]
        )
