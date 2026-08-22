"""The continuation evidence contract, and its separation from the canonical one.

The canonical §17 contract **requires** `policy_runs`; amendment §13.6 Layer 3
says no continuation artifact carries it, because no policy was run. The two are
mutually unsatisfiable, so the continuation has its own promoter and the
contracts stay separate. These tests prove that separation is real in both
directions rather than asserted in a docstring -- a producer and a consumer
disagreeing about a key is exactly what consumed the canonical attempt.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_continuation_gate as G
from cardiosentinel.neural import t1_continuation_labels as L
from cardiosentinel.neural import t1_continuation_persistence as CP
from cardiosentinel.neural import t1_continuation_spec as S
from cardiosentinel.neural import t1_persistence as P


def _provenance():
    return {
        "continues": {
            "predecessor_run": "t1-v1-development",
            "predecessor_digest": S.PREDECESSOR_OOF_CONTENT_SHA256,
            "governing_amendment": "T1_EXECUTION_RECOVERY_AMENDMENT_V1_1",
            "governing_amendment_sha256": S.RECOVERY_AMENDMENT_SHA256,
        },
        "consumed_evidence": [
            {"artifact": name, "sha256": digest}
            for name, digest in sorted(S.PREDECESSOR_FILE_DIGESTS.items())
        ],
    }


def _measurement(fold_index: int = 0):
    subject, policy_id, _ = S.PREDECESSOR_FOLD_SELECTIONS[fold_index]
    return {
        "fold_index": fold_index,
        "held_out_subject": subject,
        "selected_policy_id": policy_id,
        "thresholds": {
            "p_watch": 0.9,
            "s_watch": 0.8,
            "p_event": 0.99,
            "s_event": 0.95,
        },
        "primary_confusion": {"tp": 2, "fp": 1, "tn": 2, "fn": 1},
        "episode_evidence": {
            "reference_episodes": 2,
            "predicted_event_runs": 2,
            "matched_episodes": 1,
            "unmatched_predicted_runs": 1,
        },
        "onset_latency_seconds": [0.0],
        "stream_count": 1,
    }


def _evidence(fold_index: int = 0):
    return CP.build_continuation_held_out_evidence(
        _measurement(fold_index),
        authorized_git_sha="c538181eb93884f4583a8bd328e50573efbcf3df",
        fold_selection_sha256=S.PREDECESSOR_FOLD_SELECTIONS[fold_index][2],
        provenance=_provenance(),
    )


# ---------------------------------------------------------------------------
# 1. The canonical artifact REQUIRES policy_runs
# ---------------------------------------------------------------------------


def test_canonical_contract_requires_policy_runs():
    assert "policy_runs" in P.HELD_OUT_EVALUATION_REQUIRED_FIELDS


def test_canonical_promoter_refuses_a_payload_without_policy_runs(tmp_path):
    """The reason the continuation cannot reuse the canonical path."""
    payload = {
        field: "x"
        for field in P.HELD_OUT_EVALUATION_REQUIRED_FIELDS
        if field != "policy_runs"
    }

    class _Claimed:
        held_out_dir = tmp_path / "held_out_traces"
        runtime = None

    with pytest.raises(P.T1PersistenceError, match="policy_runs"):
        P.promote_held_out_evaluation(_Claimed(), 0, payload)
    assert not (tmp_path / "held_out_traces").exists(), "a refusal left a directory"


# ---------------------------------------------------------------------------
# 2. The continuation artifact REFUSES policy_runs
# ---------------------------------------------------------------------------


def test_continuation_contract_excludes_policy_runs():
    assert "policy_runs" not in CP.CONTINUATION_HELD_OUT_REQUIRED_FIELDS
    assert "policy_runs" in S.FORBIDDEN_CONTINUATION_FIELDS


@pytest.mark.parametrize("value", [0, 1, None, "0"])
def test_continuation_evidence_refuses_policy_runs_at_any_value(value):
    """§13.6 Layer 3 requires the key's absence, not a zero."""
    payload = dict(_evidence())
    payload["policy_runs"] = value
    with pytest.raises(CP.T1ContinuationPersistenceError, match="policy_runs"):
        CP.validate_continuation_held_out_evidence(payload)


def test_promotion_refuses_policy_runs_and_writes_nothing(tmp_path):
    payload = dict(_evidence())
    payload["policy_runs"] = 0
    with pytest.raises(CP.T1ContinuationPersistenceError, match="policy_runs"):
        CP.promote_continuation_held_out_evaluation(tmp_path, 0, payload)
    assert list(tmp_path.iterdir()) == []


def test_a_clean_continuation_artifact_carries_no_policy_runs_anywhere():
    assert "policy_runs" not in json.dumps(_evidence())


# ---------------------------------------------------------------------------
# 3. The continuation artifact REQUIRES continues / consumed_evidence
# ---------------------------------------------------------------------------


def test_continuation_contract_requires_both_provenance_blocks():
    assert "continues" in CP.CONTINUATION_HELD_OUT_REQUIRED_FIELDS
    assert "consumed_evidence" in CP.CONTINUATION_HELD_OUT_REQUIRED_FIELDS


@pytest.mark.parametrize("block", ["continues", "consumed_evidence"])
def test_evidence_refuses_a_missing_provenance_block(block):
    payload = dict(_evidence())
    del payload[block]
    with pytest.raises(CP.T1ContinuationPersistenceError, match="missing"):
        CP.validate_continuation_held_out_evidence(payload)


@pytest.mark.parametrize("key", sorted(CP.CONTINUES_REQUIRED_KEYS))
def test_continues_block_refuses_a_missing_key(key):
    payload = dict(_evidence())
    payload["continues"] = {k: v for k, v in payload["continues"].items() if k != key}
    with pytest.raises(CP.T1ContinuationPersistenceError, match="continues"):
        CP.validate_continuation_held_out_evidence(payload)


def test_consumed_evidence_refuses_empty_or_malformed():
    payload = dict(_evidence())
    payload["consumed_evidence"] = []
    with pytest.raises(CP.T1ContinuationPersistenceError, match="non-empty"):
        CP.validate_continuation_held_out_evidence(payload)

    payload["consumed_evidence"] = [{"artifact": "T1_PREFLIGHT.json"}]
    with pytest.raises(CP.T1ContinuationPersistenceError, match="Malformed"):
        CP.validate_continuation_held_out_evidence(payload)


def test_the_provenance_blocks_name_the_real_predecessor():
    evidence = _evidence()
    assert evidence["continues"]["predecessor_run"] == "t1-v1-development"
    assert evidence["continues"]["predecessor_digest"] == (
        "cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f"
    )
    assert len(evidence["consumed_evidence"]) == 8
    assert evidence["is_continuation_artifact"] is True
    assert evidence["generated_during_canonical_execution"] is False
    assert evidence["test_accessed"] is False


def test_evidence_refuses_a_falsified_identity_flag():
    for field, bad in (
        ("is_continuation_artifact", False),
        ("generated_during_canonical_execution", True),
        ("test_accessed", True),
    ):
        payload = dict(_evidence())
        payload[field] = bad
        with pytest.raises(CP.T1ContinuationPersistenceError):
            CP.validate_continuation_held_out_evidence(payload)


# ---------------------------------------------------------------------------
# 4. No policy execution path is reachable
# ---------------------------------------------------------------------------

CONTINUATION_MODULES = (
    "cardiosentinel.neural.t1_continuation_spec",
    "cardiosentinel.neural.t1_continuation_predecessor",
    "cardiosentinel.neural.t1_continuation_gate",
    "cardiosentinel.neural.t1_continuation_measurement",
    "cardiosentinel.neural.t1_continuation_attestation",
    "cardiosentinel.neural.t1_continuation_persistence",
)


def test_the_gate_passes_on_the_persistence_module_too():
    proof = G.prove_negative_capability(CONTINUATION_MODULES)
    assert proof["counters"] == dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)


def test_no_continuation_module_reaches_a_policy_execution_path():
    import importlib

    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(
            Path(importlib.import_module(module_name).__file__).read_text(
                encoding="utf-8"
            )
        )
        modules, names = G._bound_names(tree)
        for forbidden in G.FORBIDDEN_MODULES:
            assert forbidden not in modules, f"{module_name} imports {forbidden}"
        bound = {n for group in names.values() for n in group}
        for group in G.FORBIDDEN_IMPORTS.values():
            assert not (bound & set(group)), f"{module_name} binds {bound & set(group)}"


def test_the_fold_evaluator_stays_unloaded_by_the_continuation_graph():
    """The strongest runtime proof, and the one the label authority must not cost."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import cardiosentinel.neural.t1_continuation_persistence\n"
        "import cardiosentinel.neural.t1_continuation_measurement\n"
        "import cardiosentinel.neural.t1_continuation_labels\n"
        "from cardiosentinel.neural.t1_continuation_gate import NEVER_LOADED_MODULES\n"
        "bad = [m for m in NEVER_LOADED_MODULES if m in sys.modules]\n"
        "print('DIRTY' if bad else 'CLEAN', bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert out.stdout.startswith("CLEAN"), out.stdout + out.stderr


def test_importing_the_label_module_does_not_pull_in_the_development_run():
    """Deferred authority import: a caller wanting no labels pays nothing."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import cardiosentinel.neural.t1_continuation_labels\n"
        "print('LOADED' if 'cardiosentinel.neural.t1_development_run' in sys.modules"
        " else 'DEFERRED')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert out.stdout.strip() == "DEFERRED", out.stdout + out.stderr


def test_persistence_writes_only_into_the_continuation_root(tmp_path):
    payload = _evidence()
    with pytest.raises(CP.T1ContinuationPersistenceError, match="outside"):
        CP.promote_continuation_held_out_evaluation(tmp_path, 0, payload)


def test_persistence_refuses_to_address_the_consumed_attempt():
    payload = _evidence()
    with pytest.raises(CP.T1ContinuationPersistenceError):
        CP.promote_continuation_held_out_evaluation(S.CONSUMED_ATTEMPT_DIR, 0, payload)


def test_label_access_is_bounded_to_the_folds_promoted_subject():
    with pytest.raises(L.T1ContinuationLabelError, match="not one of the twelve"):
        L.held_out_labels_for_fold(object(), 99)


def test_label_members_exclude_label_and_target_family():
    assert L.CONTINUATION_LABEL_MEMBERS == ("primary_mask", "primary_positive")
    assert "label" not in L.CONTINUATION_LABEL_MEMBERS
    assert "target_family" not in L.CONTINUATION_LABEL_MEMBERS


def test_labels_must_cover_every_trace_row():
    labels = {"primary_mask": {"a": True}, "primary_positive": {"a": False}}
    L.require_labels_cover_trace(labels, ["a"], 0)
    with pytest.raises(L.T1ContinuationLabelError, match="no label"):
        L.require_labels_cover_trace(labels, ["a", "b"], 0)


def test_no_continuation_run_directory_was_created():
    assert not S.CONTINUATION_RUN_ROOT.exists()
