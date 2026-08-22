"""The continuation executor: built, gated, and unable to run.

These tests prove the engine exists and refuses. They never execute a
continuation, never create a run directory, and never open a real held-out
label. The continuation is authorized once; a post-claim failure consumes it and
no second one is predeclared, so every gate that can be green before it starts
is green before it starts.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t1_continuation_gate as G
from cardiosentinel.neural import t1_continuation_results as R
from cardiosentinel.neural import t1_continuation_runner as RUN
from cardiosentinel.neural import t1_continuation_spec as S


def _provenance():
    return {
        "continues": {
            "predecessor_run": "t1-v1-development",
            "predecessor_digest": S.PREDECESSOR_OOF_CONTENT_SHA256,
            "governing_amendment": "T1_EXECUTION_RECOVERY_AMENDMENT_V1_1",
            "governing_amendment_sha256": S.RECOVERY_AMENDMENT_SHA256,
        },
        "consumed_evidence": [{"artifact": "T1_PREFLIGHT.json", "sha256": "0" * 64}],
    }


def _measurements(count: int = 12):
    out = {}
    for index in range(count):
        subject, policy_id, _ = S.PREDECESSOR_FOLD_SELECTIONS[index]
        out[index] = {
            "fold_index": index,
            "held_out_subject": subject,
            "selected_policy_id": policy_id,
            "thresholds": {
                "p_watch": 0.9,
                "s_watch": 0.8,
                "p_event": 0.99,
                "s_event": 0.95,
            },
            "primary_confusion": {"tp": 3 + index, "fp": 2, "tn": 40, "fn": 5},
            "episode_evidence": {
                "reference_episodes": 10 + index,
                "predicted_event_runs": 8,
                "matched_episodes": 4,
                "unmatched_predicted_runs": 4,
            },
            "onset_latency_seconds": [5.0, 10.0],
            "stream_count": 2,
        }
    return out


# ---------------------------------------------------------------------------
# Runtime: authorization refusal
# ---------------------------------------------------------------------------


def test_the_repository_is_armed_and_the_session_is_not():
    """Two different questions, deliberately kept apart.

    The repository carries an explicit human authorization. This test process
    does not, because the session fixture disarms it -- pytest is not the
    operator, and a suite that could execute the continuation would be one
    stray import away from consuming it.
    """
    assert committed_authorization_value() is True, "repository is not armed"
    assert S.T1_CONTINUATION_AUTHORIZED is False, "the test session is armed"


def committed_authorization_value() -> bool:
    """The flag as **committed**, not as this test process sees it.

    The session fixture forces the runtime value False so pytest can never
    execute the continuation. That makes the runtime value useless for asking
    "is the repository armed?", so this reads the assignment out of the source.
    """
    import ast
    from pathlib import Path as _P

    from cardiosentinel.neural import t1_continuation_spec as _S

    tree = ast.parse(_P(_S.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == (
            "T1_CONTINUATION_AUTHORIZED"
        ):
            return bool(ast.literal_eval(node.value))
    raise AssertionError("T1_CONTINUATION_AUTHORIZED is not assigned in the spec")


def test_preflight_refuses_at_the_first_stage_and_touches_nothing():
    with pytest.raises(S.T1ContinuationPermissionError):
        RUN.preflight()
    assert not S.CONTINUATION_RUN_ROOT.exists()


def test_execute_refuses_before_resolving_anything(tmp_path):
    with pytest.raises(S.T1ContinuationPermissionError):
        RUN.execute_continuation(tmp_path)
    assert not S.CONTINUATION_RUN_ROOT.exists()
    assert list(tmp_path.iterdir()) == []


def test_authorization_is_the_very_first_stage():
    assert RUN.CONTINUATION_STAGE_ORDER[0] == RUN.STAGE_AUTHORIZE
    assert RUN.CONTINUATION_STAGE_ORDER.index(RUN.STAGE_CLAIM) > (
        RUN.CONTINUATION_STAGE_ORDER.index(RUN.STAGE_VERIFY_PREDECESSOR)
    )


def test_every_writing_stage_comes_after_every_read_only_gate():
    order = RUN.CONTINUATION_STAGE_ORDER
    first_write = min(order.index(s) for s in RUN.WRITING_STAGES)
    for stage in (
        RUN.STAGE_AUTHORIZE,
        RUN.STAGE_IDENTITY,
        RUN.STAGE_VERIFY_PREDECESSOR,
        RUN.STAGE_PROVE_CAPABILITY,
        RUN.STAGE_CONSUME_TRACE,
        RUN.STAGE_MATCH_SELECTIONS,
    ):
        assert order.index(stage) < first_write, f"{stage} writes before it gates"


def test_a_stage_cannot_be_re_entered():
    record = RUN.ContinuationRunRecord()
    record.enter(RUN.STAGE_AUTHORIZE)
    with pytest.raises(RUN.T1ContinuationRunError, match="already entered"):
        record.enter(RUN.STAGE_AUTHORIZE)


def test_there_is_no_retry_or_resume_path():
    capability = RUN.continuation_runner_capability()
    assert capability["automatic_retry_permitted"] is False
    assert capability["resume_supported"] is False
    assert capability["read_only_before_claim"] is True
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for banned in ("retry", "resume", "reset", "force", "overwrite"):
        assert not any(banned in n.lower() for n in names), banned


# ---------------------------------------------------------------------------
# Structural: no forbidden execution path
# ---------------------------------------------------------------------------


def test_the_runner_passes_the_negative_capability_gate():
    proof = G.prove_negative_capability(RUN.CONTINUATION_PROVEN_MODULES)
    assert proof["counters"] == dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)


def test_the_runner_imports_no_forbidden_module():
    for module_name in RUN.CONTINUATION_PROVEN_MODULES:
        tree = ast.parse(
            Path(importlib.import_module(module_name).__file__).read_text("utf-8")
        )
        modules, names = G._bound_names(tree)
        for forbidden in G.FORBIDDEN_MODULES:
            assert forbidden not in modules, f"{module_name} imports {forbidden}"
        bound = {n for group in names.values() for n in group}
        for group in G.FORBIDDEN_IMPORTS.values():
            assert not (bound & set(group)), f"{module_name} binds {bound & set(group)}"


def test_no_canonical_driver_or_fold_evaluator_dependency():
    tree = ast.parse(Path(RUN.__file__).read_text("utf-8"))
    modules, _ = G._bound_names(tree)
    for banned in (
        "cardiosentinel.neural.t1_canonical_driver",
        "cardiosentinel.neural.t1_fold_evaluator",
        "cardiosentinel.neural.t1_composition",
        "cardiosentinel.neural.t1_assembly",
    ):
        assert banned not in modules


def test_no_model_checkpoint_is_ever_loaded():
    for module_name in RUN.CONTINUATION_PROVEN_MODULES:
        tree = ast.parse(
            Path(importlib.import_module(module_name).__file__).read_text("utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None)
                )
                assert name not in {"load", "load_state_dict", "torch_load"}, (
                    f"{module_name} loads a checkpoint"
                )


def test_the_runner_declares_the_six_artifacts():
    assert R.CONTINUATION_RESULT_ARTIFACTS == (
        "T1_OOF_RESULT.json",
        "T1_SUBJECT_EVIDENCE.json",
        "T1_BOOTSTRAP.json",
        "T1_CHALLENGE_EVIDENCE.json",
        "T1_FINAL_CONFIGURATION.json",
        "T1_EXPERIMENT_LOCK.json",
    )


# ---------------------------------------------------------------------------
# Frozen helpers: re-implemented, proven equal
# ---------------------------------------------------------------------------


def test_bootstrap_indices_match_the_frozen_design_exactly():
    from cardiosentinel.neural.t1_development_run import (
        subject_bootstrap_indices as canonical,
    )

    for n in (1, 2, 5, 12, 20):
        assert np.array_equal(R.subject_bootstrap_indices(n), canonical(n))


def test_window_mcc_matches_the_frozen_implementation_exactly():
    from cardiosentinel.neural.t1_development_run import window_mcc as canonical

    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(1, 40))
        p = rng.random(n) < 0.5
        a = rng.random(n) < 0.5
        assert R.window_mcc(p, a) == canonical(p, a)


def test_bootstrap_design_constants_match_the_frozen_spec():
    from cardiosentinel.neural import t1_execution_spec as SPEC

    assert R.T1_BOOTSTRAP_REPLICATES == SPEC.T1_BOOTSTRAP_REPLICATES == 1000
    assert R.T1_BOOTSTRAP_SEED == SPEC.T1_BOOTSTRAP_SEED == 2026
    assert R.T1_BOOTSTRAP_UNIT == SPEC.T1_BOOTSTRAP_UNIT == "subject"
    assert (
        R.T1_BOOTSTRAP_RESELECTS_POLICY is SPEC.T1_BOOTSTRAP_RESELECTS_POLICY is False
    )


def test_the_confusion_translation_matches_the_canonical_mapping():
    """The exact pair that consumed the canonical attempt at stage 24."""
    from cardiosentinel.neural.t1_composition import PRIMARY_CONFUSION_KEYS as canonical

    assert R.PRIMARY_CONFUSION_KEYS == canonical


def test_a_missing_confusion_count_is_refused_not_defaulted():
    with pytest.raises(R.T1ContinuationResultError, match="missing"):
        R.translate_confusion({"tp": 1, "fp": 2, "tn": 3})


def test_translation_produces_the_long_names_assembly_reads():
    out = R.translate_confusion({"tp": 1, "fp": 2, "tn": 3, "fn": 4})
    assert out == {
        "true_positive": 1,
        "false_positive": 2,
        "true_negative": 3,
        "false_negative": 4,
    }


# ---------------------------------------------------------------------------
# Evidence: the six artifacts
# ---------------------------------------------------------------------------


def test_oof_result_pools_and_carries_provenance():
    result = R.build_oof_result(_measurements(), provenance=_provenance())
    assert result["artifact_class"] == "t1_v1_continuation_oof_result"
    assert result["fold_count"] == 12
    assert set(result["primary_confusion"]) == {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    }
    assert result["primary_confusion"]["true_positive"] == sum(3 + i for i in range(12))
    assert len(result["fold_summaries"]) == 12
    assert result["continues"]["predecessor_run"] == "t1-v1-development"
    assert result["consumed_evidence"]


def test_subject_evidence_is_one_row_per_subject():
    evidence = R.build_subject_evidence(_measurements(), provenance=_provenance())
    assert evidence["subject_count"] == 12
    assert len(evidence["subjects"]) == 12
    assert "continues" in evidence and "consumed_evidence" in evidence


def test_subject_evidence_refuses_a_broken_bijection():
    measurements = _measurements(2)
    measurements[1]["held_out_subject"] = measurements[0]["held_out_subject"]
    with pytest.raises(R.T1ContinuationResultError, match="bijection"):
        R.build_subject_evidence(measurements, provenance=_provenance())


def test_bootstrap_is_the_frozen_design():
    evidence = R.build_subject_evidence(_measurements(), provenance=_provenance())
    bootstrap = R.build_bootstrap(evidence, provenance=_provenance())
    assert bootstrap["replicates"] == 1000
    assert bootstrap["seed"] == 2026
    assert bootstrap["unit"] == "subject"
    assert bootstrap["policy_reselected_inside_bootstrap"] is False
    assert bootstrap["defined_replicates"] + bootstrap["undefined_replicates"] == 1000
    assert bootstrap["percentile_2_5"] <= bootstrap["percentile_97_5"]


def test_bootstrap_preserves_undefined_replicates_rather_than_zeroing():
    measurements = _measurements(2)
    for m in measurements.values():
        m["episode_evidence"] = dict.fromkeys(R.EPISODE_EVIDENCE_KEYS, 0)
    evidence = R.build_subject_evidence(measurements, provenance=_provenance())
    assert all(
        v[R.BOOTSTRAP_SUBJECT_STATISTIC] is None for v in evidence["subjects"].values()
    )
    bootstrap = R.build_bootstrap(evidence, provenance=_provenance())
    assert bootstrap["undefined_replicates"] == 1000
    assert bootstrap["percentile_2_5"] is None


def test_challenge_records_an_absent_join_as_absent():
    evidence = R.build_subject_evidence(_measurements(), provenance=_provenance())
    challenge = R.build_challenge_evidence(evidence, provenance=_provenance())
    assert challenge["join_performed"] is False
    assert challenge["strata"] == {}
    assert challenge["selection_performed_on_challenge_evidence"] is False


def test_final_configuration_reads_thresholds_from_promoted_selections():
    config = R.build_final_configuration(
        provenance=_provenance(), upstream_identities={"m2": "x"}
    )
    assert config["fold_count"] == 12
    assert config["thresholds_generated_here"] is False
    assert config["selection_performed_here"] is False
    assert config["thresholds_source"] == "promoted_fold_selection_artifacts"
    assert config["state_trace_array_sha256"] == S.PREDECESSOR_OOF_ARRAY_SHA256


def test_experiment_lock_closes_provenance():
    attestation = dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)
    lock = R.build_experiment_lock(
        provenance=_provenance(),
        attestation=attestation,
        promoted_digests={"T1_OOF_RESULT.json": "a" * 64},
    )
    assert lock["governing_amendment_sha256"] == (
        "d3ea7734c93be8f59796e03e8c0210778716327f7adc033cb2d3dcfff7f92c96"
    )
    assert lock["attempts_authorized"] == 1
    assert lock["automatic_retry_permitted"] is False
    assert lock["promoted_artifact_count"] == 1


def test_experiment_lock_refuses_a_nonzero_counter():
    attestation = dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)
    attestation["fold_evaluations"] = 1
    with pytest.raises(R.T1ContinuationResultError, match="fold_evaluations"):
        R.build_experiment_lock(
            provenance=_provenance(), attestation=attestation, promoted_digests={}
        )


@pytest.mark.parametrize(
    "builder",
    ["oof_result", "subject_evidence", "bootstrap", "challenge", "final_configuration"],
)
def test_no_run_level_artifact_carries_policy_runs(builder):
    measurements = _measurements()
    evidence = R.build_subject_evidence(measurements, provenance=_provenance())
    built = {
        "oof_result": lambda: R.build_oof_result(
            measurements, provenance=_provenance()
        ),
        "subject_evidence": lambda: evidence,
        "bootstrap": lambda: R.build_bootstrap(evidence, provenance=_provenance()),
        "challenge": lambda: R.build_challenge_evidence(
            evidence, provenance=_provenance()
        ),
        "final_configuration": lambda: R.build_final_configuration(
            provenance=_provenance(), upstream_identities={}
        ),
    }[builder]()
    assert "policy_runs" not in json.dumps(built)
    assert built["is_continuation_artifact"] is True
    assert built["test_accessed"] is False


@pytest.mark.parametrize("block", ["continues", "consumed_evidence"])
def test_every_run_level_artifact_carries_both_provenance_blocks(block):
    measurements = _measurements()
    evidence = R.build_subject_evidence(measurements, provenance=_provenance())
    for built in (
        R.build_oof_result(measurements, provenance=_provenance()),
        evidence,
        R.build_bootstrap(evidence, provenance=_provenance()),
        R.build_challenge_evidence(evidence, provenance=_provenance()),
        R.build_final_configuration(provenance=_provenance(), upstream_identities={}),
    ):
        assert block in built, f"{built['artifact_class']} lacks {block}"


# ---------------------------------------------------------------------------
# Nothing was executed
# ---------------------------------------------------------------------------


def test_no_continuation_directory_was_created():
    assert not S.CONTINUATION_RUN_ROOT.exists()


def test_the_consumed_attempt_is_never_addressed_for_writing():
    tree = ast.parse(Path(RUN.__file__).read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name in {"mkdir", "write_json_atomic"}:
                segment = ast.unparse(node)
                assert "CONSUMED" not in segment.upper()


# ---------------------------------------------------------------------------
# The refusal, proven exhaustively
# ---------------------------------------------------------------------------


def test_continuation_refuses_before_claim_when_not_authorized(tmp_path, monkeypatch):
    """The whole gate, in one test: refused, and nothing at all happened.

    Directory, claim, artifacts, label access and TEST access are five separate
    things that must not happen, and a test asserting only the first would pass
    on an implementation that opened every held-out label before refusing. So
    each is proven independently: the label authority and the promoters are
    replaced with sentinels that raise if reached, and the whole
    `cardiosentinel-runs` tree is fingerprinted before and after.
    """
    from cardiosentinel.neural import t1_continuation_labels as L
    from cardiosentinel.neural import t1_continuation_persistence as CP

    runs_root = S.CONTINUATION_RUN_ROOT.parent

    def _fingerprint():
        if not runs_root.exists():
            return ()
        return tuple(
            sorted(
                (str(p.relative_to(runs_root)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in runs_root.rglob("*")
                if p.is_file()
            )
        )

    before = _fingerprint()

    reached: list[str] = []

    def _forbidden(name):
        def sentinel(*_a, **_k):
            reached.append(name)
            raise AssertionError(f"{name} was reached while unauthorized")

        return sentinel

    monkeypatch.setattr(L, "continuation_target_source", _forbidden("label_authority"))
    monkeypatch.setattr(L, "held_out_labels_for_fold", _forbidden("held_out_labels"))
    monkeypatch.setattr(RUN, "_target_source", _forbidden("_target_source"))
    monkeypatch.setattr(RUN, "_claim", _forbidden("claim"))
    monkeypatch.setattr(RUN, "_promote_json", _forbidden("promote_json"))
    monkeypatch.setattr(
        CP, "promote_continuation_held_out_evaluation", _forbidden("promote_held_out")
    )

    with pytest.raises(S.T1ContinuationPermissionError) as excinfo:
        RUN.execute_continuation(tmp_path)

    # 1. It refused for the right reason -- permission, not a downstream error.
    assert "not authorized" in str(excinfo.value)

    # 2. No continuation directory, and no claim inside it.
    assert not S.CONTINUATION_RUN_ROOT.exists()
    assert not (S.CONTINUATION_RUN_ROOT / S.CONTINUATION_ATTEMPT_ID).exists()

    # 3. No artifact file of any kind.
    for name in R.CONTINUATION_RESULT_ARTIFACTS + (S.CONTINUATION_ATTESTATION_NAME,):
        assert not (S.CONTINUATION_RUN_ROOT / S.CONTINUATION_ATTEMPT_ID / name).exists()

    # 4. No label access, and no claim or promotion attempted.
    assert reached == [], f"unauthorized run reached {reached}"

    # 5. No TEST access.
    assert not (runs_root / "TEST_ATTEMPT.json").exists()

    # 6. The entire runs tree is byte-for-byte as it was, including the
    #    consumed canonical attempt.
    assert _fingerprint() == before, "an unauthorized run modified the runs tree"

    # 7. The caller's own directory is untouched too.
    assert list(tmp_path.iterdir()) == []


def test_preflight_refuses_without_touching_the_consumed_attempt(monkeypatch):
    """Even the read-only preflight stops before it reads the predecessor."""
    from cardiosentinel.neural import t1_continuation_predecessor as P

    reached: list[str] = []

    def sentinel(*_a, **_k):
        reached.append("verify_predecessor")
        raise AssertionError("predecessor was read while unauthorized")

    monkeypatch.setattr(RUN, "verify_predecessor", sentinel)
    monkeypatch.setattr(P, "verify_predecessor", sentinel)

    with pytest.raises(S.T1ContinuationPermissionError):
        RUN.preflight()
    assert reached == []
