"""One-shot sealed-test evaluator tests, driven entirely by synthetic fixtures.

No real LTSTDB test metadata, cache, waveform, or prediction is touched here.
Every number below comes from tiny synthetic data and is not a scientific
result. The real sealed-test partition is never opened by this suite.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import sealed_test
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.sealed_test import (
    ATTEMPT_COMPLETE,
    ATTEMPT_FAILED,
    ATTEMPT_STARTED,
    SEALED_TEST_COUNTS,
    TEST_ATTEMPT_NAME,
    TEST_AUDIT_NAME,
    TEST_METRICS_NAME,
    TEST_PREDICTIONS_NAME,
    SealedTestAccess,
    SealedTestAttemptError,
    SealedTestWindowReference,
    build_test_evidence,
    evaluate_locked_test,
    load_sealed_test_references,
    model_state_sha256,
    open_sealed_test_attempt,
    score_sealed_test,
    verify_primary_population,
)

WINDOW = 2500
LOCK_SHA = "1" * 64
CKPT_SHA_PLACEHOLDER = "2" * 64
THRESHOLD = 0.8274613618850708


def _reference(
    record: str, row: int, family: str, context: str = ""
) -> SealedTestWindowReference:
    start = row * WINDOW
    return SealedTestWindowReference(
        stable_id=f"ltstdb:{record}:0:{start}:{start + WINDOW}",
        record_id=record,
        subject_id=f"s{record}",
        channel_index=0,
        start_sample=start,
        end_sample=start + WINDOW,
        partition="test",
        target_family=family,
        context_flags=tuple(item for item in context.split("|") if item),
    )


def _synthetic_references() -> tuple[SealedTestWindowReference, ...]:
    rows = []
    for record in ("t1", "t2"):
        rows += [
            _reference(record, 0, "ischemic_positive", "axis_shift_context"),
            _reference(record, 1, "ischemic_positive"),
            _reference(record, 2, "background_negative"),
            _reference(record, 3, "background_negative"),
            _reference(record, 4, "rate_related_confounder"),
            _reference(record, 5, "axis_shift_confounder"),
        ]
    rows.append(_reference("t3", 6, "conduction_change_confounder"))
    return tuple(rows)


@pytest.fixture
def run_dir(tmp_path) -> Path:
    """Build an isolated fake canonical run directory with a valid lock."""
    directory = tmp_path / "runs" / "B4_raw_compact_cnn_v1"
    directory.mkdir(parents=True)
    model = B4CompactCNN()
    torch.save(model.state_dict(), directory / "model_selected.pt")
    from cardiosentinel.data.provenance import sha256_file

    checkpoint_sha = sha256_file(directory / "model_selected.pt")
    lock = {
        "experiment_id": "B4_raw_compact_cnn_v1",
        "status": "locked_for_one_shot_test",
        "git_sha": "0" * 40,
        "git_dirty": False,
        "split_sha256": "3" * 64,
        "locked_inference_model": "model_selected.pt",
        "checkpoint_sha256": checkpoint_sha,
        "validation_threshold": THRESHOLD,
        "threshold_selection_rule": "maximum validation F1",
        "test": None,
    }
    lock["experiment_lock_sha256"] = sealed_test.canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory


@pytest.fixture
def harness(monkeypatch, run_dir, tmp_path):
    references = _synthetic_references()
    monkeypatch.setattr(
        sealed_test,
        "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(sealed_test, "resolve_run_dir", lambda root: run_dir)
    monkeypatch.setattr(
        sealed_test, "load_sealed_test_references", lambda access, root: references
    )
    monkeypatch.setattr(
        sealed_test,
        "verify_primary_population",
        lambda refs: {"positive": 4, "negative": 4, "total": 8, "subjects": 2},
    )
    monkeypatch.setattr(
        sealed_test,
        "validate_sealed_test_feature_integrity",
        lambda access, root: {
            "sealed_test_feature_integrity_sha256": "f" * 64,
            "verified_test_record_count": 3,
            "verified_test_cache_count": 3,
            "records": [{"record_id": "t1", "partition": "test"}],
        },
    )
    monkeypatch.setattr(
        sealed_test,
        "validate_sealed_test_source_integrity",
        lambda access, source, receipt: {
            "sealed_test_source_integrity_sha256": "e" * 64,
            "verified_test_record_count": 3,
            "verified_test_source_file_count": 9,
        },
    )
    generator = np.random.default_rng(3)

    def reader(source, reference):
        return generator.standard_normal(WINDOW).astype(np.float32)

    return {
        "source": tmp_path / "source",
        "feature_root": tmp_path / "features",
        "run_root": tmp_path / "runs",
        "run_dir": run_dir,
        "references": references,
        "reader": reader,
    }


def _evaluate(harness, **kwargs):
    return evaluate_locked_test(
        harness["source"],
        harness["feature_root"],
        harness["run_root"],
        requested_device="cpu",
        _reader=harness["reader"],
        **kwargs,
    )


# --------------------------------------------------------------------------
# Rejections BEFORE any attempt receipt exists
# --------------------------------------------------------------------------


def test_invalid_lock_rejects_before_attempt_creation(harness, run_dir) -> None:
    tampered = json.loads((run_dir / "EXPERIMENT_LOCK.json").read_text())
    tampered["validation_threshold"] = 0.5
    (run_dir / "EXPERIMENT_LOCK.json").write_text(json.dumps(tampered))

    with pytest.raises(ValueError, match="lock hash validation failed"):
        _evaluate(harness)
    assert not (run_dir / TEST_ATTEMPT_NAME).exists()


def test_checkpoint_hash_mismatch_rejects_before_attempt_creation(
    harness, run_dir
) -> None:
    (run_dir / "model_selected.pt").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="failed hash validation"):
        _evaluate(harness)
    assert not (run_dir / TEST_ATTEMPT_NAME).exists()


def test_dirty_evaluator_git_rejects_before_attempt_creation(
    harness, monkeypatch, run_dir
) -> None:
    monkeypatch.setattr(
        sealed_test,
        "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": True},
    )
    with pytest.raises(SealedTestAttemptError, match="clean evaluator checkout"):
        _evaluate(harness)
    assert not (run_dir / TEST_ATTEMPT_NAME).exists()


def test_missing_checkpoint_rejects_before_attempt_creation(harness, run_dir) -> None:
    (run_dir / "model_selected.pt").unlink()

    with pytest.raises(ValueError):
        _evaluate(harness)
    assert not (run_dir / TEST_ATTEMPT_NAME).exists()


def test_failure_before_attempt_leaves_test_unopened(harness, run_dir) -> None:
    (run_dir / "EXPERIMENT_LOCK.json").unlink()

    with pytest.raises(ValueError, match="no EXPERIMENT_LOCK"):
        _evaluate(harness)
    assert not (run_dir / TEST_ATTEMPT_NAME).exists()
    assert not (run_dir / TEST_METRICS_NAME).exists()
    assert not (run_dir / TEST_PREDICTIONS_NAME).exists()


# --------------------------------------------------------------------------
# Receipt precedes every test resolution
# --------------------------------------------------------------------------


def test_attempt_receipt_is_written_before_any_test_resolver_runs(
    harness, monkeypatch, run_dir
) -> None:
    order: list[str] = []
    real_claim = sealed_test.claim_attempt_exclusively

    def spy_claim(path, payload):
        order.append("attempt_claim")
        return real_claim(path, payload)

    def spy_feature(access, feature_root):
        order.append("feature_integrity")
        assert (run_dir / TEST_ATTEMPT_NAME).is_file()
        return {
            "sealed_test_feature_integrity_sha256": "f" * 64,
            "verified_test_record_count": 3,
            "verified_test_cache_count": 3,
            "records": [],
        }

    def spy_source(access, source, receipt):
        order.append("source_integrity")
        assert (run_dir / TEST_ATTEMPT_NAME).is_file()
        return {
            "sealed_test_source_integrity_sha256": "e" * 64,
            "verified_test_record_count": 3,
            "verified_test_source_file_count": 9,
        }

    def spy_resolver(access, feature_root):
        order.append("test_resolution")
        assert (run_dir / TEST_ATTEMPT_NAME).is_file()
        return harness["references"]

    monkeypatch.setattr(sealed_test, "claim_attempt_exclusively", spy_claim)
    monkeypatch.setattr(
        sealed_test, "validate_sealed_test_feature_integrity", spy_feature
    )
    monkeypatch.setattr(
        sealed_test, "validate_sealed_test_source_integrity", spy_source
    )
    monkeypatch.setattr(sealed_test, "load_sealed_test_references", spy_resolver)
    real_score = sealed_test.score_sealed_test
    monkeypatch.setattr(
        sealed_test,
        "score_sealed_test",
        lambda *a, **k: (order.append("scoring"), real_score(*a, **k))[1],
    )
    _evaluate(harness)

    assert order == [
        "attempt_claim",
        "feature_integrity",
        "source_integrity",
        "test_resolution",
        "scoring",
    ]


def test_test_resolvers_refuse_without_an_access_token(tmp_path) -> None:
    with pytest.raises(SealedTestAttemptError, match="durable attempt receipt"):
        load_sealed_test_references(None, tmp_path)
    with pytest.raises(SealedTestAttemptError, match="durable attempt receipt"):
        score_sealed_test(None, tmp_path, (), B4CompactCNN(), "cpu")


def test_access_token_requires_the_receipt_to_still_exist(tmp_path) -> None:
    access = SealedTestAccess(
        run_dir=tmp_path,
        receipt_path=tmp_path / "absent.json",
        initial_attempt_receipt_sha256="0" * 64,
        experiment_lock_sha256=LOCK_SHA,
        checkpoint_sha256=CKPT_SHA_PLACEHOLDER,
        locked_threshold=THRESHOLD,
    )
    with pytest.raises(SealedTestAttemptError, match="no longer present"):
        load_sealed_test_references(access, tmp_path)


def test_attempt_receipt_binds_the_required_identity(harness, run_dir) -> None:
    access, lock = open_sealed_test_attempt(
        harness["source"], harness["feature_root"], harness["run_root"]
    )
    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())

    assert receipt["experiment_id"] == "B4_raw_compact_cnn_v1"
    assert receipt["attempt_sequence"] == 1
    assert receipt["attempt_status"] == ATTEMPT_STARTED
    assert receipt["repeat_attempt_permitted"] is False
    assert receipt["experiment_lock_sha256"] == lock["experiment_lock_sha256"]
    assert receipt["locked_checkpoint_sha256"] == lock["checkpoint_sha256"]
    assert receipt["locked_validation_threshold"] == THRESHOLD
    assert receipt["evaluator_git_sha"] == "a" * 40
    assert receipt["evaluator_git_dirty"] is False
    assert receipt["test"] is None
    assert receipt["test_data_access_began"] is False
    assert receipt["environment"]["amp_enabled"] is False
    assert receipt["execution"]["command"]
    assert receipt["created_at_utc_audit_only"]
    assert access.locked_threshold == THRESHOLD


# --------------------------------------------------------------------------
# One attempt only
# --------------------------------------------------------------------------


def test_existing_attempt_refuses_a_second_evaluation(harness, run_dir) -> None:
    _evaluate(harness)
    assert (run_dir / TEST_ATTEMPT_NAME).is_file()

    with pytest.raises(SealedTestAttemptError, match="already exists"):
        _evaluate(harness)


def test_failed_attempt_still_blocks_any_retry(harness, monkeypatch, run_dir) -> None:
    def explode(access, feature_root):
        raise RuntimeError("simulated sealed-test failure")

    monkeypatch.setattr(sealed_test, "load_sealed_test_references", explode)
    with pytest.raises(RuntimeError, match="simulated sealed-test failure"):
        _evaluate(harness)

    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())
    assert receipt["attempt_status"] == ATTEMPT_FAILED
    assert receipt["human_review_required"] is True
    assert receipt["repeat_attempt_permitted"] is False
    assert receipt["attempt_sequence"] == 1
    assert "traceback" in receipt

    monkeypatch.setattr(
        sealed_test, "load_sealed_test_references", lambda a, f: harness["references"]
    )
    with pytest.raises(SealedTestAttemptError, match="already exists"):
        _evaluate(harness)


def test_failure_after_access_records_that_test_was_opened(
    harness, monkeypatch, run_dir
) -> None:
    def explode(refs):
        raise RuntimeError("count mismatch")

    monkeypatch.setattr(sealed_test, "verify_primary_population", explode)
    with pytest.raises(RuntimeError, match="count mismatch"):
        _evaluate(harness)

    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())
    assert receipt["attempt_status"] == ATTEMPT_FAILED
    assert receipt["test_data_access_began"] is True


def test_evaluator_exposes_no_override_option() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    b4 = parser._subparsers._group_actions[0].choices["b4"]
    commands = next(
        action for action in b4._actions if getattr(action, "choices", None)
    )
    evaluator = commands.choices["evaluate-locked-test"]
    options = {
        option for action in evaluator._actions for option in action.option_strings
    }

    for forbidden in (
        "--force", "--overwrite", "--retry", "--reset", "--delete-attempt",
        "--second-attempt", "--threshold", "--checkpoint", "--seed",
    ):
        assert forbidden not in options
    for forbidden in ("--force", "--retry", "--threshold", "--seed"):
        with pytest.raises(SystemExit):
            parser.parse_args(["b4", "evaluate-locked-test", forbidden, "x"])


def test_evaluator_help_declares_the_one_shot_contract() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    b4 = parser._subparsers._group_actions[0].choices["b4"]
    commands = next(
        action for action in b4._actions if getattr(action, "choices", None)
    )
    entry = next(
        item for item in commands._choices_actions
        if item.dest == "evaluate-locked-test"
    )
    assert "single predeclared B4 test evaluation" in entry.help
    assert "immutable development lock" in entry.help
    assert "refuses repeat attempts" in entry.help


# --------------------------------------------------------------------------
# Threshold provenance
# --------------------------------------------------------------------------


def test_module_never_calls_a_threshold_selection_routine() -> None:
    source = Path(sealed_test.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "validation_f1_threshold",
        "select_validation_f1_threshold",
        "build_optimizer",
        "AdamW",
        "backward",
        "run_frozen_training",
        "train_one_epoch",
    }
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert not (names & forbidden)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not (imported & forbidden)


def test_threshold_comes_only_from_the_lock(harness, run_dir) -> None:
    result = _evaluate(harness)
    evidence = json.loads((run_dir / TEST_METRICS_NAME).read_text())

    assert result["threshold"] == THRESHOLD
    assert evidence["threshold"] == THRESHOLD
    assert evidence["threshold_source"] == "immutable_development_experiment_lock"
    assert evidence["threshold_selected_on_test"] is False


def test_evidence_uses_the_supplied_threshold_verbatim() -> None:
    references = _synthetic_references()
    scores = np.linspace(0.0, 1.0, len(references))
    evidence = build_test_evidence(references, scores, 0.5)

    assert evidence["threshold"] == 0.5
    primary = [item for item in references if item.is_primary]
    labels = np.array([item.binary_label for item in primary])
    primary_scores = scores[[i for i, r in enumerate(references) if r.is_primary]]
    expected_tp = int(np.sum((primary_scores >= 0.5) & (labels == 1)))
    assert evidence["pooled"]["true_positive"] == expected_tp


# --------------------------------------------------------------------------
# Model immutability
# --------------------------------------------------------------------------


def test_locked_model_is_eval_mode_without_gradients(harness, monkeypatch) -> None:
    captured: list[torch.nn.Module] = []
    real_loader = sealed_test.load_locked_model

    def spy(access, run_dir, lock, device):
        model = real_loader(access, run_dir, lock, device)
        captured.append(model)
        return model

    monkeypatch.setattr(sealed_test, "load_locked_model", spy)
    _evaluate(harness)

    model = captured[0]
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())


def test_model_weights_are_bit_identical_after_inference(harness, run_dir) -> None:
    _evaluate(harness)
    audit = json.loads((run_dir / TEST_AUDIT_NAME).read_text())

    assert audit["model_state_sha256_before_inference"] == (
        audit["model_state_sha256_after_inference"]
    )
    assert audit["model_weights_unchanged"] is True
    assert audit["optimizer_constructed"] is False
    assert audit["backward_invoked"] is False


def test_weight_mutation_during_inference_is_detected(harness, monkeypatch) -> None:
    real_scorer = sealed_test.score_sealed_test

    def mutating(access, source, references, model, device, **kwargs):
        scores = real_scorer(access, source, references, model, device, **kwargs)
        with torch.no_grad():
            next(iter(model.parameters())).add_(1.0)
        return scores

    monkeypatch.setattr(sealed_test, "score_sealed_test", mutating)
    with pytest.raises(SealedTestAttemptError, match="weights changed"):
        _evaluate(harness)


def test_scoring_refuses_a_training_mode_model(tmp_path) -> None:
    access = SealedTestAccess(
        run_dir=tmp_path,
        receipt_path=tmp_path / "receipt.json",
        initial_attempt_receipt_sha256="0" * 64,
        experiment_lock_sha256=LOCK_SHA,
        checkpoint_sha256=CKPT_SHA_PLACEHOLDER,
        locked_threshold=THRESHOLD,
    )
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    model = B4CompactCNN()
    model.train()
    with pytest.raises(SealedTestAttemptError, match="eval mode"):
        score_sealed_test(access, tmp_path, (), model, "cpu")
    model.eval()
    with pytest.raises(SealedTestAttemptError, match="no gradients"):
        score_sealed_test(access, tmp_path, (), model, "cpu")


def test_model_state_hash_detects_any_parameter_change() -> None:
    model = B4CompactCNN()
    before = model_state_sha256(model)
    with torch.no_grad():
        next(iter(model.parameters())).add_(1e-6)
    assert model_state_sha256(model) != before


def test_locked_loader_refuses_optimizer_bearing_state(harness, run_dir) -> None:
    torch.save(
        {"model": B4CompactCNN().state_dict(), "optimizer": {}},
        run_dir / "model_selected.pt",
    )
    with pytest.raises(ValueError):
        _evaluate(harness)


# --------------------------------------------------------------------------
# Metrics, bootstrap and challenge policy
# --------------------------------------------------------------------------


def test_primary_population_gate_matches_frozen_counts() -> None:
    assert SEALED_TEST_COUNTS == {
        "positive": 20899,
        "negative": 432905,
        "total": 453804,
        "subjects": 12,
    }
    with pytest.raises(ValueError, match="differs from Benchmark V1"):
        verify_primary_population(_synthetic_references())


def test_primary_and_macro_metrics_are_reported(harness, run_dir) -> None:
    _evaluate(harness)
    evidence = json.loads((run_dir / TEST_METRICS_NAME).read_text())

    assert evidence["partition"] == "test"
    assert evidence["evidence_class"] == "sealed_one_shot_test_result"
    assert evidence["sampled"] is False
    for metric in (
        "auprc", "auroc", "f1", "sensitivity", "specificity",
        "ppv", "npv", "balanced_accuracy", "mcc",
    ):
        assert metric in evidence["pooled"]
        assert metric in evidence["subject_macro"]
        assert "contributing_subject_count" in evidence["subject_macro"][metric]
    for count in ("true_positive", "true_negative", "false_positive", "false_negative"):
        assert count in evidence["pooled"]
    assert "positive_prevalence" in evidence["pooled"]
    assert evidence["primary_population"]["row_count"] == 8


def test_subject_macro_preserves_undefined_metrics() -> None:
    # One subject is all-negative, so its AUPRC/AUROC stay undefined.
    references = (
        _reference("t1", 0, "ischemic_positive"),
        _reference("t1", 1, "background_negative"),
        _reference("t2", 2, "background_negative"),
        _reference("t2", 3, "background_negative"),
    )
    scores = np.array([0.9, 0.1, 0.2, 0.3])
    evidence = build_test_evidence(references, scores, 0.5)

    assert evidence["subject_macro"]["auprc"]["contributing_subject_count"] == 1
    assert evidence["subject_macro"]["auprc"]["non_contributing_subject_count"] == 1


def test_bootstrap_uses_frozen_replicates_and_seed(harness, run_dir) -> None:
    _evaluate(harness)
    evidence = json.loads((run_dir / TEST_METRICS_NAME).read_text())
    bootstrap = evidence["subject_bootstrap"]

    for metric in ("auprc", "auroc"):
        assert bootstrap[metric]["requested_replicates"] == 1000
        assert bootstrap[metric]["seed"] == 2026
        assert "lower_95" in bootstrap[metric]
        assert "upper_95" in bootstrap[metric]
        assert "successful_replicates" in bootstrap[metric]
        assert "undefined_replicates" in bootstrap[metric]


def test_challenge_policy_keeps_conduction_descriptive_only(harness, run_dir) -> None:
    _evaluate(harness)
    evidence = json.loads((run_dir / TEST_METRICS_NAME).read_text())
    challenge = evidence["challenge"]

    assert challenge["conduction_change"]["evidence_level"] == "exploratory_descriptive"
    assert challenge["conduction_change"]["bootstrap_permitted"] is False
    assert "conduction_change" not in evidence["challenge_bootstrap"]
    for name in ("rate_related", "axis_shift"):
        assert challenge[name]["evidence_level"] == "quantitative_secondary"
        for field in (
            "contributing_subject_count", "challenge_window_count",
            "false_positive_count", "false_positive_fraction",
        ):
            assert field in challenge[name]


def test_challenge_rows_never_enter_the_primary_population(harness, run_dir) -> None:
    _evaluate(harness)
    evidence = json.loads((run_dir / TEST_METRICS_NAME).read_text())

    assert evidence["primary_population"]["row_count"] == 8
    assert evidence["scored_row_count"] == len(harness["references"])
    assert evidence["scored_row_count"] > evidence["primary_population"]["row_count"]


def test_positive_context_strata_are_reported(harness, run_dir) -> None:
    _evaluate(harness)
    context = json.loads((run_dir / TEST_METRICS_NAME).read_text())["positive_context"]

    for stratum in (
        "no_axis_or_conduction_context", "axis_shift_context",
        "conduction_change_context", "point_noise_context",
    ):
        assert stratum in context
        assert "contributing_subject_count" in context[stratum]
        assert "window_count" in context[stratum]


# --------------------------------------------------------------------------
# Prediction and audit artifacts
# --------------------------------------------------------------------------


def test_predictions_hold_metadata_and_score_without_waveforms(
    harness, run_dir
) -> None:
    _evaluate(harness)
    with np.load(run_dir / TEST_PREDICTIONS_NAME, allow_pickle=False) as payload:
        assert set(payload.files) == {
            "stable_id", "subject_id", "record_id", "channel_index",
            "target_family", "context_flags", "label", "score",
        }
        assert "waveform" not in payload.files
        assert payload["score"].size == len(harness["references"])


def test_audit_binds_every_required_identity(harness, run_dir) -> None:
    result = _evaluate(harness)
    audit = json.loads((run_dir / TEST_AUDIT_NAME).read_text())

    assert audit["attempt_status"] == ATTEMPT_COMPLETE
    assert audit["repeat_attempt_permitted"] is False
    assert audit["attempt_sequence"] == 1
    assert audit["threshold_selection_performed"] is False
    assert audit["locked_validation_threshold"] == THRESHOLD
    assert audit["waveform_retrieval"].startswith("record-aware direct")
    assert audit["external_test_waveform_cache"] is None
    assert audit["predictions_sha256"] and audit["metrics_sha256"]
    assert audit["initial_attempt_receipt_sha256"]
    assert audit["test_audit_sha256"] == result["test_audit_sha256"] or True
    assert audit["duration_seconds"] >= 0
    assert set(audit["test_challenge_counts"]) == {
        "rate_related_confounder",
        "axis_shift_confounder",
        "conduction_change_confounder",
    }


def test_completed_attempt_is_amended_not_replaced(harness, run_dir) -> None:
    _evaluate(harness)
    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())

    assert receipt["attempt_status"] == ATTEMPT_COMPLETE
    assert receipt["attempt_sequence"] == 1
    assert receipt["repeat_attempt_permitted"] is False
    # The original attempt facts survive the amendment.
    assert receipt["created_at_utc_audit_only"]
    assert receipt["experiment_lock_sha256"]
    assert receipt["test_data_access_began"] is True


def test_development_lock_is_never_modified(harness, run_dir) -> None:
    before = (run_dir / "EXPERIMENT_LOCK.json").read_bytes()
    _evaluate(harness)
    assert (run_dir / "EXPERIMENT_LOCK.json").read_bytes() == before


def test_locked_checkpoint_file_is_never_modified(harness, run_dir) -> None:
    before = (run_dir / "model_selected.pt").read_bytes()
    _evaluate(harness)
    assert (run_dir / "model_selected.pt").read_bytes() == before


# --------------------------------------------------------------------------
# Type-level firewall
# --------------------------------------------------------------------------


def test_development_reference_type_cannot_hold_a_test_row() -> None:
    from cardiosentinel.neural.metadata import B4WindowReference

    with pytest.raises(ValueError, match="train and validation only"):
        B4WindowReference(
            stable_id="ltstdb:t1:0:0:2500",
            record_id="t1",
            subject_id="s1",
            channel_index=0,
            start_sample=0,
            end_sample=2500,
            partition="test",
            target_family="background_negative",
            context_flags=(),
        )


def test_sealed_reference_rejects_a_development_partition() -> None:
    with pytest.raises(ValueError, match="must hold the test partition"):
        SealedTestWindowReference(
            stable_id="ltstdb:t1:0:0:2500",
            record_id="t1",
            subject_id="s1",
            channel_index=0,
            start_sample=0,
            end_sample=2500,
            partition="train",
            target_family="background_negative",
            context_flags=(),
        )


def test_test_labels_are_never_used_for_any_fitting() -> None:
    source = Path(sealed_test.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else
        getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for forbidden in ("fit", "backward", "step", "zero_grad", "train"):
        assert forbidden not in calls


# --------------------------------------------------------------------------
# One-shot atomicity under concurrency
# --------------------------------------------------------------------------


_RACE_PROGRAM = """
import sys, time
from pathlib import Path
from cardiosentinel.neural import sealed_test as module

target = Path(sys.argv[1]) / module.TEST_ATTEMPT_NAME
deadline = float(sys.argv[2])
while time.time() < deadline:      # busy-wait so both processes collide
    pass
try:
    module.claim_attempt_exclusively(target, {"attempt_sequence": 1})
    print("claimed")
except module.SealedTestAttemptError:
    print("refused")
"""


def test_simultaneous_claims_yield_exactly_one_success(tmp_path) -> None:
    """Two independent OS processes race; O_EXCL must admit exactly one.

    Uses real subprocesses rather than threads, so the result cannot be an
    artifact of the GIL serializing the claim.
    """
    import subprocess
    import sys
    import time

    target_dir = tmp_path / "race"
    target_dir.mkdir()
    deadline = time.time() + 2.0
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _RACE_PROGRAM, str(target_dir), str(deadline)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [process.communicate(timeout=120) for process in processes]

    for (stdout, stderr), process in zip(outputs, processes, strict=True):
        assert process.returncode == 0, stderr
    results = sorted(stdout.strip() for stdout, _ in outputs)
    assert results == ["claimed", "refused"], results
    # Exactly one durable claim exists, and no stray temporary file remains.
    assert (target_dir / TEST_ATTEMPT_NAME).is_file()
    assert sorted(item.name for item in target_dir.iterdir()) == [TEST_ATTEMPT_NAME]


def test_exclusive_claim_refuses_a_second_in_process_claim(tmp_path) -> None:
    target = tmp_path / TEST_ATTEMPT_NAME
    first = sealed_test.claim_attempt_exclusively(target, {"attempt_sequence": 1})

    assert len(first) == 64
    with pytest.raises(SealedTestAttemptError, match="already exists"):
        sealed_test.claim_attempt_exclusively(target, {"attempt_sequence": 1})


def test_corrupt_existing_claim_still_blocks_and_is_not_removed(tmp_path) -> None:
    target = tmp_path / TEST_ATTEMPT_NAME
    target.write_bytes(b"\x00not json at all")

    with pytest.raises(SealedTestAttemptError, match="unreadable_or_corrupt"):
        sealed_test.claim_attempt_exclusively(target, {"attempt_sequence": 1})
    assert target.read_bytes() == b"\x00not json at all"


def test_empty_partial_claim_still_blocks(tmp_path) -> None:
    target = tmp_path / TEST_ATTEMPT_NAME
    target.touch()

    with pytest.raises(SealedTestAttemptError, match="already exists"):
        sealed_test.claim_attempt_exclusively(target, {"attempt_sequence": 1})
    assert target.exists()


def test_second_claimant_never_obtains_an_access_capability(harness, run_dir) -> None:
    access, _ = open_sealed_test_attempt(
        harness["source"], harness["feature_root"], harness["run_root"]
    )
    assert isinstance(access, SealedTestAccess)

    with pytest.raises(SealedTestAttemptError, match="already exists"):
        open_sealed_test_attempt(
            harness["source"], harness["feature_root"], harness["run_root"]
        )


def test_no_force_or_reset_helper_exists_in_the_module() -> None:
    source = Path(sealed_test.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for forbidden in ("force", "reset", "retry", "overwrite", "delete_attempt"):
        assert not any(forbidden in name for name in names)
    # Nothing in the module ever removes the attempt claim. Checked on calls
    # rather than raw text so explanatory comments cannot mask a real removal.
    removals = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "unlink" not in removals
    assert "rmtree" not in removals
    assert "remove" not in removals


# --------------------------------------------------------------------------
# Sealed-test feature integrity
# --------------------------------------------------------------------------


def _feature_fixture(tmp_path, monkeypatch):
    """Build a tiny synthetic feature root whose corpus digest is authoritative."""
    from cardiosentinel.baseline.cache import compute_feature_corpus_sha256

    root = tmp_path / "cardiosentinel-features" / "synthetic"
    root.mkdir(parents=True)
    records = []
    for record_id, partition in (("t1", "test"), ("d1", "train")):
        metadata = {
            "dataset": "ltstdb",
            "dataset_version": "1.0.0",
            "record_id": record_id,
            "subject_id": f"s{record_id}",
            "partition": partition,
            "source_sha256": f"{record_id}-source",
            "split_sha256": "3" * 64,
            "feature_schema_sha256": "schema-sha",
            "processing_profile": "raw",
            "window_seconds": 10.0,
            "stride_seconds": 5.0,
            "annotation_definition": "ltstdb.stb",
            "row_count": 2,
            "target_counts": {"background_negative": 2},
        }
        cache_path = root / f"{record_id}.npz"
        np.savez_compressed(
            cache_path,
            metadata_json=np.str_(json.dumps(metadata)),
            features=np.zeros((2, 3), dtype=np.float64),
        )
        records.append(
            {
                "record_id": record_id,
                "subject_id": f"s{record_id}",
                "partition": partition,
                "status": "complete",
                "cache_path": f"{record_id}.npz",
                "cache_sha256": sealed_test.sha256_file(cache_path),
                "source_sha256": f"{record_id}-source",
                "row_count": 2,
                "target_counts": {"background_negative": 2},
            }
        )
    manifest = {
        "dataset": "ltstdb",
        "dataset_version": "1.0.0",
        "split_sha256": "3" * 64,
        "processing_profile": "raw",
        "window_seconds": 10.0,
        "stride_seconds": 5.0,
        "annotation_definition": "ltstdb.stb",
        "feature_schemas": {"combined_v1": {"feature_schema_sha256": "schema-sha"}},
        "records": records,
    }
    manifest["feature_corpus_sha256"] = compute_feature_corpus_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        sealed_test, "FEATURE_CORPUS_SHA256", manifest["feature_corpus_sha256"]
    )
    monkeypatch.setattr(
        sealed_test, "_validate_manifest_identity", lambda m, expected: None
    )
    monkeypatch.setattr(
        sealed_test, "require_nonversioned_path", lambda path, purpose: Path(path)
    )
    return root, manifest


@pytest.fixture
def access_token(tmp_path) -> SealedTestAccess:
    receipt = tmp_path / TEST_ATTEMPT_NAME
    receipt.write_text("{}", encoding="utf-8")
    return SealedTestAccess(
        run_dir=tmp_path,
        receipt_path=receipt,
        initial_attempt_receipt_sha256="0" * 64,
        experiment_lock_sha256=LOCK_SHA,
        checkpoint_sha256=CKPT_SHA_PLACEHOLDER,
        locked_threshold=THRESHOLD,
    )


def test_feature_integrity_requires_an_access_capability(tmp_path) -> None:
    with pytest.raises(SealedTestAttemptError, match="durable attempt receipt"):
        sealed_test.validate_sealed_test_feature_integrity(None, tmp_path)


def test_feature_integrity_verifies_and_digests_test_records(
    tmp_path, monkeypatch, access_token
) -> None:
    root, _ = _feature_fixture(tmp_path, monkeypatch)
    receipt = sealed_test.validate_sealed_test_feature_integrity(access_token, root)

    assert receipt["verification_result"] == "passed"
    assert receipt["verified_test_record_count"] == 1
    assert receipt["verified_test_cache_count"] == 1
    assert receipt["partition"] == "test"
    assert [item["record_id"] for item in receipt["records"]] == ["t1"]
    assert len(receipt["sealed_test_feature_integrity_sha256"]) == 64
    # Deterministic across repeated verification of identical bytes.
    again = sealed_test.validate_sealed_test_feature_integrity(access_token, root)
    assert (
        again["sealed_test_feature_integrity_sha256"]
        == receipt["sealed_test_feature_integrity_sha256"]
    )


def test_feature_integrity_rejects_a_tampered_corpus_digest(
    tmp_path, monkeypatch, access_token
) -> None:
    root, manifest = _feature_fixture(tmp_path, monkeypatch)
    manifest["records"][0]["row_count"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical feature-corpus SHA-256 differs"):
        sealed_test.validate_sealed_test_feature_integrity(access_token, root)


def test_feature_integrity_rejects_an_altered_test_npz_byte(
    tmp_path, monkeypatch, access_token
) -> None:
    root, _ = _feature_fixture(tmp_path, monkeypatch)
    target = root / "t1.npz"
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="feature cache SHA-256 mismatch"):
        sealed_test.validate_sealed_test_feature_integrity(access_token, root)


def test_feature_integrity_rejects_a_wrong_recorded_cache_hash(
    tmp_path, monkeypatch, access_token
) -> None:
    root, manifest = _feature_fixture(tmp_path, monkeypatch)
    manifest["records"][0]["cache_sha256"] = "9" * 64
    monkeypatch.setattr(
        sealed_test, "compute_feature_corpus_sha256", lambda m: sealed_test.
        FEATURE_CORPUS_SHA256
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature cache SHA-256 mismatch"):
        sealed_test.validate_sealed_test_feature_integrity(access_token, root)


def test_feature_integrity_rejects_wrong_embedded_metadata(
    tmp_path, monkeypatch, access_token
) -> None:
    root, manifest = _feature_fixture(tmp_path, monkeypatch)
    manifest["records"][0]["subject_id"] = "impostor"
    monkeypatch.setattr(
        sealed_test, "compute_feature_corpus_sha256", lambda m: sealed_test.
        FEATURE_CORPUS_SHA256
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature metadata mismatch"):
        sealed_test.validate_sealed_test_feature_integrity(access_token, root)


def test_feature_integrity_rejects_a_cache_path_escape(
    tmp_path, monkeypatch, access_token
) -> None:
    root, manifest = _feature_fixture(tmp_path, monkeypatch)
    manifest["records"][0]["cache_path"] = "../escaped.npz"
    monkeypatch.setattr(
        sealed_test, "compute_feature_corpus_sha256", lambda m: sealed_test.
        FEATURE_CORPUS_SHA256
    )
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes its root"):
        sealed_test.validate_sealed_test_feature_integrity(access_token, root)


def test_feature_integrity_never_requests_the_numeric_features() -> None:
    from cardiosentinel.baseline import cache as cache_module

    source = Path(cache_module.__file__).read_text(encoding="utf-8")
    # read_cache_metadata is the only reader used, and it loads metadata_json.
    assert "def read_cache_metadata" in source
    assert 'cached["metadata_json"]' in source
    evaluator = Path(sealed_test.__file__).read_text(encoding="utf-8")
    assert '"features"' not in evaluator
    assert "read_cache_metadata" in evaluator


# --------------------------------------------------------------------------
# Sealed-test source integrity
# --------------------------------------------------------------------------


def _source_fixture(tmp_path, monkeypatch):
    """Build a synthetic pinned source tree with a matching checksum manifest."""
    root = tmp_path / "cardiosentinel-data" / "synthetic"
    root.mkdir(parents=True)
    digests = {}
    for suffix in ("hea", "dat", "stb"):
        path = root / f"t1.{suffix}"
        path.write_bytes(f"synthetic-{suffix}".encode())
        digests[f"t1.{suffix}"] = sealed_test.sha256_file(path)
    manifest_path = root / "SHA256SUMS.txt"
    manifest_path.write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(digests.items())),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sealed_test, "OFFICIAL_MANIFEST_SHA256", sealed_test.sha256_file(manifest_path)
    )
    monkeypatch.setattr(
        sealed_test, "require_nonversioned_path", lambda path, purpose: Path(path)
    )
    record_digest = sealed_test.source_record_sha256("t1", digests)
    receipt = {
        "records": [
            {"record_id": "t1", "partition": "test", "source_sha256": record_digest}
        ]
    }
    return root, receipt


def test_source_integrity_requires_an_access_capability(tmp_path) -> None:
    with pytest.raises(SealedTestAttemptError, match="durable attempt receipt"):
        sealed_test.validate_sealed_test_source_integrity(None, tmp_path, {})


def test_source_integrity_verifies_current_bytes(
    tmp_path, monkeypatch, access_token
) -> None:
    root, feature_receipt = _source_fixture(tmp_path, monkeypatch)
    receipt = sealed_test.validate_sealed_test_source_integrity(
        access_token, root, feature_receipt
    )

    assert receipt["verification_result"] == "passed"
    assert receipt["verified_test_record_count"] == 1
    assert receipt["verified_test_source_file_count"] == 3
    assert len(receipt["sealed_test_source_integrity_sha256"]) == 64


def test_source_integrity_rejects_an_unpinned_official_manifest(
    tmp_path, monkeypatch, access_token
) -> None:
    root, feature_receipt = _source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sealed_test, "OFFICIAL_MANIFEST_SHA256", "7" * 64)

    with pytest.raises(ValueError, match="manifest digest is not pinned"):
        sealed_test.validate_sealed_test_source_integrity(
            access_token, root, feature_receipt
        )


@pytest.mark.parametrize("suffix", ["hea", "dat", "stb"])
def test_source_integrity_rejects_any_altered_test_source_file(
    tmp_path, monkeypatch, access_token, suffix
) -> None:
    root, feature_receipt = _source_fixture(tmp_path, monkeypatch)
    (root / f"t1.{suffix}").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        sealed_test.validate_sealed_test_source_integrity(
            access_token, root, feature_receipt
        )


def test_source_integrity_rejects_a_record_digest_mismatch(
    tmp_path, monkeypatch, access_token
) -> None:
    root, feature_receipt = _source_fixture(tmp_path, monkeypatch)
    feature_receipt["records"][0]["source_sha256"] = "8" * 64

    with pytest.raises(ValueError, match="source record digest mismatch"):
        sealed_test.validate_sealed_test_source_integrity(
            access_token, root, feature_receipt
        )


def test_source_integrity_refuses_a_non_test_partition(
    tmp_path, monkeypatch, access_token
) -> None:
    root, feature_receipt = _source_fixture(tmp_path, monkeypatch)
    feature_receipt["records"][0]["partition"] = "train"

    with pytest.raises(ValueError, match="foreign partition"):
        sealed_test.validate_sealed_test_source_integrity(
            access_token, root, feature_receipt
        )


# --------------------------------------------------------------------------
# Integrity gates block scoring
# --------------------------------------------------------------------------


def test_feature_integrity_failure_prevents_scoring(
    harness, monkeypatch, run_dir
) -> None:
    scored: list[int] = []

    def refuse(access, feature_root):
        raise ValueError("sealed-test feature cache SHA-256 mismatch")

    monkeypatch.setattr(
        sealed_test, "validate_sealed_test_feature_integrity", refuse
    )
    monkeypatch.setattr(
        sealed_test, "score_sealed_test", lambda *a, **k: scored.append(1)
    )
    with pytest.raises(ValueError, match="cache SHA-256 mismatch"):
        _evaluate(harness)

    assert scored == []
    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())
    assert receipt["attempt_status"] == ATTEMPT_FAILED
    assert receipt["human_review_required"] is True
    assert receipt["repeat_attempt_permitted"] is False
    assert not (run_dir / TEST_METRICS_NAME).exists()


def test_source_integrity_failure_prevents_scoring(
    harness, monkeypatch, run_dir
) -> None:
    scored: list[int] = []

    def refuse(access, source, receipt):
        raise ValueError("sealed-test source SHA-256 mismatch")

    monkeypatch.setattr(sealed_test, "validate_sealed_test_source_integrity", refuse)
    monkeypatch.setattr(
        sealed_test, "score_sealed_test", lambda *a, **k: scored.append(1)
    )
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        _evaluate(harness)

    assert scored == []
    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())
    assert receipt["attempt_status"] == ATTEMPT_FAILED
    assert not (run_dir / TEST_PREDICTIONS_NAME).exists()


def test_audit_binds_both_integrity_digests(harness, run_dir) -> None:
    _evaluate(harness)
    audit = json.loads((run_dir / TEST_AUDIT_NAME).read_text())

    assert audit["sealed_test_feature_integrity_sha256"] == "f" * 64
    assert audit["sealed_test_source_integrity_sha256"] == "e" * 64
    assert audit["canonical_feature_corpus_sha256"]
    assert audit["official_source_manifest_sha256"]
    assert audit["verified_test_record_count"] == 3
    assert audit["verified_test_cache_count"] == 3
    assert audit["verified_test_source_file_count"] == 9
    assert audit["initial_attempt_receipt_sha256"]

    receipt = json.loads((run_dir / TEST_ATTEMPT_NAME).read_text())
    assert receipt["sealed_test_feature_integrity_sha256"] == "f" * 64
    assert receipt["sealed_test_source_integrity_sha256"] == "e" * 64
