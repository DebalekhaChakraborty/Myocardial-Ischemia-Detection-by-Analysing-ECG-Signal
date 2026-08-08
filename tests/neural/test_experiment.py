"""Canonical B4 runner lifecycle tests driven entirely by synthetic fixtures.

Nothing here is a scientific B4 result. Every metric below is computed from
tiny synthetic waveforms and exists only to prove execution and provenance
machinery. No real training, validation, or sealed-test data is touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import experiment
from cardiosentinel.neural.experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_LOCK_NAME,
    RUN_STATUS_NAME,
    STATUS_COMPLETE,
    STATUS_FAILED,
    build_experiment_lock,
    prepare_b4_experiment,
    resolve_run_dir,
    run_b4_train_validation,
    validate_experiment_lock,
)
from cardiosentinel.neural.metadata import B4MetadataIndex, B4WindowReference
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.protocol import (
    TRAINABLE_PARAMETER_COUNT,
    require_development_partition,
)
from cardiosentinel.neural.training import FrozenTrainingResult

WINDOW = 2500
FAKE_SELECTION = "a" * 64
FAKE_FEATURE_INTEGRITY = "b" * 64
FAKE_SOURCE_INTEGRITY = "c" * 64
FAKE_CACHE_SHA = "d" * 64


def _reference(
    record: str, partition: str, row: int, positive: bool
) -> B4WindowReference:
    start = row * WINDOW
    return B4WindowReference(
        stable_id=f"ltstdb:{record}:0:{start}:{start + WINDOW}",
        record_id=record,
        subject_id=f"s{record}",
        channel_index=0,
        start_sample=start,
        end_sample=start + WINDOW,
        partition=partition,
        target_family="ischemic_positive" if positive else "background_negative",
        context_flags=("axis_shift_context",) if positive else (),
    )


def _index(partition: str, positives: int, negatives: int) -> B4MetadataIndex:
    references = tuple(
        _reference(f"{partition}{row // 2}", partition, row, row < positives)
        for row in range(positives + negatives)
    )
    return B4MetadataIndex(
        partition=partition,
        references=references,
        positive_count=positives,
        negative_count=negatives,
        subject_count=len({item.subject_id for item in references}),
        selection_sha256=FAKE_SELECTION if partition == "train" else None,
    )


class _FakeCache:
    """Minimal stand-in exposing only what B4CachedWaveformDataset consumes."""

    def __init__(self, indexes: dict[str, B4MetadataIndex]) -> None:
        generator = np.random.default_rng(7)
        self.waveforms = {
            partition: generator.standard_normal(
                (index.total_count, WINDOW)
            ).astype(np.float32)
            for partition, index in indexes.items()
        }
        self.manifest = {
            "waveform_cache_sha256": FAKE_CACHE_SHA,
            "cache_complete": True,
            "protocol_sha256": experiment.B4_PROTOCOL_SHA256,
            "split_sha256": experiment.B4_SPLIT_SHA256,
            "feature_corpus_sha256": experiment.FEATURE_CORPUS_SHA256,
            "training_selection_sha256": FAKE_SELECTION,
            "development_feature_integrity_sha256": FAKE_FEATURE_INTEGRITY,
            "development_source_integrity_sha256": FAKE_SOURCE_INTEGRITY,
            "equivalence_audit": {"exact_mismatches": 0, "result": "passed"},
        }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Wire synthetic development inputs into the runner's module namespace."""
    indexes = {"train": _index("train", 2, 2), "validation": _index("validation", 3, 3)}
    cache = _FakeCache(indexes)
    monkeypatch.setattr(
        experiment,
        "EXPECTED_COUNTS",
        {
            partition: {
                "total": index.total_count,
                "positive": index.positive_count,
                "negative": index.negative_count,
            }
            for partition, index in indexes.items()
        },
    )
    monkeypatch.setattr(experiment, "TRAINING_SELECTION_SHA256", FAKE_SELECTION)
    monkeypatch.setattr(
        experiment,
        "git_provenance",
        lambda root: {
            "git_sha": "0" * 40,
            "git_dirty": False,
            "python_version": "3.12.6",
            "cardiosentinel_version": "0.1.0",
        },
    )
    monkeypatch.setattr(
        experiment,
        "validate_development_feature_integrity",
        lambda root: {
            "development_feature_integrity_sha256": FAKE_FEATURE_INTEGRITY
        },
    )
    monkeypatch.setattr(
        experiment,
        "validate_development_source_integrity",
        lambda source, receipt: {
            "development_source_integrity_sha256": FAKE_SOURCE_INTEGRITY
        },
    )
    monkeypatch.setattr(experiment, "build_development_indexes", lambda root: indexes)
    monkeypatch.setattr(
        experiment, "validate_waveform_cache", lambda root, built: cache
    )
    return {
        "source": tmp_path / "source",
        "feature_root": tmp_path / "features",
        "cache_root": tmp_path / "cache",
        "run_root": tmp_path / "runs",
        "indexes": indexes,
        "cache": cache,
    }


def _run(harness, **kwargs):
    return run_b4_train_validation(
        harness["source"],
        harness["feature_root"],
        harness["cache_root"],
        harness["run_root"],
        requested_device="cpu",
        **kwargs,
    )


def _prepare(harness, **kwargs):
    return prepare_b4_experiment(
        harness["source"],
        harness["feature_root"],
        harness["cache_root"],
        harness["run_root"],
        requested_device="cpu",
        **kwargs,
    )


def _stub_training(monkeypatch, auprc_sequence, *, threshold_scores=None):
    """Drive the real frozen loop with a deterministic AUPRC sequence."""
    import cardiosentinel.neural.training as training

    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    scores = (
        np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        if threshold_scores is None
        else threshold_scores
    )
    values = iter(auprc_sequence)
    monkeypatch.setattr(training, "train_one_epoch", lambda *args: 0.25)
    monkeypatch.setattr(training, "validation_scores", lambda *args: (labels, scores))
    monkeypatch.setattr(
        training, "validation_auprc", lambda *args: next(values, auprc_sequence[-1])
    )
    monkeypatch.setattr(
        experiment, "validation_scores", lambda *args: (labels, scores)
    )
    return labels, scores


# --------------------------------------------------------------------------
# Preflight identity binding
# --------------------------------------------------------------------------


def test_preflight_binds_every_frozen_identity(harness) -> None:
    report = _prepare(harness).report

    assert report["experiment_id"] == EXPERIMENT_ID
    assert report["protocol_sha256"] == experiment.B4_PROTOCOL_SHA256
    assert report["split_sha256"] == experiment.B4_SPLIT_SHA256
    assert report["feature_corpus_sha256"] == experiment.FEATURE_CORPUS_SHA256
    assert report["training_selection_sha256"] == FAKE_SELECTION
    assert report["development_feature_integrity_sha256"] == FAKE_FEATURE_INTEGRITY
    assert report["development_source_integrity_sha256"] == FAKE_SOURCE_INTEGRITY
    assert report["waveform_cache_sha256"] == FAKE_CACHE_SHA
    assert report["equivalence_audit"]["exact_mismatches"] == 0
    assert report["model"]["trainable_parameter_count"] == TRAINABLE_PARAMETER_COUNT
    assert report["model"]["fp32_parameter_payload_bytes"] == 348356
    assert report["seed"] == 2026
    assert report["device"] == "cpu"
    assert report["determinism"]["deterministic_algorithms"] is True
    assert report["environment"]["amp_enabled"] is False
    assert report["environment"]["torch_version"]
    assert report["environment"]["numpy_version"]
    assert report["environment"]["python_version"]
    assert report["test_partition_access"] is None
    assert report["resources"]["available_bytes"] >= report["resources"][
        "required_bytes"
    ]


def test_preflight_rejects_dirty_git(harness, monkeypatch) -> None:
    monkeypatch.setattr(
        experiment,
        "git_provenance",
        lambda root: {"git_sha": "0" * 40, "git_dirty": True},
    )
    with pytest.raises(ValueError, match="clean Git checkout"):
        _prepare(harness)


def test_preflight_rejects_wrong_protocol(harness, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise ValueError("B4_PROTOCOL_V1.md differs from its frozen SHA-256.")

    monkeypatch.setattr(experiment, "validate_frozen_protocol", refuse)
    with pytest.raises(ValueError, match="frozen SHA-256"):
        _prepare(harness)


def test_preflight_rejects_wrong_waveform_cache_digest(harness) -> None:
    harness["cache"].manifest["development_feature_integrity_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="waveform cache identity differs"):
        _prepare(harness)


def test_preflight_rejects_nonzero_equivalence_mismatch(harness) -> None:
    harness["cache"].manifest["equivalence_audit"]["exact_mismatches"] = 1
    with pytest.raises(ValueError, match="zero-mismatch equivalence audit"):
        _prepare(harness)


def test_preflight_rejects_wrong_training_selection(harness) -> None:
    train = harness["indexes"]["train"]
    harness["indexes"]["train"] = B4MetadataIndex(
        partition="train",
        references=train.references,
        positive_count=train.positive_count,
        negative_count=train.negative_count,
        subject_count=train.subject_count,
        selection_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="training selection SHA-256 differs"):
        _prepare(harness)


def test_preflight_rejects_row_count_drift(harness, monkeypatch) -> None:
    monkeypatch.setattr(
        experiment,
        "EXPECTED_COUNTS",
        {
            "train": {"total": 999, "positive": 2, "negative": 2},
            "validation": {"total": 6, "positive": 3, "negative": 3},
        },
    )
    with pytest.raises(ValueError, match="train row count differs"):
        _prepare(harness)


def test_preflight_rejects_wrong_model_parameter_count(harness, monkeypatch) -> None:
    monkeypatch.setattr(experiment, "TRAINABLE_PARAMETER_COUNT", 12345)
    with pytest.raises(ValueError, match="parameter count differs"):
        _prepare(harness)


def test_preflight_does_not_create_the_run_directory(harness) -> None:
    prepared = _prepare(harness)
    assert not prepared.run_dir.exists()


# --------------------------------------------------------------------------
# Sealed-test firewall
# --------------------------------------------------------------------------


def test_development_partition_guard_rejects_test() -> None:
    with pytest.raises(ValueError, match="train and validation only"):
        require_development_partition("test")


def test_runner_only_ever_builds_development_partitions(harness) -> None:
    prepared = _prepare(harness)
    assert set(prepared.indexes) == {"train", "validation"}
    assert set(prepared.report["partitions"]) == {"train", "validation"}


def test_experiment_module_never_passes_a_test_partition_argument() -> None:
    source = Path(experiment.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]
        if isinstance(argument, ast.Constant) and argument.value == "test"
    ]
    assert offenders == []


def test_train_validation_cli_cannot_select_a_test_partition() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["b4", "run-train-validation", "--partition", "test"])
    with pytest.raises(SystemExit):
        parser.parse_args(["b4", "run-train-validation", "--test"])


def test_no_test_evaluation_command_is_registered() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["b4", "evaluate-test"])


def _runner_subparser():
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    b4 = parser._subparsers._group_actions[0].choices["b4"]
    commands = next(
        action for action in b4._actions if getattr(action, "choices", None)
    )
    return commands.choices["run-train-validation"]


def test_train_validation_exposes_no_option_naming_test() -> None:
    runner = _runner_subparser()
    options = [option for action in runner._actions for option in action.option_strings]

    assert options
    assert all("test" not in option for option in options)


def test_train_validation_help_declares_no_test_access() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    b4 = parser._subparsers._group_actions[0].choices["b4"]
    commands = next(
        action for action in b4._actions if getattr(action, "choices", None)
    )
    help_text = commands._choices_actions
    entry = next(
        item for item in help_text if item.dest == "run-train-validation"
    )

    assert "single canonical B4 train/validation experiment" in entry.help
    assert "Does not access the test partition." in entry.help


# --------------------------------------------------------------------------
# One canonical run
# --------------------------------------------------------------------------


def test_completed_experiment_refuses_a_second_run(harness, monkeypatch) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    first = _run(harness)
    assert first["status"] == STATUS_COMPLETE

    with pytest.raises(ValueError, match="exactly one"):
        _run(harness)


def test_interrupted_experiment_does_not_silently_rerun(harness, monkeypatch) -> None:
    import cardiosentinel.neural.training as training

    def explode(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "train_one_epoch", explode)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(harness)

    run_dir = resolve_run_dir(harness["run_root"])
    status = json.loads((run_dir / RUN_STATUS_NAME).read_text(encoding="utf-8"))
    assert status["status"] == STATUS_FAILED
    assert status["human_review_required"] is True
    assert status["automatic_restart_performed"] is False
    assert not (run_dir / EXPERIMENT_LOCK_NAME).exists()

    # Restore a working trainer: a later invocation must still refuse, proving
    # the interruption is not silently retried with a fresh scientific run.
    _stub_training(monkeypatch, [0.5])
    with pytest.raises(ValueError, match="requires documented human review"):
        _run(harness)


# --------------------------------------------------------------------------
# Determinism and model construction order
# --------------------------------------------------------------------------


def test_model_is_constructed_after_determinism_initialization(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.5, 0.5, 0.5, 0.5, 0.5])
    order: list[str] = []
    real_determinism = experiment.initialize_determinism

    def spy_determinism(**kwargs):
        order.append("determinism")
        return real_determinism(**kwargs)

    class SpyModel(B4CompactCNN):
        def __init__(self) -> None:
            order.append("model")
            super().__init__()

    monkeypatch.setattr(experiment, "initialize_determinism", spy_determinism)
    monkeypatch.setattr(experiment, "B4CompactCNN", SpyModel)
    _run(harness)

    canonical = order.index("model", order.index("determinism"))
    assert order[canonical - 1] == "determinism"
    assert order.count("model") == 2  # one constant probe, one canonical model


# --------------------------------------------------------------------------
# Checkpoint and threshold selection
# --------------------------------------------------------------------------


def test_checkpoint_selection_follows_maximum_validation_auprc(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.10, 0.70, 0.30, 0.30, 0.30, 0.30])
    result = _run(harness)

    assert result["selected_epoch"] == 2
    assert result["selected_validation_auprc"] == 0.70


def test_exact_auprc_tie_keeps_the_earliest_epoch(harness, monkeypatch) -> None:
    _stub_training(monkeypatch, [0.50, 0.50, 0.50, 0.50, 0.50])
    result = _run(harness)

    assert result["selected_epoch"] == 1


def test_threshold_uses_validation_only_with_highest_tie(harness, monkeypatch) -> None:
    _stub_training(
        monkeypatch,
        [0.5, 0.5, 0.5, 0.5, 0.5],
        threshold_scores=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    )
    result = _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])
    payload = json.loads(
        (run_dir / experiment.VALIDATION_THRESHOLD_NAME).read_text(encoding="utf-8")
    )

    assert payload["selected_from"] == "validation"
    assert payload["test_informed"] is False
    assert payload["threshold"] == result["validation_threshold"]
    assert payload["threshold_rule"] == experiment.THRESHOLD_RULE


# --------------------------------------------------------------------------
# Evidence artifacts
# --------------------------------------------------------------------------


def test_completed_run_writes_atomic_evidence_and_no_partial_files(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])

    for name in (
        experiment.RUN_MANIFEST_NAME,
        experiment.EPOCH_HISTORY_NAME,
        experiment.VALIDATION_METRICS_NAME,
        experiment.VALIDATION_THRESHOLD_NAME,
        experiment.EXPERIMENT_LOCK_NAME,
        experiment.SELECTED_MODEL_NAME,
        experiment.TRAINING_CHECKPOINT_NAME,
        experiment.VALIDATION_PREDICTIONS_NAME,
        RUN_STATUS_NAME,
    ):
        assert (run_dir / name).is_file()
    assert [item.name for item in run_dir.iterdir() if item.name.startswith(".")] == []


def test_epoch_history_is_persisted_after_every_completed_epoch(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.10, 0.20, 0.30, 0.30, 0.30, 0.30, 0.30])
    seen: list[int] = []
    real_write = experiment.write_json_atomic

    def spy_write(path, payload):
        if path.name == experiment.EPOCH_HISTORY_NAME:
            seen.append(len(payload["epochs"]))
        return real_write(path, payload)

    monkeypatch.setattr(experiment, "write_json_atomic", spy_write)
    _run(harness)

    assert seen == list(range(1, len(seen) + 1))


def test_validation_evidence_records_required_metrics(harness, monkeypatch) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    result = _run(harness)
    evidence = result["validation_evidence"]

    assert evidence["partition"] == "validation"
    assert evidence["sampled"] is False
    assert evidence["row_count"] == 6
    assert evidence["positive_count"] == 3
    assert evidence["negative_count"] == 3
    for metric in (
        "auprc",
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
        "mcc",
    ):
        assert metric in evidence["pooled"]
        assert metric in evidence["subject_macro"]
    assert "contributing_subject_count" in evidence["subject_macro"]["auprc"]


def test_validation_predictions_hold_metadata_without_waveforms(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])
    with np.load(
        run_dir / experiment.VALIDATION_PREDICTIONS_NAME, allow_pickle=False
    ) as payload:
        keys = set(payload.files)
        assert keys == {
            "stable_id",
            "subject_id",
            "record_id",
            "channel_index",
            "target_family",
            "context_flags",
            "label",
            "score",
        }
        assert payload["label"].size == 6
        assert not any(
            str(item).startswith("ltstdb:test") for item in payload["stable_id"]
        )
    assert "waveform" not in keys


# --------------------------------------------------------------------------
# Checkpoint identity and experiment lock
# --------------------------------------------------------------------------


def test_locked_inference_state_holds_weights_without_optimizer(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])

    inference = torch.load(
        run_dir / experiment.SELECTED_MODEL_NAME, map_location="cpu", weights_only=True
    )
    training_checkpoint = torch.load(
        run_dir / experiment.TRAINING_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=True,
    )
    assert "optimizer" not in inference
    assert set(training_checkpoint) == {"model", "optimizer"}
    assert inference.keys() == training_checkpoint["model"].keys()


def test_lock_binds_checkpoint_hash_and_records_test_as_null(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    result = _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])
    lock = validate_experiment_lock(run_dir)

    assert lock["test"] is None
    assert lock["status"] == "locked_for_one_shot_test"
    assert lock["git_dirty"] is False
    assert lock["trainable_parameter_count"] == TRAINABLE_PARAMETER_COUNT
    assert lock["seed"] == 2026
    assert lock["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert lock["checkpoint_bytes"] > 0
    assert lock["locked_inference_model"] == experiment.SELECTED_MODEL_NAME
    assert lock["waveform_cache_sha256"] == FAKE_CACHE_SHA
    assert lock["training_selection_sha256"] == FAKE_SELECTION
    assert lock["validation_threshold"] == result["validation_threshold"]
    assert lock["input_contract"]["samples"] == WINDOW
    assert lock["input_contract"]["physical_unit"] == "mV"
    assert lock["input_contract"]["dtype"] == "float32"
    assert lock["environment"]["amp_enabled"] is False
    assert lock["epoch_history_digest"]
    assert lock["command"]
    assert lock["total_duration_seconds"] >= 0


def test_lock_digest_is_deterministic_and_detects_tampering(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    _run(harness)
    run_dir = resolve_run_dir(harness["run_root"])
    lock_path = run_dir / EXPERIMENT_LOCK_NAME

    first = validate_experiment_lock(run_dir)
    second = validate_experiment_lock(run_dir)
    assert first["experiment_lock_sha256"] == second["experiment_lock_sha256"]

    tampered = json.loads(lock_path.read_text(encoding="utf-8"))
    tampered["validation_threshold"] = 0.123456
    lock_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="lock hash validation failed"):
        validate_experiment_lock(run_dir)


def test_lock_rejects_a_dirty_checkout_and_wrong_parameter_count(harness) -> None:
    prepared = _prepare(harness)
    arguments = {
        "command": "unit-test",
        "epoch_history": ({"epoch": 1},),
        "selected_epoch": 1,
        "selected_validation_auprc": 0.5,
        "threshold": 0.5,
        "checkpoint_sha256": "e" * 64,
        "checkpoint_bytes": 10,
        "training_checkpoint_sha256": "f" * 64,
        "validation_evidence_sha256": "0" * 64,
        "validation_predictions_sha256": None,
        "duration_seconds": 1.0,
    }
    prepared.report["git"]["git_dirty"] = True
    with pytest.raises(ValueError, match="clean Git checkout"):
        build_experiment_lock(prepared, **arguments)

    prepared.report["git"]["git_dirty"] = False
    prepared.report["model"]["trainable_parameter_count"] = 1
    with pytest.raises(ValueError, match="wrong parameter count"):
        build_experiment_lock(prepared, **arguments)


def test_model_is_frozen_after_lock_creation(harness, monkeypatch) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    captured: list[torch.nn.Module] = []
    real_model = experiment.B4CompactCNN

    class CapturingModel(real_model):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            captured.append(self)

    monkeypatch.setattr(experiment, "B4CompactCNN", CapturingModel)
    _run(harness)

    canonical = captured[-1]
    assert canonical.training is False
    assert all(not parameter.requires_grad for parameter in canonical.parameters())


# --------------------------------------------------------------------------
# Non-scientific labelling
# --------------------------------------------------------------------------


def test_synthetic_run_is_never_labelled_a_sealed_test_result(
    harness, monkeypatch
) -> None:
    _stub_training(monkeypatch, [0.4, 0.4, 0.4, 0.4, 0.4])
    result = _run(harness)

    assert result["test"] is None
    assert result["validation_evidence"]["evidence_class"] == (
        "development_validation_result"
    )
    assert "test" not in result["validation_evidence"]["partition"]
    assert result["validation_evidence"]["score_semantics"].startswith("uncalibrated")


def test_frozen_training_result_contract_is_unchanged() -> None:
    assert FrozenTrainingResult.__dataclass_fields__.keys() == {
        "history",
        "selected_checkpoint_epoch",
        "selected_validation_auprc",
        "validation_threshold",
    }
