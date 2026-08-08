"""Candidate runner and resource-benchmark tests, driven by synthetic fixtures.

Nothing here is a scientific result. No real B4-B or B4-C training or validation
is performed, no real locked checkpoint is benchmarked, and the sealed-test
partition is never referenced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import candidate_experiment as runner
from cardiosentinel.neural import resource_benchmark as benchmark
from cardiosentinel.neural.candidate_experiment import (
    ARCHITECTURE_PROTOCOL_SHA256,
    B4A_DEPENDENCY_DIGEST,
    CANDIDATE_SELECTORS,
    CandidateNumericalIntegrityError,
    expected_candidate_identity,
    require_exact_scientific_environment,
    require_numerical_integrity,
    resolve_candidate_run_dir,
    resolve_candidate_selector,
    run_candidate_train_validation,
    validate_architecture_protocol,
    validate_candidate_lock,
)
from cardiosentinel.neural.candidates import (
    B4B_EXPERIMENT_ID,
    B4C_EXPERIMENT_ID,
    B4CSSMCNN,
    B4BTransformerCNN,
)
from cardiosentinel.neural.experiment import (
    EXPERIMENT_LOCK_NAME,
    RUN_STATUS_NAME,
    STATUS_COMPLETE,
    STATUS_FAILED,
)
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.training import CompletedEpoch

WINDOW = 2500


def _environment(**overrides):
    payload = {
        "python_version": "3.12.6",
        "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.2",
        "amp_enabled": False,
        "dependencies": {
            "installed_packages_sha256": B4A_DEPENDENCY_DIGEST,
            "key_dependencies": {
                "numpy": "2.3.2",
                "scikit-learn": "1.9.0",
                "scipy": "1.18.0",
                "torch": "2.13.0+cpu",
                "wfdb": "4.3.1",
            },
        },
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# Identity, selectors and protocol
# --------------------------------------------------------------------------


def test_only_two_candidate_selectors_exist() -> None:
    assert CANDIDATE_SELECTORS == {
        "b4b": "B4B_cnn_transformer_v1",
        "b4c": "B4C_cnn_ssm_v1",
    }
    assert resolve_candidate_selector("b4b") == B4B_EXPERIMENT_ID
    assert resolve_candidate_selector("b4c") == B4C_EXPERIMENT_ID
    for bad in ("b4a", "b4d", "test", "B4B", ""):
        with pytest.raises(ValueError, match="Unknown B4 candidate selector"):
            resolve_candidate_selector(bad)


def test_expected_identities_construct_no_model() -> None:
    b4b = expected_candidate_identity(B4B_EXPERIMENT_ID)
    b4c = expected_candidate_identity(B4C_EXPERIMENT_ID)

    assert b4b["trainable_parameter_count"] == 309_809
    assert b4b["fp32_parameter_payload_bytes"] == 1_239_236
    assert b4b["architecture"] == "B4BTransformerCNN"
    assert b4c["trainable_parameter_count"] == 155_313
    assert b4c["fp32_parameter_payload_bytes"] == 621_252
    assert b4c["architecture"] == "B4CSSMCNN"
    for identity in (b4b, b4c):
        assert identity["identity_source"] == "frozen_protocol_constants"
        assert identity["verified_against_constructed_model"] is False


def test_architecture_protocol_hash_is_enforced(tmp_path) -> None:
    assert validate_architecture_protocol() == ARCHITECTURE_PROTOCOL_SHA256
    impostor = tmp_path / "fake.md"
    impostor.write_text("not the protocol", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its frozen"):
        validate_architecture_protocol(impostor)


def test_candidates_have_independent_run_directories(tmp_path) -> None:
    b4b = resolve_candidate_run_dir(tmp_path, B4B_EXPERIMENT_ID)
    b4c = resolve_candidate_run_dir(tmp_path, B4C_EXPERIMENT_ID)

    assert b4b != b4c
    assert b4b.name == "B4B_cnn_transformer_v1"
    assert b4c.name == "B4C_cnn_ssm_v1"


# --------------------------------------------------------------------------
# Exact environment governance
# --------------------------------------------------------------------------


def test_exact_environment_is_accepted() -> None:
    assert require_exact_scientific_environment(_environment()) == (
        B4A_DEPENDENCY_DIGEST
    )


def test_environment_digest_mismatch_refuses_the_run() -> None:
    broken = _environment()
    broken["dependencies"]["installed_packages_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="exact B4-A dependency snapshot"):
        require_exact_scientific_environment(broken)


@pytest.mark.parametrize(
    "key,value",
    [("python_version", "3.13.0"), ("torch_version", "2.9.0"),
     ("numpy_version", "2.0.0")],
)
def test_key_version_mismatch_refuses_the_run(key, value) -> None:
    with pytest.raises(ValueError, match="Refusing the scientific run"):
        require_exact_scientific_environment(_environment(**{key: value}))


def test_key_dependency_mismatch_refuses_the_run() -> None:
    broken = _environment()
    broken["dependencies"]["key_dependencies"]["scipy"] = "1.0.0"
    with pytest.raises(ValueError, match="scipy"):
        require_exact_scientific_environment(broken)


# --------------------------------------------------------------------------
# Numerical integrity
# --------------------------------------------------------------------------


def _epoch(loss=0.5, auprc=0.5) -> CompletedEpoch:
    return CompletedEpoch(
        epoch=1, mean_training_loss=loss, validation_auprc=auprc,
        checkpoint_saved=True, early_stopping_patience=0,
    )


def test_numerical_integrity_accepts_a_healthy_state() -> None:
    require_numerical_integrity(B4CSSMCNN(), _epoch())


def test_non_finite_training_loss_aborts() -> None:
    with pytest.raises(CandidateNumericalIntegrityError, match="training loss"):
        require_numerical_integrity(B4BTransformerCNN(), _epoch(loss=float("nan")))


def test_non_finite_validation_score_aborts() -> None:
    with pytest.raises(CandidateNumericalIntegrityError, match="validation score"):
        require_numerical_integrity(B4BTransformerCNN(), _epoch(auprc=float("inf")))


def test_non_finite_parameter_aborts() -> None:
    model = B4BTransformerCNN()
    with torch.no_grad():
        next(iter(model.parameters())).fill_(float("nan"))
    with pytest.raises(CandidateNumericalIntegrityError, match="parameter"):
        require_numerical_integrity(model, _epoch())


def test_non_finite_ssm_transition_aborts() -> None:
    # Finite parameters that overflow the derived gain: exp(90) saturates, so
    # only the derived-tensor check can catch this, not the parameter check.
    model = B4CSSMCNN()
    with torch.no_grad():
        model.blocks[0].log_step.fill_(90.0)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(CandidateNumericalIntegrityError, match="derived SSM"):
        require_numerical_integrity(model, _epoch())


def test_integrity_check_never_repairs_the_model() -> None:
    model = B4CSSMCNN()
    with torch.no_grad():
        model.blocks[0].log_step.fill_(90.0)
    with pytest.raises(CandidateNumericalIntegrityError):
        require_numerical_integrity(model, _epoch())

    # The offending value is left exactly as it was: nothing is clamped.
    assert float(model.blocks[0].log_step.detach().max()) == 90.0


# --------------------------------------------------------------------------
# Full runner lifecycle on synthetic fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Wire synthetic development inputs into the runner's module namespace."""
    from cardiosentinel.neural.metadata import B4MetadataIndex, B4WindowReference

    def reference(record, row, positive):
        start = row * WINDOW
        return B4WindowReference(
            stable_id=f"ltstdb:{record}:0:{start}:{start + WINDOW}",
            record_id=record, subject_id=f"s{record}", channel_index=0,
            start_sample=start, end_sample=start + WINDOW, partition="train"
            if record.startswith("tr") else "validation",
            target_family="ischemic_positive" if positive else "background_negative",
            context_flags=(),
        )

    train = tuple(reference("tr1", i, i < 2) for i in range(4))
    validation = tuple(reference("va1", i, i < 2) for i in range(4))
    indexes = {
        "train": B4MetadataIndex("train", train, 2, 2, 1, "selection"),
        "validation": B4MetadataIndex("validation", validation, 2, 2, 1),
    }

    class Cache:
        def __init__(self):
            generator = np.random.default_rng(5)
            self.waveforms = {
                part: generator.standard_normal((4, WINDOW)).astype(np.float32)
                for part in ("train", "validation")
            }
            self.manifest = {
                "waveform_cache_sha256": "d" * 64,
                "cache_complete": True,
                "equivalence_audit": {"exact_mismatches": 0, "result": "passed"},
            }

    monkeypatch.setattr(
        runner, "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(runner, "validate_frozen_protocol", lambda: "p" * 64)
    monkeypatch.setattr(
        runner, "validate_architecture_protocol",
        lambda *a, **k: ARCHITECTURE_PROTOCOL_SHA256,
    )
    monkeypatch.setattr(
        runner, "validate_development_feature_integrity",
        lambda root: {"development_feature_integrity_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        runner, "validate_development_source_integrity",
        lambda source, receipt: {"development_source_integrity_sha256": "s" * 64},
    )
    monkeypatch.setattr(runner, "build_development_indexes", lambda root: indexes)
    monkeypatch.setattr(runner, "validate_waveform_cache", lambda root, ix: Cache())
    monkeypatch.setattr(runner, "_require_frozen_counts", lambda ix: None)
    monkeypatch.setattr(runner, "_require_cache_identity", lambda m, f, s: None)
    monkeypatch.setattr(
        runner, "runtime_environment", lambda device, workers: _environment()
    )

    labels = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    import cardiosentinel.neural.training as training

    monkeypatch.setattr(training, "train_one_epoch", lambda *a: 0.25)
    monkeypatch.setattr(training, "validation_scores", lambda *a: (labels, scores))
    monkeypatch.setattr(training, "validation_auprc", lambda *a: 0.5)
    monkeypatch.setattr(runner, "validation_scores", lambda *a: (labels, scores))

    return {
        "source": tmp_path / "source",
        "feature_root": tmp_path / "features",
        "cache_root": tmp_path / "cache",
        "run_root": tmp_path / "runs",
    }


def _run(harness, selector):
    return run_candidate_train_validation(
        selector, harness["source"], harness["feature_root"],
        harness["cache_root"], harness["run_root"],
    )


def test_both_candidates_complete_independently(harness) -> None:
    first = _run(harness, "b4b")
    second = _run(harness, "b4c")

    assert first["status"] == STATUS_COMPLETE
    assert second["status"] == STATUS_COMPLETE
    assert first["experiment_id"] == B4B_EXPERIMENT_ID
    assert second["experiment_id"] == B4C_EXPERIMENT_ID
    # A completed B4-B must not block the one canonical B4-C run.
    assert first["run_dir"] != second["run_dir"]


def test_duplicate_candidate_run_is_refused(harness) -> None:
    _run(harness, "b4b")
    with pytest.raises(ValueError, match="exactly one canonical run"):
        _run(harness, "b4b")
    # The other candidate is unaffected.
    assert _run(harness, "b4c")["status"] == STATUS_COMPLETE


def test_interrupted_candidate_cannot_silently_restart(harness, monkeypatch) -> None:
    import cardiosentinel.neural.training as training

    monkeypatch.setattr(
        training, "train_one_epoch",
        lambda *a: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(harness, "b4c")

    run_dir = resolve_candidate_run_dir(harness["run_root"], B4C_EXPERIMENT_ID)
    status = json.loads((run_dir / RUN_STATUS_NAME).read_text())
    assert status["status"] == STATUS_FAILED
    assert status["human_review_required"] is True
    assert status["automatic_restart_performed"] is False

    monkeypatch.setattr(training, "train_one_epoch", lambda *a: 0.25)
    with pytest.raises(ValueError, match="requires documented human review"):
        _run(harness, "b4c")


def test_numerical_failure_aborts_the_run(harness, monkeypatch) -> None:
    import cardiosentinel.neural.training as training

    monkeypatch.setattr(training, "train_one_epoch", lambda *a: float("nan"))
    with pytest.raises(CandidateNumericalIntegrityError, match="training loss"):
        _run(harness, "b4b")

    run_dir = resolve_candidate_run_dir(harness["run_root"], B4B_EXPERIMENT_ID)
    status = json.loads((run_dir / RUN_STATUS_NAME).read_text())
    assert status["status"] == STATUS_FAILED
    assert status["numerical_integrity_failure"] is True
    assert not (run_dir / EXPERIMENT_LOCK_NAME).exists()


def test_dirty_git_refuses_before_any_run_directory(harness, monkeypatch) -> None:
    monkeypatch.setattr(
        runner, "git_provenance",
        lambda root: {"git_sha": "a" * 40, "git_dirty": True},
    )
    with pytest.raises(ValueError, match="clean Git checkout"):
        _run(harness, "b4b")
    assert not resolve_candidate_run_dir(
        harness["run_root"], B4B_EXPERIMENT_ID
    ).exists()


def test_environment_mismatch_refuses_before_any_run_directory(
    harness, monkeypatch
) -> None:
    monkeypatch.setattr(
        runner, "runtime_environment",
        lambda device, workers: _environment(torch_version="9.9.9"),
    )
    with pytest.raises(ValueError, match="Refusing the scientific run"):
        _run(harness, "b4b")
    assert not resolve_candidate_run_dir(
        harness["run_root"], B4B_EXPERIMENT_ID
    ).exists()


def test_cache_identity_mismatch_refuses(harness, monkeypatch) -> None:
    def refuse(manifest, feature, source):
        raise ValueError("B4 waveform cache identity differs: waveform_cache_sha256.")

    monkeypatch.setattr(runner, "_require_cache_identity", refuse)
    with pytest.raises(ValueError, match="cache identity differs"):
        _run(harness, "b4c")


def test_determinism_immediately_precedes_model_construction(
    harness, monkeypatch
) -> None:
    order: list[str] = []
    real = runner.initialize_determinism

    def spy(**kwargs):
        order.append("determinism")
        return real(**kwargs)

    class SpyB4B(B4BTransformerCNN):
        def __init__(self):
            order.append("model")
            super().__init__()

    monkeypatch.setattr(runner, "initialize_determinism", spy)
    monkeypatch.setitem(
        runner.CANDIDATE_SPECIFICATIONS[B4B_EXPERIMENT_ID], "factory", SpyB4B
    )
    _run(harness, "b4b")

    assert order.count("model") == 1
    canonical = order.index("model")
    assert order[canonical - 1] == "determinism"


def test_lock_binds_the_required_identities(harness) -> None:
    result = _run(harness, "b4c")
    run_dir = resolve_candidate_run_dir(harness["run_root"], B4C_EXPERIMENT_ID)
    lock = validate_candidate_lock(run_dir)

    assert lock["test"] is None
    assert lock["experiment_id"] == B4C_EXPERIMENT_ID
    assert lock["candidate_architecture"] == "B4CSSMCNN"
    assert lock["architecture_protocol_sha256"] == ARCHITECTURE_PROTOCOL_SHA256
    assert lock["environment_dependency_digest"] == B4A_DEPENDENCY_DIGEST
    assert lock["trainable_parameter_count"] == 155_313
    assert lock["fp32_parameter_payload_bytes"] == 621_252
    assert lock["model"]["verified_against_constructed_model"] is True
    assert lock["seed"] == 2026
    assert lock["git_dirty"] is False
    assert lock["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert lock["execution"]["candidate"] == "b4c"
    assert lock["validation_threshold"] == result["validation_threshold"]
    assert lock["epoch_history_digest"]


def test_lock_digest_detects_tampering(harness) -> None:
    _run(harness, "b4b")
    run_dir = resolve_candidate_run_dir(harness["run_root"], B4B_EXPERIMENT_ID)
    path = run_dir / EXPERIMENT_LOCK_NAME

    payload = json.loads(path.read_text())
    payload["selected_epoch"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lock hash validation failed"):
        validate_candidate_lock(run_dir)


def test_threshold_is_validation_only(harness) -> None:
    _run(harness, "b4b")
    run_dir = resolve_candidate_run_dir(harness["run_root"], B4B_EXPERIMENT_ID)
    payload = json.loads((run_dir / "VALIDATION_THRESHOLD.json").read_text())

    assert payload["selected_from"] == "validation"
    assert payload["test_informed"] is False


def test_checkpoint_selection_uses_frozen_validation_auprc(
    harness, monkeypatch
) -> None:
    import cardiosentinel.neural.training as training

    values = iter([0.1, 0.7, 0.3, 0.3, 0.3, 0.3])
    monkeypatch.setattr(
        training, "validation_auprc", lambda *a: next(values, 0.3)
    )
    result = _run(harness, "b4b")

    assert result["selected_epoch"] == 2
    assert result["selected_validation_auprc"] == 0.7


def test_runner_exposes_no_configuration_override() -> None:
    import inspect

    parameters = inspect.signature(run_candidate_train_validation).parameters
    for forbidden in (
        "lr", "learning_rate", "dropout", "threshold", "seed", "epochs",
        "architecture_config", "allow_dirty", "partition", "test",
        "require_clean",
    ):
        assert forbidden not in parameters


def test_candidate_cli_offers_only_two_candidates_and_no_overrides() -> None:
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    b4 = parser._subparsers._group_actions[0].choices["b4"]
    commands = next(a for a in b4._actions if getattr(a, "choices", None))
    candidate = commands.choices["candidate"]
    sub = next(a for a in candidate._actions if getattr(a, "choices", None))
    train = sub.choices["run-train-validation"]
    options = {opt for a in train._actions for opt in a.option_strings}

    for forbidden in (
        "--lr", "--dropout", "--threshold", "--seed", "--epochs",
        "--architecture-config", "--allow-dirty", "--test",
    ):
        assert forbidden not in options
    for bad in ("b4a", "b4d", "test"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["b4", "candidate", "run-train-validation", "--candidate", bad]
            )


def test_candidate_runner_never_names_a_test_partition() -> None:
    import ast

    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for argument in [*node.args, *(k.value for k in node.keywords)]
        if isinstance(argument, ast.Constant) and argument.value == "test"
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# Resource benchmark
# --------------------------------------------------------------------------


@pytest.fixture
def locked_run(tmp_path):
    """Build a synthetic locked B4-A-shaped run directory."""
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.integrity import canonical_sha256

    directory = tmp_path / "locked"
    directory.mkdir()
    model = B4CompactCNN()
    torch.save(model.state_dict(), directory / "model_selected.pt")
    lock = {
        "experiment_id": "synthetic_v1",
        "model": {"architecture": "B4CompactCNN"},
        "locked_inference_model": "model_selected.pt",
        "checkpoint_sha256": sha256_file(directory / "model_selected.pt"),
        "trainable_parameter_count": 87_089,
        "test": None,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return directory


def test_benchmark_constants_match_the_frozen_procedure() -> None:
    assert benchmark.BENCHMARK_BATCH_SIZE == 1
    assert benchmark.WARMUP_CALLS == 50
    assert benchmark.MEASURED_CALLS == 500
    assert benchmark.INTRA_OP_THREADS == 1
    assert benchmark.BENCHMARK_SEED == 2026
    assert set(benchmark.SUPPORTED_ARCHITECTURES) == {
        "B4CompactCNN", "B4BTransformerCNN", "B4CSSMCNN",
    }


def test_benchmark_input_is_deterministic_synthetic_and_batch_one() -> None:
    first = benchmark.benchmark_input()
    second = benchmark.benchmark_input()

    assert tuple(first.shape) == (1, 1, WINDOW)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_benchmark_never_touches_the_dataset() -> None:
    import ast

    tree = ast.parse(Path(benchmark.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    for forbidden in (
        "build_development_indexes", "validate_waveform_cache",
        "B4CachedWaveformDataset", "read_local_segment",
        "load_sealed_test_references", "build_optimizer",
    ):
        assert forbidden not in names, forbidden


def test_benchmark_validates_a_locked_model(locked_run) -> None:
    lock = benchmark.validate_locked_model(locked_run)

    assert lock["experiment_id"] == "synthetic_v1"
    model = benchmark.load_locked_model(locked_run, lock)
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())


def test_benchmark_refuses_an_unlocked_or_tampered_model(locked_run) -> None:
    path = locked_run / "EXPERIMENT_LOCK.json"
    payload = json.loads(path.read_text())
    payload["trainable_parameter_count"] = 5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(benchmark.LockedModelError, match="hash validation failed"):
        benchmark.validate_locked_model(locked_run)


def test_benchmark_refuses_a_missing_lock(tmp_path) -> None:
    with pytest.raises(benchmark.LockedModelError, match="No EXPERIMENT_LOCK"):
        benchmark.validate_locked_model(tmp_path)


def test_benchmark_refuses_a_corrupt_checkpoint(locked_run) -> None:
    (locked_run / "model_selected.pt").write_bytes(b"corrupt")
    with pytest.raises(benchmark.LockedModelError, match="SHA-256 does not match"):
        benchmark.validate_locked_model(locked_run)


def test_benchmark_measures_median_and_p95(locked_run, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "WARMUP_CALLS", 2)
    monkeypatch.setattr(benchmark, "MEASURED_CALLS", 5)
    result = benchmark.measure_locked_model(locked_run)

    assert result["benchmark_protocol"] == "B4_RESOURCE_BENCHMARK_V1"
    assert result["batch_size"] == 1
    assert result["timer"] == "time.perf_counter_ns"
    assert result["tie_break_statistic"] == "median_latency_ms_per_window"
    assert result["dataset_accessed"] is False
    assert result["median_latency_ms_per_window"] > 0
    assert result["p95_latency_ms_per_window"] >= result[
        "median_latency_ms_per_window"
    ]
    assert result["environment"]["intra_op_threads"] == 1
    assert result["peak_rss_available"] is True
    assert result["process_isolated"] is True
    assert len(result["benchmark_result_sha256"]) == 64
    assert result["trainable_parameter_count"] == 87_089


def test_isolated_benchmark_runs_in_a_fresh_subprocess(locked_run) -> None:
    """Fresh-process isolation is required because ru_maxrss never decreases."""
    result = benchmark.benchmark_locked_model_isolated(locked_run)

    assert result["experiment_id"] == "synthetic_v1"
    assert result["measured_calls"] == 500
    assert result["warmup_calls"] == 50
    assert result["process_isolated"] is True
    assert result["peak_rss"] > 0
    assert result["peak_rss_units"] in {"kibibytes", "bytes"}
    assert "fresh subprocess" in result["peak_rss_measurement_method"]


def test_isolated_benchmark_refuses_an_invalid_run(tmp_path) -> None:
    with pytest.raises(benchmark.LockedModelError, match="Isolated benchmark failed"):
        benchmark.benchmark_locked_model_isolated(tmp_path)
