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


def test_benchmark_measures_median_and_p95(
    locked_run, monkeypatch, conforming_env
) -> None:
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


def test_isolated_benchmark_spawns_a_fresh_subprocess(locked_run, monkeypatch) -> None:
    """Fresh-process isolation is required because ru_maxrss never decreases.

    The spawn is verified deterministically so the assertion does not depend on
    whether the host happens to satisfy the production environment gate.
    """
    import subprocess as subprocess_module

    captured: list[list[str]] = []
    payload = {"experiment_id": "synthetic_v1", "process_isolated": True}

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    def spy(command, **kwargs):
        captured.append(command)
        return Completed()

    monkeypatch.setattr(benchmark.subprocess, "run", spy)
    result = benchmark.benchmark_locked_model_isolated(
        locked_run, official_model="B4-A"
    )

    assert result == payload
    command = captured[0]
    assert command[0].endswith("python") or "python" in command[0]
    assert command[1:3] == ["-m", "cardiosentinel.neural.resource_benchmark"]
    assert command[3] == str(locked_run.resolve())
    assert command[4] == "B4-A"
    assert subprocess_module is benchmark.subprocess


def test_isolated_child_enforces_the_environment_gate(locked_run) -> None:
    """A real child process must apply the gate itself, with no bypass."""
    import inspect

    source = inspect.getsource(benchmark.measure_locked_model)
    assert "_require_exact_benchmark_environment" in source
    assert "validate_resource_benchmark_protocol" in source
    # No environment variable or parameter can skip either gate.
    parameters = inspect.signature(benchmark.measure_locked_model).parameters
    for forbidden in ("skip_environment", "allow_environment", "force"):
        assert forbidden not in parameters
    module_source = Path(benchmark.__file__).read_text(encoding="utf-8")
    assert "os.environ" not in module_source
    assert "getenv" not in module_source


def test_isolated_benchmark_refuses_an_invalid_run(tmp_path) -> None:
    with pytest.raises(benchmark.LockedModelError, match="Isolated benchmark failed"):
        benchmark.benchmark_locked_model_isolated(tmp_path)


# --------------------------------------------------------------------------
# Official resource benchmark hardening
# --------------------------------------------------------------------------


def _conforming_environment(**overrides):
    """A fully synthetic environment satisfying the gate.

    Built from literals rather than the real host so the tests are identical on
    any machine. The production gate itself is never relaxed.
    """
    environment = {
        "python_version": "3.12.6",
        "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.2",
        "platform": "synthetic-platform",
        "cpu_model": "synthetic-cpu",
        "device": "cpu",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "dependency_digest": B4A_DEPENDENCY_DIGEST,
        "dependencies": {
            "installed_packages_sha256": B4A_DEPENDENCY_DIGEST,
            "key_dependencies": {
                "numpy": "2.3.2", "scikit-learn": "1.9.0", "scipy": "1.18.0",
                "torch": "2.13.0+cpu", "wfdb": "4.3.1",
            },
        },
    }
    environment.update(overrides)
    return environment


@pytest.fixture
def conforming_env(monkeypatch):
    monkeypatch.setattr(
        benchmark, "benchmark_environment", lambda: _conforming_environment()
    )


def _official_lock(directory: Path, model_key: str) -> dict:
    """Build a synthetic lock that satisfies the frozen official mapping."""
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.integrity import canonical_sha256
    from cardiosentinel.neural.protocol import B4_PROTOCOL_SHA256

    specification = benchmark.OFFICIAL_MODELS[model_key]
    factory = benchmark.SUPPORTED_ARCHITECTURES[specification["architecture"]]
    torch.save(factory().state_dict(), directory / "model_selected.pt")
    checkpoint = directory / "model_selected.pt"
    lock = {
        "experiment_id": specification["experiment_id"],
        "status": "locked_for_one_shot_test",
        "git_dirty": False,
        "protocol_sha256": B4_PROTOCOL_SHA256,
        "model": {
            "architecture": specification["architecture"],
            "verified_against_constructed_model": True,
        },
        "locked_inference_model": "model_selected.pt",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "trainable_parameter_count": specification["trainable_parameter_count"],
        "environment_dependency_digest": B4A_DEPENDENCY_DIGEST,
        "test": None,
    }
    if specification["requires_architecture_protocol"]:
        lock["architecture_protocol_sha256"] = ARCHITECTURE_PROTOCOL_SHA256
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    (directory / "EXPERIMENT_LOCK.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return lock


@pytest.fixture
def official_runs(tmp_path):
    directories = {}
    for key in benchmark.OFFICIAL_ORDER:
        directory = tmp_path / key
        directory.mkdir()
        _official_lock(directory, key)
        directories[key] = directory
    return directories


def test_resource_protocol_sha_is_enforced(tmp_path) -> None:
    assert (
        benchmark.validate_resource_benchmark_protocol()
        == benchmark.RESOURCE_PROTOCOL_SHA256
    )
    impostor = tmp_path / "fake.md"
    impostor.write_text("not the protocol", encoding="utf-8")
    with pytest.raises(benchmark.ResourceBenchmarkError, match="frozen SHA-256"):
        benchmark.validate_resource_benchmark_protocol(impostor)


def test_protocol_mismatch_refuses_before_any_forward(
    official_runs, monkeypatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        benchmark, "validate_resource_benchmark_protocol",
        lambda *a, **k: (_ for _ in ()).throw(
            benchmark.ResourceBenchmarkError("frozen SHA-256 mismatch")
        ),
    )
    monkeypatch.setattr(
        benchmark, "load_locked_model",
        lambda *a, **k: calls.append(1),
    )
    with pytest.raises(benchmark.ResourceBenchmarkError, match="SHA-256"):
        benchmark.measure_locked_model(official_runs["B4-A"])
    assert calls == []


def test_environment_mismatch_refuses_before_any_forward(
    official_runs, monkeypatch
) -> None:
    loaded: list[int] = []
    broken = _conforming_environment()
    broken["dependencies"]["installed_packages_sha256"] = "9" * 64
    monkeypatch.setattr(benchmark, "benchmark_environment", lambda: broken)
    monkeypatch.setattr(
        benchmark, "load_locked_model", lambda *a, **k: loaded.append(1)
    )
    with pytest.raises(ValueError, match="exact B4-A dependency snapshot"):
        benchmark.measure_locked_model(official_runs["B4-B"])
    assert loaded == []


def test_environment_version_mismatch_refuses_before_any_forward(
    official_runs, monkeypatch
) -> None:
    loaded: list[int] = []
    broken = _conforming_environment(torch_version="9.9.9")
    monkeypatch.setattr(benchmark, "benchmark_environment", lambda: broken)
    monkeypatch.setattr(
        benchmark, "load_locked_model", lambda *a, **k: loaded.append(1)
    )
    with pytest.raises(ValueError, match="Refusing the scientific run"):
        benchmark.measure_locked_model(official_runs["B4-C"])
    assert loaded == []


def test_official_mapping_rejects_a_wrong_experiment(official_runs) -> None:
    path = official_runs["B4-B"] / "EXPERIMENT_LOCK.json"
    payload = json.loads(path.read_text())
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="requires experiment_id"
    ):
        benchmark._require_official_lock(payload, "B4-C")


def test_official_mapping_rejects_wrong_parameter_count(official_runs) -> None:
    payload = json.loads(
        (official_runs["B4-A"] / "EXPERIMENT_LOCK.json").read_text()
    )
    payload["trainable_parameter_count"] = 12345
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="trainable parameters"
    ):
        benchmark._require_official_lock(payload, "B4-A")


def test_official_lock_requires_verified_identity_and_clean_tree(
    official_runs,
) -> None:
    payload = json.loads(
        (official_runs["B4-C"] / "EXPERIMENT_LOCK.json").read_text()
    )
    dirty = {**payload, "git_dirty": True}
    with pytest.raises(benchmark.ResourceBenchmarkError, match="clean Git checkout"):
        benchmark._require_official_lock(dirty, "B4-C")

    unverified = json.loads(json.dumps(payload))
    unverified["model"]["verified_against_constructed_model"] = False
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="verified constructed-model"
    ):
        benchmark._require_official_lock(unverified, "B4-C")


def test_official_lock_requires_frozen_protocol_digests(official_runs) -> None:
    payload = json.loads(
        (official_runs["B4-B"] / "EXPERIMENT_LOCK.json").read_text()
    )
    wrong_b4 = {**payload, "protocol_sha256": "0" * 64}
    with pytest.raises(benchmark.ResourceBenchmarkError, match="B4_PROTOCOL_V1"):
        benchmark._require_official_lock(wrong_b4, "B4-B")

    wrong_arch = {**payload, "architecture_protocol_sha256": "0" * 64}
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="architecture selection protocol"
    ):
        benchmark._require_official_lock(wrong_arch, "B4-B")


def test_official_lock_requires_the_frozen_dependency_digest(official_runs) -> None:
    payload = json.loads(
        (official_runs["B4-A"] / "EXPERIMENT_LOCK.json").read_text()
    )
    payload["environment_dependency_digest"] = "5" * 64
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="frozen dependency digest"
    ):
        benchmark._require_official_lock(payload, "B4-A")


def test_b4a_historical_environment_equivalent_is_accepted(official_runs) -> None:
    """B4-A predates the explicit field; the nested equivalent must satisfy it."""
    payload = json.loads(
        (official_runs["B4-A"] / "EXPERIMENT_LOCK.json").read_text()
    )
    del payload["environment_dependency_digest"]
    payload["environment"] = {
        "dependencies": {"installed_packages_sha256": B4A_DEPENDENCY_DIGEST}
    }
    benchmark._require_official_lock(payload, "B4-A")


def test_checkpoint_byte_size_mismatch_refuses(official_runs) -> None:
    path = official_runs["B4-A"] / "EXPERIMENT_LOCK.json"
    payload = json.loads(path.read_text())
    payload["checkpoint_bytes"] = 12
    from cardiosentinel.neural.integrity import canonical_sha256

    payload.pop("experiment_lock_sha256")
    payload["experiment_lock_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(benchmark.LockedModelError, match="byte size does not match"):
        benchmark.validate_locked_model(official_runs["B4-A"])


def test_p95_uses_nearest_rank_ceil() -> None:
    samples = list(range(1, 501))
    assert benchmark.nearest_rank_p95(samples) == 475
    assert benchmark.nearest_rank_p95([10]) == 10
    assert benchmark.nearest_rank_p95(list(range(1, 21))) == 19


def test_median_remains_the_tie_break_statistic(
    official_runs, monkeypatch, conforming_env
) -> None:
    monkeypatch.setattr(benchmark, "WARMUP_CALLS", 2)
    monkeypatch.setattr(benchmark, "MEASURED_CALLS", 5)
    result = benchmark.measure_locked_model(official_runs["B4-A"])

    assert result["tie_break_statistic"] == "median_latency_ms_per_window"
    assert result["p95_definition"] == "nearest_rank ceil(0.95*N)"
    assert result["resource_benchmark_protocol_sha256"] == (
        benchmark.RESOURCE_PROTOCOL_SHA256
    )


def test_result_digest_covers_the_protocol_sha(
    official_runs, monkeypatch, conforming_env
) -> None:
    from cardiosentinel.neural.integrity import canonical_sha256

    monkeypatch.setattr(benchmark, "WARMUP_CALLS", 1)
    monkeypatch.setattr(benchmark, "MEASURED_CALLS", 3)
    result = benchmark.measure_locked_model(official_runs["B4-B"])

    recorded = result.pop("benchmark_result_sha256")
    assert recorded == canonical_sha256(result)
    tampered = {**result, "resource_benchmark_protocol_sha256": "0" * 64}
    assert canonical_sha256(tampered) != recorded


# --------------------------------------------------------------------------
# Official suite
# --------------------------------------------------------------------------


def _child_payload(official_model, environment=None, **overrides):
    """A synthetic child payload carrying a correctly re-derivable digest."""
    from cardiosentinel.neural.integrity import canonical_sha256

    env = environment(official_model) if environment else {
        "python_version": "3.12.6", "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.2", "dependency_digest": B4A_DEPENDENCY_DIGEST,
        "platform": "Linux-x", "cpu_model": "Xeon", "device": "cpu",
        "intra_op_threads": 1, "inter_op_threads": 1,
    }
    payload = {
        "official_model": official_model,
        "experiment_lock_sha256": f"lock-{official_model}",
        "checkpoint_sha256": f"ckpt-{official_model}",
        "resource_benchmark_protocol_sha256": benchmark.RESOURCE_PROTOCOL_SHA256,
        "process_isolated": True,
        "dataset_accessed": False,
        "trainable_parameter_count": 1,
        "fp32_parameter_payload_bytes": 4,
        "locked_checkpoint_bytes": 8,
        "median_latency_ms_per_window": 1.0,
        "p95_latency_ms_per_window": 2.0,
        "peak_rss": 1000,
        "peak_rss_units": "kibibytes",
        "peak_rss_available": True,
        "environment": env,
    }
    payload.update(overrides)
    payload["benchmark_result_sha256"] = canonical_sha256(payload)
    return payload


def _fake_child(order_log, environment=None, mutate=None):
    def runner(run_dir, *, official_model=None, timeout_seconds=900.0):
        order_log.append(official_model)
        payload = _child_payload(official_model, environment)
        if mutate is not None:
            payload = mutate(official_model, payload)
        return payload
    return runner


def _suite(directories, run_root, runner, **kwargs):
    """Drive the private implementation, which alone accepts a runner."""
    return benchmark._run_official_resource_suite_impl(
        directories, run_root,
        command=kwargs.pop("command", "unit-test"),
        timeout_seconds=kwargs.pop("timeout_seconds", 900.0),
        runner=runner,
    )


def test_official_suite_requires_exactly_three_models(tmp_path) -> None:
    order: list[str] = []
    for bad in (
        {"B4-A": tmp_path},
        {"B4-A": tmp_path, "B4-B": tmp_path},
        {"B4-A": tmp_path, "B4-B": tmp_path, "B4-C": tmp_path, "B4-D": tmp_path},
    ):
        with pytest.raises(
            benchmark.ResourceBenchmarkError, match="exactly B4-A, B4-B and B4-C"
        ):
            _suite(bad, tmp_path / "runs", _fake_child(order))
    assert order == []


def test_official_suite_uses_the_frozen_order(tmp_path) -> None:
    order: list[str] = []
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    suite = _suite(directories, tmp_path / "runs", _fake_child(order))

    assert order == ["B4-A", "B4-B", "B4-C"]
    assert suite["candidate_order"] == ["B4-A", "B4-B", "B4-C"]


def test_official_suite_binds_every_required_identity(tmp_path) -> None:
    from cardiosentinel.neural.integrity import canonical_sha256

    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    suite = _suite(directories, tmp_path / "runs", _fake_child([]))

    assert suite["resource_benchmark_protocol_sha256"] == (
        benchmark.RESOURCE_PROTOCOL_SHA256
    )
    assert suite["architecture_protocol_sha256"] == ARCHITECTURE_PROTOCOL_SHA256
    assert suite["b4_protocol_sha256"]
    assert suite["suite_attempt_sha256"]
    assert set(suite["experiment_lock_sha256"]) == set(benchmark.OFFICIAL_ORDER)
    assert set(suite["checkpoint_sha256"]) == set(benchmark.OFFICIAL_ORDER)
    assert set(suite["benchmark_result_sha256"]) == set(benchmark.OFFICIAL_ORDER)
    assert suite["dataset_accessed"] is False
    assert suite["test_accessed"] is False
    assert suite["suite_duration_seconds"] >= 0

    recorded = suite.pop("resource_benchmark_suite_sha256")
    assert recorded == canonical_sha256(suite)
    # The combined digest covers all three individual result hashes.
    tampered = json.loads(json.dumps(suite))
    tampered["benchmark_result_sha256"]["B4-C"] = "changed"
    assert canonical_sha256(tampered) != recorded


def test_official_suite_refuses_a_differing_host(tmp_path) -> None:
    def environment(model):
        base = {
            "python_version": "3.12.6", "torch_version": "2.13.0+cpu",
            "numpy_version": "2.3.2", "dependency_digest": B4A_DEPENDENCY_DIGEST,
            "platform": "Linux-x", "cpu_model": "Xeon", "device": "cpu",
            "intra_op_threads": 1, "inter_op_threads": 1,
        }
        if model == "B4-C":
            base["cpu_model"] = "DifferentCPU"
        return base

    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="requires one host"
    ):
        _suite(
            directories, tmp_path / "runs",
            _fake_child([], environment=environment),
        )


def test_official_suite_requires_single_intra_op_thread(tmp_path) -> None:
    def environment(model):
        return {
            "python_version": "3.12.6", "torch_version": "2.13.0+cpu",
            "numpy_version": "2.3.2", "dependency_digest": B4A_DEPENDENCY_DIGEST,
            "platform": "Linux-x", "cpu_model": "Xeon", "device": "cpu",
            "intra_op_threads": 4, "inter_op_threads": 1,
        }

    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    with pytest.raises(
        benchmark.ResourceBenchmarkError, match="intra-op threads == 1"
    ):
        _suite(
            directories, tmp_path / "runs",
            _fake_child([], environment=environment),
        )


def test_attempt_is_claimed_before_the_first_measurement(tmp_path) -> None:
    run_root = tmp_path / "runs"
    observed: list[bool] = []

    def runner(run_dir, *, official_model=None, timeout_seconds=900.0):
        attempt = run_root / benchmark.SUITE_DIR_NAME / benchmark.SUITE_ATTEMPT_NAME
        observed.append(attempt.is_file())
        return _fake_child([])(run_dir, official_model=official_model)

    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    _suite(directories, run_root, runner)

    assert observed == [True, True, True]


def test_existing_attempt_refuses_a_second_official_suite(tmp_path) -> None:
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    run_root = tmp_path / "runs"
    _suite(directories, run_root, _fake_child([]))
    with pytest.raises(benchmark.ResourceBenchmarkError, match="already exists"):
        _suite(directories, run_root, _fake_child([]))


def test_failed_suite_cannot_selectively_retry(tmp_path) -> None:
    run_root = tmp_path / "runs"
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}

    def explode(run_dir, *, official_model=None, timeout_seconds=900.0):
        if official_model == "B4-B":
            raise RuntimeError("simulated child failure")
        return _fake_child([])(run_dir, official_model=official_model)

    with pytest.raises(RuntimeError, match="simulated child failure"):
        _suite(directories, run_root, explode)

    attempt = json.loads(
        (run_root / benchmark.SUITE_DIR_NAME / benchmark.SUITE_ATTEMPT_NAME).read_text()
    )
    assert attempt["attempt_status"] == benchmark.SUITE_STATUS_FAILED
    assert attempt["human_review_required"] is True
    assert attempt["selective_candidate_retry_permitted"] is False
    assert attempt["repeat_attempt_permitted"] is False

    # Neither the failed model alone nor the whole suite may run again.
    with pytest.raises(benchmark.ResourceBenchmarkError, match="already exists"):
        _suite(directories, run_root, _fake_child([]))


def test_no_force_or_overwrite_api_exists() -> None:
    import ast
    import inspect

    parameters = inspect.signature(benchmark.run_official_resource_suite).parameters
    for forbidden in (
        "force", "best_of", "retry_one", "rerun_candidate", "overwrite", "repeat",
    ):
        assert forbidden not in parameters

    tree = ast.parse(Path(benchmark.__file__).read_text(encoding="utf-8"))
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for forbidden in ("force", "best_of", "retry", "overwrite"):
        assert not any(forbidden in name for name in names)
    calls = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "unlink" not in calls and "rmtree" not in calls


# --------------------------------------------------------------------------
# Sealed official execution path
# --------------------------------------------------------------------------


def test_public_suite_exposes_no_runner_injection() -> None:
    import inspect

    parameters = inspect.signature(benchmark.run_official_resource_suite).parameters
    for forbidden in (
        "_runner", "runner", "backend", "executor", "measurement_function",
        "measure", "child", "force", "best_of", "retry_one", "overwrite",
    ):
        assert forbidden not in parameters, forbidden
    assert set(parameters) == {
        "run_directories", "run_root", "command", "timeout_seconds"
    }


def test_public_suite_hard_wires_the_isolated_runner() -> None:
    import inspect

    source = inspect.getsource(benchmark.run_official_resource_suite)
    assert "runner=benchmark_locked_model_isolated" in source
    assert "_run_official_resource_suite_impl" in source


def test_public_suite_actually_invokes_the_isolated_runner(
    tmp_path, monkeypatch
) -> None:
    """The public entry point must reach the real isolated-subprocess runner."""
    seen: list[str] = []

    def spy(run_dir, *, official_model=None, timeout_seconds=900.0):
        seen.append(official_model)
        return _child_payload(official_model)

    monkeypatch.setattr(benchmark, "benchmark_locked_model_isolated", spy)
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    benchmark.run_official_resource_suite(directories, tmp_path / "runs")

    assert seen == ["B4-A", "B4-B", "B4-C"]


def test_private_helper_is_not_exported() -> None:
    import ast

    tree = ast.parse(Path(benchmark.__file__).read_text(encoding="utf-8"))
    exported = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        )
    ]
    for assignment in exported:
        names = {
            element.value for element in assignment.value.elts
            if isinstance(element, ast.Constant)
        }
        assert "_run_official_resource_suite_impl" not in names
    # The private helper is underscore-prefixed, so `from module import *`
    # cannot pick it up regardless.
    assert benchmark._run_official_resource_suite_impl.__name__.startswith("_")


# --------------------------------------------------------------------------
# Child result digest validation
# --------------------------------------------------------------------------


def _reject(tmp_path, mutate, match):
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    run_root = tmp_path / "runs"
    with pytest.raises(benchmark.ResourceBenchmarkError, match=match):
        _suite(directories, run_root, _fake_child([], mutate=mutate))
    suite_dir = run_root / benchmark.SUITE_DIR_NAME
    assert not (suite_dir / benchmark.SUITE_RESULTS_NAME).exists()
    attempt = json.loads((suite_dir / benchmark.SUITE_ATTEMPT_NAME).read_text())
    assert attempt["attempt_status"] == benchmark.SUITE_STATUS_FAILED
    assert attempt["human_review_required"] is True


def test_forged_child_digest_is_rejected(tmp_path) -> None:
    def forge(model, payload):
        payload["median_latency_ms_per_window"] = 0.0001  # digest now stale
        return payload

    _reject(tmp_path, forge, "does not re-derive")


def test_missing_child_digest_is_rejected(tmp_path) -> None:
    def strip(model, payload):
        payload.pop("benchmark_result_sha256")
        return payload

    _reject(tmp_path, strip, "no benchmark_result_sha256")


def test_wrong_official_model_is_rejected(tmp_path) -> None:
    def swap(model, payload):
        return _child_payload("B4-A") if model == "B4-C" else payload

    _reject(tmp_path, swap, "Expected a B4-C child result")


def test_non_isolated_child_is_rejected(tmp_path) -> None:
    def shared(model, payload):
        return _child_payload(model, process_isolated=False)

    _reject(tmp_path, shared, "not process isolated")


def test_dataset_accessing_child_is_rejected(tmp_path) -> None:
    def touched(model, payload):
        return _child_payload(model, dataset_accessed=True)

    _reject(tmp_path, touched, "reports dataset access")


def test_wrong_child_protocol_sha_is_rejected(tmp_path) -> None:
    def wrong(model, payload):
        return _child_payload(model, resource_benchmark_protocol_sha256="0" * 64)

    _reject(tmp_path, wrong, "different resource protocol digest")


def test_valid_children_still_produce_a_complete_suite(tmp_path) -> None:
    directories = {name: tmp_path / name for name in benchmark.OFFICIAL_ORDER}
    run_root = tmp_path / "runs"
    suite = _suite(directories, run_root, _fake_child([]))

    suite_dir = run_root / benchmark.SUITE_DIR_NAME
    assert (suite_dir / benchmark.SUITE_RESULTS_NAME).is_file()
    attempt = json.loads((suite_dir / benchmark.SUITE_ATTEMPT_NAME).read_text())
    assert attempt["attempt_status"] == benchmark.SUITE_STATUS_COMPLETE
    assert suite["resource_benchmark_suite_sha256"]


def test_child_validation_does_not_alter_the_payload() -> None:
    payload = _child_payload("B4-A")
    original = json.loads(json.dumps(payload))
    benchmark._require_valid_child_result(
        payload, "B4-A", benchmark.RESOURCE_PROTOCOL_SHA256
    )
    assert payload == original
