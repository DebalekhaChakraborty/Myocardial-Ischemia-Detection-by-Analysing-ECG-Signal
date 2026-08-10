"""Synthetic end-to-end tests for the canonical P1 Stage-1 execution path.

These exercise the ACTUAL public scientific routes — cache materialization and
validation, real multi-epoch training, threshold selection, metrics, challenge
evidence, the lock and the Stage-1 suite — not only isolated guards.

All fixtures are synthetic. No test reads a real B4 run, the real feature
corpus, a real waveform source, or the sealed test.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import p1_experiment as p1x
from cardiosentinel.neural import physiology_fusion as p1
from cardiosentinel.neural.integrity import canonical_sha256

ROWS_TRAIN, ROWS_VALIDATION = 96, 64


def _environment():
    return {
        "python_version": "3.12.6",
        "torch_version": "2.13.0+cpu",
        "numpy_version": "2.3.2",
        "amp_enabled": False,
        "device": "cpu",
        "dependencies": {
            "installed_packages_sha256": (
                "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
            ),
            "key_dependencies": {
                "numpy": "2.3.2",
                "scikit-learn": "1.9.0",
                "scipy": "1.18.0",
                "torch": "2.13.0+cpu",
                "wfdb": "4.3.1",
            },
        },
    }


@pytest.fixture(autouse=True)
def _governance(monkeypatch):
    """Synthetic runtime/Git gates; populations relaxed to fixture size."""
    monkeypatch.setattr(
        p1x, "require_p1_runtime", lambda: (_environment(), "digest-abc")
    )
    monkeypatch.setattr(
        p1x, "require_clean_checkout", lambda: {"git_sha": "a" * 40, "git_dirty": False}
    )
    monkeypatch.setattr(
        p1x,
        "EXPECTED_POPULATIONS",
        {
            "train": {
                "total": ROWS_TRAIN,
                "positive": 32,
                "negative": 64,
                "subjects": 4,
            },
            "validation": {
                "total": ROWS_VALIDATION,
                "positive": 16,
                "negative": 48,
                "subjects": 3,
            },
        },
    )


def _population(rows: int, positives: int, subjects: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.zeros(rows, dtype=np.int64)
    labels[:positives] = 1
    subject_ids = [f"s{i % subjects}" for i in range(rows)]
    stable_ids = [f"ltstdb:r0:0:{i * 2500}:{(i + 1) * 2500}" for i in range(rows)]
    # A learnable signal so training is a real optimisation, not a no-op.
    embeddings = rng.normal(size=(rows, p1.EMBEDDING_DIM)).astype(np.float32)
    embeddings[labels == 1] += 0.9
    return stable_ids, subject_ids, labels, embeddings


def _cache(tmp_path, partition, rows, positives, subjects, seed):
    stable_ids, subject_ids, labels, embeddings = _population(
        rows, positives, subjects, seed
    )
    manifest = p1x.build_embedding_cache_manifest(
        partition=partition,
        stable_ids=stable_ids,
        embeddings=embeddings,
        labels=labels,
        subject_ids=subject_ids,
        git_sha="a" * 40,
        git_dirty=False,
        dependency_digest="digest-abc",
        encoder_receipt={"encoder_state_unchanged": True},
    )
    return p1x.P1EmbeddingCache(
        partition=partition,
        stable_ids=tuple(stable_ids),
        embeddings=embeddings,
        labels=labels,
        subject_ids=tuple(subject_ids),
        manifest=manifest,
    )


@pytest.fixture
def caches(tmp_path):
    return (
        _cache(tmp_path, "train", ROWS_TRAIN, 32, 4, 1),
        _cache(tmp_path, "validation", ROWS_VALIDATION, 16, 3, 2),
    )


def _raw_physiology(stable_ids, seed):
    rng = np.random.default_rng(seed)
    valid = p1.PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")
    rows = {}
    for index, key in enumerate(stable_ids):
        row = rng.normal(size=p1.PHYSIOLOGY_DIM) + 5.0
        row[valid] = 1.0
        if index == 0:
            row[3] = np.nan  # exercise imputation on the real path
        rows[str(key)] = row
    return rows


@pytest.fixture
def physiology(caches):
    train, validation = caches
    raw_train = _raw_physiology(train.stable_ids, 11)
    raw_validation = _raw_physiology(validation.stable_ids, 12)
    transform = p1.fit_physiology_transform(
        np.stack([raw_train[k] for k in train.stable_ids])
    )
    train_bundle = p1x.build_physiology_bundle(
        partition="train",
        stable_ids=train.stable_ids,
        raw_by_stable_id=raw_train,
        transform=transform,
    )
    validation_bundle = p1x.build_physiology_bundle(
        partition="validation",
        stable_ids=validation.stable_ids,
        raw_by_stable_id=raw_validation,
        transform=transform,
    )
    return transform, train_bundle, validation_bundle


@pytest.fixture
def challenge_set(caches, physiology):
    """A synthetic stand-in carrying the frozen selection identity."""
    _, validation = caches
    transform, _, validation_bundle = physiology
    return p1x.P1ChallengeSet(
        stable_ids=validation.stable_ids,
        target_families=tuple(
            ["rate_related_confounder"] * 30
            + ["axis_shift_confounder"] * 20
            + ["conduction_change_confounder"] * 14
        ),
        subject_ids=validation.subject_ids,
        embeddings=validation.embeddings,
        physiology=validation_bundle,
        selection_sha256=p1x.CHALLENGE_SELECTION_SHA256,
        counts={"rate_related_confounder": {"windows": 30, "subjects": 3}},
    )


# --------------------------------------------------------------------------
# Cache provenance: ordered identity and content
# --------------------------------------------------------------------------


def test_ordered_digest_changes_when_row_order_changes() -> None:
    ids = ["a", "b", "c"]
    assert p1x.ordered_stable_id_digest(ids) != p1x.ordered_stable_id_digest(
        ["c", "b", "a"]
    )
    assert p1x.ordered_stable_id_digest(ids) == p1x.ordered_stable_id_digest(list(ids))


def test_ordered_digest_refuses_duplicates() -> None:
    with pytest.raises(p1x.P1ExecutionError, match="duplicates"):
        p1x.ordered_stable_id_digest(["a", "a"])


def test_content_digest_changes_when_one_embedding_changes() -> None:
    matrix = np.zeros((4, p1.EMBEDDING_DIM), dtype=np.float32)
    before = p1x.embedding_content_digest(matrix)
    matrix[2, 7] = 1e-6
    assert p1x.embedding_content_digest(matrix) != before


def test_content_digest_distinguishes_dtype_and_shape() -> None:
    a = np.zeros((4, p1.EMBEDDING_DIM), dtype=np.float32)
    assert p1x.embedding_content_digest(a) != p1x.embedding_content_digest(
        a.astype(np.float64)
    )
    assert p1x.embedding_content_digest(a) != p1x.embedding_content_digest(a[:3])


def test_cache_manifest_binds_frozen_identities(caches) -> None:
    train, _ = caches
    manifest = train.manifest
    assert manifest["encoder_checkpoint_sha256"] == p1.B4B_CHECKPOINT_SHA256
    assert manifest["embedding_tap"] == p1.EMBEDDING_TAP
    assert manifest["embedding_dim"] == 128
    assert manifest["git_dirty"] is False
    assert manifest["test_accessed"] is False
    body = {k: v for k, v in manifest.items() if k != "cache_sha256"}
    assert canonical_sha256(body) == manifest["cache_sha256"]


def test_cache_manifest_refuses_wrong_population(tmp_path) -> None:
    stable_ids, subject_ids, labels, embeddings = _population(10, 5, 2, 3)
    with pytest.raises(p1x.P1ExecutionError, match="differs from the frozen"):
        p1x.build_embedding_cache_manifest(
            partition="train",
            stable_ids=stable_ids,
            embeddings=embeddings,
            labels=labels,
            subject_ids=subject_ids,
            git_sha="a" * 40,
            git_dirty=False,
            dependency_digest="d",
            encoder_receipt={},
        )


def test_cache_manifest_refuses_dirty_checkout(caches) -> None:
    train, _ = caches
    with pytest.raises(p1x.P1ExecutionError, match="clean Git"):
        p1x.build_embedding_cache_manifest(
            partition="train",
            stable_ids=train.stable_ids,
            embeddings=train.embeddings,
            labels=train.labels,
            subject_ids=train.subject_ids,
            git_sha="a" * 40,
            git_dirty=True,
            dependency_digest="d",
            encoder_receipt={},
        )


def test_cache_manifest_refuses_the_test_partition(caches) -> None:
    train, _ = caches
    with pytest.raises(p1.PhysiologyFusionError, match="never access"):
        p1x.build_embedding_cache_manifest(
            partition="test",
            stable_ids=train.stable_ids,
            embeddings=train.embeddings,
            labels=train.labels,
            subject_ids=train.subject_ids,
            git_sha="a" * 40,
            git_dirty=False,
            dependency_digest="d",
            encoder_receipt={},
        )


# --------------------------------------------------------------------------
# Materialization and reload
# --------------------------------------------------------------------------


class _StubEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def encode(self, waveforms):
        return torch.zeros(waveforms.shape[0], p1.EMBEDDING_DIM)


def _materialize(tmp_path, monkeypatch, rows=ROWS_TRAIN):
    stable_ids, subject_ids, labels, _ = _population(rows, 32, 4, 5)
    batches = [torch.zeros(rows, 1, 2500)]
    monkeypatch.setattr(p1x, "validate_p1_protocol", lambda *a, **k: "p")
    monkeypatch.setattr(p1x, "load_official_b4b_encoder", lambda d: _StubEncoder())
    return p1x.materialize_p1_embedding_cache(
        tmp_path / "b4b",
        batches,
        partition="train",
        stable_ids=stable_ids,
        labels=labels,
        subject_ids=subject_ids,
        cache_root=tmp_path / "cache",
    )


def test_materialized_cache_round_trips_and_revalidates(tmp_path, monkeypatch) -> None:
    cache = _materialize(tmp_path, monkeypatch)
    assert cache.embeddings.shape == (ROWS_TRAIN, 128)
    monkeypatch.setattr(p1x, "P1_PROTOCOL_SHA256", cache.manifest["p1_protocol_sha256"])
    monkeypatch.setattr(p1x, "FROZEN_DEPENDENCY_DIGEST", "digest-abc")
    reloaded = p1x.load_p1_embedding_cache(tmp_path / "cache", "train")
    assert reloaded.stable_ids == cache.stable_ids
    assert np.array_equal(reloaded.embeddings, cache.embeddings)


def test_existing_cache_is_refused_not_overwritten(tmp_path, monkeypatch) -> None:
    _materialize(tmp_path, monkeypatch)
    with pytest.raises(p1x.P1ExecutionError, match="already exists"):
        _materialize(tmp_path, monkeypatch)


def test_partial_cache_directory_blocks_overwrite(tmp_path, monkeypatch) -> None:
    """A crashed run leaves a directory with no manifest; it must still block."""
    directory = tmp_path / "cache" / "train"
    directory.mkdir(parents=True)
    (directory / p1x.CACHE_ARRAY_NAME).write_bytes(b"partial")
    with pytest.raises(p1x.P1ExecutionError, match="even if partial"):
        _materialize(tmp_path, monkeypatch)
    assert (directory / p1x.CACHE_ARRAY_NAME).read_bytes() == b"partial"


def test_tampered_cache_artifact_is_refused(tmp_path, monkeypatch) -> None:
    cache = _materialize(tmp_path, monkeypatch)
    monkeypatch.setattr(p1x, "P1_PROTOCOL_SHA256", cache.manifest["p1_protocol_sha256"])
    monkeypatch.setattr(p1x, "FROZEN_DEPENDENCY_DIGEST", "digest-abc")
    array = tmp_path / "cache" / "train" / p1x.CACHE_ARRAY_NAME
    with np.load(array, allow_pickle=False) as archive:
        columns = {k: archive[k] for k in archive.files}
    columns["embedding"][0, 0] += 1.0
    np.savez_compressed(array, **columns)
    with pytest.raises(p1x.P1ExecutionError, match="SHA-256 does not match"):
        p1x.load_p1_embedding_cache(tmp_path / "cache", "train")


def test_cache_manifest_tamper_is_refused(tmp_path, monkeypatch) -> None:
    cache = _materialize(tmp_path, monkeypatch)
    monkeypatch.setattr(p1x, "P1_PROTOCOL_SHA256", cache.manifest["p1_protocol_sha256"])
    monkeypatch.setattr(p1x, "FROZEN_DEPENDENCY_DIGEST", "digest-abc")
    path = tmp_path / "cache" / "train" / p1x.CACHE_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["rows"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(p1x.P1ExecutionError, match="digest validation"):
        p1x.load_p1_embedding_cache(tmp_path / "cache", "train")


def test_cache_load_refuses_the_test_partition(tmp_path) -> None:
    with pytest.raises(p1.PhysiologyFusionError, match="never access"):
        p1x.load_p1_embedding_cache(tmp_path, "test")


# --------------------------------------------------------------------------
# Deterministic training on the real path
# --------------------------------------------------------------------------


def test_head_construction_is_deterministic() -> None:
    a = p1x.build_deterministic_p1_head(p1.P1A_EXPERIMENT_ID)
    b = p1x.build_deterministic_p1_head(p1.P1A_EXPERIMENT_ID)
    for left, right in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(left, right)


def test_epoch_order_is_deterministic_and_epoch_dependent() -> None:
    assert np.array_equal(p1x.p1_epoch_order(50, 1), p1x.p1_epoch_order(50, 1))
    assert not np.array_equal(p1x.p1_epoch_order(50, 1), p1x.p1_epoch_order(50, 2))
    assert sorted(p1x.p1_epoch_order(50, 3).tolist()) == list(range(50))


def test_real_multi_epoch_training_selects_a_checkpoint(caches) -> None:
    train, validation = caches
    result = p1x.train_p1_arm(
        p1.P1A_EXPERIMENT_ID,
        train.embeddings,
        train.labels,
        validation.embeddings,
        validation.labels,
        max_epochs=6,
    )
    history = (
        result["epoch_history"]
    )
    assert len(history) >= 1
    required = {"epoch", "mean_training_loss", "validation_auprc"}
    assert all(required <= set(e) for e in history)
    assert result["selected_epoch"] >= 1
    saved = [e["epoch"] for e in history if e["checkpoint_saved"]]
    assert result["selected_epoch"] == saved[-1]
    assert np.isfinite(result["selected_validation_auprc"])


def test_training_is_reproducible(caches) -> None:
    train, validation = caches
    runs = [
        p1x.train_p1_arm(
            p1.P1A_EXPERIMENT_ID,
            train.embeddings,
            train.labels,
            validation.embeddings,
            validation.labels,
            max_epochs=3,
        )["selected_validation_auprc"]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_early_stopping_terminates_training(caches) -> None:
    train, validation = caches
    noise = np.random.default_rng(0).normal(
        size=validation.embeddings.shape
    ).astype(np.float32)
    result = p1x.train_p1_arm(
        p1.P1A_EXPERIMENT_ID,
        train.embeddings,
        train.labels,
        noise,
        validation.labels,
        max_epochs=30,
    )
    assert result["completed_epochs"] < 30
    assert result["stop_reason"] == "early_stopping"
    assert result["epoch_history"][-1]["early_stopping_patience"] == 4


def test_non_finite_state_aborts_without_repair(caches) -> None:
    train, validation = caches
    head = p1x.build_deterministic_p1_head(p1.P1A_EXPERIMENT_ID)
    with pytest.raises(p1x.P1ExecutionError, match="Non-finite mean training loss"):
        p1x._require_finite_state(head, float("nan"), 0.5, 1)
    with pytest.raises(p1x.P1ExecutionError, match="Non-finite validation AUPRC"):
        p1x._require_finite_state(head, 0.5, float("inf"), 1)


def test_p1a_refuses_physiology_and_p1b_requires_it(caches, physiology) -> None:
    train, _ = caches
    _, train_bundle, _ = physiology
    train_physiology = train_bundle.values
    with pytest.raises(p1x.P1ExecutionError, match="neural-only control"):
        p1x._features_for(p1.P1A_EXPERIMENT_ID, train.embeddings, train_physiology)
    with pytest.raises(p1x.P1ExecutionError, match="requires transformed physiology"):
        p1x._features_for(p1.P1B_EXPERIMENT_ID, train.embeddings, None)
    fused = p1x._features_for(p1.P1B_EXPERIMENT_ID, train.embeddings, train_physiology)
    assert fused.shape[1] == 128 + 18


def _code_without_prose(obj) -> str:
    """Source with docstrings stripped: assert on code, not on its own prose."""
    import ast

    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and (
            ast.get_docstring(node) is not None
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_encoder_never_enters_the_optimizer() -> None:
    source = _code_without_prose(p1x.train_p1_arm)
    assert "head.parameters()" in source
    # No encoder object, module or forward call appears in the training loop.
    for forbidden in ("encode(", "B4BTransformerCNN", "extract_frozen_embeddings"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Threshold, metrics, challenge
# --------------------------------------------------------------------------


def test_threshold_is_selected_on_validation_only(caches) -> None:
    _, validation = caches
    scores = np.linspace(0.05, 0.95, validation.labels.shape[0])
    threshold = p1x.select_p1_threshold(validation.labels, scores)
    assert threshold in set(scores.tolist())


def test_validation_evidence_has_pooled_and_macro(caches) -> None:
    _, validation = caches
    scores = np.linspace(0.05, 0.95, validation.labels.shape[0])
    evidence = p1x.p1_validation_evidence(
        validation.labels, scores, validation.subject_ids, 0.5
    )
    for key in ("auprc", "auroc", "f1", "sensitivity", "specificity", "mcc"):
        assert key in evidence["pooled"]
    assert "auprc" in evidence["subject_macro"]
    assert evidence["partition"] == "validation"
    assert evidence["score_semantics"].startswith("uncalibrated")


def test_challenge_evidence_uses_frozen_policy() -> None:
    families = np.asarray(
        ["rate_related_confounder"] * 3
        + ["axis_shift_confounder"] * 2
        + ["conduction_change_confounder"],
        dtype=np.str_,
    )
    scores = np.asarray([0.9, 0.2, 0.8, 0.7, 0.1, 0.3])
    subjects = np.asarray(["s1", "s1", "s2", "s1", "s2", "s3"], dtype=np.str_)
    evidence = p1x.p1_challenge_evidence(families, scores, subjects, 0.5)
    assert evidence["rate_related"]["false_positive_count"] == 2
    assert evidence["rate_related"]["evidence_status"] == "quantitative_secondary"
    assert evidence["axis_shift"]["false_positive_count"] == 1
    assert evidence["conduction_change"]["evidence_status"] == "exploratory_descriptive"
    assert evidence["conduction_change"]["bootstrap_permitted"] is False
    assert evidence["partition"] == "validation"


# --------------------------------------------------------------------------
# Canonical arm, lock and Stage-1 suite
# --------------------------------------------------------------------------


def _run_suite(tmp_path, caches, physiology, challenge_set):
    train, validation = caches
    transform, train_bundle, validation_bundle = physiology
    return p1x.run_p1_stage1_suite(
        run_root=tmp_path / "runs",
        train_cache=train,
        validation_cache=validation,
        transform=transform,
        train_physiology=train_bundle,
        validation_physiology=validation_bundle,
        challenge=challenge_set,
    )


def test_stage1_suite_runs_both_arms_and_locks_them(
    tmp_path, caches, physiology, challenge_set
) -> None:
    suite = _run_suite(tmp_path, caches, physiology, challenge_set)
    assert suite["arm_order"] == [p1.P1A_EXPERIMENT_ID, p1.P1B_EXPERIMENT_ID]
    assert set(suite["arm_results"]) == set(p1x.P1_ARM_ORDER)
    assert suite["physiology_retained"] is None
    assert suite["retention_decision_performed"] is False
    assert suite["test_accessed"] is False
    body = {k: v for k, v in suite.items() if k != "p1_stage1_suite_sha256"}
    assert canonical_sha256(body) == suite["p1_stage1_suite_sha256"]
    for arm in p1x.P1_ARM_ORDER:
        directory = tmp_path / "runs" / arm
        for name in (
            p1x.EXPERIMENT_LOCK_NAME,
            p1x.EPOCH_HISTORY_NAME,
            p1x.VALIDATION_METRICS_NAME,
            p1x.VALIDATION_THRESHOLD_NAME,
            p1x.VALIDATION_PREDICTIONS_NAME,
            p1x.PHYSIOLOGY_TRANSFORM_NAME,
            p1x.SELECTED_MODEL_NAME,
            p1x.TRAINING_CHECKPOINT_NAME,
        ):
            assert (directory / name).exists(), f"{arm} missing {name}"


def test_locks_validate_and_bind_provenance(
    tmp_path, caches, physiology, challenge_set
) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    for arm in p1x.P1_ARM_ORDER:
        lock = p1x.validate_p1_lock(tmp_path / "runs" / arm)
        assert lock["experiment_id"] == arm
        assert lock["test"] is None
        assert lock["encoder_fine_tuned"] is False
        assert lock["encoder_checkpoint_sha256"] == p1.B4B_CHECKPOINT_SHA256
        assert lock["status"] == p1x.LOCK_STATUS
        if arm == p1.P1B_EXPERIMENT_ID:
            assert lock["physiology_transform_sha256"]
            assert lock["head"]["input_dim"] == 146
        else:
            assert lock["physiology_transform_sha256"] is None
            assert lock["head"]["input_dim"] == 128


def test_tampered_lock_is_refused(tmp_path, caches, physiology, challenge_set) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    path = tmp_path / "runs" / p1.P1A_EXPERIMENT_ID / p1x.EXPERIMENT_LOCK_NAME
    lock = json.loads(path.read_text())
    lock["selected_validation_auprc"] = 0.99
    path.write_text(json.dumps(lock))
    with pytest.raises(p1x.P1ExecutionError, match="hash validation failed"):
        p1x.validate_p1_lock(tmp_path / "runs" / p1.P1A_EXPERIMENT_ID)


def test_tampered_head_checkpoint_is_refused(
    tmp_path, caches, physiology, challenge_set
) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    directory = tmp_path / "runs" / p1.P1B_EXPERIMENT_ID
    (directory / p1x.SELECTED_MODEL_NAME).write_bytes(b"corrupt")
    with pytest.raises(p1x.P1ExecutionError, match="failed hash validation"):
        p1x.validate_p1_lock(directory)


def test_claimed_arm_cannot_be_rerun(
    tmp_path, caches, physiology, challenge_set
) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    with pytest.raises(p1.PhysiologyFusionError, match="already been claimed"):
        _run_suite(tmp_path, caches, physiology, challenge_set)


def test_post_claim_failure_writes_failed_receipt(
    tmp_path,
    caches,
    physiology,
    challenge_set,
) -> None:
    train, validation = caches
    transform, train_bundle, _ = physiology
    # Train physiology supplied for validation: refused after the claim exists.
    with pytest.raises(p1x.P1ExecutionError, match="Physiology bundle is for"):
        p1x.run_p1_arm(
            p1.P1B_EXPERIMENT_ID,
            run_root=tmp_path / "runs",
            train_cache=train,
            validation_cache=validation,
            transform=transform,
            challenge=challenge_set,
            train_physiology=train_bundle,
            validation_physiology=train_bundle,
        )
    directory = tmp_path / "runs" / p1.P1B_EXPERIMENT_ID
    status = json.loads((directory / "RUN_STATUS.json").read_text())
    assert status["status"] == "FAILED_OR_INTERRUPTED"
    assert status["human_review_required"] is True
    assert status["repeat_attempt_permitted"] is False
    assert directory.is_dir()  # claim never released


def test_suite_exposes_no_single_arm_or_retry_route() -> None:
    parameters = inspect.signature(p1x.run_p1_stage1_suite).parameters
    for forbidden in ("arm", "only", "retry", "force", "overwrite", "resume"):
        assert forbidden not in parameters
    source = inspect.getsource(p1x.run_p1_stage1_suite)
    assert "for arm in P1_ARM_ORDER" in source


# --------------------------------------------------------------------------
# Preflight and firewall on the real routes
# --------------------------------------------------------------------------


def test_preflight_is_read_only(tmp_path) -> None:
    report = p1x.p1_preflight(tmp_path / "runs", tmp_path / "cache")
    # A real gate: absent caches must NOT be reported as directly runnable.
    assert report["status"] == "embedding_cache_materialization_required"
    assert report["embedding_caches_ready"] is False
    assert report["embedding_cache"]["train"]["present"] is False
    assert report["models_constructed"] == 0
    assert report["artifacts_created"] == 0
    assert report["test_partition_access"] is None
    assert report["partitions_permitted"] == ["train", "validation"]
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "cache").exists()


def test_preflight_reports_a_claimed_attempt(
    tmp_path, caches, physiology, challenge_set
) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    report = p1x.p1_preflight(tmp_path / "runs", tmp_path / "cache")
    assert report["status"] == "attempt_already_claimed"
    assert all(report["canonical_arm_claimed"].values())


def test_official_module_never_names_sealed_test_machinery() -> None:
    import ast

    source = Path(p1x.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("sealed_test" in name for name in imported)
    # `sealed_test_state: "unopened"` is a negative-assertion FIELD and is
    # expected; the machinery itself must be absent.
    for forbidden in (
        "evaluate_locked_test",
        "SealedTestAccess",
        "TEST_ATTEMPT",
        "TEST_METRICS",
        "neural.sealed_test",
    ):
        assert forbidden not in source


# --------------------------------------------------------------------------
# Re-review blockers: binding, identity and checkpoint/patience separation
# --------------------------------------------------------------------------


def test_permuted_physiology_is_refused(caches, physiology) -> None:
    """A row permutation must be detected, not silently misaligned."""
    train, _ = caches
    _, train_bundle, _ = physiology
    permuted = p1x.P1PhysiologyBundle(
        partition="train",
        stable_ids=tuple(reversed(train_bundle.stable_ids)),
        values=train_bundle.values,
        schema_sha256=train_bundle.schema_sha256,
        transform_sha256=train_bundle.transform_sha256,
    )
    with pytest.raises(p1x.P1ExecutionError, match="exact row order"):
        p1x.require_aligned_physiology(permuted, train.stable_ids, "train")
    assert p1x.require_aligned_physiology(
        train_bundle, train.stable_ids, "train"
    ) is train_bundle.values


def test_physiology_join_refuses_missing_rows(caches, physiology) -> None:
    train, _ = caches
    transform, _, _ = physiology
    raw = _raw_physiology(train.stable_ids, 3)
    raw.pop(train.stable_ids[0])
    with pytest.raises(p1x.P1ExecutionError, match="no morphology_v1 record"):
        p1x.build_physiology_bundle(
            partition="train",
            stable_ids=train.stable_ids,
            raw_by_stable_id=raw,
            transform=transform,
        )


def test_wrong_challenge_selection_is_refused(
    tmp_path, caches, physiology, challenge_set
) -> None:
    forged = p1x.P1ChallengeSet(
        stable_ids=challenge_set.stable_ids,
        target_families=challenge_set.target_families,
        subject_ids=challenge_set.subject_ids,
        embeddings=challenge_set.embeddings,
        physiology=challenge_set.physiology,
        selection_sha256="0" * 64,
        counts=challenge_set.counts,
    )
    train, validation = caches
    transform, train_bundle, validation_bundle = physiology
    with pytest.raises(p1x.P1ExecutionError, match="frozen validation challenge"):
        p1x.run_p1_stage1_suite(
            run_root=tmp_path / "runs",
            train_cache=train,
            validation_cache=validation,
            transform=transform,
            train_physiology=train_bundle,
            validation_physiology=validation_bundle,
            challenge=forged,
        )


def test_challenge_cannot_be_omitted_from_the_suite() -> None:
    parameters = inspect.signature(p1x.run_p1_stage1_suite).parameters
    assert parameters["challenge"].default is inspect.Parameter.empty
    assert parameters["challenge"].kind is inspect.Parameter.KEYWORD_ONLY


def test_unlocked_encoder_cannot_be_stamped_as_official(tmp_path) -> None:
    """An arbitrary directory is not the canonical B4-B run."""
    (tmp_path / "b4b").mkdir()
    with pytest.raises(Exception):
        p1x.load_official_b4b_encoder(tmp_path / "b4b")
    source = inspect.getsource(p1x.materialize_p1_embedding_cache)
    assert "load_official_b4b_encoder" in source


def test_checkpoint_saves_below_the_patience_delta() -> None:
    """A 5e-7 gain selects a new checkpoint but must NOT reset patience.

    Conflating checkpoint selection with the early-stopping delta was a real
    defect in the previous head; this pins the reviewed B4 separation.
    """
    from cardiosentinel.neural.training import CheckpointTracker

    tracker = CheckpointTracker()
    first = tracker.update(1, 0.500000)
    assert first.save_checkpoint is True and first.patience == 0
    second = tracker.update(2, 0.500000 + 5e-7)
    assert second.save_checkpoint is True, "a numerical maximum must be saved"
    assert second.patience == 1, "a sub-delta gain must not reset patience"
    assert tracker.best_epoch == 2
    third = tracker.update(3, 0.400000)
    assert third.save_checkpoint is False and third.patience == 2


def test_official_stage1_orchestration_exists_without_manual_assembly() -> None:
    import cardiosentinel.neural.cli as neural_cli

    source = inspect.getsource(neural_cli.run_p1_command)
    assert "execute_p1_stage1" in source
    assert "SystemExit" not in source
    parameters = inspect.signature(p1x.execute_p1_stage1).parameters
    assert set(parameters) == {
        "run_root",
        "cache_root",
        "feature_root",
        "b4b_run_dir",
        "command",
    }
    for forbidden in ("arm", "only", "force", "retry", "overwrite"):
        assert forbidden not in parameters


@pytest.mark.parametrize(
    "artifact",
    [
        "VALIDATION_METRICS.json",
        "CHALLENGE_METRICS.json",
        "EPOCH_HISTORY.json",
        "PHYSIOLOGY_TRANSFORM.json",
    ],
)
def test_tampering_any_claim_bearing_artifact_is_refused(
    tmp_path, caches, physiology, challenge_set, artifact
) -> None:
    _run_suite(tmp_path, caches, physiology, challenge_set)
    directory = tmp_path / "runs" / p1.P1B_EXPERIMENT_ID
    payload = json.loads((directory / artifact).read_text())
    payload["tampered"] = True
    (directory / artifact).write_text(json.dumps(payload))
    with pytest.raises(p1x.P1ExecutionError, match="failed hash validation"):
        p1x.validate_p1_lock(directory)


def test_cache_load_refuses_altered_subject_assignment(tmp_path, monkeypatch) -> None:
    cache = _materialize(tmp_path, monkeypatch)
    monkeypatch.setattr(p1x, "P1_PROTOCOL_SHA256", cache.manifest["p1_protocol_sha256"])
    monkeypatch.setattr(p1x, "FROZEN_DEPENDENCY_DIGEST", "digest-abc")
    array = tmp_path / "cache" / "train" / p1x.CACHE_ARRAY_NAME
    with np.load(array, allow_pickle=False) as archive:
        columns = {k: archive[k] for k in archive.files}
    columns["subject_id"] = np.asarray(
        ["sX"] * len(columns["subject_id"]), dtype=np.str_
    )
    np.savez_compressed(array, **columns)
    manifest_path = tmp_path / "cache" / "train" / p1x.CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = p1x.sha256_file(array)
    manifest.pop("cache_sha256")
    manifest["cache_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(p1x.P1ExecutionError, match="subject assignment"):
        p1x.load_p1_embedding_cache(tmp_path / "cache", "train")
