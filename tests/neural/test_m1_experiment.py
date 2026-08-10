"""Execution-path tests for the canonical M1 Stage-1 route.

These cover the parts of M1 that are about *evidence integrity* rather than
causality: the train-only standardizer, the immutable stream cache and its
no-overwrite semantics, one-shot arm claims, lock digests, and the suite that
refuses to exist without all three arms.

Everything is synthetic. No real corpus, model, run directory or test partition
is touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m1_experiment
from cardiosentinel.neural.m1_experiment import (
    FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256,
    FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256,
    FROZEN_P1_EMBEDDING_CACHE_SHA256,
    FROZEN_P1_STAGE1_SUITE_SHA256,
    FROZEN_P1B_LOCK_SHA256,
    FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
    M1_ARM_ORDER,
    M1_STAGE1_RESULT_NAME,
    M1StreamRepresentation,
    _claim_stream_cache,
    _fuse,
    build_distance_standardizer,
    build_m1_lock,
    build_m1_stage1_result,
    build_stream_cache_manifest,
    load_stream_cache,
    require_frozen_upstream_identities,
    scan_test_artifacts,
    subject_false_positive_evidence,
    train_m1_arm,
    validate_m1_lock,
    validate_m1_stage1_results,
)
from cardiosentinel.neural.p1_experiment import EXPERIMENT_LOCK_NAME
from cardiosentinel.neural.patient_memory import (
    ALPHA_LONG,
    ALPHA_SHORT,
    M1D_EXPERIMENT_ID,
    M1S_EXPERIMENT_ID,
    REPRESENTATION_DIM,
    M1MemoryError,
    build_causal_streams,
    build_m1_head,
    claim_m1_run_directory,
    fit_distance_standardizer,
    generate_stream_memory,
    resolve_m1_run_dir,
)
from tests.neural.test_patient_memory import reference, vector


@pytest.fixture
def standardizer():
    rows = np.stack([vector(seed) for seed in range(120)]).astype(np.float64)
    return fit_distance_standardizer(rows, partition="train")


@pytest.fixture
def synthetic(standardizer):
    rows = [
        reference(record, channel, index)
        for record in ("rA", "rB")
        for channel in (0, 1)
        for index in range(6)
    ]
    values = {row.stable_id: vector(hash(row.stable_id) % 4000) for row in rows}
    streams = build_causal_streams(rows)
    memory = generate_stream_memory(
        streams,
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    representation = M1StreamRepresentation(
        partition="train",
        stable_ids=memory.stable_ids,
        matrix=np.stack([values[key] for key in memory.stable_ids]),
        streams=streams,
        reused_primary_rows=len(memory.stable_ids),
        newly_extracted_rows=0,
        primary_audit={"primary_rows_reused": len(memory.stable_ids)},
    )
    return memory, representation


UPSTREAM = {
    "p1_stage1_suite_sha256": FROZEN_P1_STAGE1_SUITE_SHA256,
    "p1b_experiment_lock_sha256": FROZEN_P1B_LOCK_SHA256,
    "physiology_transform_sha256": FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
    "p1_train_embedding_cache_sha256": FROZEN_P1_EMBEDDING_CACHE_SHA256["train"],
    "encoder_checkpoint_sha256": "b1301723909c641a0014c31f6daa9549d47ab231f0b07"
    "483e0de729aff5591c9",
}

MANIFEST_FIELDS = {
    "standardizer_sha256": "a" * 64,
    "feature_integrity_sha256": FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256,
    "source_integrity_sha256": FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256,
    "p1_stage1_suite_sha256": "b" * 64,
    "p1b_lock_sha256": "c" * 64,
    "physiology_transform_sha256": "d" * 64,
    "embedding_cache_sha256": "e" * 64,
    "git_sha": "f" * 40,
    "git_dirty": False,
    "dependency_digest": "0" * 64,
}


# --------------------------------------------------------------------------
# Distance standardizer
# --------------------------------------------------------------------------


def test_standardizer_is_fitted_on_train_only(standardizer, synthetic):
    _, representation = synthetic
    with pytest.raises(M1MemoryError, match="train"):
        build_distance_standardizer(
            M1StreamRepresentation(
                partition="validation",
                stable_ids=representation.stable_ids,
                matrix=representation.matrix,
                streams=representation.streams,
                reused_primary_rows=0,
                newly_extracted_rows=0,
                primary_audit={},
            ),
            primary_train_stable_ids=representation.stable_ids,
            upstream_identities=UPSTREAM,
        )


def test_standardizer_requires_the_frozen_primary_train_population(synthetic):
    _, representation = synthetic
    # The synthetic stream is far smaller than the frozen 374,452 train rows,
    # so the guard must refuse rather than quietly fit on whatever it is given.
    with pytest.raises(M1MemoryError, match="374452"):
        build_distance_standardizer(
            representation,
            primary_train_stable_ids=representation.stable_ids,
            upstream_identities=UPSTREAM,
        )


def test_standardizer_is_not_fitted_on_the_full_stream(standardizer):
    payload = standardizer.as_dict()
    assert payload["fitted_on_partition"] == "train"
    assert payload["fitted_on_full_stream"] is False
    assert payload["validation_statistics_used"] is False
    assert payload["patient_specific_normalization"] is False


def test_standardizer_zero_variance_dimension_takes_scale_one():
    rows = np.stack([vector(seed) for seed in range(50)]).astype(np.float64)
    rows[:, 7] = 3.25
    fitted = fit_distance_standardizer(rows, partition="train")
    assert 7 in fitted.zero_variance_dimensions
    assert fitted.scales[7] == 1.0


def test_standardizer_digest_detects_tampering(standardizer):
    payload = standardizer.as_dict()
    payload["means"] = [0.0] * REPRESENTATION_DIM
    with pytest.raises(M1MemoryError, match="digest"):
        type(standardizer).from_dict(payload)


# --------------------------------------------------------------------------
# Stream cache integrity
# --------------------------------------------------------------------------


def test_stream_cache_manifest_binds_every_required_identity(synthetic):
    memory, representation = synthetic
    manifest = build_stream_cache_manifest(memory, representation, **MANIFEST_FIELDS)
    for field in (
        "m1_protocol_sha256",
        "p1_stage1_suite_sha256",
        "p1b_experiment_lock_sha256",
        "encoder_checkpoint_sha256",
        "physiology_transform_sha256",
        "p1_embedding_cache_sha256",
        "split_sha256",
        "feature_corpus_sha256",
        "distance_standardizer_sha256",
        "partition",
        "full_stream_row_count",
        "stream_count",
        "record_ids",
        "channel_indices",
        "ordered_stable_id_sha256",
        "ordered_chronology_sha256",
        "representation_content_sha256",
        "d_short_content_sha256",
        "d_long_content_sha256",
        "history_count_sha256",
        "update_policy",
        "alpha_short",
        "alpha_long",
        "git_sha",
        "git_dirty",
        "environment_dependency_digest",
        "test_accessed",
    ):
        assert field in manifest, field
    assert manifest["alpha_short"] == ALPHA_SHORT
    assert manifest["alpha_long"] == ALPHA_LONG
    assert manifest["test_accessed"] is False
    assert manifest["contamination_safe"] is False
    assert manifest["label_independent_history"] is True
    assert manifest["stream_count"] == 4
    assert manifest["channel_indices"] == [0, 1]


def test_stream_cache_manifest_refuses_misaligned_rows(synthetic):
    memory, representation = synthetic
    shifted = M1StreamRepresentation(
        partition="train",
        stable_ids=representation.stable_ids[1:],
        matrix=representation.matrix[1:],
        streams=representation.streams,
        reused_primary_rows=0,
        newly_extracted_rows=0,
        primary_audit={},
    )
    with pytest.raises(M1MemoryError, match="misaligned"):
        build_stream_cache_manifest(memory, shifted, **MANIFEST_FIELDS)


def test_existing_stream_cache_directory_is_never_overwritten(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir(parents=True)
    with pytest.raises(M1MemoryError, match="human review"):
        _claim_stream_cache(directory, "train")


def test_partial_stream_cache_stops_for_human_review(tmp_path):
    (tmp_path / "train").mkdir(parents=True)
    with pytest.raises(M1MemoryError, match="human review"):
        load_stream_cache(tmp_path, "train")


def test_absent_stream_cache_reports_a_missing_manifest(tmp_path):
    with pytest.raises(M1MemoryError, match="No M1 stream cache manifest"):
        load_stream_cache(tmp_path, "validation")


def test_stream_cache_never_accepts_the_test_partition(tmp_path):
    with pytest.raises(Exception):
        load_stream_cache(tmp_path, "test")


# --------------------------------------------------------------------------
# Representation assembly
# --------------------------------------------------------------------------


def test_fuse_refuses_a_non_finite_representation():
    embedding = np.zeros((3, 128), dtype=np.float32)
    physiology = np.zeros((3, 18), dtype=np.float32)
    assert _fuse(embedding, physiology).shape == (3, REPRESENTATION_DIM)
    physiology[1, 4] = np.nan
    with pytest.raises(M1MemoryError, match="refuses"):
        _fuse(embedding, physiology)


def test_fuse_produces_the_retained_146_dimension_width():
    fused = _fuse(
        np.ones((2, 128), dtype=np.float32), np.ones((2, 18), dtype=np.float32)
    )
    assert fused.shape[1] == 146


# --------------------------------------------------------------------------
# One-shot claims
# --------------------------------------------------------------------------


def test_arm_directory_is_a_one_shot_claim(tmp_path):
    run_dir = resolve_m1_run_dir(tmp_path, M1S_EXPERIMENT_ID)
    claim_m1_run_directory(run_dir, M1S_EXPERIMENT_ID)
    with pytest.raises(M1MemoryError, match="already been claimed"):
        claim_m1_run_directory(run_dir, M1S_EXPERIMENT_ID)


def test_claim_refuses_an_unknown_arm(tmp_path):
    with pytest.raises(M1MemoryError):
        resolve_m1_run_dir(tmp_path, "M1X_not_an_arm")


# --------------------------------------------------------------------------
# Locks and the Stage-1 suite
# --------------------------------------------------------------------------


def _lock_for(experiment_id: str, run_dir: Path) -> dict:
    head = build_m1_head(experiment_id)
    lock = build_m1_lock(
        experiment_id,
        head=head,
        result={
            "selected_epoch": 3,
            "selected_validation_auprc": 0.4,
            "completed_epochs": 5,
            "stop_reason": "early_stopping",
        },
        threshold=0.5,
        validation_evidence={"pooled": {"auprc": 0.4}},
        challenge_evidence={"rate_related": {"false_positive_fraction": 0.1}},
        cold_start={"0_5_minutes": {"window_count": 1}},
        descriptives={"window_count": 1},
        subject_false_positives={"pooled_background_negative_fpr": 0.01},
        train_cache={"stream_cache_sha256": "a" * 64},
        validation_cache={"stream_cache_sha256": "b" * 64},
        standardizer={"standardizer_sha256": "c" * 64},
        artifact_hashes={},
        provenance={"git_sha": "f" * 40, "git_dirty": False},
        environment={"device": "cpu"},
        dependency_digest="0" * 64,
        p1_stage1_suite_sha256=FROZEN_P1_STAGE1_SUITE_SHA256,
        p1b_lock_sha256=FROZEN_P1B_LOCK_SHA256,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / EXPERIMENT_LOCK_NAME).write_text(json.dumps(lock))
    return lock


def test_arm_lock_validates_and_detects_tampering(tmp_path):
    run_dir = tmp_path / M1S_EXPERIMENT_ID
    lock = _lock_for(M1S_EXPERIMENT_ID, run_dir)
    assert validate_m1_lock(run_dir)["experiment_id"] == M1S_EXPERIMENT_ID
    assert lock["test_accessed"] is False
    assert lock["test_metrics"] is None
    assert lock["repeat_attempt_permitted"] is False

    lock["threshold"] = 0.9
    (run_dir / EXPERIMENT_LOCK_NAME).write_text(json.dumps(lock))
    with pytest.raises(M1MemoryError, match="digest"):
        validate_m1_lock(run_dir)


def test_arm_lock_binds_the_frozen_global_control(tmp_path):
    lock = _lock_for(M1D_EXPERIMENT_ID, tmp_path / M1D_EXPERIMENT_ID)
    assert lock["global_control_experiment_id"] == "P1B_phys_fusion_v1"
    assert lock["memory_features"] == ["d_short", "d_long"]
    assert lock["head"]["trainable_parameter_count"] == 9601
    assert lock["boundary"]["contamination_safe"] is False


def test_stage1_suite_requires_all_three_arms(tmp_path):
    locks = {arm: _lock_for(arm, tmp_path / arm) for arm in M1_ARM_ORDER[:2]}
    with pytest.raises(M1MemoryError, match="all three arms"):
        build_m1_stage1_result(
            locks,
            control={"experiment_id": "P1B_phys_fusion_v1"},
            stream_caches={},
            standardizer={},
            provenance={"git_sha": "f" * 40, "git_dirty": False},
            environment={},
            dependency_digest="0" * 64,
        )


def test_stage1_suite_round_trips_and_makes_no_selection(tmp_path):
    locks = {arm: _lock_for(arm, tmp_path / arm) for arm in M1_ARM_ORDER}
    rows = np.stack([vector(seed) for seed in range(30)]).astype(np.float64)
    standardizer = fit_distance_standardizer(
        rows, partition="train", input_identities=UPSTREAM
    ).as_dict()
    payload = build_m1_stage1_result(
        locks,
        control={
            "experiment_id": "P1B_phys_fusion_v1",
            "experiment_lock_sha256": FROZEN_P1B_LOCK_SHA256,
            "retrained_by_m1": False,
        },
        stream_caches={
            "train": {"stream_cache_sha256": "a" * 64},
            "validation": {"stream_cache_sha256": "b" * 64},
        },
        standardizer=standardizer,
        provenance={"git_sha": "f" * 40, "git_dirty": False},
        environment={},
        dependency_digest="0" * 64,
    )
    (tmp_path / M1_STAGE1_RESULT_NAME).write_text(json.dumps(payload))

    validated = validate_m1_stage1_results(tmp_path)
    assert validated["m1_stage1_suite_sha256"] == payload["m1_stage1_suite_sha256"]
    # The suite reports evidence; the human makes the Pareto judgement.
    assert validated["memory_selection_performed"] is False
    assert validated["memory_selected"] is None
    assert validated["weighted_score_used"] is False
    assert validated["test_accessed"] is False
    assert validated["arm_order"] == list(M1_ARM_ORDER)
    assert set(validated["arm_results"]) == set(M1_ARM_ORDER)


# --------------------------------------------------------------------------
# Training contract
# --------------------------------------------------------------------------


def test_training_is_deterministic_and_selects_a_checkpoint():
    generator = np.random.default_rng(0)
    train = generator.normal(size=(160, REPRESENTATION_DIM + 1)).astype(np.float32)
    labels = (train[:, 0] > 0).astype(np.int64)
    validation = generator.normal(size=(80, REPRESENTATION_DIM + 1)).astype(np.float32)
    validation_labels = (validation[:, 0] > 0).astype(np.int64)

    first = train_m1_arm(
        M1S_EXPERIMENT_ID, train, labels, validation, validation_labels, max_epochs=2
    )
    second = train_m1_arm(
        M1S_EXPERIMENT_ID, train, labels, validation, validation_labels, max_epochs=2
    )
    assert first["selected_epoch"] == second["selected_epoch"]
    assert first["selected_validation_auprc"] == second["selected_validation_auprc"]
    for left, right in zip(
        first["head"].state_dict().values(), second["head"].state_dict().values()
    ):
        assert np.array_equal(left.numpy(), right.numpy())


def test_training_configuration_matches_the_p1_contract():
    configuration = m1_experiment.m1_training_configuration()
    assert configuration["seed"] == 2026
    assert configuration["batch_size"] == 256
    assert configuration["max_epochs"] == 30
    assert configuration["early_stopping_patience"] == 4
    assert configuration["early_stopping_delta"] == 1e-6
    assert configuration["mixed_precision"] is False
    assert configuration["class_weighting"] is None
    assert configuration["calibration"] is None
    assert configuration["encoder"] == "frozen B4-B; not fine-tuned"


# --------------------------------------------------------------------------
# Route shape
# --------------------------------------------------------------------------


def test_there_is_no_single_arm_public_route():
    from cardiosentinel.cli import build_parser

    parser = build_parser()
    m1 = next(
        action
        for action in parser._subparsers._group_actions[0].choices["m1"]._actions
        if getattr(action, "choices", None)
    )
    assert set(m1.choices) == {"preflight", "run-stage1"}


def test_run_stage1_claims_arms_in_the_frozen_order():
    tree = ast.parse(Path(m1_experiment.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_m1_stage1"
    )
    names = [
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == "M1_ARM_ORDER"
    ]
    assert names, "the suite must iterate the frozen arm order"
    assert M1_ARM_ORDER == (
        "M1S_short_memory_v1",
        "M1L_long_memory_v1",
        "M1D_dual_memory_v1",
    )


def test_preflight_treats_an_absent_cache_as_healthy_not_failed():
    tree = ast.parse(Path(m1_experiment.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "m1_preflight"
    )
    literals = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "stream_cache_materialization_required" in literals
    assert "ready_for_canonical_m1_stage1" in literals
    # A read-only gate must never construct a claim or an artifact.
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "claim_m1_run_directory" not in calls
    assert "materialize_stream_cache" not in calls
    assert "generate_stream_memory" not in calls


# --------------------------------------------------------------------------
# Frozen P1-B control artifacts
# --------------------------------------------------------------------------


def _p1b_run(tmp_path: Path) -> Path:
    """A synthetic P1-B run laid out exactly like the canonical one.

    The lock stores only digests; the transform and evidence live in their own
    artifacts. Reading them from the wrong place is precisely the defect this
    guards, so the layout is reproduced rather than stubbed.
    """
    from cardiosentinel.neural.physiology_fusion import (
        PHYSIOLOGY_FEATURE_NAMES,
        fit_physiology_transform,
    )

    run_dir = tmp_path / "P1B_phys_fusion_v1"
    run_dir.mkdir(parents=True)
    rows = np.random.default_rng(3).normal(size=(40, len(PHYSIOLOGY_FEATURE_NAMES)))
    rows[:, PHYSIOLOGY_FEATURE_NAMES.index("morphology_valid")] = 1.0
    transform = fit_physiology_transform(rows, partition="train").as_dict()
    (run_dir / "PHYSIOLOGY_TRANSFORM.json").write_text(json.dumps(transform))
    (run_dir / "VALIDATION_METRICS.json").write_text(json.dumps({"pooled": {}}))
    (run_dir / "CHALLENGE_METRICS.json").write_text(json.dumps({"rate_related": {}}))

    from cardiosentinel.data.provenance import sha256_file

    (run_dir / EXPERIMENT_LOCK_NAME).write_text(
        json.dumps(
            {
                "experiment_id": "P1B_phys_fusion_v1",
                "physiology_transform_sha256": transform["transform_sha256"],
                "artifact_sha256": {
                    name: sha256_file(run_dir / name)
                    for name in (
                        "PHYSIOLOGY_TRANSFORM.json",
                        "VALIDATION_METRICS.json",
                        "CHALLENGE_METRICS.json",
                    )
                },
            }
        )
    )
    return run_dir


def test_frozen_transform_is_read_from_its_own_verified_artifact(tmp_path):
    run_dir = _p1b_run(tmp_path)
    transform = m1_experiment.load_frozen_physiology_transform(run_dir)
    lock = json.loads((run_dir / EXPERIMENT_LOCK_NAME).read_text())
    assert (
        transform.as_dict()["transform_sha256"]
        == lock["physiology_transform_sha256"]
    )


def test_tampered_transform_artifact_is_refused(tmp_path):
    run_dir = _p1b_run(tmp_path)
    payload = json.loads((run_dir / "PHYSIOLOGY_TRANSFORM.json").read_text())
    payload["means"][0] = 99.0
    (run_dir / "PHYSIOLOGY_TRANSFORM.json").write_text(json.dumps(payload))
    with pytest.raises(M1MemoryError, match="lock digest"):
        m1_experiment.load_frozen_physiology_transform(run_dir)


def test_control_evidence_is_read_from_verified_artifacts(tmp_path):
    evidence = m1_experiment.load_frozen_control_evidence(_p1b_run(tmp_path))
    assert set(evidence) == {"validation_evidence", "challenge_evidence"}


def test_missing_control_artifact_is_refused(tmp_path):
    run_dir = _p1b_run(tmp_path)
    (run_dir / "CHALLENGE_METRICS.json").unlink()
    with pytest.raises(M1MemoryError):
        m1_experiment.load_frozen_control_evidence(run_dir)


# --------------------------------------------------------------------------
# Preflight firewall and partial-state semantics
# --------------------------------------------------------------------------


def test_test_artifact_scan_actually_walks_the_supplied_roots(tmp_path):
    assert scan_test_artifacts(tmp_path) == []
    claimed = tmp_path / "M1S_short_memory_v1"
    claimed.mkdir()
    (claimed / "TEST_ATTEMPT.json").write_text("{}")
    found = scan_test_artifacts(tmp_path)
    assert any(name.endswith("TEST_ATTEMPT.json") for name in found)


def test_preflight_refuses_when_a_test_artifact_exists(tmp_path, monkeypatch):
    _stub_preflight_runtime(monkeypatch)
    run_root = tmp_path / "runs"
    (run_root / "M1S_short_memory_v1").mkdir(parents=True)
    (run_root / "M1S_short_memory_v1" / "TEST_ATTEMPT.json").write_text("{}")
    report = m1_experiment.m1_preflight(run_root, tmp_path / "caches")
    assert report["status"] == "test_artifact_present_human_review_required"
    assert report["test_artifacts_present"] is True
    assert report["ready_for_canonical_m1_stage1"] is False
    assert report["human_review_required"] is True


def test_preflight_reports_partial_cache_for_human_review(tmp_path, monkeypatch):
    _stub_preflight_runtime(monkeypatch)
    caches = tmp_path / "caches"
    (caches / "train").mkdir(parents=True)  # directory, no manifest
    report = m1_experiment.m1_preflight(tmp_path / "runs", caches)
    assert report["status"] == "partial_stream_cache_human_review_required"
    assert report["stream_cache_state"]["partial_partitions"] == ["train"]
    assert report["ready_for_canonical_m1_stage1"] is False


def test_preflight_refuses_an_orphan_standardizer(tmp_path, monkeypatch):
    _stub_preflight_runtime(monkeypatch)
    caches = tmp_path / "caches"
    caches.mkdir(parents=True)
    (caches / "M1_DISTANCE_STANDARDIZER.json").write_text("{}")
    report = m1_experiment.m1_preflight(tmp_path / "runs", caches)
    assert report["status"] == "partial_stream_cache_human_review_required"
    assert report["stream_cache_state"]["orphan_standardizer"] is True


def test_preflight_absent_cache_is_healthy_initial_state(tmp_path, monkeypatch):
    _stub_preflight_runtime(monkeypatch)
    report = m1_experiment.m1_preflight(tmp_path / "runs", tmp_path / "caches")
    # With no upstream roots supplied the gates are legitimately unproven, but
    # an absent cache must never be mistaken for a partial one.
    assert report["status"] != "partial_stream_cache_human_review_required"
    assert report["stream_cache_state"]["partial_partitions"] == []
    assert report["stream_cache_state"]["one_partition_only"] is False
    assert report["human_review_required"] is False
    assert report["models_created"] == 0
    assert report["artifacts_created"] == 0
    assert not (tmp_path / "caches").exists()


def test_preflight_sha256_is_computed_over_the_complete_report(
    tmp_path, monkeypatch
):
    from cardiosentinel.neural.integrity import canonical_sha256

    _stub_preflight_runtime(monkeypatch)
    report = m1_experiment.m1_preflight(tmp_path / "runs", tmp_path / "caches")
    body = {k: v for k, v in report.items() if k != "preflight_sha256"}
    assert report["preflight_sha256"] == canonical_sha256(body)
    assert "status" in body and "stream_caches" in body


def _stub_preflight_runtime(monkeypatch):
    """Preflight is read-only, but it still gates on the frozen runtime.

    CI runs a different Python than the frozen scientific environment, so the
    gate is stubbed here rather than weakened in production. The real preflight
    still evaluates it.
    """
    monkeypatch.setattr(
        m1_experiment, "require_p1_runtime", lambda: ({"device": "cpu"}, "0" * 64)
    )
    monkeypatch.setattr(
        m1_experiment,
        "require_clean_checkout",
        lambda: {"git_sha": "f" * 40, "git_dirty": False},
    )


# --------------------------------------------------------------------------
# Exact frozen upstream identity enforcement
# --------------------------------------------------------------------------


def _upstream_kwargs(**overrides):
    payload = {
        "p1_suite": {
            "p1_stage1_suite_sha256": FROZEN_P1_STAGE1_SUITE_SHA256,
            "test_accessed": False,
        },
        "p1b_lock": {"experiment_lock_sha256": FROZEN_P1B_LOCK_SHA256},
        "physiology_transform_sha256": FROZEN_PHYSIOLOGY_TRANSFORM_SHA256,
        "embedding_caches": {
            partition: {"cache_sha256": digest}
            for partition, digest in FROZEN_P1_EMBEDDING_CACHE_SHA256.items()
        },
        "encoder_lock": {
            "checkpoint_sha256": (
                "b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9"
            ),
            "experiment_lock_sha256": (
                "58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b"
            ),
            "test": None,
        },
        "feature_receipt": {
            "development_feature_integrity_sha256": (
                FROZEN_DEVELOPMENT_FEATURE_INTEGRITY_SHA256
            )
        },
        "source_receipt": {
            "development_source_integrity_sha256": (
                FROZEN_DEVELOPMENT_SOURCE_INTEGRITY_SHA256
            )
        },
        "challenge_selection_sha256": (
            "49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a"
        ),
    }
    payload.update(overrides)
    return payload


def test_frozen_upstream_identities_pass_when_exact():
    receipt = require_frozen_upstream_identities(**_upstream_kwargs())
    assert receipt["all_frozen_identities_enforced"] is True
    assert receipt["p1_stage1_suite_sha256"] == FROZEN_P1_STAGE1_SUITE_SHA256


@pytest.mark.parametrize(
    "override",
    [
        {"p1_suite": {"p1_stage1_suite_sha256": "0" * 64, "test_accessed": False}},
        {"p1b_lock": {"experiment_lock_sha256": "0" * 64}},
        {"physiology_transform_sha256": "0" * 64},
        {
            "embedding_caches": {
                "train": {"cache_sha256": "0" * 64},
                "validation": {
                    "cache_sha256": FROZEN_P1_EMBEDDING_CACHE_SHA256["validation"]
                },
            }
        },
        {
            "feature_receipt": {
                "development_feature_integrity_sha256": "0" * 64
            }
        },
        {"source_receipt": {"development_source_integrity_sha256": "0" * 64}},
        {"challenge_selection_sha256": "0" * 64},
    ],
)
def test_a_different_valid_artifact_is_not_a_substitute(override):
    with pytest.raises(M1MemoryError, match="frozen M1 protocol binds"):
        require_frozen_upstream_identities(**_upstream_kwargs(**override))


def test_a_suite_recording_test_access_is_refused():
    with pytest.raises(M1MemoryError, match="test access"):
        require_frozen_upstream_identities(
            **_upstream_kwargs(
                p1_suite={
                    "p1_stage1_suite_sha256": FROZEN_P1_STAGE1_SUITE_SHA256,
                    "test_accessed": True,
                }
            )
        )


def test_standardizer_refuses_null_upstream_identities(synthetic):
    _, representation = synthetic
    with pytest.raises(M1MemoryError, match="absent or null"):
        build_distance_standardizer(
            representation,
            primary_train_stable_ids=representation.stable_ids,
            upstream_identities={**UPSTREAM, "p1b_experiment_lock_sha256": None},
        )


def test_superseded_protocol_digest_is_named_as_such():
    from cardiosentinel.neural.patient_memory import (
        M1_PROTOCOL_SHA256,
        SUPERSEDED_M1_PROTOCOL_SHA256,
    )

    # Both drafts were superseded BEFORE any M1 evidence existed. Historical
    # entries are never erased, so a stale digest is recognised as superseded
    # rather than merely rejected as unknown.
    for digest in (
        "52eedc628d906ac02619264fc26cd4629e56f05d6c1916448d62a2844c9815f4",
        "cc2e78e720bbb55d3dd51e61a5ea6cd04c77cb77eef41508def3951361ccda61",
    ):
        assert digest in SUPERSEDED_M1_PROTOCOL_SHA256
    assert M1_PROTOCOL_SHA256 not in SUPERSEDED_M1_PROTOCOL_SHA256

    document = Path("docs/M1_DUAL_MEMORY_PROTOCOL_V1.md").read_text()
    for digest in SUPERSEDED_M1_PROTOCOL_SHA256:
        assert digest in document, "the revision record must retain every entry"
    assert document.count("SUPERSEDED BEFORE USE") >= len(
        SUPERSEDED_M1_PROTOCOL_SHA256
    )


# --------------------------------------------------------------------------
# Subject-wise false-positive evidence
# --------------------------------------------------------------------------


def test_subject_false_positive_evidence_is_deterministic():
    labels = np.array([0, 0, 0, 0, 1, 0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.1, 0.2, 0.8, 0.95, 0.7, 0.05, 0.99])
    subjects = ["a", "a", "b", "b", "b", "c", "c", "c"]
    first = subject_false_positive_evidence(labels, scores, subjects, 0.5)
    second = subject_false_positive_evidence(labels, scores, subjects, 0.5)
    assert first == second

    # a: 1/2, b: 1/2, c: 1/2 -> pooled 3/6
    assert first["pooled_background_negative_fpr"] == pytest.approx(0.5)
    assert first["background_negative_count"] == 6
    assert first["contributing_subject_count"] == 3
    assert first["subject_false_positive_rates"] == {"a": 0.5, "b": 0.5, "c": 0.5}
    assert first["quantile_interpolation"] == "linear"
    assert first["evidence_status"] == "supporting"
    assert first["threshold_optimized_from_this_evidence"] is False


def test_subject_without_negative_support_is_excluded():
    labels = np.array([0, 0, 1], dtype=np.int64)
    scores = np.array([0.9, 0.1, 0.99])
    evidence = subject_false_positive_evidence(labels, scores, ["a", "a", "b"], 0.5)
    assert evidence["contributing_subject_count"] == 1
    assert set(evidence["subject_false_positive_rates"]) == {"a"}


def test_subject_false_positive_summary_reports_the_frozen_fields():
    labels = np.zeros(20, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, 20)
    subjects = [f"s{i // 4}" for i in range(20)]
    evidence = subject_false_positive_evidence(labels, scores, subjects, 0.5)
    for field in (
        "pooled_background_negative_fpr",
        "background_negative_count",
        "contributing_subject_count",
        "subject_fpr_median",
        "subject_fpr_q25",
        "subject_fpr_q75",
        "subject_fpr_iqr",
        "subject_fpr_p90",
        "subject_fpr_max",
        "subject_false_positive_rates",
    ):
        assert field in evidence, field
    assert evidence["subject_fpr_iqr"] == pytest.approx(
        evidence["subject_fpr_q75"] - evidence["subject_fpr_q25"]
    )


def test_subject_false_positive_evidence_refuses_an_empty_negative_population():
    with pytest.raises(M1MemoryError, match="background-negative"):
        subject_false_positive_evidence(
            np.ones(3, dtype=np.int64), np.ones(3), ["a", "a", "a"], 0.5
        )


# --------------------------------------------------------------------------
# Corrected cold-start origin
# --------------------------------------------------------------------------


def test_recording_age_is_stream_relative_not_absolute(standardizer):
    from cardiosentinel.neural.patient_memory import (
        build_causal_streams as _streams,
    )

    # A stream that starts deep into the record: absolute start samples would
    # place its first window in the ">60 minutes" bin, which is exactly the
    # error the superseded protocol text described.
    offset = 4_000_000  # 16000 s at 250 Hz
    rows = [
        reference("rA", 0, index + offset // 1250, partition="train")
        for index in range(4)
    ]
    values = {row.stable_id: vector(index) for index, row in enumerate(rows)}
    memory = generate_stream_memory(
        _streams(rows),
        partition="train",
        representations=values,
        standardizer=standardizer,
    )
    assert memory.recording_age_seconds[0] == 0.0
    assert memory.cold_start_bins[0] == "0_5_minutes"
    assert memory.recording_age_seconds[1] == pytest.approx(5.0)
