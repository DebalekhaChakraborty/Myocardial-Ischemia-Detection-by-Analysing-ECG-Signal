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


MANIFEST_FIELDS = {
    "standardizer_sha256": "a" * 64,
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
        )


def test_standardizer_requires_the_frozen_primary_train_population(synthetic):
    _, representation = synthetic
    # The synthetic stream is far smaller than the frozen 374,452 train rows,
    # so the guard must refuse rather than quietly fit on whatever it is given.
    with pytest.raises(M1MemoryError, match="374452"):
        build_distance_standardizer(
            representation, primary_train_stable_ids=representation.stable_ids
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
        train_cache={"stream_cache_sha256": "a" * 64},
        validation_cache={"stream_cache_sha256": "b" * 64},
        standardizer={"standardizer_sha256": "c" * 64},
        artifact_hashes={},
        provenance={"git_sha": "f" * 40, "git_dirty": False},
        environment={"device": "cpu"},
        dependency_digest="0" * 64,
        p1_stage1_suite_sha256="d" * 64,
        p1b_lock_sha256="e" * 64,
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
    payload = build_m1_stage1_result(
        locks,
        control={"experiment_id": "P1B_phys_fusion_v1"},
        stream_caches={"train": {"stream_cache_sha256": "a" * 64}},
        standardizer={"standardizer_sha256": "c" * 64},
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
