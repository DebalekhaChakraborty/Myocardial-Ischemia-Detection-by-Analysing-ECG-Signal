"""Synthetic tests for the P1 physiology-fusion development machinery.

Every fixture is synthetic. No test reads a real B4 run, a real feature corpus,
a real waveform source, or the sealed test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import physiology_fusion as p1
from cardiosentinel.neural.candidates import B4BTransformerCNN
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.protocol import WINDOW_SAMPLES

NAMES = p1.PHYSIOLOGY_FEATURE_NAMES
VALID = NAMES.index("morphology_valid")
TEMPLATE = NAMES.index("beat_template_correlation_median")


def _train_matrix(rows: int = 64, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, p1.PHYSIOLOGY_DIM)) * 3.0 + 10.0
    matrix[:, VALID] = 1.0
    return matrix


# --------------------------------------------------------------------------
# Protocol and schema identity
# --------------------------------------------------------------------------


def test_protocol_sha_is_enforced() -> None:
    assert p1.validate_p1_protocol() == p1.P1_PROTOCOL_SHA256


def test_tampered_protocol_is_refused(tmp_path) -> None:
    forged = tmp_path / "p.md"
    forged.write_text("not the frozen protocol\n", encoding="utf-8")
    with pytest.raises(p1.PhysiologyFusionError, match="frozen SHA-256"):
        p1.validate_p1_protocol(forged)


def test_frozen_schema_identity_and_order() -> None:
    assert p1.MORPHOLOGY_SCHEMA_SHA256 == (
        "13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434"
    )
    assert p1.PHYSIOLOGY_DIM == 18
    assert NAMES[0] == "detected_r_peak_count"
    assert NAMES[-1] == "beat_template_variability"
    p1.validate_physiology_schema(NAMES)


def test_wrong_feature_order_is_refused() -> None:
    with pytest.raises(p1.PhysiologyFusionError, match="names/order"):
        p1.validate_physiology_schema(tuple(reversed(NAMES)))
    with pytest.raises(p1.PhysiologyFusionError, match="names/order"):
        p1.validate_physiology_schema(NAMES[:-1])


def test_feature_groups_tile_the_schema_exactly() -> None:
    groups = p1.physiology_feature_groups()
    assigned = [n for g in groups.values() for n in g]
    assert sorted(assigned) == sorted(NAMES)
    assert len(assigned) == len(set(assigned)) == 18


def test_morphology_columns_are_contiguous_and_correct() -> None:
    columns = p1.morphology_columns()
    assert len(columns) == 18
    assert list(columns) == list(range(columns[0], columns[0] + 18))


# --------------------------------------------------------------------------
# Missingness / imputation (the corrected policy)
# --------------------------------------------------------------------------


def test_invalid_row_with_all_nan_is_imputed_not_dropped() -> None:
    matrix = _train_matrix()
    matrix[0, :] = np.nan
    matrix[0, VALID] = 0.0
    fitted = p1.fit_physiology_transform(matrix)
    out = fitted.transform(matrix)
    assert out.shape == matrix.shape  # no row dropping
    assert np.all(np.isfinite(out))
    assert fitted.imputed_counts["rr_mean_ms"] == 1


def test_isolated_nan_in_a_valid_row_is_imputed() -> None:
    """The real train audit found exactly this case; it must be handled."""
    matrix = _train_matrix()
    matrix[3, TEMPLATE] = np.nan
    assert matrix[3, VALID] == 1.0
    fitted = p1.fit_physiology_transform(matrix)
    out = fitted.transform(matrix)
    assert fitted.imputed_counts["beat_template_correlation_median"] == 1
    assert np.isfinite(out[3, TEMPLATE])


def test_validity_feature_is_retained_and_never_imputed() -> None:
    matrix = _train_matrix()
    matrix[0, :] = np.nan
    matrix[0, VALID] = 0.0
    fitted = p1.fit_physiology_transform(matrix)
    out = fitted.transform(matrix)
    # The zero-validity row must remain distinguishable from valid rows.
    assert out[0, VALID] != out[1, VALID]
    assert fitted.imputed_counts["morphology_valid"] == 0


def test_non_finite_validity_is_refused() -> None:
    matrix = _train_matrix()
    fitted = p1.fit_physiology_transform(matrix)
    broken = matrix.copy()
    broken[0, VALID] = np.nan
    with pytest.raises(p1.PhysiologyFusionError, match="must be finite"):
        fitted.transform(broken)


def test_zero_finite_support_is_refused_not_filled() -> None:
    matrix = _train_matrix()
    matrix[:, TEMPLATE] = np.nan
    with pytest.raises(p1.PhysiologyFusionError, match="zero finite train support"):
        p1.fit_physiology_transform(matrix)


def test_zero_variance_feature_becomes_a_constant_zero_column() -> None:
    matrix = _train_matrix()
    matrix[:, TEMPLATE] = 5.0
    fitted = p1.fit_physiology_transform(matrix)
    assert "beat_template_correlation_median" in fitted.zero_variance_features
    out = fitted.transform(matrix)
    assert np.allclose(out[:, TEMPLATE], 0.0)


def test_transform_output_is_finite_and_float32() -> None:
    matrix = _train_matrix()
    fitted = p1.fit_physiology_transform(matrix)
    out = fitted.transform(matrix)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# Train-only fitting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_transform_cannot_be_fitted_outside_train(partition) -> None:
    with pytest.raises(p1.PhysiologyFusionError):
        p1.fit_physiology_transform(_train_matrix(), partition=partition)


def test_validation_values_cannot_change_fitted_statistics() -> None:
    train = _train_matrix()
    fitted = p1.fit_physiology_transform(train)
    before = fitted.as_dict()
    wild = _train_matrix(seed=99) * 1000.0
    wild[:, VALID] = 1.0
    fitted.transform(wild)
    assert fitted.as_dict() == before


def test_transform_digest_and_provenance_are_bound() -> None:
    fitted = p1.fit_physiology_transform(_train_matrix())
    payload = fitted.as_dict()
    body = {k: v for k, v in payload.items() if k != "transform_sha256"}
    assert canonical_sha256(body) == payload["transform_sha256"]
    assert payload["fitted_on_partition"] == "train"
    assert payload["schema_sha256"] == p1.MORPHOLOGY_SCHEMA_SHA256
    assert payload["validity_feature"] == "morphology_valid"


def test_wrong_width_matrix_is_refused() -> None:
    fitted = p1.fit_physiology_transform(_train_matrix())
    with pytest.raises(p1.PhysiologyFusionError, match=r"\[N, 18\]"):
        fitted.transform(np.zeros((4, 17)))


# --------------------------------------------------------------------------
# P1-A / P1-B heads
# --------------------------------------------------------------------------


def test_head_dimensions_and_parameter_counts() -> None:
    a = p1.build_p1_head(p1.P1A_EXPERIMENT_ID)
    b = p1.build_p1_head(p1.P1B_EXPERIMENT_ID)
    assert p1.p1_head_identity(a)["input_dim"] == 128
    assert p1.p1_head_identity(b)["input_dim"] == 146
    assert p1.p1_head_identity(a)["trainable_parameter_count"] == 8321
    assert p1.p1_head_identity(b)["trainable_parameter_count"] == 9473


def test_both_arms_share_the_same_head_family() -> None:
    a = p1.build_p1_head(p1.P1A_EXPERIMENT_ID)
    b = p1.build_p1_head(p1.P1B_EXPERIMENT_ID)
    assert [type(m).__name__ for m in a.head] == [type(m).__name__ for m in b.head]
    ia, ib = p1.p1_head_identity(a), p1.p1_head_identity(b)
    for key in ("hidden_dim", "activation", "dropout", "output"):
        assert ia[key] == ib[key]


def test_head_rejects_the_wrong_input_width() -> None:
    a = p1.build_p1_head(p1.P1A_EXPERIMENT_ID)
    with pytest.raises(p1.PhysiologyFusionError, match=r"\[B, 128\]"):
        a(torch.zeros(2, 146))


def test_unknown_arm_is_refused() -> None:
    with pytest.raises(p1.PhysiologyFusionError, match="Unknown P1 experiment"):
        p1.build_p1_head("P1Z_bogus_v1")


def test_p1a_manifest_refuses_physiology_leakage() -> None:
    head = p1.build_p1_head(p1.P1A_EXPERIMENT_ID)
    fitted = p1.fit_physiology_transform(_train_matrix())
    with pytest.raises(p1.PhysiologyFusionError, match="neural-only control"):
        p1.p1_run_manifest(
            p1.P1A_EXPERIMENT_ID,
            head=head,
            transform=fitted,
            git_sha="a" * 40,
            environment={},
        )


def test_p1b_manifest_requires_a_fitted_transform() -> None:
    head = p1.build_p1_head(p1.P1B_EXPERIMENT_ID)
    with pytest.raises(p1.PhysiologyFusionError, match="requires a fitted"):
        p1.p1_run_manifest(
            p1.P1B_EXPERIMENT_ID,
            head=head,
            transform=None,
            git_sha="a" * 40,
            environment={},
        )


def test_manifests_bind_provenance_and_digest() -> None:
    for arm, transform in (
        (p1.P1A_EXPERIMENT_ID, None),
        (p1.P1B_EXPERIMENT_ID, p1.fit_physiology_transform(_train_matrix())),
    ):
        manifest = p1.p1_run_manifest(
            arm,
            head=p1.build_p1_head(arm),
            transform=transform,
            git_sha="b" * 40,
            environment={"python_version": "3.12.6"},
        )
        body = {k: v for k, v in manifest.items() if k != "p1_manifest_sha256"}
        assert canonical_sha256(body) == manifest["p1_manifest_sha256"]
        assert manifest["encoder_fine_tuned"] is False
        assert manifest["test_accessed"] is False
        assert manifest["sealed_test_state"] == "unopened"
        assert manifest["encoder_checkpoint_sha256"] == p1.B4B_CHECKPOINT_SHA256


def test_training_contract_is_common_and_frozen() -> None:
    configuration = p1.p1_training_configuration()
    assert configuration["seed"] == 2026
    assert configuration["optimizer"] == "AdamW"
    assert configuration["scheduler"] is None
    assert configuration["early_stopping_patience"] == 4
    assert configuration["encoder"] == "frozen B4-B; not fine-tuned"


# --------------------------------------------------------------------------
# Frozen encoder + embedding cache
# --------------------------------------------------------------------------


def test_embedding_extraction_leaves_the_encoder_unchanged() -> None:
    encoder = B4BTransformerCNN()
    waveforms = torch.zeros(3, 1, WINDOW_SAMPLES)
    embeddings, receipt = p1.extract_frozen_embeddings(encoder, waveforms)
    assert embeddings.shape == (3, 128)
    assert receipt["encoder_state_unchanged"] is True
    assert (
        receipt["encoder_state_sha256_before"]
        == receipt["encoder_state_sha256_after"]
    )
    assert receipt["gradients_enabled"] is False
    assert receipt["encoder_fine_tuned"] is False
    assert all(not p.requires_grad for p in encoder.parameters())


def _cache(**overrides):
    payload = {
        "partition": "validation",
        "stable_ids": ["w0", "w1", "w2"],
        "embeddings": np.zeros((3, 128), dtype=np.float32),
        "split_sha256": "s" * 64,
        "feature_corpus_sha256": "c" * 64,
        "git_sha": "d" * 40,
        "environment": {"python_version": "3.12.6"},
    }
    payload.update(overrides)
    return p1.embedding_cache_contract(**payload)


def test_embedding_cache_binds_the_frozen_encoder_identity() -> None:
    contract = _cache()
    assert contract["embedding_dim"] == 128
    assert contract["encoder_checkpoint_sha256"] == p1.B4B_CHECKPOINT_SHA256
    assert contract["encoder_experiment_lock_sha256"] == p1.B4B_EXPERIMENT_LOCK_SHA256
    assert contract["embedding_tap"] == p1.EMBEDDING_TAP
    assert contract["test_accessed"] is False
    body = {k: v for k, v in contract.items() if k != "cache_sha256"}
    assert canonical_sha256(body) == contract["cache_sha256"]


def test_embedding_cache_refuses_the_test_partition() -> None:
    with pytest.raises(p1.PhysiologyFusionError, match="never access"):
        _cache(partition="test")


def test_embedding_cache_refuses_misaligned_or_duplicate_ids() -> None:
    with pytest.raises(p1.PhysiologyFusionError, match="misaligned"):
        _cache(stable_ids=["w0", "w1"])
    with pytest.raises(p1.PhysiologyFusionError, match="duplicate"):
        _cache(stable_ids=["w0", "w0", "w1"])


def test_embedding_cache_refuses_bad_shape_or_non_finite() -> None:
    with pytest.raises(p1.PhysiologyFusionError, match=r"\[N, 128\]"):
        _cache(embeddings=np.zeros((3, 64), dtype=np.float32))
    with pytest.raises(p1.PhysiologyFusionError, match="non-finite"):
        _cache(embeddings=np.full((3, 128), np.nan, dtype=np.float32))


# --------------------------------------------------------------------------
# One-shot semantics and status identity
# --------------------------------------------------------------------------


def test_canonical_attempt_is_claimed_once(tmp_path) -> None:
    run_dir = tmp_path / p1.P1A_EXPERIMENT_ID
    p1.claim_p1_run_directory(run_dir, p1.P1A_EXPERIMENT_ID)
    assert run_dir.is_dir()
    with pytest.raises(p1.PhysiologyFusionError, match="already been claimed"):
        p1.claim_p1_run_directory(run_dir, p1.P1A_EXPERIMENT_ID)


def test_claim_has_no_force_or_delete_path() -> None:
    source = inspect.getsource(p1.claim_p1_run_directory)
    assert "run_dir.mkdir(exist_ok=False)" in source
    assert "os.O_DIRECTORY" in source
    for forbidden in ("--force", "rmtree", "unlink(", "rename("):
        assert forbidden not in source


def test_failure_receipt_requires_human_review(tmp_path) -> None:
    run_dir = tmp_path / p1.P1B_EXPERIMENT_ID
    p1.claim_p1_run_directory(run_dir, p1.P1B_EXPERIMENT_ID)
    try:
        raise ValueError("synthetic failure")
    except ValueError as error:
        payload = p1.record_p1_failure(run_dir, p1.P1B_EXPERIMENT_ID, error)
    assert payload["status"] == "FAILED_OR_INTERRUPTED"
    assert payload["human_review_required"] is True
    assert payload["repeat_attempt_permitted"] is False
    assert payload["automatic_retry_performed"] is False
    assert run_dir.is_dir()  # the claim is never released


def test_status_identity_is_required_and_per_arm(tmp_path) -> None:
    """Same defect class as the B4 RUN_STATUS bug: no defaulted identity."""
    parameters = inspect.signature(p1.write_p1_status).parameters
    assert parameters["experiment_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["experiment_id"].default is inspect.Parameter.empty
    for arm in (p1.P1A_EXPERIMENT_ID, p1.P1B_EXPERIMENT_ID):
        directory = tmp_path / arm
        directory.mkdir()
        payload = p1.write_p1_status(directory, "STARTED", experiment_id=arm)
        assert payload["experiment_id"] == arm


def test_run_root_must_be_non_versioned() -> None:
    with pytest.raises(ValueError):
        p1.resolve_p1_run_dir(Path(p1.REPOSITORY_ROOT) / "docs", p1.P1A_EXPERIMENT_ID)


# --------------------------------------------------------------------------
# Structural test firewall
# --------------------------------------------------------------------------


@pytest.mark.parametrize("partition", ["test", "sealed_test", "TEST"])
def test_no_p1_path_accepts_the_test_partition(partition) -> None:
    with pytest.raises(p1.PhysiologyFusionError):
        p1.require_p1_partition(partition)


def test_permitted_partitions_are_train_and_validation_only() -> None:
    assert p1.PERMITTED_PARTITIONS == ("train", "validation")
    assert p1.require_p1_partition("train") == "train"
    assert p1.require_p1_partition("validation") == "validation"


def test_module_never_imports_or_names_sealed_test_machinery() -> None:
    source = Path(p1.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("sealed_test" in name for name in imported)
    for forbidden in (
        "evaluate_locked_test",
        "SealedTestAccess",
        "TEST_ATTEMPT",
        "TEST_METRICS",
    ):
        assert forbidden not in source


def test_module_declares_no_selection_or_scoring_helper() -> None:
    source = inspect.getsource(p1).lower()
    for forbidden in ("def select_", "def rank_", "winner", "weighted_score"):
        assert forbidden not in source
