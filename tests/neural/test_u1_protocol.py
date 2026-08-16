"""The U1 protocol binding is structural: synthetic data only, no execution.

Nothing here fits a calibrator, scores a window, or opens a run artifact. The
tests prove the frozen design refuses what it must refuse.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import u1_protocol as U
from cardiosentinel.neural.u1_protocol import (
    U1ProtocolError,
    assign_calibration_folds,
    equal_mass_group_boundaries,
    equal_mass_group_sizes,
    equal_mass_groups,
    equal_mass_sort_order,
    equal_width_bin_edges,
    equal_width_bin_index,
    fold_assignment_digest,
    require_calibration_subjects,
    routing_threshold_rank,
    select_routing_threshold,
    u1_protocol_identity,
    validate_against_frozen_split,
    validate_fold_assignment,
    validate_u1_protocol_document,
)

FROZEN_SPLIT = Path("protocols/splits/ltstdb_v1.json")


def _frozen_test_subjects():
    if not FROZEN_SPLIT.is_file():
        pytest.skip("frozen split manifest is not on this filesystem")
    manifest = json.loads(FROZEN_SPLIT.read_text())
    return sorted(manifest["partitions"]["test"]["subjects"])


# ---------------------------------------------------------------------------
# Protocol document
# ---------------------------------------------------------------------------


def test_protocol_document_is_frozen():
    assert validate_u1_protocol_document() == U.U1_PROTOCOL_SHA256


def test_protocol_document_states_its_hard_commitments():
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "FROZEN PROSPECTIVE SCIENTIFIC PROTOCOL" in text
    # the G4 admission score must never become "confidence"
    assert "must not calibrate, reinterpret, rescale or route on the G4" in text
    # true logits are genuinely unavailable; temperature scaling is not proper
    assert "True logits are not persisted" in text
    # routing threshold is not chosen in this task
    assert "No routing threshold is chosen in this task" in text
    # no automatic downstream transition
    assert "does **not** begin U2 conformal prediction" in text
    assert "Completion of U1 does not authorise TEST." in text


def test_protocol_refuses_a_mutated_document(tmp_path):
    forged = tmp_path / "forged.md"
    forged.write_text("not the protocol")
    with pytest.raises(U1ProtocolError, match="differs from the frozen"):
        validate_u1_protocol_document(forged)


def test_protocol_refuses_a_missing_document(tmp_path):
    with pytest.raises(U1ProtocolError, match="missing"):
        validate_u1_protocol_document(tmp_path / "absent.md")


# ---------------------------------------------------------------------------
# TEST firewall
# ---------------------------------------------------------------------------


def test_real_test_subjects_can_never_enter_calibration():
    """The decisive firewall test, using the real frozen TEST identities."""
    for subject in _frozen_test_subjects():
        with pytest.raises(U1ProtocolError, match="not in the frozen U1"):
            require_calibration_subjects([subject])


def test_a_single_test_subject_poisons_an_otherwise_valid_set():
    test_subject = _frozen_test_subjects()[0]
    contaminated = list(U.U1_CALIBRATION_SUBJECTS) + [test_subject]
    with pytest.raises(U1ProtocolError, match="not in the frozen U1"):
        assign_calibration_folds(contaminated)


def test_test_subjects_are_refused_not_silently_dropped():
    """A filter would return 12 folds; the protocol must raise instead."""
    test_subject = _frozen_test_subjects()[0]
    with pytest.raises(U1ProtocolError):
        assign_calibration_folds(list(U.U1_CALIBRATION_SUBJECTS) + [test_subject])


def test_frozen_split_proves_validation_and_test_are_disjoint():
    proof = validate_against_frozen_split(FROZEN_SPLIT)
    assert proof["calibration_test_intersection"] == []
    assert proof["sealed_test_partition"] is True
    assert proof["calibration_subject_count"] == 12
    assert proof["test_accessed"] is False
    assert proof["sealed_test_state"] == "unopened"


def test_split_validator_refuses_a_wrong_split(tmp_path):
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": "0" * 64,
                "sealed_test_partition": True,
                "partitions": {"validation": {"subjects": []}, "test": {}},
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="not the frozen benchmark split"):
        validate_against_frozen_split(forged)


def test_split_validator_refuses_an_unsealed_test_partition(tmp_path):
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": U.U1_SPLIT_SHA256,
                "sealed_test_partition": False,
                "partitions": {
                    "validation": {"subjects": list(U.U1_CALIBRATION_SUBJECTS)},
                    "test": {"subjects": []},
                },
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="does not seal the TEST"):
        validate_against_frozen_split(forged)


def test_split_validator_catches_a_leaked_test_subject(tmp_path):
    """If VALIDATION and TEST ever overlapped, the validator must refuse."""
    forged = tmp_path / "split.json"
    forged.write_text(
        json.dumps(
            {
                "split_sha256": U.U1_SPLIT_SHA256,
                "sealed_test_partition": True,
                "partitions": {
                    "validation": {"subjects": list(U.U1_CALIBRATION_SUBJECTS)},
                    "test": {"subjects": [U.U1_CALIBRATION_SUBJECTS[0]]},
                },
            }
        )
    )
    with pytest.raises(U1ProtocolError, match="appear in calibration"):
        validate_against_frozen_split(forged)


# ---------------------------------------------------------------------------
# Fold design
# ---------------------------------------------------------------------------


def test_folds_are_subject_disjoint_and_complete():
    folds = assign_calibration_folds()
    assert len(folds) == 12
    for fold in folds:
        assert fold.held_out_subject not in fold.fit_subjects
        assert len(fold.fit_subjects) == 11
        assert set(fold.fit_subjects) | {fold.held_out_subject} == set(
            U.U1_CALIBRATION_SUBJECTS
        )


def test_every_permitted_subject_is_held_out_exactly_once():
    folds = assign_calibration_folds()
    held_out = [fold.held_out_subject for fold in folds]
    assert sorted(held_out) == sorted(U.U1_CALIBRATION_SUBJECTS)
    assert len(set(held_out)) == 12


def test_fold_identity_is_deterministic_and_order_independent():
    a = assign_calibration_folds()
    b = assign_calibration_folds(list(reversed(U.U1_CALIBRATION_SUBJECTS)))
    assert fold_assignment_digest(a) == fold_assignment_digest(b)
    assert a == b


def test_fold_assignment_validator_accepts_the_frozen_design():
    proof = validate_fold_assignment(assign_calibration_folds())
    assert proof["fold_design"] == "leave_one_subject_out"
    assert proof["fold_count"] == 12
    assert proof["no_fold_fits_on_its_held_out_subject"] is True
    assert proof["fold_assignment_basis"] == ("frozen_subject_identity_ascending_only")


def test_fold_validator_refuses_a_fold_fitted_on_its_own_subject():
    """The core cross-fitting guarantee must be enforced, not assumed."""
    folds = list(assign_calibration_folds())
    first = folds[0]
    folds[0] = first._replace(
        fit_subjects=(first.held_out_subject,) + first.fit_subjects[1:]
    )
    with pytest.raises(U1ProtocolError, match="fits on its own held-out"):
        validate_fold_assignment(tuple(folds))


def test_fold_validator_refuses_an_incomplete_assignment():
    folds = assign_calibration_folds()[:-1]
    with pytest.raises(U1ProtocolError, match="Expected 12 folds"):
        validate_fold_assignment(folds)


def test_incomplete_subject_set_is_refused():
    with pytest.raises(U1ProtocolError, match="are absent"):
        assign_calibration_folds(U.U1_CALIBRATION_SUBJECTS[:6])


def test_duplicate_subjects_are_refused():
    doubled = list(U.U1_CALIBRATION_SUBJECTS) + [U.U1_CALIBRATION_SUBJECTS[0]]
    with pytest.raises(U1ProtocolError, match="Duplicate"):
        require_calibration_subjects(doubled)


def test_fold_assignment_cannot_depend_on_performance():
    """No score-derived quantity may be referenced as code by the assignment.

    Checked over AST identifiers rather than raw text, so prose that *forbids*
    such an input does not count as using one.
    """
    tree = ast.parse(Path(U.__file__).read_text())
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    referenced |= {
        k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords
    } - {None}
    forbidden = {
        "auprc",
        "auroc",
        "false_positive_rate",
        "prediction",
        "predictions",
        "error_rate",
        "calibration_error",
        "prevalence",
        "sensitivity",
        "specificity",
    }
    assert not (referenced & forbidden), referenced & forbidden


# ---------------------------------------------------------------------------
# Frozen science must not drift
# ---------------------------------------------------------------------------


def test_no_m2_scientific_constant_changes():
    from cardiosentinel.neural.m2_gate import (
        M1L_CLASSIFICATION_THRESHOLD,
        M2_GATE_RECEIPT_SHA256,
        M2_PROTOCOL_SHA256,
        M2_RETAINED_EXPERIMENT_ID,
        NORMAL_EVIDENCE_THRESHOLD,
    )

    assert U.U1_CLASSIFICATION_THRESHOLD == M1L_CLASSIFICATION_THRESHOLD
    assert U.U1_NORMAL_EVIDENCE_THRESHOLD == NORMAL_EVIDENCE_THRESHOLD
    assert M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    assert M2_PROTOCOL_SHA256 == (
        "a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c"
    )
    assert M2_GATE_RECEIPT_SHA256 == (
        "5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24"
    )


def test_u1_binds_the_retained_m2g_arm_not_the_control():
    from cardiosentinel.neural import m2_selection as S

    assert U.U1_M2G_ARM_RESULT_SHA256 == S.M2_RETAINED_ARM_RESULT_SHA256
    assert U.U1_M2G_LOCK_SHA256 == S.M2_RETAINED_LOCK_SHA256
    assert U.U1_M2_SUITE_SHA256 == S.M2_SUITE_SHA256
    assert U.U1_M2_RETENTION_DECISION_SHA256 == S.M2_RETENTION_DECISION_SHA256
    # the naive control arm is never a U1 calibration input
    assert U.U1_M2G_ARM_RESULT_SHA256 != S.M2_CONTROL_ARM_RESULT_SHA256


def test_classification_threshold_is_not_changeable_by_u1():
    assert U.U1_MAY_CHANGE_CLASSIFICATION_THRESHOLD is False
    assert U.U1_CLASSIFICATION_THRESHOLD == 0.7554003000259399


def test_g4_admission_score_is_never_the_calibration_input():
    assert U.U1_CALIBRATION_INPUT_FIELD == "score"
    assert "normal_evidence" in U.U1_FORBIDDEN_CALIBRATION_INPUTS
    assert "g4_normal_evidence" in U.U1_FORBIDDEN_CALIBRATION_INPUTS
    # the two thresholds are distinct quantities and must stay distinct
    assert U.U1_CLASSIFICATION_THRESHOLD != U.U1_NORMAL_EVIDENCE_THRESHOLD


def test_full_replay_is_never_a_metric_denominator():
    assert U.U1_FULL_REPLAY_IS_METRIC_DENOMINATOR is False
    assert U.U1_CALIBRATION_POPULATION == "primary_metric"
    assert U.U1_PRIMARY_ROW_COUNT == 473_897
    assert U.U1_FULL_REPLAY_ROW_COUNT == 492_904


# ---------------------------------------------------------------------------
# The module cannot execute science
# ---------------------------------------------------------------------------


def test_protocol_module_imports_only_the_standard_library():
    """No scorer, replay, torch or run-artifact reader is reachable from here."""
    tree = ast.parse(Path(U.__file__).read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {
        "hashlib",
        "json",
        "math",
        "pathlib",
        "typing",
        "__future__",
    }, modules
    assert "torch" not in modules
    assert "numpy" not in modules
    assert "cardiosentinel" not in modules


def test_protocol_module_cannot_fit_score_or_mutate():
    tree = ast.parse(Path(U.__file__).read_text())
    forbidden = {
        "fit",
        "train",
        "backward",
        "step",
        "score_batch",
        "write_text",
        "write_bytes",
        "unlink",
        "rmtree",
        "rename",
        "save",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            assert name not in forbidden, name


def test_protocol_chooses_no_routing_threshold_and_no_retention():
    assert U.U1_ROUTING_THRESHOLD_CHOSEN_HERE is False
    assert U.U1_AUTOMATIC_RETENTION is False
    assert U.U1_AUTOMATIC_U2_TRANSITION is False
    assert U.U1_M2_RERUN_PERMITTED is False
    assert U.U1_EPISODE_PERSISTENCE_IMPLEMENTED is False
    assert U.U1_NEURAL_UNCERTAINTY_MODEL is False


def test_identity_record_is_complete_and_self_consistent():
    identity = u1_protocol_identity()
    assert identity["protocol_sha256"] == U.U1_PROTOCOL_SHA256
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"
    assert identity["true_logits_persisted"] is False
    assert identity["primary_method"] == "platt_logistic_on_recovered_logit"
    assert identity["comparator_method"] == ("temperature_only_on_recovered_logit")
    assert identity["tie_break"] == "simpler_nested_model"
    assert identity["retained_coverage"] == 0.90
    assert identity["fold_assignment"]["fold_count"] == 12
    assert identity["uncertainty_definition"] == (
        "calibrated_probability_of_frozen_decision_error"
    )


def test_coverage_grid_is_frozen_sorted_and_includes_the_no_routing_reference():
    grid = U.U1_COVERAGE_GRID
    assert list(grid) == sorted(grid)
    assert grid[-1] == 1.00
    assert U.U1_RETAINED_COVERAGE in grid
    assert len(set(grid)) == len(grid)


# ---------------------------------------------------------------------------
# Two calibration artifacts: OOF development vs deployable configuration
# ---------------------------------------------------------------------------


def test_oof_and_deployment_calibrators_are_distinct_roles():
    assert U.U1_OOF_ARTIFACT != U.U1_DEPLOY_ARTIFACT
    assert U.U1_CALIBRATION_ARTIFACTS == (
        U.U1_OOF_ARTIFACT,
        U.U1_DEPLOY_ARTIFACT,
    )
    assert U.U1_OOF_CALIBRATOR_COUNT == 12
    assert U.U1_DEPLOY_CALIBRATOR_COUNT == 1


def test_development_evaluation_is_out_of_fold_only():
    assert U.U1_DEVELOPMENT_EVIDENCE_SOURCE == U.U1_OOF_ARTIFACT
    assert U.U1_DEVELOPMENT_EVIDENCE_SOURCE != U.U1_DEPLOY_ARTIFACT
    assert U.U1_DEPLOY_FIT_IS_EVALUATION is False


def test_downstream_t1_t2_development_use_is_out_of_fold_only():
    assert U.U1_DOWNSTREAM_DEVELOPMENT_CALIBRATION_SOURCE == U.U1_OOF_ARTIFACT
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "must be the OOF" in text
    assert (
        "never be given a probability produced by a\ncalibrator that was "
        "fitted using that subject" in text
    )


def test_final_calibrator_fit_population_is_all_twelve_validation_subjects():
    assert U.U1_DEPLOY_FIT_SUBJECTS == U.U1_CALIBRATION_SUBJECTS
    assert len(U.U1_DEPLOY_FIT_SUBJECTS) == 12
    # and never a TEST subject
    for subject in _frozen_test_subjects():
        assert subject not in U.U1_DEPLOY_FIT_SUBJECTS


def test_final_family_cannot_differ_from_the_oof_selected_family():
    assert U.U1_FAMILY_SELECTION_EVIDENCE == "out_of_fold_only"
    assert U.U1_FINAL_FIT_MAY_RESELECT_FAMILY is False
    assert U.U1_FINAL_FIT_FALLBACK_PERMITTED is False


def test_no_in_sample_final_fit_performance_claim_is_authorised():
    assert U.U1_DEPLOY_IN_SAMPLE_PERFORMANCE_CLAIM_AUTHORISED is False
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "parameterisation, not evaluation" in text


def test_u_star_dev_and_u_star_deploy_are_different_artifact_concepts():
    assert U.U1_DEV_THRESHOLD_NAME != U.U1_DEPLOY_THRESHOLD_NAME
    assert U.U1_DEV_THRESHOLD_NAME == "u_star_dev"
    assert U.U1_DEPLOY_THRESHOLD_NAME == "u_star_deploy"
    dev = select_routing_threshold([0.1, 0.2, 0.3], ["a", "b", "c"], 0.60)
    deploy = select_routing_threshold(
        [0.1, 0.2, 0.3], ["a", "b", "c"], 0.60, name=U.U1_DEPLOY_THRESHOLD_NAME
    )
    assert dev.name == "u_star_dev"
    assert deploy.name == "u_star_deploy"


def test_unknown_threshold_name_is_refused():
    with pytest.raises(U1ProtocolError, match="Unknown routing threshold"):
        select_routing_threshold([0.1], ["a"], 1.0, name="u_star_test")


# ---------------------------------------------------------------------------
# The frozen empirical order statistic
# ---------------------------------------------------------------------------


def test_frozen_primary_size_shows_why_lower_quantile_was_insufficient():
    """N = 473,897 at c* = 0.90 is exactly the case that failed."""
    n = U.U1_PRIMARY_ROW_COUNT
    assert n == 473_897
    target = U.U1_RETAINED_COVERAGE

    lower_like = int(target * n)  # 426,507
    assert lower_like == 426_507
    assert lower_like / n < target  # 0.8999993669... -- below target

    k = routing_threshold_rank(n, target)
    assert k == 426_508
    assert k / n >= target  # 0.9000014771... -- at or above target


def test_order_statistic_guarantees_achieved_coverage_at_or_above_target():
    for n in (1, 2, 7, 10, 13, 100, 999, 473_897):
        for target in U.U1_COVERAGE_GRID:
            k = routing_threshold_rank(n, target)
            assert 1 <= k <= n
            assert k / n >= target


def test_achieved_coverage_never_falls_below_target_on_synthetic_data():
    values = [i / 1000 for i in range(1000)]
    ids = [f"w{i:04d}" for i in range(1000)]
    for target in U.U1_COVERAGE_GRID:
        result = select_routing_threshold(values, ids, target)
        assert result.achieved_coverage >= target
        assert result.accepted_count == sum(1 for v in values if v <= result.u_star)


def test_ties_can_only_increase_achieved_coverage():
    # 10 rows, half of them tied at 0.5, target 0.60 -> k = 6 lands on a tie
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.9, 1.0]
    ids = [f"w{i}" for i in range(10)]
    result = select_routing_threshold(values, ids, 0.60)
    assert result.u_star == 0.5
    assert result.threshold_tie_count == 4
    # inclusive acceptance sweeps every tied row in
    assert result.accepted_count == 8
    assert result.achieved_coverage == 0.8
    assert result.achieved_coverage > 0.60


def test_threshold_tie_handling_is_deterministic():
    values = [0.5] * 20
    ids = [f"w{i:02d}" for i in range(20)]
    first = select_routing_threshold(values, ids, 0.90)
    second = select_routing_threshold(list(reversed(values)), list(reversed(ids)), 0.90)
    assert first.u_star == second.u_star
    assert first.accepted_count == second.accepted_count == 20
    assert first.achieved_coverage == 1.0


def test_threshold_requires_a_stable_id_for_every_value():
    with pytest.raises(U1ProtocolError, match="tie-break is not"):
        select_routing_threshold([0.1, 0.2], ["only-one"], 0.5)


def test_threshold_refuses_empty_and_out_of_range_targets():
    with pytest.raises(U1ProtocolError, match="at least one row"):
        routing_threshold_rank(0, 0.9)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(U1ProtocolError, match="must lie in"):
            routing_threshold_rank(10, bad)


def test_threshold_report_fields_are_frozen():
    assert U.U1_THRESHOLD_REPORT_FIELDS == (
        "target_coverage",
        "achieved_coverage",
        "accepted_count",
        "threshold_tie_count",
    )
    result = select_routing_threshold([0.1, 0.2, 0.3], ["a", "b", "c"], 0.90)
    for field in U.U1_THRESHOLD_REPORT_FIELDS:
        assert hasattr(result, field)


def test_no_library_quantile_convention_governs_the_threshold():
    assert not hasattr(U, "U1_QUANTILE_CONVENTION")
    assert U.U1_THRESHOLD_RULE == "empirical_order_statistic_ceil_1_based"
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "has been **removed**" in text


# ---------------------------------------------------------------------------
# Deterministic ECE binning
# ---------------------------------------------------------------------------


def test_equal_width_bins_are_deterministic_and_close_the_final_interval():
    edges = equal_width_bin_edges()
    assert len(edges) == 15
    assert edges[0][0] == 0.0
    assert edges[-1][1] == 1.0
    # p == 1.0 must land in the final bin, not fall off the end
    assert equal_width_bin_index(1.0) == 14
    assert equal_width_bin_index(0.0) == 0
    # lower edge inclusive, upper exclusive except the last
    assert equal_width_bin_index(1 / 15) == 1
    assert equal_width_bin_index(1 / 15 - 1e-12) == 0
    assert U.U1_ECE_EQUAL_WIDTH_FINAL_BIN_CLOSED is True


def test_equal_width_bin_index_refuses_out_of_range_probability():
    for bad in (-1e-9, 1.0000001):
        with pytest.raises(U1ProtocolError, match="outside"):
            equal_width_bin_index(bad)


def test_equal_mass_group_sizes_differ_by_at_most_one_and_sum_exactly():
    for n in (15, 16, 29, 100, 473_897):
        sizes = equal_mass_group_sizes(n)
        assert len(sizes) == 15
        assert sum(sizes) == n
        assert max(sizes) - min(sizes) <= 1


def test_equal_mass_groups_are_contiguous_and_cover_every_row():
    n = 100
    boundaries = equal_mass_group_boundaries(
        [i / n for i in range(n)], [f"w{i:03d}" for i in range(n)]
    )
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == n
    for (_, stop), (start, _) in zip(boundaries, boundaries[1:]):
        assert stop == start


def test_equal_mass_refuses_too_few_rows_and_mismatched_ids():
    with pytest.raises(U1ProtocolError, match="cannot fill"):
        equal_mass_group_sizes(14)
    # message unified across both helpers during hardening; the refusal is the
    # same one, now reporting the counts that disagreed
    with pytest.raises(U1ProtocolError, match="tie-break is not"):
        equal_mass_group_boundaries([0.1, 0.2], ["a"])


def test_equal_mass_semantics_are_not_library_delegated():
    assert U.U1_ECE_LIBRARY_QUANTILE_PERMITTED is False
    assert U.U1_ECE_EQUAL_MASS_SORT_KEY == ("calibrated_probability", "stable_id")
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "never delegated to a library-default quantile" in text


# ---------------------------------------------------------------------------
# Dependence and bootstrap claim boundaries
# ---------------------------------------------------------------------------


def test_windows_are_not_claimed_to_be_independent_evidence():
    assert U.U1_WINDOWS_ARE_INDEPENDENT_EVIDENCE is False
    assert U.U1_INFERENTIAL_UNIT == "subject"
    assert U.U1_INFERENTIAL_UNIT_COUNT == 12
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "effective independent support\nremains subject-level" in text
    assert "does **not** remove within-subject dependence" in text
    # the discarded claims must be gone, in any of their earlier phrasings
    for banned in (
        "fit variance is negligible",
        "high-variance objection does not apply",
        "usual objection to LOSO does not apply",
    ):
        assert banned not in text, banned


def test_bootstrap_claim_boundary_is_frozen():
    assert U.U1_BOOTSTRAP_REPLICATES == 1000
    assert U.U1_BOOTSTRAP_SEED == 2026
    assert U.U1_BOOTSTRAP_UNIT == "subject"
    assert U.U1_BOOTSTRAP_REFITS_FOLDS_PER_REPLICATE is False
    assert U.U1_BOOTSTRAP_CLAIM == (
        "between_subject_variation_conditional_on_fitted_oof_calibration"
    )
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "not** a complete bootstrap of calibrator re-fitting" in text


def test_c_star_is_a_design_assumption_not_measured_capacity():
    assert U.U1_RETAINED_COVERAGE == 0.90
    assert U.U1_RETAINED_COVERAGE_IS_MEASURED_CAPACITY is False
    assert U.U1_RETAINED_COVERAGE_BASIS == (
        "a_priori_operational_design_assumption_reference_operating_point"
    )
    text = U.U1_PROTOCOL_PATH.read_text()
    assert "a-priori operational design assumption" in text
    assert "not** as measured deployment capacity" in text
    assert "measured later in E1" in text


# ---------------------------------------------------------------------------
# Equal-mass ECE: the helper must actually perform the frozen sort
# ---------------------------------------------------------------------------


def test_equal_mass_helper_actually_sorts_by_probability_then_stable_id():
    """The frozen order must be produced, not assumed of the caller."""
    probabilities = [0.9, 0.1, 0.5, 0.5, 0.3]
    ids = ["e", "a", "d", "c", "b"]
    order = equal_mass_sort_order(probabilities, ids)
    # ascending p, then stable_id: a(0.1) b(0.3) c(0.5) d(0.5) e(0.9)
    assert [ids[i] for i in order] == ["a", "b", "c", "d", "e"]
    assert [probabilities[i] for i in order] == [0.1, 0.3, 0.5, 0.5, 0.9]


def test_equal_mass_group_membership_is_independent_of_incoming_row_order():
    """Shuffling the input must not move a single row between groups."""
    n = 45  # 45 = 15 * 3, exactly three rows per group
    base_p = [round((i % 9) / 10, 4) for i in range(n)]
    base_ids = [f"w{i:03d}" for i in range(n)]

    canonical = equal_mass_groups(base_p, base_ids)
    membership = {
        identity: group.group_index
        for group in canonical
        for identity in group.member_stable_ids
    }

    order = list(range(n))
    for shift in (1, 7, 23, 44):
        shuffled = order[shift:] + order[:shift]
        shuffled_groups = equal_mass_groups(
            [base_p[i] for i in shuffled], [base_ids[i] for i in shuffled]
        )
        shuffled_membership = {
            identity: group.group_index
            for group in shuffled_groups
            for identity in group.member_stable_ids
        }
        assert shuffled_membership == membership


def test_stable_id_decides_which_side_of_a_boundary_a_tie_falls_on():
    """A tie deliberately spanning an equal-mass boundary must be resolved."""
    bins = 3
    # 6 rows, 2 per group; the four 0.5 ties span the group-0/1 and 1/2 edges
    probabilities = [0.1, 0.5, 0.5, 0.5, 0.5, 0.9]
    ids = ["id1", "id5", "id3", "id2", "id4", "id6"]

    groups = equal_mass_groups(probabilities, ids, bins=bins)
    assert [g.count for g in groups] == [2, 2, 2]

    # ascending (p, stable_id): id1(0.1) id2 id3 id4 id5 (all 0.5) id6(0.9)
    assert groups[0].member_stable_ids == ("id1", "id2")
    assert groups[1].member_stable_ids == ("id3", "id4")
    assert groups[2].member_stable_ids == ("id5", "id6")

    # the tie is genuinely split across a boundary by stable_id alone
    assert groups[0].maximum_probability == 0.5
    assert groups[1].minimum_probability == 0.5

    # and reversing the input changes nothing
    reversed_groups = equal_mass_groups(
        list(reversed(probabilities)), list(reversed(ids)), bins=bins
    )
    assert [g.member_stable_ids for g in reversed_groups] == [
        g.member_stable_ids for g in groups
    ]


def test_every_row_belongs_to_exactly_one_equal_mass_group():
    n = 100
    probabilities = [round((i * 7 % 100) / 100, 4) for i in range(n)]
    ids = [f"w{i:03d}" for i in range(n)]
    groups = equal_mass_groups(probabilities, ids)

    seen_indices = [i for g in groups for i in g.member_indices]
    seen_ids = [s for g in groups for s in g.member_stable_ids]
    assert sorted(seen_indices) == list(range(n))
    assert sorted(seen_ids) == sorted(ids)
    assert len(seen_indices) == n == len(set(seen_indices))
    assert sum(g.count for g in groups) == n


def test_boundaries_refuse_unsorted_rows():
    """Applying frozen boundaries to unsorted rows must be impossible."""
    probabilities = [0.9, 0.1, 0.5] * 5
    ids = [f"w{i:02d}" for i in range(15)]
    with pytest.raises(U1ProtocolError, match="not in the frozen"):
        equal_mass_group_boundaries(probabilities, ids)


def test_boundaries_accept_already_sorted_rows():
    n = 30
    probabilities = [i / n for i in range(n)]
    ids = [f"w{i:03d}" for i in range(n)]
    boundaries = equal_mass_group_boundaries(probabilities, ids)
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == n
    # and they agree with the sorting construction path
    groups = equal_mass_groups(probabilities, ids)
    assert [stop - start for start, stop in boundaries] == [g.count for g in groups]


# ---------------------------------------------------------------------------
# stable_id integrity
# ---------------------------------------------------------------------------


def test_duplicate_stable_ids_are_refused_not_deduplicated():
    with pytest.raises(U1ProtocolError, match="Duplicate stable_ids"):
        select_routing_threshold([0.1, 0.2, 0.3], ["a", "a", "b"], 0.5)
    with pytest.raises(U1ProtocolError, match="Duplicate stable_ids"):
        equal_mass_groups([0.1, 0.2, 0.3], ["a", "a", "b"], bins=3)


def test_empty_stable_ids_are_refused():
    for bad in ("", "   "):
        with pytest.raises(U1ProtocolError, match="is empty"):
            select_routing_threshold([0.1, 0.2], [bad, "b"], 0.5)
        with pytest.raises(U1ProtocolError, match="is empty"):
            equal_mass_sort_order([0.1, 0.2], [bad, "b"])


def test_missing_stable_ids_are_refused_for_both_helpers():
    with pytest.raises(U1ProtocolError, match="tie-break is not"):
        select_routing_threshold([0.1, 0.2], ["only-one"], 0.5)
    with pytest.raises(U1ProtocolError, match="tie-break is not"):
        equal_mass_sort_order([0.1, 0.2], ["only-one"])


# ---------------------------------------------------------------------------
# numeric domain
# ---------------------------------------------------------------------------


def test_non_finite_uncertainty_is_refused_before_sorting():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(U1ProtocolError, match="NaN and infinities"):
            select_routing_threshold([0.1, bad], ["a", "b"], 0.5)


def test_out_of_range_uncertainty_is_refused():
    for bad in (-0.001, 1.001, 2.0, -1.0):
        with pytest.raises(U1ProtocolError, match=r"outside \[0, 1\]"):
            select_routing_threshold([0.1, bad], ["a", "b"], 0.5)


def test_non_finite_ece_probability_is_refused():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(U1ProtocolError, match="NaN and infinities"):
            equal_mass_groups([0.1, 0.2, bad], ["a", "b", "c"], bins=3)


def test_out_of_range_ece_probability_is_refused():
    for bad in (-1e-9, 1.0000001, 5.0):
        with pytest.raises(U1ProtocolError, match=r"outside \[0, 1\]"):
            equal_mass_groups([0.1, 0.2, bad], ["a", "b", "c"], bins=3)


def test_domain_checks_do_not_disturb_the_frozen_routing_rule():
    """Hardening added checks only; the rule itself is untouched."""
    assert U.U1_THRESHOLD_RULE == "empirical_order_statistic_ceil_1_based"
    assert U.U1_THRESHOLD_ACCEPTANCE == "u <= u_star"
    assert U.U1_THRESHOLD_SORT_KEY == ("uncertainty", "stable_id")
    n = U.U1_PRIMARY_ROW_COUNT
    assert routing_threshold_rank(n, 0.90) == 426_508
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.9, 1.0]
    ids = [f"w{i}" for i in range(10)]
    result = select_routing_threshold(values, ids, 0.60)
    assert (result.u_star, result.accepted_count, result.achieved_coverage) == (
        0.5,
        8,
        0.8,
    )
