"""Tests for the frozen T1-v1 canonical development execution specification.

These prove the execution *mechanics* only. The science is proved by
``test_t1_protocol.py`` and is not restated here.

Two things are load-bearing throughout: the specification document digest, and
the proof that the comment-only repair to ``t1_protocol.py`` left its executable
AST untouched.

Synthetic structures only. No real run artifact is required, so these run
identically in CI and locally.
"""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_execution_spec as X
from cardiosentinel.neural import t1_protocol as P
from cardiosentinel.neural.t1_execution_spec import T1ExecutionSpecError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# t1_protocol.py at the specification's starting master,
# 9fa7e88ee648a71e397516ccce210dcf0f06c409, digested with every COMMENT token
# removed and every line right-stripped. The §N comment repair must not move
# this digest by a single bit.
#
# This is deliberately NOT a digest of `ast.dump` output. `ast.dump` serialises
# internal node fields, which change between interpreter versions -- 3.12 added
# `type_params` to function and class nodes, for one -- so an `ast.dump` digest
# binds the file AND the interpreter that read it, and fails the moment CI runs
# a different Python than the author did. Removing comment tokens from the
# source text is a property of the file alone.
T1_PROTOCOL_COMMENT_STRIPPED_SHA256 = (
    "66548c4ced7513ccbf83781417e5cd23fd3293f49fa0079873834f3c4d6ec17c"
)


def _comment_stripped_digest(path: Path) -> str:
    """Digest the source with every comment removed, interpreter-independently.

    Comment spans come from `tokenize`, so a string that merely looks like a
    comment is untouched. Lines are right-stripped afterwards, because deleting
    a trailing comment leaves the whitespace that preceded it.
    """
    source = path.read_text()
    spans: dict[int, list[tuple[int, int]]] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.setdefault(token.start[0], []).append((token.start[1], token.end[1]))
    stripped = []
    for number, line in enumerate(source.splitlines(), start=1):
        for start, end in sorted(spans.get(number, []), reverse=True):
            line = line[:start] + line[end:]
        stripped.append(line.rstrip())
    return hashlib.sha256("\n".join(stripped).encode()).hexdigest()


def _module_calls(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text())
    return {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }


def _module_imports(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


# ---------------------------------------------------------------------------
# 1-5  Identity
# ---------------------------------------------------------------------------


def test_01_execution_spec_document_digest_is_exact():
    assert X.validate_t1_execution_spec_document() == X.T1_EXECUTION_SPEC_SHA256
    assert X.T1_EXECUTION_SPEC_PATH.is_file()


def test_01b_a_mutated_execution_spec_is_refused(tmp_path):
    forged = tmp_path / "forged.md"
    forged.write_text(X.T1_EXECUTION_SPEC_PATH.read_text() + "\nappended\n")
    with pytest.raises(T1ExecutionSpecError, match="immutable"):
        X.validate_t1_execution_spec_document(forged)


def test_02_the_t1_protocol_digest_remains_exact():
    assert X.T1_PROTOCOL_DOCUMENT_SHA256 == P.T1_PROTOCOL_SHA256
    assert P.validate_t1_protocol_document() == P.T1_PROTOCOL_SHA256


def test_03_experiment_identity_is_exact():
    assert X.T1_EXPERIMENT_IDENTITY == "T1_state_machine_v1"


def test_04_run_root_is_exact():
    assert X.T1_RUN_ROOT_RELATIVE == Path(
        "cardiosentinel-runs/phase9-t1-development-v1"
    )


def test_05_attempt_identity_is_exact_and_deterministic():
    assert X.T1_DEVELOPMENT_ATTEMPT_ID == "t1-v1-development"
    assert X.T1_ATTEMPT_NAME_IS_DETERMINISTIC is True
    assert X.T1_ATTEMPT_NAME_CARRIES_TIMESTAMP is False
    assert X.T1_ATTEMPT_NAME_CARRIES_UUID is False
    assert X.T1_ATTEMPT_NAME_CARRIES_RANDOM_SUFFIX is False
    assert X.T1_AUTOMATIC_RETRY_PERMITTED is False
    assert X.T1_RECOVERY_IDENTITY_PREDECLARED is False
    assert X.T1_ALTERNATE_RUN_ROOT_PERMITTED is False
    assert X.T1_CLAIM_CONSUMED_ONCE_CREATED is True


# ---------------------------------------------------------------------------
# 6-7  CLI surface
# ---------------------------------------------------------------------------


def test_06_the_future_cli_has_only_the_flag_and_the_expected_git_sha():
    assert X.T1_FUTURE_CLI_OPTIONS == (
        "--execute-canonical-development",
        "--expected-git-sha",
    )
    assert X.T1_FUTURE_CLI_MODULE == "cardiosentinel.neural.t1_development_run"
    for option in X.T1_FUTURE_CLI_OPTIONS:
        assert X.require_cli_option_permitted(option) == option
    assert X.T1_FUTURE_EXECUTION_GIT_SHA_KNOWN is False


@pytest.mark.parametrize("option", X.T1_FORBIDDEN_CLI_OPTIONS)
def test_07_no_scientific_cli_option_exists(option):
    with pytest.raises(T1ExecutionSpecError, match="after seeing results"):
        X.require_cli_option_permitted(option)


def test_07b_the_specification_sha_is_not_frozen_as_the_execution_sha():
    """The future harness merge SHA is not known yet and must not be guessed."""
    assert X.T1_SPECIFICATION_STARTING_GIT_SHA == (
        "9fa7e88ee648a71e397516ccce210dcf0f06c409"
    )
    assert X.T1_GIT_SHA_IS_THE_AUTHORIZATION_MECHANISM is True
    assert not hasattr(X, "T1_EXECUTION_GIT_SHA")


# ---------------------------------------------------------------------------
# 8-9  Claim ordering
# ---------------------------------------------------------------------------


def test_08_pre_claim_stages_precede_per_row_input_opening():
    for stage in (
        X.STAGE_VERIFY_GIT,
        X.STAGE_VALIDATE_PROTOCOL,
        X.STAGE_VALIDATE_SPEC,
        X.STAGE_PROVE_TEST_UNOPENED,
        X.STAGE_PROVE_ATTEMPT_ABSENT,
    ):
        X.require_stage_precedes(stage, X.STAGE_ASSEMBLE_LABEL_BLIND)


def test_09_the_claim_precedes_every_per_row_access_stage():
    for stage in X.T1_PER_ROW_ACCESS_STAGES:
        X.require_stage_precedes(X.STAGE_CLAIM, stage)
        assert X.require_claim_before_per_row_access(stage) == stage
    X.require_stage_precedes(X.STAGE_PROVE_ATTEMPT_ABSENT, X.STAGE_CLAIM)


# ---------------------------------------------------------------------------
# 10-14  Upstream verification and no-refit
# ---------------------------------------------------------------------------


def test_10_m2_retention_validation_is_required():
    assert "validate_retained_m2_arm" in X.T1_REQUIRED_UPSTREAM_VALIDATORS
    assert X.T1_REQUIRED_M2_RETAINED_ARM == "M2-G"
    X.require_stage_precedes(X.STAGE_VALIDATE_M2, X.STAGE_ASSEMBLE_LABEL_BLIND)


def test_11_u1_retention_and_canonical_validation_are_required():
    assert "validate_u1_retention_decision" in X.T1_REQUIRED_UPSTREAM_VALIDATORS
    assert "validate_retained_u1_calibration" in X.T1_REQUIRED_UPSTREAM_VALIDATORS
    assert X.T1_REQUIRED_U1_FAMILY == "platt_logistic_on_recovered_logit"
    assert X.T1_U1_FOLD_COUNT == 12
    assert X.T1_U1_FIT_SUBJECTS_PER_FOLD == 11


def test_12_t2_retention_validation_is_required():
    assert "validate_retained_t2_arm" in X.T1_REQUIRED_UPSTREAM_VALIDATORS
    assert X.T1_REQUIRED_T2_RETAINED_ARM == "causal_s4d_longitudinal_v1"
    assert X.T1_REQUIRED_T2_SCORE_SEMANTICS == "uncalibrated_temporal_model_score"
    assert X.T1_PARALLEL_WEAKER_VERIFIER_PERMITTED is False


@pytest.mark.parametrize("fitting", X.T1_U1_FORBIDDEN_FITTING_CALLABLES)
def test_13_u1_refit_is_forbidden(fitting):
    assert X.T1_U1_REFIT_PERMITTED is False
    with pytest.raises(T1ExecutionSpecError, match="refit"):
        X.require_no_refit(fitting)


def test_14_the_u1_deployment_calibrator_is_forbidden_for_development():
    assert X.T1_U1_DEPLOYMENT_CALIBRATOR_PERMITTED_FOR_DEVELOPMENT is False
    assert P.T1_U1_DEPLOYMENT_CALIBRATOR_PERMITTED_FOR_DEVELOPMENT is False
    assert X.T1_M2_REPLAY_PERMITTED is False
    assert X.T1_T2_REPLAY_PERMITTED is False


def test_15_full_timeline_calibration_uses_already_fitted_platt_parameters():
    assert X.T1_U1_APPLY_CALLABLE == "U1Calibrator.apply_to_scores"
    assert X.T1_CALIBRATION_IS_A_FIT is False
    assert X.T1_CALIBRATION_CONTRACT == P.T1_FULL_TIMELINE_CALIBRATION_CONTRACT
    assert X.T1_U1_CLAMP_DELTA == P.T1_U1_CLAMP_DELTA
    assert X.T1_DETECTOR_THRESHOLD == P.T1_DETECTOR_THRESHOLD
    assert X.T1_U1_FOLD_MANIFEST_NAME == "U1_FOLD_MANIFEST.json"


# ---------------------------------------------------------------------------
# 16-20  The label-blind assembly firewall
# ---------------------------------------------------------------------------


def test_16_target_family_is_forbidden_during_transition_assembly():
    assert "target_family" in X.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    assert "target_family" in P.T1_FORBIDDEN_TRANSITION_INPUTS
    with pytest.raises(T1ExecutionSpecError, match="not be deployable"):
        X.require_label_blind_member("target_family")


def test_17_label_is_forbidden_during_transition_assembly():
    assert "label" in X.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    assert "label" in P.T1_FORBIDDEN_TRANSITION_INPUTS
    with pytest.raises(T1ExecutionSpecError, match="not be deployable"):
        X.require_label_blind_member("label")


def test_18_whole_npz_materialisation_is_forbidden_by_design():
    assert X.T1_WHOLE_NPZ_MATERIALISATION_PERMITTED is False
    assert X.T1_MEMBER_RESTRICTED_READER_REQUIRED is True
    for reader in X.T1_T2_READERS_FORBIDDEN_FOR_LABEL_BLIND_ASSEMBLY:
        with pytest.raises(T1ExecutionSpecError, match="materialises every column"):
            X.require_member_restricted_reader(reader)


def test_19_and_20_forbidden_members_stay_closed_until_the_evaluation_join():
    """The T2 identity NPZ really does carry evaluation annotation."""
    from cardiosentinel.neural.t2_outer_evidence import T2_OUTER_IDENTITY_COLUMNS

    for member in ("label", "target_family", "primary_mask"):
        assert member in T2_OUTER_IDENTITY_COLUMNS, (
            "the specification's warning depends on these members existing"
        )
        assert member in X.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
        assert member not in X.T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND


def test_20b_permitted_members_are_accepted():
    for member in X.T1_T2_IDENTITY_MEMBERS_PERMITTED_LABEL_BLIND:
        assert X.require_label_blind_member(member) == member


def test_20c_the_m2_gate_outcome_is_not_a_transition_feature():
    assert "update_admitted" in X.T1_M2_COLUMNS_NOT_TRANSITION_FEATURES
    assert "m2_update_admitted" in P.T1_FORBIDDEN_TRANSITION_INPUTS
    assert "update_admitted" not in X.T1_M2_COLUMNS_USED


# ---------------------------------------------------------------------------
# 21-23  Alignment and census
# ---------------------------------------------------------------------------


def test_21_stable_id_equality_is_required():
    assert X.T1_STABLE_ID_EQUALITY_REQUIRED is True


def test_22_availability_mask_equality_is_required():
    assert X.T1_AVAILABILITY_MASK_EQUALITY_REQUIRED is True
    assert X.T1_AVAILABILITY_MISMATCH_IS_HARD_STOP is True
    assert X.T1_SYNTHETIC_SCORE_PERMITTED is False


def test_23_the_row_census_is_frozen_and_closes_exactly():
    assert X.T1_TIMELINE_ROW_COUNT == 492_904
    assert X.T1_EXPECTED_SCORED_ROWS == 492_898
    assert X.T1_EXPECTED_UNAVAILABLE_ROWS == 6
    assert (
        X.T1_EXPECTED_SCORED_ROWS + X.T1_EXPECTED_UNAVAILABLE_ROWS
        == X.T1_TIMELINE_ROW_COUNT
    )
    assert X.T1_TIMELINE_ROW_COUNT == P.T1_TIMELINE_ROW_COUNT
    assert X.T1_EXPECTED_SCORED_ROWS == P.T1_EXPECTED_SCORE_PRESENT_ROWS
    assert X.T1_EXPECTED_UNAVAILABLE_ROWS == P.T1_EXPECTED_UNAVAILABLE_ROWS


# ---------------------------------------------------------------------------
# 24-28  The fold-scoped label firewall
# ---------------------------------------------------------------------------


def test_24_fit_labels_open_before_fit_policy_selection_only():
    X.require_stage_precedes(X.STAGE_FOLD_OPEN_FIT_LABELS, X.STAGE_FOLD_SELECT)
    X.require_stage_precedes(
        X.STAGE_FOLD_OPEN_FIT_LABELS, X.STAGE_FOLD_GENERATE_THRESHOLDS
    )
    assert X.T1_GLOBAL_LABEL_TABLE_PERMITTED is False
    assert X.T1_FOLD_SCOPED_TARGET_AUTHORITY_REQUIRED is True
    assert X.T1_T2_IDENTITY_NPZ_AS_LABEL_TABLE_PERMITTED is False


def test_25_held_out_labels_remain_closed_during_fit_selection():
    X.require_stage_precedes(X.STAGE_FOLD_SELECT, X.STAGE_FOLD_OPEN_HELD_OUT_LABELS)
    X.require_stage_precedes(
        X.STAGE_FOLD_RUN_CANDIDATES, X.STAGE_FOLD_OPEN_HELD_OUT_LABELS
    )
    assert P.T1_HELD_OUT_LABELS_AVAILABLE_DURING_SELECTION is False
    with pytest.raises(T1ExecutionSpecError, match="not cross-fitting"):
        X.require_held_out_access_authorized({"selection_promoted": False})


def test_26_fold_selection_promotion_precedes_held_out_label_access():
    X.require_stage_precedes(
        X.STAGE_FOLD_PROMOTE_SELECTION, X.STAGE_FOLD_AUTHORIZE_HELD_OUT
    )
    X.require_stage_precedes(
        X.STAGE_FOLD_AUTHORIZE_HELD_OUT, X.STAGE_FOLD_OPEN_HELD_OUT_LABELS
    )
    assert X.T1_HELD_OUT_BARRIER_IS_STRUCTURAL is True
    assert X.T1_FOLD_SELECTION_REREAD_AND_DIGEST_VERIFIED is True

    # Promotion without a verified re-read is not promotion.
    with pytest.raises(T1ExecutionSpecError, match="digest verified"):
        X.require_held_out_access_authorized(
            {"selection_promoted": True, "selection_digest_verified": False}
        )
    with pytest.raises(T1ExecutionSpecError, match=X.T1_HELD_OUT_ACCESS_FLAG):
        X.require_held_out_access_authorized(
            {"selection_promoted": True, "selection_digest_verified": True}
        )
    authorized = {
        "selection_promoted": True,
        "selection_digest_verified": True,
        X.T1_HELD_OUT_ACCESS_FLAG: True,
    }
    assert X.require_held_out_access_authorized(authorized) is authorized


def test_26b_the_fold_selection_artifact_binds_the_whole_lineage():
    for binding in (
        "fold_index",
        "held_out_subject",
        "fit_subjects",
        "selected_policy",
        "selected_thresholds",
        "t1_protocol_sha256",
        "t1_execution_spec_sha256",
        "input_evidence_store_sha256",
        "test_accessed",
    ):
        assert binding in X.T1_FOLD_SELECTION_ARTIFACT_BINDINGS


def test_27_rejected_candidate_policies_never_run_on_the_held_out_subject():
    assert X.T1_REJECTED_POLICIES_RUN_ON_HELD_OUT is False
    with pytest.raises(T1ExecutionSpecError, match="second selection set"):
        X.require_single_held_out_policy_run(12)
    with pytest.raises(T1ExecutionSpecError, match="second selection set"):
        X.require_single_held_out_policy_run(2)


def test_28_exactly_one_selected_policy_runs_on_the_held_out_subject():
    assert X.T1_HELD_OUT_POLICY_RUNS_PER_FOLD == 1
    assert X.require_single_held_out_policy_run(1) == 1


# ---------------------------------------------------------------------------
# 29-31  Fold cardinality
# ---------------------------------------------------------------------------


def test_29_there_are_exactly_twelve_folds():
    assert X.T1_FOLD_COUNT == 12
    assert X.T1_FOLD_COUNT == P.T1_FOLD_COUNT
    assert len(P.t1_folds()) == 12


def test_30_there_are_exactly_twelve_candidate_policies_per_fold():
    assert X.T1_CANDIDATE_POLICIES_PER_FOLD == 12
    assert X.T1_CANDIDATE_POLICIES_PER_FOLD == P.T1_CANDIDATE_POLICY_COUNT
    assert len(P.candidate_policies()) == 12


def test_31_no_fold_retry_is_permitted():
    assert X.T1_FOLD_RETRY_PERMITTED is False
    assert P.T1_FOLD_RETRY_PERMITTED is False
    assert X.T1_AUTOMATIC_RETRY_PERMITTED is False


# ---------------------------------------------------------------------------
# 32-33  OOF before final configuration
# ---------------------------------------------------------------------------


def test_32_the_oof_result_precedes_the_final_all_validation_configuration():
    X.require_stage_precedes(X.STAGE_OOF_RESULT, X.STAGE_FINAL_CONFIGURATION)
    X.require_stage_precedes(X.STAGE_OOF_STATE_EVIDENCE, X.STAGE_OOF_RESULT)


def test_33_the_final_configuration_is_not_development_evidence():
    assert X.T1_FINAL_CONFIGURATION_IS_DEVELOPMENT_EVIDENCE is False
    assert X.T1_FINAL_CONFIGURATION_OVERWRITES_OOF_RESULT is False
    assert X.T1_PROMOTED_EVIDENCE_MAY_BE_OVERWRITTEN is False


# ---------------------------------------------------------------------------
# 34-36  Challenge and bootstrap
# ---------------------------------------------------------------------------


def test_34_the_challenge_join_happens_after_the_state_trace():
    X.require_stage_precedes(X.STAGE_OOF_STATE_EVIDENCE, X.STAGE_CHALLENGE)
    assert X.T1_CHALLENGE_JOIN_AFTER_STATE_TRACE is True


def test_35_challenge_identity_is_not_a_selection_or_transition_input():
    assert X.T1_CHALLENGE_IS_SELECTION_INPUT is False
    assert X.T1_CHALLENGE_IS_TRANSITION_INPUT is False
    assert X.T1_CHALLENGE_IS_THRESHOLD_GENERATION_INPUT is False
    assert P.T1_CHALLENGE_IS_SELECTION_INPUT is False
    assert P.T1_CHALLENGE_IS_TRANSITION_INPUT is False


def test_36_the_bootstrap_is_frozen_at_one_thousand_seed_2026_by_subject():
    assert X.T1_BOOTSTRAP_REPLICATES == 1000 == P.T1_BOOTSTRAP_REPLICATES
    assert X.T1_BOOTSTRAP_SEED == 2026 == P.T1_BOOTSTRAP_SEED
    assert X.T1_BOOTSTRAP_UNIT == "subject" == P.T1_BOOTSTRAP_UNIT
    assert X.T1_BOOTSTRAP_RESELECTS_POLICY is False
    assert P.T1_BOOTSTRAP_RESELECTS_POLICY is False
    assert X.T1_BOOTSTRAP_RESAMPLES_WITH_MULTIPLICITY is True


# ---------------------------------------------------------------------------
# 37-40  Firewalls and the binder's own incapacity
# ---------------------------------------------------------------------------


def test_37_test_remains_unopened():
    assert X.T1_TEST_ACCESSED is False
    assert X.T1_SEALED_TEST_STATE == "unopened"
    assert X.T1_TEST_CLI_OPTION_EXPOSED is False
    assert P.T1_TEST_ACCESSED is False
    with pytest.raises(T1ExecutionSpecError, match="sealed"):
        X.require_no_test_access("test")
    with pytest.raises(T1ExecutionSpecError, match="sealed"):
        X.require_no_test_access("TEST")
    assert X.require_no_test_access("validation") == "validation"


def test_38_no_routing_is_defined():
    assert X.T1_ROUTING_DEFINED is False
    assert P.T1_ROUTING_DEFINED is False


def test_39_no_llm_participates_in_execution():
    assert X.T1_LLM_PARTICIPATES_IN_EXECUTION is False
    assert P.T1_LLM_PARTICIPATES_IN_STATE is False


def test_40_the_spec_binder_cannot_perform_scientific_execution():
    """Structural, not asserted: the binder has no way to touch scientific state."""
    assert X.T1_SCIENTIFIC_EXECUTION_PERFORMED_BY_THIS_MODULE is False

    imported = _module_imports(X)
    assert imported <= {"__future__", "hashlib", "pathlib", "typing"}
    for forbidden in ("numpy", "torch", "scipy", "sklearn", "pandas", "cardiosentinel"):
        assert forbidden not in imported, forbidden

    called = _module_calls(X)
    for forbidden in (
        "load",
        "np_load",
        "savez",
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "rmtree",
        "next_state",
        "empirical_order_statistic",
        "candidate_policies",
        "fit",
        "predict",
        "apply_to_scores",
    ):
        assert forbidden not in called, forbidden

    source = Path(X.__file__).read_text()
    assert source.count("open(") == 1
    assert 'open(path, "rb")' in source


def test_40b_no_planned_artifact_is_created_by_this_specification():
    assert X.T1_ARTIFACTS_CREATED_BY_THIS_SPECIFICATION == ()
    run_root = REPOSITORY_ROOT / X.T1_RUN_ROOT_RELATIVE
    assert not run_root.exists(), (
        f"{run_root} exists; a canonical T1 attempt directory in any state "
        "consumes the attempt, and this specification creates none"
    )


def test_40c_the_planned_artifact_names_are_frozen():
    assert len(X.T1_PLANNED_ARTIFACTS) == 13
    assert "T1_EXPERIMENT_LOCK.json" in X.T1_PLANNED_ARTIFACTS
    assert "T1_OOF_RESULT.json" in X.T1_PLANNED_ARTIFACTS
    assert "T1_FINAL_CONFIGURATION.json" in X.T1_PLANNED_ARTIFACTS


# ---------------------------------------------------------------------------
# The comment-only repair proof (§35)
# ---------------------------------------------------------------------------


def test_the_protocol_comment_repair_changed_nothing_but_comments():
    """Comments may differ; everything else may not.

    Stronger than an AST comparison in one respect: docstrings are string
    expressions, not comment tokens, so they are inside this digest too. The
    repair may not touch one.
    """
    digest = _comment_stripped_digest(Path(P.__file__))
    assert digest == T1_PROTOCOL_COMMENT_STRIPPED_SHA256, (
        "t1_protocol.py changed outside its comments; the repair was supposed to "
        "touch comment text only"
    )


def test_the_protocol_still_parses_to_a_single_consistent_module():
    """A same-interpreter sanity check that costs nothing and catches a lot."""
    tree = ast.parse(Path(P.__file__).read_text())
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for required in (
        "validate_t1_protocol_document",
        "t1_folds",
        "candidate_policies",
        "empirical_order_statistic",
        "next_state",
        "group_reference_episodes",
        "match_runs_to_episodes",
        "policy_sort_key",
    ):
        assert required in functions, required


def test_the_repaired_section_references_resolve_against_the_frozen_document():
    """Every §N in the module names a section that exists in the document."""
    import re

    document = P.T1_PROTOCOL_PATH.read_text()
    present = {
        int(match.group(1)) for match in re.finditer(r"^## (\d+)\.", document, re.M)
    }
    assert present == set(range(0, 24)), sorted(present)

    source = Path(P.__file__).read_text()
    referenced = {int(number) for number in re.findall(r"§(\d+)", source)}
    assert referenced, "the module should still carry section references"
    unresolved = sorted(referenced - present)
    assert not unresolved, f"module references non-existent sections {unresolved}"


# ---------------------------------------------------------------------------
# The transition view versus the persisted evidence row (§33 of the request)
# ---------------------------------------------------------------------------


def test_the_evidence_row_is_a_superset_of_the_transition_view_not_a_conflict():
    """A permission list wider than the transition view is not a contradiction."""
    assert set(X.T1_INPUT_EVIDENCE_COLUMNS) >= {
        "m2g_detector_score",
        "detector_decision_d_t",
        "oof_calibrated_probability_p_t",
        "decision_error_uncertainty_u_t",
        "s4d_temporal_evidence_s_t",
    }
    # Every persisted evidence quantity is a permitted protocol row input, under
    # either the evidence naming or the transition-view naming.
    permitted = set(P.T1_ALLOWED_ROW_INPUTS) | set(P.T1Row._fields)
    for column in X.T1_INPUT_EVIDENCE_COLUMNS:
        if column in ("record_id", "channel_index", "start_sample", "subject_id"):
            continue  # physical identity and calibrator lookup, never predictive
        assert column in permitted, column
    # And nothing forbidden ever reaches an evidence store.
    for column in X.T1_EVIDENCE_STORE_FORBIDDEN_COLUMNS:
        with pytest.raises(T1ExecutionSpecError):
            X.require_evidence_column_permitted(column)
    assert X.T1_STATE_ELAPSED_CREATES_NEW_TRANSITION_CONDITION is False


def test_elapsed_time_comes_from_physical_sample_coordinates():
    assert X.T1_ELAPSED_STREAM_SECONDS_SOURCE == "physical_sample_coordinates"
    assert X.T1_ELAPSED_FROM_ROW_ORDINAL_PERMITTED is False
    assert X.T1_SAMPLING_FREQUENCY_HZ == P.T1_SAMPLING_FREQUENCY_HZ
    assert X.T1_STRIDE_SECONDS == P.T1_STRIDE_SECONDS
    assert X.T1_EMITTED_STATE_CONVENTION == "state_after_processing_current_row"


def test_physical_exposure_includes_unavailable_positions():
    assert X.T1_EXPOSURE_INCLUDES_UNAVAILABLE_POSITIONS is True
    assert X.T1_EXPOSURE_IS_PRIMARY_ONLY is False
    assert X.T1_FALSE_ONSET_NUMERATOR == "unmatched_predicted_event_runs"


def test_an_undefined_metric_is_preserved_not_zeroed():
    assert X.T1_UNDEFINED_METRIC_BECOMES_ZERO is False
    assert X.T1_UNDEFINED_COMPARISON_REQUIRES_HUMAN_REVIEW is True
    assert X.require_defined_metric("episode_f1", 0.5) == 0.5
    with pytest.raises(T1ExecutionSpecError, match="human review"):
        X.require_defined_metric("episode_f1", None)
    assert X.T1_CATEGORICAL_STATE_AUPRC_REPORTED is False


def test_failure_semantics_are_frozen():
    assert X.T1_PRE_CLAIM_REFUSAL_CONSUMES_ATTEMPT is False
    assert X.T1_POST_CLAIM_FAILURE_CONSUMES_ATTEMPT is True
    assert X.T1_FAILED_ATTEMPT_MAY_BE_DELETED_OR_REWRITTEN is False
    for field in ("stage", "current_fold", "test_state", "runtime_integrity_state"):
        assert field in X.T1_FAILURE_RECEIPT_FIELDS


def test_runtime_enforcement_points_are_frozen_and_ordered():
    assert X.T1_RUNTIME_ENFORCEMENT_POINTS[0] == "start"
    assert X.T1_RUNTIME_ENFORCEMENT_POINTS[-1] == "completion"
    assert len(X.T1_RUNTIME_ENFORCEMENT_POINTS) == 8
    assert X.T1_RUNTIME_ENFORCEMENT_MAY_BE_WEAKENED is False
    assert X.T1_PACKAGE_INSTALL_PERMITTED is False
    assert X.T1_ALTERNATE_INTERPRETER_PERMITTED is False
    assert X.T1_CANONICAL_INTERPRETER == "/home/AI_POC/venvs/tactics/bin/python"


def test_the_stage_order_is_a_total_order_without_duplicates():
    assert len(X.T1_STAGE_ORDER) == len(set(X.T1_STAGE_ORDER))
    assert X.T1_STAGE_ORDER[0] == X.STAGE_START
    assert X.T1_STAGE_ORDER[-1] == X.STAGE_COMPLETION
    with pytest.raises(T1ExecutionSpecError, match="not a frozen"):
        X.require_stage_known("invent_a_stage")


def test_the_specification_identity_names_what_a_future_artifact_binds():
    identity = X.specification_identity()
    assert identity["execution_spec_sha256"] == X.T1_EXECUTION_SPEC_SHA256
    assert identity["protocol_document_sha256"] == P.T1_PROTOCOL_SHA256
    assert identity["attempt_id"] == "t1-v1-development"
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"
    assert identity["scientific_execution_performed"] is False


def test_the_subject_authority_is_identity_only():
    assert X.T1_SUBJECT_AUTHORITY_CALLABLE == "subject_id_for_record"
    assert X.T1_SUBJECT_IDENTITY_IS_TRANSITION_FEATURE is False
    assert X.T1_SUBJECT_IDENTITY_DERIVED_FROM_LABEL is False
    assert X.T1_SUBJECT_IDENTITY_CROSS_CHECK_MUST_AGREE_EXACTLY is True
    from cardiosentinel.data import ltstdb

    assert hasattr(ltstdb, "subject_id_for_record")
