"""Canonical M2 execution harness: identities, firewalls, sentinel, persistence.

Synthetic and identity-level verification only. No canonical M2 scientific
execution occurs here, no VALIDATION or TEST partition is touched, and no
retention decision is expressed.

Tests needing the local frozen corpus/checkpoint (not committed to git -- see
`.gitignore`) are skipped when that data is absent, matching this repo's
existing convention for local-data-dependent tests.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_evaluation as V
from cardiosentinel.neural import m2_execution as X
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural import m2_persistence as PS
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural import m2_scorer as SC
from cardiosentinel.neural import runtime_sentinel as S
from cardiosentinel.neural.m2_gate_derivation import (
    DEFAULT_M1_RUN_ROOT,
    DEFAULT_STREAM_CACHE_ROOT,
)
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

LOCAL_DATA = DEFAULT_STREAM_CACHE_ROOT.exists() and DEFAULT_M1_RUN_ROOT.exists()
LOCAL_SKIP = "requires the local frozen corpus/checkpoint (gitignored)"

FROZEN_DIGEST = "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"

# Only the dedicated scientific interpreter carries the frozen package
# identity. CI builds its own environment from pyproject extras, so a digest
# mismatch there is the sentinel working correctly, not a defect: canonical
# runs are expected to execute exclusively in the frozen interpreter.
IN_FROZEN_SCIENTIFIC_RUNTIME = (
    S.observe_runtime_identity(S.EnforcementPoint.START).observed_digest
    == FROZEN_DIGEST
)


def _green_runtime() -> S.RuntimeIntegrityRecord:
    """A record whose START check has already been taken and matched."""
    record = S.RuntimeIntegrityRecord()
    S.require_runtime_identity(S.EnforcementPoint.START, record=record, detail="test")
    return record


# --------------------------------------------------------------------------
# 1-4. Frozen identities accepted; altered identities rejected
# --------------------------------------------------------------------------


def test_exact_frozen_identities_accepted():
    assert G.validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert G.validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256
    assert G.M2_RETAINED_EXPERIMENT_ID == "M1L_long_memory_v2"
    SC.assert_thresholds_are_distinct()


def test_altered_m2_protocol_digest_rejected(tmp_path):
    document = tmp_path / "protocol.md"
    document.write_text("tampered")
    with pytest.raises(G.M2GateError, match="differs from the frozen"):
        G.validate_m2_protocol(document)


def test_altered_receipt_digest_rejected(tmp_path):
    document = tmp_path / "receipt.json"
    document.write_text("{}")
    with pytest.raises(G.M2GateError, match="differs from the frozen"):
        G.validate_m2_gate_receipt(document)


@pytest.mark.skipif(not LOCAL_DATA, reason=LOCAL_SKIP)
def test_altered_m1l_checkpoint_or_lock_rejected(tmp_path):
    source = DEFAULT_M1_RUN_ROOT / "M1L_long_memory_v2"
    run_dir = tmp_path / "M1L_long_memory_v2"
    run_dir.mkdir()
    for name in ("EXPERIMENT_LOCK.json", "model_selected.pt"):
        (run_dir / name).write_bytes((source / name).read_bytes())
    # A tampered checkpoint no longer matches the lock's artifact digest.
    (run_dir / "model_selected.pt").write_bytes(b"tampered")
    with pytest.raises(Exception):
        SC.validate_retained_m1l_identity(run_dir)


# --------------------------------------------------------------------------
# 5-7. Scorer contract and threshold separation
# --------------------------------------------------------------------------


@pytest.mark.skipif(not LOCAL_DATA, reason=LOCAL_SKIP)
def test_scorer_receives_raw_representation_and_pre_update_d_long():
    scorer = SC.load_frozen_m1l_scorer(DEFAULT_M1_RUN_ROOT)
    identity = scorer.identity()
    assert identity["input_dim"] == 147
    assert "raw frozen 146-d z_t" in identity["input_contract"]
    assert "post-update d_long is never used" in identity["input_contract"]
    assert identity["retrained"] is False
    assert identity["fitted"] is False
    assert identity["threshold_selected_here"] is False


@pytest.mark.skipif(not LOCAL_DATA, reason=LOCAL_SKIP)
def test_scorer_output_parity_with_the_frozen_scoring_path():
    """The adapter reproduces the established canonical numerical path."""
    import torch

    from cardiosentinel.neural.m2_gate_derivation import M1L_INTRA_OP_THREADS
    from cardiosentinel.neural.patient_memory import (
        REPRESENTATION_DIM,
        build_deterministic_m1_head,
        m1_arm_features,
    )

    scorer = SC.load_frozen_m1l_scorer(DEFAULT_M1_RUN_ROOT)
    generator = np.random.default_rng(2026)
    block = generator.normal(size=(8, REPRESENTATION_DIM)).astype(np.float64)
    d_long = generator.normal(size=8).astype(np.float64)

    reference = build_deterministic_m1_head("M1L_long_memory_v2")
    reference.load_state_dict(
        torch.load(
            DEFAULT_M1_RUN_ROOT / "M1L_long_memory_v2" / "model_selected.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    reference.eval()
    features = m1_arm_features("M1L_long_memory_v2", block, d_long.reshape(-1, 1))
    previous = torch.get_num_threads()
    torch.set_num_threads(M1L_INTRA_OP_THREADS)
    try:
        with torch.no_grad():
            expected = (
                torch.sigmoid(reference(torch.from_numpy(features)))
                .to(torch.float64)
                .numpy()
            )
    finally:
        torch.set_num_threads(previous)
    observed = scorer.score_batch(block, d_long)
    assert np.array_equal(observed, expected)


def test_classification_and_update_thresholds_cannot_be_interchanged():
    assert SC.M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert SC.NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614
    assert SC.NORMAL_EVIDENCE_THRESHOLD < SC.M1L_CLASSIFICATION_THRESHOLD
    assert SC.NORMAL_EVIDENCE_THRESHOLD == G.NORMAL_EVIDENCE_THRESHOLD
    # The evaluation layer refuses any threshold that is not the frozen one.
    with pytest.raises(V.M2EvaluationError):
        V.require_frozen_m1l_classification_threshold(SC.NORMAL_EVIDENCE_THRESHOLD)


# --------------------------------------------------------------------------
# 8-9. Label firewall and post-replay-only annotation join
# --------------------------------------------------------------------------


def test_runtime_replay_has_no_label_or_annotation_inputs():
    firewall = X.assert_label_firewall()
    assert firewall["annotation_identifiers_absent"] is True
    assert firewall["replay_imports_evaluation"] is False
    assert set(firewall["replay_side_modules"]) == {"m2_execution", "m2_policy"}


def test_annotations_join_only_after_replay():
    """The dependency points one way: replay never imports evaluation."""
    for module in (X, P):
        imports = X._module_imports(Path(module.__file__))
        assert not any(name.endswith("m2_evaluation") for name in imports), module

    # And the evaluation module is where annotations legitimately appear.
    evaluation_source = Path(V.__file__).read_text()
    assert "target_families" in evaluation_source


def test_timeline_row_carries_no_identity_or_annotation_field():
    fields = set(P.M2TimelineRow.__dataclass_fields__)
    assert not (fields & {"subject_id", "target_family", "label", "annotation"})


# --------------------------------------------------------------------------
# 10-11. Alignment failures are loud
# --------------------------------------------------------------------------


def test_post_replay_join_misalignment_fails_loudly():
    """Identity mismatch is refused; a positional join is not even possible."""
    evidence = [_evidence(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="no evidence row"):
        V.build_evaluation_bundle("M2-G", evidence, [_annotation(start_sample=7777)])


def test_evaluation_functions_require_an_identity_joined_bundle():
    """The old positional (labels=..., subject_ids=...) API is gone."""
    import inspect

    for function in (
        V.window_evidence,
        V.false_alarm_evidence,
        V.cold_start_stratified_evidence,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & {"labels", "subject_ids", "target_families"}), function
        assert "bundle" in parameters


@pytest.mark.skipif(not LOCAL_DATA, reason=LOCAL_SKIP)
def test_unknown_record_selection_fails_loudly():
    with pytest.raises(X.M2ExecutionError, match="absent from the"):
        X.assemble_timeline_rows("train", record_ids=("s99999",))


# --------------------------------------------------------------------------
# 12-13. Arms and the partition firewall
# --------------------------------------------------------------------------


def test_only_two_arms_exist():
    assert P.M2_ARMS == ("M2-0", "M2-G") == G.M2_CORE_ARMS
    assert X.PARTITIONS_PERMITTED_HERE == ("train",)
    with pytest.raises(P.M2PolicyError):
        P.require_m2_arm("M2-GR")


def test_test_partition_is_hard_rejected():
    for spelling in ("test", "TEST", " Test "):
        with pytest.raises(X.M2ExecutionError, match="hard-rejects"):
            X.require_permitted_partition(spelling)
    assert "test" in X.FORBIDDEN_PARTITIONS


def test_validation_partition_is_not_permitted_by_this_authorization():
    with pytest.raises(X.M2ExecutionError, match="not permitted"):
        X.require_permitted_partition("validation")


def test_execution_module_does_not_import_the_sealed_test_evaluator():
    """The B4 sealed-test evaluator is unreachable from every M2 module.

    Checked over real imports rather than raw text: `sealed_test_state` is a
    legitimate provenance field whose value must stay "unopened", and a text
    scan would confuse recording that fact with reusing the evaluator.
    """
    for module in (X, PS, SC, V, P):
        imports = X._module_imports(Path(module.__file__))
        assert not any(
            name.endswith("sealed_test") or ".sealed_test." in name for name in imports
        ), module.__name__

    # The only permitted appearance is the provenance value itself.
    source = Path(X.__file__).read_text()
    for occurrence in source.split("sealed_test")[1:]:
        assert occurrence.startswith("_state"), occurrence[:40]
    assert '"sealed_test_state": "unopened"' in source


# --------------------------------------------------------------------------
# 14-18. Runtime-integrity sentinel
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not IN_FROZEN_SCIENTIFIC_RUNTIME,
    reason=(
        "only the frozen scientific interpreter carries the canonical "
        "335-package identity; CI legitimately builds its own environment, and "
        "the sentinel is SUPPOSED to report a mismatch there"
    ),
)
def test_sentinel_passes_in_the_frozen_scientific_runtime():
    check = S.observe_runtime_identity(S.EnforcementPoint.START)
    assert check.matches is True
    assert check.observed_digest == FROZEN_DIGEST
    assert check.package_count == 335


def test_sentinel_detects_a_non_frozen_runtime_in_any_environment():
    """The mechanism, checked without assuming the ambient environment.

    This must hold everywhere, including CI: an environment that is not the
    frozen scientific runtime has to be reported as a mismatch and must refuse
    a claim-bearing promotion. A canonical run is expected to execute only in
    the frozen interpreter.
    """
    record = S.RuntimeIntegrityRecord(expected_digest="f" * 64)
    check = S.observe_runtime_identity(
        S.EnforcementPoint.START, expected_digest="f" * 64
    )
    assert check.matches is False
    assert check.observed_digest != "f" * 64
    with pytest.raises(S.RuntimeIntegrityError):
        S.require_runtime_identity(S.EnforcementPoint.START, record=record)
    assert record.all_matched is False


def test_sentinel_uses_the_official_digest_recipe_not_a_second_one():
    from cardiosentinel.neural.provenance import dependency_environment

    official = dependency_environment()["installed_packages_sha256"]
    assert (
        S.observe_runtime_identity(S.EnforcementPoint.START).observed_digest == official
    )
    assert S.SENTINEL_DESIGN_DOCUMENT == "docs/RUNTIME_INTEGRITY_SENTINEL_V1.md"


def test_simulated_mismatch_at_start_refuses_execution():
    record = S.RuntimeIntegrityRecord(expected_digest="0" * 64)
    with pytest.raises(S.RuntimeIntegrityError, match="promotion is refused"):
        S.require_runtime_identity(S.EnforcementPoint.START, record=record)
    assert record.all_matched is False
    assert record.first_mismatch().enforcement_point == "start"


def test_simulated_mismatch_before_persistence_refuses_promotion(tmp_path):
    claimed = PS.claim_run_directory(
        tmp_path, "M2_sim_prepromotion", "M2-G", runtime=_green_runtime()
    )
    record = S.RuntimeIntegrityRecord(expected_digest="1" * 64)
    check = S.observe_runtime_identity(
        S.EnforcementPoint.PRE_PROMOTION, expected_digest="1" * 64
    )
    record.record(check)
    assert check.matches is False
    PS.record_failure(claimed, "simulated pre-promotion mismatch", runtime_check=check)
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["claim_bearing_result_promoted"] is False
    failure = json.loads((claimed.run_dir / PS.RUNTIME_FAILURE_NAME).read_text())
    assert failure["claim_bearing"] is False
    assert failure["promotion_refused"] is True
    assert failure["automatic_environment_repair_performed"] is False
    # No claim-bearing artifact exists.
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()


def test_end_mismatch_marks_the_attempt_non_canonical():
    """A COMPLETION mismatch is recorded, never silently blessed."""
    record = S.RuntimeIntegrityRecord(expected_digest="2" * 64)
    check = S.observe_runtime_identity(
        S.EnforcementPoint.COMPLETION, expected_digest="2" * 64
    )
    record.record(check)
    assert check.matches is False
    payload = record.as_dict()
    assert payload["all_observations_matched"] is False
    assert payload["completion_digest"] == check.observed_digest


def test_no_automatic_environment_repair_occurs():
    source = Path(S.__file__).read_text()
    for banned in ("pip install", "subprocess", "check_call", "install("):
        assert banned not in source, banned
    assert (
        S.RuntimeIntegrityRecord().as_dict()["automatic_environment_repair_performed"]
        is False
    )


# --------------------------------------------------------------------------
# 19-21. Persistence safety and provenance
# --------------------------------------------------------------------------


def test_partial_run_cannot_masquerade_as_complete(tmp_path):
    claimed = PS.claim_run_directory(
        tmp_path, "M2_partial", "M2-G", runtime=_green_runtime()
    )
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_STARTED
    assert status["claim_bearing_result_promoted"] is False
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    # The claim is never released, so a second attempt is refused.
    with pytest.raises(PS.M2PersistenceError, match="already claimed"):
        PS.claim_run_directory(tmp_path, "M2_partial", "M2-G", runtime=_green_runtime())


def test_forbidden_partition_audit_blocks_promotion():
    with pytest.raises(Exception):
        PS.audit_forbidden_partitions({"partition_accessed": "test"})
    with pytest.raises(PS.M2PersistenceError, match="validation_accessed"):
        PS.audit_forbidden_partitions(
            {
                "partition_accessed": "train",
                "validation_accessed": True,
                "test_accessed": False,
                "sealed_test_state": "unopened",
            }
        )
    with pytest.raises(PS.M2PersistenceError, match="sealed test"):
        PS.audit_forbidden_partitions(
            {
                "partition_accessed": "train",
                "validation_accessed": False,
                "test_accessed": False,
                "sealed_test_state": "opened",
            }
        )


def test_no_automatic_m2_retention_decision_occurs():
    suite = PS.build_suite_result(
        suite_id="M2_suite_test",
        arm_results={"M2-0": {"arm": "M2-0"}, "M2-G": {"arm": "M2-G"}},
    )
    assert suite["memory_selection_performed"] is False
    assert suite["memory_selected"] is None
    assert suite["automatic_arm_preference_applied"] is False
    assert suite["human_review_required"] is True
    with pytest.raises(PS.M2PersistenceError, match="binds exactly"):
        PS.build_suite_result(suite_id="bad", arm_results={"M2-G": {}})


def test_result_schema_binds_all_required_provenance():
    required = set(PS.REQUIRED_PROVENANCE_FIELDS)
    for field in (
        "m2_protocol_sha256",
        "m2_gate_receipt_sha256",
        "retained_m1l_lock_sha256",
        "retained_m1l_checkpoint_sha256",
        "p1b_lock_sha256",
        "b4b_checkpoint_sha256",
        "distance_standardizer_sha256",
        "split_sha256",
        "ordered_chronology_sha256",
        "runtime_dependency_digest_start",
        "runtime_dependency_digest_pre_promotion",
        "runtime_dependency_digest_end",
        "partition_accessed",
        "validation_accessed",
        "test_accessed",
        "sealed_test_state",
        "artifact_sha256",
    ):
        assert field in required, field


# --------------------------------------------------------------------------
# 22-24. Drift direction, rollback, and scope
# --------------------------------------------------------------------------


def test_prototype_drift_annotations_cannot_affect_replay():
    from cardiosentinel.neural.m2_evidence import PrototypeTrajectory
    from cardiosentinel.neural.patient_memory import REPRESENTATION_DIM

    prototypes = np.zeros((3, REPRESENTATION_DIM))
    prototypes[2] = 1.0
    trajectory = PrototypeTrajectory(
        times=np.asarray([0.0, 10.0, 20.0]), prototypes=prototypes
    )
    before = trajectory.prototypes.copy()
    result = V.contamination_evidence(
        {("s00001", 0): trajectory},
        stress_intervals=[
            V.M2StressInterval("s00001", 0, "ischemic", 5.0, 25.0),
            V.M2StressInterval("s00001", 0, "conduction_change", 5.0, 25.0),
        ],
    )
    # The trajectory is consumed, never mutated.
    assert np.array_equal(trajectory.prototypes, before)
    assert result["trajectory_produced_label_blind"] is True
    assert result["annotations_applied_after_replay"] is True
    assert result["recovery_threshold_defined"] is False
    conduction = [i for i in result["intervals"] if i["family"] == "conduction_change"]
    assert conduction[0]["evidence_status"] == "exploratory_descriptive"


def test_no_rollback_path_exists_in_the_harness():
    for module in (X, V, PS, SC, S):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        names = {
            getattr(node, "id", None) or getattr(node, "attr", None)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        for banned in ("rollback_prototype", "restore_snapshot", "oracle_correction"):
            assert banned not in names, (module.__name__, banned)
    assert G.M2_ROLLBACK_IN_CORE is False


def test_no_u1_u2_t1_t2_code_enters_this_work():
    for module in (X, V, PS, SC, S):
        source = Path(module.__file__).read_text().lower()
        for banned in (
            "conformal",
            "temperature_scaling",
            "isotonic",
            "watch_state",
            "event_state",
            "recovery_state",
        ):
            assert banned not in source, (module.__name__, banned)


def test_uncertainty_and_episode_packages_remain_placeholders():
    for package in ("uncertainty", "episodes"):
        path = REPOSITORY_ROOT / "src" / "cardiosentinel" / package / "__init__.py"
        assert len(path.read_text().strip().splitlines()) <= 2, package


# --------------------------------------------------------------------------
# Real-artifact integration (frozen scientific runtime + local data only)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (LOCAL_DATA and IN_FROZEN_SCIENTIFIC_RUNTIME),
    reason=(
        "the smoke exercises the sentinel, which correctly refuses to run "
        "outside the frozen scientific runtime, and needs the gitignored "
        "frozen corpus/checkpoint"
    ),
)
def test_bounded_train_smoke_is_non_claim_bearing():
    report = X.train_integration_smoke(("s20011",), max_rows_per_stream=8)
    assert report["artifact_class"] == "NON_CLAIM_BEARING_TRAIN_INTEGRATION_SMOKE"
    assert report["claim_bearing"] is False
    assert report["scientific_evidence"] is False
    assert report["metrics_computed"] is False
    assert report["arm_comparison_performed"] is False
    assert report["retention_decision_performed"] is False
    assert report["partition_accessed"] == "train"
    assert report["validation_accessed"] is False
    assert report["test_accessed"] is False
    assert set(report["arms"]) == {"M2-0", "M2-G"}
    assert report["runtime_identity_checks"]["all_observations_matched"] is True


# --------------------------------------------------------------------------
# Human execution-harness review corrections:
#   A) the frozen classification threshold is enforced on EVERY thresholded path
#   B) the post-replay annotation join is keyed by immutable row identity
# --------------------------------------------------------------------------

from cardiosentinel.neural.m2_policy import (  # noqa: E402
    M2GateDecision,
    M2RowEvidence,
)

ALT_THRESHOLD = 0.5


def _decision(score: float | None) -> M2GateDecision:
    return M2GateDecision(
        arm="M2-G",
        g1_available=True,
        g2_finite_representation=True,
        g3_finite_sample_precondition=True,
        g3_feature_results={},
        g3_sqi_admissible=True,
        g4_normal_evidence=True,
        g5_not_in_refractory=True,
        g6_morphology_computable=True,
        admitted=True,
        score=score,
        refractory_until_before=float("-inf"),
    )


def _evidence(record_id="s00001", channel_index=0, start_sample=0, score=0.1):
    return M2RowEvidence(
        record_id=record_id,
        channel_index=channel_index,
        start_sample=start_sample,
        available_time=(start_sample + 2500) / 250.0,
        observation_state=1,
        arm="M2-G",
        decision=_decision(score),
        d_long=0.5,
        morphology_valid=1.0,
        update_admitted=True,
        refractory_rearmed_after_decision=False,
        refractory_until_after=float("-inf"),
        past_observed_count_before=0,
        past_update_count_before=0,
        past_update_count_after=1,
        time_since_last_admitted_update=None,
    )


def _annotation(
    record_id="s00001",
    channel_index=0,
    start_sample=0,
    label=0,
    family="background_negative",
    subject="subj-1",
    bin_name="over_60_minutes",
):
    return V.M2AnnotationRow(
        record_id=record_id,
        channel_index=channel_index,
        start_sample=start_sample,
        label=label,
        target_family=family,
        subject_id=subject,
        cold_start_bin=bin_name,
    )


def _paired_bundle(count=4):
    """A minimal well-formed bundle with both classes and two subjects."""
    evidence, annotations = [], []
    for index in range(count):
        start = index * 1250
        evidence.append(_evidence(start_sample=start, score=0.1 + 0.2 * index))
        annotations.append(
            _annotation(
                start_sample=start,
                label=index % 2,
                family="ischemic_positive" if index % 2 else "background_negative",
                subject=f"subj-{index % 2}",
            )
        )
    return V.build_evaluation_bundle("M2-G", evidence, annotations)


# --- 1-2. threshold enforcement on every thresholded path -----------------


def test_every_thresholded_metric_rejects_a_non_frozen_threshold():
    bundle = _paired_bundle()
    for function in (
        V.window_evidence,
        V.false_alarm_evidence,
        V.cold_start_stratified_evidence,
    ):
        with pytest.raises(V.M2EvaluationError, match="frozen retained M1L"):
            function(bundle, threshold=ALT_THRESHOLD)


def test_normal_evidence_threshold_is_rejected_as_a_classification_threshold():
    bundle = _paired_bundle()
    for function in (
        V.window_evidence,
        V.false_alarm_evidence,
        V.cold_start_stratified_evidence,
    ):
        with pytest.raises(V.M2EvaluationError):
            function(bundle, threshold=SC.NORMAL_EVIDENCE_THRESHOLD)


def test_the_frozen_classification_threshold_is_accepted():
    bundle = _paired_bundle()
    assert (
        V.require_frozen_m1l_classification_threshold(SC.M1L_CLASSIFICATION_THRESHOLD)
        == SC.M1L_CLASSIFICATION_THRESHOLD
    )
    payload = V.cold_start_stratified_evidence(bundle)
    assert payload["threshold"] == SC.M1L_CLASSIFICATION_THRESHOLD
    assert payload["threshold_selected_here"] is False


# --- 3-7. identity-keyed join -------------------------------------------


def test_equal_length_but_wrong_identities_are_rejected():
    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    wrong = [_annotation(start_sample=9999), _annotation(start_sample=8888)]
    assert len(evidence) == len(wrong)
    with pytest.raises(V.M2EvaluationError, match="no evidence row"):
        V.build_evaluation_bundle("M2-G", evidence, wrong)


def test_permuted_annotation_ordering_is_realigned_by_identity_not_position():
    evidence = [
        _evidence(start_sample=0, score=0.10),
        _evidence(start_sample=1250, score=0.90),
    ]
    ordered = [
        _annotation(start_sample=0, label=0, subject="a"),
        _annotation(start_sample=1250, label=1, subject="b"),
    ]
    bundle_ordered = V.build_evaluation_bundle("M2-G", evidence, ordered)
    bundle_permuted = V.build_evaluation_bundle(
        "M2-G", evidence, list(reversed(ordered))
    )
    # Identity, not order, decides the pairing.
    assert bundle_ordered.keys == bundle_permuted.keys
    assert np.array_equal(bundle_ordered.labels, bundle_permuted.labels)
    assert np.array_equal(bundle_ordered.scores, bundle_permuted.scores)
    # And the score/label pairing is the correct one.
    assert bundle_ordered.scores[0] == 0.10
    assert bundle_ordered.labels[0] == 0
    assert bundle_ordered.scores[1] == 0.90
    assert bundle_ordered.labels[1] == 1


def test_duplicate_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="Duplicate annotation"):
        V.build_evaluation_bundle(
            "M2-G", evidence, [_annotation(start_sample=0), _annotation(start_sample=0)]
        )


def test_duplicate_evidence_identities_are_rejected():
    duplicated = [_evidence(start_sample=0), _evidence(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="Duplicate evidence"):
        V.build_evaluation_bundle("M2-G", duplicated, [_annotation(start_sample=0)])


def test_missing_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    with pytest.raises(V.M2EvaluationError, match="carry no annotation"):
        V.build_evaluation_bundle("M2-G", evidence, [_annotation(start_sample=0)])


def test_extra_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0)]
    extra = [_annotation(start_sample=0), _annotation(start_sample=1250)]
    with pytest.raises(V.M2EvaluationError, match="no evidence row"):
        V.build_evaluation_bundle("M2-G", evidence, extra)


def test_subset_population_requires_an_explicit_opt_in():
    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    annotations = [_annotation(start_sample=0)]
    bundle = V.build_evaluation_bundle(
        "M2-G", evidence, annotations, require_full_population=False
    )
    assert len(bundle.keys) == 1


def test_evaluation_key_corresponds_to_the_frozen_stable_id():
    row = _evidence(record_id="s20011", channel_index=1, start_sample=2500)
    key = V.evaluation_key(row)
    assert key == ("s20011", 1, 2500)
    assert V.stable_id_for_key(key) == "ltstdb:s20011:1:2500:5000"


# --- 8-9. subject and family remain evaluation-only ----------------------


def test_subject_and_family_remain_evaluation_only():
    replay_fields = set(P.M2TimelineRow.__dataclass_fields__)
    assert "subject_id" not in replay_fields
    assert "target_family" not in replay_fields
    evaluation_fields = set(V.M2AnnotationRow.__dataclass_fields__)
    assert {"subject_id", "target_family"} <= evaluation_fields
    # Neither reaches the gate.
    import inspect

    assert not (
        set(inspect.signature(P.evaluate_gate).parameters)
        & {"subject_id", "target_family"}
    )


# --- 10-11. stress intervals bound to their stream -----------------------


def _trajectory(times, values):
    from cardiosentinel.neural.m2_evidence import PrototypeTrajectory
    from cardiosentinel.neural.patient_memory import REPRESENTATION_DIM

    prototypes = np.zeros((len(times), REPRESENTATION_DIM))
    for index, value in enumerate(values):
        prototypes[index] = value
    return PrototypeTrajectory(
        times=np.asarray(times, dtype=np.float64), prototypes=prototypes
    )


def test_stress_interval_is_bound_to_the_correct_stream_trajectory():
    trajectories = {
        ("s00001", 0): _trajectory([0.0, 10.0, 20.0], [0.0, 0.0, 1.0]),
        ("s00001", 1): _trajectory([0.0, 10.0, 20.0], [0.0, 0.0, 5.0]),
    }
    result = V.contamination_evidence(
        trajectories,
        stress_intervals=[
            V.M2StressInterval("s00001", 0, "ischemic", 5.0, 25.0),
            V.M2StressInterval("s00001", 1, "ischemic", 5.0, 25.0),
        ],
    )
    assert result["intervals_bound_to_stream_identity"] is True
    channel_0 = [i for i in result["intervals"] if i["channel_index"] == 0][0]
    channel_1 = [i for i in result["intervals"] if i["channel_index"] == 1][0]
    # Each interval saw only its own stream's drift.
    assert channel_0["peak_drift_during_stress"] == 1.0
    assert channel_1["peak_drift_during_stress"] == 5.0


def test_stress_interval_cannot_cross_apply_to_another_stream():
    trajectories = {("s00001", 0): _trajectory([0.0, 10.0], [0.0, 1.0])}
    with pytest.raises(V.M2EvaluationError, match="never be applied to another"):
        V.contamination_evidence(
            trajectories,
            stress_intervals=[V.M2StressInterval("s00002", 0, "ischemic", 5.0, 25.0)],
        )


# --- 12. scored/unscored parity with frozen M1 ---------------------------


def test_unscored_row_is_refused_exactly_as_frozen_m1_refuses_it():
    """Parity with `m1_experiment.require_available_rows` governance semantics.

    Frozen M1 refuses a score-bearing population containing a physically
    unavailable row rather than dropping it. M2 reproduces that verbatim
    instead of silently excluding, so the denominator is never altered
    automatically.
    """
    import inspect

    from cardiosentinel.neural.m1_experiment import require_available_rows

    frozen_doc = inspect.getdoc(require_available_rows) or ""
    assert "never silently altered" in frozen_doc

    evidence = [_evidence(start_sample=0, score=None)]
    with pytest.raises(V.M2EvaluationError) as caught:
        V.build_evaluation_bundle("M2-G", evidence, [_annotation(start_sample=0)])
    message = str(caught.value)
    assert "STOP FOR HUMAN REVIEW" in message
    assert "never dropped from a metric" in message
    assert "no denominator is altered automatically" in message


# --- 13-14. smoke bookkeeping and evaluated-population provenance ---------


def test_bounded_smoke_keeps_stable_ids_aligned_with_its_bounded_rows():
    """Interleaved streams mean the retained rows are not a prefix."""
    source = Path(X.__file__).read_text()
    assert "bundle.stable_ids[: len(bounded)]" not in source
    assert "zip(bundle.rows, bundle.stable_ids, strict=True)" in source


@pytest.mark.skipif(
    not (LOCAL_DATA and IN_FROZEN_SCIENTIFIC_RUNTIME),
    reason="needs the frozen runtime and gitignored corpus",
)
def test_bounded_smoke_stable_ids_match_its_rows_exactly():
    from cardiosentinel.neural.m2_execution import assemble_timeline_rows

    bundle = assemble_timeline_rows("train", record_ids=("s20011",))
    bounded, bounded_ids, per_stream = [], [], {}
    for row, stable_id in zip(bundle.rows, bundle.stable_ids, strict=True):
        taken = per_stream.get(row.stream_key, 0)
        if taken >= 8:
            continue
        per_stream[row.stream_key] = taken + 1
        bounded.append(row)
        bounded_ids.append(stable_id)
    # Every retained ID is the one its own row derives.
    for row, stable_id in zip(bounded, bounded_ids, strict=True):
        assert (
            V.stable_id_for_key((row.record_id, row.channel_index, row.start_sample))
            == stable_id
        )
    # And a naive prefix slice would have been wrong for interleaved streams.
    assert bounded_ids != list(bundle.stable_ids[: len(bounded)])


def test_result_provenance_binds_the_evaluated_population_identity():
    assert "evaluated_population_identity" in PS.REQUIRED_PROVENANCE_FIELDS
    bundle = _paired_bundle()
    identity = bundle.population_identity()
    assert identity["evaluated_rows"] == 4
    assert len(identity["evaluated_ordered_stable_id_sha256"]) == 64
    assert identity["positional_join_used"] is False
    # A different evaluated population yields a different identity.
    other = V.build_evaluation_bundle(
        "M2-G", [_evidence(start_sample=0)], [_annotation(start_sample=0)]
    )
    assert (
        other.population_identity()["evaluated_ordered_stable_id_sha256"]
        != identity["evaluated_ordered_stable_id_sha256"]
    )


# --- 20. still no scientific result --------------------------------------


def test_no_scientific_m2_result_is_generated_by_this_module():
    bundle = _paired_bundle()
    payload = V.arm_evaluation("M2-G", list(bundle.evidence))
    assert payload["window_evidence"] is None
    assert payload["false_alarm_evidence"] is None
    assert payload["contamination_evidence"] is None
    assert payload["label_joined_sections_populated"] is False
    assert payload["validation_accessed"] is False
    assert payload["test_accessed"] is False


# --------------------------------------------------------------------------
# Human final persistence/governance review corrections:
#   1) PRE_PROMOTION check at the arm claim-directory boundary
#   2) COMPLETION observed BEFORE the canonical artifact is finalized
#   3) the runtime block must be COMPLETE and GREEN, not merely present
#   4) evaluated_population_identity is mandatory for claim-bearing evaluation
#   5) headline metrics refuse a deliberately subsetted bundle
# --------------------------------------------------------------------------


def _execution_identity():
    return {
        "partition_accessed": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "scorer_identity": {
            "retained_lock_sha256": SC.RETAINED_M1L_LOCK_SHA256,
            "classification_threshold": SC.M1L_CLASSIFICATION_THRESHOLD,
            "memory_admission_threshold": SC.NORMAL_EVIDENCE_THRESHOLD,
            "classification_threshold_used_for_memory_admission": False,
        },
    }


def _claim_bearing_result(arm="M2-G", population=None):
    return {
        "arm": arm,
        "scientific_computation_completed": True,
        "m2_protocol_sha256": G.M2_PROTOCOL_SHA256,
        "m2_gate_receipt_sha256": G.M2_GATE_RECEIPT_SHA256,
        "evaluated_population_identity": population,
        "memory_selection_performed": False,
        "memory_selected": None,
        "rollback": False,
    }


# --- 1-3. claim directory sentinel boundary ------------------------------


def test_claim_directory_requires_a_successful_start_record(tmp_path):
    empty = S.RuntimeIntegrityRecord()
    with pytest.raises(PS.M2PersistenceError, match="START runtime check"):
        PS.claim_run_directory(tmp_path, "M2_no_start", "M2-G", runtime=empty)
    assert not (tmp_path / "M2_no_start").exists()


def test_claim_directory_performs_a_pre_promotion_check_before_creation(tmp_path):
    runtime = _green_runtime()
    PS.claim_run_directory(tmp_path, "M2_claim_check", "M2-G", runtime=runtime)
    points = [check.enforcement_point for check in runtime.checks]
    assert points == ["start", "pre_promotion"]
    details = [check.detail for check in runtime.checks]
    assert PS.CLAIM_DIRECTORY_PROMOTION_DETAIL in details


def test_claim_time_runtime_mismatch_creates_no_canonical_directory(tmp_path):
    """A mismatch at the claim boundary means the directory never exists."""
    runtime = S.RuntimeIntegrityRecord(expected_digest="c" * 64)
    # Force a matching START so only the claim-time check can fail.
    runtime.record(
        S.RuntimeCheck(
            enforcement_point="start",
            observed_digest="c" * 64,
            expected_digest="c" * 64,
            matches=True,
            package_count=0,
            observed_at="2026-01-01T00:00:00Z",
        )
    )
    with pytest.raises(S.RuntimeIntegrityError):
        PS.claim_run_directory(tmp_path, "M2_claim_mismatch", "M2-G", runtime=runtime)
    assert not (tmp_path / "M2_claim_mismatch").exists()


# --- 4-7. completion binding and runtime-block completeness ---------------


def test_canonical_complete_result_cannot_carry_a_none_completion_digest():
    runtime = _green_runtime()
    runtime.record(
        S.observe_runtime_identity(S.EnforcementPoint.PRE_PROMOTION, detail="x")
    )
    # No COMPLETION observation has been taken yet.
    assert runtime.digest_at(S.EnforcementPoint.COMPLETION) is None
    with pytest.raises(PS.M2PersistenceError, match="completion"):
        PS.validate_complete_runtime_identity(runtime)


def test_completion_mismatch_prevents_canonical_complete_promotion(tmp_path):
    runtime = _green_runtime()
    claimed = PS.claim_run_directory(tmp_path, "M2_end_bad", "M2-G", runtime=runtime)
    # Point the record at an impossible digest so COMPLETION cannot match.
    runtime.expected_digest = "d" * 64
    with pytest.raises(S.RuntimeIntegrityError, match="COMPLETION"):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=_claim_bearing_result(),
            execution_identity=_execution_identity(),
            runtime=runtime,
            requires_evaluation=False,
        )
    # Nothing canonical exists and the run is not COMPLETE.
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["claim_bearing_result_promoted"] is False
    failure = json.loads((claimed.run_dir / PS.RUNTIME_FAILURE_NAME).read_text())
    assert failure["canonical_promotion_invalidated"] is True
    assert failure["staged_result_retained_as_forensic_material"] is True
    assert failure["claim_bearing"] is False
    # Forensic evidence is retained, not deleted.
    assert (claimed.staging_dir / PS.ARM_RESULT_NAME).exists()


def test_all_three_enforcement_points_appear_in_the_finalized_block(tmp_path):
    runtime = _green_runtime()
    claimed = PS.claim_run_directory(tmp_path, "M2_full_block", "M2-G", runtime=runtime)
    population = _paired_bundle().population_identity()
    status = PS.finalize_and_promote_arm_result(
        claimed,
        result=_claim_bearing_result(population=population),
        execution_identity=_execution_identity(),
        runtime=runtime,
        requires_evaluation=True,
    )
    assert status["status"] == PS.STATUS_COMPLETE
    assert status["canonical"] is True

    promoted = json.loads((claimed.run_dir / PS.ARM_RESULT_NAME).read_text())
    block = promoted["runtime_identity_checks"]
    points = {check["enforcement_point"] for check in block["checks"]}
    assert points == {"start", "pre_promotion", "completion"}
    assert block["all_observations_matched"] is True
    assert promoted["runtime_dependency_digest_start"] == FROZEN_DIGEST
    assert promoted["runtime_dependency_digest_pre_promotion"] == FROZEN_DIGEST
    # The genuine, observed completion digest is inside the promoted artifact.
    assert promoted["runtime_dependency_digest_end"] == FROZEN_DIGEST
    # And the recorded hash is the hash of exactly those promoted bytes.
    from cardiosentinel.data.provenance import sha256_file

    assert status["artifact_sha256"][PS.ARM_RESULT_NAME] == sha256_file(
        claimed.run_dir / PS.ARM_RESULT_NAME
    )


def test_all_observations_matched_is_required_for_canonical_evidence():
    runtime = _green_runtime()
    runtime.record(
        S.RuntimeCheck(
            enforcement_point="pre_promotion",
            observed_digest="e" * 64,
            expected_digest=FROZEN_DIGEST,
            matches=False,
            package_count=0,
            observed_at="2026-01-01T00:00:00Z",
        )
    )
    runtime.record(
        S.observe_runtime_identity(S.EnforcementPoint.COMPLETION, detail="x")
    )
    with pytest.raises(PS.M2PersistenceError, match="every runtime observation"):
        PS.validate_complete_runtime_identity(runtime)


# --- 8-10. evaluated population identity ---------------------------------


def test_none_population_identity_is_rejected_for_claim_bearing_evaluation():
    for empty in (None, {}):
        with pytest.raises(
            PS.M2PersistenceError, match="evaluated_population_identity"
        ):
            PS.validate_evaluated_population_identity(empty)


def test_malformed_population_identity_is_rejected():
    valid = _paired_bundle().population_identity()
    for mutation in (
        {"evaluated_rows": 0},
        {"evaluated_rows": -1},
        {"evaluated_ordered_stable_id_sha256": "short"},
        {"evaluated_ordered_stable_id_sha256": "z" * 64},
        {"positional_join_used": True},
        {"identity_key": ""},
    ):
        broken = {**valid, **mutation}
        with pytest.raises(PS.M2PersistenceError):
            PS.validate_evaluated_population_identity(broken)


def test_valid_full_population_identity_is_accepted():
    identity = _paired_bundle().population_identity()
    validated = PS.validate_evaluated_population_identity(identity)
    assert validated["evaluated_rows"] == 4
    assert validated["positional_join_used"] is False
    assert validated["population_scope"] == V.POPULATION_SCOPE_FULL


def test_claim_bearing_result_requires_population_identity_when_evaluated(tmp_path):
    runtime = _green_runtime()
    claimed = PS.claim_run_directory(tmp_path, "M2_no_pop", "M2-G", runtime=runtime)
    with pytest.raises(PS.M2PersistenceError, match="evaluated_population_identity"):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=_claim_bearing_result(population=None),
            execution_identity=_execution_identity(),
            runtime=runtime,
            requires_evaluation=True,
        )
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()


# --- 11-14. headline metrics require full-population bundles --------------


def _subset_bundle():
    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    return V.build_evaluation_bundle(
        "M2-G", evidence, [_annotation(start_sample=0)], require_full_population=False
    )


@pytest.mark.parametrize(
    "function",
    [V.window_evidence, V.false_alarm_evidence, V.cold_start_stratified_evidence],
)
def test_headline_metrics_refuse_a_subsetted_bundle(function):
    subset = _subset_bundle()
    assert subset.population_scope == V.POPULATION_SCOPE_SUPPORTING_SUBSET
    assert subset.is_full_population is False
    with pytest.raises(V.M2EvaluationError, match="full-population"):
        function(subset)


def test_full_population_bundles_continue_to_work():
    bundle = _paired_bundle()
    assert bundle.is_full_population is True
    payload = V.cold_start_stratified_evidence(bundle)
    assert payload["population_identity"]["population_scope"] == (
        V.POPULATION_SCOPE_FULL
    )


# --- 15. stress evidence stays separate and stream-bound ------------------


def test_stress_interval_evidence_remains_separate_from_population_scope():
    """Interval selection is supporting evidence, not a headline population."""
    trajectories = {("s00001", 0): _trajectory([0.0, 10.0, 20.0], [0.0, 0.0, 1.0])}
    result = V.contamination_evidence(
        trajectories,
        stress_intervals=[V.M2StressInterval("s00001", 0, "ischemic", 5.0, 25.0)],
    )
    # It needs no evaluation bundle at all and imposes no population scope.
    assert result["intervals_bound_to_stream_identity"] is True
    assert "population_scope" not in result
    assert result["intervals"][0]["record_id"] == "s00001"


# --- 18-20. unchanged invariants ------------------------------------------


def test_sentinel_digest_recipe_remains_unchanged():
    from cardiosentinel.neural.provenance import dependency_environment

    assert (
        S.observe_runtime_identity(S.EnforcementPoint.START).observed_digest
        == dependency_environment()["installed_packages_sha256"]
    )
    assert (
        S.SENTINEL_DESIGN_SHA256
        == "cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075"
    )


def test_no_m1_scientific_file_changed_by_this_work():
    """The M1 stack is untouched: its frozen documents still validate."""
    from cardiosentinel.data.provenance import sha256_file

    assert (
        sha256_file(REPOSITORY_ROOT / "docs" / "M1_DUAL_MEMORY_PROTOCOL_V2.md")
        == "31a81358870cd23c2258cf4f307ab8c4dc7bf245bc4bf18a4d1f48fe2aada39c"
    )
    assert (
        sha256_file(REPOSITORY_ROOT / "docs" / "M1_MEMORY_RETENTION_DECISION_V1.md")
        == "a3685fc0f8ff1fa0dce2bf9954bb28a925787070c021f3e80ca5716a4fa5f0ed"
    )


def test_no_canonical_m2_scientific_execution_occurs(tmp_path):
    """The validator gates promotion; it computes no metric and selects no arm."""
    import inspect

    tree = ast.parse(inspect.getsource(PS.validate_claim_bearing_arm_result).lstrip())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for banned in (
        "binary_metrics",
        "p1_validation_evidence",
        "p1_challenge_evidence",
        "average_precision_score",
        "roc_auc_score",
        "argmax",
    ):
        assert banned not in called, banned
    suite = PS.build_suite_result(
        suite_id="S", arm_results={"M2-0": {"arm": "M2-0"}, "M2-G": {"arm": "M2-G"}}
    )
    assert suite["memory_selection_performed"] is False
    assert suite["memory_selected"] is None
