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
    with pytest.raises(V.M2EvaluationError, match="no new threshold"):
        V.window_evidence(
            [], labels=[], subject_ids=[], threshold=SC.NORMAL_EVIDENCE_THRESHOLD
        )


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
    with pytest.raises(V.M2EvaluationError, match="misaligned"):
        V.window_evidence([], labels=[1, 0], subject_ids=["a", "b"])


def test_false_alarm_join_misalignment_fails_loudly():
    with pytest.raises(V.M2EvaluationError, match="not row-aligned"):
        V.false_alarm_evidence(
            [], labels=[1], target_families=["background_negative"], subject_ids=[]
        )


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
    claimed = PS.claim_run_directory(tmp_path, "M2_sim_prepromotion", "M2-G")
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
    claimed = PS.claim_run_directory(tmp_path, "M2_partial", "M2-G")
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_STARTED
    assert status["claim_bearing_result_promoted"] is False
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    # The claim is never released, so a second attempt is refused.
    with pytest.raises(PS.M2PersistenceError, match="already claimed"):
        PS.claim_run_directory(tmp_path, "M2_partial", "M2-G")


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
        trajectory,
        stress_intervals=[
            {"family": "ischemic", "start_time": 5.0, "end_time": 25.0},
            {"family": "conduction_change", "start_time": 5.0, "end_time": 25.0},
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
