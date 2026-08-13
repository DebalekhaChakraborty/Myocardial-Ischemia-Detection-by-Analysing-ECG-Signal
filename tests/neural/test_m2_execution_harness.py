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

from cardiosentinel.neural import m2_development_run as RUN
from cardiosentinel.neural import m2_evaluation as V
from cardiosentinel.neural import m2_execution as X
from cardiosentinel.neural import m2_gate as G
from cardiosentinel.neural import m2_persistence as PS
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural import m2_populations as PP
from cardiosentinel.neural import m2_scorer as SC
from cardiosentinel.neural import m2_stress_intervals as SI
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


FROZEN_RUNTIME_ONLY = pytest.mark.skipif(
    not IN_FROZEN_SCIENTIFIC_RUNTIME,
    reason=(
        "asserts the frozen scientific identity; canonical COMPLETE evidence "
        "requires the frozen runtime by design, and CI legitimately builds its "
        "own environment"
    ),
)


def _synthetic_frozen_check(point: str, detail: str = "test") -> S.RuntimeCheck:
    """A TEST-ONLY observation asserting the frozen identity.

    Production never constructs these: `observe_runtime_identity` always reads
    the real environment. Tests use them so persistence MECHANICS can be
    exercised outside the frozen scientific runtime (CI included) WITHOUT
    weakening the production invariant that a canonical claim requires the
    frozen digest.
    """
    return S.RuntimeCheck(
        enforcement_point=point,
        observed_digest=FROZEN_DIGEST,
        expected_digest=FROZEN_DIGEST,
        matches=True,
        package_count=335,
        observed_at="2026-01-01T00:00:00Z",
        detail=detail,
    )


def _frozen_runtime_record() -> S.RuntimeIntegrityRecord:
    """A record that looks, to production code, like the frozen runtime.

    Built from synthetic frozen observations rather than by relaxing anything:
    `expected_digest` really is the frozen identity, and the recorded START
    really did expect/observe/match it.
    """
    record = S.RuntimeIntegrityRecord()
    record.record(_synthetic_frozen_check(S.EnforcementPoint.START.value))
    return record


@pytest.fixture()
def frozen_runtime(monkeypatch):
    """Drive the whole production path with synthetic frozen observations.

    `observe_runtime_identity` is monkeypatched -- a clearly TEST-ONLY seam --
    so `claim_run_directory` and `finalize_and_promote_arm_result` run their
    real logic unchanged. No production invariant is relaxed.
    """

    def fake_observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        return _synthetic_frozen_check(
            S.EnforcementPoint(point).value, detail or "test"
        )

    monkeypatch.setattr(PS, "observe_runtime_identity", fake_observe)
    # Canonical evidence requires a clean checkout (the M1/P1 convention). A
    # development tree is usually dirty, so provenance is pinned here too --
    # again a TEST-ONLY seam, never a relaxation: production still reads the
    # real checkout, and a dirty state is still rejected (proved separately).
    monkeypatch.setattr(
        PS,
        "git_provenance",
        lambda _root: {"git_sha": "0" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(
        PS,
        "require_runtime_identity",
        lambda point, *, record=None, detail=None: (
            record.record(
                _synthetic_frozen_check(
                    S.EnforcementPoint(point).value, detail or "test"
                )
            )
            if record is not None
            else _synthetic_frozen_check(S.EnforcementPoint(point).value)
        ),
    )
    return _frozen_runtime_record()


def _green_runtime() -> S.RuntimeIntegrityRecord:
    """Deprecated alias retained for the ambient-runtime mechanism tests."""
    return _frozen_runtime_record()


def _canonical_population_token(evidence):
    """A canonical authority token for synthetic evidence.

    Real runs obtain this from `M2InputBundle.canonical_input_population_
    identity()`, which proves the bundle against the frozen manifest. Tests
    construct an equivalent token for synthetic rows; the point under test is
    that evaluation accepts ONLY a token, never a caller-computed digest.
    """
    from cardiosentinel.neural.m2_evaluation import _observed_population_digest

    keys = [V.evaluation_key(row) for row in evidence]
    return X.M2ReplayPopulation(
        partition="train",
        row_count=len(keys),
        ordered_stable_id_sha256=_observed_population_digest(keys),
        stream_cache_sha256="c" * 64,
    )


def _population_identities():
    """The four distinct population identities a canonical lock must bind."""
    return {
        "replay_population_identity": _replay_identity(),
        "primary_evaluation_population_identity": _primary_identity(),
        "challenge_evaluation_population_identity": _challenge_identity(),
        "stress_interval_selection_identity": _stress_identity(),
    }


def _primary_population_from_evidence(evidence):
    """A PRIMARY authority token matching a synthetic evidence set exactly."""
    return _primary_population(
        [
            _annotation(
                record_id=row.record_id,
                channel_index=int(row.channel_index),
                start_sample=int(row.start_sample),
            )
            for row in evidence
        ]
    )


def _replay_identity(rows=473_897):
    """A FULL REPLAY identity payload. Never a metric denominator."""
    return X.M2ReplayPopulation(
        partition="validation",
        row_count=rows,
        ordered_stable_id_sha256="1" * 64,
        stream_cache_sha256="2" * 64,
    ).identity()


def _primary_identity():
    """A PRIMARY identity payload carrying the frozen validation counts."""
    return {
        "population": PP.POPULATION_PRIMARY,
        "partition": "validation",
        "authority": PP.PRIMARY_AUTHORITY,
        "authority_detail": "frozen P1 validation embedding cache",
        "row_count": PP.PRIMARY_VALIDATION_POPULATION["total"],
        "counts": dict(PP.PRIMARY_VALIDATION_POPULATION),
        "ordered_stable_id_sha256": "3" * 64,
        "p1_embedding_cache_sha256": "4" * 64,
        "membership_derived_from_m2_scores": False,
        "binary_labels_present": True,
        "evaluated_rows": PP.PRIMARY_VALIDATION_POPULATION["total"],
        "evaluated_ordered_stable_id_sha256": "3" * 64,
        "identity_key": "(record_id, channel_index, start_sample)",
        "identity_corresponds_to_frozen_stable_id": True,
        "positional_join_used": False,
        "matches_frozen_authority_exactly": True,
    }


def _challenge_identity():
    """A CHALLENGE identity payload carrying the frozen selection digest."""
    from cardiosentinel.neural.validation_challenge import (
        CHALLENGE_EXPECTED_COUNTS,
        CHALLENGE_SELECTION_SHA256,
        CHALLENGE_TOTAL_WINDOWS,
    )

    return {
        "population": PP.POPULATION_CHALLENGE,
        "partition": "validation",
        "authority": PP.CHALLENGE_AUTHORITY,
        "authority_detail": "build_validation_challenge_index(...)",
        "row_count": CHALLENGE_TOTAL_WINDOWS,
        "counts": {k: dict(v) for k, v in CHALLENGE_EXPECTED_COUNTS.items()},
        "challenge_selection_sha256": CHALLENGE_SELECTION_SHA256,
        "ordered_stable_id_sha256": "5" * 64,
        "binary_labels_invented": False,
        "membership_derived_from_m2_scores": False,
        "evaluated_rows": CHALLENGE_TOTAL_WINDOWS,
        "evaluated_ordered_stable_id_sha256": "5" * 64,
        "identity_key": "(record_id, channel_index, start_sample)",
        "identity_corresponds_to_frozen_stable_id": True,
        "positional_join_used": False,
        "matches_frozen_authority_exactly": True,
    }


def _source_identity():
    """A development source-integrity receipt, as the real verifier produces."""
    return {
        "identity_class": PS.DEVELOPMENT_SOURCE_IDENTITY_CLASS,
        "feature_receipt": {"verification_result": "passed"},
        "source_receipt": {"verification_result": "passed"},
        "annotation_set": "stb",
        "test_partition_hashed": False,
        "verified_before_stress_selection": True,
    }


def _stress_identity():
    """A real source-defined stress selection identity."""
    identity = dict(SI.build_stress_selection().identity())
    identity["development_source_identity"] = _source_identity()
    return identity


def _full_result(arm="M2-G", **overrides):
    """A complete canonical arm result payload, with FOUR distinct populations."""
    replay = _replay_identity()
    primary = _primary_identity()
    challenge = _challenge_identity()
    stress = _stress_identity()
    result = {
        "artifact_class": PS.ARM_RESULT_CLASS,
        "arm": arm,
        "scientific_computation_completed": True,
        "label_blind_replay_completed": True,
        "m1l_classification_threshold": SC.M1L_CLASSIFICATION_THRESHOLD,
        "threshold_selected_here": False,
        "classifier_retrained": False,
        "memory_selection_performed": False,
        "memory_selected": None,
        "rollback": False,
        "partition_accessed": "validation",
        "replay_population_identity": replay,
        "primary_evaluation_population_identity": primary,
        "challenge_evaluation_population_identity": challenge,
        "stress_interval_selection_identity": stress,
        "development_source_identity": _source_identity(),
        **RUN.recovery_lineage(),
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "policy_evidence": {
            "update_admission_fraction": 0.2,
            "population_identity": replay,
        },
        "window_evidence": {"population_identity": primary},
        "false_alarm_evidence": {
            "background_population_identity": primary,
            "challenge_population_identity": challenge,
        },
        "cold_start_evidence": {"population_identity": primary},
        "contamination_evidence": {
            "intervals": [],
            "replay_population_identity": replay,
            "stress_interval_selection_identity": stress,
        },
    }
    result.update(overrides)
    return result


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
    stray = [_annotation(start_sample=7777)]
    with pytest.raises(V.M2EvaluationError, match="no replay evidence"):
        V.build_primary_bundle(
            "M2-G", evidence, stray, primary_population=_primary_population(stray)
        )


def test_evaluation_functions_require_an_identity_joined_bundle():
    """The old positional (labels=..., subject_ids=...) API is gone."""
    import inspect

    for function in (
        V.window_evidence,
        V.background_false_positive_evidence,
        V.cold_start_stratified_evidence,
        V.challenge_false_positive_evidence,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & {"labels", "subject_ids", "target_families"}), function
        assert "bundle" in parameters
    # The combined false-alarm section takes BOTH denominators, explicitly.
    combined = set(inspect.signature(V.false_alarm_evidence).parameters)
    assert {"primary_bundle", "challenge_bundle"} <= combined


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


def test_simulated_mismatch_before_persistence_refuses_promotion(
    tmp_path, frozen_runtime
):
    claimed = PS.claim_run_directory(
        tmp_path, "M2_sim_prepromotion", "M2-G", runtime=frozen_runtime
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


def test_partial_run_cannot_masquerade_as_complete(tmp_path, frozen_runtime):
    claimed = PS.claim_run_directory(
        tmp_path, "M2_partial", "M2-G", runtime=frozen_runtime
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
    with pytest.raises(PS.M2PersistenceError, match="test_accessed"):
        PS.audit_forbidden_partitions(
            {
                "partition_accessed": "validation",
                "validation_accessed": True,
                "test_accessed": True,
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
    subject="subj-1",
    bin_name="over_60_minutes",
):
    """A PRIMARY annotation. It carries a label and no target family."""
    return V.M2PrimaryAnnotation(
        record_id=record_id,
        channel_index=channel_index,
        start_sample=start_sample,
        label=label,
        subject_id=subject,
        cold_start_bin=bin_name,
    )


def _challenge_annotation(
    record_id="s00001",
    channel_index=0,
    start_sample=0,
    family="rate_related_confounder",
    subject="subj-1",
):
    """A CHALLENGE annotation. There is nowhere to put a binary label."""
    return V.M2ChallengeAnnotation(
        record_id=record_id,
        channel_index=channel_index,
        start_sample=start_sample,
        target_family=family,
        subject_id=subject,
    )


def _primary_population(annotations):
    """A frozen-authority PRIMARY token for a synthetic annotation set.

    Real runs obtain this from `m2_populations.primary_evaluation_population`,
    which proves the frozen 473,897/21,628/452,269/12 identity against the P1
    embedding cache. Synthetic tests supply their own expected counts; the
    point under test is that evaluation accepts ONLY an authority token and
    then requires the evaluated rows to be exactly that population.
    """
    rows = list(annotations)
    labels = [int(a.label) for a in rows]
    subjects = [str(a.subject_id) for a in rows]
    return PP.verify_primary_population(
        stable_ids=[a.stable_id for a in rows],
        labels=labels,
        subject_ids=subjects,
        cache_sha256="a" * 64,
        expected_counts={
            "total": len(rows),
            "positive": sum(1 for v in labels if v == 1),
            "negative": sum(1 for v in labels if v == 0),
            "subjects": len(set(subjects)),
        },
    )


def _challenge_population(annotations):
    """A frozen-authority CHALLENGE token for a synthetic selection."""
    rows = list(annotations)
    families = [str(a.target_family) for a in rows]
    counts = {
        family: {
            "windows": sum(1 for f in families if f == family),
            "subjects": len(
                {str(a.subject_id) for a in rows if str(a.target_family) == family}
            ),
        }
        for family in PP.CHALLENGE_FAMILIES
    }
    return PP.verify_challenge_population(
        stable_ids=[a.stable_id for a in rows],
        target_families=families,
        subject_ids=[str(a.subject_id) for a in rows],
        selection_sha256="d" * 64,
        counts=counts,
        expected_selection_sha256="d" * 64,
        expected_counts=counts,
        expected_total=len(rows),
    )


def _paired_bundle(count=4):
    """A minimal well-formed PRIMARY bundle with both classes and two subjects."""
    evidence, annotations = [], []
    for index in range(count):
        start = index * 1250
        evidence.append(_evidence(start_sample=start, score=0.1 + 0.2 * index))
        annotations.append(
            _annotation(
                start_sample=start,
                label=index % 2,
                subject=f"subj-{index % 2}",
            )
        )
    return V.build_primary_bundle(
        "M2-G",
        evidence,
        annotations,
        primary_population=_primary_population(annotations),
    )


def _challenge_bundle(count=3):
    """A minimal well-formed CHALLENGE bundle over confounder rows."""
    evidence, annotations = [], []
    for index in range(count):
        start = 500_000 + index * 1250
        evidence.append(_evidence(start_sample=start, score=0.1 + 0.1 * index))
        annotations.append(
            _challenge_annotation(
                start_sample=start,
                family=PP.CHALLENGE_FAMILIES[index % len(PP.CHALLENGE_FAMILIES)],
                subject=f"subj-{index % 2}",
            )
        )
    return V.build_challenge_bundle(
        "M2-G",
        evidence,
        annotations,
        challenge_population=_challenge_population(annotations),
    )


# --- 1-2. threshold enforcement on every thresholded path -----------------


def test_every_thresholded_metric_rejects_a_non_frozen_threshold():
    bundle = _paired_bundle()
    challenge = _challenge_bundle()
    for function in (
        V.window_evidence,
        V.background_false_positive_evidence,
        V.cold_start_stratified_evidence,
    ):
        with pytest.raises(V.M2EvaluationError, match="frozen retained M1L"):
            function(bundle, threshold=ALT_THRESHOLD)
    with pytest.raises(V.M2EvaluationError, match="frozen retained M1L"):
        V.challenge_false_positive_evidence(challenge, threshold=ALT_THRESHOLD)
    with pytest.raises(V.M2EvaluationError, match="frozen retained M1L"):
        V.false_alarm_evidence(
            primary_bundle=bundle,
            challenge_bundle=challenge,
            threshold=ALT_THRESHOLD,
        )


def test_normal_evidence_threshold_is_rejected_as_a_classification_threshold():
    bundle = _paired_bundle()
    for function in (
        V.window_evidence,
        V.background_false_positive_evidence,
        V.cold_start_stratified_evidence,
    ):
        with pytest.raises(V.M2EvaluationError):
            function(bundle, threshold=SC.NORMAL_EVIDENCE_THRESHOLD)
    with pytest.raises(V.M2EvaluationError):
        V.challenge_false_positive_evidence(
            _challenge_bundle(), threshold=SC.NORMAL_EVIDENCE_THRESHOLD
        )


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
    with pytest.raises(V.M2EvaluationError, match="no replay evidence"):
        V.build_primary_bundle(
            "M2-G", evidence, wrong, primary_population=_primary_population(wrong)
        )


def test_permuted_annotation_ordering_is_realigned_by_identity_not_position():
    evidence = [
        _evidence(start_sample=0, score=0.10),
        _evidence(start_sample=1250, score=0.90),
    ]
    ordered = [
        _annotation(start_sample=0, label=0, subject="a"),
        _annotation(start_sample=1250, label=1, subject="b"),
    ]
    token = _primary_population(ordered)
    bundle_ordered = V.build_primary_bundle(
        "M2-G", evidence, ordered, primary_population=token
    )
    bundle_permuted = V.build_primary_bundle(
        "M2-G", evidence, list(reversed(ordered)), primary_population=token
    )
    # Identity, not order, decides the pairing.
    assert np.array_equal(bundle_ordered.stable_ids, bundle_permuted.stable_ids)
    assert np.array_equal(bundle_ordered.labels, bundle_permuted.labels)
    assert np.array_equal(bundle_ordered.scores, bundle_permuted.scores)
    # And the score/label pairing is the correct one.
    assert bundle_ordered.scores[0] == 0.10
    assert bundle_ordered.labels[0] == 0
    assert bundle_ordered.scores[1] == 0.90
    assert bundle_ordered.labels[1] == 1


def test_duplicate_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0)]
    single = [_annotation(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="Duplicate primary annotation"):
        V.build_primary_bundle(
            "M2-G",
            evidence,
            [_annotation(start_sample=0), _annotation(start_sample=0)],
            primary_population=_primary_population(single),
        )


def test_duplicate_evidence_identities_are_rejected():
    duplicated = [_evidence(start_sample=0), _evidence(start_sample=0)]
    single = [_annotation(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="Duplicate evidence"):
        V.build_primary_bundle(
            "M2-G",
            duplicated,
            single,
            primary_population=_primary_population(single),
        )


def test_missing_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    both = [_annotation(start_sample=0), _annotation(start_sample=1250)]
    partial = [_annotation(start_sample=0)]
    with pytest.raises(V.M2EvaluationError, match="not the frozen population"):
        V.build_primary_bundle(
            "M2-G", evidence, partial, primary_population=_primary_population(both)
        )


def test_extra_annotation_identities_are_rejected():
    evidence = [_evidence(start_sample=0)]
    extra = [_annotation(start_sample=0), _annotation(start_sample=1250)]
    with pytest.raises(V.M2EvaluationError, match="no replay evidence"):
        V.build_primary_bundle(
            "M2-G", evidence, extra, primary_population=_primary_population(extra)
        )


def test_a_subset_can_never_opt_into_being_the_primary_population():
    """There is no `require_full_population=False` escape hatch any more.

    The primary denominator is the frozen P1 population, full stop. A caller
    cannot narrow it by annotating fewer rows and asking for a subset bundle.
    """
    import inspect

    parameters = set(inspect.signature(V.build_primary_bundle).parameters)
    assert "require_full_population" not in parameters
    assert "primary_population" in parameters

    evidence = [_evidence(start_sample=0), _evidence(start_sample=1250)]
    both = [_annotation(start_sample=0), _annotation(start_sample=1250)]
    with pytest.raises(V.M2EvaluationError, match="never widened or narrowed"):
        V.build_primary_bundle(
            "M2-G",
            evidence,
            [_annotation(start_sample=0)],
            primary_population=_primary_population(both),
        )


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
    primary_fields = set(V.M2PrimaryAnnotation.__dataclass_fields__)
    challenge_fields = set(V.M2ChallengeAnnotation.__dataclass_fields__)
    assert "subject_id" in primary_fields and "subject_id" in challenge_fields
    # The two annotation types are purpose-specific and NOT interchangeable:
    # only primary carries a label, only challenge carries a target family.
    assert "label" in primary_fields and "label" not in challenge_fields
    assert "target_family" in challenge_fields
    assert "target_family" not in primary_fields
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
    single = [_annotation(start_sample=0)]
    with pytest.raises(V.M2EvaluationError) as caught:
        V.build_primary_bundle(
            "M2-G", evidence, single, primary_population=_primary_population(single)
        )
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


def test_result_provenance_binds_all_four_population_identities():
    assert PS.POPULATION_IDENTITY_FIELDS == (
        "replay_population_identity",
        "primary_evaluation_population_identity",
        "challenge_evaluation_population_identity",
        "stress_interval_selection_identity",
    )
    for field in PS.POPULATION_IDENTITY_FIELDS:
        assert field in PS.REQUIRED_PROVENANCE_FIELDS
    # And the single catch-all identity is gone, so no population can stand in
    # for another.
    assert "evaluated_population_identity" not in PS.REQUIRED_PROVENANCE_FIELDS
    bundle = _paired_bundle()
    identity = bundle.population_identity()
    assert identity["evaluated_rows"] == 4
    assert len(identity["evaluated_ordered_stable_id_sha256"]) == 64
    assert identity["positional_join_used"] is False
    # A different evaluated population yields a different identity.
    single = [_evidence(start_sample=0)]
    other = V.build_primary_bundle(
        "M2-G",
        single,
        [_annotation(start_sample=0)],
        primary_population=_primary_population_from_evidence(single),
    )
    assert (
        other.population_identity()["evaluated_ordered_stable_id_sha256"]
        != identity["evaluated_ordered_stable_id_sha256"]
    )


# --- 20. still no scientific result --------------------------------------


def test_no_scientific_m2_result_is_generated_by_this_module():
    evidence = [_evidence(start_sample=index * 1250) for index in range(4)]
    payload = V.arm_evaluation("M2-G", evidence)
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


SYNTHETIC_SHA = "a" * 64


def _execution_identity():
    """A complete execution identity, as the real harness produces."""
    return {
        "partition_accessed": "validation",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "input_identity": {
            "partition": "validation",
            "distance_standardizer_sha256": SYNTHETIC_SHA,
            "split_sha256": SYNTHETIC_SHA,
            "feature_corpus_sha256": SYNTHETIC_SHA,
            "ordered_chronology_sha256": SYNTHETIC_SHA,
            "stream_cache_sha256": SYNTHETIC_SHA,
        },
        "scorer_identity": {
            "retained_lock_sha256": SC.RETAINED_M1L_LOCK_SHA256,
            "retained_checkpoint_sha256": SC.RETAINED_M1L_CHECKPOINT_SHA256,
            "p1b_lock_sha256": SC.FROZEN_P1B_LOCK_SHA256,
            "b4b_checkpoint_sha256": SC.FROZEN_B4B_CHECKPOINT_SHA256,
            "classification_threshold": SC.M1L_CLASSIFICATION_THRESHOLD,
            "memory_admission_threshold": SC.NORMAL_EVIDENCE_THRESHOLD,
            "classification_threshold_used_for_memory_admission": False,
        },
    }


def _claim_bearing_result(arm="M2-G", population=None):
    """The arm result payload. Provenance lives in the separate run lock."""
    return {
        "arm": arm,
        "scientific_computation_completed": True,
        "evaluated_population_identity": population,
    }


def _complete_lock(tmp_path=None, *, population=None, **overrides):
    """A fully provenance-complete canonical lock, for validator tests."""
    import cardiosentinel.neural.m2_persistence as _ps

    original = _ps.git_provenance
    _ps.git_provenance = lambda _root: {"git_sha": "0" * 40, "git_dirty": False}
    try:
        return _build_complete_lock(population=population, **overrides)
    finally:
        _ps.git_provenance = original


def _build_complete_lock(*, population=None, **overrides):
    runtime = _frozen_runtime_record()
    for point in ("pre_promotion", "completion"):
        runtime.record(_synthetic_frozen_check(point))
    lock = PS.build_canonical_run_lock(
        experiment_id="M2_lock_fixture",
        arm="M2-G",
        execution_identity=_execution_identity(),
        runtime=runtime,
        population_identities=(population or _population_identities()),
        development_source_identity=_source_identity(),
        recovery_lineage=RUN.recovery_lineage(),
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:10:00Z",
        artifact_sha256={PS.ARM_RESULT_NAME: "b" * 64},
    )
    if overrides:
        lock = {**lock, **overrides}
        lock.pop("experiment_lock_sha256", None)
        from cardiosentinel.neural.integrity import canonical_sha256

        lock["experiment_lock_sha256"] = canonical_sha256(lock)
    return lock


# --- 1-3. claim directory sentinel boundary ------------------------------


def test_claim_directory_requires_a_successful_start_record(tmp_path):
    empty = S.RuntimeIntegrityRecord()
    with pytest.raises(PS.M2PersistenceError, match="START runtime check"):
        PS.claim_run_directory(tmp_path, "M2_no_start", "M2-G", runtime=empty)
    assert not (tmp_path / "M2_no_start").exists()


def test_claim_directory_performs_a_pre_promotion_check_before_creation(
    tmp_path, frozen_runtime
):
    runtime = frozen_runtime
    PS.claim_run_directory(tmp_path, "M2_claim_check", "M2-G", runtime=runtime)
    points = [check.enforcement_point for check in runtime.checks]
    assert points == ["start", "pre_promotion"]
    details = [check.detail for check in runtime.checks]
    assert PS.CLAIM_DIRECTORY_PROMOTION_DETAIL in details


def test_non_frozen_record_cannot_create_a_canonical_claim(tmp_path):
    """§1: a matching NON-FROZEN digest is never sufficient for a claim.

    The record below is entirely self-consistent -- it expects "c"*64 and its
    START observed and matched "c"*64 -- yet it must not produce a canonical
    claim, because canonical standing requires the frozen scientific identity.
    """
    runtime = S.RuntimeIntegrityRecord(expected_digest="c" * 64)
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
    with pytest.raises(PS.M2PersistenceError, match="frozen scientific identity"):
        PS.claim_run_directory(tmp_path, "M2_claim_mismatch", "M2-G", runtime=runtime)
    assert not (tmp_path / "M2_claim_mismatch").exists()


def test_canonical_claim_requires_the_frozen_expected_digest():
    """§8.2 -- the record's expectation itself must be the frozen identity."""
    ambient = S.observe_runtime_identity(S.EnforcementPoint.START).observed_digest
    runtime = S.RuntimeIntegrityRecord(expected_digest=ambient)
    runtime.record(
        S.RuntimeCheck(
            enforcement_point="start",
            observed_digest=ambient,
            expected_digest=ambient,
            matches=True,
            package_count=1,
            observed_at="2026-01-01T00:00:00Z",
        )
    )
    if ambient != FROZEN_DIGEST:
        with pytest.raises(PS.M2PersistenceError, match="frozen scientific identity"):
            PS.require_frozen_runtime_record(runtime)


def test_canonical_claim_requires_start_expected_and_observed_frozen():
    """§8.3/§8.4 -- START must have expected AND observed the frozen digest."""
    wrong_expected = S.RuntimeIntegrityRecord()
    wrong_expected.record(
        S.RuntimeCheck(
            enforcement_point="start",
            observed_digest=FROZEN_DIGEST,
            expected_digest="d" * 64,
            matches=True,
            package_count=335,
            observed_at="2026-01-01T00:00:00Z",
        )
    )
    with pytest.raises(PS.M2PersistenceError, match="expected"):
        PS.require_frozen_runtime_record(wrong_expected)

    wrong_observed = S.RuntimeIntegrityRecord()
    wrong_observed.record(
        S.RuntimeCheck(
            enforcement_point="start",
            observed_digest="e" * 64,
            expected_digest=FROZEN_DIGEST,
            matches=False,
            package_count=1,
            observed_at="2026-01-01T00:00:00Z",
        )
    )
    with pytest.raises(PS.M2PersistenceError, match="observe the frozen"):
        PS.require_frozen_runtime_record(wrong_observed)


def test_test_only_mechanism_does_not_weaken_the_production_invariant():
    """§8.5 -- the synthetic fixture asserts the frozen digest, nothing less."""
    check = _synthetic_frozen_check("start")
    assert check.expected_digest == FROZEN_DIGEST
    assert check.observed_digest == FROZEN_DIGEST
    # And production still refuses anything else, regardless of the fixture.
    import inspect

    source = inspect.getsource(PS.require_frozen_runtime_record)
    assert "FROZEN_DEPENDENCY_DIGEST" in source


# --- 4-7. completion binding and runtime-block completeness ---------------


def test_canonical_complete_result_cannot_carry_a_none_completion_digest():
    runtime = _frozen_runtime_record()
    runtime.record(
        S.observe_runtime_identity(S.EnforcementPoint.PRE_PROMOTION, detail="x")
    )
    # No COMPLETION observation has been taken yet.
    assert runtime.digest_at(S.EnforcementPoint.COMPLETION) is None
    with pytest.raises(PS.M2PersistenceError, match="completion"):
        PS.validate_complete_runtime_identity(runtime)


def test_completion_mismatch_prevents_canonical_complete_promotion(
    tmp_path, frozen_runtime, monkeypatch
):
    """A COMPLETION mismatch invalidates canonical standing entirely."""
    runtime = frozen_runtime
    claimed = PS.claim_run_directory(tmp_path, "M2_end_bad", "M2-G", runtime=runtime)

    # The record still legitimately expects the frozen identity; only the
    # COMPLETION observation differs, which is the real-world failure mode.
    def observe_bad_completion(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        value = S.EnforcementPoint(point).value
        if value == "completion":
            return S.RuntimeCheck(
                enforcement_point=value,
                observed_digest="9" * 64,
                expected_digest=FROZEN_DIGEST,
                matches=False,
                package_count=71,
                observed_at="2026-01-01T00:00:00Z",
                detail=detail,
            )
        return _synthetic_frozen_check(value, detail or "test")

    monkeypatch.setattr(PS, "observe_runtime_identity", observe_bad_completion)

    with pytest.raises(S.RuntimeIntegrityError, match="COMPLETION"):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=_full_result(),
            execution_identity=_execution_identity(),
            runtime=runtime,
            requires_evaluation=True,
        )

    # Nothing canonical exists: no result, no lock, and the run is not COMPLETE.
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    assert not (claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).exists()
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["claim_bearing_result_promoted"] is False
    assert status["canonical"] is False
    failure = json.loads((claimed.run_dir / PS.RUNTIME_FAILURE_NAME).read_text())
    assert failure["canonical_promotion_invalidated"] is True
    assert failure["staged_result_retained_as_forensic_material"] is True
    assert failure["claim_bearing"] is False
    # Forensic evidence is retained, not deleted.
    assert (claimed.staging_dir / PS.ARM_RESULT_NAME).exists()


def test_all_three_enforcement_points_appear_in_the_finalized_block(
    tmp_path, frozen_runtime
):
    runtime = frozen_runtime
    claimed = PS.claim_run_directory(tmp_path, "M2_full_block", "M2-G", runtime=runtime)
    status = PS.finalize_and_promote_arm_result(
        claimed,
        result=_full_result(),
        execution_identity=_execution_identity(),
        runtime=runtime,
        requires_evaluation=True,
    )
    assert status["status"] == PS.STATUS_COMPLETE
    assert status["canonical"] is True

    # The result file carries no hash of itself; the separate lock binds it.
    lock = json.loads((claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    block = lock["runtime_identity_checks"]
    points = {check["enforcement_point"] for check in block["checks"]}
    assert points == {"start", "pre_promotion", "completion"}
    assert block["all_observations_matched"] is True
    assert lock["runtime_dependency_digest_start"] == FROZEN_DIGEST
    assert lock["runtime_dependency_digest_pre_promotion"] == FROZEN_DIGEST
    assert lock["runtime_dependency_digest_end"] == FROZEN_DIGEST

    # The lock binds the exact promoted artifact bytes, and re-validates.
    from cardiosentinel.data.provenance import sha256_file

    promoted_digest = sha256_file(claimed.run_dir / PS.ARM_RESULT_NAME)
    assert lock["artifact_sha256"][PS.ARM_RESULT_NAME] == promoted_digest
    assert status["artifact_sha256"][PS.ARM_RESULT_NAME] == promoted_digest
    assert status["experiment_lock_sha256"] == lock["experiment_lock_sha256"]
    PS.validate_canonical_run_lock(lock, run_dir=claimed.run_dir)

    promoted = json.loads((claimed.run_dir / PS.ARM_RESULT_NAME).read_text())
    assert "experiment_lock_sha256" not in promoted  # no self-referential hash


def test_all_observations_matched_is_required_for_canonical_evidence():
    runtime = _frozen_runtime_record()
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


@pytest.mark.parametrize(
    ("validator", "field"),
    [
        ("validate_replay_population_identity", "replay_population_identity"),
        (
            "validate_primary_population_identity",
            "primary_evaluation_population_identity",
        ),
        (
            "validate_challenge_population_identity",
            "challenge_evaluation_population_identity",
        ),
        (
            "validate_stress_selection_identity",
            "stress_interval_selection_identity",
        ),
    ],
)
def test_none_population_identity_is_rejected_for_claim_bearing_evaluation(
    validator, field
):
    for empty in (None, {}):
        with pytest.raises(PS.M2PersistenceError, match=field):
            getattr(PS, validator)(empty)


def test_malformed_primary_population_identity_is_rejected():
    valid = _primary_identity()
    for mutation in (
        {"evaluated_rows": 0},
        {"evaluated_rows": -1},
        {"evaluated_ordered_stable_id_sha256": "short"},
        {"positional_join_used": True},
        {"identity_key": ""},
        {"matches_frozen_authority_exactly": False},
        {"counts": {"total": 1, "positive": 1, "negative": 0, "subjects": 1}},
        {"membership_derived_from_m2_scores": True},
        {"binary_labels_present": False},
        {"authority": "some_other_authority"},
    ):
        with pytest.raises(PS.M2PersistenceError):
            PS.validate_primary_population_identity({**valid, **mutation})


def test_malformed_challenge_population_identity_is_rejected():
    valid = _challenge_identity()
    for mutation in (
        {"challenge_selection_sha256": "z" * 64},
        {"row_count": 8136},
        {"binary_labels_invented": True},
        {"authority": "frozen_p1_validation_population"},
        {"population": PP.POPULATION_PRIMARY},
    ):
        with pytest.raises(PS.M2PersistenceError):
            PS.validate_challenge_population_identity({**valid, **mutation})


def test_malformed_stress_selection_identity_is_rejected():
    valid = _stress_identity()
    for mutation in (
        {"decision_sha256": "z" * 64},
        {"decision_document": "docs/other.md"},
        {"marker_vicinity_reused_as_stress_duration": True},
        {"persistence_duration_invented": True},
        {"merge_gap_applied": True},
        {"selection_influenced_by_m2_outputs": True},
        {"selection_performed_after_label_blind_replay": False},
        {"source_defined_families": ["ischemic"]},
    ):
        with pytest.raises(PS.M2PersistenceError):
            PS.validate_stress_selection_identity({**valid, **mutation})


def test_valid_population_identities_are_accepted():
    assert PS.validate_replay_population_identity(_replay_identity())["row_count"] > 0
    primary = PS.validate_primary_population_identity(_primary_identity())
    assert primary["counts"] == dict(PP.PRIMARY_VALIDATION_POPULATION)
    challenge = PS.validate_challenge_population_identity(_challenge_identity())
    assert challenge["binary_labels_invented"] is False
    stress = PS.validate_stress_selection_identity(_stress_identity())
    assert stress["decision_sha256"] == SI.DECISION_SHA256


def test_claim_bearing_result_requires_population_identity_when_evaluated(
    tmp_path, frozen_runtime
):
    runtime = frozen_runtime
    claimed = PS.claim_run_directory(tmp_path, "M2_no_pop", "M2-G", runtime=runtime)
    # An otherwise-complete result whose primary population identity is absent.
    result = {
        **_full_result(),
        "primary_evaluation_population_identity": None,
    }
    with pytest.raises(
        PS.M2PersistenceError, match="primary_evaluation_population_identity"
    ):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=result,
            execution_identity=_execution_identity(),
            runtime=runtime,
            requires_evaluation=True,
        )
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    assert not (claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).exists()


# --- 11-14. headline metrics require full-population bundles --------------


@pytest.mark.parametrize(
    "function",
    [
        V.window_evidence,
        V.background_false_positive_evidence,
        V.cold_start_stratified_evidence,
    ],
)
def test_primary_metrics_refuse_a_challenge_bundle(function):
    """A primary headline metric can never run on the challenge denominator."""
    with pytest.raises(V.M2EvaluationError, match="PRIMARY metric population"):
        function(_challenge_bundle())


def test_challenge_fpr_refuses_the_primary_bundle():
    """p1_challenge_evidence is never called over the primary population."""
    with pytest.raises(V.M2EvaluationError, match="CHALLENGE metric population"):
        V.challenge_false_positive_evidence(_paired_bundle())


def test_population_specific_bundles_carry_their_own_authority():
    primary = _paired_bundle()
    challenge = _challenge_bundle()
    assert primary.authority == PP.PRIMARY_AUTHORITY
    assert challenge.authority == PP.CHALLENGE_AUTHORITY
    payload = V.cold_start_stratified_evidence(primary)
    assert payload["population"] == PP.POPULATION_PRIMARY
    assert payload["population_identity"]["authority"] == PP.PRIMARY_AUTHORITY
    # The challenge bundle has no labels at all, so none can be invented.
    assert not hasattr(challenge, "labels")


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

    tree = ast.parse(inspect.getsource(PS.validate_canonical_run_lock).lstrip())
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


# --------------------------------------------------------------------------
# Human final canonical-provenance review:
#   the complete lock contract, and full_population proven against the
#   canonical replay population rather than trusted from the caller.
# --------------------------------------------------------------------------


def test_minimal_result_fixture_is_rejected_as_incomplete_provenance():
    """§8.6 -- the small synthetic result cannot masquerade as canonical."""
    minimal = _claim_bearing_result()
    with pytest.raises(PS.M2PersistenceError):
        PS.validate_canonical_run_lock(minimal)


@pytest.mark.parametrize(
    "field",
    [
        "git_sha",
        "retained_m1l_checkpoint_sha256",
        "retained_m1l_lock_sha256",
        "split_sha256",
        "feature_corpus_sha256",
        "ordered_chronology_sha256",
        "signal_v1_schema_sha256",
        "morphology_v1_schema_sha256",
        "combined_v1_schema_sha256",
    ],
)
def test_missing_required_identity_is_rejected(field):
    """§8.7-8.10 -- a missing identity fails, key-presence is not enough."""
    from cardiosentinel.neural.integrity import canonical_sha256

    lock = _complete_lock()
    del lock[field]
    lock.pop("experiment_lock_sha256", None)
    lock["experiment_lock_sha256"] = canonical_sha256(lock)
    with pytest.raises(PS.M2PersistenceError, match="missing"):
        PS.validate_canonical_run_lock(lock)


@pytest.mark.parametrize(
    "field",
    ["m2_protocol_sha256", "retained_m1l_lock_sha256", "b4b_checkpoint_sha256"],
)
def test_wrong_frozen_identity_is_rejected(field):
    """§8.11 -- a well-formed but WRONG frozen identity fails."""
    lock = _complete_lock(**{field: "f" * 64})
    with pytest.raises(PS.M2PersistenceError, match="expected the frozen"):
        PS.validate_canonical_run_lock(lock)


def test_malformed_sha_is_rejected():
    lock = _complete_lock(split_sha256="not-a-digest")
    with pytest.raises(PS.M2PersistenceError, match="not a SHA-256 digest"):
        PS.validate_canonical_run_lock(lock)


def test_dirty_git_state_is_rejected():
    """§4 -- canonical evidence requires a clean checkout (P1/M1 convention)."""
    lock = _complete_lock(git_dirty=True)
    with pytest.raises(PS.M2PersistenceError, match="clean Git checkout"):
        PS.validate_canonical_run_lock(lock)


def test_complete_canonical_provenance_is_accepted():
    """§8.12 -- a fully provenance-complete lock validates."""
    lock = _complete_lock()
    validated = PS.validate_canonical_run_lock(lock)
    assert validated["arm"] in ("M2-0", "M2-G")
    for field in PS.REQUIRED_PROVENANCE_FIELDS:
        assert field in validated


def test_changing_one_result_byte_invalidates_the_lock_binding(tmp_path):
    """§8.13-8.14 -- the lock binds the exact promoted bytes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result_path = run_dir / PS.ARM_RESULT_NAME
    result_path.write_text('{"arm": "M2-G"}\n')

    from cardiosentinel.data.provenance import sha256_file

    lock = _complete_lock(
        artifact_sha256={PS.ARM_RESULT_NAME: sha256_file(result_path)}
    )
    PS.validate_canonical_run_lock(lock, run_dir=run_dir)

    result_path.write_text('{"arm": "M2-G" }\n')  # one byte differs
    with pytest.raises(PS.M2PersistenceError, match="does not match its lock digest"):
        PS.validate_canonical_run_lock(lock, run_dir=run_dir)


# --- §5/§6 the metric population must be PROVEN, not asserted -------------


def test_self_consistent_subset_cannot_claim_the_primary_population():
    """The exact bypass: subset first, annotate the subset, claim the whole."""
    full_annotations = [_annotation(start_sample=i * 1250) for i in range(4)]
    subset_evidence = [_evidence(start_sample=i * 1250) for i in range(2)]
    subset_annotations = full_annotations[:2]
    # Evidence and annotations cover each other perfectly, and it is still
    # refused: mutual consistency is not membership in the frozen population.
    with pytest.raises(V.M2EvaluationError, match="not the frozen population"):
        V.build_primary_bundle(
            "M2-G",
            subset_evidence,
            subset_annotations,
            primary_population=_primary_population(full_annotations),
        )


def test_primary_scope_requires_the_frozen_authority_to_be_supplied():
    """Membership cannot rest on the caller's assertion alone."""
    import inspect

    evidence = [_evidence(start_sample=i * 1250) for i in range(2)]
    annotations = [_annotation(start_sample=i * 1250) for i in range(2)]
    parameters = inspect.signature(V.build_primary_bundle).parameters
    assert parameters["primary_population"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        V.build_primary_bundle("M2-G", evidence, annotations)


def test_a_forged_authority_token_is_refused(monkeypatch):
    """A non-authoritative reference cannot stand in for the frozen authority.

    The public API accepts no caller-computed digest at all, so the bypass
    cannot even be expressed: the population must arrive as a token whose
    source is the frozen P1 validation authority.
    """
    annotations = [_annotation(start_sample=i * 1250) for i in range(2)]
    evidence = [_evidence(start_sample=i * 1250) for i in range(2)]

    assert not hasattr(V, "canonical_replay_population_digest")
    assert not hasattr(V, "build_evaluation_bundle")

    forged = type(
        "ForgedPopulation",
        (),
        {
            "source": "caller_supplied",
            "stable_ids": tuple(a.stable_id for a in annotations),
            "labels": tuple(int(a.label) for a in annotations),
            "subject_ids": tuple(str(a.subject_id) for a in annotations),
            "identity": lambda self: {},
        },
    )()
    with pytest.raises(V.M2EvaluationError, match="frozen P1 validation authority"):
        V.build_primary_bundle("M2-G", evidence, annotations, primary_population=forged)
    with pytest.raises(V.M2EvaluationError, match="frozen validation challenge"):
        V.build_challenge_bundle("M2-G", evidence, [], challenge_population=forged)


def test_the_frozen_authority_cannot_be_relabelled_by_the_annotations():
    """A caller cannot flip a label on the way into the primary denominator."""
    annotations = [_annotation(start_sample=0, label=0)]
    evidence = [_evidence(start_sample=0)]
    authority = _primary_population(annotations)
    relabelled = [_annotation(start_sample=0, label=1)]
    with pytest.raises(V.M2EvaluationError, match="never reassigned"):
        V.build_primary_bundle(
            "M2-G", evidence, relabelled, primary_population=authority
        )


def test_the_exact_frozen_population_is_accepted():
    evidence = [_evidence(start_sample=i * 1250) for i in range(4)]
    annotations = [_annotation(start_sample=i * 1250) for i in range(4)]
    bundle = V.build_primary_bundle(
        "M2-G",
        evidence,
        annotations,
        primary_population=_primary_population(annotations),
    )
    identity = bundle.population_identity()
    assert identity["matches_frozen_authority_exactly"] is True
    assert identity["evaluated_rows"] == 4
    assert identity["positional_join_used"] is False
    assert identity["authority"] == PP.PRIMARY_AUTHORITY


def test_an_unproven_population_identity_is_rejected_for_claim_bearing():
    """A scope label alone is never enough; the exactness proof is required."""
    identity = dict(_primary_identity())
    identity["matches_frozen_authority_exactly"] = False
    with pytest.raises(PS.M2PersistenceError, match="EXACTLY the frozen population"):
        PS.validate_primary_population_identity(identity)


def test_policy_evidence_stays_available_without_a_metric_denominator():
    """Label-free policy evidence binds the FULL REPLAY population, not a metric."""
    evidence = [_evidence(start_sample=i * 1250) for i in range(2)]
    summary = V.policy_evidence(evidence)
    assert summary["evidence_class"] == "m2_policy_evidence"
    assert summary["population"] == PP.POPULATION_REPLAY
    # And it refuses to bind anything but the verified replay authority.
    forged = type("Forged", (), {"source": "caller_supplied"})()
    with pytest.raises(V.M2EvaluationError, match="FULL REPLAY authority"):
        V.policy_evidence(evidence, replay_population=forged)


# --------------------------------------------------------------------------
# Human final result-contract review:
#   canonical population authority, result payload contract, and a separate
#   PRE_PROMOTION observation for the experiment lock.
# --------------------------------------------------------------------------


def _input_bundle(rows, stable_ids, *, scope, manifest_extra=None):
    manifest = {
        "partition": "train",
        "full_stream_row_count": len(stable_ids),
        "ordered_stable_id_sha256": "0" * 64,
        "stream_cache_sha256": "c" * 64,
        "ordered_chronology_sha256": "d" * 64,
        "split_sha256": SYNTHETIC_SHA,
        "feature_corpus_sha256": SYNTHETIC_SHA,
        "representation_dim": 146,
        "distance_standardizer_sha256": SYNTHETIC_SHA,
        **(manifest_extra or {}),
    }
    return X.M2InputBundle(
        partition="train",
        rows=tuple(rows),
        stable_ids=tuple(stable_ids),
        standardizer=None,
        stream_cache_manifest=manifest,
        selection_scope=scope,
    )


# --- §11.1-3 bounded vs full input semantics -----------------------------


def test_record_filtered_input_bundle_is_explicitly_non_full():
    """§11.1 -- a bounded bundle cannot define canonical scope."""
    ids = ["ltstdb:s1:0:0:2500"]
    bundle = _input_bundle([], ids, scope=X.SELECTION_SCOPE_BOUNDED)
    assert bundle.is_full_partition is False
    assert bundle.identity()["selection_scope"] == X.SELECTION_SCOPE_BOUNDED
    with pytest.raises(X.M2ExecutionError, match="requires the full partition"):
        bundle.canonical_input_population_identity()


def test_full_bundle_row_count_must_equal_the_manifest():
    """§11.2 -- a row-count disagreement with the frozen manifest is fatal."""
    ids = ["ltstdb:s1:0:0:2500", "ltstdb:s1:0:1250:3750"]
    bundle = _input_bundle(
        [],
        ids,
        scope=X.SELECTION_SCOPE_FULL,
        manifest_extra={"full_stream_row_count": 99},
    )
    with pytest.raises(X.M2ExecutionError, match="frozen manifest records"):
        bundle.canonical_input_population_identity()


def test_full_bundle_digest_must_equal_the_frozen_manifest_identity():
    """§11.3 -- the bundle's own digest must match the frozen manifest."""
    ids = ["ltstdb:s1:0:0:2500", "ltstdb:s1:0:1250:3750"]
    bundle = _input_bundle([], ids, scope=X.SELECTION_SCOPE_FULL)
    with pytest.raises(X.M2ExecutionError, match="does not match the frozen manifest"):
        bundle.canonical_input_population_identity()

    # With the true digest recorded, the authority is issued.
    from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

    good = _input_bundle(
        [],
        ids,
        scope=X.SELECTION_SCOPE_FULL,
        manifest_extra={"ordered_stable_id_sha256": ordered_stable_id_digest(ids)},
    )
    token = good.canonical_input_population_identity()
    assert token.source == "verified_full_input_bundle"
    assert token.row_count == 2


def test_bounded_bundle_reports_its_own_digest_not_the_manifest():
    """A filtered bundle must not report the manifest identity as its own."""
    ids = ["ltstdb:s1:0:0:2500"]
    bundle = _input_bundle([], ids, scope=X.SELECTION_SCOPE_BOUNDED)
    identity = bundle.identity()
    assert identity["selected_row_count"] == 1
    assert (
        identity["selected_ordered_stable_id_sha256"]
        != identity["manifest_ordered_stable_id_sha256"]
    )


# --- §11.7-9 the result payload contract ---------------------------------


def test_minimal_result_fails_through_the_real_finalization_path(
    tmp_path, frozen_runtime
):
    """§7/§11.7 -- {"arm": "M2-G"} must not become canonical."""
    claimed = PS.claim_run_directory(
        tmp_path, "M2_minimal", "M2-G", runtime=frozen_runtime
    )
    with pytest.raises(PS.M2PersistenceError, match="missing required fields"):
        PS.finalize_and_promote_arm_result(
            claimed,
            result={"arm": "M2-G"},
            execution_identity=_execution_identity(),
            runtime=frozen_runtime,
            requires_evaluation=True,
        )
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    assert not (claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).exists()
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_STARTED
    assert status["claim_bearing_result_promoted"] is False


@pytest.mark.parametrize(
    "section",
    [
        "policy_evidence",
        "window_evidence",
        "false_alarm_evidence",
        "cold_start_evidence",
        "contamination_evidence",
        "scientific_computation_completed",
        "replay_population_identity",
        "primary_evaluation_population_identity",
        "challenge_evaluation_population_identity",
        "stress_interval_selection_identity",
    ],
)
def test_result_missing_any_mandatory_section_fails(tmp_path, frozen_runtime, section):
    """§11.8-9 -- each mandatory section is individually required."""
    claimed = PS.claim_run_directory(
        tmp_path, f"M2_missing_{section}", "M2-G", runtime=frozen_runtime
    )
    result = _full_result()
    del result[section]
    with pytest.raises(PS.M2PersistenceError):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=result,
            execution_identity=_execution_identity(),
            runtime=frozen_runtime,
            requires_evaluation=True,
        )
    assert not (claimed.run_dir / PS.ARM_RESULT_NAME).exists()
    assert not (claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).exists()


def test_empty_mandatory_section_is_rejected():
    """An omitted section may not be smuggled in as an empty object."""
    result = _full_result()
    result["window_evidence"] = {}
    with pytest.raises(PS.M2PersistenceError, match="protocol-valid exclusion"):
        PS.validate_claim_bearing_arm_result_payload(result)


# --- §11.10-11 result / lock / section population coherence ---------------


def test_section_population_identity_must_agree_with_the_result(tmp_path):
    """Each metric and ITS OWN declared population must describe same rows."""
    primary = _primary_identity()
    other = dict(primary)
    other["evaluated_rows"] = primary["evaluated_rows"] + 1
    result = _full_result()
    result["window_evidence"] = {"population_identity": other}
    with pytest.raises(PS.M2PersistenceError, match="differs from"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_a_section_may_not_borrow_another_populations_denominator():
    """Window evidence declaring the CHALLENGE population is fatal."""
    result = _full_result()
    result["window_evidence"] = {"population_identity": _challenge_identity()}
    with pytest.raises(PS.M2PersistenceError, match="differs from"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_a_section_must_declare_the_population_it_was_computed_over():
    result = _full_result()
    result["cold_start_evidence"] = {"strata": {}}
    with pytest.raises(PS.M2PersistenceError, match="does not declare"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_result_and_lock_population_identities_are_identical(tmp_path, frozen_runtime):
    """§11.10 -- the lock copies the result's identity; they cannot disagree."""
    claimed = PS.claim_run_directory(
        tmp_path, "M2_coherent", "M2-G", runtime=frozen_runtime
    )
    PS.finalize_and_promote_arm_result(
        claimed,
        result=_full_result(),
        execution_identity=_execution_identity(),
        runtime=frozen_runtime,
        requires_evaluation=True,
    )
    promoted = json.loads((claimed.run_dir / PS.ARM_RESULT_NAME).read_text())
    lock = json.loads((claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    for field in PS.POPULATION_IDENTITY_FIELDS:
        assert promoted[field] == lock[field]
    # And the four remain distinct: none stands in for another.
    identities = [promoted[f] for f in PS.POPULATION_IDENTITY_FIELDS]
    assert len({json.dumps(i, sort_keys=True) for i in identities}) == 4


# --- §11.12-16 separate promotion observations ----------------------------


def test_result_and_lock_each_have_their_own_pre_promotion_check(
    tmp_path, frozen_runtime
):
    """§11.12-14 -- two claim-bearing artifacts, two observations."""
    claimed = PS.claim_run_directory(
        tmp_path, "M2_two_checks", "M2-G", runtime=frozen_runtime
    )
    PS.finalize_and_promote_arm_result(
        claimed,
        result=_full_result(),
        execution_identity=_execution_identity(),
        runtime=frozen_runtime,
        requires_evaluation=True,
    )
    details = [check.detail for check in frozen_runtime.checks]
    assert PS.CLAIM_DIRECTORY_PROMOTION_DETAIL in details
    assert f"promote:{PS.ARM_RESULT_NAME}" in details
    assert f"promote:{PS.EXPERIMENT_LOCK_NAME}" in details
    points = [check.enforcement_point for check in frozen_runtime.checks]
    assert points.count("pre_promotion") == 3
    assert "start" in points and "completion" in points


def test_lock_binds_its_own_promotion_observation(tmp_path, frozen_runtime):
    """§11.15-16 -- the lock records the lock-promotion check it was gated by.

    The observation is taken BEFORE the lock is built, so it genuinely appears
    in `runtime_identity_checks` and the self-digest covers it. It is never
    fabricated after the fact.
    """
    claimed = PS.claim_run_directory(
        tmp_path, "M2_lock_binds", "M2-G", runtime=frozen_runtime
    )
    PS.finalize_and_promote_arm_result(
        claimed,
        result=_full_result(),
        execution_identity=_execution_identity(),
        runtime=frozen_runtime,
        requires_evaluation=True,
    )
    lock = json.loads((claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    recorded = [c["detail"] for c in lock["runtime_identity_checks"]["checks"]]
    assert f"promote:{PS.EXPERIMENT_LOCK_NAME}" in recorded

    # The self-digest covers that block: recomputing over the body matches.
    from cardiosentinel.neural.integrity import canonical_sha256

    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    assert lock["experiment_lock_sha256"] == canonical_sha256(body)
    PS.validate_canonical_run_lock(lock, run_dir=claimed.run_dir)


def test_lock_promotion_mismatch_leaves_the_run_non_canonical(
    tmp_path, frozen_runtime, monkeypatch
):
    """§10 -- a lock-promotion failure never yields COMPLETE/canonical."""
    claimed = PS.claim_run_directory(
        tmp_path, "M2_lock_fail", "M2-G", runtime=frozen_runtime
    )

    def observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        value = S.EnforcementPoint(point).value
        if detail == f"promote:{PS.EXPERIMENT_LOCK_NAME}":
            return S.RuntimeCheck(
                enforcement_point=value,
                observed_digest="7" * 64,
                expected_digest=FROZEN_DIGEST,
                matches=False,
                package_count=71,
                observed_at="2026-01-01T00:00:00Z",
                detail=detail,
            )
        return _synthetic_frozen_check(value, detail or "test")

    monkeypatch.setattr(PS, "observe_runtime_identity", observe)
    with pytest.raises(S.RuntimeIntegrityError, match="experiment-lock promotion"):
        PS.finalize_and_promote_arm_result(
            claimed,
            result=_full_result(),
            execution_identity=_execution_identity(),
            runtime=frozen_runtime,
            requires_evaluation=True,
        )
    # No lock, run not COMPLETE, already-promoted evidence preserved.
    assert not (claimed.run_dir / PS.EXPERIMENT_LOCK_NAME).exists()
    status = json.loads((claimed.run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["canonical"] is False
    failure = json.loads((claimed.run_dir / PS.RUNTIME_FAILURE_NAME).read_text())
    assert failure["canonical_promotion_invalidated"] is True
    assert failure["automatic_retry_performed"] is False
    assert (claimed.run_dir / PS.ARM_RESULT_NAME).exists()  # preserved, not deleted
