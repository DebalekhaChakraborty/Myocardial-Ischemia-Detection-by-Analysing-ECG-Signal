"""The M2-v1 four-population contract and the canonical development route.

Synthetic fixtures only. No VALIDATION stream cache is opened, no VALIDATION
rows are enumerated, no VALIDATION annotations are loaded, no primary or
challenge counts are derived from local data, no M2 arm is run on VALIDATION
and no M2 metric is calculated. The B4 sealed TEST partition is untouched and
no `TEST_ATTEMPT` is created.

The frozen scientific rules are inherited unchanged: nothing here alters M2-0,
M2-G, G1-G6, a threshold, the refractory, the memory policy, the challenge
definitions or the prototype-drift formula.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_development_run as R
from cardiosentinel.neural import m2_evaluation as V
from cardiosentinel.neural import m2_evidence_store as ES
from cardiosentinel.neural import m2_execution as X
from cardiosentinel.neural import m2_persistence as PS
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural import m2_populations as PP
from cardiosentinel.neural import m2_stress_intervals as SI
from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

NEURAL = REPOSITORY_ROOT / "src/cardiosentinel/neural"


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def _decision(score=0.1, arm="M2-G"):
    return P.M2GateDecision(
        arm=arm,
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


def _evidence(start_sample=0, score=0.1, arm="M2-G", record_id="s00001", channel=0):
    return P.M2RowEvidence(
        record_id=record_id,
        channel_index=channel,
        start_sample=start_sample,
        available_time=(start_sample + 2500) / 250.0,
        observation_state=1,
        arm=arm,
        decision=_decision(score, arm),
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


def _primary_rows(count=4):
    return [
        V.M2PrimaryAnnotation(
            record_id="s00001",
            channel_index=0,
            start_sample=index * 1250,
            label=index % 2,
            subject_id=f"subj-{index % 2}",
            cold_start_bin="over_60_minutes",
        )
        for index in range(count)
    ]


def _challenge_rows(count=3):
    return [
        V.M2ChallengeAnnotation(
            record_id="s00002",
            channel_index=0,
            start_sample=index * 1250,
            target_family=PP.CHALLENGE_FAMILIES[index % len(PP.CHALLENGE_FAMILIES)],
            subject_id=f"subj-{index % 2}",
        )
        for index in range(count)
    ]


def _primary_authority(rows):
    labels = [int(r.label) for r in rows]
    subjects = [str(r.subject_id) for r in rows]
    return PP.verify_primary_population(
        stable_ids=[r.stable_id for r in rows],
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


def _challenge_authority(rows):
    families = [str(r.target_family) for r in rows]
    counts = {
        family: {
            "windows": sum(1 for f in families if f == family),
            "subjects": len(
                {str(r.subject_id) for r in rows if str(r.target_family) == family}
            ),
        }
        for family in PP.CHALLENGE_FAMILIES
    }
    return PP.verify_challenge_population(
        stable_ids=[r.stable_id for r in rows],
        target_families=families,
        subject_ids=[str(r.subject_id) for r in rows],
        selection_sha256="d" * 64,
        counts=counts,
        expected_selection_sha256="d" * 64,
        expected_counts=counts,
        expected_total=len(rows),
    )


def _replay_authority(all_rows):
    from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

    ids = sorted(r.stable_id for r in all_rows)
    return X.M2ReplayPopulation(
        partition="validation",
        row_count=len(ids),
        ordered_stable_id_sha256=ordered_stable_id_digest(ids),
        stream_cache_sha256="c" * 64,
    ), ids


def _primary_bundle():
    rows = _primary_rows()
    evidence = [_evidence(start_sample=r.start_sample) for r in rows]
    return V.build_primary_bundle(
        "M2-G", evidence, rows, primary_population=_primary_authority(rows)
    )


def _challenge_bundle():
    rows = _challenge_rows()
    evidence = [
        _evidence(start_sample=r.start_sample, record_id="s00002") for r in rows
    ]
    return V.build_challenge_bundle(
        "M2-G", evidence, rows, challenge_population=_challenge_authority(rows)
    )


def _module_calls(path: Path, function: str) -> set[str]:
    """Names of the functions whose bodies call `function`."""
    tree = ast.parse(Path(path).read_text())
    callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                target = inner.func
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == function:
                    callers.add(node.name)
    return callers


# --------------------------------------------------------------------------
# 1-3. The populations are distinct concepts, and never share a row
# --------------------------------------------------------------------------


def test_full_replay_and_primary_are_distinct_concepts():
    assert PP.POPULATION_REPLAY != PP.POPULATION_PRIMARY
    assert PP.REPLAY_AUTHORITY != PP.PRIMARY_AUTHORITY
    replay, _ids = _replay_authority(_primary_rows())
    primary = _primary_bundle().population_identity()
    assert replay.identity()["population"] == PP.POPULATION_REPLAY
    assert primary["population"] == PP.POPULATION_PRIMARY
    assert replay.identity()["is_metric_denominator"] is False
    assert replay.identity() != primary


def test_full_replay_and_challenge_are_distinct_concepts():
    assert PP.POPULATION_REPLAY != PP.POPULATION_CHALLENGE
    assert PP.REPLAY_AUTHORITY != PP.CHALLENGE_AUTHORITY
    replay, _ids = _replay_authority(_challenge_rows())
    challenge = _challenge_bundle().population_identity()
    assert replay.identity() != challenge
    assert challenge["binary_labels_invented"] is False


def test_primary_and_challenge_cannot_share_a_stable_id():
    primary = _primary_rows()
    # A challenge row whose identity collides with a primary row.
    colliding = [
        V.M2ChallengeAnnotation(
            record_id=primary[0].record_id,
            channel_index=primary[0].channel_index,
            start_sample=primary[0].start_sample,
            target_family="rate_related_confounder",
            subject_id="subj-0",
        )
    ]
    replay, ids = _replay_authority(primary)
    with pytest.raises(PP.M2PopulationError, match="BOTH the primary and challenge"):
        PP.prove_population_containment(
            replay_population=replay,
            replay_stable_ids=ids,
            primary=_primary_authority(primary),
            challenge=_challenge_authority(colliding),
        )


def test_metric_rows_must_live_inside_the_full_replay_population():
    primary = _primary_rows()
    replay, ids = _replay_authority(primary[:2])
    with pytest.raises(PP.M2PopulationError, match="absent from the full replay"):
        PP.prove_population_containment(
            replay_population=replay,
            replay_stable_ids=ids,
            primary=_primary_authority(primary),
            challenge=_challenge_authority(_challenge_rows()),
        )


def test_containment_refuses_replay_ids_that_do_not_match_the_token():
    primary = _primary_rows()
    replay, ids = _replay_authority(primary)
    with pytest.raises(PP.M2PopulationError, match="not the canonical replay identity"):
        PP.prove_population_containment(
            replay_population=replay,
            replay_stable_ids=[*ids, "ltstdb:sXXXXX:0:0:2500"],
            primary=_primary_authority(primary),
            challenge=_challenge_authority(_challenge_rows()),
        )


# --------------------------------------------------------------------------
# 4-7. Authorities and label semantics
# --------------------------------------------------------------------------


def test_primary_authority_is_the_frozen_validation_membership_not_m2_scores():
    signature = inspect.signature(PP.verify_primary_population)
    assert not (set(signature.parameters) & {"scores", "decisions", "evidence"})
    identity = _primary_bundle().population_identity()
    assert identity["authority"] == PP.PRIMARY_AUTHORITY
    assert identity["membership_derived_from_m2_scores"] is False
    # The frozen loader binds the P1 validation cache, not an M2 artifact.
    source = inspect.getsource(PP.primary_evaluation_population)
    assert "load_p1_embedding_cache" in source


def test_primary_authority_refuses_counts_that_differ_from_the_frozen_identity():
    rows = _primary_rows()
    with pytest.raises(PP.M2PopulationError, match="never adjusted to fit"):
        PP.verify_primary_population(
            stable_ids=[r.stable_id for r in rows],
            labels=[int(r.label) for r in rows],
            subject_ids=[str(r.subject_id) for r in rows],
            cache_sha256="a" * 64,
        )


def test_challenge_authority_is_the_frozen_validation_challenge_selection():
    from cardiosentinel.neural.validation_challenge import CHALLENGE_SELECTION_SHA256

    source = inspect.getsource(PP.challenge_evaluation_population)
    assert "build_validation_challenge_index" in source
    rows = _challenge_rows()
    with pytest.raises(PP.M2PopulationError, match="differs from the frozen identity"):
        PP.verify_challenge_population(
            stable_ids=[r.stable_id for r in rows],
            target_families=[str(r.target_family) for r in rows],
            subject_ids=[str(r.subject_id) for r in rows],
            selection_sha256="z" * 64,
            counts={},
            expected_selection_sha256=CHALLENGE_SELECTION_SHA256,
        )


def test_primary_bundle_requires_binary_labels():
    assert "label" in V.M2PrimaryAnnotation.__dataclass_fields__
    with pytest.raises(V.M2EvaluationError, match="binary label 0 or 1"):
        V.M2PrimaryAnnotation(
            record_id="s00001",
            channel_index=0,
            start_sample=0,
            label=2,
            subject_id="subj-0",
            cold_start_bin="over_60_minutes",
        )
    assert "labels" in V.M2PrimaryAnnotationTable.__dataclass_fields__


def test_challenge_bundle_does_not_invent_binary_labels():
    assert "label" not in V.M2ChallengeAnnotation.__dataclass_fields__
    assert "labels" not in V.M2ChallengeAnnotationTable.__dataclass_fields__
    bundle = _challenge_bundle()
    assert not hasattr(bundle, "labels")
    assert bundle.population_identity()["binary_labels_invented"] is False


def test_the_two_annotation_types_are_not_interchangeable():
    primary_rows = _primary_rows(2)
    challenge_rows = _challenge_rows(2)
    with pytest.raises(V.M2EvaluationError, match="M2PrimaryAnnotation"):
        V.M2PrimaryAnnotationTable.from_rows(challenge_rows)
    with pytest.raises(V.M2EvaluationError, match="M2ChallengeAnnotation"):
        V.M2ChallengeAnnotationTable.from_rows(primary_rows)


# --------------------------------------------------------------------------
# 8-12. Each metric accepts exactly ONE population
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        V.window_evidence,
        V.background_false_positive_evidence,
        V.cold_start_stratified_evidence,
    ],
)
def test_primary_only_metrics_refuse_the_challenge_population(function):
    with pytest.raises(V.M2EvaluationError, match="PRIMARY metric population"):
        function(_challenge_bundle())


def test_challenge_fpr_refuses_the_primary_population():
    with pytest.raises(V.M2EvaluationError, match="CHALLENGE metric population"):
        V.challenge_false_positive_evidence(_primary_bundle())


def test_p1_challenge_evidence_is_reachable_only_from_the_challenge_path():
    """§7 -- p1_challenge_evidence is never run over the primary population."""
    callers = _module_calls(NEURAL / "m2_evaluation.py", "p1_challenge_evidence")
    assert callers == {"challenge_false_positive_evidence"}
    guard = _module_calls(NEURAL / "m2_evaluation.py", "require_challenge_bundle")
    assert "challenge_false_positive_evidence" in guard


def test_subject_false_positive_evidence_is_reachable_only_from_the_primary_path():
    callers = _module_calls(
        NEURAL / "m2_evaluation.py", "subject_false_positive_evidence"
    )
    assert callers == {"background_false_positive_evidence"}


def test_false_alarm_section_carries_two_named_denominators():
    """§8 -- one section, two explicit subsections, each with its own identity."""
    payload = V.false_alarm_evidence(
        primary_bundle=_primary_bundle(), challenge_bundle=_challenge_bundle()
    )
    assert set(payload) >= {"background_and_subject_fpr", "challenge_fpr"}
    background = payload["background_and_subject_fpr"]["population_identity"]
    challenge = payload["challenge_fpr"]["population_identity"]
    assert background["population"] == PP.POPULATION_PRIMARY
    assert challenge["population"] == PP.POPULATION_CHALLENGE
    assert background != challenge
    assert payload["single_denominator_served_both"] is False


def test_cold_start_is_computed_over_primary_rows_not_the_full_timeline():
    payload = V.cold_start_stratified_evidence(_primary_bundle())
    assert payload["population"] == PP.POPULATION_PRIMARY
    assert payload["post_hoc_early_threshold_defined"] is False
    assert "0-5 minute bin" in payload["inherited_limitation"]
    from cardiosentinel.neural.patient_memory import COLD_START_BINS

    assert set(payload["strata"]) == {name for name, _l, _h in COLD_START_BINS}


# --------------------------------------------------------------------------
# 13-15. Policy, trajectory and stress binding
# --------------------------------------------------------------------------


def test_policy_evidence_binds_the_full_replay_population():
    rows = _primary_rows()
    replay, _ids = _replay_authority(rows)
    evidence = [_evidence(start_sample=r.start_sample) for r in rows]
    payload = V.policy_evidence(evidence, replay_population=replay)
    assert payload["population"] == PP.POPULATION_REPLAY
    assert payload["population_identity"] == replay.identity()
    assert payload["classification_threshold_used_for_admission"] is False


def test_prototype_contamination_binds_the_replay_trajectory_and_the_selection():
    from cardiosentinel.neural.m2_evidence import PrototypeTrajectory

    rows = _primary_rows()
    replay, _ids = _replay_authority(rows)
    selection = SI.build_stress_selection()
    payload = V.contamination_evidence(
        {
            ("s00001", 0): PrototypeTrajectory(
                times=np.array([10.0, 20.0]), prototypes=np.zeros((2, 4))
            )
        },
        stress_intervals=[],
        replay_population=replay,
        stress_selection_identity=selection.identity(),
    )
    assert payload["trajectory_population"] == PP.POPULATION_REPLAY
    assert payload["replay_population_identity"] == replay.identity()
    assert payload["stress_interval_selection_identity"] == selection.identity()
    assert payload["trajectory_produced_label_blind"] is True
    assert payload["annotations_applied_after_replay"] is True


def test_stress_intervals_are_selected_only_after_replay():
    """The selector is post-replay by module boundary and by declaration."""
    firewall = X.assert_label_firewall()
    assert "m2_stress_intervals" in firewall["post_replay_modules"]
    identity = SI.build_stress_selection().identity()
    assert identity["selection_performed_after_label_blind_replay"] is True
    assert identity["selection_influenced_by_m2_outputs"] is False
    # And the runner constructs the selection only after both replays finish.
    source = inspect.getsource(R._run)
    replay_call = source.index("replay_both_arms(\n")
    assert replay_call < source.index("build_stress_selection_from_parsed(")
    assert replay_call < source.index("primary_evaluation_population(")
    assert replay_call < source.index("challenge_evaluation_population(")


# --------------------------------------------------------------------------
# 16-22. The canonical DEVELOPMENT partition firewall
# --------------------------------------------------------------------------


def test_canonical_development_partition_is_exactly_validation():
    assert X.CANONICAL_DEVELOPMENT_PARTITION == "validation"
    assert X.require_canonical_development_partition("validation") == "validation"
    assert PP.DEVELOPMENT_PARTITION == "validation"


def test_train_cannot_become_a_claim_bearing_development_result():
    with pytest.raises(X.M2ExecutionError, match="never become a claim-bearing"):
        X.require_canonical_development_partition("train")
    # And the permissive smoke firewall was NOT widened to admit validation.
    assert X.PARTITIONS_PERMITTED_HERE == ("train",)
    with pytest.raises(X.M2ExecutionError, match="not permitted"):
        X.require_permitted_partition("validation")


def test_the_train_smoke_remains_non_claim_bearing():
    assert X.SMOKE_ARTIFACT_CLASS == "NON_CLAIM_BEARING_TRAIN_INTEGRATION_SMOKE"
    source = inspect.getsource(X.train_integration_smoke)
    assert "NOT a scientific run" in source
    assert X.SMOKE_ARTIFACT_CLASS != PS.ARM_RESULT_CLASS


def test_test_is_hard_rejected_before_any_path_resolution():
    for guard in (
        X.require_permitted_partition,
        X.require_canonical_development_partition,
    ):
        with pytest.raises(X.M2ExecutionError, match="sealed test|hard-reject"):
            guard("test")
        with pytest.raises(X.M2ExecutionError):
            guard("TEST")
    # No sealed-test utility is importable from the canonical route.
    for module in (X, R):
        imports = X._module_imports(Path(module.__file__))
        assert not any("sealed_test" in name for name in imports), module


def test_canonical_development_provenance_requires_validation_accessed_true():
    result = _complete_result()
    result["validation_accessed"] = False
    with pytest.raises(PS.M2PersistenceError, match="validation_accessed=true"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_canonical_development_provenance_requires_test_accessed_false():
    result = _complete_result()
    result["test_accessed"] = True
    with pytest.raises(PS.M2PersistenceError, match="test_accessed=false"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_sealed_test_state_must_remain_unopened():
    result = _complete_result()
    result["sealed_test_state"] = "opened"
    with pytest.raises(PS.M2PersistenceError, match="sealed test must remain unopened"):
        PS.validate_claim_bearing_arm_result_payload(result)


# --------------------------------------------------------------------------
# 23-28. The canonical runner
# --------------------------------------------------------------------------


def test_the_runner_exposes_no_arbitrary_partition_argument():
    parser = R.build_parser()
    options = {flag for action in parser._actions for flag in action.option_strings}
    assert options == {"-h", "--help", R.EXECUTION_FLAG, R.EXPECTED_GIT_SHA_FLAG}
    assert not any("partition" in flag for flag in options)
    assert "partition" not in set(
        inspect.signature(R.execute_canonical_development).parameters
    )


def test_no_public_single_arm_canonical_scientific_route_exists():
    public = {
        name
        for name in dir(R)
        if not name.startswith("_") and callable(getattr(R, name))
    }
    assert not any("single" in name or "one_arm" in name for name in public)
    assert "arm" not in set(
        inspect.signature(R.execute_canonical_development).parameters
    )


def test_both_arms_are_fixed_and_ordered():
    assert R.CANONICAL_ARM_ORDER == ("M2-0", "M2-G")
    assert tuple(P.M2_ARMS) == R.CANONICAL_ARM_ORDER
    assert len(R.CANONICAL_ARM_ORDER) == 2
    R._assert_frozen_arm_order()


def test_one_arms_result_cannot_influence_the_other_arms_replay():
    """The replay loop consumes only frozen rows, never an arm's evidence."""
    source = inspect.getsource(R.replay_both_arms)
    tree = ast.parse(inspect.getsource(R.replay_both_arms).lstrip())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (
        called
        & {
            "window_evidence",
            "false_alarm_evidence",
            "cold_start_stratified_evidence",
            "contamination_evidence",
            "build_primary_bundle",
            "build_challenge_bundle",
        }
    )
    # Each arm gets its own state and its own store; nothing crosses over.
    assert "for arm in CANONICAL_ARM_ORDER" in source
    assert "stores[arm].add_stream" in source


def test_the_runner_requires_an_expected_git_sha():
    with pytest.raises(R.M2DevelopmentRunError, match="is required"):
        R.require_expected_git_sha(None)
    with pytest.raises(R.M2DevelopmentRunError, match="is required"):
        R.require_expected_git_sha("")
    parser = R.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([R.EXECUTION_FLAG])


def test_a_git_mismatch_stops_before_any_data_access(monkeypatch):
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance",
        lambda _root: {"git_sha": "a" * 40, "git_dirty": False},
    )
    with pytest.raises(R.M2DevelopmentRunError, match="BEFORE any data access"):
        R.require_expected_git_sha("b" * 40)


def test_a_dirty_checkout_stops_before_any_data_access(monkeypatch):
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance",
        lambda _root: {"git_sha": "a" * 40, "git_dirty": True},
    )
    with pytest.raises(R.M2DevelopmentRunError, match="No data was opened"):
        R.require_expected_git_sha("a" * 40)


def test_the_runner_does_not_execute_on_import():
    source = ast.parse(Path(R.__file__).read_text())
    top_level_calls = [
        node
        for node in source.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert top_level_calls == []
    assert R.NO_SCIENTIFIC_EXECUTION_YET is True


# --------------------------------------------------------------------------
# 29-30. Bounded memory and the disk-backed evidence store
# --------------------------------------------------------------------------


def test_bounded_replay_retains_no_corpus_scale_row_objects(tmp_path):
    store = ES.M2EvidenceStore(root=tmp_path / "M2-G", arm="M2-G")
    for index in range(3):
        key = ("s00001", index)
        rows = [_evidence(start_sample=r * 1250, channel=index) for r in range(4)]
        store.add_stream(key, rows, [(float(r), np.zeros(4)) for r in range(4)])
        del rows

    import dataclasses

    def _attributes(obj):
        return [getattr(obj, f.name) for f in dataclasses.fields(obj)]

    assert not any(isinstance(v, P.M2RowEvidence) for v in _attributes(store))
    for column in _attributes(store._scores):
        assert all(isinstance(item, np.ndarray) for item in column)
        assert not any(isinstance(item, P.M2RowEvidence) for item in column)
    table = store.score_table()
    assert isinstance(table.scores, np.ndarray)
    assert table.scores.dtype == np.float64
    assert table.row_count == 12


def test_the_streaming_admission_summary_equals_the_frozen_one(tmp_path):
    from cardiosentinel.neural.m2_evidence import summarize_admission

    rows = [_evidence(start_sample=index * 1250) for index in range(6)]
    store = ES.M2EvidenceStore(root=tmp_path / "M2-G", arm="M2-G")
    store.add_stream(("s00001", 0), rows[:3], [(0.0, np.zeros(4))])
    store.add_stream(("s00001", 1), rows[3:], [(0.0, np.zeros(4))])
    streamed = store.admission.summary()
    frozen = summarize_admission(rows)
    for key, value in frozen.items():
        assert streamed[key] == value, key


def test_the_evidence_store_binds_its_schema_and_content_digest(tmp_path):
    root = tmp_path / "M2-G"
    store = ES.M2EvidenceStore(root=root, arm="M2-G")
    store.add_stream(
        ("s00001", 0),
        [_evidence(start_sample=0)],
        [(10.0, np.array([1.0, 2.0, 3.0, 4.0]))],
    )
    manifest = store.finalize()
    assert manifest["schema"] == ES.EVIDENCE_STORE_SCHEMA
    assert manifest["lossy_conversion_applied"] is False
    assert manifest["labels_present"] is False
    assert manifest["trajectory_points_selected_by_annotation"] is False
    ES.validate_evidence_store_manifest(manifest, root=root)

    # A tampered manifest and a tampered file are both refused.
    with pytest.raises(ES.M2EvidenceStoreError, match="failed digest validation"):
        ES.validate_evidence_store_manifest({**manifest, "row_count": 999})
    (root / ES.ROW_EVIDENCE_NAME).write_bytes(b"tampered")
    with pytest.raises(ES.M2EvidenceStoreError, match="row evidence does not match"):
        ES.validate_evidence_store_manifest(manifest, root=root)


def test_prototype_precision_survives_the_disk_round_trip(tmp_path):
    prototypes = np.array([np.pi, np.e, 1e-17, -1234.56789012345], dtype=np.float64)
    store = ES.M2EvidenceStore(root=tmp_path / "M2-G", arm="M2-G")
    store.add_stream(
        ("s00001", 0), [_evidence(start_sample=0)], [(0.1234567890123, prototypes)]
    )
    trajectory = store.load_trajectory(("s00001", 0))
    assert trajectory.prototypes.dtype == np.float64
    assert np.array_equal(trajectory.prototypes[0], prototypes)
    assert trajectory.times[0] == 0.1234567890123


def test_an_unscored_row_is_refused_by_the_compact_score_table(tmp_path):
    store = ES.M2EvidenceStore(root=tmp_path / "M2-G", arm="M2-G")
    store.add_stream(
        ("s00001", 0), [_evidence(start_sample=0, score=None)], [(0.0, np.zeros(4))]
    )
    table = store.score_table()
    with pytest.raises(ES.M2EvidenceStoreError, match="STOP FOR HUMAN REVIEW"):
        table.scores_for(["ltstdb:s00001:0:0:2500"])


# --------------------------------------------------------------------------
# 31-32. Result and lock coherence
# --------------------------------------------------------------------------


def _complete_result(arm="M2-G"):
    """A complete canonical arm result carrying four distinct populations."""
    replay = X.M2ReplayPopulation(
        partition="validation",
        row_count=473_897,
        ordered_stable_id_sha256="1" * 64,
        stream_cache_sha256="2" * 64,
    ).identity()
    primary = {
        "population": PP.POPULATION_PRIMARY,
        "partition": "validation",
        "authority": PP.PRIMARY_AUTHORITY,
        "row_count": PP.PRIMARY_VALIDATION_POPULATION["total"],
        "counts": dict(PP.PRIMARY_VALIDATION_POPULATION),
        "ordered_stable_id_sha256": "3" * 64,
        "membership_derived_from_m2_scores": False,
        "binary_labels_present": True,
        "evaluated_rows": PP.PRIMARY_VALIDATION_POPULATION["total"],
        "evaluated_ordered_stable_id_sha256": "3" * 64,
        "identity_key": "(record_id, channel_index, start_sample)",
        "positional_join_used": False,
        "matches_frozen_authority_exactly": True,
    }
    from cardiosentinel.neural.validation_challenge import (
        CHALLENGE_EXPECTED_COUNTS,
        CHALLENGE_SELECTION_SHA256,
        CHALLENGE_TOTAL_WINDOWS,
    )

    challenge = {
        "population": PP.POPULATION_CHALLENGE,
        "partition": "validation",
        "authority": PP.CHALLENGE_AUTHORITY,
        "row_count": CHALLENGE_TOTAL_WINDOWS,
        "counts": {k: dict(v) for k, v in CHALLENGE_EXPECTED_COUNTS.items()},
        "challenge_selection_sha256": CHALLENGE_SELECTION_SHA256,
        "binary_labels_invented": False,
        "evaluated_rows": CHALLENGE_TOTAL_WINDOWS,
        "evaluated_ordered_stable_id_sha256": "5" * 64,
        "identity_key": "(record_id, channel_index, start_sample)",
        "positional_join_used": False,
        "matches_frozen_authority_exactly": True,
    }
    stress = SI.build_stress_selection().identity()
    return {
        "artifact_class": PS.ARM_RESULT_CLASS,
        "arm": arm,
        "scientific_computation_completed": True,
        "label_blind_replay_completed": True,
        "m1l_classification_threshold": (
            __import__(
                "cardiosentinel.neural.m2_scorer", fromlist=["x"]
            ).M1L_CLASSIFICATION_THRESHOLD
        ),
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
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
        "policy_evidence": {"population_identity": replay},
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


def test_result_population_identities_are_mutually_coherent():
    assert PS.validate_claim_bearing_arm_result_payload(_complete_result())
    # Swapping a section's denominator is fatal.
    result = _complete_result()
    result["window_evidence"] = {
        "population_identity": result["challenge_evaluation_population_identity"]
    }
    with pytest.raises(PS.M2PersistenceError, match="differs from"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_one_population_cannot_stand_in_for_another():
    """The replay identity can never be presented as the primary denominator."""
    result = _complete_result()
    result["primary_evaluation_population_identity"] = result[
        "replay_population_identity"
    ]
    with pytest.raises(PS.M2PersistenceError, match="must declare"):
        PS.validate_claim_bearing_arm_result_payload(result)
    # The four identities in a valid result are genuinely distinct.
    valid = _complete_result()
    identities = [valid[f] for f in PS.POPULATION_IDENTITY_FIELDS]
    for index, first in enumerate(identities):
        for second in identities[index + 1 :]:
            assert first != second


def test_the_lock_binds_all_four_population_identities():
    assert PS.POPULATION_IDENTITY_FIELDS == (
        "replay_population_identity",
        "primary_evaluation_population_identity",
        "challenge_evaluation_population_identity",
        "stress_interval_selection_identity",
    )
    for field in PS.POPULATION_IDENTITY_FIELDS:
        assert field in PS.REQUIRED_PROVENANCE_FIELDS
    assert "evaluated_population_identity" not in PS.REQUIRED_PROVENANCE_FIELDS


def test_the_stress_identity_binds_the_frozen_human_decision():
    result = _complete_result()
    result["stress_interval_selection_identity"] = {
        **result["stress_interval_selection_identity"],
        "decision_sha256": "z" * 64,
    }
    with pytest.raises(PS.M2PersistenceError, match="drifted apart"):
        PS.validate_claim_bearing_arm_result_payload(result)


# --------------------------------------------------------------------------
# 33-36. Unchanged invariants, and no development access in this task
# --------------------------------------------------------------------------


def test_no_frozen_scientific_rule_changed():
    from cardiosentinel.neural import m2_gate as G
    from cardiosentinel.neural import m2_scorer as SC

    assert SC.M1L_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert SC.NORMAL_EVIDENCE_THRESHOLD == 0.0002997174742631614
    assert P.M2_ARMS == ("M2-0", "M2-G")
    assert G.validate_m2_protocol() == G.M2_PROTOCOL_SHA256
    assert G.validate_m2_gate_receipt() == G.M2_GATE_RECEIPT_SHA256
    SC.assert_thresholds_are_distinct()


def test_no_validation_or_test_data_is_opened_by_this_test_module():
    """§13/§11 -- the contract is proved from constants and synthetic rows."""
    forbidden = {
        "load_p1_embedding_cache",
        "build_validation_challenge_index",
        "assemble_timeline_rows",
        "iter_timeline_streams",
        "canonical_replay_population",
        "primary_evaluation_population",
        "challenge_evaluation_population",
        "read_annotations",
        "read_record",
        "load_stream_store",
        "execute_canonical_development",
        "parsed_validation_annotations",
        "train_integration_smoke",
    }
    # Checked on CALL nodes, not raw text: naming a function in an assertion
    # about it is not an invocation of it.
    tree = ast.parse(Path(__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (called & forbidden), sorted(called & forbidden)


def test_the_canonical_runner_was_not_invoked():
    assert R.NO_SCIENTIFIC_EXECUTION_YET is True
    assert R.PLANNED_EXECUTION_ORDER[0] == "pre_claim_identity_checks"
    assert R.PLANNED_EXECUTION_ORDER[-1] == "two_arm_suite_without_selection"


def test_no_retention_rollback_or_arm_selection_is_expressed():
    result = _complete_result()
    assert result["memory_selected"] is None
    assert result["memory_selection_performed"] is False
    assert result["rollback"] is False
    suite = PS.build_suite_result(
        suite_id="synthetic",
        arm_results={"M2-0": {"arm": "M2-0"}, "M2-G": {"arm": "M2-G"}},
    )
    assert suite["memory_selected"] is None
    assert suite["automatic_arm_preference_applied"] is False
    assert suite["human_review_required"] is True
    assert suite["test_accessed"] is False
    assert suite["sealed_test_state"] == "unopened"
