"""The assembled canonical one-shot route, executed end to end on fixtures.

The population-contract tests prove the components. This file proves the
ASSEMBLED route actually runs: the previous revision could not start at all,
because both arms claimed one shared directory and the CLI could not supply the
roots `_run()` demanded.

Every data and scientific loader is replaced by a synthetic fixture through the
runner's private TEST-ONLY injection seam, and the REAL orchestration function
executes. **No real VALIDATION data is opened**, no `.stb` is read, no P1 cache
or challenge selection is rebuilt, no metric is computed on development data and
the sealed TEST partition is untouched.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_development_run as R
from cardiosentinel.neural import m2_evaluation as V
from cardiosentinel.neural import m2_execution as X
from cardiosentinel.neural import m2_persistence as PS
from cardiosentinel.neural import m2_policy as P
from cardiosentinel.neural import m2_populations as PP
from cardiosentinel.neural import m2_scorer as SC
from cardiosentinel.neural import runtime_sentinel as S
from cardiosentinel.neural.m2_gate_derivation import DEFAULT_P1_CACHE_ROOT
from cardiosentinel.neural.patient_memory import (
    OBSERVATION_AVAILABLE,
    REPRESENTATION_DIM,
    M1DistanceStandardizer,
)
from tests.neural.m2_attempt1_fixtures import _plant_frozen_attempt1

FROZEN_DIGEST = "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
SUITE = R.CANONICAL_SUITE_ID
GIT_SHA = "0" * 40

RECORDS = (("s00001", 0), ("s00001", 1), ("s00002", 0))
ROWS_PER_STREAM = 6


# --------------------------------------------------------------------------
# Synthetic frozen-runtime seam (identical convention to the harness tests)
# --------------------------------------------------------------------------


def _frozen_check(point, detail="test"):
    return S.RuntimeCheck(
        enforcement_point=S.EnforcementPoint(point).value,
        observed_digest=FROZEN_DIGEST,
        expected_digest=FROZEN_DIGEST,
        matches=True,
        package_count=335,
        observed_at="2026-01-01T00:00:00Z",
        detail=detail,
    )


@pytest.fixture()
def frozen_runtime(monkeypatch):
    """Drive the real production path with synthetic frozen observations."""

    def fake_observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        return _frozen_check(point, detail or "test")

    def fake_require(point, *, record=None, detail=None):
        check = _frozen_check(point, detail or "test")
        if record is not None:
            record.record(check)
        return check

    for module in (PS, R):
        monkeypatch.setattr(
            module, "observe_runtime_identity", fake_observe, raising=False
        )
    monkeypatch.setattr(PS, "require_runtime_identity", fake_require)
    monkeypatch.setattr(
        PS, "git_provenance", lambda _root: {"git_sha": GIT_SHA, "git_dirty": False}
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.runtime_sentinel.require_runtime_identity", fake_require
    )
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance",
        lambda _root: {"git_sha": GIT_SHA, "git_dirty": False},
    )
    return fake_require


@pytest.fixture()
def synthetic_frozen_populations(monkeypatch):
    """Substitute the frozen population identities for a 12-row fixture corpus.

    A TEST-ONLY seam. The real 473,897-row primary identity and the real
    challenge selection digest cannot be met by a synthetic corpus, and the
    production validators rightly refuse anything else -- proved separately by
    the population-contract tests, which assert the real frozen values. Here the
    expectation is replaced in ONE place so the token builder and the persistence
    validator agree, and the assembled ROUTE is what is under test.
    """
    primary_counts = {"total": 12, "positive": 6, "negative": 6, "subjects": 2}
    challenge_counts = {
        family: {"windows": 2, "subjects": 2} for family in PP.CHALLENGE_FAMILIES
    }
    monkeypatch.setattr(PP, "PRIMARY_VALIDATION_POPULATION", primary_counts)
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_EXPECTED_COUNTS",
        challenge_counts,
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_SELECTION_SHA256",
        "d" * 64,
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_TOTAL_WINDOWS", 6
    )
    return primary_counts


# --------------------------------------------------------------------------
# Synthetic scientific fixtures
# --------------------------------------------------------------------------


class _Scorer:
    """A deterministic stand-in for the frozen M1L scorer."""

    def __call__(self, representation, d_long):
        """`[raw frozen 146-d z_t ; pre-update d_long(t)]`, exactly as frozen."""
        vector = np.concatenate(
            [np.asarray(representation, dtype=np.float64).ravel(), [float(d_long)]]
        )
        return float(np.clip(np.abs(vector).mean() / 10.0, 0.0, 1.0))

    def identity(self):
        return {
            "retained_lock_sha256": SC.RETAINED_M1L_LOCK_SHA256,
            "retained_checkpoint_sha256": SC.RETAINED_M1L_CHECKPOINT_SHA256,
            "p1b_lock_sha256": SC.FROZEN_P1B_LOCK_SHA256,
            "b4b_checkpoint_sha256": SC.FROZEN_B4B_CHECKPOINT_SHA256,
            "classification_threshold": SC.M1L_CLASSIFICATION_THRESHOLD,
            "memory_admission_threshold": SC.NORMAL_EVIDENCE_THRESHOLD,
            "classification_threshold_used_for_memory_admission": False,
        }


def _standardizer():
    return M1DistanceStandardizer(
        means=tuple([0.0] * REPRESENTATION_DIM),
        scales=tuple([1.0] * REPRESENTATION_DIM),
        prior=tuple([0.0] * REPRESENTATION_DIM),
        zero_variance_dimensions=(),
        fitted_rows=1,
        fitted_population="train",
        input_identities={"partition": "train"},
    )


def _timeline_rows(key):
    record_id, channel = key
    return [
        P.M2TimelineRow(
            record_id=record_id,
            channel_index=channel,
            start_sample=index * 1250,
            observation_state=OBSERVATION_AVAILABLE,
            representation=np.full(REPRESENTATION_DIM, 0.1 * (index + 1)),
            finite_sample_fraction=1.0,
            sqi=dict.fromkeys(_sqi_columns(), 0.0),
            morphology_valid=1.0,
        )
        for index in range(ROWS_PER_STREAM)
    ]


def _sqi_columns():
    from cardiosentinel.neural import m2_gate as G

    return G.G3_SQI_COLUMNS


def _stable_id(key, index):
    record_id, channel = key
    start = index * 1250
    return V.stable_id_for_key((record_id, channel, start))


def _all_stable_ids():
    return [_stable_id(key, i) for key in RECORDS for i in range(ROWS_PER_STREAM)]


def _stream_source():
    for key in RECORDS:
        yield key, _timeline_rows(key)


def _replay_authority(_stream_cache_root):
    from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

    ids = _all_stable_ids()
    population = X.M2ReplayPopulation(
        partition="validation",
        row_count=len(ids),
        ordered_stable_id_sha256=ordered_stable_id_digest(sorted(ids)),
        stream_cache_sha256="c" * 64,
    )
    manifest = {
        "partition": "validation",
        "full_stream_row_count": len(ids),
        "ordered_stable_id_sha256": population.ordered_stable_id_sha256,
        "stream_cache_sha256": "c" * 64,
        "ordered_chronology_sha256": "e" * 64,
        "split_sha256": "f" * 64,
        "feature_corpus_sha256": "0" * 64,
        "representation_dim": REPRESENTATION_DIM,
        "distance_standardizer_sha256": "1" * 64,
    }
    return population, sorted(ids), manifest


def _primary_ids():
    """The first four rows of each stream are the primary metric population."""
    return [_stable_id(key, i) for key in RECORDS for i in range(4)]


def _challenge_ids():
    """The last two rows of each stream are challenge confounders. Disjoint."""
    return [_stable_id(key, i) for key in RECORDS for i in (4, 5)]


def _primary_population(_p1_cache_root):
    ids = _primary_ids()
    labels = [index % 2 for index in range(len(ids))]
    subjects = [f"subj-{index % 2}" for index in range(len(ids))]
    return PP.verify_primary_population(
        stable_ids=ids,
        labels=labels,
        subject_ids=subjects,
        cache_sha256="a" * 64,
        expected_counts={
            "total": len(ids),
            "positive": sum(labels),
            "negative": len(ids) - sum(labels),
            "subjects": len(set(subjects)),
        },
    )


def _challenge_population(_feature_root):
    ids = _challenge_ids()
    families = [
        PP.CHALLENGE_FAMILIES[index % len(PP.CHALLENGE_FAMILIES)]
        for index in range(len(ids))
    ]
    subjects = [f"subj-{index % 2}" for index in range(len(ids))]
    counts = {
        family: {
            "windows": sum(1 for f in families if f == family),
            "subjects": len(
                {s for f, s in zip(families, subjects, strict=True) if f == family}
            ),
        }
        for family in PP.CHALLENGE_FAMILIES
    }
    return PP.verify_challenge_population(
        stable_ids=ids,
        target_families=families,
        subject_ids=subjects,
        selection_sha256="d" * 64,
        counts=counts,
        expected_selection_sha256="d" * 64,
        expected_counts=counts,
        expected_total=len(ids),
    )


def _primary_annotations(primary, *, stream_cache_root):
    return V.M2PrimaryAnnotationTable(
        stable_ids=np.asarray(primary.stable_ids, dtype=np.str_),
        labels=np.asarray(primary.labels, dtype=np.int64),
        subject_ids=np.asarray(primary.subject_ids, dtype=np.str_),
        cold_start_bins=np.asarray(
            ["over_60_minutes"] * primary.row_count, dtype=np.str_
        ),
    )


def _source_identity(*, source_root, feature_root):
    return {
        "identity_class": "m2_v1_development_source_integrity",
        "feature_receipt": {"verification_result": "passed"},
        "source_receipt": {"verification_result": "passed"},
        "annotation_set": "stb",
        "test_partition_hashed": False,
        "verified_before_stress_selection": True,
    }


class _ParsedAnnotations:
    """One synthetic record's `.stb` semantics: one eligible ST episode."""

    def __init__(self, record_id, channel):
        from cardiosentinel.data.models import AnnotationMarker, STEvent

        self.events = (
            STEvent(
                dataset_id="ltstdb",
                record_id=record_id,
                subject_id="subj-0",
                lead=channel,
                event_family="st_episode",
                event_subtype="ischemic",
                onset_sample=1250,
                peak_sample=2500,
                end_sample=3750,
                onset_seconds=5.0,
                peak_seconds=10.0,
                end_seconds=15.0,
                peak_deviation_uv=-150.0,
                direction="depression",
                annotation_source="stb",
                annotation_definition="ltstdb.stb",
                is_primary_definition=True,
                original_annotations=(),
            ),
        )
        self.quality_intervals = ()
        self.markers = (
            AnnotationMarker(
                record_id=record_id,
                subject_id="subj-0",
                lead=channel,
                sample=2000,
                category="st_shift",
                subtype="axis_related",
                annotation_source="stb",
                original_annotation=None,
            ),
        )
        self.source_censored_intervals = ()


def _parsed_annotations(*, source_root, feature_root):
    for record_id, channel in RECORDS:
        yield _ParsedAnnotations(record_id, channel)


def _loaders(**overrides):
    base = {
        "load_frozen_m1l_scorer": lambda _root: _Scorer(),
        "load_distance_standardizer": lambda _root: _standardizer(),
        "stream_source": _stream_source(),
        "canonical_replay_population": _replay_authority,
        "primary_evaluation_population": _primary_population,
        "challenge_evaluation_population": _challenge_population,
        "build_primary_annotations": _primary_annotations,
        "parsed_validation_annotations": _parsed_annotations,
        "verify_development_source": _source_identity,
    }
    base.update(overrides)
    return base


def _roots(tmp_path):
    # The recovery route verifies attempt #1 from artifacts before it runs, so
    # a synthetic run stands up a byte-identical original attempt first.
    _plant_frozen_attempt1(tmp_path / "runs")
    return {
        "source_root": tmp_path / "source",
        "feature_root": tmp_path / "features",
        "stream_cache_root": tmp_path / "stream-cache",
        "p1_cache_root": tmp_path / "p1-cache",
        "m1_run_root": tmp_path / "m1-run",
        "run_root": tmp_path / "runs",
    }


def _execute(tmp_path, **overrides):
    return R.execute_canonical_development(
        expected_git_sha=GIT_SHA,
        execute=True,
        _suite_id=SUITE,
        _roots=_roots(tmp_path),
        _loaders=_loaders(**overrides),
    )


# --------------------------------------------------------------------------
# §12 -- the assembled route actually runs, end to end
# --------------------------------------------------------------------------


def test_the_canonical_route_executes_end_to_end(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    result = _execute(tmp_path)

    assert result["executed"] is True
    assert result["arms"] == ["M2-0", "M2-G"]
    assert result["memory_selected"] is None
    assert result["memory_selection_performed"] is False

    runs = _roots(tmp_path)["run_root"]
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = runs / PS.arm_experiment_id(SUITE, arm)
        assert (run_dir / PS.ARM_RESULT_NAME).is_file()
        assert (run_dir / PS.EXPERIMENT_LOCK_NAME).is_file()
        promoted = json.loads((run_dir / PS.ARM_RESULT_NAME).read_text())
        assert promoted["arm"] == arm
        assert promoted["partition_accessed"] == "validation"
        assert promoted["validation_accessed"] is True
        assert promoted["test_accessed"] is False
        assert promoted["sealed_test_state"] == "unopened"
        assert promoted["memory_selected"] is None
        for field in PS.POPULATION_IDENTITY_FIELDS:
            assert promoted[field]

    suite_path = PS.suite_directory(runs, SUITE) / PS.SUITE_RESULT_NAME
    assert suite_path.is_file()
    suite = json.loads(suite_path.read_text())
    PS.validate_suite_result(suite)
    assert sorted(suite["arm_results"]) == ["M2-0", "M2-G"]


def test_execution_history_reports_the_completed_run(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    runs = _roots(tmp_path)["run_root"]
    before = R.canonical_execution_history(runs, SUITE)
    assert before["any_attempt_claimed"] is False
    assert before["suite_result_promoted"] is False

    _execute(tmp_path)

    after = R.canonical_execution_history(runs, SUITE)
    assert after["any_attempt_claimed"] is True
    assert after["suite_result_promoted"] is True
    for arm in R.CANONICAL_ARM_ORDER:
        assert after["arms"][arm]["result_promoted"] is True
        assert after["arms"][arm]["lock_promoted"] is True
        assert after["arms"][arm]["status"] == PS.STATUS_COMPLETE


def test_both_claims_exist_before_the_validation_loader_is_called(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§2 -- validation is opened only after BOTH arm claims succeed."""
    runs = _roots(tmp_path)["run_root"]
    observed: dict[str, list[str]] = {"claims_at_replay_population": []}

    def watching_replay_authority(stream_cache_root):
        observed["claims_at_replay_population"] = sorted(
            p.name for p in runs.iterdir() if p.is_dir()
        )
        return _replay_authority(stream_cache_root)

    _execute(tmp_path, canonical_replay_population=watching_replay_authority)

    claimed = observed["claims_at_replay_population"]
    for arm in R.CANONICAL_ARM_ORDER:
        assert PS.arm_experiment_id(SUITE, arm) in claimed


def test_the_two_arms_replay_the_identical_frozen_population(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """One shared full replay population; the arms differ only by policy."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    identities = [
        json.loads(
            (runs / PS.arm_experiment_id(SUITE, arm) / PS.ARM_RESULT_NAME).read_text()
        )["replay_population_identity"]
        for arm in R.CANONICAL_ARM_ORDER
    ]
    assert identities[0] == identities[1]


def test_no_test_path_or_partition_is_ever_requested(
    tmp_path, frozen_runtime, synthetic_frozen_populations, monkeypatch
):
    """A real spy: every partition and path the route asks for is recorded.

    The previous version of this test carried an unconditional `or True`, so it
    could not fail. This one refuses the run outright if any component asks for
    the test partition, a test path or a sealed-test helper.
    """
    requested_partitions: list[str] = []
    opened_paths: list[str] = []

    real_guard = X.require_canonical_development_partition

    def spy_partition(partition):
        requested_partitions.append(str(partition))
        return real_guard(partition)

    monkeypatch.setattr(X, "require_canonical_development_partition", spy_partition)
    monkeypatch.setattr(R, "require_canonical_development_partition", spy_partition)

    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        opened_paths.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)

    result = _execute(tmp_path)
    monkeypatch.undo()

    assert result["partition"] == "validation"
    # Only the canonical partition is ever requested for real, plus the
    # firewall's own self-check that "test" is still refused.
    assert set(requested_partitions) <= {"validation", "test", "TEST", " test "}
    assert "validation" in requested_partitions
    for offender in ("test", "TEST"):
        # ...and every such request RAISED rather than resolving a path.
        with pytest.raises(X.M2ExecutionError):
            real_guard(offender)
    assert not any("/test/" in path or path.endswith("/test") for path in opened_paths)
    assert not any("sealed" in path.lower() for path in opened_paths)

    runs = _roots(tmp_path)["run_root"]
    for arm in R.CANONICAL_ARM_ORDER:
        promoted = json.loads(
            (runs / PS.arm_experiment_id(SUITE, arm) / PS.ARM_RESULT_NAME).read_text()
        )
        assert promoted["sealed_test_state"] == "unopened"
        assert promoted["test_accessed"] is False


# --------------------------------------------------------------------------
# §13 A-D -- claim collision and pre-existing attempts
# --------------------------------------------------------------------------


def test_a_shared_suite_id_does_not_collide_between_arms(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.A -- the blocker: both arms used to claim one directory."""
    first = PS.arm_experiment_id(SUITE, "M2-0")
    second = PS.arm_experiment_id(SUITE, "M2-G")
    assert first != second
    assert first.startswith(SUITE) and second.startswith(SUITE)
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    assert (runs / first).is_dir()
    assert (runs / second).is_dir()


@pytest.mark.parametrize("existing", ["M2-0", "M2-G"])
def test_a_pre_existing_arm_claim_prevents_validation_access(
    tmp_path, frozen_runtime, synthetic_frozen_populations, existing
):
    """§13.B/C -- a consumed attempt stops the run before any data is read."""
    roots = _roots(tmp_path)
    (roots["run_root"] / PS.arm_experiment_id(SUITE, existing)).mkdir(parents=True)
    opened = []

    def forbidden(_root):
        opened.append("validation")
        raise AssertionError("validation must not be opened")

    # The recovery precondition fires before claim checking, which is earlier
    # still than the pair preflight -- either way, nothing is opened.
    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="already|consumed"
    ):
        _execute(tmp_path, canonical_replay_population=forbidden)
    assert opened == []


def test_a_pre_existing_suite_result_prevents_validation_access(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.D -- a completed suite is never re-run."""
    roots = _roots(tmp_path)
    PS.suite_directory(roots["run_root"], SUITE).mkdir(parents=True)
    opened = []

    def forbidden(_root):
        opened.append("validation")
        raise AssertionError("validation must not be opened")

    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="already|consumed"
    ):
        _execute(tmp_path, canonical_replay_population=forbidden)
    assert opened == []


def test_a_pre_existing_evidence_workspace_is_refused(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.N -- a generic workspace never silently reuses another attempt."""
    roots = _roots(tmp_path)
    PS.evidence_workspace(roots["run_root"], SUITE).mkdir(parents=True)
    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="already|consumed"
    ):
        _execute(tmp_path)


def test_the_evidence_workspace_belongs_to_this_suite(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§9 -- deterministic, suite-owned, outside the immutable arm results."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    workspace = PS.evidence_workspace(runs, SUITE)
    assert workspace.is_dir()
    assert workspace.name.startswith(SUITE)
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = runs / PS.arm_experiment_id(SUITE, arm)
        assert workspace != run_dir
        assert run_dir not in workspace.parents


# --------------------------------------------------------------------------
# §13 E-H -- the CLI and the readiness ordering
# --------------------------------------------------------------------------


def test_the_cli_reaches_orchestration_with_only_the_two_flags(monkeypatch):
    """§13.E -- the advertised CLI must be able to run what it advertises."""
    seen: dict[str, object] = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"preflight_class": "x", "partition": "validation", "arms": ["M2-0"]}

    # Captured BEFORE patching, so the real public contract is inspected.
    parameters = inspect.signature(R.execute_canonical_development).parameters
    monkeypatch.setattr(R, "execute_canonical_development", fake)
    assert R.main([R.EXECUTION_FLAG, R.EXPECTED_GIT_SHA_FLAG, GIT_SHA]) == 0
    assert seen == {"expected_git_sha": GIT_SHA, "execute": True}
    # No root, id or workspace argument is required from the operator.
    required = {
        name
        for name, value in parameters.items()
        if value.default is inspect.Parameter.empty
    }
    # Only the deliberate authorization control is required; the CLI supplies
    # it. No root, id or workspace argument is demanded of the operator.
    assert required == {"expected_git_sha"}
    for name in ("run_root", "experiment_id", "evidence_root", "dataset_root"):
        assert name not in parameters, name


def test_canonical_roots_are_deterministic_and_complete():
    roots = R.canonical_roots()
    assert set(roots) == {
        "source_root",
        "feature_root",
        "stream_cache_root",
        "p1_cache_root",
        "m1_run_root",
        "run_root",
    }
    assert R.canonical_roots() == roots


def test_main_dispatch_is_after_every_runtime_helper():
    """§13.F -- module execution cannot enter the run with an undefined helper."""
    tree = ast.parse(Path(R.__file__).read_text())
    definitions = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    dispatch = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.If) and ast.dump(node.test).find("__main__") != -1
    ]
    assert dispatch, "no __main__ dispatch found"
    assert max(definitions.values()) < dispatch[0]
    for required in (
        "main",
        "_run",
        "parsed_validation_annotations",
        "verify_development_source",
        "replay_both_arms",
        "pre_claim_readiness",
        "canonical_roots",
    ):
        assert definitions[required] < dispatch[0], required


def test_scorer_and_locks_are_verified_before_any_validation_loader(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.G -- a bad checkpoint must not consume two canonical attempts."""
    roots = _roots(tmp_path)

    def bad_scorer(_root):
        class _Bad(_Scorer):
            def identity(self):
                payload = super().identity()
                payload["retained_checkpoint_sha256"] = "z" * 64
                return payload

        return _Bad()

    opened = []

    def forbidden(_root):
        opened.append("validation")
        raise AssertionError("validation must not be opened")

    with pytest.raises(R.M2DevelopmentRunError, match="retained_checkpoint_sha256"):
        _execute(
            tmp_path,
            load_frozen_m1l_scorer=bad_scorer,
            canonical_replay_population=forbidden,
        )
    assert opened == []
    # And no RECOVERY arm was claimed. (The run root also holds the planted
    # original attempt, which must remain untouched.)
    for arm in R.CANONICAL_ARM_ORDER:
        assert not (roots["run_root"] / PS.arm_experiment_id(SUITE, arm)).exists()
    assert not PS.evidence_workspace(roots["run_root"], SUITE).exists()


def test_protocol_receipt_and_decision_digests_are_verified_before_validation():
    """§13.H -- the frozen identity checks are in the pre-claim stage."""
    source = inspect.getsource(R.preflight)
    assert "validate_m2_protocol" in source
    assert "validate_m2_gate_receipt" in source
    assert "DECISION_SHA256" in source
    readiness = inspect.getsource(R.pre_claim_readiness)
    assert readiness.index("    readiness = preflight(") < readiness.index(
        "    claim_check = require_unclaimed_suite("
    )
    run = inspect.getsource(R._run)
    readiness_call = run.index("    readied = pre_claim_readiness(")
    claim_call = run.index("        claims[arm] = claim_run_directory(")
    replay_call = run.index("    replay_population, replay_stable_ids, manifest =")
    assert readiness_call < claim_call < replay_call


def test_readiness_result_records_that_validation_stayed_closed():
    source = inspect.getsource(R.pre_claim_readiness)
    assert '"validation_opened_during_readiness": False' in source


# --------------------------------------------------------------------------
# §13 I -- the frozen P1 cache root
# --------------------------------------------------------------------------


def test_the_p1_cache_root_is_the_frozen_p1_root_not_the_stream_cache():
    """§7/§13.I -- they are different artifacts and must never be confused."""
    roots = R.canonical_roots()
    assert roots["p1_cache_root"] == DEFAULT_P1_CACHE_ROOT
    assert roots["p1_cache_root"].name == "p1-b4b-embeddings-v1"
    assert roots["stream_cache_root"].name == "m1-stream-memory-v2"
    assert roots["p1_cache_root"] != roots["stream_cache_root"]
    # There is no fallback from one to the other anywhere on the route.
    source = inspect.getsource(R._run)
    assert "p1_cache_root or stream_cache_root" not in source
    assert 'roots["p1_cache_root"]' in source


def test_the_primary_population_is_loaded_from_the_p1_root(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    seen: dict[str, Path] = {}

    def watching_primary(p1_cache_root):
        seen["root"] = Path(p1_cache_root)
        return _primary_population(p1_cache_root)

    _execute(tmp_path, primary_evaluation_population=watching_primary)
    assert seen["root"] == _roots(tmp_path)["p1_cache_root"]
    assert seen["root"] != _roots(tmp_path)["stream_cache_root"]


# --------------------------------------------------------------------------
# §13 J-L -- development source integrity
# --------------------------------------------------------------------------


def test_source_integrity_is_verified_before_stress_selection(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.J -- the raw `.stb` is proven to be the official frozen source."""
    order: list[str] = []

    def watching_source(*, source_root, feature_root):
        order.append("source_integrity")
        return _source_identity(source_root=source_root, feature_root=feature_root)

    def watching_annotations(*, source_root, feature_root):
        order.append("stb_read")
        yield from _parsed_annotations(
            source_root=source_root, feature_root=feature_root
        )

    _execute(
        tmp_path,
        verify_development_source=watching_source,
        parsed_validation_annotations=watching_annotations,
    )
    assert order[0] == "source_integrity"
    assert "stb_read" in order


def test_the_real_verifier_uses_the_existing_development_integrity_machinery():
    source = inspect.getsource(R.verify_development_source)
    assert "validate_development_feature_integrity" in source
    assert "validate_development_source_integrity" in source
    assert '"test_partition_hashed": False' in source


def test_source_integrity_is_bound_in_arm_provenance_and_suite(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.K/L -- the receipt appears in the arm result, lock and suite."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = runs / PS.arm_experiment_id(SUITE, arm)
        promoted = json.loads((run_dir / PS.ARM_RESULT_NAME).read_text())
        assert promoted["development_source_identity"]["annotation_set"] == "stb"
        lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
        stress = lock["stress_interval_selection_identity"]
        assert stress["development_source_identity"]["annotation_set"] == "stb"
    suite = json.loads(
        (PS.suite_directory(runs, SUITE) / PS.SUITE_RESULT_NAME).read_text()
    )
    assert suite["development_source_identity"]["annotation_set"] == "stb"


# --------------------------------------------------------------------------
# §13 M -- one trajectory at a time
# --------------------------------------------------------------------------


def test_contamination_loads_at_most_one_trajectory_at_a_time():
    """§8/§13.M -- prototype matrices are never held for several streams."""
    from cardiosentinel.neural.m2_evidence import PrototypeTrajectory

    intervals = [
        V.M2StressInterval(
            record_id=record_id,
            channel_index=channel,
            family="ischemic",
            start_time=5.0,
            end_time=15.0,
        )
        for record_id, channel in RECORDS
    ]
    resident: list[tuple[str, int]] = []
    peak = {"value": 0}

    def load(key):
        resident.append(key)
        peak["value"] = max(peak["value"], len(resident))
        # The previous key must already have been released.
        assert len(resident) == 1, resident
        trajectory = PrototypeTrajectory(
            times=np.array([1.0, 20.0, 400.0, 2000.0]),
            prototypes=np.zeros((4, REPRESENTATION_DIM)),
        )
        resident.clear()
        return trajectory

    payload = V.streaming_contamination_evidence(
        stress_intervals=intervals, load_trajectory=load
    )
    assert peak["value"] == 1
    assert payload["trajectories_loaded"] == "one_stream_at_a_time"
    assert len(payload["intervals"]) == len(intervals)


def test_streaming_and_whole_dict_contamination_agree_exactly():
    """The streaming form introduces no numerical approximation."""
    from cardiosentinel.neural.m2_evidence import PrototypeTrajectory

    key = ("s00001", 0)
    trajectory = PrototypeTrajectory(
        times=np.array([1.0, 20.0, 400.0, 2000.0]),
        prototypes=np.array(
            [[0.0, 0.0], [0.5, -0.25], [1.5, -0.75], [2.0, -1.0]], dtype=np.float64
        ),
    )
    interval = V.M2StressInterval(
        record_id=key[0],
        channel_index=key[1],
        family="ischemic",
        start_time=5.0,
        end_time=100.0,
    )
    whole = V.contamination_evidence({key: trajectory}, stress_intervals=[interval])
    streamed = V.streaming_contamination_evidence(
        stress_intervals=[interval], load_trajectory=lambda _k: trajectory
    )
    assert whole["intervals"] == streamed["intervals"]


def test_the_route_uses_the_streaming_contamination_form():
    source = inspect.getsource(R._run)
    assert "streaming_contamination_evidence(" in source
    assert "load_trajectory=stores[arm].load_trajectory" in source


# --------------------------------------------------------------------------
# §13 O-S -- evidence store revalidation and the suite
# --------------------------------------------------------------------------


def test_evidence_store_files_are_revalidated_after_finalization(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.O -- the manifest is checked against the actual persisted files."""
    source = inspect.getsource(R._run)
    assert "validate_evidence_store_manifest(" in source
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    for arm in R.CANONICAL_ARM_ORDER:
        promoted = json.loads(
            (runs / PS.arm_experiment_id(SUITE, arm) / PS.ARM_RESULT_NAME).read_text()
        )
        store = promoted["evidence_store_identity"]
        assert store["lossy_conversion_applied"] is False
        assert store["prototype_dtype"] == "float64"
        assert store["labels_present"] is False
        assert store["annotations_present"] is False


def test_build_suite_result_is_on_the_canonical_route():
    """§13.P -- the promise in PLANNED_EXECUTION_ORDER is actually kept."""
    source = inspect.getsource(R._run)
    assert "build_suite_body(" in source
    assert "finalize_and_promote_suite_result(" in source
    assert "two_arm_suite_without_selection" in R.PLANNED_EXECUTION_ORDER
    # The body is built unsigned; only finalization signs it, after the
    # suite's own PRE_PROMOTION observation exists.
    assert "build_suite_result(" not in source


def test_the_suite_contains_exactly_the_two_arms_and_selects_nothing(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.Q/R."""
    result = _execute(tmp_path)
    suite = result["suite"]
    assert sorted(suite["arm_results"]) == ["M2-0", "M2-G"]
    assert suite["arms"] == ["M2-0", "M2-G"]
    assert suite["memory_selected"] is None
    assert suite["memory_selection_performed"] is False
    assert suite["automatic_arm_preference_applied"] is False
    assert suite["new_scientific_metric_computed"] is False
    assert suite["rollback_evaluated"] is False
    assert suite["human_review_required"] is True
    for field in PS.POPULATION_IDENTITY_FIELDS:
        assert suite[field]
    assert set(suite["arm_experiment_lock_sha256"]) == {"M2-0", "M2-G"}


def test_the_suite_has_its_own_pre_promotion_observation(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§13.S -- never a reused arm observation."""
    observed: list[str] = []

    def recording_observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        observed.append(str(detail))
        return _frozen_check(point, detail or "test")

    import cardiosentinel.neural.m2_persistence as module

    original = module.observe_runtime_identity
    module.observe_runtime_identity = recording_observe
    try:
        _execute(tmp_path)
    finally:
        module.observe_runtime_identity = original

    assert f"promote:{PS.SUITE_RESULT_NAME}" in observed
    assert observed.count(f"promote:{PS.SUITE_RESULT_NAME}") == 1
    assert observed.count(f"promote:{PS.ARM_RESULT_NAME}") == 2


def test_an_incomplete_arm_yields_no_canonical_suite(tmp_path, frozen_runtime):
    runs = _roots(tmp_path)["run_root"]
    (runs / PS.arm_experiment_id(SUITE, "M2-0")).mkdir(parents=True)
    with pytest.raises(PS.M2PersistenceError, match="not COMPLETE"):
        PS.finalize_and_promote_suite_result(
            runs,
            SUITE,
            suite_body=_suite_body(),
            runtime=_suite_runtime(),
        )


def _suite_body(**overrides):
    body = PS.build_suite_body(
        suite_id=SUITE,
        arm_results={
            arm: {
                "experiment_id": PS.arm_experiment_id(SUITE, arm),
                "run_dir": "x",
                "arm_result_sha256": "a" * 64,
            }
            for arm in PS.M2_ARMS
        },
        arm_lock_sha256=dict.fromkeys(PS.M2_ARMS, "b" * 64),
        population_identities={
            field: {"x": index}
            for index, field in enumerate(PS.POPULATION_IDENTITY_FIELDS)
        },
        development_source_identity=_source_identity(source_root="", feature_root=""),
        recovery_lineage=R.recovery_lineage(),
        git_sha=GIT_SHA,
    )
    body.update(overrides)
    return body


def _suite_runtime():
    record = S.RuntimeIntegrityRecord()
    record.record(_frozen_check(S.EnforcementPoint.START.value))
    return record


# --------------------------------------------------------------------------
# §13 T-U -- firewalls and no real development access
# --------------------------------------------------------------------------


def test_test_remains_rejected_before_any_path_resolution():
    """§13.T."""
    for guard in (
        X.require_permitted_partition,
        X.require_canonical_development_partition,
    ):
        with pytest.raises(X.M2ExecutionError):
            guard("test")
    imports = X._module_imports(Path(R.__file__))
    assert not any("sealed_test" in name for name in imports)


def test_no_real_development_data_is_opened_by_this_module():
    """§13.U -- every scientific loader is injected, none is called for real."""
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
        "load_frozen_m1l_scorer",
        "validate_development_source_integrity",
        "validate_development_feature_integrity",
    }
    tree = ast.parse(Path(__file__).read_text())
    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not (called & forbidden), sorted(called & forbidden)


def test_the_injection_seam_is_private_and_absent_from_the_cli():
    parameters = set(inspect.signature(R.execute_canonical_development).parameters)
    assert {"_roots", "_loaders", "_suite_id"} <= parameters
    assert all(name.startswith("_") for name in ("_roots", "_loaders", "_suite_id"))
    # And no PUBLIC suite selector exists.
    assert "suite_id" not in parameters
    options = {
        flag for action in R.build_parser()._actions for flag in action.option_strings
    }
    assert options == {"-h", "--help", R.EXECUTION_FLAG, R.EXPECTED_GIT_SHA_FLAG}
    for banned in ("partition", "arm", "threshold", "retry", "seed", "source", "root"):
        assert not any(banned in flag for flag in options), banned


# --------------------------------------------------------------------------
# §1/§7.1-3 -- the production suite id is immutable
# --------------------------------------------------------------------------


def test_no_public_suite_id_override_exists():
    """§7.1 -- production execution cannot choose another canonical suite."""
    parameters = inspect.signature(R.execute_canonical_development).parameters
    assert "suite_id" not in parameters
    assert "_suite_id" in parameters
    with pytest.raises(TypeError):
        R.execute_canonical_development(
            expected_git_sha=GIT_SHA, execute=True, suite_id="alternate"
        )


def test_a_noncanonical_suite_id_is_refused_before_anything_happens(tmp_path):
    """§1 -- refused before claim checking, filesystem creation and VALIDATION."""
    with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
        R.require_canonical_suite_id("alternate")
    assert R.require_canonical_suite_id(R.CANONICAL_SUITE_ID) == R.CANONICAL_SUITE_ID
    # Nothing was created by the refusal.
    assert not (tmp_path / "runs").exists()


def test_a_consumed_recovery_cannot_be_rerun_under_a_second_name(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """An alternate name cannot bypass a consumed canonical attempt."""
    roots = _roots(tmp_path)
    (roots["run_root"] / PS.arm_experiment_id(R.CANONICAL_SUITE_ID, "M2-0")).mkdir(
        parents=True
    )
    with pytest.raises(
        (R.M2DevelopmentRunError, PS.M2PersistenceError), match="already|consumed"
    ):
        R._run(
            expected_git_sha=GIT_SHA,
            suite_id=R.require_canonical_suite_id(R.CANONICAL_SUITE_ID),
            roots=roots,
            loaders=_loaders(),
        )
    # There is no public route that would let a caller pick another name, and
    # every alternate name is refused outright.
    assert (
        "suite_id" not in inspect.signature(R.execute_canonical_development).parameters
    )
    for name in ("m2-v1-development-two-arm-recovery2", "attempt3"):
        with pytest.raises(R.M2DevelopmentRunError, match="is refused"):
            R.require_canonical_suite_id(name)


def test_the_test_seam_cannot_alter_production_semantics(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.3 -- `_suite_id` only names a synthetic suite; production is pinned."""
    result = _execute(tmp_path)
    assert result["suite_id"] == SUITE
    # The promoted suite is validated WITHOUT the production id expectation
    # only because this is a synthetic run; production passes it.
    finalize = inspect.getsource(R._run)
    assert "expected_suite_id=CANONICAL_SUITE_ID if production else None" in finalize
    execute = inspect.getsource(R.execute_canonical_development)
    assert "production=_suite_id is None" in execute


# --------------------------------------------------------------------------
# §2/§7.4-7 -- the suite persists its own runtime promotion evidence
# --------------------------------------------------------------------------


def _promoted_suite(tmp_path):
    runs = _roots(tmp_path)["run_root"]
    return json.loads(
        (PS.suite_directory(runs, SUITE) / PS.SUITE_RESULT_NAME).read_text()
    )


def test_the_suite_persists_its_start_and_pre_promotion_observations(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.4/§7.5."""
    _execute(tmp_path)
    suite = _promoted_suite(tmp_path)
    checks = suite["runtime_identity_checks"]
    observed = {
        (item["enforcement_point"], item["detail"]) for item in checks["checks"]
    }
    assert ("start", "m2_suite") in observed
    assert ("pre_promotion", f"promote:{PS.SUITE_RESULT_NAME}") in observed
    assert checks["expected_digest"] == FROZEN_DIGEST
    assert checks["all_observations_matched"] is True


def test_the_suite_digest_covers_its_runtime_block(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.6 -- the signature is taken AFTER the promotion observation exists."""
    _execute(tmp_path)
    suite = _promoted_suite(tmp_path)
    from cardiosentinel.neural.integrity import canonical_sha256

    body = {k: v for k, v in suite.items() if k != "m2_suite_sha256"}
    assert suite["m2_suite_sha256"] == canonical_sha256(body)
    # Stripping the runtime block changes the digest, so it is genuinely covered.
    without = {k: v for k, v in body.items() if k != "runtime_identity_checks"}
    assert canonical_sha256(without) != suite["m2_suite_sha256"]
    # And an unsigned body is refused by finalization.
    with pytest.raises(PS.M2PersistenceError, match="must be unsigned"):
        PS.finalize_and_promote_suite_result(
            _roots(tmp_path)["run_root"],
            SUITE,
            suite_body=suite,
            runtime=_suite_runtime(),
        )


def test_a_suite_runtime_mismatch_prevents_promotion(tmp_path, monkeypatch):
    """§7.7 -- a non-frozen observation refuses the suite."""
    runs = tmp_path / "runs"
    runs.mkdir()

    def mismatched(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        check = _frozen_check(point, detail or "test")
        return S.RuntimeCheck(
            enforcement_point=check.enforcement_point,
            observed_digest="9" * 64,
            expected_digest=FROZEN_DIGEST,
            matches=False,
            package_count=1,
            observed_at=check.observed_at,
            detail=check.detail,
        )

    monkeypatch.setattr(PS, "observe_runtime_identity", mismatched)
    monkeypatch.setattr(PS, "_verify_suite_against_arm_artifacts", lambda *a, **k: None)
    with pytest.raises(S.RuntimeIntegrityError, match="NOT promoted"):
        PS.finalize_and_promote_suite_result(
            runs, SUITE, suite_body=_suite_body(), runtime=_suite_runtime()
        )
    assert not (PS.suite_directory(runs, SUITE)).exists()


def test_a_suite_without_a_runtime_block_is_rejected():
    suite = _suite_body()
    suite["runtime_identity_checks"] = None
    from cardiosentinel.neural.integrity import canonical_sha256

    suite["m2_suite_sha256"] = canonical_sha256(
        {k: v for k, v in suite.items() if k != "m2_suite_sha256"}
    )
    with pytest.raises(PS.M2PersistenceError, match="own runtime_identity_checks"):
        PS.validate_suite_result(suite)


def test_a_suite_reusing_an_arm_promotion_observation_is_rejected():
    """The suite's PRE_PROMOTION must name M2_SUITE_RESULT.json."""
    record = _suite_runtime()
    record.record(
        _frozen_check(
            S.EnforcementPoint.PRE_PROMOTION.value, f"promote:{PS.ARM_RESULT_NAME}"
        )
    )
    suite = _suite_body()
    suite["runtime_identity_checks"] = record.as_dict()
    from cardiosentinel.neural.integrity import canonical_sha256

    suite["m2_suite_sha256"] = canonical_sha256(
        {k: v for k, v in suite.items() if k != "m2_suite_sha256"}
    )
    with pytest.raises(PS.M2PersistenceError, match="never reused"):
        PS.validate_suite_result(suite)


# --------------------------------------------------------------------------
# §3/§7.8-16 -- the suite is verified against the ACTUAL arm artifacts
# --------------------------------------------------------------------------


def _signed(body):
    from cardiosentinel.neural.integrity import canonical_sha256

    record = _suite_runtime()
    record.record(
        _frozen_check(
            S.EnforcementPoint.PRE_PROMOTION.value, f"promote:{PS.SUITE_RESULT_NAME}"
        )
    )
    body = dict(body)
    body["runtime_identity_checks"] = record.as_dict()
    body["m2_suite_sha256"] = canonical_sha256(body)
    return body


@pytest.mark.parametrize("bad", [None, "short", "z" * 63, 12345])
def test_a_missing_or_malformed_arm_result_sha_is_rejected(bad):
    """§7.8/§7.9."""
    body = _suite_body()
    body["arm_results"]["M2-0"]["arm_result_sha256"] = bad
    with pytest.raises(PS.M2PersistenceError, match="arm_result_sha256"):
        PS.validate_suite_result(_signed(body))


@pytest.mark.parametrize("bad", [None, "short", "z" * 63])
def test_a_missing_or_malformed_lock_sha_is_rejected(bad):
    """§7.11."""
    body = _suite_body()
    body["arm_experiment_lock_sha256"]["M2-G"] = bad
    with pytest.raises(PS.M2PersistenceError, match="arm_experiment_lock_sha256"):
        PS.validate_suite_result(_signed(body))


def test_wrong_arm_experiment_ids_are_rejected():
    """§7.14."""
    body = _suite_body()
    body["arm_experiment_ids"]["M2-0"] = "something-else"
    with pytest.raises(PS.M2PersistenceError, match="deterministic"):
        PS.validate_suite_result(_signed(body))


def test_a_malformed_suite_git_sha_is_rejected():
    body = _suite_body()
    body["git_sha"] = "not-a-sha"
    with pytest.raises(PS.M2PersistenceError, match="git_sha is malformed"):
        PS.validate_suite_result(_signed(body))


def test_a_mutated_arm_result_byte_fails_suite_validation(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.10 -- the suite is the aggregation of the ACTUAL frozen arms."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    suite = _promoted_suite(tmp_path)
    PS.validate_suite_result(suite, run_root=runs)

    result_path = runs / PS.arm_experiment_id(SUITE, "M2-0") / PS.ARM_RESULT_NAME
    result_path.write_text(result_path.read_text() + " ")
    with pytest.raises(PS.M2PersistenceError, match="but the suite declares"):
        PS.validate_suite_result(suite, run_root=runs)


def test_an_altered_arm_lock_fails_suite_validation(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.12/§7.13."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    suite = _promoted_suite(tmp_path)

    lock_path = runs / PS.arm_experiment_id(SUITE, "M2-G") / PS.EXPERIMENT_LOCK_NAME
    lock = json.loads(lock_path.read_text())
    lock["experiment_lock_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(PS.M2PersistenceError, match="differs from the suite"):
        PS.validate_suite_result(suite, run_root=runs)


def test_the_suite_git_sha_must_equal_both_arm_locks(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.15."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    suite = _promoted_suite(tmp_path)
    for arm in R.CANONICAL_ARM_ORDER:
        lock = json.loads(
            (
                runs / PS.arm_experiment_id(SUITE, arm) / PS.EXPERIMENT_LOCK_NAME
            ).read_text()
        )
        assert lock["git_sha"] == suite["git_sha"]


def test_the_suite_population_identities_must_equal_both_arm_locks(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.16."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    suite = _promoted_suite(tmp_path)
    for arm in R.CANONICAL_ARM_ORDER:
        lock = json.loads(
            (
                runs / PS.arm_experiment_id(SUITE, arm) / PS.EXPERIMENT_LOCK_NAME
            ).read_text()
        )
        for field in PS.POPULATION_IDENTITY_FIELDS:
            assert lock[field] == suite[field], field


# --------------------------------------------------------------------------
# §4/§7.17-18 -- development-source identity coherence
# --------------------------------------------------------------------------


def test_source_identity_agrees_across_result_stress_lock_and_suite(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    """§7.17."""
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    suite = _promoted_suite(tmp_path)
    for arm in R.CANONICAL_ARM_ORDER:
        run_dir = runs / PS.arm_experiment_id(SUITE, arm)
        result = json.loads((run_dir / PS.ARM_RESULT_NAME).read_text())
        lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
        identity = result["development_source_identity"]
        assert (
            result["stress_interval_selection_identity"]["development_source_identity"]
            == identity
        )
        assert lock["development_source_identity"] == identity
        assert suite["development_source_identity"] == identity


@pytest.mark.parametrize(
    "mutation",
    [
        {"identity_class": "something_else"},
        {"annotation_set": "sta"},
        {"test_partition_hashed": True},
        {"verified_before_stress_selection": False},
        {"feature_receipt": {"verification_result": "failed"}},
        {"source_receipt": {"verification_result": "failed"}},
        {"feature_receipt": None},
    ],
)
def test_a_malformed_source_integrity_identity_is_rejected(mutation):
    """§7.18."""
    identity = {**_source_identity(source_root="", feature_root=""), **mutation}
    with pytest.raises(PS.M2PersistenceError):
        PS.validate_development_source_identity(identity)


def test_a_result_whose_stress_source_disagrees_is_rejected(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    _execute(tmp_path)
    runs = _roots(tmp_path)["run_root"]
    run_dir = runs / PS.arm_experiment_id(SUITE, "M2-0")
    result = json.loads((run_dir / PS.ARM_RESULT_NAME).read_text())
    result["stress_interval_selection_identity"] = {
        **result["stress_interval_selection_identity"],
        "development_source_identity": {
            **result["development_source_identity"],
            "annotation_set": "stb",
            "verified_before_stress_selection": True,
            "feature_receipt": {"verification_result": "passed", "extra": 1},
        },
    }
    with pytest.raises(PS.M2PersistenceError, match="differs from|same verified"):
        PS.validate_claim_bearing_arm_result_payload(result)


def test_no_assertion_in_this_module_is_vacuous():
    """No assertion may contain an unconditionally-true disjunct.

    Checked on the AST rather than by scanning text, because a literal scan
    would match this test's own description and pass for the wrong reason.
    """
    tree = ast.parse(Path(__file__).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if isinstance(test, ast.Constant) and bool(test.value):
            offenders.append(node.lineno)
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
            for value in test.values:
                if isinstance(value, ast.Constant) and bool(value.value):
                    offenders.append(node.lineno)
    assert offenders == [], offenders
