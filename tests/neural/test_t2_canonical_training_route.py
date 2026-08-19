"""The assembled canonical T2 route, driven end to end against synthetics.

This file exists because every primitive passing proved nothing about whether
the canonical route had a body. It drives `_execute_training_attempt` -- the
same orchestration `execute_canonical_training` calls after preflight -- against
a genuinely valid synthetic M1 stream cache and a genuinely valid synthetic
LTSTDB-shaped target authority, in a temporary run root.

**No real science happens here.** No canonical CLI invocation, no real TRAIN
optimiser step, no real internal-dev score, no real threshold, no real
VALIDATION row access, no challenge scientific result, no TEST. The only real
artifacts read are frozen manifests and frozen digests.

Two substitutions are made, both environmental rather than scientific and both
following the reviewed M2 canonical-runner convention:

* `git_provenance` is replaced with a clean fake, because a canonical lock
  refuses a dirty checkout and a developer's working tree is dirty by nature;
* the runtime-integrity observation is replaced with a frozen one, because the
  only interpreter that carries the frozen 335-package digest is
  `venvs/tactics` -- CI and a developer shell do not, and the sentinel is
  right to refuse them. Its real refusal behaviour is proved by its own tests
  and by `test_a_missing_enforcement_stage_is_refused` here; what this file
  tests is that the canonical body *visits* the enforcement points, in order.

Nothing else. The split, the target join, the frontier loop, the checkpoint
choreography, the threshold pass, the promotion and the lock are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import t2_development_run as RUN
from cardiosentinel.neural import t2_evaluation as EV
from cardiosentinel.neural import t2_persistence as PS
from cardiosentinel.neural import t2_timeline as TL
from cardiosentinel.neural import t2_training as TR
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.m1_store import (
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
)
from cardiosentinel.neural.patient_memory import M1MemoryError
from cardiosentinel.neural.runtime_sentinel import EnforcementPoint, RuntimeCheck
from cardiosentinel.neural.t2_models import build_t2_model
from cardiosentinel.neural.t2_protocol import (
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_ARMS,
    T2_INPUT_DIM,
    T2_INTERNAL_DEV_SUBJECTS,
    T2_TBPTT_LENGTH,
)
from tests.neural import t2_fixtures as FX

GIT_SHA = "a" * 40
FROZEN_DIGEST = "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"

# Both are pre-claim refusals from `require_expected_git_sha`, and which one
# fires depends on whether the checkout happens to be clean -- dirty in a
# working tree, clean in CI. The property under test is that the route refuses
# and opens nothing, not which of the two identity gates spoke first.
_PRE_CLAIM_REFUSAL = "but the run expects|working tree is dirty"


def _frozen_check(point, detail="test"):
    return RuntimeCheck(
        enforcement_point=EnforcementPoint(point).value,
        observed_digest=FROZEN_DIGEST,
        expected_digest=FROZEN_DIGEST,
        matches=True,
        package_count=335,
        observed_at="2026-01-01T00:00:00Z",
        detail=detail,
    )


@pytest.fixture()
def frozen_runtime(monkeypatch):
    """Drive the real production path with synthetic frozen observations.

    Identical convention to the reviewed M2 canonical-runner tests. The
    observation is faked; the choreography, the ordering requirement and every
    refusal built on the record are the real ones.
    """

    def fake_observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        return _frozen_check(point, detail or "test")

    def fake_require(point, *, record=None, detail=None):
        check = _frozen_check(point, detail or "test")
        if record is not None:
            record.record(check)
        return check

    monkeypatch.setattr(PS, "observe_runtime_identity", fake_observe)
    monkeypatch.setattr(PS, "require_runtime_identity", fake_require)
    monkeypatch.setattr(
        "cardiosentinel.neural.runtime_sentinel.require_runtime_identity", fake_require
    )

    def frozen_provenance(device=None):
        """The dependency digest is faked; the DEVICE is the real selection.

        Substituting the device too would defeat the whole point of the
        actual-device tests, so the declared device is whatever
        `canonical_execution_device` actually chose.
        """
        selected = torch.device(device) if device is not None else torch.device("cpu")
        return {
            "interpreter": "/home/AI_POC/venvs/tactics/bin/python",
            "python_version": "3.11.0",
            "package_count": 335,
            "dependency_digest": FROZEN_DIGEST,
            "torch_version": torch.__version__,
            "cuda_version": None,
            "declared_execution_device": str(selected),
            "device_type": selected.type,
            "device_index": selected.index,
            "device_name": None,
            "cuda_available": torch.cuda.is_available(),
            "device_override_permitted": False,
            "silent_cpu_fallback_permitted": False,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_threads": 1,
            "torch_interop_threads": 1,
        }

    monkeypatch.setattr(PS, "runtime_provenance", frozen_provenance)
    monkeypatch.setattr(RUN, "runtime_provenance", frozen_provenance)
    return fake_require


@pytest.fixture()
def clean_git(monkeypatch, frozen_runtime):
    """A clean checkout, so the canonical lock's Git gate is not the subject."""
    fake = {"git_sha": GIT_SHA, "git_dirty": False}
    # Every module that imported the name directly, not only the source module:
    # `t2_development_run.require_expected_git_sha` resolves `git_provenance`
    # in its OWN namespace, and the outer route goes through it.
    monkeypatch.setattr(PS, "git_provenance", lambda _root: dict(fake))
    monkeypatch.setattr(RUN, "git_provenance", lambda _root: dict(fake))
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance", lambda _root: dict(fake)
    )
    return fake


@pytest.fixture()
def environment(tmp_path):
    """The full 56-subject synthetic TRAIN environment, tiny per stream."""
    return FX.build_environment(tmp_path / "env")


def _sources(tmp_path, environment, **overrides):
    return RUN.T2TrainingSources(
        run_root=tmp_path / "runs",
        stream_cache_root=environment.stream_cache_root,
        corpus_manifest=environment.corpus_manifest,
        canonical=False,
        **overrides,
    )


def _checks():
    return {
        "preflight_class": "t2_training_preflight",
        "experiment_identity": PS.T2_EXPERIMENT_IDENTITY,
        "attempt_id": PS.T2_TRAINING_ATTEMPT_ID,
        "git_sha": GIT_SHA,
        "authorized_git_sha": GIT_SHA,
        "t2_protocol_sha256": PS.T2_PROTOCOL_SHA256,
        "t2_execution_spec_sha256": PS.T2_EXECUTION_SPEC_SHA256,
        "per_row_train_input_opened_before_claim": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


@pytest.fixture()
def completed_attempt(tmp_path, environment, clean_git):
    """One complete synthetic canonical attempt, run once and shared."""
    sources = _sources(tmp_path, environment)
    report = RUN._execute_training_attempt(_checks(), sources)
    return report, sources


# --- A. the assembled canonical TRAIN worker ------------------------------


def test_the_synthetic_full_attempt_reaches_complete(completed_attempt):
    report, sources = completed_attempt
    assert report["status"] == PS.STATUS_COMPLETE
    assert report["report_class"] == "t2_canonical_training_completion"
    run_dir = sources.run_root / sources.attempt_id
    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_COMPLETE
    assert status["canonical"] is True
    assert not (
        PS.t2_review_directory(sources.run_root, sources.attempt_id)
    ).exists(), "a completed attempt writes no failure receipt"


def test_both_arms_are_trained_and_both_checkpoints_exist(completed_attempt):
    report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    assert sorted(report["arm_results"]) == sorted(T2_ARMS)
    for arm in T2_ARMS:
        assert (run_dir / PS.CHECKPOINT_NAME[arm]).is_file()
        assert (run_dir / PS.CHECKPOINT_LOCK_NAME[arm]).is_file()
        assert (run_dir / PS.ARM_RESULT_NAME[arm]).is_file()


def test_both_thresholds_exist_and_no_arm_is_selected(completed_attempt):
    report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    result = json.loads((run_dir / PS.RESULT_NAME).read_text())
    assert sorted(result["internal_dev_thresholds"]) == sorted(T2_ARMS)
    for arm in T2_ARMS:
        evidence = result["internal_dev_thresholds"][arm]
        assert evidence["rule"] == "exact_maximum_f1_highest_threshold_tie_break"
        assert evidence["partition"] == "t2_internal_dev_8_subjects"
        assert evidence["is_outer_validation_evidence"] is False
        assert isinstance(evidence["threshold"], float)
    assert result["arm_selection_status"] == PS.ARM_SELECTION_PENDING
    assert result["arm_selected"] is None
    assert result["arm_compared_on_train_evidence"] is False
    assert report["arm_selected"] is None


def test_the_canonical_validator_accepts_the_completed_attempt(completed_attempt):
    _report, sources = completed_attempt
    verification = PS.validate_canonical_t2_attempt(
        sources.run_root, sources.attempt_id
    )
    assert verification["verified"] is True
    assert verification["checkpoint_locks_verified"] is True
    assert sorted(verification["checkpoint_sha256"]) == sorted(T2_ARMS)
    assert verification["arm_selection_status"] == PS.ARM_SELECTION_PENDING


def test_the_attempt_is_consumed_once_claimed(completed_attempt):
    _report, sources = completed_attempt
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        PS.require_unclaimed_t2_attempt(sources.run_root, sources.attempt_id)


# --- B. the public TRAIN route --------------------------------------------


def test_no_unconditional_not_wired_stop_remains():
    """The old body raised unconditionally after preflight. It must be gone."""
    source = Path(RUN.__file__).read_text()
    normalised = " ".join(source.split())
    for phrase in (
        "not wired",
        "not authorized to execute in this change set",
        "the training body is not",
    ):
        assert phrase not in normalised, phrase


def test_the_public_route_delegates_to_the_real_orchestration():
    """`execute_canonical_training` is preflight plus the same worker."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(RUN.execute_canonical_training))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "preflight" in called
    assert "_execute_training_attempt" in called
    assert "T2TrainingSources" in called
    raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    assert not raises, "the public route no longer stops unconditionally"


def test_the_public_route_carries_no_scientific_injection_parameter():
    import inspect

    parameters = inspect.signature(RUN.execute_canonical_training).parameters
    assert list(parameters) == ["expected_git_sha"]


def test_the_cli_exposes_no_scientific_knob_and_no_train_activation_switch():
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    assert not (options & set(RUN.FORBIDDEN_OPTIONS))
    assert options >= {
        "--execute-canonical-training",
        "--execute-canonical-outer-validation",
        "--expected-git-sha",
    }
    for module in (RUN, PS, TR, TL):
        source = Path(module.__file__).read_text()
        assert "T2_TRAIN_EXECUTION_AUTHORIZED" not in source
        assert "os.environ" not in source


def test_the_sources_seam_exposes_no_scientific_field():
    """Injectable: where to read. Not injectable: any frozen scientific choice."""
    fields = set(RUN.T2TrainingSources.__dataclass_fields__)
    assert fields == {
        "run_root",
        "attempt_id",
        "stream_cache_root",
        "corpus_manifest",
        "split_manifest",
        "canonical",
    }
    for forbidden in ("epochs", "seed", "lr", "tbptt", "threshold", "device"):
        assert not any(forbidden in name for name in fields)


# --- C. input bytes --------------------------------------------------------


@pytest.mark.parametrize(
    "array_file",
    [REPRESENTATION_FILE, STABLE_ID_FILE, START_SAMPLE_FILE],
)
def test_an_array_mutation_under_an_unchanged_manifest_is_refused(tmp_path, array_file):
    """The manifest still declares the pre-mutation digests. The bytes differ."""
    environment = FX.build_environment(
        tmp_path / "env", streams=FX.default_streams(FX.frozen_train_subjects()[:3])
    )
    # Proves the fixture is genuinely valid before the mutation, so the refusal
    # below is caused by the mutation and not by an invalid fixture.
    TL.T2Timeline("train", root=environment.stream_cache_root).close()
    FX.mutate_array_file(environment.partition_dir / array_file)
    with pytest.raises(M1MemoryError):
        TL.T2Timeline("train", root=environment.stream_cache_root)


def test_a_self_consistent_but_non_frozen_store_is_refused_on_the_canonical_path(
    tmp_path, monkeypatch
):
    """A perfectly valid store that is not the promoted one is still refused."""
    environment = FX.build_environment(
        tmp_path / "env", streams=FX.default_streams(FX.frozen_train_subjects()[:3])
    )
    # Opened with an explicit root it is accepted, and says so.
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        assert timeline.identity()["canonical_source"] is False
        assert timeline.identity()["persisted_bytes_revalidated"] is True
    # Opened as THE canonical source, the frozen identity gate fires.
    monkeypatch.setattr(TL, "STREAM_CACHE_ROOT", environment.stream_cache_root)
    with pytest.raises(TL.T2TimelineError, match="thinned by nothing at all"):
        TL.T2Timeline("train")


def test_the_frozen_stream_identity_gate_names_the_promoted_digests():
    identity = TL.require_frozen_stream_identity(
        "train",
        {
            "stream_cache_sha256": TL.EXPECTED_STREAM_CACHE_SHA256["train"],
            "representation_content_sha256": (
                TL.EXPECTED_REPRESENTATION_SHA256["train"]
            ),
        },
    )
    assert identity["stream_cache_sha256"] == (
        "d006c698017110bfd95774ca207036a820139779b95cf1b3f3a36c06efa779a4"
    )
    assert identity["representation_content_sha256"] == (
        "e52a566fbc285a7a9f92715752dee43c020faa3550aaeb660f5f400dee07b5d3"
    )


def test_the_byte_level_route_is_the_repositorys_strongest_m1_validator():
    source = Path(TL.__file__).read_text()
    assert "load_stream_store" in source
    with TL.T2Timeline.__init__.__globals__["Path"](TL.__file__).open() as handle:
        assert "m1_experiment import load_stream_store" in handle.read()


# --- D. the target join ----------------------------------------------------


def test_every_timeline_row_resolves_to_exactly_one_target_family(environment):
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        codes, lineage = TL.resolve_timeline_target_families(
            timeline, manifest_path=environment.corpus_manifest
        )
        assert codes.shape == (timeline.row_count,)
        assert int((codes == TL.FAMILY_CODE_UNRESOLVED).sum()) == 0
        assert lineage["row_count"] == timeline.row_count
        assert lineage["every_row_resolved_exactly_once"] is True
        assert lineage["raw_annotations_reread"] is False
        assert lineage["stb_reinterpreted"] is False
        assert lineage["labels_derived_from_context_flags"] is False
        assert lineage["target_family_is_model_input"] is False
        total = sum(lineage["target_family_counts"].values())
        assert total == timeline.row_count


def test_a_missing_target_row_is_refused(tmp_path, environment):
    record = environment.streams[0].record_id

    def drop_first(columns):
        for name, values in list(columns.items()):
            columns[name] = values[1:]

    FX.rewrite_record_cache(environment, record, mutate=drop_first)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="does not contain"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_a_duplicated_target_row_is_refused(environment):
    record = environment.streams[0].record_id

    def duplicate_first(columns):
        for name, values in list(columns.items()):
            columns[name] = np.concatenate([values, values[:1]])

    FX.rewrite_record_cache(environment, record, mutate=duplicate_first)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="repeats stable id"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_a_same_count_wrong_stable_id_target_is_refused(environment):
    """Equal row count is not identity. This one has the right length."""
    record = environment.streams[0].record_id

    def rename_ids(columns):
        columns["stable_ids"] = np.asarray(
            [f"{value}x" for value in columns["stable_ids"].tolist()], dtype=np.str_
        )

    FX.rewrite_record_cache(environment, record, mutate=rename_ids)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="does not contain"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_an_extra_target_row_is_never_silently_ignored(environment):
    record = environment.streams[0].record_id

    def append_extra(columns):
        for name, values in list(columns.items()):
            columns[name] = np.concatenate([values, values[-1:]])
        columns["stable_ids"] = np.asarray(
            columns["stable_ids"].tolist()[:-1] + ["ltstdb:zzzz:0:0:2500"],
            dtype=np.str_,
        )

    FX.rewrite_record_cache(environment, record, mutate=append_extra)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="never consumed"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_a_target_row_naming_a_different_window_is_refused(environment):
    record = environment.streams[0].record_id

    def shift_window(columns):
        starts = columns["window_start_samples"].copy()
        starts[0] = starts[0] + 7
        columns["window_start_samples"] = starts

    FX.rewrite_record_cache(environment, record, mutate=shift_window)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="but the timeline row"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_a_target_row_from_another_partition_is_refused(environment):
    record = environment.streams[0].record_id

    def repartition(columns):
        partitions = columns["partitions"].copy()
        partitions[0] = "validation"
        columns["partitions"] = partitions

    FX.rewrite_record_cache(environment, record, mutate=repartition)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="rows from partition"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_an_unknown_target_family_is_refused(environment):
    record = environment.streams[0].record_id

    def unknown_family(columns):
        families = columns["target_families"].astype("<U40").copy()
        families[0] = "invented_family"
        columns["target_families"] = families

    FX.rewrite_record_cache(environment, record, mutate=unknown_family)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        with pytest.raises(TL.T2TimelineError, match="carries family"):
            TL.resolve_timeline_target_families(
                timeline, manifest_path=environment.corpus_manifest
            )


def test_the_target_family_cannot_enter_the_model_input(environment):
    """The reader's `values` come from the store, never from the family codes."""
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        codes, _ = TL.resolve_timeline_target_families(
            timeline, manifest_path=environment.corpus_manifest
        )
        reader = TR.T2TimelineReader(timeline, codes)
        stream = timeline.streams()[0]
        view = reader.slice_view(0, stream, 0, stream.row_count)
        assert view.values.shape[1] == T2_INPUT_DIM

        # Flip every family code to its opposite class. The representation the
        # model would consume must not move by a single float.
        flipped = np.where(
            codes == TL.POSITIVE_FAMILY_CODE,
            TL.NEGATIVE_FAMILY_CODE,
            TL.POSITIVE_FAMILY_CODE,
        ).astype(np.uint8)
        other = TR.T2TimelineReader(timeline, flipped)
        flipped_view = other.slice_view(0, stream, 0, stream.row_count)
        assert np.array_equal(view.values, flipped_view.values)
        assert not np.array_equal(view.labels, flipped_view.labels)


def test_coded_roles_agree_with_the_string_role_assignment(environment):
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        codes, _ = TL.resolve_timeline_target_families(
            timeline, manifest_path=environment.corpus_manifest
        )
        states = timeline.observation_state(0, timeline.row_count)
        names = np.asarray(TL.FAMILY_NAME, dtype="<U32")[codes]
        expected = TL.assign_row_roles(categories=names, observation_state=states)
        observed = TL.role_names_for_codes(TL.role_codes_for_families(codes, states))
        assert np.array_equal(observed, expected)


# --- E. frontier optimisation ---------------------------------------------


class _CountingOptimizer:
    """Wraps the frozen optimiser and counts steps. Changes no arithmetic."""

    def __init__(self, inner):
        self.inner = inner
        self.steps = 0
        self.zeroed = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zeroed += 1
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.steps += 1
        self.inner.step()


def _reader_for(environment, timeline):
    codes, _ = TL.resolve_timeline_target_families(
        timeline, manifest_path=environment.corpus_manifest
    )
    return TR.T2TimelineReader(timeline, codes)


def test_one_optimizer_step_per_nonempty_frontier(tmp_path):
    """Two streams, several frontiers, all PRIMARY: steps == frontiers."""
    rows = T2_TBPTT_LENGTH * 2 + 5
    subjects = FX.frozen_train_subjects()[:2]
    streams = tuple(
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subject),
            channel_index=0,
            families=tuple(FX.PRIMARY_NEGATIVE for _ in range(rows)),
        )
        for subject in subjects
    )
    # One positive somewhere so the weighted loss is not degenerate.
    streams = (
        FX.SyntheticStream(
            record_id=streams[0].record_id,
            channel_index=0,
            families=(FX.PRIMARY_POSITIVE,) + streams[0].families[1:],
        ),
        streams[1],
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        model = build_t2_model(T2_ARM_GRU)
        optimizer = _CountingOptimizer(TR.build_optimizer(model))
        stats = TR.train_one_epoch(
            model, optimizer, reader, timeline.streams(), pos_weight=2.0
        )
        assert stats.frontier_count == 3
        assert stats.optimizer_step_count == 3
        assert optimizer.steps == 3
        assert optimizer.zeroed == 3
        assert stats.zero_direct_loss_frontier_count == 0


def test_several_length_groups_still_produce_exactly_one_step(tmp_path):
    """Streams of different lengths share a frontier and share one step."""
    subjects = FX.frozen_train_subjects()[:3]
    lengths = (T2_TBPTT_LENGTH, T2_TBPTT_LENGTH - 40, T2_TBPTT_LENGTH - 90)
    streams = tuple(
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subject),
            channel_index=0,
            families=(FX.PRIMARY_POSITIVE,)
            + tuple(FX.PRIMARY_NEGATIVE for _ in range(length - 1)),
        )
        for subject, length in zip(subjects, lengths, strict=True)
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        model = build_t2_model(T2_ARM_GRU)
        optimizer = _CountingOptimizer(TR.build_optimizer(model))
        stats = TR.train_one_epoch(
            model, optimizer, reader, timeline.streams(), pos_weight=2.0
        )
        assert stats.frontier_count == 1
        assert stats.length_group_count == 3
        assert optimizer.steps == 1
        assert stats.direct_loss_row_count == sum(lengths)


def test_the_denominator_is_every_direct_loss_row_in_the_frontier(tmp_path):
    """Not per stream, not per group: one total over the whole frontier."""
    subjects = FX.frozen_train_subjects()[:2]
    streams = (
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[0]),
            channel_index=0,
            families=(FX.PRIMARY_POSITIVE, FX.PRIMARY_NEGATIVE, FX.CHALLENGE_RATE),
        ),
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[1]),
            channel_index=0,
            families=(FX.PRIMARY_NEGATIVE, FX.OTHER_BOUNDARY),
        ),
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        model = build_t2_model(T2_ARM_GRU)
        stats = TR.train_one_epoch(
            model,
            _CountingOptimizer(TR.build_optimizer(model)),
            reader,
            timeline.streams(),
            pos_weight=2.0,
        )
        # 3 PRIMARY rows across both streams; the challenge and boundary rows
        # are context and carry no direct loss.
        assert stats.direct_loss_row_count == 3
        assert stats.context_row_count == 5


def test_a_frontier_with_no_direct_loss_row_produces_no_step(tmp_path):
    subjects = FX.frozen_train_subjects()[:1]
    streams = (
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[0]),
            channel_index=0,
            families=tuple(FX.CHALLENGE_RATE for _ in range(4)),
        ),
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        model = build_t2_model(T2_ARM_GRU)
        optimizer = _CountingOptimizer(TR.build_optimizer(model))
        stats = TR.train_one_epoch(
            model, optimizer, reader, timeline.streams(), pos_weight=2.0
        )
        assert stats.frontier_count == 1
        assert stats.optimizer_step_count == 0
        assert optimizer.steps == 0
        assert stats.zero_direct_loss_frontier_count == 1
        assert stats.direct_loss_row_count == 0


def test_the_epoch_record_uses_frontier_terminology(completed_attempt):
    report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    payload = json.loads((run_dir / PS.ARM_RESULT_NAME[T2_ARM_GRU]).read_text())
    first = payload["epochs"][0]
    for key in (
        "optimizer_step_count",
        "zero_direct_loss_frontier_count",
        "direct_loss_row_count",
        "weighted_loss_sum",
        "mean_weighted_loss_per_direct_row",
        "frontier_count",
    ):
        assert key in first, key
    assert not any("chunk" in key for key in first)
    assert report["status"] == PS.STATUS_COMPLETE


# --- F. availability -------------------------------------------------------


def test_equal_raw_lengths_with_different_gaps_do_not_stack_together(tmp_path):
    """Same raw slice length, different compacted length: different groups."""
    subjects = FX.frozen_train_subjects()[:2]
    streams = (
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[0]),
            channel_index=0,
            families=(FX.PRIMARY_POSITIVE,)
            + tuple(FX.PRIMARY_NEGATIVE for _ in range(9)),
            unavailable=frozenset({4}),
        ),
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[1]),
            channel_index=0,
            families=(FX.PRIMARY_POSITIVE,)
            + tuple(FX.PRIMARY_NEGATIVE for _ in range(9)),
        ),
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        views = [
            reader.slice_view(index, stream, 0, stream.row_count)
            for index, stream in enumerate(timeline.streams())
        ]
        assert views[0].raw_length == views[1].raw_length == 10
        assert views[0].length == 9 and views[1].length == 10
        model = build_t2_model(T2_ARM_GRU)
        optimizer = _CountingOptimizer(TR.build_optimizer(model))
        stats = TR.train_one_epoch(
            model, optimizer, reader, timeline.streams(), pos_weight=2.0
        )
        assert stats.length_group_count == 2, "compacted length decides the group"
        assert optimizer.steps == 1
        assert stats.unavailable_row_count == 1
        assert stats.direct_loss_row_count == 19


def test_an_unavailable_row_is_dropped_before_the_model_sees_it(tmp_path):
    subjects = FX.frozen_train_subjects()[:1]
    streams = (
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subjects[0]),
            channel_index=0,
            families=tuple(FX.PRIMARY_NEGATIVE for _ in range(6)),
            unavailable=frozenset({2, 3}),
        ),
    )
    environment = FX.build_environment(tmp_path / "env", streams=streams)
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        view = reader.slice_view(0, timeline.streams()[0], 0, 6)
        assert view.positions.tolist() == [0, 1, 4, 5]
        assert view.length == 4
        assert np.all(np.isfinite(view.values)), "no NaN row reaches the model"


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_skipping_an_unavailable_row_equals_carrying_state_unchanged(arm):
    """The compaction IS the state no-op, proved against a contiguous pass."""
    model = build_t2_model(arm)
    model.eval()
    generator = torch.Generator().manual_seed(11)
    available = torch.randn(1, 6, T2_INPUT_DIM, generator=generator)
    with torch.no_grad():
        contiguous, _ = model(available)
        first, state = model(available[:, :3])
        second, _ = model(available[:, 3:], state)
    assert torch.allclose(contiguous[:, 3:], second, atol=1e-5)
    assert torch.allclose(contiguous[:, :3], first, atol=1e-5)


# --- G. checkpoint ---------------------------------------------------------


def test_the_selected_best_epoch_is_the_checkpoint_used_for_the_threshold(
    completed_attempt,
):
    """The retained bytes reproduce the exact predictions they were chosen on."""
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    for arm in T2_ARMS:
        payload = json.loads((run_dir / PS.ARM_RESULT_NAME[arm]).read_text())
        selection = payload["checkpoint_selection"]
        assert payload["best_epoch"] == selection["best_epoch"]
        assert (
            payload["best_internal_dev_pooled_auprc"]
            == (selection["best_internal_dev_pooled_auprc"])
        )
        assert payload["threshold_derived_from_best_checkpoint"] is True
        # The score digest is the link: it was captured at the best epoch and
        # re-derived from the reloaded checkpoint during the threshold pass.
        lock = json.loads((run_dir / PS.CHECKPOINT_LOCK_NAME[arm]).read_text())
        assert lock["best_epoch"] == payload["best_epoch"]
        assert (
            lock["internal_dev_score_sha256"] == (payload["internal_dev_score_sha256"])
        )


def test_a_mutated_checkpoint_is_refused(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    path = run_dir / PS.CHECKPOINT_NAME[T2_ARM_GRU]
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(PS.T2PersistenceError, match="does not match"):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)


def test_a_mutated_checkpoint_lock_is_refused(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    path = run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_S4D]
    payload = json.loads(path.read_text())
    payload["best_epoch"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(
        PS.T2PersistenceError, match=PS.CHECKPOINT_LOCK_NAME[T2_ARM_S4D]
    ):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)
    # And the lock is refused on its own terms too, not only by its file digest.
    with pytest.raises(PS.T2PersistenceError, match="own digest validation"):
        PS.validate_checkpoint_lock(payload, T2_ARM_S4D, run_dir=run_dir)


def test_a_checkpoint_lock_self_digest_mutation_is_refused(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    path = run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_GRU]
    payload = json.loads(path.read_text())
    payload["checkpoint_lock_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)


def test_a_checkpoint_lock_pointing_at_another_checkpoint_is_refused(
    completed_attempt,
):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    gru = json.loads((run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_GRU]).read_text())
    s4d = json.loads((run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_S4D]).read_text())
    gru["checkpoint_sha256"] = s4d["checkpoint_sha256"]
    # Repairing the self-digest is deliberate: without it the lock would be
    # refused for being internally inconsistent, and the point here is that a
    # perfectly self-consistent lock naming the WRONG checkpoint is still
    # refused.
    body = {k: v for k, v in gru.items() if k != "checkpoint_lock_sha256"}
    gru["checkpoint_lock_sha256"] = canonical_sha256(body)
    with pytest.raises(PS.T2PersistenceError, match="A mutated checkpoint is refused"):
        PS.validate_checkpoint_lock(gru, T2_ARM_GRU, run_dir=run_dir)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", T2_ARM_S4D),
        ("trainable_parameters", 1),
        ("t2_protocol_sha256", "0" * 64),
        ("t2_execution_spec_sha256", "0" * 64),
        ("internal_split_sha256", "0" * 64),
    ],
)
def test_checkpoint_lock_identity_drift_is_refused(completed_attempt, field, value):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    payload = json.loads((run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_GRU]).read_text())
    payload[field] = value
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_checkpoint_lock(payload, T2_ARM_GRU, run_dir=run_dir)


def test_the_canonical_validator_binds_the_checkpoint_locks_itself(
    completed_attempt,
):
    """No caller has to remember a second validator."""
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    for arm in T2_ARMS:
        assert arm in lock["checkpoint_lock_sha256"]
        assert arm in lock["checkpoint_lock_self_sha256"]
        assert PS.CHECKPOINT_LOCK_NAME[arm] in lock["artifact_sha256"]
    (run_dir / PS.CHECKPOINT_LOCK_NAME[T2_ARM_S4D]).unlink()
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)


# --- H. runtime choreography ----------------------------------------------


def test_the_enforcement_points_were_visited_in_the_frozen_order(completed_attempt):
    report, sources = completed_attempt
    required = PS.required_runtime_stage_order(T2_ARMS)
    assert required == (
        PS.STAGE_TRAINING_START,
        PS.stage_pre_model_construction(T2_ARM_GRU),
        PS.stage_pre_checkpoint_promotion(T2_ARM_GRU),
        PS.stage_pre_model_construction(T2_ARM_S4D),
        PS.stage_pre_checkpoint_promotion(T2_ARM_S4D),
        PS.RESULT_NAME,
    )
    observed = list(report["runtime_enforcement_stages"])
    filtered = tuple(label for label in observed if label in set(required))
    assert filtered == required

    run_dir = sources.run_root / sources.attempt_id
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    checks = lock["runtime_identity_checks"]["checks"]
    assert checks[0]["enforcement_point"] == "start"
    assert checks[-1]["enforcement_point"] == "completion"
    assert lock["runtime_identity_checks"]["all_observations_matched"] is True


def test_a_missing_enforcement_stage_is_refused():
    from cardiosentinel.neural.runtime_sentinel import RuntimeIntegrityRecord

    record = RuntimeIntegrityRecord()
    for detail in (PS.STAGE_TRAINING_START, PS.RESULT_NAME):
        record.record(
            RuntimeCheck(
                enforcement_point="start",
                observed_digest=record.expected_digest,
                expected_digest=record.expected_digest,
                matches=True,
                package_count=335,
                observed_at="2026-01-01T00:00:00Z",
                detail=detail,
            )
        )
    with pytest.raises(PS.T2PersistenceError, match="enforcement choreography"):
        PS.require_runtime_stage_order(record, T2_ARMS)


def test_determinism_is_established_before_the_first_runtime_reading():
    """The real reading, not the frozen fake.

    Establishing determinism lazily inside the first arm's construction made
    arm A observe `deterministic_algorithms: False` and arm B `True`, and the
    same-runtime check then correctly refused a comparison that was never
    actually mixed. This is that defect, pinned.
    """
    from cardiosentinel.neural.t2_models import seed_everything

    seed_everything()
    first = PS.runtime_provenance()
    seed_everything()
    second = PS.runtime_provenance()
    assert first["deterministic_algorithms"] is True
    PS.require_single_runtime(first, second)


def test_both_arms_observed_the_same_runtime(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    runtimes = [
        json.loads((run_dir / PS.ARM_RESULT_NAME[arm]).read_text())["runtime"]
        for arm in T2_ARMS
    ]
    PS.require_single_runtime(runtimes[0], runtimes[1])
    assert runtimes[0]["deterministic_algorithms"] is True


# --- I. failure semantics --------------------------------------------------


def _fail_on_arm(monkeypatch, arm: str) -> dict[str, bool]:
    """Fail at a real boundary: constructing that arm's model.

    Returns an `armed` switch rather than relying on `monkeypatch.undo()`,
    which would also tear down the frozen-runtime and clean-Git seams and
    leave the second attempt asking the wrong environment to be canonical.
    """
    original = TR.build_candidate
    armed = {"armed": True}

    def failing(name: str):
        if armed["armed"] and name == arm:
            raise RuntimeError(f"synthetic {name} construction failure")
        return original(name)

    monkeypatch.setattr(TR, "build_candidate", failing)
    return armed


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_a_failure_in_either_arm_consumes_the_whole_attempt(
    tmp_path, environment, clean_git, monkeypatch, arm
):
    _fail_on_arm(monkeypatch, arm)
    sources = _sources(tmp_path, environment)
    with pytest.raises(RuntimeError, match="synthetic"):
        RUN._execute_training_attempt(_checks(), sources)

    run_dir = sources.run_root / sources.attempt_id
    assert run_dir.is_dir(), "the claim is consumed, not rolled back"
    assert not (run_dir / PS.RESULT_NAME).exists()
    assert not (run_dir / PS.EXPERIMENT_LOCK_NAME).exists()
    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["canonical"] is False
    assert status["repeat_attempt_permitted"] is False
    assert status["automatic_retry_performed"] is False
    assert status["human_review_required"] is True

    review = PS.t2_review_directory(sources.run_root, sources.attempt_id)
    receipt = json.loads((review / PS.FAILURE_RECEIPT_NAME).read_text())
    assert receipt["claim_bearing"] is False
    assert receipt["attempt_consumed"] is True
    assert receipt["exception_message"].startswith("synthetic")
    assert receipt["arm_selection_status"] == PS.ARM_SELECTION_PENDING


def test_a_failure_during_the_second_arm_promotes_no_complete_result(
    tmp_path, environment, clean_git, monkeypatch
):
    """Arm A's evidence survives as forensic material. Nothing is claim-bearing."""
    _fail_on_arm(monkeypatch, T2_ARM_S4D)
    sources = _sources(tmp_path, environment)
    with pytest.raises(RuntimeError):
        RUN._execute_training_attempt(_checks(), sources)

    run_dir = sources.run_root / sources.attempt_id
    assert (run_dir / PS.CHECKPOINT_NAME[T2_ARM_GRU]).is_file()
    assert (run_dir / PS.ARM_RESULT_NAME[T2_ARM_GRU]).is_file()
    assert not (run_dir / PS.CHECKPOINT_NAME[T2_ARM_S4D]).exists()
    assert not (run_dir / PS.RESULT_NAME).exists()
    with pytest.raises(PS.T2PersistenceError, match="No canonical T2 artifact"):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)
    receipt = json.loads(
        (
            PS.t2_review_directory(sources.run_root, sources.attempt_id)
            / PS.FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["exposure"]["arms_completed"] == [T2_ARM_GRU]


def test_no_selective_rerun_of_the_failed_arm_is_possible(
    tmp_path, environment, clean_git, monkeypatch
):
    armed = _fail_on_arm(monkeypatch, T2_ARM_S4D)
    sources = _sources(tmp_path, environment)
    with pytest.raises(RuntimeError):
        RUN._execute_training_attempt(_checks(), sources)
    # Repair the cause. The attempt is still consumed: there is no route back
    # in for the failed arm alone, and none for the whole attempt either.
    armed["armed"] = False
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        RUN._execute_training_attempt(_checks(), sources)


def test_no_retry_force_or_recovery_vocabulary_exists_in_the_route():
    source = " ".join(Path(RUN.__file__).read_text().split())
    for phrase in ("recovery1", "--retry", "--force", "--fresh-seed", "--reset"):
        assert f'"{phrase}"' not in source or phrase in str(RUN.FORBIDDEN_OPTIONS)


# --- J. outer VALIDATION ---------------------------------------------------


def test_the_public_gate_refuses_before_any_loader_access(monkeypatch):
    """No path is resolved, no timeline is opened, no label is read.

    Activation is now True, so the refusal comes from the authorized-commit
    gate rather than the activation gate. The argument is a nonsense value the
    checkout cannot be at, and the loader spy proves nothing was opened.
    """
    opened: list[object] = []
    monkeypatch.setattr(
        EV, "_open_validation_timeline", lambda *a, **k: opened.append(a) or None
    )
    for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
        with pytest.raises(RUN.T2RunError, match=_PRE_CLAIM_REFUSAL):
            entry(Path("/nonexistent"))
    assert opened == [], "the refusal fired before anything was opened"
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True
    assert not (PS.T2_RUN_ROOT / PS.T2_OUTER_VALIDATION_ATTEMPT_ID).exists()


def test_the_activation_gate_is_the_first_statement_of_every_entry_point():
    """Not merely present somewhere: the very first executable statement."""
    import ast
    import inspect
    import textwrap

    for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
        tree = ast.parse(textwrap.dedent(inspect.getsource(entry)))
        body = list(tree.body[0].body)
        # Drop the docstring, which is not an executable statement.
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        assert body, entry.__name__
        assert "require_outer_validation_authorized" in ast.dump(body[0]), (
            entry.__name__
        )


def test_outer_validation_selection_is_delegated_to_the_protocol(monkeypatch):
    from cardiosentinel.neural import t2_protocol

    seen: dict[str, object] = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return {"selected_arm": T2_ARM_GRU, "rule": "spy"}

    monkeypatch.setattr(t2_protocol, "select_t2_arm", spy)
    monkeypatch.setattr(EV, "select_t2_arm", spy)
    decision = EV.select_from_outer_validation(
        pooled_auprc={T2_ARM_GRU: 0.4, T2_ARM_S4D: 0.4},
        subject_macro_auprc={T2_ARM_GRU: 0.4, T2_ARM_S4D: 0.4},
        parameter_counts={T2_ARM_GRU: 59_521, T2_ARM_S4D: 45_313},
    )
    assert decision["rule"] == "spy"
    assert set(seen) == {
        "pooled_auprc",
        "subject_macro_auprc",
        "parameter_counts",
    }


def test_the_validation_result_schema_still_validates_the_frozen_shape():
    payload = {name: {} for name in EV.REQUIRED_OUTER_VALIDATION_FIELDS}
    payload.update(
        {
            "artifact_class": EV.OUTER_VALIDATION_RESULT_CLASS,
            "per_arm_evidence": {arm: {} for arm in T2_ARMS},
            "test_accessed": False,
            "sealed_test_state": "unopened",
            "primary_population_identity": {"row_count": 473_897},
        }
    )
    assert EV.validate_outer_validation_result(payload) is payload
    with pytest.raises(EV.T2EvaluationError, match="PRIMARY population"):
        EV.validate_outer_validation_result(
            {**payload, "primary_population_identity": {"row_count": 5}}
        )


# --- K. challenge wording --------------------------------------------------


def test_no_broad_trained_on_field_exists_anywhere():
    families = ["rate_related", "axis_shift", "conduction_change"]
    evidence = EV.challenge_family_evidence(families, [0, 0, 0], [0.9, 0.1, 0.5], 0.5)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert key != "trained_on", "the broad claim is false and is gone"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(evidence)
    for module in (EV, TR, TL, PS, RUN):
        assert '"trained_on"' not in Path(module.__file__).read_text()


def test_the_precise_causal_context_semantics_are_exact():
    evidence = EV.challenge_family_evidence(["rate_related"], [0], [0.9], 0.5)
    expected = {
        "direct_training_loss_received": False,
        "challenge_identity_model_input": False,
        "challenge_label_model_input": False,
        "may_be_label_blind_causal_context": True,
        "checkpoint_selection_input": False,
        "arm_selection_input": False,
    }
    for key, value in expected.items():
        assert evidence[key] is value, key
        assert evidence["subsets"]["rate_related"][key] is value, key
    assert EV.CHALLENGE_CAUSAL_SEMANTICS == expected


def test_a_challenge_row_can_causally_alter_a_later_primary_output():
    """Which is exactly why `trained_on: false` would have been a lie."""
    model = build_t2_model(T2_ARM_GRU)
    model.eval()
    generator = torch.Generator().manual_seed(7)
    challenge = torch.randn(1, 1, T2_INPUT_DIM, generator=generator)
    primary = torch.randn(1, 1, T2_INPUT_DIM, generator=generator)
    with torch.no_grad():
        with_context, _ = model(torch.cat([challenge, primary], dim=1))
        without_context, _ = model(primary)
    assert not torch.allclose(with_context[:, 1:], without_context, atol=1e-6)


# --- L. TEST firewall ------------------------------------------------------


def test_test_is_refused_everywhere(tmp_path, environment):
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.refuse_sealed_partition("test")
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.T2Timeline("test", root=environment.stream_cache_root)
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.require_frozen_stream_identity("test", {})
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    assert "--test" not in options
    assert RUN.TRAIN_PARTITION == "train"


def test_every_promoted_artifact_records_the_sealed_test_state(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    for name in (
        PS.RESULT_NAME,
        PS.EXPERIMENT_LOCK_NAME,
        PS.RUN_STATUS_NAME,
        PS.POPULATION_NAME,
        *(PS.ARM_RESULT_NAME[arm] for arm in T2_ARMS),
        *(PS.CHECKPOINT_LOCK_NAME[arm] for arm in T2_ARMS),
    ):
        payload = json.loads((run_dir / name).read_text())
        assert payload["test_accessed"] is False, name
        assert payload["sealed_test_state"] == "unopened", name


# --- the split, the class weight and the population lineage ---------------


def test_the_internal_split_is_the_frozen_forty_eight_eight(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    split = json.loads((run_dir / PS.INTERNAL_SPLIT_NAME).read_text())
    assert split["split_sha256"] == (
        "54f8091ee7d4620ab6e24aaa32b121874b6a1610003e3df63f94f9727618e28e"
    )
    assert split["fit_count"] == 48
    assert split["internal_dev_count"] == 8
    assert tuple(split["internal_dev_subjects"]) == T2_INTERNAL_DEV_SUBJECTS


def test_the_class_weight_uses_the_fit_forty_eight_only(completed_attempt):
    report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    population = json.loads((run_dir / PS.POPULATION_NAME).read_text())
    weight = population["class_weight"]
    assert weight["partition"] == "t2_fit_48_subjects"
    assert weight["fit_subject_count"] == 48
    assert weight["internal_dev_rows_counted"] is False
    assert weight["outer_validation_rows_counted"] is False
    assert weight["all_train_subjects_counted"] is False
    assert weight["counted_population"] == "available_primary_direct_loss_rows"
    assert weight["positive_class_weight"] == pytest.approx(
        weight["fit_negative_count"] / weight["fit_positive_count"]
    )
    assert report["fit_positive_count"] == weight["fit_positive_count"]
    assert report["fit_negative_count"] == weight["fit_negative_count"]
    assert population["internal_dev_contributes_optimizer_gradient"] is False


def test_the_class_weight_refuses_an_internal_dev_contaminated_selection(
    environment,
):
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        reader = _reader_for(environment, timeline)
        contaminated = timeline.streams_for_subjects(set(timeline.subjects()))
        with pytest.raises(TR.T2TrainingError, match="Internal-dev subjects"):
            TR.fit_class_weight_evidence(
                reader,
                contaminated,
                fit_subjects=timeline.subjects(),
                internal_dev_subjects=T2_INTERNAL_DEV_SUBJECTS,
            )


def test_the_target_authority_lineage_is_bound_by_the_result(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    result = json.loads((run_dir / PS.RESULT_NAME).read_text())
    lineage = result["target_authority_identity"]
    assert lineage["authority"] == "ltstdb_baseline_v1_feature_corpus"
    assert lineage["partition"] == "train"
    assert lineage["join_key"] == "stable_id"
    assert lineage["join_is_record_wise_bounded"] is True
    assert lineage["record_count"] == 56
    assert set(lineage["record_cache_sha256"]) == {
        stream.record_id for stream in sources.open_timeline().streams()
    }
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    assert lock["target_authority_identity"] == lineage


def test_the_population_identity_proves_the_promoted_rows(completed_attempt):
    _report, sources = completed_attempt
    run_dir = sources.run_root / sources.attempt_id
    population = json.loads((run_dir / PS.POPULATION_NAME).read_text())
    assert population["population_identity_proves"] == [
        "same_rows",
        "same_ordering",
        "same_physical_timeline",
        "same_category_authority",
    ]
    identity = population["train_timeline_identity"]
    assert (
        population["rederived_ordered_stable_id_sha256"]
        == (identity["ordered_stable_id_sha256"])
    )
    assert identity["persisted_bytes_revalidated"] is True
    assert identity["byte_level_validation_route"] == (
        "m1_experiment.load_stream_store"
    )
    assert population["negative_sampling_applied"] is False


def test_the_frozen_family_census_gate_refuses_a_wrong_population():
    with pytest.raises(TL.T2TimelineError, match="frozen one and nothing proceeds"):
        TL.require_frozen_family_census(
            "train",
            {
                "row_count": 2_208_431,
                "primary_row_count": 2_143_599,
                "ischemic_positive": 93_613,
                "background_negative": 2_049_986,
                "challenge_row_count": 46_025,
                "other_non_primary_row_count": 1,
            },
        )
    TL.require_frozen_family_census(
        "train",
        {
            "row_count": 2_208_431,
            "primary_row_count": 2_143_599,
            "ischemic_positive": 93_613,
            "background_negative": 2_049_986,
            "challenge_row_count": 46_025,
            "other_non_primary_row_count": 18_807,
        },
    )
