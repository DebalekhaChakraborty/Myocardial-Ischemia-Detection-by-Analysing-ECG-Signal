"""The one-shot outer-VALIDATION attempt, proven while its gate stays False.

The public route still refuses: `T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED` is
unchanged, and every public entry point refuses as its first statement before
resolving a path or opening anything. What this file proves is that the body
behind the gate is *already complete* -- claim, per-row evidence, accounting,
stream-aware descriptors, failure semantics, canonical validator -- so the
future activation change set changes a switch and not the science.

**No real science happens here.** No canonical CLI invocation, no real TRAIN
optimiser step, no real internal-dev score, no real threshold, no real
VALIDATION per-row access, no real challenge evidence, no TEST. Every timeline
is a synthetic on-disk fixture in `tmp_path`.

The same two environmental substitutions as the training-route tests, and no
others: a clean `git_provenance`, and a frozen runtime observation, because the
frozen 335-package digest belongs to `venvs/tactics` alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t2_development_run as RUN
from cardiosentinel.neural import t2_evaluation as EV
from cardiosentinel.neural import t2_outer_evidence as ES
from cardiosentinel.neural import t2_persistence as PS
from cardiosentinel.neural import t2_timeline as TL
from cardiosentinel.neural.t2_protocol import T2_ARM_GRU, T2_ARM_S4D, T2_ARMS
from tests.neural import t2_fixtures as FX
from tests.neural.test_t2_canonical_training_route import (  # noqa: F401
    GIT_SHA,
    clean_git,
    environment,
    frozen_runtime,
)


def _train_checks():
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


def _validation_streams(count: int = 4, *, gap_in_first: bool = True):
    return tuple(
        FX.SyntheticStream(
            record_id=FX.record_for_subject(subject),
            channel_index=0,
            families=FX.default_streams((subject,))[0].families,
            # Index 4 of the frozen pattern is a PRIMARY row for the first
            # subject, so the planted gap makes the PRIMARY target and
            # PRIMARY scored populations genuinely differ.
            unavailable=frozenset({4})
            if (index == 0 and gap_in_first)
            else frozenset(),
        )
        for index, subject in enumerate(FX.frozen_train_subjects()[:count])
    )


@pytest.fixture()
def trained(tmp_path, environment, clean_git):  # noqa: F811
    """One completed synthetic TRAIN attempt, under a temporary run root."""
    sources = RUN.T2TrainingSources(
        run_root=tmp_path / "runs",
        stream_cache_root=environment.stream_cache_root,
        corpus_manifest=environment.corpus_manifest,
        canonical=False,
    )
    RUN._execute_training_attempt(_train_checks(), sources)
    return sources


@pytest.fixture()
def validation(tmp_path, monkeypatch):
    """A synthetic VALIDATION partition, with an unavailable exact-flat row."""
    streams = _validation_streams()
    built = FX.build_environment(
        tmp_path / "val", partition="validation", streams=streams
    )
    # The frozen constant is the PRIMARY **target** population, defined by the
    # label authority and independent of whether the physical observation
    # survived. The scored population is the subset that carries one.
    primary_rows = sum(
        1
        for stream in streams
        for family in stream.families
        if family in {FX.PRIMARY_POSITIVE, FX.PRIMARY_NEGATIVE}
    )
    # The frozen 473,897-row PRIMARY expectation is substituted in ONE place,
    # the reviewed M2 canonical-runner seam: a synthetic corpus cannot meet the
    # real count and the production validator rightly refuses anything else.
    # The real value is asserted separately, below.
    monkeypatch.setattr(EV, "T2_VALIDATION_PRIMARY_ROW_COUNT", primary_rows)
    return built


@pytest.fixture()
def completed_outer(trained, validation):
    """One complete synthetic outer attempt, run once and shared."""
    report = EV._outer_validation_worker(
        GIT_SHA,
        run_root=trained.run_root,
        training_attempt_id=trained.attempt_id,
        validation_root=validation.stream_cache_root,
        corpus_manifest=validation.corpus_manifest,
    )
    return report, trained, validation


# --- 1-2. the canonical run root is repository-anchored --------------------


def test_a_foreign_cwd_cannot_move_the_canonical_claim_location(tmp_path):
    """Scientific identity does not depend on where the human is standing."""
    before = PS.T2_RUN_ROOT
    assert before.is_absolute()
    original = Path.cwd()
    foreign = tmp_path / "somewhere-else"
    foreign.mkdir()
    try:
        os.chdir(foreign)
        assert PS.T2_RUN_ROOT == before
        assert (
            PS.t2_run_directory(PS.T2_RUN_ROOT, PS.T2_TRAINING_ATTEMPT_ID)
            == before / PS.T2_TRAINING_ATTEMPT_ID
        )
        assert (
            PS.t2_run_directory(PS.T2_RUN_ROOT, PS.T2_OUTER_VALIDATION_ATTEMPT_ID)
            == before / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )
    finally:
        os.chdir(original)


def test_a_foreign_cwd_cannot_bypass_an_existing_claim(tmp_path, monkeypatch):
    """A consumed attempt stays consumed from any working directory."""
    run_root = tmp_path / "runs"
    (run_root / PS.T2_TRAINING_ATTEMPT_ID).mkdir(parents=True)
    monkeypatch.setattr(PS, "T2_RUN_ROOT", run_root)
    original = Path.cwd()
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    try:
        os.chdir(foreign)
        with pytest.raises(PS.T2PersistenceError, match="already claimed"):
            PS.require_unclaimed_t2_attempt(PS.T2_RUN_ROOT, PS.T2_TRAINING_ATTEMPT_ID)
    finally:
        os.chdir(original)


def test_the_run_root_is_not_cwd_relative_in_source():
    source = Path(PS.__file__).read_text()
    assert 'Path("cardiosentinel-runs' not in source
    assert "REPOSITORY_ROOT" in source


# --- 3-4. actual device execution ------------------------------------------


def test_the_model_device_equals_the_persisted_execution_device(trained):
    run_dir = trained.run_root / trained.attempt_id
    for arm in T2_ARMS:
        payload = json.loads((run_dir / PS.ARM_RESULT_NAME[arm]).read_text())
        proof = payload["execution_device_proof"]
        assert proof["execution_device_agrees"] is True
        assert PS._same_device(
            proof["declared_execution_device"], payload["model_parameter_device"]
        )
        assert PS._same_device(
            payload["execution_device"], payload["model_parameter_device"]
        )
        lock = json.loads((run_dir / PS.CHECKPOINT_LOCK_NAME[arm]).read_text())
        assert lock["execution_device_proof"]["execution_device_agrees"] is True


def test_false_cuda_provenance_with_a_cpu_model_is_refused():
    """No real CUDA hardware needed: the claim and the observation disagree."""
    lying = {"declared_execution_device": "cuda:0", "device_type": "cuda"}
    with pytest.raises(PS.T2PersistenceError, match="did not run on"):
        PS.require_execution_device_agreement(lying, "cpu")
    honest = {"declared_execution_device": "cpu", "device_type": "cpu"}
    assert PS.require_execution_device_agreement(honest, "cpu")[
        "execution_device_agrees"
    ]
    # `cuda` and `cuda:0` are the same device; `cuda:0` and `cuda:1` are not.
    assert PS._same_device("cuda", "cuda:0")
    assert not PS._same_device("cuda:0", "cuda:1")


def test_the_execution_device_is_selected_once_and_has_no_override():
    import torch

    device = PS.canonical_execution_device()
    assert device.type in {"cpu", "cuda"}
    assert device == torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for module in (PS, RUN, EV):
        source = Path(module.__file__).read_text()
        assert "T2_TRAIN_EXECUTION_AUTHORIZED" not in source
        # `--device` is named only where it is forbidden or documented as
        # forbidden; it is never wired into the parser.
        assert 'add_argument("--device"' not in source
    assert "--device" in RUN.FORBIDDEN_OPTIONS
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    assert "--device" not in options


def test_a_split_device_model_is_refused():
    from cardiosentinel.neural.t2_models import build_t2_model

    model = build_t2_model(T2_ARM_GRU)
    assert PS.model_parameter_device(model) == "cpu"


def test_determinism_is_required_not_traded_away(monkeypatch):
    import torch

    from cardiosentinel.neural.t2_models import seed_everything

    seed_everything()
    assert (
        PS.require_deterministic_execution(torch.device("cpu"))[
            "silent_cpu_fallback_performed"
        ]
        is False
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    with pytest.raises(PS.T2PersistenceError, match="no CPU fallback"):
        PS.require_deterministic_execution(torch.device("cuda:0"))


# --- 5-6. the failing arm is named correctly -------------------------------


def _fail_on_arm(monkeypatch, arm):
    from cardiosentinel.neural import t2_training as TR

    original = TR.build_candidate
    armed = {"armed": True}

    def failing(name):
        if armed["armed"] and name == arm:
            raise RuntimeError(f"synthetic {name} construction failure")
        return original(name)

    monkeypatch.setattr(TR, "build_candidate", failing)
    return armed


@pytest.mark.parametrize(
    "failing_arm,expected_completed",
    [(T2_ARM_GRU, []), (T2_ARM_S4D, [T2_ARM_GRU])],
)
def test_the_failure_receipt_names_the_arm_that_actually_failed(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
    monkeypatch,
    failing_arm,
    expected_completed,
):
    """`arms_completed[-1]` is not the failing arm; `current_arm` is."""
    _fail_on_arm(monkeypatch, failing_arm)
    sources = RUN.T2TrainingSources(
        run_root=tmp_path / "runs",
        stream_cache_root=environment.stream_cache_root,
        corpus_manifest=environment.corpus_manifest,
        canonical=False,
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        RUN._execute_training_attempt(_train_checks(), sources)
    receipt = json.loads(
        (
            PS.t2_review_directory(sources.run_root, sources.attempt_id)
            / PS.FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["arm"] == failing_arm
    assert receipt["exposure"]["current_arm"] == failing_arm
    assert receipt["exposure"]["arms_completed"] == expected_completed
    assert receipt["attempt_consumed"] is True


# --- 7-8. the public outer gate --------------------------------------------


def test_the_public_outer_route_opens_nothing_while_the_gate_is_false(monkeypatch):
    opened: list[object] = []
    monkeypatch.setattr(
        EV, "_open_validation_timeline", lambda *a, **k: opened.append(a)
    )
    monkeypatch.setattr(
        EV, "_outer_validation_worker", lambda *a, **k: opened.append(a)
    )
    monkeypatch.setattr(
        EV, "_outer_validation_preflight", lambda *a, **k: opened.append(a)
    )
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is False
    for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
        with pytest.raises(PS.T2ActivationError, match="not authorized"):
            entry(GIT_SHA)
    with pytest.raises(PS.T2ActivationError):
        RUN.execute_canonical_outer_validation(GIT_SHA)
    assert RUN.main(["--execute-canonical-outer-validation"]) == 3
    assert opened == [], "the gate fired before anything was resolved or opened"


def test_a_wrong_expected_git_sha_opens_nothing(
    tmp_path, trained, validation, monkeypatch
):
    """Activation true is not enough. The authorized commit is proved next."""
    opened: list[object] = []
    real_open = EV._open_validation_timeline

    def spy(*args, **kwargs):
        opened.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(EV, "_open_validation_timeline", spy)
    with pytest.raises(RUN.T2RunError, match="but the run expects"):
        EV._outer_validation_worker(
            "b" * 40,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )
    assert opened == []
    assert not (trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID).exists(), (
        "a refused attempt claims nothing"
    )


def test_a_missing_expected_git_sha_opens_nothing(trained, validation):
    with pytest.raises(RUN.T2RunError, match="--expected-git-sha is required"):
        EV._outer_validation_worker(
            None,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )
    assert not (trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID).exists()


def test_the_activation_gate_is_still_the_first_statement():
    import ast
    import inspect
    import textwrap

    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is False
    entries = [
        *EV.OUTER_VALIDATION_ENTRY_POINTS,
        RUN.execute_canonical_outer_validation,
    ]
    for entry in entries:
        tree = ast.parse(textwrap.dedent(inspect.getsource(entry)))
        body = list(tree.body[0].body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        assert "require_outer_validation_authorized" in ast.dump(body[0]), (
            entry.__name__
        )


# --- 9-10. the outer claim is one-shot -------------------------------------


def test_the_outer_attempt_is_one_shot(completed_outer):
    report, trained, validation = completed_outer
    assert report["status"] == PS.STATUS_COMPLETE
    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    assert run_dir.is_dir()
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        PS.require_unclaimed_outer_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        EV._outer_validation_worker(
            GIT_SHA,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )


def test_the_outer_attempt_is_a_sibling_of_the_training_attempt(completed_outer):
    _report, trained, _validation = completed_outer
    training_dir = trained.run_root / PS.T2_TRAINING_ATTEMPT_ID
    outer_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    assert training_dir.parent == outer_dir.parent
    assert outer_dir.name == "t2-v1-outer-validation"


def test_a_post_claim_outer_failure_consumes_the_attempt(
    trained, validation, monkeypatch
):
    from cardiosentinel.neural import t2_training as TR

    def failing(*_args, **_kwargs):
        raise RuntimeError("synthetic outer scoring failure")

    monkeypatch.setattr(TR, "score_streams", failing)
    with pytest.raises(RuntimeError, match="synthetic outer"):
        EV._outer_validation_worker(
            GIT_SHA,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )
    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    assert run_dir.is_dir(), "the claim is consumed, not rolled back"
    assert not (run_dir / PS.OUTER_RESULT_NAME).exists()
    assert not (run_dir / PS.OUTER_LOCK_NAME).exists()
    status = json.loads((run_dir / PS.OUTER_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["repeat_attempt_permitted"] is False
    assert status["selective_arm_rerun_permitted"] is False
    receipt = json.loads(
        (
            PS.t2_review_directory(trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID)
            / PS.OUTER_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["claim_bearing"] is False
    assert receipt["attempt_consumed"] is True
    assert receipt["arm"] == T2_ARM_GRU
    assert receipt["exposure"]["validation_accessed"] is True
    assert receipt["exposure"]["arm_selection_exposed"] is False
    assert receipt["exposure"]["checkpoints_loaded"] == [T2_ARM_GRU]
    assert receipt["automatic_retry_performed"] is False
    assert receipt["alternate_attempt_name_permitted"] is False


# --- 11-13. the canonical outer validator ----------------------------------


def test_the_canonical_outer_validator_accepts_the_completed_attempt(
    completed_outer,
):
    _report, trained, _validation = completed_outer
    verification = PS.validate_canonical_t2_outer_validation_attempt(
        trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    )
    assert verification["verified"] is True
    assert verification["selected_arm"] in T2_ARMS
    assert verification["training_attempt_verification"]["verified"] is True
    assert sorted(verification["checkpoint_sha256"]) == sorted(T2_ARMS)


def test_an_outer_result_mutation_fails_canonical_validation(completed_outer):
    _report, trained, _validation = completed_outer
    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    path = run_dir / PS.OUTER_RESULT_NAME
    payload = json.loads(path.read_text())
    payload["selected_arm"] = (
        T2_ARM_S4D if payload["selected_arm"] == T2_ARM_GRU else T2_ARM_GRU
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


def test_an_outer_lock_mutation_fails_canonical_validation(completed_outer):
    _report, trained, _validation = completed_outer
    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    path = run_dir / PS.OUTER_LOCK_NAME
    payload = json.loads(path.read_text())
    payload["authorized_git_sha"] = "c" * 40
    path.write_text(json.dumps(payload))
    with pytest.raises(PS.T2PersistenceError, match="digest validation"):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


@pytest.mark.parametrize("group_key", ["row_identity", T2_ARM_GRU, T2_ARM_S4D])
def test_a_per_row_evidence_mutation_fails_canonical_validation(
    completed_outer, group_key
):
    _report, trained, _validation = completed_outer
    evidence_root = (
        trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID / PS.OUTER_EVIDENCE_DIRNAME
    )
    manifest = json.loads((evidence_root / ES.T2_OUTER_STORE_MANIFEST_NAME).read_text())
    path = evidence_root / manifest["row_groups"][group_key]["file"]
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    column = "score" if group_key in T2_ARMS else "start_sample"
    mutated = arrays[column].copy()
    mutated[0] = mutated[0] + 1
    arrays[column] = mutated
    np.savez(path, **arrays)
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


def test_a_mutated_training_checkpoint_fails_outer_validation(completed_outer):
    """An outer result bound to a mutated TRAIN attempt describes no model."""
    _report, trained, _validation = completed_outer
    checkpoint = trained.run_root / trained.attempt_id / PS.CHECKPOINT_NAME[T2_ARM_GRU]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"\x00")
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


def test_a_mutated_training_checkpoint_lock_fails_outer_validation(completed_outer):
    _report, trained, _validation = completed_outer
    path = trained.run_root / trained.attempt_id / PS.CHECKPOINT_LOCK_NAME[T2_ARM_S4D]
    payload = json.loads(path.read_text())
    payload["best_epoch"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


# --- 14-17. the per-row evidence store -------------------------------------


def _store(trained):
    root = (
        trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID / PS.OUTER_EVIDENCE_DIRNAME
    )
    manifest = json.loads((root / ES.T2_OUTER_STORE_MANIFEST_NAME).read_text())
    return root, manifest


def test_the_store_carries_both_arms(completed_outer):
    _report, trained, _validation = completed_outer
    root, manifest = _store(trained)
    ES.validate_t2_outer_evidence_store(manifest, root=root)
    assert list(manifest["arms_persisted"]) == list(T2_ARMS)
    assert set(manifest["row_groups"]) == {"row_identity", *T2_ARMS}
    for arm in T2_ARMS:
        identity, scores = ES.selected_arm_scores(root, manifest, arm)
        assert scores.row_count == identity.row_count


def test_the_score_is_named_an_uncalibrated_temporal_model_score(completed_outer):
    _report, trained, _validation = completed_outer
    _root, manifest = _store(trained)
    assert manifest["score_semantics"] == "uncalibrated_temporal_model_score"
    assert manifest["score_definition"] == "sigmoid(current_window_t2_logit)"
    assert manifest["score_is_calibrated_probability"] is False
    assert manifest["score_is_confidence"] is False
    assert manifest["score_is_uncertainty"] is False


def test_unavailable_rows_carry_score_present_false(completed_outer):
    _report, trained, validation = completed_outer
    root, manifest = _store(trained)
    identity, scores = ES.selected_arm_scores(root, manifest, T2_ARM_GRU)
    present = np.asarray(identity.arrays["score_present"], dtype=bool)
    values = np.asarray(scores.arrays["score"], dtype=np.float64)
    assert int((~present).sum()) == 1, "the fixture plants exactly one gap"
    assert np.all(np.isnan(values[~present])), "no score is invented"
    assert np.all(np.isfinite(values[present]))
    assert manifest["nan_is_storage_sentinel_for_absence"] is True
    assert manifest["nan_is_ever_a_model_score"] is False
    positions = np.nonzero(~present)[0]
    with pytest.raises(ES.T2OuterEvidenceError, match="no T2 score"):
        ES.require_scores_present(scores, positions.tolist())
    ES.require_scores_present(scores, np.nonzero(present)[0][:5].tolist())
    assert manifest["row_count"] == validation.row_count


def test_full_timeline_row_accounting_closes(completed_outer):
    report, trained, validation = completed_outer
    _root, manifest = _store(trained)
    accounting = ES.require_outer_row_accounting(manifest)
    assert (
        accounting["scored_available_row_count"]
        + accounting["unavailable_no_score_row_count"]
        == accounting["row_count"]
    )
    assert accounting["row_count"] == validation.row_count
    assert report["row_accounting"]["row_count"] == validation.row_count


def test_primary_row_accounting_closes(completed_outer):
    _report, trained, _validation = completed_outer
    _root, manifest = _store(trained)
    accounting = ES.require_outer_row_accounting(manifest)
    assert (
        accounting["primary_scored_available_row_count"]
        + accounting["primary_unavailable_no_score_count"]
        == accounting["primary_target_row_count"]
    )
    # The one planted gap sits on a PRIMARY row, so target and scored differ.
    assert accounting["primary_unavailable_no_score_count"] == 1
    assert (
        accounting["primary_scored_available_row_count"]
        < accounting["primary_target_row_count"]
    )


def test_the_frozen_prospective_counts_are_the_ones_the_protocol_binds():
    """The synthetic store cannot meet these; the frozen constants still are."""
    from cardiosentinel.neural.t2_protocol import (
        T2_VALIDATION_FULL_STREAM_ROW_COUNT,
        T2_VALIDATION_PRIMARY_ROW_COUNT,
    )

    assert T2_VALIDATION_FULL_STREAM_ROW_COUNT == 492_904
    assert T2_VALIDATION_PRIMARY_ROW_COUNT == 473_897
    assert TL.EXPECTED_FAMILY_CENSUS["validation"]["row_count"] == 492_904
    assert TL.EXPECTED_FAMILY_CENSUS["validation"]["primary_row_count"] == 473_897


def test_a_store_missing_an_arm_is_refused(tmp_path):
    identity = {
        "stable_id": np.asarray(["a", "b"]),
        "record_id": np.asarray(["s20011", "s20011"]),
        "channel_index": np.asarray([0, 0]),
        "start_sample": np.asarray([0, 1250]),
        "subject_id": np.asarray(["ltstdb:s2001", "ltstdb:s2001"]),
        "target_family": np.asarray(["background_negative", "ischemic_positive"]),
        "cold_start_bin": np.asarray(["0_5_minutes", "0_5_minutes"]),
        "observation_state": np.asarray([1, 1]),
        "score_present": np.asarray([True, True]),
        "primary_mask": np.asarray([True, True]),
        "label": np.asarray([0, 1]),
    }
    scores = {
        "score": np.asarray([0.2, 0.8]),
        "score_present": np.asarray([True, True]),
        "predicted_positive": np.asarray([False, True]),
    }
    with pytest.raises(ES.T2OuterEvidenceError, match="both frozen arms"):
        ES.write_t2_outer_evidence_store(
            tmp_path / "store",
            identity=identity,
            arm_scores={T2_ARM_GRU: scores},
            lineage={},
        )


def test_a_finite_value_behind_a_false_present_mask_is_refused(tmp_path):
    identity = {
        "stable_id": np.asarray(["a", "b"]),
        "record_id": np.asarray(["s20011", "s20011"]),
        "channel_index": np.asarray([0, 0]),
        "start_sample": np.asarray([0, 1250]),
        "subject_id": np.asarray(["ltstdb:s2001", "ltstdb:s2001"]),
        "target_family": np.asarray(["background_negative", "ischemic_positive"]),
        "cold_start_bin": np.asarray(["0_5_minutes", "0_5_minutes"]),
        "observation_state": np.asarray([1, 2]),
        "score_present": np.asarray([True, False]),
        "primary_mask": np.asarray([True, True]),
        "label": np.asarray([0, 1]),
    }
    lying = {
        "score": np.asarray([0.2, 0.8]),
        "score_present": np.asarray([True, False]),
        "predicted_positive": np.asarray([False, False]),
    }
    with pytest.raises(ES.T2OuterEvidenceError, match="carries a finite value"):
        ES.write_t2_outer_evidence_store(
            tmp_path / "store",
            identity=identity,
            arm_scores=dict.fromkeys(T2_ARMS, lying),
            lineage={},
        )


def test_row_evidence_permits_selected_arm_lookup_by_stable_identity(
    completed_outer,
):
    """This is what lets T1 consume the winner without a second outer run."""
    report, trained, _validation = completed_outer
    root, manifest = _store(trained)
    selected = report["selected_arm"]
    identity, scores = ES.selected_arm_scores(root, manifest, selected)
    index = ES.row_index_by_stable_id(identity)
    assert len(index) == identity.row_count
    present = np.asarray(identity.arrays["score_present"], dtype=bool)
    wanted = str(identity.arrays["stable_id"][np.nonzero(present)[0][0]])
    position = index[wanted]
    ES.require_scores_present(scores, [position])
    assert np.isfinite(scores.arrays["score"][position])
    assert (
        report["result"]["row_evidence_store"][
            "supports_t1_without_rerunning_outer_validation"
        ]
        is True
    )


# --- 18-24. stream-aware temporal descriptors ------------------------------


def _stream(record, channel, predictions, *, present=None, primary=None, labels=None):
    size = len(predictions)
    return EV.T2DescriptorStream(
        record_id=record,
        channel_index=channel,
        predictions=np.asarray(predictions, dtype=bool),
        score_present=np.asarray(
            [True] * size if present is None else present, dtype=bool
        ),
        primary_mask=np.asarray(
            [True] * size if primary is None else primary, dtype=bool
        ),
        labels=np.asarray([0] * size if labels is None else labels, dtype=np.int64),
    )


def test_a_run_never_crosses_a_stream_boundary():
    """Stream A ends positive, stream B starts positive: two runs, never one."""
    descriptors = EV.temporal_descriptors(
        [
            _stream("s20011", 0, [0, 1]),
            _stream("s20021", 0, [1, 0]),
        ]
    )
    assert descriptors["positive_prediction_run_count"] == 2
    assert descriptors["runs_cross_stream_boundaries"] is False
    assert descriptors["stream_count"] == 2
    # No transition is counted across the boundary either: one inside each.
    assert descriptors["transition_count"] == 2


def test_a_non_primary_negative_prediction_breaks_a_run():
    """positive / challenge-negative / positive is two runs, not one."""
    descriptors = EV.temporal_descriptors(
        [_stream("s20011", 0, [1, 0, 1], primary=[True, False, True])]
    )
    assert descriptors["positive_prediction_run_count"] == 2
    assert descriptors["primary_only_sequence_used_for_runs"] is False


def test_an_unavailable_gap_breaks_a_run():
    """positive / unavailable / positive is two runs, not one."""
    descriptors = EV.temporal_descriptors(
        [
            _stream(
                "s20011",
                0,
                [1, 0, 1],
                present=[True, False, True],
            )
        ]
    )
    assert descriptors["positive_prediction_run_count"] == 2
    assert descriptors["unavailable_gap_stitches_runs"] is False
    # And no transition is invented across the gap.
    assert descriptors["transition_count"] == 0


def test_runs_use_every_available_scored_role_not_primary_only():
    """A challenge positive between two PRIMARY positives is ONE run."""
    descriptors = EV.temporal_descriptors(
        [_stream("s20011", 0, [1, 1, 1], primary=[True, False, True])]
    )
    assert descriptors["positive_prediction_run_count"] == 1
    assert descriptors["median_positive_run_duration_seconds"] == 15.0


def test_the_transition_denominator_is_full_physical_exposure():
    """Not scored rows * stride: unavailable and non-primary time still counts."""
    descriptors = EV.temporal_descriptors(
        [
            _stream(
                "s20011",
                0,
                [1, 0, 0, 1],
                present=[True, False, True, True],
                primary=[True, False, False, True],
            )
        ]
    )
    assert descriptors["physical_exposure_seconds"] == 20.0
    assert descriptors["transition_denominator"] == ("full_physical_timeline_exposure")
    assert descriptors["transition_count_per_hour"] == pytest.approx(
        descriptors["transition_count"] / (20.0 / 3600.0)
    )


def test_labelled_positive_persistence_stays_its_own_window_statistic():
    descriptors = EV.temporal_descriptors(
        [
            _stream(
                "s20011",
                0,
                [1, 0, 1, 1],
                primary=[True, True, True, False],
                labels=[1, 1, 0, 1],
            )
        ]
    )
    assert descriptors["labelled_positive_window_count"] == 2
    assert descriptors["labelled_positive_window_prediction_fraction"] == 0.5
    assert (
        descriptors["prediction_persistence_around_labelled_ischemic_intervals"] == 0.5
    )
    assert descriptors["prediction_persistence_definition"] == (
        "fraction_of_labelled_positive_windows_predicted_positive"
    )
    assert descriptors["prediction_persistence_derived_from_run_segmentation"] is False
    assert (
        descriptors["prediction_persistence_is_episode_onset_offset_measurement"]
        is False
    )
    assert descriptors["episode_grouping_performed"] is False
    assert descriptors["formal_episode_reasoning_belongs_to"] == "t1"


def test_a_duplicate_descriptor_stream_key_is_refused():
    with pytest.raises(EV.T2EvaluationError, match="appears twice"):
        EV.temporal_descriptors([_stream("s20011", 0, [1]), _stream("s20011", 0, [1])])


def test_descriptors_are_never_a_selection_input(completed_outer):
    report, _trained, _validation = completed_outer
    for arm in T2_ARMS:
        descriptors = report["result"]["temporal_descriptors"][arm]
        assert descriptors["is_selection_input"] is False
        assert descriptors["may_alter_threshold"] is False
        assert descriptors["runs_cross_stream_boundaries"] is False
        assert descriptors["primary_only_sequence_used_for_runs"] is False
        assert descriptors["unavailable_gap_stitches_runs"] is False
        assert descriptors["stream_count"] == 4


def test_unavailable_does_not_reset_the_model_state():
    """The gap breaks a descriptive run. It does NOT touch the recurrence."""
    import torch

    from cardiosentinel.neural.t2_models import build_t2_model

    model = build_t2_model(T2_ARM_GRU)
    model.eval()
    generator = torch.Generator().manual_seed(17)
    before = torch.randn(1, 3, 146, generator=generator)
    after = torch.randn(1, 3, 146, generator=generator)
    with torch.no_grad():
        contiguous, _ = model(torch.cat([before, after], dim=1))
        _first, carried = model(before)
        resumed, _ = model(after, carried)
        _fresh_out, _ = model(after, model.initial_state(1))
    # Skipping the unavailable row and carrying the state reproduces the
    # contiguous pass; resetting the state would not.
    assert torch.allclose(contiguous[:, 3:], resumed, atol=1e-5)
    assert not torch.allclose(resumed, _fresh_out, atol=1e-6)


# --- 25-26. one pass, and delegated selection ------------------------------


def test_one_causal_pass_per_arm_supplies_every_quantity(completed_outer):
    report, _trained, _validation = completed_outer
    for arm in T2_ARMS:
        evidence = report["result"]["per_arm_evidence"][arm]
        assert evidence["single_causal_pass"] is True
        assert evidence["same_pass_supplies_primary_and_challenge"] is True
        assert evidence["same_pass_supplies_temporal_descriptors"] is True
        assert evidence["second_challenge_replay_performed"] is False
        assert evidence["second_temporal_replay_performed"] is False
        assert evidence["threshold_altered_by_outer_validation"] is False
        assert evidence["unavailable_rows_scored"] == 0
        assert evidence["pooled"]["auprc"] is not None
        assert evidence["subject_macro"]["auprc"]["value"] is not None
        assert set(evidence["challenge"]["subsets"]) == {
            "rate_related",
            "axis_shift",
            "conduction_change",
        }
        assert evidence["cold_start"]["warmup_threshold_applied"] is False
        assert report["result"]["subject_bootstrap"][arm]["replicates"] == 1000
        assert report["result"]["subject_bootstrap"][arm]["seed"] == 2026
        assert report["result"]["subject_bootstrap"][arm]["unit"] == "subject"


def test_the_outer_selection_delegates_to_the_frozen_protocol_rule(completed_outer):
    report, _trained, _validation = completed_outer
    from cardiosentinel.neural.t2_protocol import select_t2_arm

    decision = report["result"]["selection_decision"]
    replayed = select_t2_arm(
        pooled_auprc={
            arm: report["result"]["per_arm_evidence"][arm]["pooled"]["auprc"]
            for arm in T2_ARMS
        },
        subject_macro_auprc={
            arm: report["result"]["per_arm_evidence"][arm]["subject_macro"]["auprc"][
                "value"
            ]
            for arm in T2_ARMS
        },
        parameter_counts={T2_ARM_GRU: 59_521, T2_ARM_S4D: 45_313},
    )
    assert decision["selected_arm"] == replayed["selected_arm"]
    assert report["result"]["latency_used_in_selection"] is False
    assert report["result"]["challenge_used_in_selection"] is False


# --- 27. outer runtime choreography ----------------------------------------


def test_the_outer_enforcement_points_were_visited_in_the_frozen_order(
    completed_outer,
):
    _report, trained, _validation = completed_outer
    required = PS.required_outer_runtime_stage_order(T2_ARMS)
    assert required == (
        PS.STAGE_OUTER_START,
        PS.stage_pre_checkpoint_load(T2_ARM_GRU),
        PS.stage_pre_checkpoint_load(T2_ARM_S4D),
        PS.OUTER_EVIDENCE_DIRNAME,
        PS.OUTER_RESULT_NAME,
    )
    lock = json.loads(
        (
            trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID / PS.OUTER_LOCK_NAME
        ).read_text()
    )
    observed = list(lock["runtime_enforcement_stages"])
    filtered = tuple(label for label in observed if label in set(required))
    assert filtered == required
    checks = lock["runtime_identity_checks"]["checks"]
    assert checks[0]["enforcement_point"] == "start"
    assert checks[-1]["enforcement_point"] == "completion"
    assert lock["runtime_identity_checks"]["all_observations_matched"] is True


def test_a_missing_outer_enforcement_stage_is_refused():
    from cardiosentinel.neural.runtime_sentinel import (
        RuntimeCheck,
        RuntimeIntegrityRecord,
    )

    record = RuntimeIntegrityRecord()
    for detail in (PS.STAGE_OUTER_START, PS.OUTER_RESULT_NAME):
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
    with pytest.raises(PS.T2PersistenceError, match="outer runtime enforcement"):
        PS.require_outer_runtime_stage_order(record, T2_ARMS)


def test_both_arms_scored_on_the_same_runtime_and_device(completed_outer):
    report, _trained, _validation = completed_outer
    runtimes = [report["result"]["per_arm_evidence"][arm]["runtime"] for arm in T2_ARMS]
    PS.require_single_runtime(runtimes[0], runtimes[1])
    for record in runtimes:
        PS.require_execution_device_agreement(record, record["model_parameter_device"])


# --- 28. TEST firewall -----------------------------------------------------


def test_test_remains_refused_everywhere(completed_outer):
    report, trained, _validation = completed_outer
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.refuse_sealed_partition("test")
    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    for name in (PS.OUTER_RESULT_NAME, PS.OUTER_LOCK_NAME, PS.OUTER_STATUS_NAME):
        payload = json.loads((run_dir / name).read_text())
        assert payload["test_accessed"] is False, name
        assert payload["sealed_test_state"] == "unopened", name
    _root, manifest = _store(trained)
    assert manifest["test_rows_present"] is False
    assert report["result"]["test_accessed"] is False
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    assert "--test" not in options
