"""The one-shot outer-VALIDATION attempt, and the activated gate around it.

`T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED` is now True. That changes what these
tests may safely do, and the change is deliberate in both directions:

* The body behind the gate is unchanged -- claim, per-row evidence, accounting,
  stream-aware descriptors, failure semantics, canonical validator -- and is
  still exercised only against synthetic on-disk fixtures.
* The gate tests no longer assert "activation refuses". They assert what refuses
  *instead*, now that activation does not: the authorized commit, and the
  reviewed-TRAIN binding. Both refuse before the outer claim and before any
  VALIDATION path is resolved, and the loader is spied to prove it.

**No real science happens here.** No canonical CLI invocation against the real
authorized commit, no real TRAIN optimiser step, no real internal-dev score, no
real threshold, no real VALIDATION per-row access, no real challenge evidence,
no TEST. Every timeline is a synthetic on-disk fixture in `tmp_path`.

**The one thing this file must never do**, now that the gate is open, is drive
the claim-bearing canonical route with the repository's actual authorized commit
against the real run root: that combination is capable of consuming the one-shot
outer attempt. `test_no_test_can_consume_the_real_one_shot_outer_attempt` is a
structural guard that fails if any test in this suite grows that shape.

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
    _PRE_CLAIM_REFUSAL,
    GIT_SHA,
    clean_git,
    environment,
    frozen_runtime,
    outer_attempt_unchanged,
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


def test_the_public_outer_route_opens_nothing_without_the_authorized_commit(
    monkeypatch,
):
    """Activation is open; the authorized commit is what refuses now.

    `GIT_SHA` is the synthetic "a"*40 and `git_provenance` is deliberately NOT
    patched here, so the real checkout provably is not at it. The refusal
    therefore comes from `require_expected_git_sha`, which runs before the outer
    claim and before any VALIDATION path is resolved.
    """
    opened: list[object] = []
    monkeypatch.setattr(
        EV, "_open_validation_timeline", lambda *a, **k: opened.append(a)
    )
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True
    with outer_attempt_unchanged():
        for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
            with pytest.raises(RUN.T2RunError, match=_PRE_CLAIM_REFUSAL):
                entry(GIT_SHA)
        with pytest.raises(RUN.T2RunError, match=_PRE_CLAIM_REFUSAL):
            RUN.execute_canonical_outer_validation(GIT_SHA)
        # The bare CLI flag names no commit, so it stops rather than claiming.
        assert RUN.main(["--execute-canonical-outer-validation"]) == 2
    assert opened == [], "nothing was resolved or opened before the refusal"


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
    """Activation is True, but the gate call stays where it is.

    A flipped constant is reviewable; a deleted gate is not. If activation is
    ever reverted, the refusal must still fire before anything is resolved.
    """
    import ast
    import inspect
    import textwrap

    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True
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


# --- 8b. the activation change set itself ----------------------------------


def _reviewed_verification(**overrides):
    """Exactly what the canonical verifier reports for the reviewed attempt."""
    base = {
        "verified": True,
        "result_sha256": PS.T2_REVIEWED_TRAIN_RESULT_SHA256,
        "experiment_lock_sha256": PS.T2_REVIEWED_TRAIN_EXPERIMENT_LOCK_SELF_SHA256,
        "authorized_git_sha": PS.T2_REVIEWED_TRAIN_AUTHORIZED_GIT_SHA,
        "arm_selection_status": PS.ARM_SELECTION_PENDING,
        "arm_selected": None,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    base.update(overrides)
    return base


def test_the_activation_constant_is_exactly_true():
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True


def test_the_activation_decision_document_validates_to_its_frozen_sha():
    digest = PS.validate_t2_train_artifact_review_document()
    assert digest == PS.T2_TRAIN_ARTIFACT_REVIEW_SHA256
    assert PS.T2_TRAIN_ARTIFACT_REVIEW_PATH.is_file()


def test_a_mutated_activation_decision_document_is_refused(tmp_path):
    forged = tmp_path / "forged_review.md"
    forged.write_text(
        PS.T2_TRAIN_ARTIFACT_REVIEW_PATH.read_text() + "\nan added line\n"
    )
    with pytest.raises(PS.T2PersistenceError, match="immutable"):
        PS.validate_t2_train_artifact_review_document(forged)


def test_the_reviewed_train_constants_are_the_canonical_ones():
    assert PS.T2_REVIEWED_TRAIN_RESULT_SHA256 == (
        "ff9258f95631405b6705811d638d754400a067be4c1a43bb9d52021bb246adb8"
    )
    assert PS.T2_REVIEWED_TRAIN_EXPERIMENT_LOCK_SELF_SHA256 == (
        "d8de03554931fe65a6f1c1242d80c1c95f1a6a26f93b8013cff5bc221a92202f"
    )
    assert PS.T2_REVIEWED_TRAIN_AUTHORIZED_GIT_SHA == (
        "f4759e2a97d17db26cb6a6b7c0e9b6207eb0b045"
    )


def test_the_reviewed_binding_accepts_the_exact_reviewed_attempt():
    binding = PS.require_reviewed_t2_training_attempt(_reviewed_verification())
    assert binding["binding_class"] == "t2_reviewed_training_attempt_binding"
    assert binding["review_document_sha256"] == PS.T2_TRAIN_ARTIFACT_REVIEW_SHA256
    assert binding["arm_selected"] is None
    assert binding["arm_selection_status"] == PS.ARM_SELECTION_PENDING
    assert binding["test_accessed"] is False
    assert binding["sealed_test_state"] == "unopened"


def test_an_unverified_train_attempt_is_refused():
    with pytest.raises(PS.T2ActivationError, match="did not verify"):
        PS.require_reviewed_t2_training_attempt(_reviewed_verification(verified=False))


def test_a_mismatched_train_result_digest_is_refused():
    with pytest.raises(PS.T2ActivationError, match="top-level result digest"):
        PS.require_reviewed_t2_training_attempt(
            _reviewed_verification(result_sha256="c" * 64)
        )


def test_a_mismatched_train_experiment_lock_identity_is_refused():
    with pytest.raises(PS.T2ActivationError, match="experiment-lock self-digest"):
        PS.require_reviewed_t2_training_attempt(
            _reviewed_verification(experiment_lock_sha256="d" * 64)
        )


def test_a_wrong_authorized_train_commit_is_refused():
    with pytest.raises(PS.T2ActivationError, match="authorized TRAIN commit"):
        PS.require_reviewed_t2_training_attempt(
            _reviewed_verification(authorized_git_sha="e" * 40)
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"arm_selection_status": "selected"}, "arm_selection_status"),
        ({"arm_selected": T2_ARM_GRU}, "already names a selected arm"),
        ({"test_accessed": True}, "TEST as accessed"),
        ({"sealed_test_state": "opened"}, "sealed_test_state"),
    ],
)
def test_a_non_pending_or_test_opened_train_state_is_refused(overrides, match):
    with pytest.raises(PS.T2ActivationError, match=match):
        PS.require_reviewed_t2_training_attempt(_reviewed_verification(**overrides))


def test_the_canonical_outer_preflight_binds_the_reviewed_attempt():
    """The binding is wired into the canonical path, not merely available."""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(EV._outer_validation_preflight))
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "validate_canonical_t2_attempt" in calls
    assert "require_reviewed_t2_training_attempt" in calls
    # ... and it is bound after canonical verification, never before it.
    assert calls.index("validate_canonical_t2_attempt") < calls.index(
        "require_reviewed_t2_training_attempt"
    )


def test_activation_added_no_alternate_mechanism():
    import ast

    source = Path(PS.__file__).read_text()
    for bypass in ("os.environ", "getenv", "setattr(", "--force", "--retry"):
        assert bypass not in source, bypass
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign | ast.Assign)
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        )
        if getattr(target, "id", None) == "T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED"
    ]
    assert len(assignments) == 1
    # Registered options, not a source scan: `--force` and `--retry` appear in
    # this module as a *denylist* and in prose explaining why they are refused,
    # and a substring test would read those as the very thing they prevent.
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    for flag in ("--activate", "--authorize", "--force", "--retry", "--reset"):
        assert flag not in options, flag
    assert options == {
        "-h",
        "--help",
        "--execute-canonical-training",
        "--execute-canonical-outer-validation",
        "--expected-git-sha",
    }


def test_no_test_can_consume_the_real_one_shot_outer_attempt():
    """Structural guard: no test may drive the claim-bearing route for real.

    After activation the dangerous shape is a claim-bearing outer call whose
    commit argument is the repository's actual authorized SHA and whose run root
    is the real one. Every legitimate test either injects a synthetic root
    through the private worker or passes a commit the checkout is not at. This
    walks the suite's own source and fails if that shape ever appears.
    """
    import ast

    authorized = PS.T2_REVIEWED_TRAIN_AUTHORIZED_GIT_SHA
    suite = Path(__file__).parent
    offenders: list[str] = []
    for path in sorted(suite.glob("test_t2_*.py")):
        text = path.read_text()
        # The real authorized commit must not appear as a call argument at all
        # in the test suite; the reviewed-constant assertions use the literal
        # only in comparisons, which carry no execution risk.
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name not in {
                "execute_canonical_outer_validation",
                "_outer_validation_worker",
                "_outer_validation_preflight",
            }:
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Constant) and arg.value == authorized:
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    assert offenders == [], (
        "a test drives the claim-bearing outer route with the real authorized "
        f"commit and could consume the one-shot attempt: {offenders}"
    )


def permitted_run_root_directories() -> frozenset[str]:
    """The only directory names the T2 run root may ever hold.

    Two are claim-bearing: the attempts themselves, whose existence *is* the
    claim. Two are additive forensic siblings carrying failure receipts outside
    a consumed claim, and they are legitimate by design -- an earlier version of
    this test wrongly forbade them.

    The review names are derived through `t2_review_directory` rather than
    rebuilt from a hard-coded suffix, so this cannot drift away from the
    persistence layer's own naming.
    """
    names = set(PS.T2_ATTEMPT_IDS)
    names.update(
        PS.t2_review_directory(PS.T2_RUN_ROOT, attempt).name
        for attempt in PS.T2_ATTEMPT_IDS
    )
    return frozenset(names)


def test_the_run_root_permits_only_the_frozen_attempt_and_review_names():
    """No recovery1, no retry, no numbered sibling, no invented attempt.

    This replaces an earlier assertion that the outer attempt did not exist.
    That held while the gate was closed and is deliberately no longer true: the
    authorized one-shot run has since consumed it. What remains invariant -- and
    is the property that actually protects the science -- is which names may
    appear beside it.
    """
    assert PS.T2_OUTER_VALIDATION_ATTEMPT_ID == "t2-v1-outer-validation"
    assert PS.T2_ATTEMPT_IDS == (
        PS.T2_TRAINING_ATTEMPT_ID,
        PS.T2_OUTER_VALIDATION_ATTEMPT_ID,
    )
    permitted = permitted_run_root_directories()
    assert permitted == {
        "t2-v1-training",
        "t2-v1-outer-validation",
        "t2-v1-training__review",
        "t2-v1-outer-validation__review",
    }
    # The claim-bearing pair and the forensic pair are disjoint, and a review
    # directory is never itself a claim.
    assert set(PS.T2_ATTEMPT_IDS) < permitted
    for attempt in PS.T2_ATTEMPT_IDS:
        review = PS.t2_review_directory(PS.T2_RUN_ROOT, attempt)
        assert review.name.endswith(PS.REVIEW_SUFFIX)
        assert review.name not in PS.T2_ATTEMPT_IDS

    if PS.T2_RUN_ROOT.is_dir():
        present = {p.name for p in PS.T2_RUN_ROOT.iterdir() if p.is_dir()}
        assert present <= permitted, sorted(present - permitted)


@pytest.mark.parametrize(
    "sibling",
    [
        "t2-v1-outer-validation-recovery1",
        "t2-v1-outer-validation-retry",
        "t2-v1-outer-validation-2",
        "t2-v1-outer-validation__retry",
        "t2-v1-training-recovery1",
        "some-random-attempt",
    ],
)
def test_an_invented_sibling_name_is_not_permitted(sibling):
    assert sibling not in permitted_run_root_directories()


@pytest.mark.parametrize(
    "permitted",
    [
        "t2-v1-training",
        "t2-v1-outer-validation",
        "t2-v1-training__review",
        "t2-v1-outer-validation__review",
    ],
)
def test_the_frozen_attempt_and_review_names_are_permitted(permitted):
    assert permitted in permitted_run_root_directories()


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
