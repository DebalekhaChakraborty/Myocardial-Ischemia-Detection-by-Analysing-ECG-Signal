"""Git-identity, execution-device and public-API provenance closure.

Four narrow properties, each of which a well-formed artifact could previously
have violated without any single check noticing:

* one authorized commit binds the whole TRAIN attempt, and HEAD moving after
  the claim consumes it rather than producing a result and a lock written at
  different commits;
* the top-level TRAIN lock proves the model actually ran on the device it
  names, and cross-binds to both arms;
* the public canonical outer function takes the authorized commit and nothing
  else -- no source override reaches it, and flipping the activation constant
  unlocks no raw VALIDATION loader;
* an outer failure after row-evidence promotion records that evidence's exact
  manifest digest, because the receipt is the only record a consumed attempt
  leaves.

No real science. Synthetic fixtures only.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t2_development_run as RUN
from cardiosentinel.neural import t2_evaluation as EV
from cardiosentinel.neural import t2_outer_evidence as ES
from cardiosentinel.neural import t2_persistence as PS
from cardiosentinel.neural import t2_timeline as TL
from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.t2_protocol import T2_ARM_GRU, T2_ARM_S4D, T2_ARMS
from tests.neural.test_t2_canonical_training_route import (  # noqa: F401
    _PRE_CLAIM_REFUSAL,
    GIT_SHA,
    clean_git,
    environment,
    frozen_runtime,
)
from tests.neural.test_t2_outer_validation_governance import (  # noqa: F401
    _train_checks,
    trained,
    validation,
)

DRIFTED_SHA = "d" * 40


def _sources(tmp_path, environment):  # noqa: F811
    return RUN.T2TrainingSources(
        run_root=tmp_path / "runs",
        stream_cache_root=environment.stream_cache_root,
        corpus_manifest=environment.corpus_manifest,
        canonical=False,
    )


# --- 1-3. Git drift after the claim ----------------------------------------


def test_head_moving_after_the_claim_consumes_the_attempt(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
    monkeypatch,
):
    """Preflight authorized SHA A; HEAD is cleanly at SHA B by promotion."""
    sources = _sources(tmp_path, environment)
    checks = _train_checks()
    assert checks["authorized_git_sha"] == GIT_SHA

    drifted = {"git_sha": DRIFTED_SHA, "git_dirty": False}
    monkeypatch.setattr(PS, "git_provenance", lambda _root: dict(drifted))

    with pytest.raises(PS.T2PersistenceError, match="HEAD moved"):
        RUN._execute_training_attempt(checks, sources)

    run_dir = sources.run_root / sources.attempt_id
    assert run_dir.is_dir(), "the claim is consumed, not rolled back"
    assert not (run_dir / PS.RESULT_NAME).exists()
    assert not (run_dir / PS.EXPERIMENT_LOCK_NAME).exists()
    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["automatic_retry_performed"] is False
    assert status["repeat_attempt_permitted"] is False
    receipt = json.loads(
        (
            PS.t2_review_directory(sources.run_root, sources.attempt_id)
            / PS.FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["attempt_consumed"] is True
    assert "HEAD moved" in receipt["exception_message"]
    with pytest.raises(PS.T2PersistenceError, match="No canonical T2 artifact"):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        PS.require_unclaimed_t2_attempt(sources.run_root, sources.attempt_id)


def test_a_tree_that_becomes_dirty_after_the_claim_consumes_the_attempt(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
    monkeypatch,
):
    sources = _sources(tmp_path, environment)
    monkeypatch.setattr(
        PS, "git_provenance", lambda _root: {"git_sha": GIT_SHA, "git_dirty": True}
    )
    with pytest.raises(PS.T2PersistenceError, match="became dirty"):
        RUN._execute_training_attempt(_train_checks(), sources)
    run_dir = sources.run_root / sources.attempt_id
    assert not (run_dir / PS.RESULT_NAME).exists()
    assert not (run_dir / PS.EXPERIMENT_LOCK_NAME).exists()


def test_the_authorized_git_identity_check_in_isolation(monkeypatch):
    monkeypatch.setattr(
        PS, "git_provenance", lambda _root: {"git_sha": GIT_SHA, "git_dirty": False}
    )
    identity = PS.require_authorized_git_identity(GIT_SHA)
    assert identity["authorized_git_sha"] == GIT_SHA
    assert identity["git_sha"] == GIT_SHA
    assert identity["git_identity_reverified_before_promotion"] is True
    with pytest.raises(PS.T2PersistenceError, match="HEAD moved"):
        PS.require_authorized_git_identity(DRIFTED_SHA)
    with pytest.raises(PS.T2PersistenceError, match="Not a Git commit identity"):
        PS.require_authorized_git_identity("nope")


# --- 4-5. one commit across every artifact ---------------------------------


def test_result_lock_and_authorized_sha_are_identical_on_the_normal_path(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    sources = _sources(tmp_path, environment)
    report = RUN._execute_training_attempt(_train_checks(), sources)
    run_dir = sources.run_root / sources.attempt_id
    result = json.loads((run_dir / PS.RESULT_NAME).read_text())
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    assert (
        result["git_sha"]
        == result["authorized_git_sha"]
        == lock["git_sha"]
        == lock["authorized_git_sha"]
        == GIT_SHA
    )
    assert report["authorized_git_sha"] == GIT_SHA
    verification = PS.validate_canonical_t2_attempt(
        sources.run_root, sources.attempt_id
    )
    assert verification["authorized_git_sha"] == GIT_SHA
    assert verification["git_identity_verified"] is True


def test_a_result_and_lock_at_different_commits_are_refused(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    """Both artifacts independently well formed; their commits disagree.

    Every digest is repaired afterwards, so nothing else is wrong with either
    file: the artifact digests match, both self-digests validate, and the
    result's own two SHAs agree with each other. The only remaining discrepancy
    is that the result and the lock name different commits.
    """
    from cardiosentinel.data.provenance import sha256_file

    sources = _sources(tmp_path, environment)
    RUN._execute_training_attempt(_train_checks(), sources)
    run_dir = sources.run_root / sources.attempt_id

    result_path = run_dir / PS.RESULT_NAME
    result = json.loads(result_path.read_text())
    result["git_sha"] = DRIFTED_SHA
    result["authorized_git_sha"] = DRIFTED_SHA
    result_path.write_text(json.dumps(result))
    PS.validate_t2_result_payload(result)

    lock_path = run_dir / PS.EXPERIMENT_LOCK_NAME
    lock = json.loads(lock_path.read_text())
    lock["artifact_sha256"][PS.RESULT_NAME] = sha256_file(result_path)
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    body["experiment_lock_sha256"] = canonical_sha256(body)
    lock_path.write_text(json.dumps(body))
    PS.validate_t2_run_lock(json.loads(lock_path.read_text()), run_dir=run_dir)

    with pytest.raises(PS.T2PersistenceError, match="authorized for"):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)


def test_a_lock_whose_own_two_shas_disagree_is_refused():
    lock = {
        "lock_class": PS.LOCK_CLASS,
        "git_sha": DRIFTED_SHA,
        "authorized_git_sha": GIT_SHA,
        "git_dirty": False,
    }
    lock["experiment_lock_sha256"] = canonical_sha256(
        {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    )
    with pytest.raises(PS.T2PersistenceError):
        PS.validate_t2_run_lock(lock)


# --- 6-8. top-level execution-device provenance ----------------------------


def test_the_top_level_lock_carries_the_observed_model_device(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    sources = _sources(tmp_path, environment)
    RUN._execute_training_attempt(_train_checks(), sources)
    run_dir = sources.run_root / sources.attempt_id
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    proof = lock["execution_device_proof"]
    assert proof["execution_device_agrees"] is True
    assert proof["derived_from"] == "both_arm_observed_parameter_devices"
    assert PS._same_device(
        proof["declared_execution_device"], proof["model_parameter_device"]
    )
    assert sorted(lock["per_arm_execution_device_proof"]) == sorted(T2_ARMS)
    result = json.loads((run_dir / PS.RESULT_NAME).read_text())
    assert result["execution_device_proof"] == proof


def test_every_artifact_that_names_a_device_names_the_same_one(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    sources = _sources(tmp_path, environment)
    RUN._execute_training_attempt(_train_checks(), sources)
    verification = PS.validate_canonical_t2_attempt(
        sources.run_root, sources.attempt_id
    )
    device = verification["execution_device_proof"]
    assert device["execution_device_agrees"] is True
    assert set(device["cross_bound_artifacts"]) == {
        "experiment lock",
        f"{T2_ARM_GRU} arm result",
        f"{T2_ARM_GRU} checkpoint lock",
        f"{T2_ARM_S4D} arm result",
        f"{T2_ARM_S4D} checkpoint lock",
    }


def test_a_top_level_proof_claiming_cuda_over_a_cpu_model_is_refused():
    lying = {
        "declared_execution_device": "cuda:0",
        "model_parameter_device": "cpu",
        "execution_device_agrees": True,
    }
    with pytest.raises(PS.T2PersistenceError, match="did not run on"):
        PS.require_execution_device_cross_binding(lying, {})


def test_two_arms_on_different_devices_are_refused():
    with pytest.raises(PS.T2PersistenceError, match="different devices"):
        PS.build_execution_device_proof(
            {
                T2_ARM_GRU: {
                    "declared_execution_device": "cpu",
                    "model_parameter_device": "cpu",
                    "execution_device_agrees": True,
                },
                T2_ARM_S4D: {
                    "declared_execution_device": "cuda:0",
                    "model_parameter_device": "cuda:0",
                    "execution_device_agrees": True,
                },
            }
        )
    with pytest.raises(PS.T2PersistenceError, match="both frozen arms"):
        PS.build_execution_device_proof({T2_ARM_GRU: {}})


def test_a_repaired_self_digest_cannot_hide_a_false_top_level_device(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    """The lock is internally perfect again. The arm artifacts still disagree."""
    sources = _sources(tmp_path, environment)
    RUN._execute_training_attempt(_train_checks(), sources)
    run_dir = sources.run_root / sources.attempt_id
    path = run_dir / PS.EXPERIMENT_LOCK_NAME
    lock = json.loads(path.read_text())
    lock["execution_device_proof"] = {
        "declared_execution_device": "cuda:0",
        "model_parameter_device": "cuda:0",
        "execution_device_agrees": True,
        "derived_from": "both_arm_observed_parameter_devices",
    }
    lock["per_arm_execution_device_proof"] = {
        arm: dict(lock["execution_device_proof"]) for arm in T2_ARMS
    }
    body = {k: v for k, v in lock.items() if k != "experiment_lock_sha256"}
    lock["experiment_lock_sha256"] = canonical_sha256(body)
    path.write_text(
        json.dumps(body | {"experiment_lock_sha256": lock["experiment_lock_sha256"]})
    )
    # Self-consistent, and still refused: the arm results and checkpoint locks
    # record the device the science actually ran on.
    with pytest.raises(PS.T2PersistenceError, match="canonical attempt"):
        PS.validate_canonical_t2_attempt(sources.run_root, sources.attempt_id)


def test_the_normal_synthetic_cpu_path_verifies(
    tmp_path,
    environment,  # noqa: F811
    clean_git,  # noqa: F811
):
    sources = _sources(tmp_path, environment)
    RUN._execute_training_attempt(_train_checks(), sources)
    verification = PS.validate_canonical_t2_attempt(
        sources.run_root, sources.attempt_id
    )
    assert verification["verified"] is True
    assert verification["execution_device_proof"]["model_parameter_device"] == "cpu"


# --- 9-13. the public canonical outer API ----------------------------------


def test_the_public_outer_signature_is_exactly_the_authorized_commit():
    parameters = inspect.signature(EV.execute_canonical_outer_validation).parameters
    assert list(parameters) == ["expected_git_sha"]
    assert list(
        inspect.signature(RUN.execute_canonical_outer_validation).parameters
    ) == ["expected_git_sha"]


@pytest.mark.parametrize(
    "forbidden",
    [
        "validation_root",
        "corpus_manifest",
        "run_root",
        "training_attempt_id",
        "device",
        "threshold",
        "arm",
        "retry",
        "force",
    ],
)
def test_no_source_override_reaches_the_public_canonical_outer_route(forbidden):
    for entry in (
        EV.execute_canonical_outer_validation,
        RUN.execute_canonical_outer_validation,
    ):
        assert forbidden not in inspect.signature(entry).parameters
        with pytest.raises(TypeError):
            entry(GIT_SHA, **{forbidden: "injected"})


def test_the_private_worker_still_accepts_fixture_injection():
    """Testability is not weakened; it is moved off the canonical surface."""
    parameters = inspect.signature(EV._outer_validation_worker).parameters
    assert set(parameters) >= {
        "expected_git_sha",
        "run_root",
        "training_attempt_id",
        "validation_root",
        "corpus_manifest",
    }
    for name in ("run_root", "training_attempt_id", "validation_root"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_public_entry_points_contain_no_raw_loader():
    assert EV.OUTER_VALIDATION_ENTRY_POINTS == (EV.execute_canonical_outer_validation,)
    assert not hasattr(EV, "open_validation_timeline")
    assert not hasattr(EV, "load_validation_labels")
    # The helpers still exist, privately, inside the claim-bearing route.
    assert callable(EV._open_validation_timeline)
    assert callable(EV._load_validation_targets)


def test_flipping_activation_alone_unlocks_no_raw_loader():
    """The only public thing activation can reach is the claim-bearing route."""
    public = [
        name
        for name in dir(EV)
        if not name.startswith("_")
        and callable(getattr(EV, name))
        and "validation" in name.lower()
        and name.startswith(("open", "load", "read", "score"))
    ]
    assert public == [], public


def test_activation_true_still_opens_nothing_unauthorized(monkeypatch):
    """Activation is open, so the authorized commit is what refuses.

    `GIT_SHA` is synthetic and `git_provenance` is not patched here, so the
    checkout provably is not at it and the route stops in
    `require_expected_git_sha` -- before the outer claim, before any VALIDATION
    path is resolved.
    """
    opened: list[object] = []
    monkeypatch.setattr(
        EV, "_open_validation_timeline", lambda *a, **k: opened.append(a)
    )
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True
    for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
        with pytest.raises(RUN.T2RunError, match=_PRE_CLAIM_REFUSAL):
            entry(GIT_SHA)
    with pytest.raises(RUN.T2RunError, match=_PRE_CLAIM_REFUSAL):
        RUN.execute_canonical_outer_validation(GIT_SHA)
    assert RUN.main(["--execute-canonical-outer-validation"]) == 2
    assert opened == []
    assert not (PS.T2_RUN_ROOT / PS.T2_OUTER_VALIDATION_ATTEMPT_ID).exists()


# --- 14-16. the outer failure receipt after row-evidence promotion ---------


def test_a_failure_after_row_evidence_promotion_records_its_digest(
    trained,  # noqa: F811
    validation,  # noqa: F811
    monkeypatch,
):
    """The receipt is the only record a consumed attempt leaves."""

    def failing(*_args, **_kwargs):
        raise RuntimeError("synthetic post-row-evidence failure")

    monkeypatch.setattr(PS, "finalize_and_promote_t2_outer_result", failing)
    with pytest.raises(RuntimeError, match="post-row-evidence"):
        EV._outer_validation_worker(
            GIT_SHA,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )

    run_dir = trained.run_root / PS.T2_OUTER_VALIDATION_ATTEMPT_ID
    evidence_root = run_dir / PS.OUTER_EVIDENCE_DIRNAME
    manifest_path = evidence_root / ES.T2_OUTER_STORE_MANIFEST_NAME
    assert manifest_path.is_file(), "row evidence was promoted before the failure"

    assert not (run_dir / PS.OUTER_RESULT_NAME).exists()
    assert not (run_dir / PS.OUTER_LOCK_NAME).exists()
    status = json.loads((run_dir / PS.OUTER_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED

    receipt = json.loads(
        (
            PS.t2_review_directory(trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID)
            / PS.OUTER_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    key = f"{PS.OUTER_EVIDENCE_DIRNAME}/{ES.T2_OUTER_STORE_MANIFEST_NAME}"
    from cardiosentinel.data.provenance import sha256_file

    expected = sha256_file(manifest_path)
    assert receipt["exposure"]["row_evidence_promoted"] is True
    assert key in receipt["promoted_artifacts"]
    assert receipt["promoted_artifacts"][key] == expected
    assert receipt["row_evidence_manifest_sha256"] == expected
    assert receipt["attempt_consumed"] is True
    assert receipt["automatic_retry_performed"] is False
    assert receipt["selective_arm_rerun_permitted"] is False
    assert receipt["alternate_attempt_name_permitted"] is False
    assert receipt["exposure"]["arms_scored"] == list(T2_ARMS)

    # Consumed, and not COMPLETE.
    with pytest.raises(PS.T2PersistenceError, match="No canonical outer artifact"):
        PS.validate_canonical_t2_outer_validation_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        PS.require_unclaimed_outer_attempt(
            trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID
        )


def test_an_early_outer_failure_records_no_row_evidence_digest(
    trained,  # noqa: F811
    validation,  # noqa: F811
    monkeypatch,
):
    """The counterpart: nothing promoted, nothing claimed to have been."""
    from cardiosentinel.neural import t2_training as TR

    def failing(*_args, **_kwargs):
        raise RuntimeError("synthetic early failure")

    monkeypatch.setattr(TR, "score_streams", failing)
    with pytest.raises(RuntimeError, match="early failure"):
        EV._outer_validation_worker(
            GIT_SHA,
            run_root=trained.run_root,
            training_attempt_id=trained.attempt_id,
            validation_root=validation.stream_cache_root,
            corpus_manifest=validation.corpus_manifest,
        )
    receipt = json.loads(
        (
            PS.t2_review_directory(trained.run_root, PS.T2_OUTER_VALIDATION_ATTEMPT_ID)
            / PS.OUTER_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["exposure"]["row_evidence_promoted"] is False
    assert receipt["row_evidence_manifest_sha256"] is None


# --- 17-19. nothing frozen moved -------------------------------------------


def test_the_frozen_documents_are_byte_identical():
    from cardiosentinel.neural.t2_protocol import (
        T2_PROTOCOL_SHA256,
        validate_t2_protocol_document,
    )

    assert validate_t2_protocol_document() == T2_PROTOCOL_SHA256
    assert T2_PROTOCOL_SHA256 == (
        "6546086a55fe2c9c109f4121cdb6b42d4d53ce0112c9611eb895bd8c805cfefb"
    )
    assert PS.validate_t2_execution_spec() == PS.T2_EXECUTION_SPEC_SHA256
    assert PS.T2_EXECUTION_SPEC_SHA256 == (
        "af6ebf1a6314edb86cce7aa88a6260dd1bd155fd0aebe472d3745b6c823b8054"
    )


def test_nothing_previously_closed_was_reopened():
    assert PS.T2_RUN_ROOT.is_absolute()
    assert PS.T2_RUN_ROOT.parts[-2:] == (
        "cardiosentinel-runs",
        "phase8-t2-development-v1",
    )
    assert PS.T2_TRAINING_ATTEMPT_ID == "t2-v1-training"
    assert PS.T2_OUTER_VALIDATION_ATTEMPT_ID == "t2-v1-outer-validation"
    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is True
    assert ES.T2_SCORE_SEMANTICS == "uncalibrated_temporal_model_score"
    assert ES.T2_SCORE_DEFINITION == "sigmoid(current_window_t2_logit)"
    assert list(ES.T2_OUTER_IDENTITY_COLUMNS).count("score_present") == 1
    assert "primary_mask" in ES.T2_OUTER_IDENTITY_COLUMNS


def test_test_remains_structurally_refused():
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.refuse_sealed_partition("test")
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.require_frozen_stream_identity("test", {})
    options = {
        option
        for action in RUN.build_parser()._actions
        for option in action.option_strings
    }
    assert "--test" not in options
    assert TL.PERMITTED_PARTITIONS == ("train", "validation")
    for module in (PS, RUN, EV, ES):
        source = Path(module.__file__).read_text()
        assert 'partition = "test"' not in source
