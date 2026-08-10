"""Regression tests for the RUN_STATUS experiment-identity defect.

The shared `_write_status` helper previously stamped the historical B4-A
constant onto every candidate's RUN_STATUS.json, so B4-B and B4-C heartbeats
claimed `B4_raw_compact_cnn_v1`. The authoritative artifacts (manifest, epoch
history, checkpoint, experiment lock) were always correct; only the heartbeat
was wrong.

These tests pin the corrected behaviour for future runs. They deliberately do
not touch historical artifacts, which keep their original values.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import candidate_experiment as candidate_runner
from cardiosentinel.neural import experiment as runner
from cardiosentinel.neural.candidate_experiment import (
    B4B_EXPERIMENT_ID,
    B4C_EXPERIMENT_ID,
)
from cardiosentinel.neural.experiment import (
    EXPERIMENT_ID as B4A_EXPERIMENT_ID,
)
from cardiosentinel.neural.experiment import (
    RUN_STATUS_NAME,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
    _write_status,
)


@pytest.mark.parametrize(
    "experiment_id",
    [B4A_EXPERIMENT_ID, B4B_EXPERIMENT_ID, B4C_EXPERIMENT_ID],
)
@pytest.mark.parametrize(
    "status", [STATUS_RUNNING, STATUS_COMPLETE, STATUS_FAILED]
)
def test_status_records_the_calling_experiment(
    tmp_path, experiment_id, status
) -> None:
    payload = _write_status(
        tmp_path, status, experiment_id=experiment_id, command="unit-test"
    )
    written = json.loads((tmp_path / RUN_STATUS_NAME).read_text())
    assert payload == written
    assert written["experiment_id"] == experiment_id
    assert written["status"] == status
    assert written["command"] == "unit-test"
    assert "updated_at" in written


def test_status_identity_is_required_and_keyword_only() -> None:
    """A default is what caused the defect; there must not be one."""
    parameters = inspect.signature(_write_status).parameters
    identity = parameters["experiment_id"]
    assert identity.kind is inspect.Parameter.KEYWORD_ONLY
    assert identity.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        _write_status(Path("."), STATUS_RUNNING)  # type: ignore[call-arg]


def test_extra_status_fields_still_pass_through(tmp_path) -> None:
    """The lifecycle payload shape is unchanged apart from the identity."""
    _write_status(
        tmp_path,
        STATUS_FAILED,
        experiment_id=B4C_EXPERIMENT_ID,
        command="unit-test",
        error_type="ValueError",
        human_review_required=True,
        automatic_restart_performed=False,
    )
    written = json.loads((tmp_path / RUN_STATUS_NAME).read_text())
    assert written["error_type"] == "ValueError"
    assert written["human_review_required"] is True
    assert written["automatic_restart_performed"] is False


def _status_call_identities(module) -> list[str]:
    """Collect the `experiment_id=` argument of every `_write_status` call."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    identities: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "_write_status"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        assert "experiment_id" in keywords, "a _write_status call omits the identity"
        identities.append(ast.unparse(keywords["experiment_id"]))
    return identities


def test_b4a_runner_always_writes_its_own_identity() -> None:
    identities = _status_call_identities(runner)
    assert identities
    assert set(identities) == {"EXPERIMENT_ID"}


def test_candidate_runner_writes_the_resolved_candidate_identity() -> None:
    identities = _status_call_identities(candidate_runner)
    assert identities
    assert set(identities) == {"experiment_id"}
    # The historical constant must not be reachable from the candidate runner.
    source = Path(candidate_runner.__file__).read_text(encoding="utf-8")
    assert "B4_raw_compact_cnn_v1" not in source


def test_candidate_identities_are_unchanged() -> None:
    assert B4A_EXPERIMENT_ID == "B4_raw_compact_cnn_v1"
    assert B4B_EXPERIMENT_ID == "B4B_cnn_transformer_v1"
    assert B4C_EXPERIMENT_ID == "B4C_cnn_ssm_v1"


def test_training_semantics_are_untouched_by_the_status_fix() -> None:
    """The fix is metadata-only: no scientific knob may have moved."""
    from cardiosentinel.neural.training import (
        BATCH_SIZE,
        EARLY_STOPPING_DELTA,
        EARLY_STOPPING_PATIENCE,
        MAX_EPOCHS,
    )

    configuration = runner.training_configuration()
    assert configuration["seed"] == 2026
    assert configuration["loss"] == "BCEWithLogitsLoss(reduction=mean)"
    assert configuration["optimizer"] == "AdamW"
    assert configuration["learning_rate"] == 1e-3
    assert configuration["weight_decay"] == 1e-4
    assert configuration["batch_size"] == BATCH_SIZE == 256
    assert configuration["max_epochs"] == MAX_EPOCHS == 15
    assert configuration["scheduler"] is None
    assert configuration["class_weighting"] is None
    assert configuration["augmentation"] is None
    assert configuration["mixed_precision"] is False
    assert EARLY_STOPPING_PATIENCE == 4
    assert EARLY_STOPPING_DELTA == 1e-6


def test_candidate_specifications_are_unchanged() -> None:
    specifications = candidate_runner.CANDIDATE_SPECIFICATIONS
    assert set(specifications) == {B4B_EXPERIMENT_ID, B4C_EXPERIMENT_ID}
    identities = {
        key: candidate_runner.expected_candidate_identity(key)
        for key in specifications
    }
    assert identities[B4B_EXPERIMENT_ID]["trainable_parameter_count"] == 309809
    assert identities[B4B_EXPERIMENT_ID]["fp32_parameter_payload_bytes"] == 1239236
    assert identities[B4C_EXPERIMENT_ID]["trainable_parameter_count"] == 155313
    assert identities[B4C_EXPERIMENT_ID]["fp32_parameter_payload_bytes"] == 621252


def test_atomic_claim_behaviour_is_unchanged() -> None:
    source = inspect.getsource(candidate_runner.claim_candidate_run_directory)
    # The claim itself must stay a non-forgiving mkdir; only the enclosing run
    # root may be created permissively.
    assert "run_dir.mkdir(exist_ok=False)" in source
    assert "run_dir.parent.mkdir(parents=True, exist_ok=True)" in source
    for forbidden in ("--force", "shutil.rmtree", "unlink()", "rename("):
        assert forbidden not in source
