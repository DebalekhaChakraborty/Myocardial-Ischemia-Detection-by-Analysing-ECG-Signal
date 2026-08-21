"""Tests for the T1 episode-state execution harness.

The harness runs the frozen protocol state machine; these tests check that it
runs it *faithfully* (the frozen semantics survive the plumbing), *causally*
(no future window can influence an earlier decision), *reproducibly* (identical
input gives identical output) and *model-agnostically* (nothing here knows what
produced the scores).
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_engine as E
from cardiosentinel.neural import t1_execution as X
from cardiosentinel.neural import t1_stream as S
from cardiosentinel.neural.t1_config import (
    RUN_CLASS_CANONICAL,
    RUN_CLASS_HARNESS,
    THRESHOLD_SOURCE_DERIVED,
    T1ConfigError,
    build_t1_episode_config,
    load_t1_episode_config,
)
from cardiosentinel.neural.t1_protocol import (
    T1_PROTOCOL_SHA256,
    T1_STATE_EVENT,
    T1_STATE_NORMAL,
    T1_STATE_RECOVERY,
    T1_STATE_WATCH,
    T1Thresholds,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "t1_episode.yaml"

STRIDE = 1250
QUIET = (0.02, 0.02, 0.02)
HOT = (0.99, 0.99, 0.99)


# ---------------------------------------------------------------------------
# Fixtures and stream builders
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return load_t1_episode_config(CONFIG_PATH)


@pytest.fixture
def run_config(config, tmp_path):
    """A config whose run root is a scratch directory, never the real one."""
    return dataclasses.replace(config, run_root=tmp_path / "runs")


def window(
    index: int,
    values: tuple[float, float, float] = QUIET,
    *,
    quality: str = "good",
    subject: str = "ltstdb:s2004",
    record: str = "s20041",
    channel: int = 0,
    flags: tuple[str, ...] = (),
    uncertainty: float | None = None,
):
    score, probability, temporal = values
    return S.T1WindowEvidence(
        window_id=f"{record}:{channel}:{index * STRIDE}",
        subject_id=subject,
        record_id=record,
        channel_index=channel,
        start_sample=index * STRIDE,
        model_score=score,
        calibrated_probability=probability,
        temporal_evidence=temporal,
        calibrated_uncertainty=uncertainty,
        signal_quality=quality,
        context_flags=flags,
    )


def stream(*segments: tuple[int, tuple[float, float, float]], **kwargs):
    """Build a chronological stream from (count, values) segments."""
    windows = []
    index = 0
    for count, values in segments:
        for _ in range(count):
            windows.append(window(index, values, **kwargs))
            index += 1
    return windows


def states(outputs) -> list[str]:
    return [entry.state_after for entry in outputs.state_trace]


# ---------------------------------------------------------------------------
# The seven required stream scenarios
# ---------------------------------------------------------------------------


def test_normal_stable_stream_never_leaves_normal(run_config):
    outputs = E.run_t1_episode_state_machine(stream((200, QUIET)), run_config)
    assert set(states(outputs)) == {T1_STATE_NORMAL}
    assert outputs.transitions == ()
    assert outputs.episodes == ()
    assert outputs.alerts == ()
    assert outputs.recovery_spans == ()


def test_a_short_false_spike_never_reaches_event(run_config):
    """Three hot windows against a CONSERVATIVE confirm budget of six."""
    outputs = E.run_t1_episode_state_machine(
        stream((60, QUIET), (3, HOT), (60, QUIET)), run_config
    )
    reached = set(states(outputs))
    assert T1_STATE_WATCH in reached, "a spike should still raise attention"
    assert T1_STATE_EVENT not in reached, "a spike must not become an episode"
    assert outputs.episodes == ()
    assert outputs.alerts == ()
    # It de-escalates on its own without an operator doing anything.
    assert states(outputs)[-1] == T1_STATE_NORMAL


def test_a_sustained_event_is_detected_once(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream((30, QUIET), (40, HOT), (60, QUIET)), run_config
    )
    assert len(outputs.episodes) == 1
    episode = outputs.episodes[0]
    assert episode.closed is True
    assert episode.window_count > 0
    assert episode.duration_seconds == pytest.approx(
        episode.window_count * run_config.stride_seconds
    )
    assert [alert.entered_state for alert in outputs.alerts] == [T1_STATE_EVENT]


def test_event_recovery_walks_the_full_ladder(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream((30, QUIET), (40, HOT), (120, QUIET)), run_config
    )
    ladder = [(entry.state_before, entry.state_after) for entry in outputs.transitions]
    assert (T1_STATE_EVENT, T1_STATE_RECOVERY) in ladder
    assert (T1_STATE_RECOVERY, T1_STATE_NORMAL) in ladder
    assert len(outputs.recovery_spans) == 1
    assert outputs.recovery_spans[0].outcome == E.RECOVERY_CLEARED
    assert states(outputs)[-1] == T1_STATE_NORMAL


def test_recovery_never_becomes_watch_automatically(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream((30, QUIET), (40, HOT), (120, QUIET)), run_config
    )
    for entry in outputs.state_trace:
        assert not (
            entry.state_before == T1_STATE_RECOVERY
            and entry.state_after == T1_STATE_WATCH
        ), "RECOVERY must never automatically de-escalate into WATCH"


def test_repeated_events_are_separate_episodes(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream(
            (30, QUIET),
            (40, HOT),
            (150, QUIET),
            (40, HOT),
            (150, QUIET),
        ),
        run_config,
    )
    assert len(outputs.episodes) == 2
    first, second = outputs.episodes
    assert first.closed and second.closed
    assert first.offset_start_sample < second.onset_start_sample
    assert len(outputs.recovery_spans) == 2


def test_re_escalation_from_recovery_is_recorded_as_such(run_config):
    """A relapse during RECOVERY re-enters EVENT on the shorter budget."""
    outputs = E.run_t1_episode_state_machine(
        stream((30, QUIET), (40, HOT), (8, QUIET), (40, HOT), (150, QUIET)),
        run_config,
    )
    outcomes = [span.outcome for span in outputs.recovery_spans]
    assert E.RECOVERY_RE_ESCALATED in outcomes


def test_missing_windows_hold_state_and_reset_streaks(run_config):
    """Unavailable windows are not evidence: they cannot confirm an escalation."""
    windows = []
    index = 0
    for _ in range(30):
        windows.append(window(index, QUIET))
        index += 1
    # Five hot, one dropout, five hot: never six consecutive available windows.
    for _ in range(5):
        windows.append(window(index, HOT))
        index += 1
    windows.append(window(index, HOT, quality="unavailable"))
    index += 1
    for _ in range(5):
        windows.append(window(index, HOT))
        index += 1
    outputs = E.run_t1_episode_state_machine(windows, run_config)

    assert outputs.unavailable_window_count == 1
    assert T1_STATE_EVENT not in set(states(outputs)), (
        "a dropout mid-confirmation resets the streak, so EVENT must not fire"
    )
    dropout = next(entry for entry in outputs.state_trace if not entry.score_present)
    assert dropout.evidence_level == E.EVIDENCE_UNAVAILABLE
    assert dropout.state_before == dropout.state_after, "state is held"
    assert set(dropout.streaks_after.values()) == {0}, "every streak reset"
    assert dropout.calibrated_probability is None
    assert dropout.temporal_evidence is None
    assert dropout.detector_decision is None


def test_an_unavailable_window_invents_nothing(run_config):
    outputs = E.run_t1_episode_state_machine(
        [window(0, HOT, quality="unavailable")], run_config
    )
    entry = outputs.state_trace[0]
    assert entry.score_present is False
    assert entry.calibrated_probability is None
    assert entry.decision_error_uncertainty is None
    assert entry.required_event_confirm_windows is None


def test_noisy_uncertainty_does_not_flap_the_state(run_config):
    """Evidence oscillating around the thresholds must not produce chatter."""
    noisy = []
    for index in range(120):
        values = HOT if index % 2 == 0 else QUIET
        noisy.append(window(index, values))
    outputs = E.run_t1_episode_state_machine(noisy, run_config)
    assert T1_STATE_EVENT not in set(states(outputs)), (
        "alternating evidence never reaches six consecutive confirmations"
    )
    assert outputs.alerts == ()
    # Hysteresis: WATCH is entered but the stream does not oscillate per window.
    changes = len(outputs.transitions)
    assert changes < len(noisy) // 4, (
        f"state changed {changes} times in {len(noisy)} windows"
    )


def test_a_supplied_uncertainty_that_contradicts_the_protocol_is_refused(run_config):
    bad = window(0, HOT, uncertainty=0.42)
    with pytest.raises(S.T1StreamError, match="derives"):
        E.run_t1_episode_state_machine([bad], run_config)


def test_a_supplied_uncertainty_that_agrees_is_accepted(run_config):
    # d_t is True at HOT, so u_t = 1 - p = 1 - 0.99.
    good = window(0, HOT, uncertainty=1.0 - 0.99)
    outputs = E.run_t1_episode_state_machine([good], run_config)
    assert outputs.state_trace[0].decision_error_uncertainty == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_a_prefix_of_the_stream_produces_a_prefix_of_the_trace(run_config):
    """The operational definition of causality: later windows cannot rewrite history."""
    full = stream((20, QUIET), (30, HOT), (40, QUIET), (30, HOT), (40, QUIET))
    complete = E.run_t1_episode_state_machine(full, run_config)
    complete_rows = complete.as_json_payload()["state_trace"]
    for cut in (1, 5, 25, 55, 90, 130, len(full) - 1):
        partial = E.run_t1_episode_state_machine(full[:cut], run_config)
        assert partial.as_json_payload()["state_trace"] == complete_rows[:cut], (
            f"truncating the stream at {cut} changed an earlier decision"
        )


def test_mutating_a_future_window_cannot_change_an_earlier_decision(run_config):
    original = stream((20, QUIET), (30, HOT), (40, QUIET))
    mutated = list(original)
    mutated[60] = window(60, HOT)
    first = E.run_t1_episode_state_machine(original, run_config)
    second = E.run_t1_episode_state_machine(mutated, run_config)
    assert (
        first.as_json_payload()["state_trace"][:60]
        == second.as_json_payload()["state_trace"][:60]
    )


def test_each_window_is_pulled_exactly_once_and_in_order(run_config):
    windows = stream((20, QUIET), (20, HOT))
    pulled: list[int] = []

    def counted():
        for index, item in enumerate(windows):
            pulled.append(index)
            yield item

    outputs = E.run_t1_episode_state_machine(counted(), run_config)
    assert pulled == list(range(len(windows))), "windows were re-read or reordered"
    assert len(outputs.state_trace) == len(windows)


def test_the_streaming_path_never_materialises_the_input(run_config):
    """No buffer, no sort: a lookahead would have to be built to be used."""
    checks = (
        (S, "iter_adapted_windows", "windows"),
        (E, "run_t1_episode_state_machine", "windows"),
    )
    for module, function_name, parameter in checks:
        tree = ast.parse(Path(module.__file__).read_text())
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name not in {"list", "sorted", "tuple", "reversed", "len"}:
                continue
            for argument in node.args:
                assert not (
                    isinstance(argument, ast.Name) and argument.id == parameter
                ), (
                    f"{module.__name__}.{function_name} calls {name}() on its input "
                    "stream, which would buffer the future"
                )


def test_out_of_order_windows_are_refused_not_sorted(run_config):
    windows = [window(0, QUIET), window(2, QUIET), window(1, QUIET)]
    with pytest.raises(S.T1StreamError, match="does not follow"):
        E.run_t1_episode_state_machine(windows, run_config)


def test_state_never_crosses_a_stream(run_config):
    """Two channels of one record are two independent state namespaces."""
    windows = []
    for index in range(40):
        values = HOT if index >= 20 else QUIET
        windows.append(window(index, values, channel=0))
        windows.append(window(index, QUIET, channel=1))
    outputs = E.run_t1_episode_state_machine(windows, run_config)
    channel_one = [
        entry.state_after for entry in outputs.state_trace if entry.channel_index == 1
    ]
    assert set(channel_one) == {T1_STATE_NORMAL}
    assert outputs.stream_count == 2


# ---------------------------------------------------------------------------
# Reproducibility and explainability
# ---------------------------------------------------------------------------


def test_identical_input_produces_identical_output(run_config):
    windows = stream((25, QUIET), (35, HOT), (60, QUIET))
    first = E.run_t1_episode_state_machine(windows, run_config)
    second = E.run_t1_episode_state_machine(list(windows), run_config)
    assert json.dumps(first.as_json_payload(), sort_keys=True) == json.dumps(
        second.as_json_payload(), sort_keys=True
    )


def test_every_transition_carries_a_reason(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream((30, QUIET), (40, HOT), (150, QUIET)), run_config
    )
    assert outputs.transitions
    for entry in outputs.transitions:
        assert entry.reason and not entry.reason.isspace()
        assert entry.state_before != entry.state_after


def test_the_escalation_reason_names_the_budget_it_met(run_config):
    outputs = E.run_t1_episode_state_machine(
        stream((90, QUIET), (40, HOT), (10, QUIET)), run_config
    )
    escalation = next(
        entry for entry in outputs.transitions if entry.state_after == T1_STATE_EVENT
    )
    assert "mature budget of 6" in escalation.reason


def test_a_cold_start_event_uses_the_cold_budget(run_config):
    """Below 300 s the temporal term is not required, on the longer budget."""
    cold = stream((30, (0.99, 0.99, 0.0)))
    outputs = E.run_t1_episode_state_machine(cold, run_config)
    escalation = [
        entry for entry in outputs.transitions if entry.state_after == T1_STATE_EVENT
    ]
    assert escalation, "cold start should reach EVENT without the temporal term"
    assert "cold-start budget of 12" in escalation[0].reason
    assert outputs.state_trace[0].cold_start is True


def test_the_same_evidence_is_not_an_event_once_the_stream_is_mature(run_config):
    """The identical evidence that fired cold must not fire mature."""
    windows = stream((70, QUIET), (40, (0.99, 0.99, 0.0)))
    outputs = E.run_t1_episode_state_machine(windows, run_config)
    assert T1_STATE_EVENT not in set(states(outputs))
    assert outputs.state_trace[-1].cold_start is False


# ---------------------------------------------------------------------------
# The refractory period is an alerting policy, not a transition input
# ---------------------------------------------------------------------------


def test_the_refractory_changes_alerts_only_and_never_the_state_trace(run_config):
    windows = stream((20, QUIET), (40, HOT), (30, QUIET), (40, HOT), (60, QUIET))
    none = dataclasses.replace(run_config, refractory_seconds=0.0)
    long = dataclasses.replace(run_config, refractory_seconds=3600.0)
    without = E.run_t1_episode_state_machine(windows, none)
    with_refractory = E.run_t1_episode_state_machine(windows, long)

    for field in ("state_trace", "episodes", "transitions", "recovery_spans"):
        assert (
            without.as_json_payload()[field] == with_refractory.as_json_payload()[field]
        ), f"{field} changed when only the refractory period changed"

    assert [alert.suppressed for alert in without.alerts] == [False, False]
    assert [alert.suppressed for alert in with_refractory.alerts] == [False, True]


def test_suppressed_alerts_are_still_recorded(run_config):
    windows = stream((20, QUIET), (40, HOT), (30, QUIET), (40, HOT), (60, QUIET))
    long = dataclasses.replace(run_config, refractory_seconds=3600.0)
    outputs = E.run_t1_episode_state_machine(windows, long)
    assert len(outputs.alerts) == 2
    assert outputs.summary()["alert_suppressed_count"] == 1
    assert outputs.summary()["alert_emitted_count"] == 1


# ---------------------------------------------------------------------------
# The firewall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["label", "target_family", "m2_update_admitted", "u_star_deploy", "gru_score"],
)
def test_a_transition_payload_carrying_a_forbidden_field_is_refused(forbidden):
    with pytest.raises(S.T1StreamError, match="not deployable"):
        S.require_no_forbidden_fields({"stable_id": "w0", forbidden: 1})


def test_context_flags_reach_reporting_but_not_the_transition(run_config):
    flagged = stream((10, HOT), flags=("rate_related", "axis_shift"))
    plain = stream((10, HOT))
    with_flags = E.run_t1_episode_state_machine(flagged, run_config)
    without_flags = E.run_t1_episode_state_machine(plain, run_config)

    assert with_flags.state_trace[0].context_flags == ("rate_related", "axis_shift")
    assert without_flags.state_trace[0].context_flags == ()
    assert [entry.state_after for entry in with_flags.state_trace] == [
        entry.state_after for entry in without_flags.state_trace
    ], "a reporting flag changed a state decision"


def test_the_harness_modules_import_no_model_machinery():
    """Model-agnosticism, proved structurally rather than asserted."""
    for module in (S, E, X):
        tree = ast.parse(Path(module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden in (
            "torch",
            "sklearn",
            "cardiosentinel.neural.t2_models",
            "cardiosentinel.neural.model",
            "cardiosentinel.neural.training",
            "cardiosentinel.neural.sealed_test",
        ):
            assert not any(name.startswith(forbidden) for name in imported), (
                f"{module.__name__} imports {forbidden}"
            )


def test_the_harness_never_touches_the_sealed_test_partition():
    for module in (S, E, X):
        source = Path(module.__file__).read_text()
        for marker in ("TEST_ATTEMPT", "evaluate-locked-test", "sealed_test_cache"):
            assert marker not in source, f"{module.__name__} mentions {marker}"


# ---------------------------------------------------------------------------
# Configuration refusals
# ---------------------------------------------------------------------------


def _raw_config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text())["t1_episode"]


def test_the_shipped_config_loads_and_is_not_protocol_evidence(config):
    assert config.run_class == RUN_CLASS_HARNESS
    assert config.protocol_evidence is False
    assert config.protocol_document_sha256 == T1_PROTOCOL_SHA256
    assert config.profile.name == "CONSERVATIVE"


def test_a_canonical_run_is_refused_when_permission_is_withdrawn(monkeypatch):
    """The config loader consults the same single constant the gate does.

    Authorization makes a canonical run_class loadable, which is the point of
    granting it. What must survive is that withdrawing permission makes it
    unloadable again, from the one constant rather than a second switch.
    """
    from cardiosentinel.neural import t1_config as C

    monkeypatch.setattr(C, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    raw = _raw_config()
    raw["run"]["run_class"] = RUN_CLASS_CANONICAL
    raw["thresholds"]["source"] = THRESHOLD_SOURCE_DERIVED
    for key in ("p_watch", "s_watch", "p_event", "s_event"):
        raw["thresholds"]["literal"][key] = None
    with pytest.raises(T1ConfigError, match="execution specification"):
        build_t1_episode_config(raw)


def test_literal_thresholds_cannot_sit_beside_a_derived_source():
    raw = _raw_config()
    raw["thresholds"]["source"] = THRESHOLD_SOURCE_DERIVED
    with pytest.raises(T1ConfigError, match="silent override"):
        build_t1_episode_config(raw)


def test_a_fourth_persistence_profile_cannot_be_configured():
    raw = _raw_config()
    raw["persistence"]["profile"] = "AGGRESSIVE"
    with pytest.raises(T1ConfigError, match="frozen profiles"):
        build_t1_episode_config(raw)


def test_declared_window_counts_are_checked_against_the_frozen_profile():
    raw = _raw_config()
    raw["persistence"]["expected_windows"]["event_confirm"] = 2
    with pytest.raises(T1ConfigError, match="frozen CONSERVATIVE profile"):
        build_t1_episode_config(raw)


def test_interpolated_thresholds_are_refused():
    raw = _raw_config()
    raw["thresholds"]["interpolation"] = True
    with pytest.raises(T1ConfigError, match="order statistic"):
        build_t1_episode_config(raw)


@pytest.mark.parametrize("key", ["imputation", "forward_fill", "synthetic_zero"])
def test_inventing_values_for_missing_windows_is_refused(key):
    raw = _raw_config()
    raw["availability"][key] = True
    with pytest.raises(T1ConfigError, match="invented"):
        build_t1_episode_config(raw)


def test_the_refractory_cannot_be_widened_to_touch_transitions():
    raw = _raw_config()
    raw["alerting"]["refractory_applies_to"] = "state_transitions"
    with pytest.raises(T1ConfigError, match="alerting policy"):
        build_t1_episode_config(raw)


def test_context_flags_cannot_be_promoted_to_transition_inputs():
    raw = _raw_config()
    raw["reporting"]["context_flags_influence_transitions"] = True
    with pytest.raises(T1ConfigError, match="not be deployable"):
        build_t1_episode_config(raw)


def test_a_mutated_protocol_digest_is_refused():
    raw = _raw_config()
    raw["protocol"]["document_sha256"] = "0" * 64
    with pytest.raises(T1ConfigError, match="frozen"):
        build_t1_episode_config(raw)


def test_unknown_config_keys_are_refused_rather_than_ignored():
    raw = _raw_config()
    raw["alerting"]["escalate_to_pager"] = True
    with pytest.raises(T1ConfigError, match="unknown keys"):
        build_t1_episode_config(raw)


def test_an_event_threshold_below_the_watch_threshold_is_refused():
    raw = _raw_config()
    raw["thresholds"]["literal"]["p_event"] = 0.1
    with pytest.raises(T1ConfigError, match="escalation ladder"):
        build_t1_episode_config(raw)


def test_a_derived_config_refuses_to_run_without_a_fit_population(config):
    derived = dataclasses.replace(
        config, threshold_source=THRESHOLD_SOURCE_DERIVED, literal_thresholds=None
    )
    with pytest.raises(E.T1EngineError, match="hand-chosen threshold"):
        E.run_t1_episode_state_machine(stream((3, QUIET)), derived)


# ---------------------------------------------------------------------------
# The run scaffold
# ---------------------------------------------------------------------------


def test_a_run_captures_git_config_runtime_and_input_identity(run_config):
    windows = stream((20, QUIET), (30, HOT), (40, QUIET))
    result = X.execute_t1_run(windows, run_config)
    run_dir = run_config.run_root / run_config.attempt_id
    manifest = X.read_run_artifact(run_dir, X.RUN_MANIFEST_NAME)

    assert manifest["git"]["git_sha"]
    assert manifest["config"]["config_sha256"] == run_config.config_sha256
    assert manifest["runtime"]["installed_packages_sha256"]
    assert manifest["runtime"]["python_version"]
    assert manifest["input"]["input_artifact_sha256"]
    assert manifest["input"]["input_window_count"] == len(windows)
    assert manifest["protocol"]["document_sha256"] == T1_PROTOCOL_SHA256
    assert result["claims"]["performance_claimed"] is False


def test_a_harness_run_is_never_labelled_protocol_evidence(run_config):
    result = X.execute_t1_run(stream((10, QUIET)), run_config)
    assert result["protocol_evidence"] is False
    assert result["claims"]["evidence_class"].startswith("harness_verification_only")
    status = X.read_run_artifact(
        run_config.run_root / run_config.attempt_id, X.RUN_STATUS_NAME
    )
    assert status["status"] == X.STATUS_COMPLETE
    assert status["protocol_evidence"] is False
    assert status["sealed_test_state"] == "unopened"


def test_all_five_outputs_are_promoted(run_config):
    X.execute_t1_run(stream((20, QUIET), (30, HOT), (60, QUIET)), run_config)
    run_dir = run_config.run_root / run_config.attempt_id
    for name in X.OUTPUT_NAMES:
        assert (run_dir / name).is_file(), name
    episodes = X.read_run_artifact(run_dir, X.EPISODES_NAME)["episodes"]
    assert len(episodes) == 1


def test_a_second_run_cannot_overwrite_the_first(run_config):
    X.execute_t1_run(stream((10, QUIET)), run_config)
    with pytest.raises(X.T1ExecutionError, match="already exists"):
        X.execute_t1_run(stream((10, QUIET)), run_config)


def test_the_canonical_attempt_namespace_is_reserved(run_config):
    reserved = dataclasses.replace(run_config, attempt_id="t1-v1-development")
    with pytest.raises(X.T1ExecutionError, match="reserved"):
        X.execute_t1_run(stream((10, QUIET)), reserved)


def test_a_different_input_order_is_a_different_input_digest():
    windows = stream((4, QUIET))
    forward = X.window_stream_digest(windows)
    backward = X.window_stream_digest(list(reversed(windows)))
    assert forward["input_artifact_sha256"] != backward["input_artifact_sha256"]


def test_the_run_result_digests_every_output(run_config):
    result = X.execute_t1_run(stream((20, QUIET), (30, HOT), (60, QUIET)), run_config)
    assert set(result["output_sha256"]) == set(X.OUTPUT_NAMES)
    assert all(len(digest) == 64 for digest in result["output_sha256"].values())


def test_two_runs_of_one_stream_agree_on_every_output_digest(run_config, tmp_path):
    windows = stream((20, QUIET), (30, HOT), (60, QUIET))
    first = X.execute_t1_run(windows, run_config)
    second_config = dataclasses.replace(run_config, run_root=tmp_path / "second")
    second = X.execute_t1_run(windows, second_config)
    assert first["output_sha256"] == second["output_sha256"]


# ---------------------------------------------------------------------------
# Consuming a future model's outputs without a code change
# ---------------------------------------------------------------------------


def test_an_arbitrary_producer_needs_no_harness_change(run_config):
    """Anything that can emit (score, probability, temporal) is a valid producer."""

    def pretend_model(index: int) -> tuple[float, float, float]:
        hot = 30 <= index < 70
        return (0.97, 0.96, 0.95) if hot else (0.03, 0.04, 0.05)

    produced = [
        S.T1WindowEvidence(
            window_id=f"produced:{index}",
            subject_id="ltstdb:s2019",
            record_id="s20191",
            channel_index=1,
            start_sample=index * STRIDE,
            model_score=pretend_model(index)[0],
            calibrated_probability=pretend_model(index)[1],
            temporal_evidence=pretend_model(index)[2],
        )
        for index in range(140)
    ]
    outputs = E.run_t1_episode_state_machine(produced, run_config)
    assert len(outputs.episodes) == 1
    assert outputs.subject_count == 1
    assert outputs.stream_count == 1


def test_thresholds_may_be_supplied_explicitly_for_integration(run_config):
    strict = T1Thresholds(p_watch=0.99, s_watch=0.99, p_event=0.999, s_event=0.999)
    outputs = E.run_t1_episode_state_machine(
        stream((40, HOT)), run_config, thresholds=strict
    )
    assert T1_STATE_EVENT not in set(states(outputs))


# ---------------------------------------------------------------------------
# Canonical namespace protection
#
# The canonical run directory is itself claim-bearing: it existing in any state
# consumes the one canonical T1 attempt. Guarding the attempt id alone is not
# enough, because the directory is reachable by three different routes.
# ---------------------------------------------------------------------------


def _canonical_root() -> Path:
    from cardiosentinel.neural.t1_execution_spec import T1_RUN_ROOT_RELATIVE

    return REPOSITORY_ROOT / T1_RUN_ROOT_RELATIVE


def test_the_canonical_run_root_named_directly_is_refused(run_config):
    """Route 1: run_root IS the canonical root, under an innocuous attempt id."""
    attacked = dataclasses.replace(
        run_config, run_root=_canonical_root(), attempt_id="smoke"
    )
    with pytest.raises(X.T1ExecutionError, match="claim-bearing"):
        X.require_non_canonical_attempt(attacked)


def test_a_directory_inside_the_canonical_run_root_is_refused(run_config):
    """Route 2: burying the run deeper does not put it outside the namespace."""
    attacked = dataclasses.replace(
        run_config,
        run_root=_canonical_root() / "nested" / "deeper",
        attempt_id="smoke",
    )
    with pytest.raises(X.T1ExecutionError, match="claim-bearing"):
        X.require_non_canonical_attempt(attacked)


def test_the_canonical_root_reached_as_an_attempt_id_is_refused(run_config):
    """Route 3: name the canonical root's PARENT and let attempt_id complete it."""
    canonical = _canonical_root()
    attacked = dataclasses.replace(
        run_config, run_root=canonical.parent, attempt_id=canonical.name
    )
    with pytest.raises(X.T1ExecutionError):
        X.require_non_canonical_attempt(attacked)


def test_a_relative_canonical_run_root_is_refused(run_config):
    """The guard must not depend on the process working directory."""
    from cardiosentinel.neural.t1_execution_spec import T1_RUN_ROOT_RELATIVE

    attacked = dataclasses.replace(
        run_config, run_root=Path(T1_RUN_ROOT_RELATIVE), attempt_id="smoke"
    )
    with pytest.raises(X.T1ExecutionError, match="claim-bearing"):
        X.require_non_canonical_attempt(attacked)


def test_a_traversal_path_into_the_canonical_root_is_refused(run_config):
    """`..` must be normalised before the comparison, not after the mkdir."""
    from cardiosentinel.neural.t1_execution_spec import T1_RUN_ROOT_RELATIVE

    traversal = (
        Path("cardiosentinel-runs")
        / "elsewhere"
        / ".."
        / Path(T1_RUN_ROOT_RELATIVE).name
    )
    attacked = dataclasses.replace(run_config, run_root=traversal, attempt_id="smoke")
    with pytest.raises(X.T1ExecutionError, match="claim-bearing"):
        X.require_non_canonical_attempt(attacked)


def test_an_unrelated_run_root_is_allowed(run_config, tmp_path):
    """The guard refuses the canonical namespace, not ordinary work."""
    fine = dataclasses.replace(
        run_config, run_root=tmp_path / "somewhere", attempt_id="smoke"
    )
    assert X.require_non_canonical_attempt(fine) is None


def test_the_end_to_end_run_refuses_and_creates_nothing(run_config):
    """The refusal fires before any directory is made, not after."""
    canonical = _canonical_root()
    existed_before = canonical.exists()
    attacked = dataclasses.replace(run_config, run_root=canonical, attempt_id="smoke")
    with pytest.raises(X.T1ExecutionError, match="claim-bearing"):
        X.execute_t1_run(stream((5, QUIET)), attacked)
    assert canonical.exists() == existed_before, (
        "the guard let the canonical run root be created before refusing"
    )


@pytest.mark.parametrize("attempt", ["t1-v1-development", "T1-V1-Development-retry"])
def test_a_canonical_attempt_id_is_refused_case_insensitively(run_config, attempt):
    attacked = dataclasses.replace(run_config, attempt_id=attempt)
    with pytest.raises(X.T1ExecutionError, match="reserved"):
        X.require_non_canonical_attempt(attacked)


# ---------------------------------------------------------------------------
# Anti-drift: the specification owns the identity, the harness binds it
# ---------------------------------------------------------------------------


def test_the_harness_binds_the_specification_identity_rather_than_copying_it():
    """A local copy of a frozen identity is a copy that can drift silently."""
    from cardiosentinel.neural import t1_execution_spec as SPEC

    assert X.CANONICAL_RUN_ROOT == REPOSITORY_ROOT / SPEC.T1_RUN_ROOT_RELATIVE
    assert SPEC.T1_DEVELOPMENT_ATTEMPT_ID in X.CANONICAL_RESERVED_PREFIXES
    assert Path(SPEC.T1_RUN_ROOT_RELATIVE).name in X.CANONICAL_RESERVED_PREFIXES


def test_no_frozen_identity_is_hardcoded_as_a_literal_in_the_harness():
    """The identity strings must come from the specification module by import."""
    from cardiosentinel.neural import t1_execution_spec as SPEC

    source = Path(X.__file__).read_text()
    for literal in (
        f'"{SPEC.T1_DEVELOPMENT_ATTEMPT_ID}"',
        f'"{SPEC.T1_RUN_ROOT_RELATIVE}"',
        f'"{SPEC.T1_EXPERIMENT_IDENTITY}"',
    ):
        assert literal not in source, (
            f"{literal} is hardcoded in t1_execution.py; import it from the "
            "execution specification instead"
        )


def test_the_run_manifest_binds_the_execution_specification(run_config):
    from cardiosentinel.neural import t1_execution_spec as SPEC

    X.execute_t1_run(stream((10, QUIET)), run_config)
    manifest = X.read_run_artifact(
        run_config.run_root / run_config.attempt_id, X.RUN_MANIFEST_NAME
    )
    binding = manifest["execution_specification"]
    assert binding["document_sha256"] == SPEC.T1_EXECUTION_SPEC_SHA256
    assert binding["canonical_attempt_id"] == SPEC.T1_DEVELOPMENT_ATTEMPT_ID
    assert binding["canonical_run_root"] == str(SPEC.T1_RUN_ROOT_RELATIVE)
    assert binding["canonical_namespace_claimed_by_this_run"] is False


# ---------------------------------------------------------------------------
# The authorization gate says what is actually true
# ---------------------------------------------------------------------------


def test_the_three_authorization_facts_stay_separate():
    """Specification, capability and permission are three different things.

    All three read True now, which is precisely when a permission derived from
    the other two would be indistinguishable from a deliberate one. Separation
    is therefore proven structurally: three distinct assignments, and the
    permission constant is a plain literal rather than an expression computed
    from its neighbours.
    """
    import ast
    from pathlib import Path

    from cardiosentinel.neural import t1_config as C

    assert C.T1_EXECUTION_SPECIFICATION_EXISTS is True
    assert C.T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS is True
    assert C.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True
    assert C.T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE == (
        "cardiosentinel.neural.t1_development_run"
    )

    assigned = {
        node.target.id: node.value
        for node in ast.parse(Path(C.__file__).read_text(encoding="utf-8")).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    for name in (
        "T1_EXECUTION_SPECIFICATION_EXISTS",
        "T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS",
        "T1_EXECUTION_SPECIFICATION_AUTHORIZED",
    ):
        assert name in assigned, f"{name} is no longer its own constant"
    permission = assigned["T1_EXECUTION_SPECIFICATION_AUTHORIZED"]
    assert isinstance(permission, ast.Constant), (
        "permission is computed rather than stated; it must be deliberate"
    )
    assert permission.value is True


def test_the_canonical_refusal_no_longer_claims_the_specification_is_missing(
    monkeypatch,
):
    """The old message said 'none exists'. One exists; it is merged.

    Reached by withdrawing permission, since the refusal this inspects only
    fires while the gate is closed.
    """
    from cardiosentinel.neural import t1_config as C

    monkeypatch.setattr(C, "T1_EXECUTION_SPECIFICATION_AUTHORIZED", False)
    raw = _raw_config()
    raw["run"]["run_class"] = RUN_CLASS_CANONICAL
    raw["thresholds"]["source"] = THRESHOLD_SOURCE_DERIVED
    for key in ("p_watch", "s_watch", "p_event", "s_event"):
        raw["thresholds"]["literal"][key] = None
    with pytest.raises(T1ConfigError) as caught:
        build_t1_episode_config(raw)
    message = str(caught.value)
    assert "exists and is merged" in message
    assert "is implemented but has NOT been authorized" in message
    assert "neither is a permission" in message
    assert "none exists" not in message


def test_the_canonical_development_harness_exists_and_is_now_authorized():
    """The constants must track the repository, not merely sound right.

    This test asserted the harness was absent, then that it existed but was
    unauthorized. Both tripwires fired as designed and both were updated by
    the change that made them false. What must never regress is the direction
    of the implication: a module existing is still a capability, and
    permission is still granted separately rather than derived from it.
    """
    import importlib.util

    from cardiosentinel.neural import t1_config as C

    found = importlib.util.find_spec(C.T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE)
    assert found is not None, "the harness module went missing"
    assert C.T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS is True
    assert C.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True
