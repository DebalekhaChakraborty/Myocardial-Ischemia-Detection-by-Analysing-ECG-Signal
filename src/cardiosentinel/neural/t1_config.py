"""Configuration loading for the T1 episode-state execution harness.

Every value the harness needs comes from a config file; nothing operational is
hardcoded here. What *is* hardcoded is the set of refusals that keep a config
from quietly contradicting the frozen protocol.

**Configuration is selection, not tuning.** The frozen protocol generates its
thresholds prospectively from FIT-subject background negatives and fixes three
persistence profiles, a cold-start rule and an evidence formula. A config that
could simply name a numeric watch threshold would dissolve that guarantee. So a
config selects *which frozen option* applies to a run -- which quantile pair,
which persistence profile -- and the harness recomputes everything else by the
frozen rule.

Two run classes exist and they are not interchangeable:

``canonical_t1_development``
    Protocol evidence. Every frozen value must match exactly, thresholds must
    be derived, and a separately authorized T1 execution specification is
    required. The specification is frozen, the harness is implemented and
    canonical execution has been authorized, so this run class is available.

``harness_verification``
    Synthetic streams, reviewer acceptance checks, and integration of a future
    model's outputs. Free to deviate from the frozen values, and permanently
    stamped ``protocol_evidence: false``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from cardiosentinel.neural.integrity import canonical_sha256
from cardiosentinel.neural.t1_execution_spec import (
    T1_EXECUTION_SPEC_NAME,
    T1_EXECUTION_SPEC_SHA256,
)
from cardiosentinel.neural.t1_protocol import (
    Q_EVENT,
    Q_WATCH,
    T1_COLD_START_REQUIRES_S4D,
    T1_COLD_START_SECONDS,
    T1_DETECTOR_THRESHOLD,
    T1_FORWARD_FILL_PERMITTED,
    T1_IMPUTATION_PERMITTED,
    T1_PERSISTENCE_PROFILES,
    T1_PROTOCOL_NAME,
    T1_PROTOCOL_SHA256,
    T1_SAMPLING_FREQUENCY_HZ,
    T1_STATES,
    T1_STRIDE_SAMPLES,
    T1_SYNTHETIC_ZERO_PERMITTED,
    T1_THRESHOLD_INTERPOLATION_PERMITTED,
    T1_WINDOW_SAMPLES,
    T1PersistenceProfile,
    T1Thresholds,
    validate_t1_protocol_document,
)

CONFIG_ROOT_KEY: Final = "t1_episode"

RUN_CLASS_CANONICAL: Final = "canonical_t1_development"
RUN_CLASS_HARNESS: Final = "harness_verification"
RUN_CLASSES: Final = (RUN_CLASS_CANONICAL, RUN_CLASS_HARNESS)

THRESHOLD_SOURCE_DERIVED: Final = "derived_from_fit_background_negatives"
THRESHOLD_SOURCE_LITERAL: Final = "explicit_literal"
THRESHOLD_SOURCES: Final = (THRESHOLD_SOURCE_DERIVED, THRESHOLD_SOURCE_LITERAL)

REFRACTORY_SCOPE: Final = "alert_emission_only"

# Three different facts, deliberately not conflated. Collapsing any two of them
# is how a gate opens by accident:
#
#   1. the specification document exists                     -> True
#   2. the canonical development harness is implemented      -> True
#   3. canonical scientific execution is authorized          -> True
#
# A specification is a contract. A harness is a capability. Neither is a
# permission. Only (3) gates execution, and it stays a deliberate constant
# rather than a check derived from (1) or (2), because the existence of a
# document and the existence of a module are facts about the repository while
# permission is a fact about a human decision.
#
# (3) was flipped by the reviewed enabling change that also replaced the
# unconditional refusal in `t1_development_run.main` with the frozen pre-claim
# verification. Granting permission weakened no check: the entry point still
# proves the authorized commit against a clean HEAD, the runtime identity
# against the frozen dependency digest, all three upstream retention
# decisions, that TEST is unopened and that the canonical attempt does not
# exist -- and it proves every one of them before anything is claimed.
T1_EXECUTION_SPECIFICATION_EXISTS: Final = True
T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE: Final = (
    "cardiosentinel.neural.t1_development_run"
)
T1_CANONICAL_DEVELOPMENT_HARNESS_EXISTS: Final = True
T1_EXECUTION_SPECIFICATION_AUTHORIZED: Final = True

_SECTIONS: Final = (
    "protocol",
    "run",
    "stream",
    "detector",
    "thresholds",
    "persistence",
    "cold_start",
    "availability",
    "alerting",
    "reporting",
)

_KEYS: Final = {
    "protocol": ("name", "document_sha256", "verify_document"),
    "run": ("run_class", "run_root", "attempt_id"),
    "stream": (
        "sampling_frequency_hz",
        "window_samples",
        "stride_samples",
        "stream_key",
        "order_by",
        "require_strictly_increasing",
        "max_gap_seconds_before_stream_reset",
    ),
    "detector": ("threshold", "threshold_source"),
    "thresholds": (
        "source",
        "q_watch",
        "q_event",
        "interpolation",
        "tie_order",
        "literal",
    ),
    "persistence": ("profile", "expected_windows"),
    "cold_start": ("seconds", "requires_temporal_evidence"),
    "availability": (
        "signal_quality_accept",
        "hold_state",
        "reset_streaks",
        "advance_state_time",
        "imputation",
        "forward_fill",
        "synthetic_zero",
    ),
    "alerting": (
        "emit_on_entry_to",
        "refractory_seconds",
        "refractory_applies_to",
        "record_suppressed_alerts",
    ),
    "reporting": ("record_context_flags", "context_flags_influence_transitions"),
}

_LITERAL_KEYS: Final = ("p_watch", "s_watch", "p_event", "s_event")
_PROFILE_WINDOW_KEYS: Final = (
    "watch_clear",
    "event_confirm",
    "event_release",
    "re_event_confirm",
    "recovery_clear",
    "cold_event_confirm",
)


class T1ConfigError(RuntimeError):
    """Raised when a T1 execution config is malformed or contradicts the protocol."""


def require_canonical_execution_authorized() -> None:
    """The permission gate, and the only one.

    Lives beside the constant it reads so the permission and its enforcement
    cannot drift apart, and is the single place any caller asks the question --
    the config loader and the canonical entry point both come here rather than
    testing the constant themselves.

    Permission is not capability: this says nothing about whether the harness
    exists or whether the specification is frozen, and passing it verifies
    nothing about the commit, the runtime, the upstream chain or the attempt.
    Those are proven separately, and afterwards.
    """
    if not T1_EXECUTION_SPECIFICATION_AUTHORIZED:
        raise T1ConfigError(
            "Canonical T1 execution is not authorized. The frozen execution "
            f"specification {T1_EXECUTION_SPEC_NAME} exists and is merged "
            f"(digest {T1_EXECUTION_SPEC_SHA256}), and the canonical "
            f"development harness {T1_CANONICAL_DEVELOPMENT_HARNESS_MODULE} is "
            "implemented, but neither is a permission: a specification is a "
            "contract and a harness is a capability. Canonical execution is "
            "authorized separately, by a human naming the merged harness commit."
        )


@dataclass(frozen=True, slots=True)
class T1EpisodeConfig:
    """A validated T1 execution configuration.

    ``protocol_evidence`` is the single fact every downstream artifact carries.
    It is true only for a canonical development run, and no code path sets it
    from user input.
    """

    run_class: str
    protocol_evidence: bool
    run_root: Path
    attempt_id: str
    sampling_frequency_hz: int
    window_samples: int
    stride_samples: int
    stream_key: tuple[str, ...]
    order_by: str
    require_strictly_increasing: bool
    max_gap_seconds_before_stream_reset: float | None
    detector_threshold: float
    threshold_source: str
    q_watch: float
    q_event: float
    literal_thresholds: T1Thresholds | None
    profile: T1PersistenceProfile
    cold_start_seconds: float
    cold_start_requires_temporal_evidence: bool
    signal_quality_accept: tuple[str, ...]
    alert_on_entry_to: tuple[str, ...]
    refractory_seconds: float
    record_suppressed_alerts: bool
    record_context_flags: bool
    config_sha256: str
    protocol_document_sha256: str
    source_path: Path | None

    @property
    def stride_seconds(self) -> float:
        return self.stride_samples / self.sampling_frequency_hz

    @property
    def is_canonical(self) -> bool:
        return self.run_class == RUN_CLASS_CANONICAL


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise T1ConfigError(f"Config section {name!r} is missing or is not a mapping.")
    unknown = sorted(set(section) - set(_KEYS[name]))
    if unknown:
        raise T1ConfigError(
            f"Config section {name!r} has unknown keys {unknown}. Unknown keys are "
            "refused rather than ignored: a silently dropped setting is a config "
            "that does not describe the run it produced."
        )
    missing = sorted(set(_KEYS[name]) - set(section))
    if missing:
        raise T1ConfigError(f"Config section {name!r} is missing keys {missing}.")
    return section


def _require_bool(section: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise T1ConfigError(f"{section}.{key} must be a boolean, got {value!r}.")
    return value


def _require_positive_int(section: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise T1ConfigError(
            f"{section}.{key} must be a positive integer, got {value!r}."
        )
    return value


def _require_probability(section: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T1ConfigError(f"{section}.{key} must be a number, got {value!r}.")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise T1ConfigError(f"{section}.{key} must lie in [0, 1], got {number!r}.")
    return number


def _require_non_negative_seconds(section: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T1ConfigError(f"{section}.{key} must be a number, got {value!r}.")
    number = float(value)
    if number < 0.0:
        raise T1ConfigError(f"{section}.{key} must not be negative, got {number!r}.")
    return number


def _require_str_tuple(section: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise T1ConfigError(
            f"{section}.{key} must be a list of strings, got {value!r}."
        )
    return tuple(value)


def _resolve_profile(name: Any) -> T1PersistenceProfile:
    for profile in T1_PERSISTENCE_PROFILES:
        if profile.name == name:
            return profile
    known = [profile.name for profile in T1_PERSISTENCE_PROFILES]
    raise T1ConfigError(
        f"persistence.profile {name!r} is not one of the frozen profiles {known}. "
        "The three profiles are part of the protocol; a fourth cannot be "
        "introduced by configuration."
    )


def _check_expected_windows(profile: T1PersistenceProfile, declared: Any) -> None:
    """The declared window counts are an assertion, never a source of values."""
    if not isinstance(declared, dict):
        raise T1ConfigError("persistence.expected_windows must be a mapping.")
    unknown = sorted(set(declared) - set(_PROFILE_WINDOW_KEYS))
    if unknown:
        raise T1ConfigError(f"persistence.expected_windows has unknown keys {unknown}.")
    missing = sorted(set(_PROFILE_WINDOW_KEYS) - set(declared))
    if missing:
        raise T1ConfigError(f"persistence.expected_windows is missing keys {missing}.")
    for key in _PROFILE_WINDOW_KEYS:
        frozen = getattr(profile, f"{key}_windows")
        if declared[key] != frozen:
            raise T1ConfigError(
                f"persistence.expected_windows.{key} says {declared[key]!r} but the "
                f"frozen {profile.name} profile is {frozen!r}. The frozen profile "
                "wins; the config is corrected, never the profile."
            )


def _resolve_literals(section: dict[str, Any], source: str) -> T1Thresholds | None:
    literal = section["literal"]
    if not isinstance(literal, dict):
        raise T1ConfigError("thresholds.literal must be a mapping.")
    unknown = sorted(set(literal) - set(_LITERAL_KEYS))
    if unknown:
        raise T1ConfigError(f"thresholds.literal has unknown keys {unknown}.")
    missing = sorted(set(_LITERAL_KEYS) - set(literal))
    if missing:
        raise T1ConfigError(f"thresholds.literal is missing keys {missing}.")
    if source == THRESHOLD_SOURCE_DERIVED:
        populated = sorted(key for key in _LITERAL_KEYS if literal[key] is not None)
        if populated:
            raise T1ConfigError(
                f"thresholds.source is {THRESHOLD_SOURCE_DERIVED!r} but "
                f"thresholds.literal populates {populated}. A derived run computes "
                "its thresholds from FIT-subject background negatives by the frozen "
                "order-statistic rule; a literal sitting beside it is either dead "
                "text or a silent override, and neither is acceptable."
            )
        return None
    values = {
        key: _require_probability("thresholds.literal", key, literal[key])
        for key in _LITERAL_KEYS
    }
    if values["p_event"] < values["p_watch"] or values["s_event"] < values["s_watch"]:
        raise T1ConfigError(
            "Literal EVENT thresholds must not sit below the WATCH thresholds; "
            f"got {values}. An EVENT easier to reach than a WATCH inverts the "
            "escalation ladder."
        )
    return T1Thresholds(
        p_watch=values["p_watch"],
        s_watch=values["s_watch"],
        p_event=values["p_event"],
        s_event=values["s_event"],
    )


def _require_canonical_agreement(
    *,
    stream: dict[str, Any],
    detector: dict[str, Any],
    thresholds: dict[str, Any],
    cold_start: dict[str, Any],
    q_watch: float,
    q_event: float,
) -> None:
    """A canonical run must agree with the frozen protocol on every frozen value."""
    try:
        require_canonical_execution_authorized()
    except T1ConfigError as unauthorized:
        raise T1ConfigError(
            f"run.run_class {RUN_CLASS_CANONICAL!r} is not available. "
            f"{unauthorized} Use {RUN_CLASS_HARNESS!r} for synthetic and "
            "integration runs."
        ) from unauthorized
    frozen = {
        "stream.sampling_frequency_hz": (
            stream["sampling_frequency_hz"],
            T1_SAMPLING_FREQUENCY_HZ,
        ),
        "stream.window_samples": (stream["window_samples"], T1_WINDOW_SAMPLES),
        "stream.stride_samples": (stream["stride_samples"], T1_STRIDE_SAMPLES),
        "detector.threshold": (detector["threshold"], T1_DETECTOR_THRESHOLD),
        "cold_start.seconds": (cold_start["seconds"], T1_COLD_START_SECONDS),
        "cold_start.requires_temporal_evidence": (
            cold_start["requires_temporal_evidence"],
            T1_COLD_START_REQUIRES_S4D,
        ),
    }
    for name, (declared, expected) in frozen.items():
        if declared != expected:
            raise T1ConfigError(
                f"A canonical T1 run must use the frozen {name} {expected!r}, "
                f"not {declared!r}."
            )
    if thresholds["source"] != THRESHOLD_SOURCE_DERIVED:
        raise T1ConfigError(
            f"A canonical T1 run must derive its thresholds "
            f"({THRESHOLD_SOURCE_DERIVED!r}). Literal thresholds would replace a "
            "prospectively generated value with a hand-chosen one, which is the "
            "single thing the T1 protocol exists to prevent."
        )
    if q_watch not in Q_WATCH:
        raise T1ConfigError(f"q_watch {q_watch!r} is not one of the frozen {Q_WATCH}.")
    if q_event not in Q_EVENT:
        raise T1ConfigError(f"q_event {q_event!r} is not one of the frozen {Q_EVENT}.")


def load_t1_episode_config(path: str | Path) -> T1EpisodeConfig:
    """Load, validate and digest a T1 execution config."""
    source = Path(path)
    if not source.is_file():
        raise T1ConfigError(f"T1 episode config is missing at {source}.")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or CONFIG_ROOT_KEY not in document:
        raise T1ConfigError(
            f"Config must have a top-level {CONFIG_ROOT_KEY!r} mapping."
        )
    return build_t1_episode_config(document[CONFIG_ROOT_KEY], source_path=source)


def build_t1_episode_config(
    raw: dict[str, Any], *, source_path: Path | None = None
) -> T1EpisodeConfig:
    """Validate an already-parsed config body. Pure apart from the digest check."""
    if not isinstance(raw, dict):
        raise T1ConfigError("T1 episode config body must be a mapping.")
    unknown_sections = sorted(set(raw) - set(_SECTIONS))
    if unknown_sections:
        raise T1ConfigError(f"Config has unknown sections {unknown_sections}.")

    protocol = _require_section(raw, "protocol")
    run = _require_section(raw, "run")
    stream = _require_section(raw, "stream")
    detector = _require_section(raw, "detector")
    thresholds = _require_section(raw, "thresholds")
    persistence = _require_section(raw, "persistence")
    cold_start = _require_section(raw, "cold_start")
    availability = _require_section(raw, "availability")
    alerting = _require_section(raw, "alerting")
    reporting = _require_section(raw, "reporting")

    if protocol["name"] != T1_PROTOCOL_NAME:
        raise T1ConfigError(
            f"protocol.name {protocol['name']!r} is not {T1_PROTOCOL_NAME!r}."
        )
    if protocol["document_sha256"] != T1_PROTOCOL_SHA256:
        raise T1ConfigError(
            f"protocol.document_sha256 {protocol['document_sha256']!r} is not the "
            f"frozen {T1_PROTOCOL_SHA256!r}."
        )
    if _require_bool("protocol", "verify_document", protocol["verify_document"]):
        validate_t1_protocol_document()

    run_class = run["run_class"]
    if run_class not in RUN_CLASSES:
        raise T1ConfigError(f"run.run_class {run_class!r} is not one of {RUN_CLASSES}.")
    attempt_id = run["attempt_id"]
    if not isinstance(attempt_id, str) or not attempt_id:
        raise T1ConfigError("run.attempt_id must be a non-empty string.")

    source_mode = thresholds["source"]
    if source_mode not in THRESHOLD_SOURCES:
        raise T1ConfigError(
            f"thresholds.source {source_mode!r} is not one of {THRESHOLD_SOURCES}."
        )
    if _require_bool("thresholds", "interpolation", thresholds["interpolation"]):
        raise T1ConfigError(
            "thresholds.interpolation must be false. The frozen rule is the raw "
            "empirical order statistic k = ceil(q*N); interpolating between "
            "neighbours is not reproducible across library versions."
        )
    if T1_THRESHOLD_INTERPOLATION_PERMITTED:  # pragma: no cover - frozen False
        raise T1ConfigError("The protocol no longer forbids interpolation; stop.")
    q_watch = _require_probability("thresholds", "q_watch", thresholds["q_watch"])
    q_event = _require_probability("thresholds", "q_event", thresholds["q_event"])
    if q_event <= q_watch:
        raise T1ConfigError(
            f"q_event {q_event!r} must exceed q_watch {q_watch!r}; EVENT is the "
            "rarer condition."
        )
    literal = _resolve_literals(thresholds, source_mode)

    profile = _resolve_profile(persistence["profile"])
    _check_expected_windows(profile, persistence["expected_windows"])

    for key in ("imputation", "forward_fill", "synthetic_zero"):
        if _require_bool("availability", key, availability[key]):
            raise T1ConfigError(
                f"availability.{key} must be false. An unavailable window carries no "
                "probability, no uncertainty and no temporal score, and nothing may "
                "be invented for it."
            )
    for key in ("hold_state", "reset_streaks", "advance_state_time"):
        if not _require_bool("availability", key, availability[key]):
            raise T1ConfigError(
                f"availability.{key} must be true; it is frozen protocol behaviour "
                "for an unavailable window."
            )
    if (
        T1_IMPUTATION_PERMITTED
        or T1_FORWARD_FILL_PERMITTED
        or T1_SYNTHETIC_ZERO_PERMITTED
    ):
        raise T1ConfigError(  # pragma: no cover - frozen False
            "The protocol no longer forbids imputation; stop."
        )

    alert_states = _require_str_tuple(
        "alerting", "emit_on_entry_to", alerting["emit_on_entry_to"]
    )
    unknown_states = sorted(set(alert_states) - set(T1_STATES))
    if unknown_states:
        raise T1ConfigError(
            f"alerting.emit_on_entry_to names non-states {unknown_states}; the four "
            f"T1 states are {T1_STATES}."
        )
    if alerting["refractory_applies_to"] != REFRACTORY_SCOPE:
        raise T1ConfigError(
            f"alerting.refractory_applies_to must be {REFRACTORY_SCOPE!r}. The "
            "refractory period is an alerting policy: it suppresses repeat "
            "notifications and never changes a state transition. A refractory "
            "period that could hold back an escalation would be a new protocol."
        )

    if _require_bool(
        "reporting",
        "context_flags_influence_transitions",
        reporting["context_flags_influence_transitions"],
    ):
        raise T1ConfigError(
            "reporting.context_flags_influence_transitions must be false. Context "
            "and confounder flags are evaluation annotation; a runtime rule that "
            "read them would not be deployable, because they do not exist on a "
            "live stream."
        )

    gap_reset = stream["max_gap_seconds_before_stream_reset"]
    if gap_reset is not None:
        gap_reset = _require_non_negative_seconds(
            "stream", "max_gap_seconds_before_stream_reset", gap_reset
        )

    if run_class == RUN_CLASS_CANONICAL:
        _require_canonical_agreement(
            stream=stream,
            detector=detector,
            thresholds=thresholds,
            cold_start=cold_start,
            q_watch=q_watch,
            q_event=q_event,
        )
    elif source_mode == THRESHOLD_SOURCE_LITERAL and literal is None:
        raise T1ConfigError("Literal thresholds were selected but none were supplied.")

    return T1EpisodeConfig(
        run_class=run_class,
        protocol_evidence=run_class == RUN_CLASS_CANONICAL,
        run_root=Path(str(run["run_root"])),
        attempt_id=attempt_id,
        sampling_frequency_hz=_require_positive_int(
            "stream", "sampling_frequency_hz", stream["sampling_frequency_hz"]
        ),
        window_samples=_require_positive_int(
            "stream", "window_samples", stream["window_samples"]
        ),
        stride_samples=_require_positive_int(
            "stream", "stride_samples", stream["stride_samples"]
        ),
        stream_key=_require_str_tuple("stream", "stream_key", stream["stream_key"]),
        order_by=str(stream["order_by"]),
        require_strictly_increasing=_require_bool(
            "stream",
            "require_strictly_increasing",
            stream["require_strictly_increasing"],
        ),
        max_gap_seconds_before_stream_reset=gap_reset,
        detector_threshold=_require_probability(
            "detector", "threshold", detector["threshold"]
        ),
        threshold_source=source_mode,
        q_watch=q_watch,
        q_event=q_event,
        literal_thresholds=literal,
        profile=profile,
        cold_start_seconds=_require_non_negative_seconds(
            "cold_start", "seconds", cold_start["seconds"]
        ),
        cold_start_requires_temporal_evidence=_require_bool(
            "cold_start",
            "requires_temporal_evidence",
            cold_start["requires_temporal_evidence"],
        ),
        signal_quality_accept=_require_str_tuple(
            "availability",
            "signal_quality_accept",
            availability["signal_quality_accept"],
        ),
        alert_on_entry_to=alert_states,
        refractory_seconds=_require_non_negative_seconds(
            "alerting", "refractory_seconds", alerting["refractory_seconds"]
        ),
        record_suppressed_alerts=_require_bool(
            "alerting", "record_suppressed_alerts", alerting["record_suppressed_alerts"]
        ),
        record_context_flags=_require_bool(
            "reporting", "record_context_flags", reporting["record_context_flags"]
        ),
        config_sha256=canonical_sha256(raw),
        protocol_document_sha256=T1_PROTOCOL_SHA256,
        source_path=source_path,
    )
