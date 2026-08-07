"""SYNTHETIC target semantics and annotation-firewall tests."""

import ast
from dataclasses import replace
from pathlib import Path

from cardiosentinel.data.models import (
    AnnotationMarker,
    AnnotationSample,
    DatasetRecord,
    SignalQualityInterval,
    SourceCensoredInterval,
    STEvent,
)
from cardiosentinel.evaluation.models import BenchmarkWindow
from cardiosentinel.evaluation.targets import (
    assign_window_target,
    iter_causal_window_indices,
)


def record(sample_count: int = 4_000, subject: str = "ltstdb:s0000") -> DatasetRecord:
    return DatasetRecord(
        "ltstdb",
        "1.0.0",
        f"{subject.split(':')[1]}1",
        subject,
        100.0,
        ("I",),
        ("I",),
        1,
        sample_count,
        sample_count / 100.0,
        "SYNTHETIC",
    )


def event(subtype: str, onset: int, end: int) -> STEvent:
    source = AnnotationSample(onset, "s", aux_note="SYNTHETIC")
    return STEvent(
        "ltstdb",
        "s00001",
        "ltstdb:s0000",
        0,
        "st_episode",
        subtype,
        onset,
        (onset + end) // 2,
        end,
        onset / 100.0,
        (onset + end) / 200.0,
        end / 100.0,
        None,
        None,
        "stb",
        "ltstdb.stb",
        True,
        (source,),
    )


def target_at(
    start: int,
    *,
    events: tuple[STEvent, ...] = (),
    quality: tuple[SignalQualityInterval, ...] = (),
    markers: tuple[AnnotationMarker, ...] = (),
    censored: tuple[SourceCensoredInterval, ...] = (),
):
    window = BenchmarkWindow(
        "ltstdb",
        "s00001",
        "ltstdb:s0000",
        0,
        "I",
        100.0,
        start,
        start + 1_000,
        start + 1_000,
    )
    return assign_window_target(
        window, events, quality, markers, censored, "ltstdb.stb"
    )


def test_positive_requires_full_containment_and_boundary_is_ambiguous() -> None:
    ischemic = event("ischemic", 500, 2_500)
    positive = target_at(1_000, events=(ischemic,))
    boundary = target_at(0, events=(ischemic,))
    assert positive.target_family == "ischemic_positive"
    assert positive.eligible_for_primary_evaluation is True
    assert boundary.target_family == "boundary_ambiguous"
    assert boundary.eligible_for_primary_evaluation is False


def test_background_and_rate_related_states_remain_distinct() -> None:
    assert target_at(0).target_family == "background_negative"
    rate = target_at(1_000, events=(event("heart_rate_related", 500, 2_500),))
    assert rate.target_family == "rate_related_confounder"
    assert rate.eligible_for_confounder_evaluation is True
    assert rate.eligible_for_training is False


def test_edb_reference_and_apparent_st_events_map_to_compatible_states() -> None:
    reference = replace(
        event("reference_st_change", 500, 2_500),
        dataset_id="edb",
        record_id="e0001",
        subject_id="edb:e0001",
        event_family="st_change",
        annotation_source="atr",
        annotation_definition="edb.reference",
    )
    apparent = replace(
        reference,
        event_family="axis_shift",
        event_subtype="apparent_st_change",
        annotation_definition="edb.axis_shift",
        is_primary_definition=False,
    )
    window = BenchmarkWindow(
        "edb", "e0001", "edb:e0001", 0, "I", 100.0, 1_000, 2_000, 2_000
    )
    reference_target = assign_window_target(
        window, (reference,), (), (), (), "edb.reference"
    )
    apparent_target = assign_window_target(
        window, (apparent,), (), (), (), "edb.reference"
    )
    assert reference_target.target_family == "ischemic_positive"
    assert apparent_target.target_family == "axis_shift_confounder"


def test_marker_vicinities_are_challenges_not_invented_episode_durations() -> None:
    raw = AnnotationSample(1_500, "s", aux_note="SYNTHETIC")
    axis = AnnotationMarker(
        "s00001", "ltstdb:s0000", 0, 1_500, "st_shift", "axis_related", "stb", raw
    )
    conduction = AnnotationMarker(
        "s00001",
        "ltstdb:s0000",
        0,
        1_500,
        "st_shift",
        "conduction_related",
        "stb",
        raw,
    )
    assert target_at(1_000, markers=(axis,)).target_family == "axis_shift_confounder"
    assert (
        target_at(1_000, markers=(conduction,)).target_family
        == "conduction_change_confounder"
    )


def test_unreadable_and_source_censored_regions_are_ineligible() -> None:
    raw = AnnotationSample(500, "s", aux_note="SYNTHETIC")
    unreadable = SignalQualityInterval(
        "s00001", "ltstdb:s0000", 0, 500, 2_500, "unreadable", "stb", (raw,)
    )
    unknown = SourceCensoredInterval(
        "s00001",
        "ltstdb:s0000",
        0,
        500,
        2_500,
        "right_censored_ischemic_episode",
        "stb",
        (raw,),
    )
    excluded = target_at(1_000, quality=(unreadable,))
    censored = target_at(1_000, censored=(unknown,))
    assert excluded.target_family == "quality_excluded"
    assert excluded.eligible_for_training is False
    assert censored.target_family == "source_censored_or_unknown"


def test_window_geometry_is_annotation_independent_and_causal() -> None:
    windows = tuple(iter_causal_window_indices(record(), 10.0, 5.0))
    assert [(item.start_sample, item.end_sample) for item in windows[:3]] == [
        (0, 1_000),
        (500, 1_500),
        (1_000, 2_000),
    ]
    assert all(item.available_at_sample == item.end_sample for item in windows)


def test_predictive_signal_layer_cannot_import_targets_or_annotations() -> None:
    signal_root = Path(__file__).resolve().parents[2] / "src/cardiosentinel/signal"
    forbidden_imports = ("cardiosentinel.data", "cardiosentinel.evaluation")
    forbidden_names = {"STEvent", "WindowTarget", "peak_deviation_uv"}
    for path in signal_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: set[str] = set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        assert not any(item.startswith(forbidden_imports) for item in imports)
        assert forbidden_names.isdisjoint(names)
