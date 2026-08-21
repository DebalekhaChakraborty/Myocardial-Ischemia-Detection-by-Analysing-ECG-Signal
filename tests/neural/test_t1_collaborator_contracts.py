"""Output contracts for the values the composition root threads.

The canonical attempt was consumed by a defect this file exists to make
impossible: the fold evaluator counted PRIMARY margins as `tp/fp/tn/fn`, the
assembly layer read `true_positive/...`, and the composition root forwarded the
producer's dictionary unchanged. Both layers were internally consistent and
separately tested. Only their junction was wrong, and the junction was reached
for the first time at stage 24 of 29 -- after the claim, after twelve folds.

The lesson is narrower than "add a test". Bound values that resolve lazily are
not exercised by the tests that cover their producer or their consumer, so the
contract between them needs a test of its own. What follows derives the
consumer's requirement from the consumer's own syntax tree and checks the
producer against it, so a rename on either side fails here rather than at the
twenty-fourth stage.

Nothing here runs the science: no canonical run is started, no VALIDATION row is
read, no label is opened and TEST stays sealed.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_assembly as ASSEMBLY
from cardiosentinel.neural import t1_composition as COMP
from cardiosentinel.neural import t1_final_configuration as FINAL
from cardiosentinel.neural import t1_fold_evaluator as EVAL
from cardiosentinel.neural import t1_persistence as PERSIST

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Reading a requirement out of the consumer, rather than restating it here
# ---------------------------------------------------------------------------


def _required_keys(function, parameter: str) -> frozenset[str]:
    """Every constant key the function reads from `parameter`.

    Syntax tree, never text. A substring scan over this repository reports the
    word it is looking for from the prose that explains why the word is
    forbidden, which has produced false positives here before. Subscripts are
    unambiguous: `mapping["key"]` is a read, and a sentence about `"key"` is
    not.
    """
    tree = ast.parse(inspect.getsource(function))
    return frozenset(
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == parameter
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    )


def _source_tree(function) -> ast.AST:
    """Parse a function, method or closure. Methods arrive indented."""
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _string_keys(node: ast.Dict) -> frozenset[str]:
    return frozenset(
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    )


def _dict_literal_keys(function, variable: str) -> frozenset[str]:
    """The string keys of a dict literal assigned to `variable`."""
    for node in ast.walk(_source_tree(function)):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            return _string_keys(node.value)
    raise AssertionError(f"{variable!r} is not assigned a dict literal")


def _nested_dict_keys(function, key: str) -> frozenset[str]:
    """The string keys of the dict literal that `key` maps to."""
    for node in ast.walk(_source_tree(function)):
        if not isinstance(node, ast.Dict):
            continue
        for name, value in zip(node.keys, node.values):
            if (
                isinstance(name, ast.Constant)
                and name.value == key
                and isinstance(value, ast.Dict)
            ):
                return _string_keys(value)
    raise AssertionError(f"{key!r} does not map to a dict literal")


# ---------------------------------------------------------------------------
# A held-out trace, shaped exactly as the evaluator returns one
# ---------------------------------------------------------------------------


class _Run:
    """The only thing the pooling helpers read is `held_out_traces`."""

    def __init__(self, traces: dict[int, dict]) -> None:
        self.held_out_traces = traces


def _trace(**overrides) -> dict:
    trace = {
        "primary_confusion": {"tp": 3, "fp": 2, "tn": 90, "fn": 5},
        "episode_evidence": {
            "reference_episodes": 7,
            "predicted_event_runs": 4,
            "matched_episodes": 3,
            "unmatched_predicted_runs": 1,
        },
        "onset_latency_seconds": (12.0, 30.0),
    }
    trace.update(overrides)
    return trace


def _run(count: int = 3) -> _Run:
    return _Run({index: _trace() for index in range(count)})


# ---------------------------------------------------------------------------
# 1. The contract that was broken
# ---------------------------------------------------------------------------


def test_the_pooled_confusion_supplies_every_key_the_assembly_layer_reads():
    """The exact defect that consumed the canonical attempt, as a test."""
    required = _required_keys(ASSEMBLY._build_assemble_oof_result, "primary_confusion")
    assert required, "the consumer reads no confusion key; the probe is broken"
    produced = frozenset(COMP._pooled_confusion(_run()))
    missing = required - produced
    assert missing == frozenset(), (
        f"the composition root supplies {sorted(produced)} but the assembly "
        f"layer reads {sorted(required)}; {sorted(missing)} would raise at "
        "stage 24, after the claim"
    )


def test_the_pooled_episode_evidence_supplies_every_key_the_consumer_reads():
    required = _required_keys(ASSEMBLY._build_assemble_oof_result, "episode_evidence")
    produced = frozenset(COMP._pooled_episode_evidence(_run()))
    assert required - produced == frozenset()


def test_the_translation_covers_exactly_what_the_evaluator_counts():
    """Both ends of the map are checked against the code that owns them."""
    counted = _dict_literal_keys(
        EVAL.T1CanonicalFoldEvaluator.evaluate_held_out, "confusion"
    )
    assert frozenset(COMP.PRIMARY_CONFUSION_KEYS) == counted, (
        "the evaluator counts keys the translation does not name"
    )
    reported = frozenset(COMP.PRIMARY_CONFUSION_KEYS.values())
    assert len(reported) == len(COMP.PRIMARY_CONFUSION_KEYS), (
        "two counted keys map to one reported key"
    )
    assert reported == _required_keys(
        ASSEMBLY._build_assemble_oof_result, "primary_confusion"
    )


def test_the_pooled_confusion_sums_rather_than_replaces():
    pooled = COMP._pooled_confusion(_run(count=3))
    assert pooled["true_positive"] == 9
    assert pooled["false_positive"] == 6
    assert pooled["true_negative"] == 270
    assert pooled["false_negative"] == 15


# ---------------------------------------------------------------------------
# 2. A missing count is refused, never defaulted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", sorted(COMP.PRIMARY_CONFUSION_KEYS))
def test_a_missing_confusion_count_is_refused(dropped):
    """Silently defaulting to zero is how a margin nothing produced is reported."""
    confusion = {k: 1 for k in COMP.PRIMARY_CONFUSION_KEYS if k != dropped}
    run = _Run({0: _trace(primary_confusion=confusion)})
    with pytest.raises(COMP.T1CompositionError) as caught:
        COMP._pooled_confusion(run)
    message = str(caught.value)
    assert dropped in message
    assert "zero" in message


def test_a_missing_count_names_the_fold_it_came_from():
    run = _Run({0: _trace(), 7: _trace(primary_confusion={"tp": 1})})
    with pytest.raises(COMP.T1CompositionError, match="Fold 7"):
        COMP._pooled_confusion(run)


# ---------------------------------------------------------------------------
# 3. Every mapping the composition root threads has a checked contract
# ---------------------------------------------------------------------------

# Producer, consumer, and the parameter the consumer reads it as. Adding a
# lazily-bound mapping to the composition root without adding it here is what
# `test_every_lazily_bound_mapping_is_covered` refuses.
MAPPING_CONTRACTS = {
    "primary_confusion": (COMP._pooled_confusion, ASSEMBLY._build_assemble_oof_result),
    "episode_evidence": (
        COMP._pooled_episode_evidence,
        ASSEMBLY._build_assemble_oof_result,
    ),
}

# The third lazily bound mapping. Its producer is a closure over the run and its
# consumer reads it through a loop rather than by constant subscript, so neither
# end can be probed the way the two above are -- both sides are compared as key
# sets instead, which is the same contract by a different route.
STATIC_CONTRACTS = {
    "configuration": (
        lambda: _nested_dict_keys(
            FINAL.select_final_validation_configuration, "configuration"
        ),
        lambda: frozenset(ASSEMBLY.FINAL_CONFIGURATION_FIELDS),
    ),
}


@pytest.mark.parametrize("parameter", sorted(STATIC_CONTRACTS))
def test_each_statically_compared_mapping_satisfies_its_consumer(parameter):
    produced, required = (probe() for probe in STATIC_CONTRACTS[parameter])
    assert required - produced == frozenset(), (
        f"{parameter}: the selector supplies {sorted(produced)}, the assembly "
        f"layer reads {sorted(required)}"
    )


@pytest.mark.parametrize("parameter", sorted(MAPPING_CONTRACTS))
def test_each_threaded_mapping_satisfies_its_consumer(parameter):
    producer, consumer = MAPPING_CONTRACTS[parameter]
    required = _required_keys(consumer, parameter)
    produced = frozenset(producer(_run()))
    assert required - produced == frozenset(), (
        f"{parameter}: producer supplies {sorted(produced)}, consumer reads "
        f"{sorted(required)}"
    )


def test_every_lazily_bound_mapping_is_covered():
    """A new `_LazyMapping` in the composition root must arrive with a contract.

    Read from the syntax tree of `build_canonical_collaborators`: every keyword
    argument bound to a `_LazyMapping` names a mapping that resolves during the
    run, which is precisely the binding no producer test and no consumer test
    exercises.
    """
    source = Path(COMP.__file__).read_text(encoding="utf-8")
    builder = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_canonical_collaborators"
    )
    lazily_bound = {
        keyword.arg
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "_LazyMapping"
    }
    uncovered = lazily_bound - set(MAPPING_CONTRACTS) - set(STATIC_CONTRACTS)
    assert uncovered == set(), (
        f"{sorted(uncovered)} resolve during the run with no contract test. A "
        "lazily bound mapping is not exercised by its producer's tests or its "
        "consumer's, which is how the stage-24 defect survived both."
    )


def test_the_pooled_latency_is_a_sequence_of_floats():
    latency = COMP._pooled_latency(_run(count=2))
    assert len(latency) == 4
    assert all(isinstance(value, float) for value in latency)


# ---------------------------------------------------------------------------
# 4. Nothing here consumes anything
# ---------------------------------------------------------------------------


def test_no_run_directory_is_created_by_these_tests():
    """The consumed attempt is not touched, and no new one is claimed."""
    canonical = PERSIST.canonical_run_directory(REPOSITORY_ROOT)
    existed = canonical.exists()
    COMP._pooled_confusion(_run())
    COMP._pooled_episode_evidence(_run())
    assert canonical.exists() is existed
    assert not (
        REPOSITORY_ROOT / "cardiosentinel-runs" / "phase9-t1-continuation-v1"
    ).exists()
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()
