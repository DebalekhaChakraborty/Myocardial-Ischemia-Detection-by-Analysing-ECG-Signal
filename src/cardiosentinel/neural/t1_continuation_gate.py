"""The negative capability gate: proof that the continuation *cannot do too much*.

The pre-claim capability gate in `t1_capability_gate` proves a collaborator graph
can finish. This is its mirror image. It proves the continuation graph cannot
reach four things it is forbidden to do, and it exists because permission
exceeding exercise is only safe in one direction: §9 of the amendment permits
re-running the selected policy, and §9.1 narrows the authorized exercise to
exclude it. Without a mechanical gate that narrowing is prose, and prose widens
back silently.

Amendment §13.6 requires each constraint proven at **three independent layers**,
and says a proof at one layer does not substitute for another:

| # | Constraint | Counter |
|---|---|---|
| 16 | Zero state-machine regeneration | `state_machine_invocations` |
| 17 | Zero threshold generation | `threshold_generation_calls` |
| 18 | Zero policy selection | `policy_selection_calls` |
| 19 | Zero fold evaluator execution | `fold_evaluations` |

**Layer 1, structural** -- this module. The continuation's import surface reaches
no transition entry point, no threshold generator, no candidate evaluator and no
fold evaluator. Proven by walking the syntax tree and the resolved import graph,
**never by scanning source text**. Text scanning is not merely imprecise here, it
is systematically wrong: a module's own refusal list contains the words it
refuses, and an attestation asserting an action did not occur necessarily names
that action. This programme has produced that false positive five times.

**Layer 2, runtime** -- and it has two halves, because the forbidden entry points
do not all live in the same place.

`t1_protocol` is loaded on purpose: §9.1 *requires* the continuation to group and
match episodes with the frozen functions the consumed attempt used. So
`next_state`, `empirical_order_statistic`, `policy_sort_key` and
`candidate_policies` are in the process whether the continuation wants them or
not, and only a real call counter can prove they did not fire.
`instrumented_protocol_entry_points` wraps those four for the duration of a run;
each wrapper records and then refuses, since §13.7 makes a non-zero counter a
stop rather than a warning. The frozen module's file is never modified -- the
attributes are rebound in-process and restored in a `finally`.

`t1_development_run` joins it for the same reason: reaching the §16 label
authority imports `t1_fold_evaluation` -> `t1_fold_authority` ->
`t1_development_run`, so `generate_thresholds`, `select_policy` and
`run_policy_over_streams` are in the process too, and they earn call counters
rather than a carve-out.

The remaining entry points -- `trace_stream`, `evaluate_held_out`,
`T1CanonicalFoldEvaluator` -- live in `t1_fold_evaluator` and the driver graph,
which the continuation never imports at all. For those,
`prove_no_forbidden_module_loaded(binding=True)` proves the stronger property
directly: a function in a module that was never loaded cannot have been called.

Together the two halves cover every entry point. Neither alone does, and a
counter that nothing can increment would be the unreferenced-import argument
wearing a counter's clothes -- which is precisely the substitution §13.6 forbids.

**Layer 3, evidence** -- `t1_continuation_measurement`. The measured trace must
be the predecessor's trace, by digest and by per-fold threshold equality.

This module executes no scientific body and reads no evidence.
"""

from __future__ import annotations

import ast
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

GATE_NAME: Final = "T1ContinuationNegativeCapabilityGate"

#: Names that, if bound in the continuation's import surface, would mean it can
#: reach a forbidden operation. Grouped by the constraint each one would break.
FORBIDDEN_IMPORTS: Final = {
    "state_machine_invocations": (
        "next_state",
        "trace_stream",
        "T1Streaks",
        "run_policy_over_streams",
    ),
    "threshold_generation_calls": (
        "generate_thresholds",
        "empirical_order_statistic",
        "background_negative_population",
    ),
    "policy_selection_calls": (
        "select_policy",
        "policy_sort_key",
        "candidate_policies",
        "T1_SELECTION_ORDER",
    ),
    "fold_evaluations": (
        "T1CanonicalFoldEvaluator",
        "evaluate_fold",
        "evaluate_held_out",
        "widen_with_held_out_traces",
    ),
}

#: Whole modules the continuation may not import. Importing any of them binds
#: every forbidden name above transitively, whatever the import line says.
#:
#: This is a **Layer 1** list: no module in the proven continuation graph may
#: name any of these. It is not the same question as which modules may be
#: *loaded in the process* -- see `NEVER_LOADED_MODULES`.
FORBIDDEN_MODULES: Final = (
    "cardiosentinel.neural.t1_fold_evaluator",
    "cardiosentinel.neural.t1_development_run",
    "cardiosentinel.neural.t1_canonical_driver",
    "cardiosentinel.neural.t1_composition",
    "cardiosentinel.neural.t1_engine",
    "cardiosentinel.neural.t1_stream",
)

#: Modules whose absence from `sys.modules` is the runtime proof (Layer 2a).
#:
#: `t1_development_run` is deliberately **not** here. Reaching the §16 label
#: authority means importing `t1_fold_evaluation`, which imports
#: `t1_fold_authority`, which imports `t1_development_run` -- so the continuation
#: cannot both open held-out labels through the existing authority and keep that
#: module unloaded. Rather than write a second label reader, or weaken the gate
#: with a carve-out, the same rule is applied consistently:
#:
#:     a forbidden entry point in a module that must be loaded earns a real call
#:     counter; a module that need not be loaded stays unloaded.
#:
#: So `t1_development_run`'s three entry points move to the instrumented set
#: below, and the genuinely dangerous module -- `t1_fold_evaluator`, which holds
#: the fold evaluator, `trace_stream` and `evaluate_held_out` -- keeps the
#: stronger never-imported proof.
NEVER_LOADED_MODULES: Final = (
    "cardiosentinel.neural.t1_fold_evaluator",
    "cardiosentinel.neural.t1_canonical_driver",
    "cardiosentinel.neural.t1_composition",
    "cardiosentinel.neural.t1_engine",
    "cardiosentinel.neural.t1_stream",
)

#: The frozen protocol module is permitted -- amendment §9.1 *requires* episode
#: grouping and matching to be performed by the functions the consumed attempt
#: used -- but only these names may be bound from it. `next_state` lives there
#: too, so importing the module wholesale would bind the transition entry point.
PERMITTED_PROTOCOL_NAMES: Final = (
    "group_reference_episodes",
    "match_runs_to_episodes",
    "T1_EPISODE_CADENCE_SAMPLES",
    "T1ProtocolError",
    "T1_STATE_EVENT",
    "validate_t1_protocol_document",
)

PROTOCOL_MODULE: Final = "cardiosentinel.neural.t1_protocol"


class T1ContinuationCapabilityError(RuntimeError):
    """Raised when the continuation graph could reach a forbidden operation.

    Distinct from the permission and identity errors: "this graph can do too
    much" is not "you may not run", and a caller that conflated them would read
    a structural defect as a withheld authorization.
    """


# ---------------------------------------------------------------------------
# Layer 2 -- runtime counters
# ---------------------------------------------------------------------------


@dataclass
class ContinuationCounters:
    """Positive proof that nothing forbidden ran.

    Incremented only by the instrumented entry points. Read at completion, where
    every one must be zero. They start at zero and are *expected* to stay there:
    the point is not to catch a runaway but to make "nothing ran" a measured
    fact carried by the evidence rather than an assertion in a docstring.
    """

    state_machine_invocations: int = 0
    threshold_generation_calls: int = 0
    policy_selection_calls: int = 0
    fold_evaluations: int = 0
    #: Every forbidden call actually observed, for a failure receipt to carry.
    observed: list[str] = field(default_factory=list)

    def record(self, counter: str, detail: str = "") -> None:
        """Record a forbidden invocation. Reaching here is already a defect."""
        if not hasattr(self, counter):
            raise T1ContinuationCapabilityError(f"Unknown counter {counter!r}.")
        setattr(self, counter, getattr(self, counter) + 1)
        self.observed.append(f"{counter}:{detail}" if detail else counter)

    def as_dict(self) -> dict[str, int]:
        return {
            "state_machine_invocations": self.state_machine_invocations,
            "threshold_generation_calls": self.threshold_generation_calls,
            "policy_selection_calls": self.policy_selection_calls,
            "fold_evaluations": self.fold_evaluations,
        }

    def require_all_zero(self) -> dict[str, int]:
        """A non-zero counter is a refusal, not a warning (amendment §13.7)."""
        counts = self.as_dict()
        nonzero = {name: value for name, value in counts.items() if value != 0}
        if nonzero:
            raise T1ContinuationCapabilityError(
                f"The continuation performed forbidden operations: {nonzero}. "
                f"Observed: {self.observed}. Amendment §13.6 requires each of "
                "these to be zero; a non-zero counter stops the continuation "
                "rather than annotating it."
            )
        return counts


# ---------------------------------------------------------------------------
# Layer 1 -- structural proof
# ---------------------------------------------------------------------------


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if source is None:  # pragma: no cover - namespace packages have no file
        raise T1ContinuationCapabilityError(f"{module_name} has no source file.")
    return Path(source)


def _bound_names(tree: ast.AST) -> tuple[set[str], dict[str, set[str]]]:
    """Every module imported and every name bound from it, from the syntax tree."""
    modules: set[str] = set()
    names: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
                names.setdefault(alias.name, set()).add("*")
    return modules, names


def _deferred_imports(tree: ast.AST) -> set[str]:
    """Modules imported inside a function body rather than at module level.

    A deferred import is still an import: `t1_development_run.main` defers its
    driver imports and reaches the whole graph anyway. Counting only
    module-level imports would let the continuation hide a forbidden reach one
    indent deeper.
    """
    deferred: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                deferred.add(inner.module)
            elif isinstance(inner, ast.Import):
                deferred.update(alias.name for alias in inner.names)
    return deferred


def prove_import_surface(module_names: Iterable[str]) -> dict[str, Any]:
    """Prove no continuation module reaches a forbidden entry point.

    Walks the syntax tree of each module -- module-level *and* deferred imports
    -- and refuses on any forbidden module, any forbidden bound name, and any
    wholesale import of the frozen protocol module.
    """
    violations: list[str] = []
    surface: dict[str, dict[str, Any]] = {}

    for module_name in module_names:
        path = _module_path(module_name)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules, names = _bound_names(tree)
        deferred = _deferred_imports(tree)
        all_modules = modules | deferred
        surface[module_name] = {
            "imports": sorted(all_modules),
            "deferred_imports": sorted(deferred),
        }

        for forbidden in FORBIDDEN_MODULES:
            if forbidden in all_modules:
                violations.append(f"{module_name} imports {forbidden}")

        bound: set[str] = set()
        for source, imported in names.items():
            bound |= imported
            if source == PROTOCOL_MODULE:
                if "*" in imported:
                    violations.append(
                        f"{module_name} imports {PROTOCOL_MODULE} wholesale, "
                        "which binds the transition entry point next_state. "
                        "Bind only the episode functions §9.1 requires."
                    )
                extra = sorted(imported - set(PERMITTED_PROTOCOL_NAMES))
                if extra:
                    violations.append(
                        f"{module_name} binds {extra} from {PROTOCOL_MODULE}; "
                        f"only {list(PERMITTED_PROTOCOL_NAMES)} are permitted."
                    )

        for counter, forbidden_names in FORBIDDEN_IMPORTS.items():
            hit = sorted(bound & set(forbidden_names))
            if hit:
                violations.append(
                    f"{module_name} binds {hit}, which would break {counter} = 0"
                )

    if violations:
        raise T1ContinuationCapabilityError(
            f"{GATE_NAME} refuses: the continuation's import surface reaches "
            "operations amendment §9.1 forbids it to perform.\n  "
            + "\n  ".join(violations)
        )
    return surface


def prove_no_forbidden_calls(module_names: Iterable[str]) -> dict[str, list[str]]:
    """Prove no continuation module *calls* a forbidden operation.

    Complements the import proof: a module could reach a forbidden function via
    an attribute on an object it was handed rather than through an import. Keys
    on the call node in the syntax tree, so a docstring naming an operation --
    which every refusal necessarily does -- is invisible here.
    """
    forbidden = {name for group in FORBIDDEN_IMPORTS.values() for name in group}
    violations: list[str] = []
    calls: dict[str, list[str]] = {}

    for module_name in module_names:
        tree = ast.parse(_module_path(module_name).read_text(encoding="utf-8"))
        observed: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:  # pragma: no cover - call on a subscript or literal
                continue
            observed.add(name)
            if name in forbidden:
                violations.append(f"{module_name} calls {name}")
        calls[module_name] = sorted(observed)

    if violations:
        raise T1ContinuationCapabilityError(
            f"{GATE_NAME} refuses: forbidden call sites.\n  " + "\n  ".join(violations)
        )
    return calls


def prove_no_forbidden_module_loaded(
    counters: ContinuationCounters, *, binding: bool = False
) -> dict[str, Any]:
    """Prove no forbidden module was imported into this interpreter.

    Runtime evidence of the strongest available kind for the entry points that
    live in modules the continuation never imports: a function in a module that
    was never loaded cannot have been called, whatever any counter says.

    `binding` is False by default because the test suite legitimately imports
    `t1_fold_evaluator` to prove the re-implemented helpers are equivalent, and
    a check that refused there would make that proof impossible. The **run path
    passes `binding=True`**, where a dirty interpreter is a refusal.
    """
    loaded = sorted(name for name in NEVER_LOADED_MODULES if name in sys.modules)
    if loaded and binding:
        raise T1ContinuationCapabilityError(
            f"{GATE_NAME} refuses: {loaded} are loaded in this interpreter. "
            "The continuation's runtime proof that it never ran a fold "
            "evaluator, a state machine or a threshold generator rests on "
            "those modules being unreachable. A loaded module is reachable."
        )
    return {
        "forbidden_modules_loaded": loaded,
        "clean_interpreter": not loaded,
        "binding": binding,
        "counters": counters.as_dict(),
    }


#: The forbidden entry points that live inside the **permitted** protocol
#: module, mapped to the counter each would break.
#:
#: These are the reason a module-load proof is not sufficient on its own.
#: `t1_protocol` is loaded on purpose -- amendment §9.1 requires the continuation
#: to group and match episodes with the frozen functions the consumed attempt
#: used -- so `next_state` and its neighbours are in the process whether the
#: continuation wants them or not. They are the four entry points that need a
#: real call counter rather than an absence argument.
PROTOCOL_INSTRUMENTED_ENTRY_POINTS: Final = {
    "next_state": "state_machine_invocations",
    "empirical_order_statistic": "threshold_generation_calls",
    "policy_sort_key": "policy_selection_calls",
    "candidate_policies": "policy_selection_calls",
}

#: The same treatment for `t1_development_run`, which the §16 label authority
#: drags into the process. These three are the reason it was on the
#: never-loaded list in the first place; instrumenting them keeps the proof
#: rather than trading it away.
DEVELOPMENT_RUN_MODULE: Final = "cardiosentinel.neural.t1_development_run"
DEVELOPMENT_RUN_INSTRUMENTED_ENTRY_POINTS: Final = {
    "generate_thresholds": "threshold_generation_calls",
    "select_policy": "policy_selection_calls",
    "run_policy_over_streams": "state_machine_invocations",
}

#: Every instrumented entry point, by the module that holds it.
INSTRUMENTED_ENTRY_POINTS: Final = {
    PROTOCOL_MODULE: PROTOCOL_INSTRUMENTED_ENTRY_POINTS,
    DEVELOPMENT_RUN_MODULE: DEVELOPMENT_RUN_INSTRUMENTED_ENTRY_POINTS,
}


@contextmanager
def instrumented_protocol_entry_points(counters: ContinuationCounters):
    """Instrument the protocol's forbidden entry points for the duration of a run.

    Amendment §13.6 Layer 2 asks for the entry points themselves to be
    instrumented, and distinguishes that from Layer 1 precisely because "a
    counter that reads zero is evidence, whereas an unreferenced import is only
    an argument". A counter nothing can increment is the argument wearing a
    counter's clothes, so these wrappers make the four reachable entry points
    genuinely countable.

    Each wrapper records the call and then refuses, because §13.7 makes a
    non-zero counter a stop rather than a warning. Recording before raising is
    deliberate: the receipt can then name which entry point fired instead of
    only reporting that something did.

    The frozen module's **file is never modified**. The attributes are rebound
    in this process for the duration of the block and restored in a `finally`,
    so the module digest that seven suites pin is untouched.
    """
    original: dict[tuple[str, str], Any] = {}

    def _tripwire(name: str, counter: str, wrapped: Any) -> Any:
        def guard(*args: Any, **kwargs: Any) -> Any:
            counters.record(counter, name)
            raise T1ContinuationCapabilityError(
                f"The continuation called {name}, which amendment §9.1 forbids "
                f"it to call. {counter} is now "
                f"{getattr(counters, counter)}; the continuation stops."
            )

        guard.__name__ = getattr(wrapped, "__name__", name)
        guard.__doc__ = getattr(wrapped, "__doc__", None)
        guard.__wrapped__ = wrapped
        return guard

    try:
        for module_name, entry_points in INSTRUMENTED_ENTRY_POINTS.items():
            # Only instrument a module that is already loaded. Importing
            # `t1_development_run` here purely to wrap it would load the very
            # module the caller may be trying to keep out of its process.
            if module_name != PROTOCOL_MODULE and module_name not in sys.modules:
                continue
            module = importlib.import_module(module_name)
            for name, counter in entry_points.items():
                if not hasattr(module, name):
                    raise T1ContinuationCapabilityError(
                        f"{module_name} has no entry point {name!r} to "
                        "instrument. The gate is bound to a module that has "
                        "moved, so it can no longer prove what it claims."
                    )
                wrapped = getattr(module, name)
                original[(module_name, name)] = wrapped
                setattr(module, name, _tripwire(name, counter, wrapped))
        yield counters
    finally:
        for (module_name, name), wrapped in original.items():
            setattr(sys.modules[module_name], name, wrapped)


def prove_negative_capability(
    module_names: Iterable[str],
    counters: ContinuationCounters | None = None,
    *,
    require_clean_interpreter: bool = False,
) -> dict[str, Any]:
    """Run every structural layer, then the runtime layer. Refuses on any.

    `require_clean_interpreter` is what the run path sets. Left False, this
    proves Layers 1 and 2b; set True it additionally proves 2a, which is the
    half that covers the entry points living in never-imported modules.
    """
    modules = tuple(module_names)
    surface = prove_import_surface(modules)
    calls = prove_no_forbidden_calls(modules)
    counters = counters if counters is not None else ContinuationCounters()
    interpreter = prove_no_forbidden_module_loaded(
        counters, binding=require_clean_interpreter
    )
    zeroed = counters.require_all_zero()
    return {
        "gate": GATE_NAME,
        "interpreter": interpreter,
        "instrumented_entry_points": {
            module: dict(points) for module, points in INSTRUMENTED_ENTRY_POINTS.items()
        },
        "modules_proven": list(modules),
        "import_surface": surface,
        "call_surface": calls,
        "counters": zeroed,
        "layer_1_structural": True,
        "layer_2_runtime": True,
    }
