"""The pre-claim capability gate.

`T1ExecutionCollaborators.require_complete` proves every collaborator is
*bound*. It cannot prove any of them can *finish*, because ``callable`` is true
of a function whose entire body is a refusal. `T1NonExecutingFoldEvaluator` is
exactly that object, and it is a legitimate one: it completes the collaborator
graph on purpose so the architecture can be reviewed without being armed.

Bound-but-unable is the dangerous combination. The canonical driver claims the
attempt at stage 10 and first calls a fold evaluator at stage 17, so a graph
that passes the binding check and refuses at stage 17 consumes the single
canonical attempt to discover something that was knowable before stage 1. That
would break the one invariant this layer exists to hold:

    an attempt must never be consumed by an execution path that cannot
    complete.

This module closes that gap with three independent checks, all of which run
before the claim and none of which executes a scientific body.

**1. Shape.** ``inspect.Signature.bind`` proves the collaborator accepts the
call the driver actually makes -- two positionals for ``evaluate_fold``, the
named keywords for each assembly step. Binding resolves arguments without
invoking anything, so a collaborator whose signature could only raise
``TypeError`` at stage 23 is refused at stage 0.

**2. Attestation.** Every collaborator must positively declare itself
execution-capable by exposing ``t1_execution_capability``. Silence is a
refusal, never a pass. This is deliberately an allowlist: a denylist of known
placeholder types would admit the next placeholder written, and the whole point
is to be safe against implementations that do not exist yet. The attribute is
either a `T1CapabilityAttestation` or a pure zero-argument callable returning
one; reading it is metadata access, not scientific work.

**3. Structural proof.** An attestation is a claim, and a claim can be wrong.
A function whose body contains no reachable ``return`` and no ``yield`` cannot
produce the mapping the driver threads into the next stage, whatever it says
about itself. That is a proof rather than a heuristic, so when the two
disagree the proof wins and the attestation is reported as false.

**What this module does not do.** It grants nothing, reads no evidence, opens
no label, resolves no path, creates no directory and consults no permission.
`t1_config.T1_EXECUTION_SPECIFICATION_AUTHORIZED` is never read here: whether a
run *could* finish and whether it *may* start are different questions, and
answering the first must stay available to a reviewer who is not answering the
second.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping

from cardiosentinel.neural.t1_execution_spec import (
    STAGE_ASSEMBLE_LABEL_BLIND,
    STAGE_BOOTSTRAP,
    STAGE_CHALLENGE,
    STAGE_CLAIM,
    STAGE_FINAL_CONFIGURATION,
    STAGE_FOLD_RUN_CANDIDATES,
    STAGE_OOF_RESULT,
    STAGE_OOF_STATE_EVIDENCE,
    T1_STAGE_ORDER,
)

# The attribute a collaborator exposes to declare itself execution-capable.
T1_CAPABILITY_ATTRIBUTE: Final = "t1_execution_capability"

GATE_NAME: Final = "T1PreClaimCapabilityGate"


class T1CapabilityError(RuntimeError):
    """Raised when a collaborator graph cannot complete a canonical run.

    Distinct from `T1DriverError` on purpose. "You may not run" and "this run
    could not finish" are different refusals, and a caller that conflated them
    would read a missing capability as a withheld permission.
    """


# ---------------------------------------------------------------------------
# What the driver calls, and where a failure would land
# ---------------------------------------------------------------------------


# The post-claim stage each collaborator is first reached from. The value of
# this map is the refusal message: it lets the gate say which stage would have
# failed *after* the attempt was spent, rather than reporting an abstract
# incompleteness.
CAPABILITY_STAGE_BINDINGS: Final = {
    "subject_of_record": STAGE_ASSEMBLE_LABEL_BLIND,
    "evaluate_fold": STAGE_FOLD_RUN_CANDIDATES,
    "assemble_oof_state_columns": STAGE_OOF_STATE_EVIDENCE,
    "assemble_oof_result": STAGE_OOF_RESULT,
    "assemble_subject_evidence": STAGE_BOOTSTRAP,
    "assemble_bootstrap": STAGE_BOOTSTRAP,
    "assemble_challenge": STAGE_CHALLENGE,
    "assemble_final_configuration": STAGE_FINAL_CONFIGURATION,
}

# The call each collaborator must accept, as (positional count, keyword names).
# Transcribed from the call sites in `t1_canonical_driver.execute` and
# `t1_development_run.stage_folds`; a drift test binds the two together.
CAPABILITY_CALL_CONTRACT: Final = {
    "subject_of_record": (1, ()),
    "evaluate_fold": (2, ()),
    "assemble_oof_state_columns": (0, ("columns", "selections")),
    "assemble_oof_result": (0, ("oof_columns", "selections")),
    "assemble_subject_evidence": (0, ("oof_columns",)),
    "assemble_bootstrap": (0, ("oof_columns",)),
    "assemble_challenge": (0, ("oof_columns",)),
    "assemble_final_configuration": (0, ("oof_columns", "selections")),
}

# Declaration order is the driver's reach order, not alphabetical order, so a
# refusal names the earliest stage that would have failed rather than whichever
# role sorts first. A gate that reported a later failure would understate how
# much of the run had already been spent.
REQUIRED_CAPABILITY_ROLES: Final = tuple(CAPABILITY_STAGE_BINDINGS)

_REACH_ORDER: Final = [
    T1_STAGE_ORDER.index(stage) for stage in CAPABILITY_STAGE_BINDINGS.values()
]
if _REACH_ORDER != sorted(_REACH_ORDER):  # pragma: no cover - guarded by a test
    raise T1CapabilityError(
        "The capability bindings are not in the order the driver reaches them, "
        "so a refusal would misreport which stage fails first."
    )

# Roles whose absence would be discovered only after the claim. Every one of
# them, by construction: the claim is stage 10 and the earliest collaborator is
# reached at stage 12.
_CLAIM_INDEX: Final = T1_STAGE_ORDER.index(STAGE_CLAIM)


class _Unbindable:
    """A placeholder that proves nothing about itself.

    Used only as a stand-in argument for `inspect.Signature.bind`, which
    resolves parameters without touching the values it binds.
    """

    __slots__ = ()


_PLACEHOLDER: Final = _Unbindable()


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class T1CapabilityAttestation:
    """One collaborator's declaration about whether it can finish.

    ``executes`` is the only load-bearing field and it is never inferred from
    the object's existence. ``reason`` is carried so a refusal can quote the
    implementation's own words rather than paraphrase them.
    """

    role: str
    provider: str
    executes: bool
    reason: str

    def __post_init__(self) -> None:
        if self.role not in CAPABILITY_STAGE_BINDINGS:
            raise T1CapabilityError(
                f"{self.role!r} is not a collaborator the canonical driver "
                f"threads. The roles are {list(REQUIRED_CAPABILITY_ROLES)}."
            )
        if not self.reason.strip():
            raise T1CapabilityError(
                f"The attestation for {self.role!r} carries no reason. An "
                "unexplained capability claim is not reviewable."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "executes": self.executes,
            "reason": self.reason,
            "first_reached_at_stage": CAPABILITY_STAGE_BINDINGS[self.role],
        }


def attest(
    role: str, provider: str, *, executes: bool, reason: str
) -> T1CapabilityAttestation:
    """Build an attestation. Sugar over the dataclass, validated identically."""
    return T1CapabilityAttestation(
        role=role, provider=provider, executes=executes, reason=reason
    )


def declare_execution_capability(
    role: str, *, executes: bool, reason: str
) -> Callable[[Any], Any]:
    """Attach an attestation to a function, method or lambda.

    Returns the object unchanged apart from the attribute, so decorating a
    collaborator never wraps it: a gate that changed the thing it verified
    would be verifying something other than what runs.
    """

    def decorate(target: Any) -> Any:
        setattr(
            target,
            T1_CAPABILITY_ATTRIBUTE,
            attest(
                role, provider=_provider_name(target), executes=executes, reason=reason
            ),
        )
        return target

    if not role:
        raise T1CapabilityError("A capability declaration names a role.")
    if not reason:
        raise T1CapabilityError("A capability declaration states a reason.")
    return decorate


def _provider_name(target: Any) -> str:
    """A stable, human-readable name for whatever implements a role."""
    for attribute in ("__qualname__", "__name__"):
        name = getattr(target, attribute, None)
        if isinstance(name, str) and name:
            return name
    return type(target).__name__


def read_attestation(role: str, collaborator: Any) -> T1CapabilityAttestation | None:
    """The collaborator's own declaration, or ``None`` if it makes none.

    ``None`` is not a soft pass. Every caller in this module treats it as a
    refusal; it is distinguished from a negative attestation only so the
    message can say "declares nothing" instead of "declares it cannot run".
    """
    declared = getattr(collaborator, T1_CAPABILITY_ATTRIBUTE, None)
    if declared is None:
        return None
    if isinstance(declared, T1CapabilityAttestation):
        return declared
    if callable(declared):
        # A zero-argument metadata accessor. Documented as pure; it is not the
        # scientific body and carries no arguments through which one could be
        # reached.
        produced = declared()
        if not isinstance(produced, T1CapabilityAttestation):
            raise T1CapabilityError(
                f"{role!r} exposes {T1_CAPABILITY_ATTRIBUTE!r} but it produced "
                f"{type(produced).__name__}, not a T1CapabilityAttestation."
            )
        return produced
    raise T1CapabilityError(
        f"{role!r} exposes {T1_CAPABILITY_ATTRIBUTE!r} as "
        f"{type(declared).__name__}, which is neither an attestation nor a "
        "callable returning one."
    )


# ---------------------------------------------------------------------------
# Structural proof: can this body return at all?
# ---------------------------------------------------------------------------


def _implementation_of(collaborator: Any) -> Any:
    """The object the driver would actually invoke.

    For a plain function that is the function; for an instance it is the
    class's ``__call__``, because that is what the call site reaches. Checking
    anything else would prove a property of code that never runs.
    """
    if inspect.isfunction(collaborator) or inspect.ismethod(collaborator):
        return collaborator
    call = getattr(type(collaborator), "__call__", None)
    if call is None:  # pragma: no cover - non-callables fail the bind check
        return None
    if getattr(call, "__objclass__", None) is type:  # pragma: no cover
        return None
    return call


def _returns_a_value(function: Any) -> bool | None:
    """Whether a body has any statement that could produce a value.

    Returns ``None`` when the source cannot be read -- a builtin, a C
    extension, a ``functools.partial``. Unreadable is not the same as proven
    non-returning, and the attestation check already refuses anything that has
    not positively declared itself.
    """
    if function is None:
        return None
    if getattr(function, "__name__", "") == "<lambda>":
        # A lambda body *is* its return value. Parsing the source line is
        # fragile when the lambda sits inside a larger expression, and the
        # answer is known without it.
        return True
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):  # pragma: no cover - defensive
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return None
    definition = next(
        (
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if definition is None:  # pragma: no cover - defensive
        return None
    return _body_produces_a_value(definition)


def _body_produces_a_value(definition: ast.AST) -> bool:
    """Walk one function body without descending into nested definitions.

    A ``return`` inside a closure belongs to the closure, not to the function
    the driver calls, so descending would report a value this body never
    produces.
    """
    stack: list[ast.AST] = list(ast.iter_child_nodes(definition))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.ClassDef):
            continue
        if isinstance(node, ast.Return) and node.value is not None:
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


# ---------------------------------------------------------------------------
# The three checks
# ---------------------------------------------------------------------------


def require_accepts_call(role: str, collaborator: Any) -> dict[str, Any]:
    """Prove the collaborator accepts the driver's call, without making it.

    ``Signature.bind`` resolves parameters against arguments and returns; it
    never invokes the callable, so this is a shape proof and nothing more.
    """
    if role not in CAPABILITY_CALL_CONTRACT:
        raise T1CapabilityError(f"{role!r} has no frozen call contract.")
    if not callable(collaborator):
        raise T1CapabilityError(
            f"Collaborator {role!r} is not callable, so stage "
            f"{CAPABILITY_STAGE_BINDINGS[role]!r} could not be entered."
        )
    positional_count, keywords = CAPABILITY_CALL_CONTRACT[role]
    try:
        signature = inspect.signature(collaborator)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {"role": role, "signature_checked": False}
    arguments = [_PLACEHOLDER] * positional_count
    named = {name: _PLACEHOLDER for name in keywords}
    try:
        signature.bind(*arguments, **named)
    except TypeError as error:
        raise T1CapabilityError(
            f"Collaborator {role!r} cannot accept the call the driver makes "
            f"({_describe_call(role)}): {error}. The driver first reaches it at "
            f"stage {CAPABILITY_STAGE_BINDINGS[role]!r}, which is after the "
            "claim, so this would have cost the attempt."
        ) from error
    return {
        "role": role,
        "signature_checked": True,
        "accepts": _describe_call(role),
    }


def _describe_call(role: str) -> str:
    positional_count, keywords = CAPABILITY_CALL_CONTRACT[role]
    parts = ["<positional>"] * positional_count + [f"{name}=..." for name in keywords]
    return f"{role}({', '.join(parts)})"


def require_completable(role: str, collaborator: Any) -> T1CapabilityAttestation:
    """Refuse a collaborator that cannot finish, for either reason.

    Two independent grounds, checked in this order because the cheaper one
    produces the clearer message: an implementation that declares itself
    non-executing is refused in its own words, and one that declares itself
    capable but has no body that can return is refused on the proof.
    """
    stage = CAPABILITY_STAGE_BINDINGS[role]
    attestation = read_attestation(role, collaborator)
    provider = _provider_name(collaborator)

    if attestation is None:
        raise T1CapabilityError(
            f"Collaborator {role!r} ({provider}) declares no execution "
            f"capability. A canonical run reaches it at stage {stage!r}, after "
            "the claim, so silence is refused rather than assumed: an "
            "implementation states that it can finish by exposing "
            f"{T1_CAPABILITY_ATTRIBUTE!r}, and one that has not said so is "
            "treated as one that cannot."
        )
    if attestation.role != role:
        raise T1CapabilityError(
            f"Collaborator {role!r} carries an attestation for "
            f"{attestation.role!r}. An attestation names the role it was "
            "written for so it cannot be moved to a different one."
        )
    if not attestation.executes:
        raise T1CapabilityError(
            f"Collaborator {role!r} ({attestation.provider}) is a capability "
            f"contract, not an implementation: {attestation.reason} A canonical "
            f"run would claim the attempt at stage {STAGE_CLAIM!r} and refuse "
            f"at stage {stage!r}, spending the one attempt on a path that "
            "cannot complete. Refused before the claim; nothing was consumed."
        )

    proof = _returns_a_value(_implementation_of(collaborator))
    if proof is False:
        raise T1CapabilityError(
            f"Collaborator {role!r} ({provider}) attests that it executes, but "
            "its body contains no reachable return and no yield, so it cannot "
            f"produce the result stage {stage!r} threads into the next stage. "
            "An attestation is a claim and a body that cannot return is a "
            "proof; the proof wins."
        )
    return attestation


def require_execution_graph_complete(
    attestations: Mapping[str, T1CapabilityAttestation],
) -> dict[str, Any]:
    """Prove every post-claim stage has an executable collaborator behind it.

    The per-role checks answer "is this one usable". This answers the question
    the invariant is actually about: is there any stage after the claim that
    would be entered without something able to finish it.
    """
    missing = sorted(set(REQUIRED_CAPABILITY_ROLES) - set(attestations))
    if missing:
        raise T1CapabilityError(
            f"The execution graph has no capability for {missing}. Every "
            "collaborator the driver threads must be verified before the claim."
        )
    refusing = sorted(
        role for role, attestation in attestations.items() if not attestation.executes
    )
    if refusing:  # pragma: no cover - require_completable refuses first
        raise T1CapabilityError(
            f"The execution graph cannot proceed past stage {STAGE_CLAIM!r}: "
            f"{refusing} declare no execution capability."
        )
    covered = sorted({CAPABILITY_STAGE_BINDINGS[role] for role in attestations})
    return {
        "execution_graph_complete": True,
        "claim_stage": STAGE_CLAIM,
        "claim_stage_index": _CLAIM_INDEX + 1,
        "post_claim_stages_covered": covered,
        "roles_verified": list(REQUIRED_CAPABILITY_ROLES),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def require_executable_capability(collaborators: Any) -> dict[str, Any]:
    """The whole gate: refuse any graph that could not survive its own claim.

    Called before the claim and before preflight. Reads no evidence, opens no
    label, resolves no path, creates nothing and consults no permission, so a
    refusal here leaves the single canonical attempt exactly as it was.
    """
    # 1. Bound. The driver owns this check; the gate does not duplicate it.
    collaborators.require_complete()

    attestations: dict[str, T1CapabilityAttestation] = {}
    shapes: list[dict[str, Any]] = []
    for role in REQUIRED_CAPABILITY_ROLES:
        bound = getattr(collaborators, role)
        # 2. Executable shape, then 3 and 4: attested and provably completable.
        shapes.append(require_accepts_call(role, bound))
        attestations[role] = require_completable(role, bound)

    # 5. The graph as a whole can proceed past the claim.
    graph = require_execution_graph_complete(attestations)
    return {
        "gate": GATE_NAME,
        **graph,
        "call_shapes": shapes,
        "attestations": [
            attestations[role].as_dict() for role in REQUIRED_CAPABILITY_ROLES
        ],
        "verified_before_claim": True,
        "labels_opened": False,
        "folds_run": False,
        "attempt_consumed": False,
        "run_directory_created": False,
    }


def capability_report(collaborators: Any) -> dict[str, Any]:
    """The same verdict as data, for a reviewer who is not executing.

    Never raises on an incomplete graph. `verify_collaborators` uses this so a
    report can say "bound but not executable" instead of choosing between an
    exception and a misleading success.
    """
    roles: dict[str, Any] = {}
    complete = True
    for role in REQUIRED_CAPABILITY_ROLES:
        bound = getattr(collaborators, role, None)
        try:
            require_accepts_call(role, bound)
            attestation = require_completable(role, bound)
        except T1CapabilityError as refusal:
            complete = False
            roles[role] = {
                "executes": False,
                "stage": CAPABILITY_STAGE_BINDINGS[role],
                "refusal": str(refusal),
            }
            continue
        roles[role] = attestation.as_dict()
    return {
        "gate": GATE_NAME,
        "execution_graph_complete": complete,
        "roles": roles,
        "claim_stage": STAGE_CLAIM,
        "attempt_consumed": False,
    }
