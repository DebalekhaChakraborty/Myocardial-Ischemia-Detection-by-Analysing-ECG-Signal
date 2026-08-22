"""Acceptance gates for the T1 measurement continuation capability (PR #53).

These tests prove the continuation is *reachable under its own governance* and
*incapable of more than the amendment authorizes*. They do not run it. The
continuation is authorized once; a post-claim failure consumes it and no second
one is authorized, so every gate that can be green before it starts is green
before it starts.

The suite branches on `ATTEMPT_PRESENT` wherever behaviour differs between the
frozen scientific interpreter, where the consumed attempt is on disk, and CI,
where `cardiosentinel-runs/` is gitignored and absent. Assuming the attempt is
present and assuming it is absent are the same mistake facing opposite ways.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import pytest
from _attempt_guard import ATTEMPT_PRESENT

from cardiosentinel.neural import (
    t1_continuation_attestation as A,
)
from cardiosentinel.neural import (
    t1_continuation_gate as G,
)
from cardiosentinel.neural import (
    t1_continuation_measurement as M,
)
from cardiosentinel.neural import (
    t1_continuation_predecessor as P,
)
from cardiosentinel.neural import (
    t1_continuation_spec as S,
)

CONTINUATION_MODULES = (
    "cardiosentinel.neural.t1_continuation_spec",
    "cardiosentinel.neural.t1_continuation_predecessor",
    "cardiosentinel.neural.t1_continuation_gate",
    "cardiosentinel.neural.t1_continuation_measurement",
    "cardiosentinel.neural.t1_continuation_attestation",
)


# ---------------------------------------------------------------------------
# 1. Identity and permission
# ---------------------------------------------------------------------------


def test_continuation_identity_is_the_one_the_amendment_names():
    assert S.CONTINUATION_RUN_CLASS == "t1_continuation_measurement"
    assert S.CONTINUATION_ATTEMPT_ID == "t1-v1-measurement-continuation"
    assert str(S.CONTINUATION_RUN_ROOT_RELATIVE) == (
        "cardiosentinel-runs/phase9-t1-continuation-v1"
    )


@pytest.mark.parametrize(
    "rejected",
    [
        "t1-v1-development-continuation",
        "T1-V1-Development-Continuation",
        "t1-v1-development",
        "phase9-t1-development-v1-continuation",
    ],
)
def test_reserved_canonical_prefixes_are_refused(rejected):
    with pytest.raises(S.T1ContinuationIdentityError):
        S.require_continuation_identity(rejected, S.CONTINUATION_RUN_ROOT_RELATIVE)


def test_the_authorized_identity_is_accepted():
    assert (
        S.require_continuation_identity(
            S.CONTINUATION_ATTEMPT_ID, S.CONTINUATION_RUN_ROOT_RELATIVE
        )
        == S.CONTINUATION_ATTEMPT_ID
    )


def test_a_continuation_may_not_write_into_the_canonical_run_root():
    with pytest.raises(S.T1ContinuationIdentityError):
        S.require_continuation_identity(
            S.CONTINUATION_ATTEMPT_ID,
            Path("cardiosentinel-runs/phase9-t1-development-v1"),
        )


def test_continuation_authorization_is_its_own_decision():
    """It must not read the canonical development flag."""
    from cardiosentinel.neural import t1_config

    assert t1_config.T1_EXECUTION_SPECIFICATION_AUTHORIZED is True
    # Disarmed for the session, so the refusal mechanism is still exercisable.
    assert S.T1_CONTINUATION_AUTHORIZED is False
    with pytest.raises(S.T1ContinuationPermissionError):
        S.require_continuation_authorized()

    source = Path(S.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    bound = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "T1_EXECUTION_SPECIFICATION_AUTHORIZED" not in bound


def test_only_one_continuation_is_authorized():
    assert S.T1_CONTINUATION_ATTEMPTS_AUTHORIZED == 1
    assert S.T1_CONTINUATION_AUTOMATIC_RETRY_PERMITTED is False
    assert S.T1_CONTINUATION_MAY_BE_DELETED_OR_REWRITTEN is False


# ---------------------------------------------------------------------------
# 2. Amendment binding
# ---------------------------------------------------------------------------


def test_the_amendment_digest_is_the_frozen_one():
    from cardiosentinel.neural.t1_recovery_amendment import (
        RECOVERY_AMENDMENT_SHA256,
        validate_recovery_amendment_document,
    )

    expected = "d3ea7734c93be8f59796e03e8c0210778716327f7adc033cb2d3dcfff7f92c96"
    assert RECOVERY_AMENDMENT_SHA256 == expected
    assert validate_recovery_amendment_document() == expected


def test_a_modified_amendment_refuses_the_continuation(tmp_path):
    from cardiosentinel.neural.t1_recovery_amendment import (
        RECOVERY_AMENDMENT_PATH,
        T1RecoveryAmendmentError,
        validate_recovery_amendment_document,
    )

    tampered = tmp_path / "amendment.md"
    tampered.write_bytes(RECOVERY_AMENDMENT_PATH.read_bytes() + b"\n<!-- edit -->\n")
    with pytest.raises(T1RecoveryAmendmentError):
        validate_recovery_amendment_document(tampered)


# ---------------------------------------------------------------------------
# 3. Predecessor verification
# ---------------------------------------------------------------------------


def test_an_absent_predecessor_refuses(tmp_path):
    with pytest.raises(P.T1ContinuationPredecessorError, match="absent"):
        P.verify_predecessor(tmp_path / "nothing-here")


def test_a_partial_predecessor_refuses(tmp_path):
    """No partial recovery: a subset of the bindings is still a refusal."""
    attempt = tmp_path / "t1-v1-development"
    attempt.mkdir()
    (attempt / "T1_PREFLIGHT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(P.T1ContinuationPredecessorError):
        P.verify_predecessor(attempt)


def test_all_twenty_bindings_are_declared():
    assert len(S.PREDECESSOR_FILE_DIGESTS) == 8
    assert len(S.PREDECESSOR_FOLD_SELECTIONS) == S.PREDECESSOR_FOLD_COUNT == 12
    for name in (
        "T1_PREFLIGHT.json",
        "T1_INPUT_EVIDENCE.json",
        "T1_INPUT_LINEAGE.json",
        "t1_input_evidence.npz",
        "T1_FOLD_SELECTIONS.json",
        "T1_OOF_STATE_EVIDENCE.json",
        "t1_oof_state_evidence.npz",
    ):
        assert name in S.PREDECESSOR_FILE_DIGESTS
    for digest in S.PREDECESSOR_FILE_DIGESTS.values():
        assert len(digest) == 64 and int(digest, 16) >= 0


@pytest.mark.skipif(not ATTEMPT_PRESENT, reason="consumed attempt is local-only")
def test_the_real_predecessor_verifies():
    verification = P.verify_predecessor()
    assert verification.verified
    assert len(verification.file_digests) == 8
    assert len(verification.fold_selection_digests) == 12
    assert verification.oof_array_sha256 == S.PREDECESSOR_OOF_ARRAY_SHA256
    assert len(verification.consumed_evidence()) == 20


# ---------------------------------------------------------------------------
# 4. The negative capability gate
# ---------------------------------------------------------------------------


def test_the_gate_passes_on_the_continuation_modules():
    proof = G.prove_negative_capability(CONTINUATION_MODULES)
    assert proof["layer_1_structural"] and proof["layer_2_runtime"]
    assert proof["counters"] == {
        "state_machine_invocations": 0,
        "threshold_generation_calls": 0,
        "policy_selection_calls": 0,
        "fold_evaluations": 0,
    }


def test_no_continuation_module_imports_a_forbidden_module():
    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(Path(_source(module_name)).read_text(encoding="utf-8"))
        modules, _ = G._bound_names(tree)
        for forbidden in G.FORBIDDEN_MODULES:
            assert forbidden not in modules, f"{module_name} imports {forbidden}"


def test_only_the_permitted_protocol_names_are_bound():
    """§9.1 requires the frozen episode functions; next_state lives beside them."""
    bound = set()
    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(Path(_source(module_name)).read_text(encoding="utf-8"))
        _, names = G._bound_names(tree)
        bound |= names.get(G.PROTOCOL_MODULE, set())
    assert bound <= set(G.PERMITTED_PROTOCOL_NAMES)
    assert "next_state" not in bound
    assert {"group_reference_episodes", "match_runs_to_episodes"} <= bound


def test_the_gate_catches_a_forbidden_import(tmp_path, monkeypatch):
    offender = tmp_path / "offender.py"
    offender.write_text(
        "from cardiosentinel.neural.t1_protocol import next_state\n", encoding="utf-8"
    )
    monkeypatch.setattr(G, "_module_path", lambda name: offender)
    with pytest.raises(G.T1ContinuationCapabilityError, match="next_state"):
        G.prove_import_surface(["offender"])


def test_the_gate_catches_a_wholesale_protocol_import(tmp_path, monkeypatch):
    offender = tmp_path / "offender.py"
    offender.write_text("import cardiosentinel.neural.t1_protocol\n", encoding="utf-8")
    monkeypatch.setattr(G, "_module_path", lambda name: offender)
    with pytest.raises(G.T1ContinuationCapabilityError, match="wholesale"):
        G.prove_import_surface(["offender"])


def test_the_gate_catches_a_deferred_import(tmp_path, monkeypatch):
    """An import one indent deeper is still an import."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        "def run():\n"
        "    from cardiosentinel.neural.t1_fold_evaluator import "
        "T1CanonicalFoldEvaluator\n"
        "    return T1CanonicalFoldEvaluator\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(G, "_module_path", lambda name: offender)
    with pytest.raises(G.T1ContinuationCapabilityError):
        G.prove_import_surface(["offender"])


def test_the_gate_is_not_fooled_by_prose(tmp_path, monkeypatch):
    """The false positive this programme has produced five times.

    A refusal necessarily names what it refuses. A text scan flags this module;
    a syntax-tree proof does not.
    """
    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        '"""This module never calls next_state, generate_thresholds,\n'
        "select_policy or evaluate_held_out, and instantiates no\n"
        'T1CanonicalFoldEvaluator."""\n'
        "FORBIDDEN = ('next_state', 'select_policy', 'generate_thresholds')\n"
        "def refuse(name):\n"
        "    if name in FORBIDDEN:\n"
        "        raise RuntimeError(f'{name} is forbidden')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(G, "_module_path", lambda name: innocent)
    assert G.prove_import_surface(["innocent"])
    assert G.prove_no_forbidden_calls(["innocent"])


def test_counters_start_zero_and_refuse_when_incremented():
    counters = G.ContinuationCounters()
    assert counters.require_all_zero() == dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)
    counters.record("policy_selection_calls", "select_policy")
    with pytest.raises(G.T1ContinuationCapabilityError, match="policy_selection_calls"):
        counters.require_all_zero()
    assert counters.observed == ["policy_selection_calls:select_policy"]


def test_the_instrumentation_actually_fires_on_next_state():
    """The counter must be reachable, or Layer 2 proves nothing.

    A counter no production path can increment is the unreferenced-import
    argument wearing a counter's clothes, which §13.6 forbids substituting for
    real runtime evidence. This calls the real frozen entry point through the
    instrumentation and proves the counter moves.
    """
    from cardiosentinel.neural import t1_protocol

    counters = G.ContinuationCounters()
    with G.instrumented_protocol_entry_points(counters):
        with pytest.raises(G.T1ContinuationCapabilityError, match="next_state"):
            t1_protocol.next_state(None, None, None, None)
    assert counters.state_machine_invocations == 1
    assert counters.observed == ["state_machine_invocations:next_state"]
    with pytest.raises(G.T1ContinuationCapabilityError):
        counters.require_all_zero()


@pytest.mark.parametrize(
    "entry_point,counter",
    sorted(G.PROTOCOL_INSTRUMENTED_ENTRY_POINTS.items()),
)
def test_every_instrumented_entry_point_increments_its_counter(entry_point, counter):
    from cardiosentinel.neural import t1_protocol

    counters = G.ContinuationCounters()
    with G.instrumented_protocol_entry_points(counters):
        with pytest.raises(G.T1ContinuationCapabilityError):
            getattr(t1_protocol, entry_point)()
    assert getattr(counters, counter) == 1


def test_instrumentation_restores_the_frozen_protocol():
    """The frozen module must be exactly as it was, even after a tripwire fires."""
    from cardiosentinel.neural import t1_protocol

    before = {
        name: getattr(t1_protocol, name)
        for name in G.PROTOCOL_INSTRUMENTED_ENTRY_POINTS
    }
    counters = G.ContinuationCounters()
    with G.instrumented_protocol_entry_points(counters):
        assert getattr(t1_protocol, "next_state") is not before["next_state"]
        with pytest.raises(G.T1ContinuationCapabilityError):
            t1_protocol.next_state()
    for name, original in before.items():
        assert getattr(t1_protocol, name) is original, f"{name} was not restored"


def test_instrumentation_restores_even_when_the_body_raises():
    from cardiosentinel.neural import t1_protocol

    original = t1_protocol.next_state
    with pytest.raises(ValueError):
        with G.instrumented_protocol_entry_points(G.ContinuationCounters()):
            raise ValueError("body blew up")
    assert t1_protocol.next_state is original


def test_every_instrumented_name_maps_to_a_real_counter():
    assert set(G.PROTOCOL_INSTRUMENTED_ENTRY_POINTS.values()) <= set(
        S.CONTINUATION_ZERO_COUNTERS
    )


def test_instrumentation_refuses_if_the_protocol_moved(monkeypatch):
    """A gate bound to a protocol that has moved cannot prove what it claims."""
    from cardiosentinel.neural import t1_protocol

    monkeypatch.delattr(t1_protocol, "policy_sort_key")
    with pytest.raises(G.T1ContinuationCapabilityError, match="no entry point"):
        with G.instrumented_protocol_entry_points(G.ContinuationCounters()):
            pass  # pragma: no cover - the context manager refuses on entry


def test_clean_interpreter_proof_is_binding_when_asked():
    """Covers the entry points in modules the continuation never imports."""
    counters = G.ContinuationCounters()

    advisory = G.prove_no_forbidden_module_loaded(counters)
    assert advisory["binding"] is False
    assert "forbidden_modules_loaded" in advisory

    # This suite imports the fold evaluator to prove helper equivalence, so the
    # interpreter is deliberately dirty here and the binding form must refuse.
    import cardiosentinel.neural.t1_fold_evaluator  # noqa: F401

    with pytest.raises(G.T1ContinuationCapabilityError, match="loaded"):
        G.prove_no_forbidden_module_loaded(counters, binding=True)

    with pytest.raises(G.T1ContinuationCapabilityError):
        G.prove_negative_capability(
            CONTINUATION_MODULES, require_clean_interpreter=True
        )


def test_the_gate_reports_both_halves_of_layer_two():
    proof = G.prove_negative_capability(CONTINUATION_MODULES)
    instrumented = proof["instrumented_entry_points"]
    # Keyed by module: the protocol is loaded because §9.1 requires it, and
    # t1_development_run because the §16 label authority drags it in.
    assert instrumented[G.PROTOCOL_MODULE] == dict(G.PROTOCOL_INSTRUMENTED_ENTRY_POINTS)
    assert instrumented[G.DEVELOPMENT_RUN_MODULE] == dict(
        G.DEVELOPMENT_RUN_INSTRUMENTED_ENTRY_POINTS
    )
    assert "interpreter" in proof
    assert proof["interpreter"]["binding"] is False


def test_every_instrumented_counter_is_an_amendment_counter():
    for points in G.INSTRUMENTED_ENTRY_POINTS.values():
        assert set(points.values()) <= set(S.CONTINUATION_ZERO_COUNTERS)


def test_the_never_loaded_set_is_the_forbidden_set_minus_the_instrumented_ones():
    """Every forbidden module is covered by exactly one runtime proof."""
    instrumented = set(G.INSTRUMENTED_ENTRY_POINTS) - {G.PROTOCOL_MODULE}
    assert set(G.NEVER_LOADED_MODULES) | instrumented == set(G.FORBIDDEN_MODULES)
    assert not (set(G.NEVER_LOADED_MODULES) & instrumented), "double-covered module"


def test_counter_names_are_the_amendment_vocabulary():
    """The defect that consumed the canonical attempt was a key-name mismatch."""
    from cardiosentinel.neural.t1_recovery_amendment import CONTINUATION_ZERO_COUNTERS

    assert S.CONTINUATION_ZERO_COUNTERS == CONTINUATION_ZERO_COUNTERS
    assert set(G.ContinuationCounters().as_dict()) == set(CONTINUATION_ZERO_COUNTERS)


def test_every_forbidden_group_maps_to_a_real_counter():
    assert set(G.FORBIDDEN_IMPORTS) == set(S.CONTINUATION_ZERO_COUNTERS)


# ---------------------------------------------------------------------------
# 5. Re-implemented helpers are proven equivalent, not assumed
# ---------------------------------------------------------------------------


def test_contiguous_runs_matches_the_canonical_implementation():
    from cardiosentinel.neural.t1_development_run import contiguous_runs as canonical

    rng = random.Random(2026)
    for _ in range(400):
        flags = [rng.random() < 0.3 for _ in range(rng.randint(0, 40))]
        assert M.contiguous_runs(flags) == canonical(flags)
    for edge in ([], [True], [False], [True] * 5, [False] * 5):
        assert M.contiguous_runs(edge) == canonical(edge)


def test_onset_latency_matches_the_canonical_implementation():
    from cardiosentinel.neural.t1_fold_evaluator import _onset_latency as canonical
    from cardiosentinel.neural.t1_protocol import (
        group_reference_episodes,
        match_runs_to_episodes,
    )

    rng = random.Random(4242)
    for _ in range(300):
        n = rng.randint(1, 40)
        starts = [i * 125 for i in range(n)]
        positive = [rng.random() < 0.35 for _ in range(n)]
        states = [rng.random() < 0.35 for _ in range(n)]
        episodes = group_reference_episodes(starts, positive)
        runs = M.contiguous_runs(states)
        matched = match_runs_to_episodes(episodes, runs)
        assert M.onset_latency_seconds(episodes, runs, matched, starts) == canonical(
            episodes, runs, matched, starts
        )


def test_the_sampling_rate_matches_the_canonical_divisor():
    assert M.SAMPLES_PER_SECOND == 250.0


# ---------------------------------------------------------------------------
# 6. The attestation artifact
# ---------------------------------------------------------------------------


def _verification(tmp_path):
    return P.PredecessorVerification(
        attempt_dir=tmp_path / "t1-v1-development",
        amendment_sha256=S.RECOVERY_AMENDMENT_SHA256,
        file_digests=dict(S.PREDECESSOR_FILE_DIGESTS),
        fold_selection_digests={
            index: digest
            for index, (_s, _p, digest) in S.PREDECESSOR_FOLD_SELECTIONS.items()
        },
        oof_array_sha256=S.PREDECESSOR_OOF_ARRAY_SHA256,
        oof_content_sha256=S.PREDECESSOR_OOF_CONTENT_SHA256,
    )


def test_the_attestation_carries_every_required_field(tmp_path):
    attestation = A.build_continuation_attestation(
        G.ContinuationCounters(),
        _verification(tmp_path),
        gate_proof={
            "gate": G.GATE_NAME,
            "modules_proven": list(CONTINUATION_MODULES),
            "layer_1_structural": True,
            "layer_2_runtime": True,
        },
        folds_measured=range(12),
    )
    for field in S.CONTINUATION_ATTESTATION_REQUIRED_FIELDS:
        assert field in attestation
    assert attestation["artifact_class"] == ("t1_v1_continuation_execution_attestation")
    assert attestation["state_trace_source"] == "predecessor_oof_state_evidence"
    assert attestation["state_trace_content_sha256"] == (
        "cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f"
    )
    assert attestation["state_trace_array_sha256"] == (
        "72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232"
    )
    assert attestation["selection_performed_here"] is False
    assert attestation["thresholds_generated_here"] is False
    assert attestation["state_transitions_regenerated"] is False
    assert attestation["predecessor_digests_verified"] is True
    assert attestation["test_accessed"] is False
    assert attestation["sealed_test_state"] == "unopened"
    assert attestation["folds_measured"] == list(range(12))


def test_the_attestation_refuses_a_nonzero_counter(tmp_path):
    counters = G.ContinuationCounters()
    counters.record("fold_evaluations", "evaluate_held_out")
    with pytest.raises(G.T1ContinuationCapabilityError):
        A.build_continuation_attestation(
            counters, _verification(tmp_path), gate_proof={}, folds_measured=[0]
        )


def test_the_attestation_refuses_a_missing_field():
    with pytest.raises(A.T1ContinuationAttestationError, match="missing"):
        A.validate_continuation_attestation({"artifact_class": "x"})


def test_the_attestation_refuses_a_policy_runs_counter(tmp_path):
    """§13.6 Layer 3 requires its absence, not a zero."""
    attestation = A.build_continuation_attestation(
        G.ContinuationCounters(),
        _verification(tmp_path),
        gate_proof={},
        folds_measured=[0],
    )
    attestation["policy_runs"] = 0
    with pytest.raises(A.T1ContinuationAttestationError, match="policy_runs"):
        A.validate_continuation_attestation(attestation)


def test_the_attestation_refuses_a_falsified_flag(tmp_path):
    attestation = A.build_continuation_attestation(
        G.ContinuationCounters(),
        _verification(tmp_path),
        gate_proof={},
        folds_measured=[0],
    )
    attestation["selection_performed_here"] = True
    with pytest.raises(A.T1ContinuationAttestationError):
        A.validate_continuation_attestation(attestation)


def test_provenance_names_what_was_continued_and_consumed(tmp_path):
    provenance = A.continuation_provenance(_verification(tmp_path))
    assert provenance["continues"]["predecessor_run"] == "t1-v1-development"
    assert provenance["continues"]["predecessor_digest"] == (
        "cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f"
    )
    consumed = provenance["consumed_evidence"]
    assert len(consumed) == 20
    assert all({"artifact", "sha256"} == set(entry) for entry in consumed)


# ---------------------------------------------------------------------------
# 7. Nothing is executed, claimed or created
# ---------------------------------------------------------------------------


def test_no_continuation_run_directory_exists():
    assert not S.CONTINUATION_RUN_ROOT.exists()


def test_no_continuation_module_creates_a_directory_or_writes():
    """Structural: none of them can promote anything."""
    mutating = {
        "mkdir",
        "makedirs",
        "write_text",
        "write_bytes",
        "rmtree",
        "unlink",
        "rename",
        "replace",
        "write_json_atomic",
        "savez",
        "savez_compressed",
    }
    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(Path(_source(module_name)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", None)
                )
                assert name not in mutating, f"{module_name} calls {name}"


def test_no_continuation_module_has_an_entrypoint():
    for module_name in CONTINUATION_MODULES:
        source = Path(_source(module_name)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.If):
                names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
                assert "__name__" not in names, f"{module_name} has a __main__"


def test_the_consumed_attempt_is_never_written_to():
    """No continuation module names a write into the canonical run root."""
    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(Path(_source(module_name)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "phase9-t1-development-v1/t1-v1-development/" not in node.value


def test_test_is_never_reachable():
    for module_name in CONTINUATION_MODULES:
        tree = ast.parse(Path(_source(module_name)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value != "test"
                assert not node.value.startswith("TEST_")


def test_continuation_identity_block_attests_test_unopened():
    identity = S.continuation_identity()
    assert identity["test_accessed"] is False
    assert identity["sealed_test_state"] == "unopened"
    assert identity["is_continuation_artifact"] is True
    assert identity["automatic_retry_permitted"] is False


# ---------------------------------------------------------------------------
# 8. Layer 3 against the real trace
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ATTEMPT_PRESENT, reason="consumed attempt is local-only")
def test_the_real_trace_matches_every_promoted_selection():
    attempt = S.CONSUMED_ATTEMPT_DIR
    trace = M.consume_oof_state_trace(attempt)
    assert trace.array_sha256 == S.PREDECESSOR_OOF_ARRAY_SHA256
    assert trace.content_sha256 == S.PREDECESSOR_OOF_CONTENT_SHA256
    assert trace.row_count == 492904

    verified = M.require_trace_matches_selections(trace, attempt)
    assert set(verified) == set(range(12))
    for fold_index, record in verified.items():
        subject, policy_id, _ = S.PREDECESSOR_FOLD_SELECTIONS[fold_index]
        assert record["held_out_subject"] == subject
        assert record["selected_policy_id"] == policy_id
        for column in M.THRESHOLD_COLUMNS:
            assert isinstance(record[column], float)


@pytest.mark.skipif(not ATTEMPT_PRESENT, reason="consumed attempt is local-only")
def test_consuming_the_trace_invokes_no_state_machine():
    counters = G.ContinuationCounters()
    trace = M.consume_oof_state_trace(S.CONSUMED_ATTEMPT_DIR, counters)
    assert counters.require_all_zero() == dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)
    assert "emitted_state" in trace.columns


# ---------------------------------------------------------------------------
# 9. The measurement itself, on synthetic evidence
#
# The three lost quantities are what the continuation exists to recover, so the
# code that recovers them is tested rather than trusted -- an untested
# measurement path is precisely what consumed the canonical attempt. No real
# held-out label is opened here: the labels are constructed, and so is the
# trace, which keeps this a test of arithmetic rather than an execution of the
# continuation.
# ---------------------------------------------------------------------------


def _synthetic_trace(states, positives, masks, *, fold_index=0, cadence=None):
    """One stream, one subject, hand-built so the answer is known by inspection.

    The row cadence is the frozen protocol stride, taken from the protocol
    rather than retyped: rows one stride apart are episode-contiguous, and a
    fixture that guessed the stride would silently make every positive row
    its own episode and test nothing.
    """
    import numpy as np

    from cardiosentinel.neural.t1_protocol import T1_EPISODE_CADENCE_SAMPLES

    cadence = T1_EPISODE_CADENCE_SAMPLES if cadence is None else cadence

    subject, policy_id, _ = S.PREDECESSOR_FOLD_SELECTIONS[fold_index]
    n = len(states)
    stable_ids = [f"row{i:03d}" for i in range(n)]
    columns = {
        "stable_id": np.asarray(stable_ids),
        "record_id": np.asarray(["rec0"] * n),
        "channel_index": np.zeros(n, dtype=int),
        "start_sample": np.asarray([i * cadence for i in range(n)]),
        "subject_id": np.asarray([subject] * n),
        "fold_index": np.full(n, fold_index, dtype=int),
        "selected_policy_id": np.asarray([policy_id] * n),
        "p_watch": np.full(n, 0.9),
        "s_watch": np.full(n, 0.8),
        "p_event": np.full(n, 0.99),
        "s_event": np.full(n, 0.95),
        "emitted_state": np.asarray(states),
        "state_elapsed_seconds": np.zeros(n),
        "transition_from": np.asarray(["NORMAL"] * n),
        "transition_to": np.asarray(["NORMAL"] * n),
        "transition_occurred": np.zeros(n, dtype=bool),
    }
    trace = M.ConsumedTrace(
        columns=columns,
        array_sha256=S.PREDECESSOR_OOF_ARRAY_SHA256,
        content_sha256=S.PREDECESSOR_OOF_CONTENT_SHA256,
        row_count=n,
    )
    labels = {
        "primary_mask": {sid: m for sid, m in zip(stable_ids, masks)},
        "primary_positive": {sid: p for sid, p in zip(stable_ids, positives)},
    }
    verified = {
        fold_index: {
            "fold_index": fold_index,
            "held_out_subject": subject,
            "selected_policy_id": policy_id,
            "p_watch": 0.9,
            "s_watch": 0.8,
            "p_event": 0.99,
            "s_event": 0.95,
        }
    }
    return trace, labels, verified


def test_measurement_recovers_confusion_episodes_and_latency():
    #      idx:   0        1        2        3        4        5
    states = ["NORMAL", "EVENT", "EVENT", "NORMAL", "EVENT", "NORMAL"]
    positives = [False, True, True, False, False, True]
    masks = [True] * 6
    trace, labels, verified = _synthetic_trace(states, positives, masks)

    result = M.measure_fold(trace, 0, labels, verified, G.ContinuationCounters())

    # Rows 1,2 predicted EVENT and positive -> 2 tp. Row 4 predicted EVENT,
    # negative -> 1 fp. Row 5 positive, not predicted -> 1 fn. Rows 0,3 -> 2 tn.
    assert result.primary_confusion == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}

    # Reference episodes: rows 1-2 contiguous, and row 5. Predicted runs:
    # rows 1-2, and row 4. The first episode matches the first run; the second
    # episode (row 5) overlaps no run.
    assert result.episode_evidence == {
        "reference_episodes": 2,
        "predicted_event_runs": 2,
        "matched_episodes": 1,
        "unmatched_predicted_runs": 1,
    }
    # The matched run begins at the same sample as its episode.
    assert result.onset_latency_seconds == [0.0]
    assert result.selected_policy_id == "qw0.9_qe0.99_FAST"
    assert result.stream_count == 1


def test_measurement_honours_the_primary_mask():
    """Rows outside the PRIMARY mask score nothing, but still form episodes."""
    states = ["EVENT", "EVENT", "NORMAL"]
    positives = [True, True, False]
    trace, labels, verified = _synthetic_trace(states, positives, [False, True, True])
    result = M.measure_fold(trace, 0, labels, verified, G.ContinuationCounters())
    assert sum(result.primary_confusion.values()) == 2
    assert result.primary_confusion == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}


def test_measurement_reports_latency_only_for_detected_episodes():
    """An undetected episode has no latency; zero would read as instant detection."""
    states = ["NORMAL", "NORMAL", "NORMAL"]
    trace, labels, verified = _synthetic_trace(states, [True, True, False], [True] * 3)
    result = M.measure_fold(trace, 0, labels, verified, G.ContinuationCounters())
    assert result.onset_latency_seconds == []
    assert result.episode_evidence["reference_episodes"] == 1
    assert result.episode_evidence["matched_episodes"] == 0


def test_measurement_latency_is_measured_in_sample_coordinates():
    states = ["NORMAL", "NORMAL", "EVENT"]
    trace, labels, verified = _synthetic_trace(states, [True, True, True], [True] * 3)
    result = M.measure_fold(trace, 0, labels, verified, G.ContinuationCounters())
    # One stride is 1250 samples at 250 Hz = 5 s. The episode begins at row 0
    # and the run at row 2, so the onset latency is two strides: 10 s.
    assert result.onset_latency_seconds == [10.0]


def test_measurement_leaves_every_counter_at_zero():
    states = ["EVENT", "NORMAL"]
    trace, labels, verified = _synthetic_trace(states, [True, False], [True, True])
    counters = G.ContinuationCounters()
    M.measure_fold(trace, 0, labels, verified, counters)
    assert counters.require_all_zero() == dict.fromkeys(S.CONTINUATION_ZERO_COUNTERS, 0)


def test_measurement_output_carries_no_policy_runs_counter():
    states = ["EVENT", "NORMAL"]
    trace, labels, verified = _synthetic_trace(states, [True, False], [True, True])
    payload = M.measure_fold(
        trace, 0, labels, verified, G.ContinuationCounters()
    ).as_dict()
    assert "policy_runs" not in json.dumps(payload)
    assert set(payload["primary_confusion"]) == {"tp", "fp", "tn", "fn"}


def test_a_gap_in_sample_cadence_breaks_an_episode():
    """An episode is physically contiguous; a gap is never bridged."""
    import numpy as np

    states = ["EVENT", "EVENT", "EVENT"]
    trace, labels, verified = _synthetic_trace(states, [True, True, True], [True] * 3)
    trace.columns["start_sample"] = np.asarray([0, 1250, 100_000])
    result = M.measure_fold(trace, 0, labels, verified, G.ContinuationCounters())
    assert result.episode_evidence["reference_episodes"] == 2


def _source(module_name: str) -> str:
    import importlib

    return importlib.import_module(module_name).__file__
