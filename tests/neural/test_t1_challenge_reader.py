"""Tests for the narrow post-trace challenge-family reader.

Challenge membership lives in `target_family` on the T2 outer row identity --
the same archive the label-blind assembly reads, and a member it is forbidden
to open there. This reader is a second, narrower door beside the label-blind
one, never a widening of it, and it opens only after a state trace exists.

Nothing here authorizes execution, claims the canonical attempt, creates the
canonical run directory or reaches TEST.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t1_assembly as A
from cardiosentinel.neural import t1_challenge as C
from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_evidence_store as STORE
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural import t2_protocol as T2
from cardiosentinel.neural.t1_protocol import T1_STATES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _identity(tmp_path: Path, **overrides) -> Path:
    """A synthetic row identity carrying every member the real one carries."""
    members = {
        "stable_id": np.asarray([f"s:{i}" for i in range(8)]),
        "target_family": np.asarray(
            [
                "background_negative",
                "rate_related_confounder",
                "axis_shift_confounder",
                "conduction_change_confounder",
                "ischemic_positive",
                "quality_excluded",
                "rate_related_confounder",
                "boundary_ambiguous",
            ]
        ),
        "label": np.zeros(8, dtype=bool),
        "primary_mask": np.ones(8, dtype=bool),
        "subject_id": np.asarray(["ltstdb:s2004"] * 8),
    }
    members.update(overrides)
    path = tmp_path / "t2_outer_row_identity.npz"
    np.savez(path, **members)
    return path


def _traced_columns(count: int = 8) -> dict[str, np.ndarray]:
    return {
        "stable_id": np.asarray([f"s:{i}" for i in range(count)]),
        "emitted_state": np.asarray(["NORMAL"] * count),
    }


# ---------------------------------------------------------------------------
# 1. It reads membership, and only membership
# ---------------------------------------------------------------------------


def test_the_reader_materialises_exactly_two_members():
    assert C.CHALLENGE_READER_MEMBERS == ("stable_id", "target_family")
    for refused in ("label", "primary_mask"):
        assert refused in C.CHALLENGE_READER_REFUSED_MEMBERS
        assert refused not in C.CHALLENGE_READER_MEMBERS


def test_the_reader_never_binds_a_label_or_a_mask():
    """Structural: no refused member is named as something the reader reads."""
    source = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "read_challenge_family_membership"
    )
    body = ast.unparse(function)
    for refused in C.CHALLENGE_READER_REFUSED_MEMBERS:
        assert f"payload['{refused}']" not in body
        assert f'payload["{refused}"]' not in body
    assert "payload['stable_id']" in body
    assert "payload['target_family']" in body


def test_membership_maps_the_three_frozen_categories(tmp_path):
    membership = C.read_challenge_family_membership(_identity(tmp_path))
    assert set(membership) == set(C.CHALLENGE_FAMILIES)
    assert membership["RATE"] == ("s:1", "s:6")
    assert membership["AXIS"] == ("s:2",)
    assert membership["CONDUCTION"] == ("s:3",)


def test_non_challenge_categories_belong_to_no_family(tmp_path):
    membership = C.read_challenge_family_membership(_identity(tmp_path))
    everything = {value for values in membership.values() for value in values}
    for outside in ("s:0", "s:4", "s:5", "s:7"):
        assert outside not in everything


def test_the_family_map_tracks_the_frozen_t2_categories():
    """A category renamed upstream must not become a silently empty family."""
    assert set(C.CHALLENGE_FAMILY_BY_TARGET_FAMILY) == set(T2.T2_CHALLENGE_CATEGORIES)
    assert set(C.CHALLENGE_FAMILY_BY_TARGET_FAMILY.values()) == set(
        C.CHALLENGE_FAMILIES
    )


def test_a_missing_member_fails_closed(tmp_path):
    path = tmp_path / "t2_outer_row_identity.npz"
    np.savez(path, stable_id=np.asarray(["s:0"]))
    with pytest.raises(C.T1ChallengeError, match="lacks"):
        C.read_challenge_family_membership(path)


# ---------------------------------------------------------------------------
# 2. It cannot run before the state trace
# ---------------------------------------------------------------------------


def test_membership_joins_only_onto_a_completed_trace(tmp_path):
    rows = C.derive_challenge_rows(
        oof_columns=_traced_columns(), identity_path=_identity(tmp_path)
    )
    assert rows["RATE"] == (1, 6)
    assert rows["AXIS"] == (2,)
    assert rows["CONDUCTION"] == (3,)


def test_a_missing_trace_column_fails_closed(tmp_path):
    with pytest.raises(C.T1ChallengeError):
        C.derive_challenge_rows(
            oof_columns={"stable_id": np.asarray(["s:0"])},
            identity_path=_identity(tmp_path),
        )


def test_an_empty_trace_fails_closed(tmp_path):
    with pytest.raises(C.T1ChallengeError, match="nothing to join"):
        C.derive_challenge_rows(
            oof_columns={
                "stable_id": np.asarray([], dtype=object),
                "emitted_state": np.asarray([], dtype=object),
            },
            identity_path=_identity(tmp_path),
        )


def test_a_trace_that_is_not_a_state_trace_fails_closed(tmp_path):
    columns = _traced_columns()
    columns["emitted_state"] = np.asarray(["PENDING"] * 8)
    with pytest.raises(C.T1ChallengeError, match="non-states"):
        C.derive_challenge_rows(oof_columns=columns, identity_path=_identity(tmp_path))
    assert "PENDING" not in T1_STATES


# ---------------------------------------------------------------------------
# 3. It cannot influence what came before it
# ---------------------------------------------------------------------------


def test_challenge_identity_is_not_an_input_to_any_decision():
    assert SPEC.T1_CHALLENGE_JOIN_AFTER_STATE_TRACE is True
    assert SPEC.T1_CHALLENGE_IS_TRANSITION_INPUT is False
    assert SPEC.T1_CHALLENGE_IS_THRESHOLD_GENERATION_INPUT is False
    assert SPEC.T1_CHALLENGE_IS_SELECTION_INPUT is False


def test_the_evaluator_cannot_reach_the_challenge_reader():
    """Selection and the transition run in a module that cannot see this."""
    from cardiosentinel.neural import t1_fold_evaluator as V

    tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "cardiosentinel.neural.t1_challenge" not in imported

    # An import-surface and read-surface check, not a substring scan. The
    # evaluator legitimately names `target_family` in the refusal that keeps it
    # out of the timeline, so scanning for the word would fail on the very
    # guard that makes the guarantee.
    code = ast.unparse(tree)
    assert "read_challenge_family_membership" not in code
    assert "derive_challenge_rows" not in code
    for read in ("payload['target_family']", "columns['target_family']"):
        assert read not in code
    refusals = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value == "target_family"
    ]
    assert refusals, "the evaluator should still name target_family as refused"


def test_the_label_blind_reader_is_untouched():
    """The stage 12 door still refuses everything it refused before."""
    assert STORE.forbidden_members() == (
        SPEC.T1_T2_IDENTITY_MEMBERS_FORBIDDEN_LABEL_BLIND
    )
    assert "target_family" in STORE.forbidden_members()
    assert "label" in STORE.forbidden_members()
    assert "primary_mask" in STORE.forbidden_members()


def test_the_frozen_sources_are_byte_identical():
    import hashlib

    frozen = {
        "t1_protocol.py": (
            "b0df6ea2ade450037e94e5ab3b193694fea980337851a2458b3f43873450b192"
        ),
        "t1_execution_spec.py": (
            "edb0cbf1afe43dee48b5d2d0ed190e0939530fc026fd2f09d3312b929ab1fbe3"
        ),
        "t1_evidence_store.py": (
            "464ca1607191aa02042a6dcbb8cfeda4d4f3aced1eae2e29ae4b77be8cf6d39c"
        ),
    }
    for name, digest in frozen.items():
        path = REPOSITORY_ROOT / "src" / "cardiosentinel" / "neural" / name
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == digest, f"{name} changed; this PR must not touch it"


# ---------------------------------------------------------------------------
# 4. Deterministic, and reachable from the assembler
# ---------------------------------------------------------------------------


def test_the_join_is_deterministic(tmp_path):
    identity = _identity(tmp_path)
    first = C.derive_challenge_rows(
        oof_columns=_traced_columns(), identity_path=identity
    )
    second = C.derive_challenge_rows(
        oof_columns=_traced_columns(), identity_path=identity
    )
    assert first == second
    for values in first.values():
        assert list(values) == sorted(values)


def test_the_challenge_assembler_derives_rather_than_receiving(tmp_path):
    collaborator = A.assemble_challenge(t2_identity=_identity(tmp_path))
    artifact = collaborator(oof_columns=_traced_columns())
    assert artifact["families"]["RATE"]["row_count"] == 2
    assert artifact["families"]["AXIS"]["row_count"] == 1
    assert artifact["families"]["CONDUCTION"]["row_count"] == 1


def test_the_assembler_still_satisfies_the_capability_gate(tmp_path):
    from cardiosentinel.neural import t1_capability_gate as G

    collaborator = A.assemble_challenge(t2_identity=_identity(tmp_path))
    attestation = G.require_completable("assemble_challenge", collaborator)
    assert attestation.executes is True


# ---------------------------------------------------------------------------
# 5. Nothing was authorized, claimed, created or opened
# ---------------------------------------------------------------------------


def test_no_execution_path_became_reachable():
    """This PR does not wire main(); the entrypoint still stops at preflight."""
    from cardiosentinel.neural import t1_development_run as R

    source = Path(R.__file__).read_text(encoding="utf-8")
    main_body = source[source.index("def main(") :]
    assert "T1CanonicalDevelopmentExecutor" not in main_body
    assert ".execute(" not in main_body


def test_authorization_remains_false():
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is False


def test_the_canonical_attempt_is_untouched(tmp_path):
    assert not _canonical_root().exists()
    C.derive_challenge_rows(
        oof_columns=_traced_columns(), identity_path=_identity(tmp_path)
    )
    assert not _canonical_root().exists()


def test_the_reader_refuses_a_test_partition_archive(tmp_path):
    path = tmp_path / "test.npz"
    np.savez(path, stable_id=np.asarray(["s:0"]), target_family=np.asarray(["x"]))
    with pytest.raises(SPEC.T1ExecutionSpecError, match="TEST is sealed"):
        C.read_challenge_family_membership(path)
