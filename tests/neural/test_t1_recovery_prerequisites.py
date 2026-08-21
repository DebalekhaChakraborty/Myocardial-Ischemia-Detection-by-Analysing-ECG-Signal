"""The gates that must be green before a T1 continuation may be authorized.

The continuation is the single authorized remaining attempt. A post-claim
failure consumes it and no second continuation is authorized, so it has to be
entered with every prerequisite already proven rather than assumed.

Three groups, matching amendment §13:

* **Amendment** -- the frozen decision exists, digests as recorded, and a moved
  document is refused. A continuation running against an amendment that changed
  would be a run whose permission is an argument rather than a fact.
* **Failed attempt preservation** -- the consumed attempt is present, unchanged
  since the failure, and every promoted artifact still digests as the amendment
  recorded it. This is the evidence the continuation consumes; if it moved, the
  continuation is measuring something else.
* **Recovery boundary** -- each of those failures is a refusal.

Nothing here executes, authorizes or continues anything. There is no
continuation module yet, by design: this file exists so that when there is one,
the gates it must pass already exist and are already green.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from _attempt_guard import ATTEMPT_PRESENT, assert_attempt_unconsumed

from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural import t1_recovery_amendment as AMENDMENT

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONSUMED_ATTEMPT = PERSIST.canonical_run_directory(REPOSITORY_ROOT)
RECEIPT_PATH = REPOSITORY_ROOT / "recovery" / "T1_FAILURE_RECEIPT_RECONSTRUCTED.json"

# Every artifact the amendment binds in §1.3 and §1.4, by file digest. Restated
# here from the frozen document on purpose: if the code and the document ever
# disagree about what was preserved, that disagreement should fail a test rather
# than be resolved silently at continuation time.
PRESERVED_ARTIFACTS = {
    "T1_PREFLIGHT.json": (
        "917b5421c9c7731eb185821ed279564c65fed5737153316cfa410811ea4f25da"
    ),
    "T1_RUN_STATUS.json": (
        "f305da7ad3d465c4500124fe4d4422dfc471580a01afe7b9d424e866e9e2c59d"
    ),
    "T1_INPUT_LINEAGE.json": (
        "e307bdd3ad244f6440ad437f66d5f7b4e2af3072b6b1833e74552095ede3c555"
    ),
    "T1_INPUT_EVIDENCE.json": (
        "bf36ac0e538b0cee61a97109de413c52ec942356d974930e5de64bc32b86423b"
    ),
    "t1_input_evidence.npz": (
        "4391b4e7cda5ac5d70c93663563cc37954afdfc7b28092ef65c2d351006c2f5c"
    ),
    "T1_FOLD_SELECTIONS.json": (
        "71e0da62ad2a86fd6bb2561137e0a152df2d5b894bd9fecfb67ad762a5682f6d"
    ),
    "T1_OOF_STATE_EVIDENCE.json": (
        "aefc922a5224b7c857b9bf99b12441e55e46fdc71def373c043ffb112e5e2405"
    ),
    "t1_oof_state_evidence.npz": (
        "72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232"
    ),
}

ARTIFACTS_PRESENT = ATTEMPT_PRESENT
requires_consumed_attempt = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason=(
        "the consumed canonical attempt is gitignored and local-only; these "
        "gates are checked where the evidence lives"
    ),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Amendment
# ---------------------------------------------------------------------------


def test_the_amendment_digest_constant_exists():
    assert AMENDMENT.RECOVERY_AMENDMENT_VERSION == "V1.1"
    assert AMENDMENT.RECOVERY_AMENDMENT_SHA256 == (
        "d3ea7734c93be8f59796e03e8c0210778716327f7adc033cb2d3dcfff7f92c96"
    )
    assert AMENDMENT.RECOVERY_AMENDMENT_PATH.name == (
        "T1_EXECUTION_RECOVERY_AMENDMENT_V1_1.md"
    )


def test_the_amendment_document_validates():
    assert (
        AMENDMENT.validate_recovery_amendment_document()
        == AMENDMENT.RECOVERY_AMENDMENT_SHA256
    )
    assert AMENDMENT.RECOVERY_AMENDMENT_PATH.is_file()


def test_a_modified_amendment_is_refused(tmp_path):
    """A decision that can be edited afterwards is not a decision."""
    forged = tmp_path / "forged.md"
    forged.write_text(
        AMENDMENT.RECOVERY_AMENDMENT_PATH.read_text(encoding="utf-8") + "\nappended\n",
        encoding="utf-8",
    )
    with pytest.raises(AMENDMENT.T1RecoveryAmendmentError, match="immutable"):
        AMENDMENT.validate_recovery_amendment_document(forged)


def test_a_missing_amendment_is_refused(tmp_path):
    with pytest.raises(AMENDMENT.T1RecoveryAmendmentError, match="missing"):
        AMENDMENT.validate_recovery_amendment_document(tmp_path / "absent.md")


def test_the_amendment_names_the_clauses_it_amends():
    """Three clauses, and the code says which without opening the document."""
    assert AMENDMENT.AMENDED_CLAUSES == {
        "T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1": ("1", "17"),
        "T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1": ("14",),
    }


def test_the_continuation_identity_avoids_the_reserved_prefixes():
    """`t1-v1-development-continuation` would be refused, and correctly."""
    from cardiosentinel.neural import t1_execution as X

    for reserved in X.CANONICAL_RESERVED_PREFIXES:
        assert not AMENDMENT.CONTINUATION_ATTEMPT_ID.lower().startswith(
            str(reserved).lower()
        )
        assert (
            not str(AMENDMENT.CONTINUATION_RUN_ROOT_RELATIVE.name)
            .lower()
            .startswith(str(reserved).lower())
        )
    assert AMENDMENT.CONTINUATION_RUN_CLASS not in SPEC.T1_STAGE_ORDER


def test_binding_the_amendment_authorizes_nothing():
    """Provenance is not permission, and this module holds only provenance.

    Read from the syntax tree, never the text. This module's docstring
    necessarily says "post-claim failure" and "continuation" because that is
    what the amendment is about, and a substring scan reports every one of
    those sentences -- the false positive this repository has hit four times.
    """
    tree = ast.parse(Path(AMENDMENT.__file__).read_text(encoding="utf-8"))

    defined = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert defined == {"_sha256_file", "validate_recovery_amendment_document"}, (
        f"the provenance module defines {sorted(defined)}; it binds a digest "
        "and must not grow into a continuation"
    )

    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    for forbidden in (
        "mkdir",
        "makedirs",
        "write_text",
        "write_bytes",
        "write_json_atomic",
        "claim_canonical_run",
        "execute",
        "promote",
    ):
        assert forbidden not in called, (
            f"the provenance module calls {forbidden!r}; it may compute a "
            "digest and nothing else"
        )
    # The one file it opens is the document whose digest it verifies.
    assert sorted(name for name in called if name == "open") == ["open"]


# ---------------------------------------------------------------------------
# 2. Failed attempt preservation
# ---------------------------------------------------------------------------


@requires_consumed_attempt
def test_the_consumed_attempt_directory_exists():
    assert CONSUMED_ATTEMPT.is_dir()
    assert (CONSUMED_ATTEMPT / "T1_OOF_STATE_EVIDENCE.json").is_file()
    assert len(list((CONSUMED_ATTEMPT / "fold_selections").glob("*.json"))) == 12


@requires_consumed_attempt
@pytest.mark.parametrize("name,digest", sorted(PRESERVED_ARTIFACTS.items()))
def test_every_preserved_artifact_still_digests_as_the_amendment_recorded(name, digest):
    path = CONSUMED_ATTEMPT / name
    assert path.is_file(), f"{name} is missing from the consumed attempt"
    assert _digest(path) == digest, (
        f"{name} changed since the amendment bound it. The consumed attempt is "
        "immutable evidence; a continuation cannot consume something that moved."
    )


@requires_consumed_attempt
def test_no_artifact_was_modified_after_the_failure():
    """19:57:57.620Z is when the process died. Nothing may postdate it."""
    failure_epoch = 1787342277.7  # 2026-08-21T19:57:57.7Z, just after the last write
    late = [
        str(path.relative_to(CONSUMED_ATTEMPT))
        for path in CONSUMED_ATTEMPT.rglob("*")
        if path.is_file() and path.stat().st_mtime > failure_epoch
    ]
    assert late == [], f"artifacts were written after the failure: {late}"


@requires_consumed_attempt
def test_the_stage_24_artifacts_were_never_written():
    """The failure is visible as an absence, and that absence is the evidence."""
    for name in (
        PERSIST.OOF_RESULT_NAME,
        PERSIST.SUBJECT_EVIDENCE_NAME,
        PERSIST.BOOTSTRAP_NAME,
        PERSIST.CHALLENGE_EVIDENCE_NAME,
        PERSIST.FINAL_CONFIGURATION_NAME,
        PERSIST.RESULT_NAME,
        PERSIST.EXPERIMENT_LOCK_NAME,
    ):
        assert not (CONSUMED_ATTEMPT / name).exists(), (
            f"{name} exists; the run failed at stage 24 and reached none of these"
        )


def test_the_reconstructed_receipt_exists_and_says_it_is_reconstructed():
    assert RECEIPT_PATH.is_file()
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["receipt_type"] == "reconstructed"
    assert "does not claim to have been emitted" in receipt["statement"]
    assert receipt["stage"] == SPEC.STAGE_OOF_RESULT
    assert receipt["exception_type"] == "KeyError"
    assert (
        receipt["failure_classification"] == "incomplete execution requiring recovery"
    )
    assert receipt["run_directory_modified"] is False
    assert receipt["attempt_consumed"] is True
    assert receipt["test_accessed"] is False
    assert receipt["sealed_test_state"] == "unopened"


def test_the_reconstructed_receipt_lives_outside_the_consumed_attempt():
    """§25: no failed attempt is rewritten to look clean."""
    assert CONSUMED_ATTEMPT not in RECEIPT_PATH.parents
    assert not (CONSUMED_ATTEMPT / PERSIST.FAILURE_RECEIPT_NAME).exists(), (
        "a receipt was placed inside the consumed attempt; the run wrote none "
        "and one added afterwards would make the attempt look like it did"
    )


def test_the_receipt_carries_every_field_section_25_requires():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for field in SPEC.T1_FAILURE_RECEIPT_FIELDS:
        assert field in receipt, f"the reconstruction omits §25 field {field!r}"


def test_the_receipt_refuses_to_invent_what_was_lost():
    """The three lost quantities are named as absent, not filled in."""
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    absent = receipt["reconstruction"]["not_reconstructed"]
    assert "per-fold PRIMARY confusion counts" in absent
    assert "per-fold episode evidence" in absent
    assert "per-fold onset latencies" in absent
    assert receipt["held_out_evaluation_evidence_persisted"] is False
    flat = json.dumps(receipt).lower()
    for fabricated in ('"true_positive":', '"matched_episodes":', '"episode_f1"'):
        assert fabricated not in flat, (
            f"the reconstruction contains {fabricated}; the lost measurements "
            "are what the continuation exists to recover and must not be guessed"
        )


def test_the_receipt_binds_the_amendment_that_governs_the_recovery():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["reconstruction"]["governing_amendment_sha256"] == (
        AMENDMENT.RECOVERY_AMENDMENT_SHA256
    )
    assert receipt["authorized_git_sha"] == AMENDMENT.CONTINUED_AUTHORIZED_GIT_SHA
    assert receipt["attempt_id"] == AMENDMENT.CONTINUED_ATTEMPT_ID


# ---------------------------------------------------------------------------
# 3. Recovery boundary: each prerequisite failure is a refusal
# ---------------------------------------------------------------------------


def test_a_missing_amendment_blocks_the_boundary(tmp_path):
    with pytest.raises(AMENDMENT.T1RecoveryAmendmentError):
        AMENDMENT.validate_recovery_amendment_document(tmp_path / "gone.md")


def test_an_amendment_digest_mismatch_blocks_the_boundary(tmp_path):
    drifted = tmp_path / "drifted.md"
    drifted.write_text("a different decision entirely\n", encoding="utf-8")
    with pytest.raises(AMENDMENT.T1RecoveryAmendmentError, match="differs"):
        AMENDMENT.validate_recovery_amendment_document(drifted)


@requires_consumed_attempt
def test_changed_consumed_evidence_would_be_detected(tmp_path):
    """The digest comparison is real: a byte changes the answer."""
    original = (CONSUMED_ATTEMPT / "T1_OOF_STATE_EVIDENCE.json").read_bytes()
    copy = tmp_path / "T1_OOF_STATE_EVIDENCE.json"
    copy.write_bytes(original + b" ")
    assert _digest(copy) != PRESERVED_ARTIFACTS["T1_OOF_STATE_EVIDENCE.json"]


def test_the_canonical_attempt_cannot_be_claimed_again():
    """Where the attempt exists, a second claim is refused.

    On CI the run directory is gitignored and absent, so the guard's correct
    answer there is its census. Both are the mechanism working; asserting
    either world unconditionally is the mistake the attempt guard exists to
    prevent.
    """
    if ATTEMPT_PRESENT:
        with pytest.raises(PERSIST.T1PersistenceError, match="already claimed"):
            PERSIST.require_unclaimed_canonical_attempt(REPOSITORY_ROOT)
    else:
        assert (
            PERSIST.require_unclaimed_canonical_attempt(REPOSITORY_ROOT)[
                "existing_run_directory"
            ]
            is False
        )


# ---------------------------------------------------------------------------
# 4. This PR builds no continuation
# ---------------------------------------------------------------------------


def test_no_continuation_run_directory_exists():
    root = REPOSITORY_ROOT / AMENDMENT.CONTINUATION_RUN_ROOT_RELATIVE
    assert not root.exists(), (
        f"{root} exists; this change binds provenance and creates no attempt"
    )


def test_no_continuation_module_exists_yet():
    """The capability is the next PR, and its absence is deliberate here."""
    neural = Path(AMENDMENT.__file__).parent
    for forbidden in (
        "t1_continuation.py",
        "t1_continuation_run.py",
        "t1_measurement_continuation.py",
    ):
        assert not (neural / forbidden).exists(), (
            f"{forbidden} exists; continuation capability belongs to a separate "
            "reviewed change"
        )


def test_the_zero_counters_the_continuation_must_prove_are_recorded():
    """Named now so the next PR implements against a fixed list."""
    assert AMENDMENT.CONTINUATION_ZERO_COUNTERS == (
        "state_machine_invocations",
        "threshold_generation_calls",
        "policy_selection_calls",
        "fold_evaluations",
    )


def test_nothing_here_consumed_anything():
    assert_attempt_unconsumed()
    assert SPEC.T1_TEST_ACCESSED is False
    assert SPEC.T1_SEALED_TEST_STATE == "unopened"
    assert not (REPOSITORY_ROOT / "TEST_ATTEMPT.json").exists()
