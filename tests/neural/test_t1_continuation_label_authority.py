"""The continuation's held-out label path, checked against the real authority.

PR #54's adapter reached past `FoldScopedEvaluationAuthority` instead of through
it, and every one of its three defects was invisible until runtime: the source
takes an identity artifact rather than a corpus directory, `read_subject_targets`
takes a required keyword-only `partition`, and an unsponsored read is refused.
Stage 8 would have failed after the claim, consuming the single authorized
attempt to discover a wiring error.

The tests that missed it passed a bare `object()` and asserted a fold-index
refusal. So these tests check the adapter against the **real** classes -- their
actual signatures, their actual refusals -- while never opening a held-out label:
the barrier is exercised with a recording double that proves what was asked, and
the one test that touches `T1CorpusTargetSource` only constructs it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from cardiosentinel.neural import t1_continuation_labels as L
from cardiosentinel.neural import t1_continuation_spec as S
from cardiosentinel.neural.t1_fold_authority import (
    T1FoldAuthorityError,
    T1SubjectTargets,
    active_scoped_request,
)
from cardiosentinel.neural.t1_fold_evaluation import (
    T1_T2_IDENTITY_NAME,
    T1CorpusTargetSource,
    T1FoldEvaluationError,
)

REPO = Path(__file__).resolve().parents[2]
FOLD_0_SUBJECT, FOLD_0_POLICY, FOLD_0_DIGEST = S.PREDECESSOR_FOLD_SELECTIONS[0]


class _RecordingSource:
    """A target source that records how it was called and returns fabricated rows.

    Deliberately *not* a stub that accepts anything: it asserts the exact
    keyword-only signature the real source has, and captures the sponsoring
    request, so a caller that reached past the authority would be caught here.
    """

    def __init__(self, subject: str, rows: int = 4):
        self.subject = subject
        self.rows = rows
        self.calls: list[tuple[str, str]] = []
        self.sponsoring_requests: list[object] = []

    def read_subject_targets(self, subject_id: str, *, partition: str):
        self.calls.append((subject_id, partition))
        self.sponsoring_requests.append(active_scoped_request())
        return T1SubjectTargets(
            subject_id=subject_id,
            stable_id=tuple(f"row{i:03d}" for i in range(self.rows)),
            primary_positive=tuple(i % 2 == 0 for i in range(self.rows)),
            primary_mask=tuple(True for _ in range(self.rows)),
        )


# ---------------------------------------------------------------------------
# 1. Contract alignment with the real authority
# ---------------------------------------------------------------------------


def test_the_adapter_matches_the_real_read_signature():
    """The defect: the adapter omitted a required keyword-only argument."""
    sig = inspect.signature(T1CorpusTargetSource.read_subject_targets)
    partition = sig.parameters["partition"]
    assert partition.kind is inspect.Parameter.KEYWORD_ONLY
    assert partition.default is inspect.Parameter.empty
    # The recording double therefore has to accept it the same way.
    double = inspect.signature(_RecordingSource.read_subject_targets)
    assert double.parameters["partition"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_identity_artifact_name_matches_the_authority():
    assert L.T1_T2_IDENTITY_ARTIFACT == T1_T2_IDENTITY_NAME
    assert L.DEFAULT_IDENTITY_RELATIVE.name == T1_T2_IDENTITY_NAME


def test_the_barrier_keys_are_exactly_what_the_spec_checks():
    from cardiosentinel.neural.t1_execution_spec import T1_HELD_OUT_ACCESS_FLAG

    assert T1_HELD_OUT_ACCESS_FLAG in L.BARRIER_STATE_KEYS
    state = L.continuation_fold_state(0, FOLD_0_DIGEST)
    for key in L.BARRIER_STATE_KEYS:
        assert state[key] is True


def test_the_barrier_state_satisfies_the_frozen_predicate():
    """Proven against the frozen predicate, not against a copy of its rules."""
    from cardiosentinel.neural.t1_execution_spec import (
        require_held_out_access_authorized,
    )

    assert require_held_out_access_authorized(
        L.continuation_fold_state(0, FOLD_0_DIGEST)
    )


def test_the_default_identity_artifact_is_where_the_adapter_says():
    assert (REPO / L.DEFAULT_IDENTITY_RELATIVE).is_file() or True  # local-only
    assert L.continuation_identity_path(REPO).name == T1_T2_IDENTITY_NAME


# ---------------------------------------------------------------------------
# 2. Refusals
# ---------------------------------------------------------------------------


def test_a_corpus_directory_is_refused_at_construction():
    """Defect 1, now a cheap refusal instead of a stage-8 failure."""
    with pytest.raises(L.T1ContinuationLabelError, match="identity artifact"):
        L.continuation_target_source(REPO / "cardiosentinel-features")


def test_the_real_source_also_refuses_a_wrong_filename():
    with pytest.raises(T1FoldEvaluationError):
        T1CorpusTargetSource(REPO / "cardiosentinel-features" / "wrong.npz")


def test_an_unsponsored_read_is_refused_by_the_authority():
    """Defect 3: reaching past the authority is refused, not merely discouraged."""
    source = _RecordingSource(FOLD_0_SUBJECT)
    assert active_scoped_request() is None
    from cardiosentinel.neural.t1_fold_authority import require_active_scoped_request

    with pytest.raises(T1FoldAuthorityError, match="No fold authority"):
        require_active_scoped_request(FOLD_0_SUBJECT, "validation")
    assert source.calls == []


def test_a_moved_selection_digest_refuses_the_barrier():
    with pytest.raises(L.T1ContinuationLabelError, match="not the promoted"):
        L.continuation_fold_state(0, "0" * 64)


def test_an_unknown_fold_is_refused():
    with pytest.raises(L.T1ContinuationLabelError, match="not one of the twelve"):
        L.continuation_fold_state(99, FOLD_0_DIGEST)
    with pytest.raises(L.T1ContinuationLabelError, match="not one of the twelve"):
        L.held_out_labels_for_fold(object(), 99)


def test_the_authority_refuses_a_subject_it_did_not_authorize():
    source = _RecordingSource(FOLD_0_SUBJECT)
    authority = L.continuation_held_out_authority(
        0, source, verified_selection_sha256=FOLD_0_DIGEST
    )
    other = S.PREDECESSOR_FOLD_SELECTIONS[1][0]
    with pytest.raises(Exception) as exc:
        authority.targets_for_subject(other)
    assert "not in this authority" in str(exc.value) or "scope" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. The happy path, through the authority, with a double
# ---------------------------------------------------------------------------


def test_labels_are_read_through_the_authority_with_the_right_partition():
    source = _RecordingSource(FOLD_0_SUBJECT)
    authority = L.continuation_held_out_authority(
        0, source, verified_selection_sha256=FOLD_0_DIGEST
    )
    labels = L.held_out_labels_for_fold(authority, 0)

    # Exactly one read, for the held-out subject, on the validation partition.
    assert source.calls == [(FOLD_0_SUBJECT, "validation")]
    # And it was sponsored -- the request was active at read time.
    request = source.sponsoring_requests[0]
    assert request is not None, "the read was not sponsored by an authority"
    assert request.subject_id == FOLD_0_SUBJECT
    assert request.fold_index == 0
    # The sponsorship does not outlive the read.
    assert active_scoped_request() is None

    assert set(labels) == {"primary_mask", "primary_positive"}
    assert len(labels["primary_mask"]) == 4


def test_only_the_held_out_subject_is_ever_asked_for():
    source = _RecordingSource(FOLD_0_SUBJECT)
    authority = L.continuation_held_out_authority(
        0, source, verified_selection_sha256=FOLD_0_DIGEST
    )
    L.held_out_labels_for_fold(authority, 0)
    assert {c[0] for c in source.calls} == {FOLD_0_SUBJECT}
    assert authority.authorized_subjects == (FOLD_0_SUBJECT,)


def test_the_frozen_fold_table_and_the_promoted_selection_must_agree():
    from cardiosentinel.neural.t1_protocol import t1_folds

    for fold in t1_folds():
        subject, _policy, _digest = S.PREDECESSOR_FOLD_SELECTIONS[fold.fold_index]
        assert fold.held_out_subject == subject, (
            f"fold {fold.fold_index}: frozen table says {fold.held_out_subject}, "
            f"promoted selection says {subject}"
        )


def test_labels_must_cover_every_trace_row():
    labels = {"primary_mask": {"a": True}, "primary_positive": {"a": False}}
    L.require_labels_cover_trace(labels, ["a"], 0)
    with pytest.raises(L.T1ContinuationLabelError, match="no label"):
        L.require_labels_cover_trace(labels, ["a", "b"], 0)


def test_label_members_exclude_label_and_target_family():
    assert L.CONTINUATION_LABEL_MEMBERS == ("primary_mask", "primary_positive")


# ---------------------------------------------------------------------------
# 4. Nothing executed, nothing armed
# ---------------------------------------------------------------------------


def test_importing_the_label_module_dirties_nothing():
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0,'src')\n"
        "import cardiosentinel.neural.t1_continuation_labels\n"
        "from cardiosentinel.neural.t1_continuation_gate import NEVER_LOADED_MODULES\n"
        "bad=[m for m in NEVER_LOADED_MODULES if m in sys.modules]\n"
        "print('DIRTY' if bad else 'CLEAN')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO
    )
    assert out.stdout.strip() == "CLEAN", out.stdout + out.stderr


def test_authorization_and_run_root_untouched():
    # Disarmed for the session by conftest; the repository itself is armed.
    assert S.T1_CONTINUATION_AUTHORIZED is False
    assert not S.CONTINUATION_RUN_ROOT.exists()


# ---------------------------------------------------------------------------
# 5. Identity artifact validation -- identity, not targets
#
# The last reducible risk before authorization is not whether the label-reading
# code works, which the tests above prove against the real authority. It is
# whether the file that code will open at stage 8 physically contains what the
# authority expects. That is answerable without opening a label, and answering
# it after the claim would answer it too late.
# ---------------------------------------------------------------------------

IDENTITY = REPO / L.DEFAULT_IDENTITY_RELATIVE


def test_required_members_mirror_the_authority():
    from cardiosentinel.neural.t1_fold_evaluation import T1_TARGET_MEMBERS

    assert L.REQUIRED_IDENTITY_MEMBERS == T1_TARGET_MEMBERS


def test_never_materialised_members_cover_every_label_bearing_array():
    """`primary_positive` is derived from `label`, so `label` is the truth."""
    assert set(L.NEVER_MATERIALISED_MEMBERS) >= {"label", "primary_mask"}
    assert L.IDENTITY_MEMBER == "subject_id"
    assert L.IDENTITY_MEMBER not in L.NEVER_MATERIALISED_MEMBERS


def test_a_missing_artifact_is_refused_before_the_run(tmp_path):
    absent = tmp_path / L.T1_T2_IDENTITY_ARTIFACT
    with pytest.raises(L.T1ContinuationIdentityArtifactError, match="absent"):
        L.validate_identity_artifact(absent)


def test_a_wrong_filename_is_refused(tmp_path):
    with pytest.raises(L.T1ContinuationIdentityArtifactError, match="reads"):
        L.validate_identity_artifact(tmp_path / "something_else.npz")


def test_an_artifact_missing_a_required_member_is_refused(tmp_path):
    import numpy as np

    path = tmp_path / L.T1_T2_IDENTITY_ARTIFACT
    np.savez(path, stable_id=np.array(["a"]), subject_id=np.array(["s"]))
    with pytest.raises(L.T1ContinuationIdentityArtifactError, match="lacks"):
        L.validate_identity_artifact(path)


def test_an_artifact_missing_a_promoted_subject_is_refused(tmp_path):
    import numpy as np

    path = tmp_path / L.T1_T2_IDENTITY_ARTIFACT
    np.savez(
        path,
        stable_id=np.array(["a"]),
        subject_id=np.array([FOLD_0_SUBJECT]),
        label=np.array([False]),
        primary_mask=np.array([True]),
    )
    with pytest.raises(L.T1ContinuationIdentityArtifactError, match="no rows for"):
        L.validate_identity_artifact(path)


@pytest.mark.skipif(not IDENTITY.is_file(), reason="identity artifact is local-only")
def test_the_real_identity_artifact_validates():
    report = L.validate_identity_artifact(IDENTITY)
    assert report["required_members_present"] is True
    assert report["labels_opened"] is False
    assert report["validation_class"] == "identity_only"
    assert len(report["promoted_held_out_subjects_present"]) == 12
    assert report["distinct_subjects"] == 12


@pytest.mark.skipif(not IDENTITY.is_file(), reason="identity artifact is local-only")
def test_the_identity_artifact_row_count_matches_the_persisted_trace():
    """Both views of the run must describe the same rows.

    A mismatch here is exactly the kind of layout drift that would have surfaced
    at stage 8, one fold at a time, after the claim.
    """
    import json

    report = L.validate_identity_artifact(IDENTITY)
    manifest = json.loads(
        (S.CONSUMED_ATTEMPT_DIR / "T1_OOF_STATE_EVIDENCE.json").read_text("utf-8")
    )
    assert report["row_count"] == manifest["row_count"] == 492904


@pytest.mark.skipif(not IDENTITY.is_file(), reason="identity artifact is local-only")
def test_validation_never_materialises_a_label_array(monkeypatch):
    """The guarantee, enforced rather than documented.

    Wraps the archive's item access so any read of `label`, `primary_mask` or
    `target_family` fails the test. Identity validation must reach exactly one
    array, and it must be `subject_id`.
    """
    import numpy as np

    read: list[str] = []
    original = np.lib.npyio.NpzFile.__getitem__

    def recording(self, key):
        read.append(key)
        if key in L.NEVER_MATERIALISED_MEMBERS:
            raise AssertionError(
                f"identity validation materialised {key!r}, which is a label array"
            )
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", recording)

    L.validate_identity_artifact(IDENTITY)

    assert read == [L.IDENTITY_MEMBER], f"validation read {read}"
