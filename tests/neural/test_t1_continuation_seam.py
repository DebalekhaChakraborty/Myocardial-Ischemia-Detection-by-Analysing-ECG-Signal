"""The claim-to-lock seam, executed end to end before it executes for real.

Every readiness report said the same thing: `execute_continuation`'s assembled
path had never run, could not be exercised without arming, and that residue was
"unprovable by construction". It was not. It was unprovable *the way we were
trying*, and the first real invocation proved it -- `_authorized_git_sha` called
`git_provenance()` without its required argument and raised `TypeError` six lines
before the claim. Pre-claim, so the attempt survived. Post-claim it would have
consumed the single authorized continuation.

The stages were each tested. The junctions between them were not, and a junction
reached for the first time at execution is exactly the defect class that consumed
the canonical attempt at stage 24.

So this drives the whole thing: authorization, identity, predecessor
verification, the negative capability gate, trace consumption, the Layer 3 match,
the claim, twelve folds of measurement and promotion, all six run-level
artifacts, the attestation and the experiment lock.

**Two things keep it honest.**

*The run root is sandboxed.* Every module that reads `CONTINUATION_RUN_ROOT` is
redirected into a temporary directory, and the test asserts afterwards that the
real one is still absent. Nothing here can claim the authorized identity.

*The labels are synthetic.* The trace is the real persisted one -- it is
label-free by construction, so consuming it costs nothing -- but the held-out
truth is invented. Opening real held-out labels outside the authorized
measurement would create exactly the ambiguity the amendment exists to prevent.
Label *values* do not determine whether the seam works; every junction it
crosses is identical either way.

It runs in a subprocess because the gate's clean-interpreter proof is binding at
`require_clean_interpreter=True`, and this suite has already imported the fold
evaluator to prove helper equivalence. A fresh process is the only place the real
gate can be exercised rather than patched away.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from _attempt_guard import ATTEMPT_PRESENT

from cardiosentinel.neural import t1_continuation_spec as S

REPO = Path(__file__).resolve().parents[2]

DRIVER = """
import json, sys
from pathlib import Path
sys.path.insert(0, {src!r})

from cardiosentinel.neural import t1_continuation_persistence as CP
from cardiosentinel.neural import t1_continuation_runner as RUN
from cardiosentinel.neural import t1_continuation_spec as S
from cardiosentinel.neural.t1_evidence_store import read_store

SANDBOX = Path({sandbox!r})

# 1. Sandbox the run root everywhere it is read. The authorized identity is
#    unreachable from this process after these three lines.
RUN.CONTINUATION_RUN_ROOT = SANDBOX
S.CONTINUATION_RUN_ROOT = SANDBOX
CP.CONTINUATION_RUN_ROOT = SANDBOX

# 2. Arm locally. The conftest interlock disarms test sessions; this subprocess
#    is not a test session, and the sandbox is what makes arming safe here.
S.T1_CONTINUATION_AUTHORIZED = True
RUN.require_continuation_authorized = S.require_continuation_authorized

# 3. A fixed commit id. Tree cleanliness is a separate property with its own
#    test; making the seam depend on it would make this fail during development
#    for a reason that has nothing to do with the seam.
RUN._authorized_git_sha = lambda: "0" * 40

# 4. Synthetic held-out truth over the real trace's own row ids.
COLUMNS = read_store(
    S.CONSUMED_ATTEMPT_DIR, "T1_OOF_STATE_EVIDENCE.json",
    columns=("stable_id", "fold_index"),
)

def _labels(authority, fold_index):
    mask = COLUMNS["fold_index"] == fold_index
    ids = [str(v) for v in COLUMNS["stable_id"][mask]]
    return {{
        "primary_mask": {{sid: True for sid in ids}},
        "primary_positive": {{sid: (i % 7 == 0) for i, sid in enumerate(ids)}},
    }}

RUN._target_source = lambda path: "sandboxed-source"
RUN.continuation_held_out_authority = (
    lambda fold_index, source, *, verified_selection_sha256: ("authority", fold_index)
)
RUN.held_out_labels_for_fold = _labels

result = RUN.execute_continuation(Path("unused-under-sandbox"))
attempt = Path(result["attempt_dir"])
print("SEAM_RESULT " + json.dumps({{
    "attempt_dir": str(attempt),
    "files": sorted(p.name for p in attempt.rglob("*") if p.is_file()),
    "record": result["record"],
    "attestation": result["attestation"],
    "lock": result["experiment_lock"],
}}, default=str))
"""


@pytest.mark.skipif(not ATTEMPT_PRESENT, reason="consumed attempt is local-only")
def test_the_claim_to_lock_seam_executes_end_to_end(tmp_path):
    sandbox = tmp_path / "phase9-t1-continuation-v1"
    script = tmp_path / "drive_seam.py"
    script.write_text(
        DRIVER.format(src=str(REPO / "src"), sandbox=str(sandbox)), encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=1800,
    )
    assert completed.returncode == 0, (
        f"the seam did not execute:\n{completed.stdout}\n{completed.stderr}"
    )
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith("SEAM_RESULT ")),
        None,
    )
    assert line, completed.stdout + completed.stderr
    result = json.loads(line[len("SEAM_RESULT ") :])

    # Every stage was entered and completed, in order.
    record = result["record"]
    assert record["stages_entered"] == list(record["stages_completed"])
    assert len(record["stages_entered"]) == record["stage_count"] == 11

    # All six run-level artifacts, plus the attestation and the lock.
    produced = set(result["files"])
    for name in (
        "T1_OOF_RESULT.json",
        "T1_SUBJECT_EVIDENCE.json",
        "T1_BOOTSTRAP.json",
        "T1_CHALLENGE_EVIDENCE.json",
        "T1_FINAL_CONFIGURATION.json",
        "T1_EXPERIMENT_LOCK.json",
        S.CONTINUATION_ATTESTATION_NAME,
    ):
        assert name in produced, f"{name} was not promoted"
    # Twelve per-fold evidence files.
    assert sum(1 for n in produced if n.startswith("T1_CONTINUATION_FOLD_")) == 12

    # The four counters, at completion.
    attestation = result["attestation"]
    for counter in S.CONTINUATION_ZERO_COUNTERS:
        assert attestation[counter] == 0, f"{counter} is {attestation[counter]}"
    assert attestation["state_transitions_regenerated"] is False
    assert attestation["selection_performed_here"] is False
    assert attestation["thresholds_generated_here"] is False
    assert attestation["predecessor_digests_verified"] is True
    assert attestation["test_accessed"] is False
    assert attestation["folds_measured"] == list(range(12))

    # Provenance closure. The lock carries the digest of every artifact
    # promoted *before* it -- five run-level artifacts plus the attestation --
    # and not its own, for the same reason the authorization commit could not
    # contain its own hash. A file cannot digest itself.
    lock = result["lock"]
    assert lock["promoted_artifact_count"] == 6
    assert set(lock["promoted_artifact_digests"]) == {
        "T1_OOF_RESULT.json",
        "T1_SUBJECT_EVIDENCE.json",
        "T1_BOOTSTRAP.json",
        "T1_CHALLENGE_EVIDENCE.json",
        "T1_FINAL_CONFIGURATION.json",
        S.CONTINUATION_ATTESTATION_NAME,
    }
    assert "T1_EXPERIMENT_LOCK.json" not in lock["promoted_artifact_digests"]
    assert lock["governing_amendment_sha256"] == S.RECOVERY_AMENDMENT_SHA256
    assert lock["attempts_authorized"] == 1
    assert lock["automatic_retry_permitted"] is False

    # No artifact claims a policy was run.
    assert "policy_runs" not in json.dumps(result)

    # And the authorized identity was never touched.
    assert not S.CONTINUATION_RUN_ROOT.exists(), "the sandbox leaked"


@pytest.mark.skipif(not ATTEMPT_PRESENT, reason="consumed attempt is local-only")
def test_the_seam_is_not_reusable_even_in_a_sandbox(tmp_path):
    """A second run into the same root is refused, not resumed."""
    sandbox = tmp_path / "phase9-t1-continuation-v1"
    (sandbox / S.CONTINUATION_ATTEMPT_ID).mkdir(parents=True)
    script = tmp_path / "drive_twice.py"
    script.write_text(
        DRIVER.format(src=str(REPO / "src"), sandbox=str(sandbox)), encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=1800,
    )
    assert completed.returncode != 0
    assert "already claimed" in completed.stderr


def test_authorized_git_sha_passes_the_repository_root():
    """The defect that refused the first real invocation, pre-claim.

    Asserted on the call site rather than by invoking it, so the test does not
    depend on the working tree being clean.
    """
    import ast

    from cardiosentinel.neural import t1_continuation_runner as RUN

    tree = ast.parse(Path(RUN.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "git_provenance"
    ]
    assert calls, "git_provenance is no longer called"
    for call in calls:
        assert call.args, "git_provenance() needs the repository root"
        assert call.args[0].id == "REPOSITORY_ROOT"


def test_the_real_continuation_root_is_still_absent():
    assert not S.CONTINUATION_RUN_ROOT.exists()
