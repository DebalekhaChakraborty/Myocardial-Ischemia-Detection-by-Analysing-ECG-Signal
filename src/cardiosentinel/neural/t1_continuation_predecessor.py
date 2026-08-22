"""Predecessor verification: the gate that makes a continuation unstartable alone.

A continuation is defined by what it continues. If the consumed attempt were
absent, or any bound digest had moved, then what remained would not be a
continuation at all -- it would be a fresh experiment wearing a continuation's
name, resting its claim on evidence nobody could check. So this module refuses
rather than degrades, and it refuses **completely**: there is no partial
recovery, no "verify what is present", no skip list.

That refusal is what makes the identity honest. A standalone continuation is
unstartable by construction.

**Read-only, always.** Nothing here writes, creates a directory, or opens a
label. It digests files that already exist and compares them with the bindings
frozen in amendment §1.3 and §1.4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cardiosentinel.neural.t1_continuation_spec import (
    CONSUMED_ATTEMPT_DIR,
    PREDECESSOR_CONTENT_DIGESTS,
    PREDECESSOR_FILE_DIGESTS,
    PREDECESSOR_FOLD_COUNT,
    PREDECESSOR_FOLD_SELECTIONS,
    PREDECESSOR_OOF_ARRAY_SHA256,
    PREDECESSOR_OOF_CONTENT_SHA256,
    fold_selection_relative_path,
)
from cardiosentinel.neural.t1_recovery_amendment import (
    validate_recovery_amendment_document,
)

READ_BLOCK: Final = 1 << 20


class T1ContinuationPredecessorError(RuntimeError):
    """Raised when the consumed attempt is absent, incomplete or altered."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PredecessorVerification:
    """What was verified, as data an artifact can carry.

    Carries digests rather than a bare boolean: a receipt that says "verified"
    without saying *what* it verified is a claim a future reader cannot check.
    """

    attempt_dir: Path
    amendment_sha256: str
    file_digests: dict[str, str]
    fold_selection_digests: dict[int, str]
    oof_array_sha256: str
    oof_content_sha256: str

    @property
    def verified(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "predecessor_attempt_dir": str(self.attempt_dir),
            "predecessor_digests_verified": True,
            "governing_amendment_sha256": self.amendment_sha256,
            "verified_file_digests": dict(sorted(self.file_digests.items())),
            "verified_fold_selection_digests": {
                str(index): digest
                for index, digest in sorted(self.fold_selection_digests.items())
            },
            "state_trace_array_sha256": self.oof_array_sha256,
            "state_trace_content_sha256": self.oof_content_sha256,
        }

    def consumed_evidence(self) -> tuple[dict[str, str], ...]:
        """The `consumed_evidence` provenance block, one entry per artifact."""
        entries = [
            {"artifact": name, "sha256": digest}
            for name, digest in sorted(self.file_digests.items())
        ]
        entries.extend(
            {
                "artifact": str(fold_selection_relative_path(index)),
                "sha256": digest,
            }
            for index, digest in sorted(self.fold_selection_digests.items())
        )
        return tuple(entries)


def verify_predecessor(
    attempt_dir: Path | None = None,
) -> PredecessorVerification:
    """Re-verify every §1.3 and §1.4 binding. Any mismatch is a refusal.

    The amendment is validated first. A continuation that checked its
    predecessor against digests read from a document that had itself moved
    would be proving consistency with an unknown, so the permission is
    established before the evidence it governs.
    """
    amendment_sha256 = validate_recovery_amendment_document()

    directory = Path(attempt_dir) if attempt_dir is not None else CONSUMED_ATTEMPT_DIR
    if not directory.is_dir():
        raise T1ContinuationPredecessorError(
            f"The consumed canonical attempt is absent at {directory}. A "
            "continuation continues something; with no predecessor there is "
            "nothing to continue, and what would run instead is a new "
            "experiment wearing a continuation's name. Refused."
        )

    mismatches: list[str] = []
    file_digests: dict[str, str] = {}
    for name, expected in sorted(PREDECESSOR_FILE_DIGESTS.items()):
        path = directory / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
            continue
        digest = _sha256_file(path)
        file_digests[name] = digest
        if digest != expected:
            mismatches.append(f"{name}: {digest} != {expected}")

    fold_digests: dict[int, str] = {}
    for index, (subject, policy_id, expected) in sorted(
        PREDECESSOR_FOLD_SELECTIONS.items()
    ):
        path = directory / fold_selection_relative_path(index)
        if not path.is_file():
            mismatches.append(f"fold {index:02d} selection: missing")
            continue
        digest = _sha256_file(path)
        fold_digests[index] = digest
        if digest != expected:
            mismatches.append(f"fold {index:02d} selection: {digest} != {expected}")
            continue
        # The digest proves the bytes; these prove the bytes mean what §1.4 says.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("held_out_subject") != subject:
            mismatches.append(
                f"fold {index:02d}: held-out subject "
                f"{payload.get('held_out_subject')!r} != {subject!r}"
            )
        if payload.get("selected_policy_id") != policy_id:
            mismatches.append(
                f"fold {index:02d}: selected policy "
                f"{payload.get('selected_policy_id')!r} != {policy_id!r}"
            )

    if len(fold_digests) != PREDECESSOR_FOLD_COUNT and not mismatches:
        mismatches.append(
            f"only {len(fold_digests)} of {PREDECESSOR_FOLD_COUNT} fold selections"
        )

    # The OOF manifest's own recorded digests must agree with the amendment.
    manifest_path = directory / "T1_OOF_STATE_EVIDENCE.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("array_sha256") != PREDECESSOR_OOF_ARRAY_SHA256:
            mismatches.append(
                f"OOF manifest array digest {manifest.get('array_sha256')} != "
                f"{PREDECESSOR_OOF_ARRAY_SHA256}"
            )
        if manifest.get("content_sha256") != PREDECESSOR_OOF_CONTENT_SHA256:
            mismatches.append(
                f"OOF manifest content digest {manifest.get('content_sha256')} != "
                f"{PREDECESSOR_OOF_CONTENT_SHA256}"
            )
        if (
            manifest.get("fold_selection_sha256")
            != PREDECESSOR_CONTENT_DIGESTS["oof_fold_selection_binding"]
        ):
            mismatches.append("OOF manifest fold-selection binding moved")
        for forbidden in ("contains_label", "contains_target_family"):
            if manifest.get(forbidden) is not False:
                mismatches.append(
                    f"OOF manifest {forbidden} is {manifest.get(forbidden)!r}, "
                    "not False -- the consumed trace must be label-free"
                )

    if mismatches:
        raise T1ContinuationPredecessorError(
            "The consumed canonical attempt does not re-verify against the "
            "recovery amendment. There is no partial recovery: a continuation "
            "resting on evidence that has moved would be a claim nobody can "
            "check.\n  " + "\n  ".join(mismatches)
        )

    return PredecessorVerification(
        attempt_dir=directory,
        amendment_sha256=amendment_sha256,
        file_digests=file_digests,
        fold_selection_digests=fold_digests,
        oof_array_sha256=PREDECESSOR_OOF_ARRAY_SHA256,
        oof_content_sha256=PREDECESSOR_OOF_CONTENT_SHA256,
    )
