"""Runtime-integrity sentinel: the implementation of the frozen V1 design.

`docs/RUNTIME_INTEGRITY_SENTINEL_V1.md` specified this control after the
2026-08-12 shared-interpreter incident, in which unrelated distributions were
installed into the then-shared scientific interpreter while a canonical run was
executing. That run had a startup gate only, so the mutation was invisible to
it. This module implements exactly that design and introduces no second one.

**One digest recipe, not two.** The identity under check is
`installed_packages_sha256`, computed by the existing
`installed_package_snapshot()` and compared against `FROZEN_DEPENDENCY_DIGEST`,
the same values `require_p1_runtime()` already uses. A second recipe would
create a second provenance truth, so none is defined here.

**Enforcement points** (design §3):

1. `START` -- before any scientific input is opened.
2. `PRE_PROMOTION` -- immediately before each claim-bearing artifact is
   promoted to its canonical location.
3. `COMPLETION` -- after the last scientific computation, recorded whether or
   not it matches.

**Failure semantics** (design §3): a mismatch refuses the claim-bearing
promotion. It never deletes, resets, repairs, retries or re-seeds anything, and
it never re-runs a stage. The attempt is consumed and the observed digest, the
expected digest and the exact enforcement point are written to a
non-claim-bearing failure record so the difference can be diagnosed read-only.

This is governance/provenance infrastructure only. It changes no M2 gate
condition, no threshold, no numerical model behaviour, no prediction and no
prototype state, and it is never a model input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from cardiosentinel.neural.p1_experiment import FROZEN_DEPENDENCY_DIGEST
from cardiosentinel.neural.provenance import dependency_environment

SENTINEL_DESIGN_DOCUMENT: Final = "docs/RUNTIME_INTEGRITY_SENTINEL_V1.md"
SENTINEL_DESIGN_SHA256: Final = (
    "cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075"
)


class EnforcementPoint(str, Enum):
    """The three frozen checkpoints from design §3."""

    START = "start"
    PRE_PROMOTION = "pre_promotion"
    COMPLETION = "completion"


class RuntimeIntegrityError(RuntimeError):
    """Raised when the runtime identity differs at a required checkpoint."""


@dataclass(frozen=True, slots=True)
class RuntimeCheck:
    """One observation of the runtime identity at one enforcement point."""

    enforcement_point: str
    observed_digest: str
    expected_digest: str
    matches: bool
    package_count: int
    observed_at: str
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enforcement_point": self.enforcement_point,
            "observed_digest": self.observed_digest,
            "expected_digest": self.expected_digest,
            "matches": self.matches,
            "package_count": self.package_count,
            "observed_at": self.observed_at,
            "detail": self.detail,
        }


def observe_runtime_identity(
    point: EnforcementPoint,
    *,
    expected_digest: str = FROZEN_DEPENDENCY_DIGEST,
    detail: str | None = None,
) -> RuntimeCheck:
    """Take one reading. Never raises on mismatch -- the caller decides."""
    environment = dependency_environment()
    observed = str(environment["installed_packages_sha256"])
    return RuntimeCheck(
        enforcement_point=EnforcementPoint(point).value,
        observed_digest=observed,
        expected_digest=str(expected_digest),
        matches=observed == str(expected_digest),
        package_count=int(environment["installed_package_count"]),
        observed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        detail=detail,
    )


@dataclass(slots=True)
class RuntimeIntegrityRecord:
    """The `runtime_identity_checks` block carried by a canonical result.

    Absence of this block means a run predates the control. It must never be
    synthesised for a completed run.
    """

    expected_digest: str = FROZEN_DEPENDENCY_DIGEST
    checks: list[RuntimeCheck] = field(default_factory=list)

    def record(self, check: RuntimeCheck) -> RuntimeCheck:
        self.checks.append(check)
        return check

    @property
    def all_matched(self) -> bool:
        return bool(self.checks) and all(check.matches for check in self.checks)

    def first_mismatch(self) -> RuntimeCheck | None:
        for check in self.checks:
            if not check.matches:
                return check
        return None

    def digest_at(self, point: EnforcementPoint) -> str | None:
        value = EnforcementPoint(point).value
        for check in self.checks:
            if check.enforcement_point == value:
                return check.observed_digest
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "control": "runtime_integrity_sentinel_v1",
            "design_document": SENTINEL_DESIGN_DOCUMENT,
            "design_document_sha256": SENTINEL_DESIGN_SHA256,
            "digest_recipe": "installed_package_snapshot/installed_packages_sha256",
            "expected_digest": self.expected_digest,
            "checks": [check.as_dict() for check in self.checks],
            "check_count": len(self.checks),
            "start_digest": self.digest_at(EnforcementPoint.START),
            "completion_digest": self.digest_at(EnforcementPoint.COMPLETION),
            "all_observations_matched": self.all_matched,
            "automatic_environment_repair_performed": False,
            "automatic_retry_performed": False,
        }


def require_runtime_identity(
    point: EnforcementPoint,
    *,
    record: RuntimeIntegrityRecord | None = None,
    detail: str | None = None,
) -> RuntimeCheck:
    """Observe, record, and refuse to continue past a mismatch.

    A mismatch raises. It repairs nothing, retries nothing and re-runs nothing:
    the attempt is consumed and requires a new human authorization, exactly as
    a crash would.
    """
    expected = record.expected_digest if record else FROZEN_DEPENDENCY_DIGEST
    check = observe_runtime_identity(point, expected_digest=expected, detail=detail)
    if record is not None:
        record.record(check)
    if not check.matches:
        raise RuntimeIntegrityError(
            f"Runtime identity differs at enforcement point "
            f"{check.enforcement_point!r}: observed {check.observed_digest}, "
            f"expected {check.expected_digest} ({check.package_count} packages "
            "installed). The claim-bearing promotion is refused. The "
            "environment is NOT repaired, the stage is NOT re-run and the "
            "attempt is consumed; this requires documented human review."
        )
    return check


def runtime_failure_record(
    check: RuntimeCheck,
    *,
    git_sha: str,
    git_dirty: bool,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """A NON-CLAIM-BEARING failure record for a refused promotion."""
    return {
        "artifact_class": "runtime_integrity_failure",
        "claim_bearing": False,
        "scientific_evidence": False,
        "experiment_id": experiment_id,
        "enforcement_point": check.enforcement_point,
        "observed_digest": check.observed_digest,
        "expected_digest": check.expected_digest,
        "package_count": check.package_count,
        "observed_at": check.observed_at,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "promotion_refused": True,
        "automatic_environment_repair_performed": False,
        "automatic_retry_performed": False,
        "human_review_required": True,
    }
