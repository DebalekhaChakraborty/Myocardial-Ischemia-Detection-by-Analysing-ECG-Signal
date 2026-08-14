"""Attempt #1's frozen preserved artifacts, reproduced for tests.

Not collected by pytest (no `test_` prefix). Shared so both the lineage tests
and the canonical-runner integration tests can stand a verified original
attempt up under a temporary root, without either reading or touching the real
preserved attempt.

The payloads live in `data/m2_attempt1_frozen.json` rather than inline, and the
writer is `json.dumps(payload, indent=2, sort_keys=True) + "\\n"`, so they
round-trip byte-identically and reproduce the frozen digests. A test asserts
exactly that, so drift here cannot silently weaken the negative cases.
"""

from __future__ import annotations

import json
from pathlib import Path

from cardiosentinel.neural import m2_development_run as R
from cardiosentinel.neural import m2_persistence as PS

ORIGINAL = R.ORIGINAL_SUITE_ID

_FROZEN = json.loads(
    (Path(__file__).parent / "data" / "m2_attempt1_frozen.json").read_text()
)

FROZEN_STATUS = _FROZEN["status"]
FROZEN_RECEIPT = _FROZEN["receipt"]

RECOVERY1 = R.RECOVERY1_SUITE_ID
FROZEN_RECOVERY1_STATUS = _FROZEN["recovery1_status"]
FROZEN_RECOVERY1_RECEIPT = _FROZEN["recovery1_receipt"]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _plant_frozen_attempt1(
    run_root, *, status=None, receipt=None, arms=("M2-0", "M2-G")
):
    """Reproduce attempt #1's preserved artifacts under a temporary root."""
    for arm in arms:
        payload = (status or {}).get(arm, FROZEN_STATUS[arm])
        _write(
            run_root / PS.arm_experiment_id(ORIGINAL, arm) / PS.RUN_STATUS_NAME, payload
        )
    if receipt is not False:
        _write(
            PS.failure_review_directory(run_root, ORIGINAL)
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME,
            receipt or FROZEN_RECEIPT,
        )
    return run_root


def _plant_frozen_recovery1(
    run_root, *, status=None, receipt=None, arms=("M2-0", "M2-G")
):
    """Reproduce recovery1's preserved artifacts under a temporary root."""
    for arm in arms:
        payload = (status or {}).get(arm, FROZEN_RECOVERY1_STATUS[arm])
        _write(
            run_root / PS.arm_experiment_id(RECOVERY1, arm) / PS.RUN_STATUS_NAME,
            payload,
        )
    if receipt is not False:
        _write(
            PS.failure_review_directory(run_root, RECOVERY1)
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME,
            receipt or FROZEN_RECOVERY1_RECEIPT,
        )
    return run_root


def _plant_both_prior_attempts(run_root):
    """Both consumed attempts, exactly as recovery2 requires them."""
    _plant_frozen_attempt1(run_root)
    _plant_frozen_recovery1(run_root)
    return run_root


def _resigned_recovery1(**overrides):
    """A recovery1 receipt mutated then re-signed."""
    from cardiosentinel.neural.integrity import canonical_sha256

    payload = {**FROZEN_RECOVERY1_RECEIPT, **overrides}
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _resigned(**overrides):
    """A receipt mutated then re-signed, so only the FROZEN digest can refuse it."""
    from cardiosentinel.neural.integrity import canonical_sha256

    payload = {**FROZEN_RECEIPT, **overrides}
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload
