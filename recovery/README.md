# Recovery records

Governance records **about** the consumed canonical T1 attempt, kept **outside**
it.

The canonical attempt `t1-v1-development` was claimed at 2026-08-21T19:47:24Z
and failed at 19:57:57Z in stage 24 of 29. Specification §25 is explicit that
**no failed attempt is deleted or rewritten to look clean**, and
`T1_FAILED_ATTEMPT_MAY_BE_DELETED_OR_REWRITTEN` is `False`. So
`cardiosentinel-runs/phase9-t1-development-v1/t1-v1-development/` is immutable:
nothing is added to it, corrected in it, or removed from it — including the
records that describe what happened to it.

That is what this directory is for.

## `T1_FAILURE_RECEIPT_RECONSTRUCTED.json`

The failure receipt §25 requires, reconstructed after the fact from surviving
evidence. **It does not claim to have been emitted by the failed execution** —
`receipt_type` is `"reconstructed"` and the file says so in its own
`statement` field, because a record that quietly reads like an artifact of the
run would be exactly the tidying §25 forbids.

The run produced no receipt of its own: `write_failure_receipt` and
`T1DevelopmentRun.failure_receipt` were both implemented, nothing called either,
and the driver had no exception handler, so the exception reached the
interpreter. PR #49 closed that path for every future run.

Reconstructed from the traceback, the filesystem timestamps and digests of all
twenty promoted artifacts, the run's own `T1_RUN_STATUS.json` and
`T1_PREFLIGHT.json`, and the frozen stage order.

**Three things are deliberately absent** — the per-fold PRIMARY confusion counts,
episode evidence and onset latencies. They were never persisted, cannot be
derived from the label-free artifacts that survive, and recovering them is the
entire purpose of the measurement continuation authorized by
`docs/T1_EXECUTION_RECOVERY_AMENDMENT_V1_1.md`. Writing plausible numbers here
would be fabricating evidence, so the receipt names them as not reconstructed
and says why.

## Precedent

This follows the handling of the 2026-08-12 outer-repository index, which was
**reconstructed, not recovered**, and recorded as such. A reconstruction that
announces itself stays useful; one that does not becomes indistinguishable from
the thing it describes.
