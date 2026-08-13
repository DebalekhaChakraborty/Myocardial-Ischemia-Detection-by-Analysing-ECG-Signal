# Runtime Integrity Sentinel V1 — prospective execution-integrity control

> **STATUS: DESIGN ONLY — NOT IMPLEMENTED, NOT AUTHORIZED FOR EXECUTION.**
> This document specifies an engineering/provenance control for future canonical
> runs. It changes no scientific choice, and it is not applied retroactively to
> any completed run.

## 0. What this is not

This control does **not** modify, and must never modify:

- the M2 gate G1–G6 or their order;
- the frozen G3 SQI columns, the Q99/linear rule, or the frozen bounds;
- `NORMAL_EVIDENCE_THRESHOLD` or the G4 q50/linear derivation rule;
- `M2_CLASSIFICATION_EVALUATION_THRESHOLD`;
- refractory semantics, duration or re-arming;
- the retained M1L architecture or any M1 artifact;
- populations, metrics, aggregation, the exit rule, or test policy.

It is a provenance check on the *execution environment*, not on the science.

## 1. Why this exists

On **2026-08-12** the CardioSentinel scientific interpreter was shared with
unrelated application work. During the canonical M1-v2 Stage-1 run:

| Event | UTC |
|---|---|
| Run launched; `require_p1_runtime()` evaluated **GREEN**; environment snapshotted (335 packages, `b0fd6eaa…`) | 18:01:39 |
| `jmespath`, `botocore`, `s3transfer`, `boto3` installed into the same interpreter by a concurrent unrelated agent session | 18:08:28–29 |
| `awscrt` installed by the same source | 19:10:07 |
| Run completed, exit 0 | 19:49:15 |

**M1-v2 had a startup gate only. No mid-run or end-of-run environment check
existed, and this document does not claim otherwise.** The run's locks record the
startup snapshot because that is when the snapshot was taken. The mutation was
discovered afterwards, during preparation of the M2 derivation receipt.

Recorded execution-integrity limitation, permanently: **whether the terminated
M1-v2 process loaded any of the five added distributions after they appeared
cannot be proven retrospectively.** All available read-only evidence is
consistent with no effect — the five are imported nowhere in `src/` or `tests/`,
none is a scientific dependency, no bytecode for them was written between install
and run completion, and no already-imported module changes when a distribution is
added to `site-packages` — but that is evidence, not proof. Human interpretation
recorded on 2026-08-12: **M1-v2 remains the canonical frozen M1 development
evidence and is not to be rerun or modified.**

## 2. Environment separation (already performed)

The scientific environment now has a dedicated interpreter that no application
project uses:

| | |
|---|---|
| Scientific interpreter | `/home/AI_POC/venvs/tactics/bin/python` |
| Application interpreter | `/home/AI_POC/venvs/debalekha/bin/python` |
| Frozen scientific identity | `b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a` (335 packages) |

The frozen identity is a function of installed distribution **names and
versions** only (`installed_package_snapshot()`), and `runtime_environment()`
records no interpreter path. Relocating the environment therefore changes no
recorded identity: the digest under the new path is bitwise equal to the digest
recorded in the M1-v2 locks, and `require_p1_runtime()` returns GREEN.

Future canonical commands must invoke the **tactics** interpreter. Historical
command strings in frozen documents record what was actually executed at the
time and are not rewritten.

## 3. The control

**Identity under check.** `installed_packages_sha256`, computed by the existing
`installed_package_snapshot()` and compared against `FROZEN_DEPENDENCY_DIGEST`.
No new digest recipe is introduced; a second recipe would create a second
provenance truth.

**Enforcement points.**

1. **Startup** — before any scientific input is opened. Refuse to start on
   mismatch. This is what exists today.
2. **Before every claim-bearing promotion** — re-evaluated immediately before
   each of: an arm claim directory being created; a checkpoint being written; an
   experiment lock being written; a canonical results document being written; a
   staging store being promoted to canonical.
3. **Completion** — after the last scientific computation, recorded in the
   canonical results whether or not it matches.

**Failure semantics.**

- A mismatch **refuses the claim-bearing promotion**. It does not delete, reset,
  repair, retry or re-seed anything, and it never re-runs a stage.
- The attempt is **consumed**. A refused promotion is a governance event
  requiring a new human authorization, exactly as a crash would be.
- The observed digest, the expected digest, and the exact enforcement point are
  written to a failure record so the difference can be diagnosed read-only.

**Recorded evidence.** Canonical results gain a `runtime_identity_checks` block:
the startup digest, one entry per promotion check with its enforcement point, the
completion digest, the total check count, and a single boolean that is true only
if every observation equalled `FROZEN_DEPENDENCY_DIGEST`. Absence of the block
means the run predates this control — it must never be synthesised for a
completed run.

**Cost.** `importlib.metadata` enumeration of ~335 distributions, at points that
already perform disk writes. Negligible relative to a promotion.

## 4. Residual exposure this does not close

A check is a sampling instrument. A mutation that lands and is reverted entirely
between two checks is invisible to it, and a mutation that lands after the final
check but before process exit cannot affect already-written artifacts anyway. The
control converts an undetected environment change into a refused promotion; it
does not make the environment immutable. The durable protection is the separation
in §2 — nothing else installs into the scientific interpreter.

Filesystem-level protection (read-only `site-packages` for the scientific
environment outside authorized maintenance windows) would close the remaining
gap. It is **proposed, not applied**, and requires a separate human decision.

## 5. Approval required before implementation

Implementation touches canonical execution paths and must not be undertaken
without explicit human authorization. Open questions for that decision: whether a
completion-time mismatch should invalidate a run whose promotions all passed, or
be recorded as evidence only; and whether the filesystem protection in §4 is
adopted.
