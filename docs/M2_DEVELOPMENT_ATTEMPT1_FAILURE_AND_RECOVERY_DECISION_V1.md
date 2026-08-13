# M2-v1 development attempt #1 — failure and recovery decision

> **STATUS: FROZEN HUMAN RECOVERY DECISION — AFTER A FAILED PRE-SCORING
> ATTEMPT, BEFORE ANY SECOND DEVELOPMENT ACCESS.**
>
> This document records what attempt #1 did and did not touch, why it failed,
> and what exactly one recovery is permitted to be. **It authorizes no
> execution.** Running the recovery requires its own separate human
> authorization after the recovery PR is reviewed and merged.

## 1. The consumed attempt

| | |
|---|---|
| Canonical suite | `m2-v1-development-two-arm` |
| Execution master SHA | `3c1ba4ce87ade6a2d17386b3a9d2b579ded442e7` |
| Started | `2026-08-13T21:58:44Z` |
| M2-0 claimed | `2026-08-13T21:58:49Z` |
| M2-G claimed | `2026-08-13T21:58:50Z` |
| Failed / exited | `2026-08-13T22:00:09Z` (exit code 1) |
| Classification | **`CONSUMED_FAILED_INFRASTRUCTURE_ATTEMPT`** |

Attempt #1 must **never** be deleted, renamed, cleaned, overwritten, reset,
reused, or represented as a completed scientific M2 result.

### Preserved artifacts

```
cardiosentinel-runs/phase6-m2-development-v1/
  m2-v1-development-two-arm__M2-0/M2_RUN_STATUS.json     STARTED
  m2-v1-development-two-arm__M2-G/M2_RUN_STATUS.json     STARTED
  m2-v1-development-two-arm__evidence/M2-0/prototype_trajectories/   (empty)
  m2-v1-development-two-arm__evidence/M2-G/prototype_trajectories/   (empty)
```

SHA-256 of the two preserved status files, recorded at forensic capture:

```
3699e656ee5ab6c6d3fba90dd7dd726cbb06233d478e53b81327f394e9f6365d  __M2-0/M2_RUN_STATUS.json
7908130758cfffa171fe47f3958ee4ef7961bfe3d352486e1f2558862251a751  __M2-G/M2_RUN_STATUS.json
```

Those files say `STARTED` because the exception escaped outside a promotion
gate. They are **historical evidence and are not rewritten** to make the state
look cleaner; the forensic classification lives separately (§5 below and
`M2_ATTEMPT_FAILURE_RECEIPT.json`).

## 2. Exact failure stage and exception

Stage: **full label-blind VALIDATION replay, before the first stream could be
assembled.**

```
m2_development_run._run        -> replay_both_arms
replay_both_arms               -> iter_timeline_streams   (m2_execution.py)
iter_timeline_streams          -> join_sqi_and_morphology (m2_gate_derivation.py)

M2GateDerivationError: The COMBINED_V1 feature corpus's TRAIN record set
does not match the M1 stream cache's record list.
```

## 3. Scientific exposure

```
validation_access_occurred = true
```

The access was limited to: development source/feature integrity verification;
opening the validation stream cache; verifying the frozen full-replay
stable-ID/manifest identity; and the attempted feature-column alignment that
raised.

Before the failure:

* no ECG waveform was scored by M2;
* no M2 score was generated;
* no prototype update occurred;
* no M2-0 or M2-G outcome was observed;
* no PRIMARY label was loaded;
* no CHALLENGE annotation was loaded;
* no stress interval was evaluated;
* no AUPRC, AUROC, F1, sensitivity, specificity, PPV, NPV, MCC, FPR or drift
  statistic was computed;
* no arm comparison occurred;
* no retention preference occurred;
* **no TEST access occurred**; the B4 sealed test remains unopened and no
  `TEST_ATTEMPT` exists.

No result, lock or suite was promoted. The attempt has **no canonical
standing**.

The correction below is therefore classified as **EXECUTION-INFRASTRUCTURE
RECOVERY**, and explicitly not model tuning, threshold tuning,
outcome-guided development or scientific redesign. Nothing about the failure
could have revealed an M2 outcome, because no outcome existed.

## 4. Root cause

`m2_gate_derivation._train_record_cache_paths()` selects COMBINED_V1 manifest
entries with `partition == "train"` and `status == "complete"`.
`join_sqi_and_morphology()` used that TRAIN-only set and required exact equality
with the stream-cache manifest's `record_ids`.

That was correct for the frozen M2 **TRAIN** gate derivation, which is the task
it was written for. But `m2_execution.iter_timeline_streams("validation")`
reused it while its stream-store manifest held the **VALIDATION** record list,
and those two sets can never be equal. The canonical DEVELOPMENT route could
not enter its first validation stream on the merged implementation.

The defect was concealed from the test suite because the assembled end-to-end
synthetic test injected `stream_source`, bypassing the real
`iter_timeline_streams()` — the injection seam replaced precisely the component
that was broken.

## 5. Recovery scope

Permitted, and nothing beyond:

1. a distinct **partition-aware** feature-join helper for scientific timeline
   assembly, which names its partition rather than hard-coding TRAIN;
2. `iter_timeline_streams` using that helper for VALIDATION;
3. deterministic non-claim-bearing failure accounting once any arm claim
   exists;
4. exactly one prospective recovery suite identity;
5. recovery lineage bound in claim-bearing provenance;
6. execution-history reporting that distinguishes the failed original from the
   recovery;
7. synthetic tests and CI.

The frozen M2 TRAIN gate derivation is **not** touched.
`_train_record_cache_paths()` keeps its TRAIN meaning, the canonical TRAIN
receipt is not regenerated, and no historical artifact is made to appear to have
been produced under different semantics.

## 6. The one permitted recovery identity

```
recovery_from_suite_id  = "m2-v1-development-two-arm"
recovery_suite_id       = "m2-v1-development-two-arm-recovery1"
recovery_reason_class   = "pre_scoring_partition_alignment_execution_defect"
```

Decided **now**, before any further development access. There is no
caller-selected suite id, and no `recovery2`, `attempt3`, timestamp, random
suffix or automatic alternate name is permitted.

If `m2-v1-development-two-arm-recovery1` is ever claimed and fails, execution
**STOPS FOR HUMAN REVIEW**. No further recovery is implicitly authorized by
this document.

## 7. What this document does not do

It authorizes no VALIDATION access, no TEST access, no M2 scoring, no metric,
no arm selection, no retention decision and no rollback. It changes no M2
scientific rule: M2-0, M2-G, G1–G6, the G3 bounds, the
`finite_sample_fraction` rule, `NORMAL_EVIDENCE_THRESHOLD`,
`M1L_CLASSIFICATION_THRESHOLD`, the 60-second refractory, the retained M1L
scorer and weights, the distance standardizer, the memory alpha, the PRIMARY
and CHALLENGE populations, the stress-family eligibility decision, the
cold-start bins, the prototype-drift formula and the rollback exclusion all
remain exactly as frozen.
