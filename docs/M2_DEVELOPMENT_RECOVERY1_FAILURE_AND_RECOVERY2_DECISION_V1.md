# M2-v1 recovery1 failure and recovery2 decision

> **STATUS: FROZEN HUMAN DECISION — AFTER A SECOND PRE-SCORING FAILURE, BEFORE
> ANY FURTHER DEVELOPMENT ACCESS.**
>
> **THIS CLARIFIES EXISTING FROZEN IMPLEMENTATION SEMANTICS. IT DOES NOT ADD OR
> CHANGE AN M2 SCIENTIFIC RULE.**
>
> It authorizes no execution. Running recovery2 requires its own separate human
> authorization after the recovery2 PR is reviewed and merged.

## 1. Two consumed attempts

| | Attempt #1 | Recovery1 |
|---|---|---|
| Suite id | `m2-v1-development-two-arm` | `m2-v1-development-two-arm-recovery1` |
| Execution master | `3c1ba4ce87ade6a2d17386b3a9d2b579ded442e7` | `d77fbdc37415c43728dbe3173ce58a85cfe2e71d` |
| Started | `2026-08-13T21:58:44Z` | `2026-08-14T12:52:11Z` |
| Exited | `2026-08-13T22:00:09Z` | `2026-08-14T12:53:32Z` |
| Failure stage | full label-blind replay | full label-blind replay, inside stream assembly |
| Reason class | `pre_scoring_partition_alignment_execution_defect` | `pre_scoring_source_null_join_sentinel_defect` |

Both are permanently consumed and preserved. Neither may be deleted, renamed,
cleaned, restored, reused, or represented as a completed scientific result, and
neither's status files or failure receipt may be rewritten.

### Recovery1 frozen forensic identity

```
recovery1 M2-0 M2_RUN_STATUS.json
  642cc8376c87826a5d7fdbd5d0730ca44b20f3429c26ea44c58974b45244d054
recovery1 M2-G M2_RUN_STATUS.json
  8ba15ca25b70c7686b2e39fe3e073607511835ff42fa19b5ee4d9138f4a0170d
recovery1 M2_ATTEMPT_FAILURE_RECEIPT.json  (file bytes)
  7773c6135a22e7ba64699511e1db1e92c8aac1ec9b90727d7805f540d5156446
recovery1 receipt_sha256                   (canonical)
  5b05873d48f1355292113a07d6025258e071cb9b13a35caaff1a10132cbb0408
```

Verified from those preserved artifacts: both arms claimed;
`validation_opened = true`; `replay_completed = false`;
`post_replay_evaluation_started = false`; `metrics_completed_per_arm = {}`; no
`M2_ARM_RESULT.json`, no `M2_EXPERIMENT_LOCK.json`, no `M2_SUITE_RESULT.json`;
filesystem promotion state false for every arm, lock and the suite;
`test_accessed = false`; `sealed_test_state = unopened`; every runtime
observation matched the frozen identity; no prototype-trajectory file written.

### Scoring exposure — two facts, both preserved

The immutable receipt records the conservative runtime value:

```
receipt_scoring_started = "indeterminate"
```

and **must not be rewritten**. Separately, the human forensic determination,
made from control flow rather than from the tracker:

```
human_forensic_scorer_invocation_observed = false
```

The traceback terminated inside
`iter_timeline_streams -> join_sqi_and_morphology_for_partition`, before the
iterator yielded its first stream to `replay_both_arms`, and no
prototype-trajectory file was written. Both facts are carried forward together;
neither replaces the other.

## 2. Root cause of recovery1

The partition fix itself worked. Same-partition record-set equality and exact
per-record stable-ID correspondence both passed.

The remaining defect was in the join's *validation* step. It initialised the
destination arrays with NaN and then used `np.isnan(output)` as proof that a row
had never been assigned. That conflated two different things:

* **STRUCTURAL MISSINGNESS** — the row was never assigned by the join;
* **SEMANTIC SOURCE NULL** — the row was correctly assigned a source feature
  whose metric was uncomputable.

`high_frequency_power_ratio` is a spectral ratio, and the frozen signal contract
permits a null when it cannot be computed. A legitimate source null therefore
raised a structural-integrity error.

## 3. The source-null semantics are ALREADY frozen

Nothing below is a new decision. Each is existing repository behaviour, restated
so the join correction cannot be mistaken for a scientific change.

**A. Signal contract.** Waveform spectral ratios may return null when they
cannot be computed, including near-zero total non-DC power.

**B. M1-v2 physical availability.** `observation_state` distinguishes
`AVAILABLE` from `UNAVAILABLE_EXACT_FLAT` by the already-frozen exact-flat
physical rule.

**C. M2 `evaluate_gate`.** When G1 fails because the physical observation is
unavailable, G2–G6 are **not applicable**. An exact-flat unavailable
observation is `G1 = false`, `G3 = None`, and is **not** counted as a G3
refusal.

**D. M2 `evaluate_g3`.** For an AVAILABLE row, a declared G3 feature passes only
when `np.isfinite(value) and value <= frozen_bound`. A source-null / NaN G3
feature on an AVAILABLE observation is therefore a **failed** feature, so
`G3 = false` and the memory update is refused. **No imputation is authorized.**

**E. M2-0.** The naive control does not operate G3–G6. An AVAILABLE row with a
finite representation remains governed by the inherited naive M1 update policy
even if an M2-G-only SQI quantity is null. An `UNAVAILABLE_EXACT_FLAT` row
remains unavailable to M2-0 as well.

## 4. The permitted correction

Only the join's structural-assignment proof changes. It keeps exact record-set
equality, exact stable-ID set equality, the duplicate and row-count checks, path
containment and deterministic stable-ID alignment; it replaces
NaN-as-unwritten-sentinel with an explicit assignment mask.

A legitimate source NaN **survives the join unchanged**. The join must not
replace NaN with zero, a TRAIN median, a bound or infinity; must not drop the
row or the feature; must not mark a NaN observation physically unavailable;
must not create a new SQI threshold; and must not regenerate SIGNAL_V1 or
COMBINED_V1.

The join's responsibility is **identity alignment**. The M2 policy owns the
scientific meaning of the resulting value, and that policy is unchanged.

## 5. The one permitted recovery identity

```
recovery_from_original_suite_id = "m2-v1-development-two-arm"
recovery1_suite_id              = "m2-v1-development-two-arm-recovery1"
recovery2_suite_id              = "m2-v1-development-two-arm-recovery2"
attempt1_reason_class   = "pre_scoring_partition_alignment_execution_defect"
recovery1_reason_class  = "pre_scoring_source_null_join_sentinel_defect"
```

The earlier recovery decision prohibited an *implicit* recovery2. Recovery2 is
now separately and explicitly authorized for **implementation only**, after
human review of a distinct second pre-scoring execution defect.

There is no recovery3, no attempt4, no timestamp or random suffix, no
caller-selected suite id and no automatic alternate name. If recovery2 is ever
claimed and fails, execution **STOPS FOR HUMAN REVIEW**; nothing here permits a
further attempt.

Recovery2 must bind **both** prior failure lineages and conceal neither.

## 6. Scope

This document changes no M2 scientific rule. `docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md`
is not modified and its digest is unchanged; the M2 TRAIN gate receipt is not
regenerated. M2-0, M2-G, G1–G6, the G3 columns and their frozen Q99 bounds, the
`finite_sample_fraction` rule, `NORMAL_EVIDENCE_THRESHOLD`, the M1L
classification threshold, the G5 refractory, G6, M1L, P1, B4, the memory alpha,
the PRIMARY and CHALLENGE populations, stress eligibility, the cold-start bins,
the prototype-drift formula and the rollback exclusion all remain exactly as
frozen.

No VALIDATION corpus was inspected to justify this correction: it rests on the
already-frozen source contract and the existing M2 policy, not on any observed
validation outcome. TEST remains sealed and unopened.
