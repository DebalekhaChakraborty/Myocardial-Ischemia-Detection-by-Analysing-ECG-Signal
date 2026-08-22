# T1 Continuation Evidence — Descriptive Report, V1

**Step 4 of `docs/T1_EVIDENCE_ANALYSIS_PLAN_V1.md`: the first read of the
measured values.** Produced under the analysis authorization in §8 of that
plan, to the reporting shape fixed in §4 and §7 before any value was visible.
Nothing in the plan was changed after the values became readable.

**Every number is read verbatim from a promoted artifact, with one exception**
named and authorized in advance by plan §7.7: the subject-macro mean of
`episode_f1`, which no artifact stores and which is the estimand the
pre-registered bootstrap targets. No other quantity is recomputed and no
`.npz` store was opened.

| | |
|---|---|
| Attempt | `t1-v1-measurement-continuation` |
| Run class | `t1_continuation_measurement` |
| Execution commit | `61704aa7259d91eaf9d4dfc2502bf78881a05d61` |
| Authorization commit | `b40b4acac16893dcb1af1f1fa91feb0d74c8a78d` |
| Continues | `t1-v1-development` at `c538181` |
| Held-out subjects | 12, cross-fitted, subject-disjoint |
| Sealed test state | `unopened` |

---

## 1. Primary result — plan §7.2

The primary endpoint is the **subject-macro mean `episode_f1`**, and the
bootstrap interval is the interval for that quantity and no other.

| | |
|---|---|
| **Subject-macro mean `episode_f1`** | **0.2524** |
| **95% subject-bootstrap interval** | **[0.0826, 0.4415]** |
| Subjects | 12 |
| Replicates | 1,000 (1,000 defined, 0 undefined) |
| Seed | 2026 |

The point estimate is `(1/N)·Σ F1_i` over the N = 12 per-subject values in §2.
It is the one derived number plan §7.7 authorizes, admissible because
`episode_f1` is defined for 12/12 subjects and the mean therefore runs over
the complete subject set rather than a data-dependent subset. The interval is
read verbatim from `T1_BOOTSTRAP.json`.

**Claim scope, quoted from the artifact, and required by plan §4.3 item 6 to
travel with the interval wherever it appears:**

> between_subject_variation_conditional_on_the_cross_fitted_t1_development_procedure

This is **not** a confidence interval for a population parameter and must not
be written as one. The bootstrap resamples 12 subjects, so the underlying
sample has twelve distinct values however many replicates are drawn: it
indicates between-subject spread, not precision. There is no p-value and no
significance language anywhere in this report.

**The episode-weighted pooled figure is 0.3423 and is reported
separately in §3.** It is a different estimand and the interval above does not
bracket it (plan §7.7).

---

## 2. Per-subject results — plan §4.2, §7.3

From `T1_SUBJECT_EVIDENCE.json`, SHA-256 `6695dd36d890dfdc5e6e6fa16514f2cee8676b7402ba93f0c0f9c10b27223120`.

**All twelve subjects, always. Undefined is shown as undefined — never
omitted, never zero-filled.** Fixed by plan §4.2 item 4 and §7.3 item 1 before
the values were visible.

| Subject | Fold | `episode_f1` | `primary_window_mcc` | onset latency, s (median) |
|---|---|---|---|---|
| `ltstdb:s2004` | 0 | 0.3750 | 0.3672 | 120.00 |
| `ltstdb:s2005` | 1 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s2019` | 2 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s2020` | 3 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s2023` | 4 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s2031` | 5 | 0.6207 | 0.0605 | -970.00 |
| `ltstdb:s2057` | 6 | 0.8000 | 0.3081 | 85.00 |
| `ltstdb:s2058` | 7 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s2059` | 8 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s3068` | 9 | 0.4091 | 0.5732 | 380.00 |
| `ltstdb:s3072` | 10 | 0.0000 | *undefined* | *undefined* |
| `ltstdb:s3073` | 11 | 0.8235 | 0.6760 | 130.00 |

`episode_f1` is defined for 12/12 subjects. `primary_window_mcc` for 5/12,
`onset_latency_seconds_median` for 5/12, and the undefined sets coincide
exactly: `ltstdb:s2005`, `ltstdb:s2019`, `ltstdb:s2020`, `ltstdb:s2023`, `ltstdb:s2058`, `ltstdb:s2059`, `ltstdb:s3072`.

Those subjects have at least one empty margin in their PRIMARY confusion, so
`window_mcc` is undefined by construction, and no matched episode, so there is
no latency to take a median of. Both helpers refuse zero because zero would
read as a real measurement.

**No subject-macro mean of MCC or latency is reported anywhere in this
document** (plan §7.3 item 3). Such a mean would run over a subset the data
chose, and would answer how the system did where the metric happened to exist.

---

## 3. Pooled description — plan §4.1, §7.2

From `T1_OOF_RESULT.json`, SHA-256 `9309b00b55173e00ee793d2468b6aaf796105928c0e5241537ef3fe80ccec6ae`. Read verbatim.

**Descriptive, and episode- or window-weighted. Not the primary estimate, and
not what the §1 interval brackets.**

Pooled PRIMARY window confusion across all 12 held-out subjects:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Reference positive** | TP 7,429 | FN 14,199 |
| **Reference negative** | FP 17,295 | TN 434,974 |

Total windows: 473,897.

Pooled episode evidence:

| Quantity | Value |
|---|---|
| `reference_episodes` | 163 |
| `predicted_event_runs` | 59 |
| `matched_episodes` | 38 |
| `unmatched_predicted_runs` | 21 |

Pooled metrics, each with its weighting stated as plan §7.7 requires:

| Metric | Value | Weighting |
|---|---|---|
| `pooled_episode_f1` | 0.3423 | episodes — `2·matched/(predicted+reference)` on pooled counts |
| `pooled_primary_window_mcc` | 0.2865 | windows, pooled across subjects |
| `matched_episode_count` | 38 | — |

These are defined even though seven individual subjects are not, because
pooling the counts removes the empty margins that leave those subjects
undefined. Pooling is what makes them defined; it does not repair the
undefined subjects and does not stand in for them.

---

## 4. Per-fold table — plan §4.1 item 2

Each row is one held-out subject under the policy its own fold promoted. The
policy column is provenance for the row and carries no commentary (plan §7.10).

| Fold | Held-out subject | Policy | TP | FP | FN | TN | Ref ep. | Pred runs | Matched | `episode_f1` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `ltstdb:s2004` | `qw0.9_qe0.99_FAST` | 1,626 | 497 | 5,127 | 26,883 | 38 | 10 | 9 | 0.3750 |
| 1 | `ltstdb:s2005` | `qw0.9_qe0.99_BALANCED` | 0 | 6,947 | 0 | 23,413 | 0 | 7 | 0 | 0.0000 |
| 2 | `ltstdb:s2019` | `qw0.9_qe0.99_FAST` | 0 | 0 | 138 | 34,398 | 6 | 0 | 0 | 0.0000 |
| 3 | `ltstdb:s2020` | `qw0.9_qe0.99_FAST` | 0 | 4,764 | 0 | 24,992 | 0 | 8 | 0 | 0.0000 |
| 4 | `ltstdb:s2023` | `qw0.9_qe0.99_FAST` | 0 | 2 | 0 | 32,018 | 0 | 1 | 0 | 0.0000 |
| 5 | `ltstdb:s2031` | `qw0.9_qe0.99_FAST` | 181 | 3,034 | 668 | 27,280 | 18 | 11 | 9 | 0.6207 |
| 6 | `ltstdb:s2057` | `qw0.9_qe0.99_FAST` | 211 | 1,303 | 65 | 18,548 | 5 | 5 | 4 | 0.8000 |
| 7 | `ltstdb:s2058` | `qw0.9_qe0.99_FAST` | 0 | 0 | 99 | 31,570 | 3 | 0 | 0 | 0.0000 |
| 8 | `ltstdb:s2059` | `qw0.9_qe0.99_FAST` | 0 | 0 | 1,241 | 31,346 | 47 | 0 | 0 | 0.0000 |
| 9 | `ltstdb:s3068` | `qw0.9_qe0.99_FAST` | 3,906 | 109 | 6,057 | 40,902 | 35 | 9 | 9 | 0.4091 |
| 10 | `ltstdb:s3072` | `qw0.9_qe0.99_FAST` | 0 | 0 | 47 | 50,321 | 1 | 1 | 0 | 0.0000 |
| 11 | `ltstdb:s3073` | `qw0.9_qe0.99_FAST` | 1,505 | 639 | 757 | 93,303 | 10 | 7 | 7 | 0.8235 |

---

## 5. Secondary — window-level MCC, plan §7.2

Window-level classification, **a different task from episode-level alerting**
and not interchangeable with the primary endpoint. Per-subject values are in
§2, complete with their gaps; the pooled figure is in §3.

Pooled `primary_window_mcc` = 0.2865, over pooled windows. Individually
defined for 5 of 12 subjects.

---

## 6. Exploratory — onset latency, plan §7.2

**Conditional on successful detection by construction.** A subject with no
matched episode has no latency at all, so latency is only ever measured where
detection already succeeded. This is not a headline result.

| | |
|---|---|
| Median latency across detected episodes | 120.00 s |
| Subjects with a defined median | 5 / 12 |

**This statistic is weighted by episodes, not by subjects.** It is `_median`
over the concatenation of every fold's `onset_latency_seconds`, so a subject
contributing more matched episodes weighs more heavily. Per plan §7.2 the
permitted phrasing is *"median latency across detected episodes"*, or
preferably *"episode-level onset latency distribution among detected
episodes"*. It is **not** a median patient onset latency and **not** a
detection latency of the system, which would imply coverage of the episodes
the system missed.

---

## 7. Challenge strata — plan §3

From `T1_CHALLENGE_EVIDENCE.json`, SHA-256 `0eb8e684944da6768511d57264b20b8d201ab935bdb73125a0f41f9b3fed2d25`.

- `join_performed`: `false`
- `strata_reported`: `[]`
- `selection_performed_on_challenge_evidence`: `false`

> Challenge strata are reported, never selected on. Absent strata are recorded as absent rather than as empty subgroups.

**No subgroup claim is available.** The absent join was recorded before
execution, not discovered after.

---

## 8. Provenance — plan §4.4

### 8.1 Artifact digests

Every number above comes from one of these files, each re-verified against
disk at the time of writing.

| Artifact | SHA-256 |
|---|---|
| `T1_OOF_RESULT.json` | `9309b00b55173e00ee793d2468b6aaf796105928c0e5241537ef3fe80ccec6ae` |
| `T1_SUBJECT_EVIDENCE.json` | `6695dd36d890dfdc5e6e6fa16514f2cee8676b7402ba93f0c0f9c10b27223120` |
| `T1_BOOTSTRAP.json` | `57ba66553e712a63b0f670cbb01bc9d680c824a90a2c9b723baa1aaa1adc0f48` |
| `T1_CHALLENGE_EVIDENCE.json` | `0eb8e684944da6768511d57264b20b8d201ab935bdb73125a0f41f9b3fed2d25` |
| `T1_FINAL_CONFIGURATION.json` | `374114293160c1f778a4803ff3a2d893d0eda2b81d6277ae326e201084495a34` |
| `T1_EXPERIMENT_LOCK.json` | `bcbdfdb08293b9c2ba7a9abef38d185e3128177c555b01dea0b81ec62f726a76` |
| `T1_V1_CONTINUATION_EXECUTION_ATTESTATION.json` | `b5a557dd40927999e00516e982c2f1619fdbeb3e5ebdd3ad108037b474eca588` |

Plus 12 `held_out_evaluations/T1_CONTINUATION_FOLD_NN_HELD_OUT.json`.
`T1_EXPERIMENT_LOCK.json` records six artifact digests, not seven: a file
cannot contain its own digest.

### 8.2 The measurement ran no model — plan §4.4 item 9

| Counter | Value |
|---|---|
| `fold_evaluations` | `0` |
| `policy_selection_calls` | `0` |
| `state_machine_invocations` | `0` |
| `threshold_generation_calls` | `0` |
| `state_transitions_regenerated` | `false` |
| `test_accessed` | `false` |
| `selection_performed_here` | `false` |
| `thresholds_generated_here` | `false` |
| `sealed_test_state` | `unopened` |

The continuation consumed a persisted state trace
(`state_trace_source: predecessor_oof_state_evidence`, content SHA-256
`cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f`) and evaluated no model. All three
negative capability gate layers passed. Predecessor verification re-verified
8 §1.3 artifact digests and
12 §1.4 fold-selection digests.

**The leakage guarantee is inherited, not re-enforced here.** The continuation
invokes no transition function, so `T1_FORBIDDEN_TRANSITION_INPUTS` does not
run in this process. The guarantee comes from the predecessor development run,
which enforced it, carried through the digest-verified state trace above.

---

## 9. Structural observations — plan §7.6

Aggregate and structural, read directly off the counts. **These were observed
after the values were read, and they change no pre-registered number in this
report.** They are recorded because they bear on how §1 should be understood,
and because the alternative — noticing them silently and adjusting — is the
thing the pre-registration exists to prevent.

### 9.1 Three subjects have no reference episodes at all

`_episode_f1` returns undefined only when `predicted + reference == 0`. For a
subject with no reference episodes but at least one predicted event run the
denominator is non-zero, so F1 evaluates to exactly `0.0` — **a false-alarm
penalty on a subject that had nothing to detect, not a failure to detect.**

The primary subject-macro mean in §1 therefore averages two different kinds of
zero:

| Kind | Subjects | Ref ep. | Pred runs | Matched |
|---|---|---|---|---|
| No reference episodes; predictions fired | `s2005`, `s2020`, `s2023` | 0 | 7, 8, 1 | 0 |
| Reference episodes present; none matched | `s2019`, `s2058`, `s2059`, `s3072` | 6, 3, 47, 1 | 0, 0, 0, 1 | 0 |

That is 3 subjects of the first kind and 4 of the second, 7 of the
twelve contributing a zero to the primary mean for two incomparable reasons.

This is a property of the frozen helper, not a defect in the run, and **no
number in this report is changed on account of it.** Whether a future analysis
should treat zero-reference subjects distinctly is a decision that could only
be made after seeing these values; any such change is post-hoc and must be
labelled that way, in a V2 that says so.

### 9.2 Onset latency is signed, and some of it is negative

`_onset_latency` computes

```
(start_samples[run_begin] - start_samples[episode_begin]) / 250.0
```

seconds from a matched episode's annotated onset to its predicted run's onset.
**It is a signed offset, not a non-negative delay.** A negative value means the
predicted event run began *before* the annotated onset of the episode it matched.

**6 of the 38 matched-episode latencies are negative:**

| Subject | Negative | Total matched | Median, s |
|---|---|---|---|
| `ltstdb:s2004` | 0 | 9 | 120.00 |
| `ltstdb:s2031` | 5 | 9 | -970.00 |
| `ltstdb:s2057` | 1 | 4 | 85.00 |
| `ltstdb:s3068` | 0 | 9 | 380.00 |
| `ltstdb:s3073` | 0 | 7 | 130.00 |

A summary that reports only the pooled median conceals that the underlying
distribution spans both signs. Any figure or sentence built on latency states
the sign convention, and no latency summary is reported as a delay.

---

## 10. What this study does not evaluate — plan §7.9

Restated unchanged. None was weakened after seeing the values.

- **Improvement over a T1-disabled system.** No T1-disabled arm was run on
  these subjects. This measures one configuration; it does not compare two.
- **The contribution of the memory modules.** No no-memory arm exists.
- **The contribution of the longitudinal SSM architecture.** The retained arm
  only; no file in the run references the declared comparator arm.
- **External generalization.** One dataset, 12 subjects. EDB shares source
  recordings with LTSTDB and is not a clean external cohort.
- **Subgroup or stratified performance.** See §7.
- **Held-out test performance.** The B4/neural sealed test is unopened.
- **Clinical utility.** Research software, public-dataset validation only.
- **Deployment behaviour.** No inference or serving path exists.

No comparative verb — improved, helped, outperformed, better — applies to this
evidence. Every one of them needs a second arm and this is a one-armed
measurement.

---

## 11. Excluded analyses — plan §4.5

Not done, and not to be done as a follow-up without a separate decision:

- Re-deriving any metric from the `.npz` stores
- Any threshold sweep, ROC or operating-point exploration — thresholds were
  frozen per fold before held-out labels were opened
- Any per-subject narrative explaining why a particular subject scored as it
  did (plan §7.6)
- Any comparison to B0–B3 or B4-B validation numbers without stating that
  those are window-level detector metrics on a different task from
  episode-level alerting
