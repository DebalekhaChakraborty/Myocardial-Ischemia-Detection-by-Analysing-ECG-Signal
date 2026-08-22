# T1 Continuation Evidence — Analysis Plan, V1

**Pre-registration.** This plan is written from a structural inventory of the
continuation artifacts. **No measured value has been read, reported or
interpreted in preparing it**, and nothing below is chosen because of what the
numbers say. That ordering is the point: an analysis chosen after seeing results
is a selection on the evidence, and this programme spent nine PRs making that
impossible everywhere else.

Analysis has **not** been authorized. This is the plan to be approved, not the
analysis.

| | |
|---|---|
| Evidence | `cardiosentinel-runs/phase9-t1-continuation-v1/t1-v1-measurement-continuation` |
| Execution commit | `61704aa7259d91eaf9d4dfc2502bf78881a05d61` |
| Authorization commit | `b40b4acac16893dcb1af1f1fa91feb0d74c8a78d` |
| Executed | 2026-08-22T16:18:39Z → 16:18:49Z |
| Prepared | 2026-08-22, from structure only |

---

## 1. What exists

19 files. Six run-level artifacts, one execution attestation, twelve per-fold
evidence files. Every digest recorded in `T1_EXPERIMENT_LOCK.json` re-verifies
against the file on disk; the lock omits its own digest, necessarily.

Every artifact carries `continues.predecessor_run = t1-v1-development`,
`continues.predecessor_digest = cf74f00a…`, 20 `consumed_evidence` entries,
`is_continuation_artifact: true`, `test_accessed: false`,
`sealed_test_state: "unopened"`. No file anywhere in the run contains
`policy_runs`.

**The three lost quantities are recovered, 12/12 folds**: per-fold PRIMARY
confusion counts, per-fold episode evidence, per-fold onset latencies.

---

## 2. Metric availability — the one thing that shapes this plan

Availability, not results. Reported because it changes which analyses are
admissible, and that decision must be made before anyone reads a value.

| Slot | Defined |
|---|---|
| `episode_f1` | **12 / 12** subjects |
| `primary_window_mcc` | **5 / 12** subjects |
| `onset_latency_seconds_median` | **5 / 12** subjects |

The seven subjects with undefined MCC are **exactly** the seven with undefined
latency, and all seven have at least one zero margin in their PRIMARY confusion.
That is the documented behaviour of the frozen helpers, not an anomaly:
`window_mcc` returns undefined when any margin is empty, and a subject whose
episodes were all unmatched has no latency at all. Both refuse to report zero,
because zero would read as a real measurement.

**Consequence.** Any subject-level pooling of MCC or latency runs over 5 of 12
subjects. Which 5 is determined by the data, so a pooled MCC is a statistic over
a data-dependent subset — not a subject-macro average, whatever it is labelled.
§4 fixes how this is handled *before* the numbers are read.

The bootstrap is unaffected: it resamples `episode_f1`, defined for all twelve,
and reports 1000 defined replicates and 0 undefined.

---

## 3. What the evidence can and cannot support

**Can support**

- Pooled PRIMARY window-level confusion and episode evidence across 12 held-out
  subjects, cross-fitted, subject-disjoint
- `episode_f1` per subject and pooled, with a 1000-replicate subject bootstrap
  at seed 2026, no reselection inside
- Per-fold description under each fold's promoted policy and thresholds
- Onset latency, on the subset where episodes were matched

**Cannot support**

- Any **test** claim. The B4/neural sealed test is unopened and stays so.
- Any **generalization** claim beyond LTSTDB. One dataset, 12 validation
  subjects. The obvious second dataset, EDB, shares source recordings with
  LTSTDB per `CROSS_DATASET_PROVENANCE.md`, so it is not a clean external set.
- Any **clinical** claim. Research software, public-dataset validation only.
- Any **subgroup** claim. `join_performed: false`, `strata_reported: []` — an
  absent join, recorded before execution.
- Any **comparative** claim against a policy that was not selected. The eleven
  rejected candidates were never run on held-out subjects, by design.
- Any **statistical significance** claim. The bootstrap's own
  `claim_scope` names what it covers: between-subject variation conditional on
  the cross-fitted procedure. It is not a hypothesis test.

---

## 4. Pre-specified analysis

Decided now, in this order, before any value is read.

### 4.1 Primary descriptive

1. Pooled PRIMARY confusion, pooled episode evidence, pooled `episode_f1`,
   pooled window MCC — read verbatim from `T1_OOF_RESULT.json`. **Nothing is
   recomputed.** The artifact is the result; recomputing would create a second
   number that could disagree with the promoted one.
2. Per-fold table from `fold_summaries`: fold index, held-out subject, selected
   policy, confusion, episode evidence, `episode_f1`.

### 4.2 Subject-level

3. Report `episode_f1` for all 12 subjects.
4. **MCC and latency are reported per subject, with the undefined subjects shown
   as undefined — never omitted, never zero-filled.** A table with seven visible
   gaps is honest; a pooled number over the five that happened to be defined is
   a statistic nobody specified.
5. If a summary of MCC or latency is wanted, it is reported as *"n of 12
   subjects, the remainder undefined"* with n stated in the same sentence. No
   bare mean.

### 4.3 Uncertainty

6. Bootstrap interval read verbatim from `T1_BOOTSTRAP.json`, always with its
   `claim_scope` string attached. The interval is not a confidence interval for
   a population parameter and must not be written as one.
7. No p-value. No significance language. `statistical_significance_claim` is
   false throughout the upstream chain and stays false here.

### 4.4 Provenance

8. Every reported number cites the artifact and digest it came from.
9. The report states that the measurement consumed a persisted trace and ran no
   model, citing the four zero counters in the attestation.

### 4.5 Explicitly excluded

- Re-deriving any metric from the `.npz` stores
- Any threshold sweep, ROC or operating-point exploration — thresholds were
  frozen per fold before held-out labels were opened
- Any per-subject narrative explaining why a particular subject scored as it did
- Any comparison to B0–B3 or B4-B validation numbers without stating that those
  are window-level detector metrics on a different task from episode-level
  alerting

---

## 5. Sequence

| # | Step | Gate |
|---|---|---|
| 1 | Approve this plan | human |
| 2 | Fill `Execution commit: 61704aa…` in the pre-authorization record | mechanical |
| 3 | Commit the analysis plan and the execution record | PR |
| 4 | Produce the descriptive report per §4 | first read of values |
| 5 | Ablation package | separate decision |
| 6 | External validation strategy | separate decision |
| 7 | Paper assembly | separate decision |

Step 4 is the first time anyone reads a measured value. It should be a human
reading them, or an explicitly authorized analysis — not a side effect of a
status check.

---

## 6. Risks to carry forward

**The evidence is on one disk.** The continuation artifacts are gitignored, like
every other run artifact. 2.3 GB of canonical runs and ~21 GB of features exist
in one place and none of it may be rerun. This is the largest unmanaged risk in
the programme and it is now larger, because the continuation is also
unrepeatable — §14 authorizes no second one.

**Seven of twelve subjects carry undefined MCC and latency.** Whatever the
values turn out to be, the reporting shape is fixed by §4.2 and should not be
renegotiated once they are visible.

**External validation remains the milestone that decides generalization.** No
amount of resolution on this evidence substitutes for a clean second cohort.

---

## 7. Approval record

**Added 2026-08-22 at approval. Sections 1–6 above are the pre-registration and
are committed exactly as written before any measured value was read. Nothing in
them was altered at approval.**

```
Plan status  : APPROVED AS WRITTEN
Approved by  : Debalekha Chakraborty
Date         : 2026-08-22
Step 4       : AUTHORIZED — the §4 descriptive report may be produced
```

§5 step 1 (`Approve this plan`, gate: human) is satisfied. Step 4, the first read
of the measured values, is authorized as an explicit analysis authorization
rather than as a side effect of a status check, and is bound to the reporting
shape fixed in §4 — in particular §4.2 item 4: **undefined subjects are shown as
undefined, never omitted, never zero-filled.** That shape was chosen before the
values were visible and is not renegotiable now that they are.

### 7.1 Structural re-verification at approval

Every structural claim in §1 and §2 was re-checked against the artifacts on disk
before approval. No measured value was read, reported or interpreted in doing so;
only definedness, counts and digests were examined.

| Claim | Result |
|---|---|
| 19 files, 6 run-level + 1 attestation + 12 per-fold | confirmed |
| 7 recorded artifact digests re-verify | 7 / 7 |
| `continues.predecessor_run = t1-v1-development`, `predecessor_digest = cf74f00a…` | 19 / 19 files |
| 20 `consumed_evidence` entries | 19 / 19 files |
| No file in the run contains `policy_runs` | confirmed, 0 files |
| `episode_f1` defined | 12 / 12 subjects |
| `primary_window_mcc` defined | 5 / 12 subjects |
| `onset_latency_seconds_median` defined | 5 / 12 subjects |
| Undefined-MCC subject set is exactly the undefined-latency set | confirmed |
| Four §13.7 counters zero; `test_accessed: false`; `sealed_test_state: unopened` | confirmed |
| Consumed attempt unchanged, 20 files at `2026-08-21T19:57:57` | confirmed |
| Frozen digests: 5 T1 modules + amendment V1.1 | 6 / 6 match |
| `installed_packages_sha256` via `provenance.dependency_environment()` | `b0fd6ea…` matches |
