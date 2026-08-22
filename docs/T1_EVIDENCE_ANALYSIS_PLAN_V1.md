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

## 7. Endpoint and claim hierarchy

**Added 2026-08-22 at approval, still before any measured value has been read.**
Sections 1–6 are unchanged. This section adds structure the pre-registration
lacked; it removes nothing and weakens no constraint.

### 7.1 What "causal" means in this programme

Load-bearing, because it is the single most misreadable word in the eventual
paper. In this codebase **causal means non-anticipative**: `t1_protocol.step` is
"one causal step. Pure: it reads the current row and nothing ahead of it", and
`t1_stream` adapts a chronological stream "strictly causally". It is a
streaming-order guarantee about information access.

**It is not causal inference.** Nothing in this evidence estimates a treatment
effect, an intervention, or a counterfactual. The phrase "causal episode
measurement" must never appear in a paper without that qualification attached,
and "causal" should not appear in an abstract at all.

### 7.2 Endpoint hierarchy

| Tier | Endpoint | Basis |
|---|---|---|
| **Primary** | `episode_f1`, pooled and per subject, with the 1000-replicate subject bootstrap | Defined for 12/12 subjects; it is the quantity T1 exists to measure — episode-level alerting; it is the only slot that supports a subject bootstrap without conditioning on a data-dependent subset |
| **Secondary** | `primary_window_mcc`, per subject and pooled | Window-level, a different task from episode alerting; defined for 5/12 subjects individually |
| **Exploratory** | `onset_latency_seconds_median` | Conditional on successful detection **by construction** — a subject with no matched episode has no latency at all, so latency is only ever measured where detection already succeeded |

**Latency is never a headline claim.** Because it is defined only on matched
episodes, a latency figure describes the timeliness of the detections that
happened, not the timeliness of the system. Those two are different quantities
and only the first is measurable here.

**Disclosure.** This hierarchy is conditioned on metric availability, and
availability is a measured property of the data (§2). It was fixed before any
value was read, but it was not fixed before *anything* was known — §2 was. That
is disclosed rather than presented as fully a priori, and it is why §2 was
restricted to definedness and confined to counts.

### 7.3 Reporting rules, reconciled with §4.2

To remove an ambiguity in the phrase "defined subjects only":

1. **The per-subject table is always complete.** Twelve rows, always, with
   undefined cells shown as undefined. "Defined subjects only" never applies to
   the table.
2. It applies only to **summaries**, and every summary states its `n` in the
   same sentence — *"n of 12 subjects, the remainder undefined"*.
3. **No subject-macro mean of MCC or latency is reported anywhere**, with or
   without an attached `n`. A mean over a data-determined subset answers "how
   did it do where the metric happened to exist", not "how did it do".
4. A pooled value is labelled **pooled** and never described as an average
   across subjects. Pooling confusion counts and averaging per-subject metrics
   are different estimators with different meanings.

### 7.4 Supported claims

Within the 12 LTSTDB validation subjects, cross-fitted and subject-disjoint:

- Episode-level detection performance, as pooled and per-subject `episode_f1`
- Between-subject variability, exactly as scoped by the bootstrap's own
  `claim_scope` string
- Window-level pooled description, labelled as window-level
- Onset latency on the subset where episodes were matched, labelled exploratory
- The provenance and auditability of the chain itself: which artifacts were
  consumed, which digests verified, that no model ran and no test was opened

### 7.5 Not supported

Each of these needs evidence that does not exist, and in several cases evidence
that is not authorized to be created.

| Claim | Why not |
|---|---|
| **External generalization** | One dataset, 12 subjects. EDB shares source recordings with LTSTDB per `CROSS_DATASET_PROVENANCE.md` and is not a clean external cohort |
| **Population subgroup** | `join_performed: false`, `strata_reported: []` — an absent join, recorded before execution |
| **Candidate architecture comparison** | The rejected B4 candidates were never run on held-out subjects, by design |
| **Any test claim** | The B4/neural sealed test is unopened and stays so |
| **Clinical claim** | Research software, public-dataset validation only |
| **Statistical significance** | The bootstrap is not a hypothesis test |
| **"T1 improved episode detection"** | **There is no comparator in this evidence.** No no-T1 arm was run on these held-out subjects. Improvement is a two-armed claim and this is a one-armed measurement |
| **"Memory helps"** | An ablation. No no-memory arm exists, and M1/M2 reruns are forbidden by standing constraint |
| **"S4D improves temporal coherence over GRU"** | `T1_T2_COMPARATOR_ARM` is declared in `t1_protocol`, but the continuation measured the retained arm only — **no file in the run references the comparator**. If this is answerable at all it is from the T2 artifacts under T2's own claim scope, not from T1 |

**The general rule this table encodes:** T1 measures one configuration on held-out
subjects. Every comparative verb — improved, helped, outperformed, better — needs
a second arm, and this evidence has one arm. A comparative claim requires the
ablation package, which is a separate decision and would require runs that are
currently unauthorized.

### 7.6 Failure-mode analysis

Permitted at the **aggregate structural** level: counts of unmatched predicted
runs, reference episodes with no matched prediction, and the relationship between
empty confusion margins and undefined metrics. These read directly off the
artifacts.

**Not permitted:** per-subject narrative explaining why a particular subject
scored as it did. That remains excluded by §4.5, and it is excluded precisely
because it is the analysis most likely to be written backwards from the numbers.

### 7.7 Pooled and subject-macro are different estimands

Established by reading `t1_continuation_results.py`, not by reading any value.
This is the most consequential thing the original plan did not say.

**The bootstrap does not describe the pooled episode F1.** Its replicate
statistic is

```
build_bootstrap:  float(np.mean(values[row]))
```

the **mean over resampled subjects of each subject's own `episode_f1`**. The
pooled figure reported in §4.1 is

```
pooled_episode_f1 = _episode_f1(pooled_episodes)
                  = 2 * matched / (predicted_event_runs + reference_episodes)
```

computed from **counts pooled across subjects**. The first is subject-weighted;
the second is episode-weighted. They are different estimators of different
quantities, they need not be close, and **the pooled value need not lie inside
the bootstrap interval.**

**Binding consequence.** The interval must never be printed adjacent to
`pooled_episode_f1` in a way that lets a reader take it as an interval around
that number. Wherever the interval appears it carries, in the same block, the
statement that its central quantity is the subject-macro mean of per-subject
`episode_f1`.

**One derived quantity is authorized, and only one.** The subject-macro mean of
the twelve stored per-subject `episode_f1` values may be computed and reported,
labelled **derived**, with its formula shown, so that the pre-registered interval
has the point estimate it actually refers to. This is the sole exception to
"nothing is recomputed" in §4.1 item 1. It is admissible because `episode_f1` is
defined for 12/12 subjects, so the mean is over the complete subject set and not
over a data-dependent subset — which is exactly why the same exception is **not**
extended to MCC or latency (§7.3 item 3). It is authorized here, before any value
is visible.

**The pooled onset latency is episode-weighted too.** `onset_latency_seconds_median`
is `_median` over the concatenation of every fold's `onset_latency_seconds`, so
subjects contributing more matched episodes count more heavily. It is a median
over episodes, not over subjects, and must be labelled that way.

**Resolution caveat.** The bootstrap resamples 12 subjects. However many
replicates are drawn, the underlying sample has twelve distinct values, so the
percentile interval is coarse and its tails are governed by a handful of
subjects. Report it as an indication of between-subject spread, which is what its
`claim_scope` already says, and never as a precision statement.

### 7.8 Where §7 supersedes §4

§7.3 item 3 is stricter than §4.2 item 5. §4.2 item 5 permits a summary of MCC or
latency provided `n` is stated in the same sentence, which leaves open a mean
over the five defined subjects; §7.3 item 3 forbids a subject-macro mean of those
two entirely. **§7.3 governs.** The tightening runs in the conservative
direction — it removes a reportable number rather than adding one — and it is
made before any value is visible.

§7.7 relaxes §4.1 item 1 in exactly one place, the subject-macro mean of
`episode_f1`, for the reason given there. No other recomputation is authorized.

---

## 8. Approval record

**Added 2026-08-22 at approval. Sections 1–6 are the pre-registration as written
at the end of ECG 12 and are committed exactly as written; nothing in them was
altered at approval. Section 7 was added at approval, and — like §§1–6 — was
written before any measured value was read. It adds structure and removes no
constraint.**

```
Plan status  : APPROVED AS WRITTEN
Approved by  : Debalekha Chakraborty
Date         : 2026-08-22
Step 4       : AUTHORIZED — the §4 descriptive report may be produced
```

§5 step 1 (`Approve this plan`, gate: human) is satisfied. Step 4, the first read
of the measured values, is authorized as an explicit analysis authorization
rather than as a side effect of a status check, and is bound to the reporting
shape fixed in §4 and §7 — in particular §4.2 item 4 and §7.3: **undefined
subjects are shown as undefined, never omitted, never zero-filled; the
per-subject table is always complete; and no subject-macro mean of MCC or
latency is reported anywhere.** That shape was chosen before the values were
visible and is not renegotiable now that they are.

The §7.5 exclusions bind the same way. In particular, **no comparative verb —
improved, helped, outperformed, better — may be applied to this evidence**, which
measures one arm. Comparative and ablation questions are not deferred pending
analysis; they are unanswerable from this run and require separate authorized
evidence.

### 8.1 Structural re-verification at approval

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
