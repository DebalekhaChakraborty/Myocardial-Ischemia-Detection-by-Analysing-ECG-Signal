# T1-v1 Canonical Development Execution Specification V1

## 0. Nature of this document

**THE SCIENCE IS ALREADY FROZEN IN `T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1`**
(`docs/T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1.md`, SHA-256
`ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4`).

**THIS DOCUMENT MAY NOT CHANGE:**

`Q_WATCH` · `Q_EVENT` · the three persistence profiles · the evidence formulas ·
the cold-start rule · transition semantics · the LOSO subjects · the episode
definition · the matching rule · the policy-selection metrics · the tie-break
order · the bootstrap count and seed · challenge semantics · the TEST firewall.

This document freezes **non-scientific execution mechanics only**: choreography,
claim structure, evidence opening order, label firewall mechanics, persistence
plan, failure semantics and the future CLI surface.

It is written **before** the T1 development run exists. No T1 threshold, state
trace, episode, metric, policy or configuration exists at the time of writing,
and this document creates none.

The protocol module already owns the science and must not be duplicated or
reinterpreted: `candidate_policies()`, `empirical_order_statistic()`, the
evidence definitions, `next_state()`, `group_reference_episodes()`,
`match_runs_to_episodes()`, `policy_sort_key()`, `t1_folds()`, and every frozen
constant and firewall.

---

## 1. Canonical execution identity

| | |
|---|---|
| Experiment identity | `T1_state_machine_v1` |
| Attempt id | `t1-v1-development` |
| Run root | `REPOSITORY_ROOT / cardiosentinel-runs / phase9-t1-development-v1` |

The attempt name is **deterministic**. There is no timestamp, no UUID, no random
suffix, no automatic retry, no `recovery1` / `recovery2`, no fresh-seed attempt
and no alternate run root.

**The claim directory is consumed once created.** A failure after the claim
requires documented human review. No automatic recovery identity is predeclared,
because predeclaring one is how a second attempt becomes reachable without a
human deciding it should be.

---

## 2. Future public CLI contract

The canonical command shape is frozen as:

```
/home/AI_POC/venvs/tactics/bin/python \
  -m cardiosentinel.neural.t1_development_run \
  --execute-canonical-development \
  --expected-git-sha <HUMAN_AUTHORIZED_MERGED_HARNESS_SHA>
```

**This command is not implemented and not executed by this specification.**
`t1_development_run.py` belongs to the next PR.

The future CLI must expose **exactly two** options: the canonical execution flag
and the expected Git SHA. It must expose **no scientific knob**:

`--q-watch` · `--q-event` · `--profile` · `--threshold` · `--p-watch` ·
`--s-watch` · `--p-event` · `--s-event` · `--subject` · `--fold` · `--retry` ·
`--force` · `--seed` · `--bootstrap` · `--test` · `--router`

are all forbidden. A scientific choice reachable from a command line is a
scientific choice a human can make after seeing results, which is the one thing
the prospective design exists to prevent.

**The exact merged Git SHA is the human authorization mechanism.** The future
harness merge SHA is not known when this document is written, so no SHA is
frozen here as the execution SHA. The specification-PR SHA is explicitly **not**
the future execution SHA.

---

## 3. Canonical interpreter and runtime integrity

Canonical interpreter: `/home/AI_POC/venvs/tactics/bin/python`.

Expected frozen dependency digest:
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

Reuse the existing runtime-integrity sentinel
(`src/cardiosentinel/neural/runtime_sentinel.py`,
`docs/RUNTIME_INTEGRITY_SENTINEL_V1.md`, SHA-256
`cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075`). No package
install, upgrade, downgrade, automatic repair or alternate interpreter is
permitted.

Enforcement is required at **minimum** at these points, in this order:

1. `START`
2. pre-label-blind-input-promotion
3. pre-each-fold-selection-promotion
4. pre-held-out-evidence-promotion
5. pre-OOF-result-promotion
6. pre-final-configuration-promotion
7. pre-experiment-lock-promotion
8. `COMPLETION`

The later harness may refine enforcement mechanics. It **may not weaken them**,
and may not remove a point from this list.

---

## 4. Pre-claim choreography

**Permitted before the claim:**

- verify the expected Git SHA and a clean working tree;
- validate the T1 protocol document digest;
- validate this execution-specification document digest;
- validate the immutable M2 retention decision;
- validate the immutable U1 retention decision and canonical attempt identity;
- validate the immutable T2 retention decision;
- prove TEST is unopened;
- prove the canonical T1 attempt directory is absent;
- inspect small immutable provenance metadata JSON.

**Forbidden before the claim:**

- opening the M2 per-row NPZ;
- opening the T2 per-row NPZ;
- reading any calibration input score;
- reading any T1 label;
- opening any target-family per-row member;
- constructing a state trace.

**The run directory is the scientific claim.** Real per-row T1 development
evidence is opened only after the claim exists. A run that read the timeline and
then declined to claim would be an unrecorded look at the data.

---

## 5. Upstream verification after the claim

The exact frozen upstream chain is verified first, reusing the existing
canonical verifiers. No weaker parallel verifier may be written.

**M2** — reuse `validate_retained_m2_arm(...)`. Require the retained arm `M2-G`
and the exact retention decision bound by the T1 protocol
(`da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47`).

**U1** — reuse the canonical U1 verifier and `validate_u1_retention_decision(...)`
/ `validate_retained_u1_calibration(...)`. Require:

- retained family `platt_logistic_on_recovered_logit`;
- OOF probabilities retained for development;
- the all-VALIDATION deployment calibrator **forbidden** for development on these
  twelve subjects;
- U1 rerun prohibited.

**T2** — reuse `validate_retained_t2_arm(...)`. Require:

- retained arm `causal_s4d_longitudinal_v1`;
- score semantics `uncalibrated_temporal_model_score`;
- `supports_t1_without_rerunning_outer_validation = true`;
- `threshold_is_t1_policy = false`;
- TEST unopened.

---

## 6. Label-blind full-timeline assembly

Exactly one immutable **label-blind** input evidence timeline is constructed
before any T1 policy-selection label is opened.

| | |
|---|---|
| Full rows | **492,904** |
| Expected scored rows | **492,898** |
| Expected unavailable rows | **6** |

### M2 source

The retained M2-G evidence store `row_evidence.npz`, whose frozen columns are
`stable_id`, `record_id`, `channel_index`, `start_sample`, `available_time`,
`score`, `scored`, `update_admitted`.

T1 uses `stable_id`, `record_id`, `channel_index`, `start_sample`, `score`,
`scored`. **`update_admitted` is not a T1 transition feature** — it is an M2 gate
outcome, and the protocol names it in `T1_FORBIDDEN_TRANSITION_INPUTS`.

### T2 source

The selected S4D score file plus **only** the identity members needed to align
the timeline.

**CRITICAL — the T2 identity NPZ carries evaluation annotation.** Its frozen
columns are:

```
stable_id, record_id, channel_index, start_sample, subject_id,
target_family, cold_start_bin, observation_state, score_present,
primary_mask, label
```

`target_family`, `primary_mask` and `label` **must not be materialized** during
label-blind assembly.

**The existing convenience readers may NOT be used at this stage.**
`read_t2_outer_row_group(...)` materializes *every* column named in the manifest
entry, and `selected_arm_scores(...)` calls it. Either would silently pull
`label` and `target_family` into memory during what is supposed to be a
label-blind step. The future harness must use a **member-restricted** reader that
names the members it materializes.

`np.load` may be opened lazily, but only named permitted members may be
materialized. Constructing `dict(np.load(...))` — or any whole-NPZ
materialization — is forbidden by design.

**Permitted identity members during label-blind assembly:**
`stable_id`, `record_id`, `channel_index`, `start_sample`, `score_present`,
`subject_id` (identity verification only), `observation_state` (physical
availability proof only).

**Forbidden identity members during label-blind assembly:**
`target_family`, `label`, `primary_mask`, `cold_start_bin`.

`cold_start_bin` is derived from recording age rather than from evaluation
annotation, so it is not forbidden on annotation grounds. It is nonetheless
excluded here because T1 derives its own cold-start condition from physical
sample coordinates (§8 of the protocol, §11 below), and a member that is not
needed is one more thing that would have to be justified. It may be joined later
for stratified reporting.

---

## 7. Subject identity authority

Subject identity is required **only** to choose the already-fitted U1 LOSO
calibrator. **It is not a transition feature.** Patient identity selects a state
namespace and a calibrator; it is never predictive.

Use the repository's frozen `subject_id_for_record(record_id)`
(`src/cardiosentinel/data/ltstdb.py`) and verify membership against
`T1_VALIDATION_SUBJECTS`.

Subject identity must not be derived from a label, a target family, an episode or
an outcome.

If the T2-persisted `subject_id` member is read for an identity cross-check,
require **exact agreement** with the canonical record-to-subject authority. A
disagreement is a hard STOP, not a preference.

---

## 8. Full-timeline U1 calibration without refit

Read the immutable U1 fold manifest
(`cardiosentinel-runs/phase7-u1-development-v1/u1-v1-development/U1_FOLD_MANIFEST.json`).

Require: 12 folds, leave-one-subject-out design, exactly one held-out subject per
fold, 11 fit subjects per fold, and the retained family
`platt_logistic_on_recovered_logit`.

For each subject, reconstruct **only** its already-fitted Platt calibrator from
the persisted fold parameters `a`, `b`, `clamp_delta`.

Reuse the existing U1 arithmetic — `U1Calibrator.apply_to_scores(...)`, or the
exact existing read-only equivalent. **Do not call** `fit_calibrator`,
`scipy.optimize`, `minimize`, family selection, or U1 development execution.

Applying an already-fitted held-out-subject calibrator to that subject's full
timeline is deterministic arithmetic on frozen parameters. It is **not** a fit,
**not** a family reselection, and **not** a U1 rerun.

For every scored M2-G row:

| Quantity | Definition |
|---|---|
| `d_t` | M2-G raw score `>= 0.7554003000259399` |
| `p_t` | that subject's already-fitted held-out LOSO Platt map applied to the raw M2-G score |
| `u_t` | `1 - p_t` if `d_t` is positive, else `p_t` |
| `s_t` | the retained S4D continuous score |

No target label and no target family participates in any of these calculations.
Crucially, T1 must **not** use `target_family` to decide whether calibrated
evidence exists — that would make a runtime transition depend on evaluation
annotation, and neither exists on a live stream.

---

## 9. Availability alignment

Require **exact** row-order and `stable_id` equality across the M2-G full replay
and the T2 retained evidence, and **exact** equality of M2 `scored` with T2
`score_present`.

Expected: **492,898 true, 6 false.**

A mismatch is a hard STOP. It is not reconciled, re-indexed or repaired.

For an unavailable row the M2 raw score is absent, and therefore `d_t`, `p_t`,
`u_t` and `s_t` are all absent. State is later carried unchanged and all streaks
reset. **No synthetic score is ever produced.**

---

## 10. Label-blind input evidence store

One immutable label-blind T1 input store, persisting **compact typed arrays**,
not hundreds of thousands of Python row objects.

Preserved at minimum:

```
stable_id, record_id, channel_index, start_sample, subject_id, score_present,
m2g_detector_score, detector_decision_d_t,
oof_calibrated_probability_p_t, decision_error_uncertainty_u_t,
s4d_temporal_evidence_s_t,
elapsed_stream_seconds
```

plus enough physical identity to verify stream boundaries.

It must contain **no** `label`, **no** `target_family`, **no** episode identity,
**no** challenge identity and **no** TEST field.

---

## 11. Elapsed time semantics

`elapsed_stream_seconds` is frozen from **physical sample coordinates**, never
from a row ordinal:

```
elapsed_stream_seconds = (start_sample - first_start_sample_of_stream) / 250.0
```

A row index does not imply physical time, and assuming it would silently
mis-date every stream containing a gap.

State elapsed time is **bookkeeping only** unless the frozen transition protocol
explicitly consumes it. It must never create a new transition condition.

The emitted state is the state **after** processing the current row.
State duration is persisted under an unambiguous field name
(`state_elapsed_seconds`) so no reader has to guess the convention.

`next_state()` semantics are not changed.

---

## 12. The fold-scoped label firewall

**This is the most important execution rule in this document.**

For fold *k*, the held-out subject's labels, target families and episode truth
**must not be opened** until that fold's policy has been selected and immutably
promoted.

- **Do not** load all VALIDATION target metadata once at run start.
- **Do not** use the T2 identity NPZ's `label` or `target_family` arrays as a
  convenience global label table.

Instead, build a **fold-scoped target-authority loader**.

Before policy selection, open target metadata **only for the 11 FIT subjects**,
using the same frozen LTSTDB feature-corpus authority already used by T2:
persisted record metadata only, no waveform reread, no raw annotation reread, no
`.stb` reinterpretation, no feature matrix unless an existing authority contract
requires it, and no context-derived label fabrication.

Only **after** the fold's selected-policy artifact has been promoted and re-read
successfully may the held-out subject's evaluation labels and target-family
metadata be opened.

---

## 13. FIT-subject threshold generation

For each fold, after opening FIT labels only, derive the four thresholds for
every candidate using the frozen protocol helpers.

Population: **FIT subjects only, PRIMARY, background negatives only.**

Use exactly `Q_WATCH = (0.90, 0.95)`, `Q_EVENT = (0.99, 0.995)` and
`empirical_order_statistic(...)`. **Do not use `numpy.quantile`** or any other
implementation: a library quantile interpolates between neighbours and is not
reproducible across versions.

Thresholds are generated separately for the `p_t` and `s_t` distributions.
No challenge rows. No held-out row. No weighting. No interpolation.

---

## 14. FIT policy evaluation

Exactly **12 policies per fold**.

For each policy, run the frozen `next_state(...)` causally over the complete
label-blind FIT-subject timelines. All runtime context rows remain in the
timeline.

Labels are used **only after** state traces exist, and then only to derive
reference PRIMARY episodes, match EVENT runs, and compute selection metrics.

Challenge identity is never a transition input and never a selection input.

No policy may modify the threshold grid, the persistence profile, the cold-start
rule, the state machine or the matching rule.

---

## 15. Exact FIT policy selection

Reuse the frozen selection order (`policy_sort_key`):

1. pooled episode F1 (higher wins)
2. pooled PRIMARY window MCC (higher wins)
3. false EVENT onsets per physical hour (lower wins)
4. EVENT exposure fraction (lower wins)
5. higher `q_event`
6. higher `q_watch`
7. CONSERVATIVE before BALANCED before FAST

Tolerance `1e-6`. No weighted score, no challenge evidence, no latency.

Episode F1 is count-algebra based: `2·TP / (2·TP + FP + FN)`. **If its
denominator is zero it is undefined.** An undefined quantity is preserved as
undefined and is **never silently converted to zero**.

If a selection stage would actually have to compare an undefined quantity in a
way the frozen ordering cannot resolve: **STOP FOR HUMAN REVIEW.**

Where the repository already has metric semantics for numerical or undefined
cases, reuse them rather than inventing a parallel convention.

---

## 16. The fold-selection promotion barrier

For each fold, persist an immutable fold-selection artifact **before** opening
its held-out labels. The artifact binds at minimum:

fold index · held-out subject identity · FIT subject identities · FIT
label-authority identity · the 12 candidate policy identities · generated
thresholds per candidate · FIT candidate selection metrics · selected policy ·
selected thresholds · selection path and tie-break stage · T1 protocol SHA · T1
execution-spec SHA · upstream input-evidence-store SHA · `test_accessed = false`.

After promotion, **re-read the artifact and verify its digest.** Only then may

```
held_out_label_access_authorized_for_this_fold = true
```

be set.

**This barrier must be structural in the future harness, not prose.** A rule that
exists only in a document is a rule the code can forget.

---

## 17. One held-out evaluation per fold

After fold selection is frozen, open only that one held-out subject's target and
evaluation metadata. Then run the **one already-selected policy once** over that
subject's existing label-blind input evidence.

Forbidden:

- running the 11 rejected candidate policies on the held-out subject;
- deriving a new threshold;
- changing persistence;
- changing cold-start logic;
- reselecting after seeing held-out results;
- retrying the fold.

Persist the held-out state trace and evaluation evidence **once**. There is no
fold retry.

---

## 18. OOF state-evidence store

After all 12 folds succeed, concatenate exactly **one** held-out selected-policy
trace per subject. This is the sole T1 **development** state evidence.

Persist one full-timeline OOF state store containing at minimum:

```
stable_id, record_id, channel_index, start_sample, subject_id, score_present,
m2g_detector_score, detector_decision_d_t, oof_calibrated_probability_p_t,
decision_error_uncertainty_u_t, s4d_temporal_evidence_s_t,
fold_index, selected_policy_id,
p_watch, s_watch, p_event, s_event,
emitted_state, state_elapsed_seconds,
transition_from, transition_to, transition_occurred
```

No TEST.

Labels and target families are preferably kept **outside** this reusable
downstream state store, so the future routing layer can consume it without
inheriting an evaluation-label dependency.

---

## 19. OOF development result

From the cross-fitted held-out traces, report exactly the protocol-defined T1
development evidence.

**Episode:** reference episode count · predicted EVENT run count · matched count ·
episode precision · episode sensitivity · episode F1.

**Onset:** matched onset latency — median, IQR, p90.

**PRIMARY window:** F1 · sensitivity · specificity · PPV · NPV · balanced
accuracy · MCC.

**State burden:** NORMAL / WATCH / EVENT / RECOVERY fractions · transitions per
hour.

**State flow counts:** `NORMAL→WATCH` · `WATCH→EVENT` · `WATCH→NORMAL` ·
`EVENT→RECOVERY` · `RECOVERY→EVENT` · `RECOVERY→NORMAL`.

**Descriptive:** overmerged EVENT runs · reference episodes split across
predicted runs.

**No categorical-state AUPRC.** A four-state categorical trace has no threshold
to sweep.

---

## 20. Physical exposure and false-onset semantics

Exposure is frozen from the **physical** timeline: one timeline position
represents one 5-second stride of physical exposure.

**Unavailable positions are included** in physical exposure, because time passed
and state was carried.

| Quantity | Definition |
|---|---|
| False EVENT onset numerator | unmatched predicted EVENT runs |
| False EVENT onsets per hour | unmatched predicted EVENT runs / physical exposure hours |
| EVENT exposure fraction | positions emitted as EVENT / all physical positions |

**Do not silently use PRIMARY-only time as physical exposure.** PRIMARY is an
evaluation population; physical exposure is wall-clock the patient actually spent
being monitored.

---

## 21. Subject evidence and bootstrap

**Subject is the inferential unit.**

After OOF evidence is complete: **1000** bootstrap replicates, seed **2026**,
sampling unit **subject**, policy reselection inside the bootstrap **NO**.

Subjects are resampled **with multiplicity**. When a subject appears more than
once in a replicate, its already-frozen OOF trace contributes with multiplicity.
No fold is rerun and no policy is re-derived.

Claim scope: **between-subject variation conditional on the cross-fitted T1
development procedure.** Undefined replicate counts are preserved, never silently
replaced.

---

## 22. Challenge reporting

Challenge family identity may be joined **only after** state traces exist.

RATE and AXIS are quantitative secondary. CONDUCTION is exploratory descriptive
only.

Per family, report: row count · WATCH row count and fraction · EVENT row count
and fraction · EVENT onsets occurring on challenge rows.

Challenge family is **not** a transition input, **not** a threshold-generation
input, and **not** a policy-selection input.

---

## 23. Final all-VALIDATION configuration

**Only after** the complete OOF result has been promoted and verified, perform
the protocol-defined final all-VALIDATION configuration selection: all 12
subjects, the same 12 candidates, the same threshold-generation rules, the same
metrics, the same selection order.

Persist `q_watch`, `q_event`, `p_watch`, `s_watch`, `p_event`, `s_event` and the
persistence profile.

This is **deployment/test configuration only**. Its in-sample all-VALIDATION
performance is **not** T1 development evidence, and it must never replace or
overwrite the OOF result.

---

## 24. Persistence and artifact plan

Future canonical artifacts, approximately:

```
T1_RUN_STATUS.json
T1_PREFLIGHT.json
T1_INPUT_LINEAGE.json
T1_INPUT_EVIDENCE.json            + typed NPZ arrays
T1_FOLD_SELECTIONS.json           (or immutable per-fold selection artifacts)
T1_OOF_STATE_EVIDENCE.json        + typed NPZ arrays
T1_OOF_RESULT.json
T1_SUBJECT_EVIDENCE.json
T1_BOOTSTRAP.json
T1_CHALLENGE_EVIDENCE.json
T1_FINAL_CONFIGURATION.json
T1_RESULT.json
T1_EXPERIMENT_LOCK.json
```

**None of these real artifacts is created by this specification PR.**

The future persistence implementation must write atomically; never overwrite
promoted evidence; bind file and content digests; bind upstream artifact
identities; bind the T1 protocol and specification identities; bind Git and
runtime identity; record the exact label-access choreography; and record TEST
unopened.

---

## 25. Failure semantics

A **pre-claim** refusal leaves the attempt unconsumed.

A **post-claim** failure **consumes** the attempt. There is no automatic retry.

The failure receipt records at minimum: stage · current fold · whether
label-blind input was opened · which folds' FIT labels were opened · which fold
selections were promoted · which folds' held-out labels were opened · which
held-out traces completed · whether OOF evidence was promoted · whether final
configuration started and completed · TEST state · runtime-integrity state ·
exception type and message.

**No failed attempt is deleted or rewritten to look clean.**

---

## 26. TEST firewall

Absolute.

The future T1 execution package exposes **no TEST option**. Before any path
resolution it refuses `partition == "test"`, TEST subjects, TEST result paths,
TEST target metadata, TEST upstream row evidence and TEST metrics.

Every artifact records `test_accessed = false` and
`sealed_test_state = "unopened"`.

No TEST work occurs in this specification PR.

---

## 27. The transition view versus the persisted evidence row

The frozen protocol's `T1_ALLOWED_ROW_INPUTS` is the **permission list** — what a
T1 row may legitimately carry. The frozen `T1Row` passed to `next_state()` is a
narrower **transition view** — what the state machine actually consumes.

They differ, and the difference is intentional:

| | |
|---|---|
| In the permission list, not in the transition view | `m2g_detector_score`, `elapsed_state_seconds` |
| Same quantity, different naming convention | `detector_decision_d_t` ↔ `detector_decision`; `oof_calibrated_probability_p_t` ↔ `calibrated_probability`; `decision_error_uncertainty_u_t` ↔ `decision_error_uncertainty`; `s4d_temporal_evidence_s_t` ↔ `temporal_evidence` |

This specification therefore distinguishes:

- **A — the full assembled and persisted T1 evidence row**, which may preserve the
  raw M2-G score and state-duration bookkeeping; and
- **B — the minimal frozen transition view consumed by `next_state()`**.

A permission list that is a superset of what the transition function consumes is
a permission list, not a contradiction. **Preserving a field in evidence does not
make it a transition condition**, and the T1 protocol must not be edited to make
either field predictive.

---

## 28. Frozen stage order

```
 1  start / runtime integrity START
 2  verify expected git sha and clean tree
 3  validate T1 protocol document
 4  validate T1 execution specification document
 5  validate M2 retention decision
 6  validate U1 retention decision and canonical attempt
 7  validate T2 retention decision
 8  prove TEST unopened
 9  prove canonical T1 attempt absent
10  claim run directory                          <-- the scientific claim
11  verify upstream chain after claim
12  assemble label-blind full timeline           <-- first per-row access
13  promote label-blind input evidence store
14  per fold k in 0..11:
      a  open FIT-subject target metadata only
      b  generate 12 candidates' thresholds from FIT background negatives
      c  run next_state for 12 policies over FIT timelines
      d  select one policy by the frozen order
      e  promote the fold-selection artifact and re-read its digest
      f  authorize held-out label access for this fold ONLY
      g  open the held-out subject's target metadata
      h  run the one selected policy once over the held-out timeline
      i  promote the held-out trace and evaluation evidence
15  concatenate 12 held-out traces -> OOF state evidence store
16  promote OOF development result
17  subject evidence and bootstrap
18  challenge reporting join
19  final all-VALIDATION configuration
20  promote experiment lock
21  runtime integrity COMPLETION
```

Stage 12 is the first per-row access and never precedes stage 10. Stage 14f never
precedes stage 14e for the same fold. Stage 19 never precedes stage 16.
