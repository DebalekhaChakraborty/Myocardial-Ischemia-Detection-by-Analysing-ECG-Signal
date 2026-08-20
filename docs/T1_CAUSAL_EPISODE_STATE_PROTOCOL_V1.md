# T1-v1 causal episode-state protocol

This protocol is frozen **before** any T1 development run, any candidate
threshold is computed, and any T1 state trace exists. It is written so a future
implementer can build the T1 execution harness without making a single new
scientific choice.

T1 converts already-existing causal evidence into an interpretable, hysteretic
system state: **NORMAL / WATCH / EVENT / RECOVERY**.

T1 is **not** another neural model, not another temporal model, not another
calibrator, not a post-hoc smoother, not a relabelled T2 binary threshold, not
an LLM, and not an edge/cloud router. It is a deterministic causal state machine
over frozen upstream evidence.

## 0. Three things that must never be confused

This distinction governs the whole protocol.

| | What it is | Where it may be used |
|---|---|---|
| **STATE-MACHINE INPUT** | `d_t`, `p_t`, `u_t`, `s_t`, availability, elapsed times | The transition function, at runtime |
| **DEVELOPMENT LABEL** | PRIMARY ischemic-positive truth, reference episodes | Threshold generation on FIT subjects, policy selection, evaluation — **never** the transition function |
| **REPORTING-ONLY CHALLENGE IDENTITY** | rate / axis / conduction family | Post-hoc stratified reporting — **never** the transition function, **never** policy selection |

A rule that read a label or a target family at runtime would not be deployable:
neither exists on a live ECG stream. Every runtime input below is available
label-blind.

## 1. Frozen upstream lineage

T1 starting master `b3004da9dcd8e7462d69eac81eb82ca9da86b8cb`.

| Upstream | Identity |
|---|---|
| M2 retention decision | `da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47` |
| M2 retained arm | `M2-G` |
| U1 protocol | `d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b` |
| U1 retention decision | `9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b` |
| U1 retained OOF evidence store | `b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba` |
| T2 protocol | `6546086a55fe2c9c109f4121cdb6b42d4d53ce0112c9611eb895bd8c805cfefb` |
| T2 retention decision | `4846921135b0ac83ceb40a0db063c2e4a3b2520971f279abe4f0c517c4f7dd20` |
| T2 retained arm | `causal_s4d_longitudinal_v1` |
| T2 outer result | `c58ed40dac753157b00ce6c70eb52fe903ecee72a5ef84e40932c1a80e259dbf` |
| T2 selected row evidence | `2240ca683fbcb790609c47f4a82af85250abb281fbbb9751dc74607a4eb591ca` |
| Split | `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7` |
| Feature corpus | `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5` |

## 2. The full-timeline input

T1 development runs over the complete chronological VALIDATION timeline:
**492,904 rows / 30 streams / 12 subjects**, stream key
`(record_id, channel_index)`, ordered by `start_sample`, window 2500 samples
(10 s), stride 1250 samples (5 s) at 250 Hz.

The timeline identity comes from the already-retained T2 row-evidence lineage.
**No timeline is regenerated from waveform data.**

## 3. Full-timeline out-of-fold calibration, without refitting

This is the key T1-v1 design rule, and the reason T1 can run label-blind.

The persisted U1 OOF evidence store holds metric and challenge rows. T1 must
**not** use `target_family` to decide whether calibrated evidence exists for a
row — that would make a runtime state transition depend on evaluation annotation
identity, which does not exist in deployment.

Instead, at execution time:

1. read the retained **M2-G full-replay `score`** values read-only;
2. identify the subject through the frozen split / record-subject authority;
3. select **that subject's already-fitted U1 LOSO Platt calibrator**;
4. apply it to **every scored M2-G row of that subject**, regardless of target
   family;
5. use the exact frozen U1 recovered-logit transformation and clamp
   (`clamp_delta = 1e-7`);
6. fit, optimise or adjust **nothing**;
7. never use a row's label or target family to compute its probability.

This yields `p_t`, a full-timeline out-of-fold Platt-calibrated probability, for
every physically scored VALIDATION row.

For development on these twelve subjects the **all-VALIDATION deployment
calibrator is forbidden** — it was fitted on every one of them, so using it
would leak the held-out subject into its own probability. Only the
subject-disjoint fold calibrator is permitted.

Extending an already-fitted calibrator from metric rows to the held-out
subject's full timeline is deterministic arithmetic. It is **not** a new
calibration fit, **not** a family reselection, **not** new calibration evidence,
and **not** a U1 rerun.

## 4. Frozen row inputs

For each physically scored row `t`, the transition function may consume exactly:

1. `stable_id`
2. retained M2-G raw detector score
3. detector decision `d_t` at the frozen operating point **0.7554003000259399**
4. full-timeline OOF calibrated probability `p_t`
5. decision-error uncertainty `u_t` = `1 - p_t` if `d_t` is positive, else `p_t`
6. retained S4D continuous temporal evidence `s_t`
   (`uncalibrated_temporal_model_score`)
7. physical availability / score-present state
8. elapsed causal stream time
9. elapsed current-state time

It must **never** consume: label, target family, subject outcome, episode
identity, any future row or score, the GRU score, an S4D binary decision, the T2
frozen reporting threshold, `u_star_dev`, `u_star_deploy`, challenge-family
identity, the M2 gate outcome, `m2_update_admitted`, or any TEST-derived
quantity.

M2-G remains the upstream adaptation lineage; its gate/admission decision is
deliberately **not** double-counted as a second T1 predictive variable in V1.

## 5. Availability contract

Exact stable-ID alignment is required among the M2-G full replay, the retained
T2 row evidence and the frozen timeline identity.

| | |
|---|---|
| Full rows | 492,904 |
| Score-present rows | 492,898 |
| Unavailable rows | 6 |

M2 and T2 physical score availability must reconcile **exactly**. If one source
says a row is scored and the other says it is not: **STOP FOR HUMAN REVIEW.**

For an unavailable row: no `p_t`, `u_t` or `s_t` is invented; the current state
is carried unchanged; state elapsed physical time advances by one 5-second
stride; **all confirmation streaks reset**; and neither an upward nor a downward
transition may fire. No imputation, no forward fill, no synthetic zero.

## 6. State space

Exactly four states: NORMAL, WATCH, EVENT, RECOVERY. Every stream starts in
**NORMAL**. State never crosses a record, channel or subject boundary. T1-v1 is
**per-stream**; patient-level multi-channel fusion is not defined here.

## 7. Prospective threshold generation

No absolute probability or S4D threshold is chosen by hand. The future
development run generates candidates using **only the FIT subjects of each fold**
and **only PRIMARY background-negative rows**.

Frozen quantile levels: `Q_WATCH = (0.90, 0.95)`, `Q_EVENT = (0.99, 0.995)`.

The same level is applied separately to the `p_t` and `s_t` distributions,
producing `p_watch`, `s_watch`, `p_event`, `s_event`.

**Exact empirical order statistic**: sort ascending by `(value, stable_id)`;
with `N` rows and `k = ceil(q * N)` (1-based), the threshold is the value at
position `k`. No library quantile, no interpolation, no smoothing, no label
weighting, no challenge rows, no TEST.

## 8. Three frozen persistence profiles

| | FAST | BALANCED | CONSERVATIVE |
|---|---|---|---|
| `watch_clear_windows` | 2 | 3 | 6 |
| `event_confirm_windows` | 2 | 3 | 6 |
| `event_release_windows` | 2 | 3 | 6 |
| `re_event_confirm_windows` | 1 | 2 | 3 |
| `recovery_clear_windows` | 3 | 6 | 12 |
| `cold_event_confirm_windows` | 4 | 6 | 12 |

WATCH entry is immediate after one WATCH-evidence row. There is no fourth
profile and no post-result duration search.

**Candidate policies per fold: 2 × 2 × 3 = 12.**

## 9. Evidence definitions

For a physically available **mature** row:

- **WATCH evidence**: `d_t == True` **OR** `p_t >= p_watch` **OR** `s_t >= s_watch`
- **EVENT evidence**: `d_t == True` **AND** `p_t >= p_event` **AND** `s_t >= s_event`
- **NORMAL evidence**: `d_t == False` **AND** `p_t < p_watch` **AND** `s_t < s_watch`

A row that is neither EVENT nor NORMAL evidence is ambiguous / WATCH-level.

There is no weighted fusion score and no learned fusion model.

## 10. Uncertainty semantics

`u_t` is preserved explicitly in the T1 evidence contract, and **no independent
T1 uncertainty threshold is created**. When `d_t` is positive, `p_t >= p_event`
is equivalently `u_t <= 1 - p_event` — so EVENT confirmation already requires
calibrated low decision-error evidence without a redundant tunable parameter.

`u_t` must remain available in persisted T1 row evidence for the later
state-aware routing phase. The rejected U1 symmetric router remains rejected.

## 11. Cold start

T2's own outer evidence recorded **zero** thresholded sensitivity in the first
0–5 minutes. T1 addresses that prospectively, without modifying T2.

Stream age is defined from sample coordinates. **Cold start is `age < 300 s`.**

During cold start, EVENT evidence is `d_t == True` **AND** `p_t >= p_event` —
the S4D event threshold is **not** required — and confirmation takes
`cold_event_confirm_windows`. WATCH evidence is unchanged.

At `age >= 300 s` the mature EVENT rule applies, requiring detector **and**
calibrated probability **and** S4D temporal evidence.

This is **not** a T2 repair: no T2 state is modified and no alternative S4D
initialization is introduced.

## 12. Transition table

All confirmation counters count **consecutive physically available** rows
satisfying their named condition; a row that fails the condition resets that
counter. Escalation has priority. A state change clears every counter.

| From | Condition | To |
|---|---|---|
| NORMAL | no WATCH evidence | NORMAL |
| NORMAL | WATCH evidence | WATCH (immediate) |
| NORMAL | EVENT evidence × confirm budget | EVENT |
| WATCH | EVENT evidence × confirm budget | EVENT |
| WATCH | NORMAL evidence × `watch_clear_windows` | NORMAL |
| WATCH | otherwise | WATCH |
| EVENT | EVENT evidence | EVENT (release streak reset) |
| EVENT | NORMAL evidence × `event_release_windows` | RECOVERY |
| EVENT | ambiguous (not NORMAL evidence) | EVENT (no release contribution) |
| RECOVERY | EVENT evidence × `re_event_confirm_windows` | EVENT |
| RECOVERY | NORMAL evidence × `recovery_clear_windows` | NORMAL |
| RECOVERY | otherwise | RECOVERY |
| any | unavailable row | unchanged, streaks reset, no transition |

**RECOVERY never automatically becomes WATCH.**

## 13. Why the T2 binary threshold is not used

The retained S4D internal-dev threshold `0.8972153067588806` must **not** appear
as `p_event`, `s_event`, a WATCH threshold, an EVENT threshold, a recovery
threshold or a routing threshold. It remains T2 experiment/reporting evidence
only. T1 uses its own prospectively generated development thresholds.

## 14. Development split

The same twelve frozen VALIDATION subjects U1 used: `ltstdb:s2004`,
`s2005`, `s2019`, `s2020`, `s2023`, `s2031`, `s2057`, `s2058`, `s2059`,
`s3068`, `s3072`, `s3073`.

**12-fold leave-one-subject-out.** For fold `k`: FIT is the other eleven
subjects, HELD-OUT is one. Candidate thresholds and policy selection use FIT
only. The held-out subject's labels, episode truth and T1 outcomes **must not be
opened until that fold's policy is completely selected and frozen**; the
selected policy is then evaluated exactly once on the held-out subject. No fold
retry and no fold-specific manual override.

## 15. Development-optimism disclosure

**T1 LOSO is subject-disjoint for T1 POLICY SELECTION only. It is not
independent end-to-end validation.**

The retained S4D arm was itself selected using the full upstream VALIDATION
population. Future T1 OOF evidence must therefore be described as **cross-fitted
T1 development evidence conditional on frozen upstream components** — never as
unseen generalization, external validation, independent validation or clinical
validation. TEST remains the only sealed LTSTDB holdout.

## 16. Reference episodes and predicted runs

Reference episodes are derived **only after** a candidate state trace exists,
from the existing frozen feature-corpus target authority. For each
`(record_id, channel_index)`, a reference ischemic episode is a **maximal**
sequence of PRIMARY ischemic-positive rows whose consecutive `start_sample`
difference is exactly **1250 samples**. Any non-PRIMARY-positive row breaks it.
No minimum-duration filter, no gap bridging, no annotation reread, no `.stb`
reinterpretation, no future repair. Onset is the first positive row's
`start_sample`; end is the last positive row's `start_sample + 2500`.

A **predicted EVENT run** is a maximal chronological run of emitted EVENT states
within one stream at the frozen cadence. No run crosses streams. An EVENT state
carried across non-primary context remains part of the same run. Unavailable
rows do not themselves create a transition.

## 17. One-to-one matching

Each predicted EVENT run matches **at most one** reference episode, and each
reference episode matches **at most one** predicted run. A match requires at
least one PRIMARY ischemic-positive row of the episode to occur while T1 is in
EVENT.

Matching is deterministic and chronological: reference episodes ordered by
onset; each takes the earliest still-unmatched overlapping predicted run. A run
spanning several episodes matches only the first — later episodes stay unmatched
unless another run detects them. **This intentionally penalizes overmerged EVENT
states.**

## 18. Policy selection

Selection on FIT subjects uses PRIMARY labels only.

Primary metric: **pooled episode F1**, where `TP` = matched predicted EVENT runs,
`FP` = unmatched predicted runs, `FN` = unmatched reference episodes; precision
`TP/(TP+FP)`, sensitivity `TP/(TP+FN)`, F1 their harmonic mean.

Complete lexicographic order, tolerance `1e-6` at each numeric step:

1. higher pooled episode F1
2. higher pooled PRIMARY window MCC (EVENT positive; NORMAL/WATCH/RECOVERY negative)
3. lower false EVENT onsets per physical hour
4. lower fraction of physical exposure in EVENT
5. higher `q_event`
6. higher `q_watch`
7. persistence profile: CONSERVATIVE before BALANCED before FAST

No challenge metric, no latency, no weighted composite, no per-fold human
override.

## 19. Out-of-fold development evidence

After all twelve folds, concatenate each subject's state trace produced by the
policy selected **without** that subject. This is the sole T1 development
performance evidence.

Reported: episode counts, predicted EVENT-run count, matches, episode precision /
sensitivity / F1; signed onset latency for matched episodes (median, IQR, p90 —
negative meaning the run began before the first labelled positive window but
eventually overlapped); PRIMARY window F1, sensitivity, specificity, PPV, NPV,
balanced accuracy, MCC; state burden fractions and transitions per hour; state
flow counts for NORMAL→WATCH, WATCH→EVENT, WATCH→NORMAL, EVENT→RECOVERY,
RECOVERY→EVENT, RECOVERY→NORMAL; and descriptively, overmerged predicted runs
and reference episodes split across multiple runs.

**No AUPRC** is reported for a categorical state output.

## 20. Subject evidence and bootstrap

Subject-level episode and window evidence is reported; windows are **not**
independent subjects. Subject-macro values are reported where mathematically
defined, and for undefined single-class quantities the contributor and excluded
counts are recorded explicitly rather than silently replaced with zero.

Frozen bootstrap: **1000 replicates, seed 2026, sampling unit subject, no
state-policy reselection inside the bootstrap.** Claim scope: between-subject
variation conditional on the cross-fitted T1 development procedure.

## 21. Cold-start and challenge reporting

Cold start is reported separately for 0–5 min, 5–60 min and >60 min: PRIMARY
EVENT-state sensitivity, specificity, episode detection where support exists,
and state burden. No post-hoc cold-start repair.

Challenge rows participate in the full **label-blind** state timeline because
their scores are available at runtime; their challenge identity is never a
transition input. After the trace exists, RATE, AXIS and CONDUCTION are
evaluated separately, reporting row count, WATCH row count/fraction, EVENT row
count/fraction and EVENT onsets occurring on challenge rows. RATE and AXIS are
quantitative secondary; CONDUCTION is exploratory descriptive only. Challenge
evidence is never a policy-selection input.

Other non-primary rows remain full causal runtime context. Their target-family
identity must not reach the transition function; because full-timeline OOF
calibration is generated label-blind from retained M2-G scores, a physically
scored other-non-primary row receives ordinary `p_t`, `u_t`, `s_t` and `d_t` and
is processed exactly like any other runtime row. This is essential for
deployment parity.

## 22. Final all-VALIDATION configuration

After cross-fitted development evidence is complete, one final T1 deployment
configuration is selected using **all twelve** subjects and the **same** grid,
threshold rules, metrics and selection order. It produces configuration only:
`q_watch`, `q_event`, the actual `p_watch`, `s_watch`, `p_event`, `s_event`, and
the persistence profile.

**Its in-sample performance is not T1 evidence.** The OOF result remains the T1
development evidence. That configuration is what a future separately-authorized
TEST or deployment run would use.

## 23. Firewalls

**TEST is sealed.** `T1_TEST_ACCESSED = False`,
`T1_SEALED_TEST_STATE = "unopened"`. The protocol module contains no TEST
reader, waveform reader, annotation reader, model inference or run-artifact
reader.

**Routing is undefined.** T1 exposes state for the later selective edge/cloud
layer but defines no network policy, cloud escalation threshold, edge capacity,
latency limit, energy limit or bandwidth policy. Later routing may prospectively
consume T1 state, `u_t`, physical availability and network/device state. The
rejected U1 symmetric router remains rejected.

**No LLM participates in state determination.** A future language model may
summarize structured evidence only; it cannot alter NORMAL, WATCH, EVENT or
RECOVERY.
