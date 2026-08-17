# T2 Causal Longitudinal Temporal Protocol V1

## 0. Nature of this document

**THIS IS A PROSPECTIVE SCIENTIFIC PROTOCOL, FROZEN BEFORE ANY T2 EXECUTION.**

It designs the next core scientific block — T2, a causal longitudinal temporal
model over successive ECG-window representations — and defines only the
*interface contract* for the later T1 deterministic state logic.

Nothing here is executed. This protocol does **not** train T2, execute T2,
implement the T2 trainer, score real VALIDATION, inspect TEST, implement T1
state logic, or choose a final edge/cloud router. The accompanying
`src/cardiosentinel/neural/t2_protocol.py` imports only the standard library,
so protocol validation cannot reach real development data or TEST even by
accident.

Upstream ladder, frozen: **B4-B → P1-B → M1L → M2-G → U1 Platt calibration**.

## 1. Retained temporal input — the audit result

The frozen causal P1-B representation `z_t` exists for **both** TRAIN and
VALIDATION, at the expected dimensionality **146 = 128 B4-B embedding + 18
retained physiology features**.

| | TRAIN | VALIDATION |
|---|---|---|
| Store | `cardiosentinel-features/m1-stream-memory-v2/train` | `.../validation` |
| Artifact class | `m1_full_stream_memory_cache` | `m1_full_stream_memory_cache` |
| Schema | 3 | 3 |
| `representation.npy` | `(2 208 431, 146)` float32 | `(492 904, 146)` float32 |
| `representation_content_sha256` | `e52a566fbc285a7a9f92715752dee43c020faa3550aaeb660f5f400dee07b5d3` | `b26a2d9b6150e6518dc2bfb394427dc93ae48a7cc3de30adcc3fefcc9f1f53ba` |
| `stream_cache_sha256` | `d006c698017110bfd95774ca207036a820139779b95cf1b3f3a36c06efa779a4` | `a3e39137a04ebebb3b97ef6c6c614339c990a6041cf649a0ba6e3c2d43baae18` |
| `p1_embedding_cache_sha256` | `0a5f021b89597d245a2afdc51fe1a65ba5cd6a090beba429f38bbccff8c372dd` | `c533db3acfdfa1057c2ac9d8e77d011d3ac5f87fc7a872399227f94f526db0c3` |
| Streams `(record_id, channel_index)` | 132 | 30 |
| Records / subjects | 60 / 56 | 13 / 12 |
| `available_row_count` | 2 208 431 | 492 898 |
| `unavailable_exact_flat_row_count` | 0 | 6 |
| `test_accessed` | false | false |
| `sealed_test_state` | unopened | unopened |

Both carry `record_id`, `channel_index`, `start_sample`, `stable_id`,
`observation_state`, `cold_start_bin` and `recording_age_seconds` — every key T2
needs for stream identity, chronological ordering, availability semantics and
the inherited cold-start strata.

Shared upstream provenance across both partitions: `split_sha256`
`66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`,
`p1_protocol_sha256` `f48ffc66…`, `p1_retention_decision_sha256` `7b403709…`,
`p1b_experiment_lock_sha256` `796f00e3…`, `b4_protocol_sha256` `f6f5e9ed…`,
`encoder_checkpoint_sha256` `b1301723…`, `physiology_schema_sha256`
`13f60be4…`, `physiology_transform_sha256` `cc6bd3a3…`, and
`environment_dependency_digest`
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

The VALIDATION `stream_cache_sha256` `a3e39137…` is the same stream cache the
frozen M2-G evidence binds, and the VALIDATION `p1_embedding_cache_sha256`
`c533db3a…` is the one U1's deployment calibrator binds. The lineage from T2's
input back through U1 and M2 is therefore exact, not asserted.

### 1.1 The P1 embedding cache is NOT the T2 source — and why that matters

`cardiosentinel-features/p1-b4b-embeddings-v1/train` holds **374 452 rows at
exactly 3:1 negative sampling** (280 839 negative = 3 × 93 613 positive), and
its manifest records a `training_selection_sha256`. It is a *selection*, not a
timeline. Its VALIDATION counterpart is unsampled (473 897 rows,
`training_selection_sha256: null`), so the two sides are not the same kind of
object.

It also carries only the 128-dimensional embedding, with no physiology block and
no `record_id` / `channel_index` / `start_sample` ordering keys.

Using it for T2 would silently destroy temporal continuity on TRAIN — the exact
failure §9 of the authorization forbids — while leaving VALIDATION intact, which
would make the comparison incoherent as well as wrong. **The M1 full stream
memory cache is the only admissible `z_t` source.**

## 2. Scientific role

T2 answers one question:

> Can causal longitudinal modelling of successive ECG-window representations
> improve current-window ischemic evidence and temporal consistency compared
> with the frozen non-temporal system?

Its output is a **causal temporal evidence score for the current window**.

T2 is **not** the T1 state machine, not an episode rule engine, not a router,
not a calibrated uncertainty model, not a replacement for M1/M2, and not a
future-aware smoother.

## 3. Separation from B4-C — different temporal scale, different question

B4-C was an intra-window S4D-inspired sequence model operating **inside** a
single 10-second ECG window, and it was not retained.

T2 operates **across successive windows**: `z_1, z_2, z_3, …, z_t`, where each
`z` is the frozen 146-dimensional P1-B representation of one 10-second window
emitted every 5 seconds.

**The B4-C rejection does NOT imply that state-space models are rejected.** B4-C
tested whether intra-window sequence structure improved a single-window
representation. T2 tests whether across-window temporal structure improves
current-window evidence. Different temporal scale, different scientific role,
different question. Reusing the B4-C result as an argument against T2-B would be
a category error.

## 4. Input contract

The **only** trainable T2-v1 input is `z_t`, the frozen 146-dimensional P1-B
representation. The B4-B encoder is not fine-tuned
(`encoder_fine_tuned: false`); the representation is consumed frozen.

The following may **not** enter the trainable T2-v1 model:

- U1 OOF calibrated probability;
- U1 uncertainty;
- `u_star_dev`; `u_star_deploy`;
- labels from future windows;
- challenge-family identity;
- episode identity; future episode duration;
- M2 gate outcome;
- any TEST-derived quantity.

**Rationale.** VALIDATION has subject-disjoint OOF Platt probabilities; TRAIN has
no equivalent already-frozen subject-disjoint calibration product. Feeding U1
outputs to T2 would create a train/validation feature-definition mismatch
invented purely to make T2 consume U1. U1 calibration remains retained and will
be available to the later T1 / fusion layer, where both sides can be defined
consistently.

## 5. Availability semantics

The stream unit is `(record_id, channel_index)`, ordered by
`window_start_samples`.

At a physically unavailable exact-flat observation
(`observation_state == 2`, `UNAVAILABLE_EXACT_FLAT`):

- **no** synthetic `z_t` is created;
- **no** imputation;
- **no** forward fill;
- the row is **not** scored;
- its target is **not** used for training or evaluation;
- the learned temporal hidden state is **not** updated from it.

The temporal state carries causally **across** the unavailable observation, and
the timeline position still advances. The row is a gap in evidence, not a gap in
time.

This is the frozen M1 physical-observation contract, inherited verbatim. **No
new near-flat or SQI threshold is invented here.** TRAIN currently contains 0
such rows and VALIDATION contains 6.

## 6. Sequence boundaries

T2 hidden state **MUST reset at every new `(record_id, channel_index)` stream**.

State never crosses recordings, channels or subjects. Within a stream, strict
chronological order only: no bidirectional model, no shuffled time, no future
context.

## 7. Label / target

T2 predicts the **current** PRIMARY target `y_t`, from the same frozen LTSTDB
`.stb` target authority used upstream.

The target at time `t` is only the target of the current window. `y_(t+1)`,
future episode membership, future onset/offset, episode duration and future
challenge labels are never model inputs. Challenge rows are never training
targets.

## 8. Internal TRAIN split — deterministic, identity-only

The 12-subject outer VALIDATION partition is **not** repeatedly tuned against.
From the 56 TRAIN subjects, one deterministic subject-disjoint internal split is
frozen:

- **48 subjects** — T2 model fitting;
- **8 subjects** — T2 internal development, early stopping, checkpoint selection.

Algorithm `sha256_identity_ranked_subject_partition_v1`: rank the TRAIN subjects
by `(sha256(f"{seed}:{subject}").hexdigest(), subject)` ascending, with seed
string **`cardiosentinel-t2-internal-split-v1`**; the first 48 are the fit
partition and the remaining 8 the internal-development partition.

The assignment depends on the subject identity string and the frozen seed and on
nothing else. **No label, prevalence, episode count or model outcome
participates.**

**Internal-development subjects (8):**

`ltstdb:s2008`, `ltstdb:s2017`, `ltstdb:s2042`, `ltstdb:s2046`, `ltstdb:s2049`,
`ltstdb:s2050`, `ltstdb:s2063`, `ltstdb:s2064`

**Fit subjects (48):**

`ltstdb:s2001`, `ltstdb:s2002`, `ltstdb:s2003`, `ltstdb:s2006`, `ltstdb:s2007`,
`ltstdb:s2010`, `ltstdb:s2011`, `ltstdb:s2012`, `ltstdb:s2013`, `ltstdb:s2014`,
`ltstdb:s2016`, `ltstdb:s2018`, `ltstdb:s2021`, `ltstdb:s2025`, `ltstdb:s2026`,
`ltstdb:s2027`, `ltstdb:s2028`, `ltstdb:s2030`, `ltstdb:s2033`, `ltstdb:s2034`,
`ltstdb:s2036`, `ltstdb:s2037`, `ltstdb:s2038`, `ltstdb:s2039`, `ltstdb:s2040`,
`ltstdb:s2041`, `ltstdb:s2043`, `ltstdb:s2044`, `ltstdb:s2045`, `ltstdb:s2047`,
`ltstdb:s2048`, `ltstdb:s2052`, `ltstdb:s2053`, `ltstdb:s2054`, `ltstdb:s2056`,
`ltstdb:s2061`, `ltstdb:s2062`, `ltstdb:s3066`, `ltstdb:s3067`, `ltstdb:s3069`,
`ltstdb:s3070`, `ltstdb:s3071`, `ltstdb:s3075`, `ltstdb:s3076`, `ltstdb:s3077`,
`ltstdb:s3078`, `ltstdb:s3079`, `ltstdb:s3080`

**Internal split digest:**
`54f8091ee7d4620ab6e24aaa32b121874b6a1610003e3df63f94f9727618e28e`

Neither partition contains any outer VALIDATION or sealed TEST subject; the
three outer partitions are pairwise disjoint at source.

## 9. Training population

Full chronological PRIMARY TRAIN streams. **The old 3:1 negative sampling
strategy is forbidden inside T2 temporal sequences** — dropping arbitrary
negative windows destroys the temporal continuity T2 exists to model.

All available PRIMARY windows in the permitted T2 training subjects are used.
Class imbalance is handled **only** through the prospectively frozen loss
weighting of §10.

## 10. Loss

Frozen: **binary cross entropy with logits** on the current-window target.

Positive class weight = `N_negative / N_positive`, computed **only** on the
48-subject T2 fitting partition.

No focal-loss comparison. No loss-family search. No validation-derived class
weight. The realised weight is reported in the eventual training artifact.

## 11. Longitudinal training semantics

Recordings are approximately day-scale and cannot be one unrestricted
backpropagation graph.

Frozen **truncated backpropagation through time**, TBPTT length **256 windows** =
**1 280 s = 21 min 20 s** of gradient horizon at the 5-second stride.

The distinction that matters:

- hidden state **MAY** carry causally from one chunk into the next;
- the carried state is **detached** at the chunk boundary;
- gradients **do NOT** propagate beyond 256 windows;
- state resets **only** at real stream boundaries.

**There is no artificial state reset every 256 windows.** Stream chronological
order is preserved throughout.

## 12. Model arms

Exactly two trainable longitudinal candidates.

**T2-A — `causal_gru_longitudinal_v1`.** Standard causal recurrent comparator.

**T2-B — `causal_s4d_longitudinal_v1`.** Compact causal diagonal state-space
model operating across ECG windows.

T2-B is **S4D-inspired / diagonal state-space longitudinal modelling**. It must
**not** be called *Mamba*: no actual Mamba implementation is used, and none is
authorised in V1. No external SSM package may be installed; existing PyTorch
only.

## 13. Shared capacity envelope

The comparison must not be won by one candidate being dramatically larger.

| Parameter | Value |
|---|---|
| input dimension | 146 |
| input projection | 64 |
| temporal hidden / state width | 64 |
| temporal layers | 2 |
| dropout | 0.10 |
| output | single current-window logit |

Both candidates must remain within approximately **0.5× to 2.0×** of each
other's trainable parameter count. Exact parameter counts are persisted. **Model
size is never increased after seeing results.**

## 14. Optimisation

| Setting | Value |
|---|---|
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| maximum epochs | 10 |
| gradient clipping | 1.0 |
| seed | 2026 |

Checkpoint selection and early stopping use the **8-subject internal-development
split only**. Primary checkpoint criterion: internal-development pooled AUPRC;
tie → earlier epoch; patience 3 completed epochs.

**Outer VALIDATION is not accessed during epoch selection.**

## 15. Outer VALIDATION is one-shot model comparison

After each candidate is fully frozen using TRAIN + internal-development only,
each is evaluated **exactly once** on the existing 12-subject outer VALIDATION
partition.

No architecture or hyperparameter modification after seeing outer VALIDATION. No
second T2 attempt merely because a result is disappointing. Any recovery after
infrastructure failure requires separate human review and must preserve exposure
status exactly as M1/M2/U1 did.

## 16. Model selection

Primary: **pooled PRIMARY VALIDATION AUPRC.** Secondary: subject-macro AUPRC.

1. higher pooled AUPRC wins;
2. if the absolute pooled-AUPRC difference is `< 0.002`, higher subject-macro
   AUPRC wins;
3. if still tied within `0.002`, retain the smaller / faster model.

No weighted composite scores, no challenge-weighted selection, no
latency-adjusted scientific scores. **Challenge evidence is never a selection
input.**

## 17. Inherited comparators

The eventual T2 result reports comparison against frozen inherited development
evidence where scientifically compatible — P1-B, and M1L / M2-G.

Their completed DEVELOPMENT experiments are **not** rerun to manufacture a
matched table. Where a metric is not directly comparable because of population
or score semantics, that limitation is stated explicitly rather than papered
over.

## 18. T2 output is not calibrated uncertainty

The T2 sigmoid / logit is a **temporal model score**. It is not automatically a
calibrated probability, a confidence, an uncertainty or conformal evidence, and
a raw T2 sigmoid is never called uncertainty.

U1 remains the retained calibrated probability source for the frozen detector.
Any later calibration of T2 requires a separate prospective protocol.

## 19. Binary threshold

T2's primary evidence is threshold-free AUPRC.

Where binary descriptive metrics are required, one threshold per arm is frozen
using **only** the 8-subject internal-development partition, by **exact maximum
F1 with highest-threshold tie-break**, matching repository convention.

That threshold is locked **before** outer VALIDATION. Outer VALIDATION must not
select or alter it.

## 20. PRIMARY VALIDATION metrics

Pooled: AUPRC, AUROC; and at the frozen TRAIN-internal threshold — F1,
sensitivity, specificity, PPV, NPV, balanced accuracy, MCC.

Subject-macro: AUPRC, AUROC, sensitivity, specificity, MCC.

Correlated windows are never treated as independent inferential units.

## 21. Subject-level uncertainty

Subject-level bootstrap only: **1000 replicates, seed 2026, sampling unit
subject**. Windows are never bootstrapped.

Scope: **between-subject variation conditional on the fitted temporal model.**
No claim of full training-procedure uncertainty is made.

## 22. Temporal descriptive evidence

At the frozen binary threshold, report: number of positive prediction runs;
median positive-run duration; isolated single-window positive fraction;
transition count per hour; prediction persistence around labelled ischemic
intervals.

These are **DESCRIPTIVE**. Thresholds are not optimised against them in T2-v1.
Formal episode-state decisions belong to T1.

## 23. Cold start

Report the inherited strata: 0–5 min, 5–60 min, >60 min.

T2 hidden state begins from the frozen zero / uninitialised state at each stream
start. No warmup threshold, no cold-start repair, no post-hoc alternative state
initialisation.

## 24. Challenge evidence

After PRIMARY evaluation, challenge behaviour is reported separately for
rate-related, axis-shift and conduction-change.

T2 does not train on these identities and does not select architecture from
them, and challenge rows are never merged into PRIMARY. For rate and axis,
false-positive behaviour is reported at the frozen T2 internal-dev threshold.
Conduction change remains descriptive only, given its tiny support.

## 25. T1 interface contract — contract only

**T1 IS NOT IMPLEMENTED HERE.** Only the future interface is frozen.

The later T1 state machine may prospectively consume:

- **A.** the frozen detector decision;
- **B.** the subject-disjoint U1 OOF Platt calibrated probability on DEVELOPMENT;
- **C.** U1 calibrated uncertainty derived from that probability and the frozen
  detector decision;
- **D.** retained M2-G patient-adaptation evidence available causally;
- **E.** the selected T2 temporal evidence score;
- **F.** physical availability state;
- **G.** elapsed causal time / state duration.

T1 states are intended to be **NORMAL, WATCH, EVENT, RECOVERY**.

However **no transition threshold, no persistence duration, no hysteresis value,
no EVENT onset rule and no RECOVERY rule is selected here.** Those belong to a
separate prospective T1 protocol, after T2 is retained.

## 26. Why T1 and T2 must remain separate

**T2** is a learned temporal representation / evidence score. **T1** is
deterministic system-state semantics with operational persistence and
hysteresis.

The SSM is **not** trained to emit NORMAL/WATCH/EVENT/RECOVERY directly. Doing so
would entangle learned sequence modelling with system policy, and would make
ablation and interpretability weaker: a state-machine error and a representation
error would become indistinguishable, and every future policy change would
require retraining.

## 27. Routing

The rejected U1 symmetric window-level router **remains rejected**. No final
edge/cloud routing is introduced here and **no route threshold is selected**.

Future routing may be reconsidered prospectively **after T1**, using temporal
state together with retained calibrated uncertainty.

## 28. TEST firewall

TEST remains absolutely sealed. No TEST waveform, `z`, label, score, episode,
metric or subject-specific statistic may be accessed.

The eventual T2 development result must still record `test_accessed = false` and
`sealed_test_state = unopened`.

## 29. Development optimism

The outer VALIDATION partition **has already been used** in upstream model,
threshold and calibration development.

T2 uses an internal TRAIN split to reduce additional outer-validation tuning, but
**T2 outer-VALIDATION results remain DEVELOPMENT evidence.** They are not unseen
generalisation and must never be described as such.

## 30. Bound provenance

| Item | Value |
|---|---|
| Starting Git SHA | `997df407376edcf585a68d019b26b02a7670c12b` |
| Split (self-digest) | `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7` |
| Split (file) | `74f055dee370ab2742b2a5346eb37de4d3f6fccb011676b203b3eb339a62d714` |
| P1 protocol | `f48ffc66e52649d74a8286182d5e7220f78abdd6c12a7ebfe04f116b853337f1` |
| P1 retention decision | `7b403709fa0fb12eef65423d830c121fc3ada904266a1b47931d438f5e797d68` |
| P1-B experiment lock | `796f00e3ea27b1be272f843f0fb82b2c3e450311308404b78a1def22eb0676d0` |
| B4 protocol | `f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b` |
| B4-B encoder checkpoint | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |
| Physiology schema | `13f60be400b5b957c1eb592bbafd8206d4d2855c1aa657a058671fb8d7cab434` |
| Physiology transform | `cc6bd3a353f0ac6cad342114ed96e135cbf3c61e2946f847d5b95358b6bd51a9` |
| TRAIN stream cache | `d006c698017110bfd95774ca207036a820139779b95cf1b3f3a36c06efa779a4` |
| VALIDATION stream cache | `a3e39137a04ebebb3b97ef6c6c614339c990a6041cf649a0ba6e3c2d43baae18` |
| TRAIN representation content | `e52a566fbc285a7a9f92715752dee43c020faa3550aaeb660f5f400dee07b5d3` |
| VALIDATION representation content | `b26a2d9b6150e6518dc2bfb394427dc93ae48a7cc3de30adcc3fefcc9f1f53ba` |
| **U1 retention decision** | **`9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b`** |
| U1 canonical result | `649631cbf5188731d006f533997cfe28df4f5acb79e7693514e86ad0cef0cb12` |
| U1 experiment lock (self-digest) | `7f4dd1505919e23a598773736dc57e2d1b4d360f496b45acdf2028ed0574b1b6` |
| M2 retention decision | `da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47` |
| M2-G arm result | `a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75` |
| **T2 internal subject split** | **`54f8091ee7d4620ab6e24aaa32b121874b6a1610003e3df63f94f9727618e28e`** |
| Stream ordering rule | `(record_id, channel_index)` ordered by `window_start_samples` |
| Availability rule | `AVAILABLE = 1`, `UNAVAILABLE_EXACT_FLAT = 2`; never synthesised, never absorbed |
| Input dimension | 146 |
| TBPTT length | 256 windows |
| Candidates | `causal_gru_longitudinal_v1`, `causal_s4d_longitudinal_v1` |
| Optimizer | AdamW, lr 3e-4, wd 1e-4, ≤10 epochs, clip 1.0 |
| Seed | 2026 |
| Selection rule | pooled AUPRC → subject-macro AUPRC within 0.002 → smaller model |
| TEST state | `test_accessed: false`, `sealed_test_state: unopened` |

## 31. Environment

No dependency installation, no `pip`, no environment repair. The canonical
interpreter remains `/home/AI_POC/venvs/tactics/bin/python`, and the expected
runtime dependency digest remains
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

If implementation would require a new package, execution **STOPS FOR HUMAN
REVIEW**. Nothing is installed.

## 32. Scope of this change set

This protocol is design only. Allowed here: static repository and artifact
metadata inspection, frozen manifest and digest inspection, schema inspection,
synthetic arrays, synthetic streams, and tests.

Forbidden here: T2 model training, real TRAIN optimisation, real VALIDATION
scoring, real T2 threshold selection, real challenge scoring, and any TEST
access. No outcome analysis was conducted.
