# U1 Calibration and Selective-Routing Protocol V1

## 0. Nature and status of this document

**FROZEN PROSPECTIVE SCIENTIFIC PROTOCOL.** It is written before any U1
calibrator is fitted, before any calibrated probability exists, and before any
routing threshold is chosen. It freezes *what will be done and how it will be
judged*, so that a later execution run cannot be steered by its own results.

No calibrator is fitted here. No VALIDATION row is scored here. No routing
threshold is derived here. No U1 metric is produced here. TEST is not opened.

A material later change requires `U1_..._PROTOCOL_V2.md` with a documented
scientific reason. A Git diff to V1 is not an acceptable silent change.

## 1. Frozen input identity — the retained M2-G system

U1 consumes the retained M2-G system exactly as frozen by the merged human
retention decision. It reopens nothing.

| Bound artifact | SHA-256 |
|---|---|
| Master at protocol freeze | `ba20fc94465ac5c3080b998096797cc6d965ec1f` |
| M2 update-policy retention decision | `da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47` |
| M2-v1 protocol | `a8ba6fad038ed0ec01156b6959239f489426d55db8ad73a0c704fd527e7db91c` |
| M2 TRAIN gate derivation receipt | `5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24` |
| M2 canonical suite result | `8a6b0a1c64da72fc0f4573c742bef491b01b4eb8179f0759da3c537a01939a02` |
| **M2-G arm result (U1 input)** | **`a061d4d8c5211381c18baa228436bb9abc78b2f87f71fe4cab6ca71b2d15cf75`** |
| **M2-G experiment lock (U1 input)** | **`5ac07d9f1ea3859e046c84fb91f22cee1bb20ef4857837b1c82fcd944dbf0fe8`** |
| M2-0 arm result (control, NOT a U1 input) | `37a6e9d4c01b823e407addbb897d14f6f54835a347a6557a745890585395644c` |
| Frozen benchmark split | `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7` |

Retained suite id: `m2-v1-development-two-arm-recovery2`.

The following are **frozen and must not be altered, refitted, reinterpreted or
re-derived by U1**: B4-B, P1-B, M1L, M2-G, gates G1–G6, all M2 thresholds,
memory alpha, the classification threshold, the normal-evidence threshold,
refractory logic, feature schemas, the benchmark split, and primary/challenge
semantics.

The M2 classification operating threshold remains exactly:

```
tau = 0.7554003000259399
```

**U1 is not authorised to improve classifier performance by changing `tau`.**

`M2-0` is control/ablation evidence. It is **not** a U1 calibration input. U1
calibrates the retained system only.

## 2. The two scores — a hard distinction

M2 contains two numerically distinct quantities. Conflating them would be a
scientific error, so the distinction is frozen here.

| | G4 normal-evidence quantity | M2-G classifier output |
|---|---|---|
| Purpose | memory-admission gating only | ischemia decision |
| Frozen threshold | `0.0002997174742631614` | `0.7554003000259399` |
| Used for classification? | **never** | yes |
| Used by U1? | **NEVER** | **yes — the sole calibration input** |

The G4 normal-evidence quantity is **not** a calibrated probability, **not** a
confidence, **not** an uncertainty and **not** a conformal score. The frozen
M2-G arm result records `classification_threshold_used_for_memory_admission:
false`, and the two thresholds are numerically distinct by three orders of
magnitude.

**U1 must not calibrate, reinterpret, rescale or route on the G4 admission
score.** Any U1 artifact that consumes the G4 quantity as classifier confidence
is invalid by construction.

### The appropriate calibration input

The appropriate — and only permitted — U1 calibration input is the persisted
per-window M2-G classifier score, recorded in the frozen evidence store under
schema `m2_v1_evidence_store/1` as the column:

```
score : float64
```

with the frozen semantics recorded in the M2-G arm result:

> `uncalibrated sigmoid model score; not calibrated probability`

## 3. What is actually persisted — a binding numerical finding

This determines the calibrator family, so it is frozen here rather than
rediscovered later.

The retained M1L head is declared `"output": "single_raw_logit"`. The scorer
computes, per window:

```
score = float64( sigmoid( head(features) ) )
```

where the head parameters and the feature matrix are **float32**, so the
sigmoid is evaluated in **float32** and only afterwards widened to float64.

Two consequences follow, and both are binding on U1:

1. **True logits are not persisted.** Only the post-sigmoid probability is
   stored. Re-deriving genuine logits would require re-running M2, which is
   **forbidden**. Therefore *temperature scaling in its proper sense — a rescale
   of true logits — is not available to U1.*
2. **Recovered logits are quantized, and saturate.** A logit may be recovered as
   `z = log(p) - log1p(-p)`, but only up to float32 sigmoid resolution. The
   float32 spacing immediately below `1.0` is `2^-24 ~= 5.96e-8`, so
   `sigmoid(z)` rounds to exactly `1.0` for `z` above roughly `16.6`, and
   distinct large logits collapse onto a few representable probabilities.
   Recovered logits are therefore faithful in the central range and
   progressively unreliable toward the extremes.

U1 does not attempt to repair this. It measures it (§4.1), reports it, and
stops if it is material.

### 3.1 Mandatory saturation census — a precondition, not a metric

Before any calibrator is fitted, the U1 execution run must compute and persist,
over the permitted development population:

- count and fraction of rows with `score == 0.0` exactly;
- count and fraction of rows with `score == 1.0` exactly;
- count and fraction of rows outside `[delta, 1 - delta]`;
- the number of distinct persisted `score` values.

with the frozen clamp constant

```
delta = 1e-7
```

**Frozen stop rule.** If the fraction of rows outside `[delta, 1 - delta]`
exceeds

```
saturated_fraction_review_bound = 0.01
```

the run **STOPS FOR HUMAN REVIEW** and fits nothing. It does not fall back to
another calibrator, does not widen the clamp, and does not re-run M2. Heavy
saturation would mean the score carries no usable ordering at the extremes, and
calibrating it would fit quantization noise. Recovering true logits is a
separate versioned decision, not a U1 repair.

This census is a **precondition**, not a scientific result.

## 4. Permitted partitions and population

| Partition | U1 access |
|---|---|
| TRAIN | **none** — no M2-G replay evidence exists for it, and producing some would require a forbidden M2 rerun |
| VALIDATION | **permitted** — the sole development partition |
| TEST | **NONE — sealed, see §12** |

**Calibration and evaluation population:** the frozen PRIMARY population of the
retained M2-G arm — 473,897 labelled rows (21,628 ischemic positive, 452,269
background negative) across 12 subjects, identity digest
`a671d35a354748e47c9ce77726462c59dfdc82c14249c204ff6ef00d35a27f1c`.

The FULL REPLAY population (492,904 rows) is **never** a calibration or metric
denominator, exactly as in M2. The CHALLENGE population (8,137 rows) is
reported separately under §11 and is never merged into the PRIMARY denominator.

U1 operates strictly on the **post-replay evaluation side**. The frozen M2
label firewall — `artifacts -> replay -> evidence -> (then) evaluation` — is
preserved: no U1 quantity may enter replay, and no label may reach a replay-side
module.

### 4.1 Development-partition limitation, recorded not hidden

The frozen classification threshold `tau` was itself selected on VALIDATION
predictions under the frozen metrics protocol. U1 calibrates on the same
partition because no other M2-G evidence exists. Out-of-fold cross-fitting
(§5) removes *subject-level* self-evaluation, but it does not make VALIDATION an
independent partition.

Therefore **all U1 development calibration and routing evidence is
development-optimistic**, and no U1 result may be described as a generalisation
estimate. The sealed TEST partition remains the only clean check, and it is
deferred, not scheduled here.

## 5. Subject-disjoint calibration strategy

### 5.1 Chosen design — leave-one-subject-out cross-fitting, K = 12

The 12 permitted VALIDATION subjects, in frozen canonical ascending order:

```
ltstdb:s2004  ltstdb:s2005  ltstdb:s2019  ltstdb:s2020
ltstdb:s2023  ltstdb:s2031  ltstdb:s2057  ltstdb:s2058
ltstdb:s2059  ltstdb:s3068  ltstdb:s3072  ltstdb:s3073
```

Fold `k` (0-indexed) holds out exactly the `k`-th subject in that order. For
each fold:

- the calibrator is fitted **only** on the PRIMARY rows of the other 11
  subjects;
- it is applied **only** to the held-out subject's PRIMARY rows;
- the underlying M2-G predictions are consumed exactly as persisted, preserving
  their causal ordering;
- the fold's output is that subject's **out-of-fold calibrated probabilities**.

Every subject is therefore evaluated exactly once, by a calibrator that never
saw any row of that subject. Pooled out-of-fold evidence covers all 473,897
PRIMARY rows exactly once, with no overlap and no gap.

### 5.2 Why leave-one-subject-out, and not fewer folds

`K = 12` is the *smallest scientifically sensible* design in the sense that
matters here — it has the fewest free design choices, not the fewest fits:

- **It removes the grouping degree of freedom entirely.** Any `K < 12` requires
  an additional rule deciding *which* subjects share a fold. That rule is a free
  parameter that must itself be justified, and it is exactly the kind of choice
  that can be tuned, even unintentionally, toward a flattering result. LOSO's
  partition is forced by subject identity alone; there is nothing left to
  choose.
- **The usual objection to LOSO does not apply.** LOSO is normally criticised
  for high-variance fits on small training sets. Here each fit uses a 1–2
  parameter monotonic calibrator on roughly 4 x 10^5 rows from 11 subjects.
  Fit variance is negligible; the binding constraint is subject count, which is
  12 under every design.
- **It maximises fit-set subject diversity** (11 of 12 subjects per fold), which
  matters far more than row count for between-subject calibration drift.
- **It gives complete out-of-fold coverage**, so the pooled development
  evaluation uses the entire frozen PRIMARY population rather than a fraction.

### 5.3 Why not a simpler fixed calibrate/evaluate subject split

A fixed split — for example 6 subjects to fit and 6 to evaluate — was
considered and is **rejected**, for reasons stated before any result exists:

- it would evaluate calibration on **6 subjects**, and with between-subject
  calibration variability that is too few to support even a descriptive claim;
- it would discard roughly half the labelled evidence from the calibrator fit;
- it would reintroduce the grouping degree of freedom in its sharpest form —
  *which* 6 subjects — with a large effect on the result;
- it produces no out-of-fold probability for the fitting half, so pooled
  evidence could not cover the frozen PRIMARY population.

Cross-fitting is scientifically superior here. This choice is recorded now, with
its reasoning, precisely so it cannot later be presented as a matter of
convenience.

### 5.4 Fold assignment is frozen before fitting, and depends only on identity

Fold assignment is a pure deterministic function of **frozen subject identity**:
ascending lexicographic order of the canonical namespaced subject id, as
recorded in the frozen split manifest.

It must **not** depend on — and structurally cannot depend on — M2-G
performance, prediction errors, calibration error, subject AUPRC, subject
false-positive rate, subject prevalence, or any desired result. No M2-G output
is an input to fold assignment.

The fold assignment is persisted with a digest and is verifiable independently
of any score.

## 6. Calibrator family

U1 is deliberately simple. No neural uncertainty model is introduced. No
flexible calibrator is adopted merely to lower ECE.

**Primary method — Platt scaling (2-parameter logistic) on the recovered,
clamped logit:**

```
z   = log(p_clamped) - log1p(-p_clamped),    p_clamped = clip(score, delta, 1 - delta)
g(s) = sigmoid(a * z + b)
```

fitted by maximum likelihood on the fold's fit rows.

**Predeclared comparator — temperature-only scaling**, the nested special case
`b = 0`, fitted identically. It is labelled an *approximation* throughout,
because true logits are not persisted (§3); it is a rescale of recovered
logits, not of genuine logits, and no U1 output may describe it otherwise.

Platt is primary rather than temperature-only because a single scale parameter
cannot correct an intercept/prevalence offset, and the recovered logit is a
quantized reconstruction whose zero point is not guaranteed meaningful.

### 6.1 Frozen selection criterion between the two

Selection is by **mean out-of-fold negative log-likelihood** over the pooled
PRIMARY out-of-fold probabilities, computed with the frozen clamp `delta`.

- Lower pooled out-of-fold NLL wins.
- **Tie-break:** if the two differ by less than `nll_tie_tolerance = 1e-4` in
  absolute pooled NLL, the **simpler nested model (temperature-only) is
  retained**. Simplicity breaks ties, never the more flexible model.
- Selection uses **development out-of-fold evidence only**. It never uses TEST,
  never uses ECE, and never uses a weighted combination invented after results
  are seen.

Both fitted families are reported side by side regardless of which is selected.

### 6.2 Numerical safeguards, frozen

- clamp `delta = 1e-7` applied identically in logit recovery, NLL and reporting;
- fits use float64 throughout;
- optimiser: L-BFGS-B, deterministic, initial `(a, b) = (1.0, 0.0)`,
  `maxiter = 500`, gradient tolerance `1e-10`;
- non-convergence, a non-finite parameter, or a non-monotonic fitted map
  (`a <= 0`) is a **hard failure that stops the run for human review** — it is
  never silently retried, re-initialised, or replaced by the comparator;
- every fitted `(a, b)` per fold is persisted.

Strict monotonicity (`a > 0`) is required and checked, because §7 depends on it.

## 7. Classification remains frozen — provable, not asserted

The frozen decision is, and remains, on the **raw persisted score**:

```
y_hat = 1  if and only if  score >= tau,   tau = 0.7554003000259399
```

Because every permitted calibrator `g` is strictly monotonic increasing
(`a > 0`, enforced in §6.2), for the fold-`k` calibrated boundary
`pi_k = g_k(tau)`:

```
score >= tau   <==>   g_k(score) >= pi_k
```

so the calibrated decision is **identical** to the frozen decision, row for row.
Calibration cannot silently redefine the ischemia classifier.

Calibrated probability is used **only** for uncertainty estimation and
local-versus-cloud routing. **No new classification threshold is optimised in
U1.**

## 8. Uncertainty definition — frozen

Let `p = g_k(score)` be the out-of-fold calibrated probability and `y_hat` the
frozen decision. The **primary U1 uncertainty** is the calibrated probability
that the frozen decision is wrong:

```
u = 1 - p   if y_hat == 1
u =     p   if y_hat == 0
```

`u` lies in `[0, 1]`. It is decision-aware, directly interpretable as an
expected error probability, and it makes calibration genuinely load-bearing: a
miscalibrated `p` produces a wrong risk estimate rather than merely a rescaled
ordering.

Raw sigmoid output is **never** called uncertainty. A raw score margin is not a
U1 uncertainty.

**Secondary, descriptive only:** calibrated binary entropy
`H(p) = -p ln p - (1 - p) ln(1 - p)`, reported for completeness. It is
**not** used for the retained routing rule. This is fixed now so that entropy
cannot be substituted later if it happens to look better.

### 8.1 Routing semantics

```
accept locally   if  u <= u_star
escalate to cloud if  u >  u_star
```

Conceptually this supports high-confidence local normal, high-confidence local
abnormal, and ambiguous escalation. U1 is **window-level selective routing
only**. Episode persistence, onset delay and false-alarms-per-hour are **not**
implemented or claimed here; they remain Phase-7 temporal work.

## 9. Risk, coverage and required evidence

**Risk** is the error rate of the frozen decision among locally accepted
windows:

```
risk(u_star) = ( FP_accepted + FN_accepted ) / N_accepted
```

**Coverage** is the accepted fraction; **escalation fraction** is `1 - coverage`.

### 9.1 Frozen coverage grid

```
0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00
```

At each grid point the run reports the full risk-coverage curve on pooled
out-of-fold development evidence. `1.00` is the no-routing reference.

### 9.2 Calibration evidence (required)

- **Brier score** `mean((p - y)^2)`;
- **NLL** `-mean[ y ln p + (1 - y) ln(1 - p) ]` with clamp `delta`;
- **ECE**, both binnings, frozen semantics:
  - equal-width: 15 bins on `[0, 1]`;
  - equal-mass: 15 quantile bins;
  - in both, `ECE = sum_b (n_b / N) * | empirical_positive_rate_b - mean_p_b |`,
    over the positive-class probability;
- **reliability evidence**: per-bin count, mean `p`, empirical positive rate,
  for both binnings;
- the same three quantities for the **uncalibrated baseline** (raw `score`
  treated as a probability), as the comparison the retention criterion needs.

### 9.3 Selective-routing evidence (required)

At every grid point: coverage, escalation fraction, accepted risk, accepted
window count.

**Class-aware evidence, mandatory** — so the system cannot appear safe merely by
abstaining disproportionately from difficult positives:

- accepted sensitivity, specificity, PPV, NPV;
- escalation fraction among true positives;
- escalation fraction among true negatives;
- accepted positive and negative counts.

**Routing-calibration consistency diagnostic (required):** predicted accepted
risk `mean(u | accepted)` versus observed accepted risk. Under good calibration
these agree; divergence is itself the evidence that the calibrated probability
is not trustworthy for routing.

### 9.4 Subject-level evidence

Per subject: coverage, escalation fraction, accepted risk, and accepted
sensitivity/specificity where defined. Subject-macro summaries follow the frozen
metrics protocol — a subject lacking a class is **undefined, never zero**, and
contributing/non-contributing counts are reported for every macro quantity.

Confidence intervals reuse the frozen convention rather than inventing one:
1000 subject-level bootstrap replicates, seed `2026`, 2.5th/97.5th percentiles,
with degenerate and undefined replicates reported. Window-level bootstrap is
prohibited.

## 10. Routing-policy selection rule — frozen now, applied later

**No routing threshold is chosen in this task.** The rule by which a later
DEVELOPMENT run will choose one is frozen here.

**Retained operating point:** the fixed target coverage

```
c_star = 0.90
```

declared **a priori on deployment-capacity grounds** — an edge device escalating
roughly one window in ten — and explicitly **not** read off the risk-coverage
curve.

`u_star` is then the deterministic empirical quantile of pooled out-of-fold `u`
over the frozen PRIMARY population: the **smallest** `u_star` whose coverage is
`>= c_star`, ties broken toward the smaller `u_star`, computed with a frozen
`numpy` `'lower'` quantile convention.

### 10.1 Why a fixed point on a reported grid, and not an optimised scalar

Optimising a single scalar — for example minimising `risk + lambda *
escalation` — requires `lambda`, which cannot be justified before results exist
and would in practice be chosen after seeing them. With only 12 subjects the
argmin is also genuinely noisy: a single subject's behaviour can move it
materially, so the "optimum" would not be stable evidence.

A frozen coverage grid plus **one** a-priori retained point is therefore
preferred. The human sees the entire trade-off curve and can judge it, while the
retained point cannot have been tuned. The cost is that `c_star = 0.90` is a
deployment assumption rather than a performance optimum, and this protocol
states that plainly rather than presenting it as optimal.

### 10.2 Reporting guards — flags, never automatic re-selection

Both guards **report and stop for human review**. Neither adjusts any threshold.

- **Asymmetric abstention:** if at `c_star` the positive escalation fraction
  exceeds the negative escalation fraction by a factor greater than
  `asymmetric_abstention_ratio = 3.0`, the run flags that the apparent safety
  gain is being bought by escalating positives.
- **Routing-calibration inadequacy:** if predicted accepted risk and observed
  accepted risk differ by more than
  `accepted_risk_agreement_tolerance = 0.02` absolute, the run flags that the
  calibrated probability is not adequate for routing.

## 11. Cold-start and challenge reporting — prospectively specified

**Cold start.** M2's 0–5 minute limitation is frozen and inherited. U1
introduces **no** special cold-start threshold and performs **no** post-hoc
cold-start repair. U1 *reports* calibration and selective behaviour by the
already frozen strata:

| Stratum | PRIMARY windows |
|---|---|
| 0–5 minutes | 1,798 |
| 5–60 minutes | 19,637 |
| over 60 minutes | 452,462 |

The 0–5 minute stratum contains a single ischemic positive, so its
discrimination-dependent quantities are undefined and are reported as counts
without a confidence interval. The retained routing point is **not** altered
because of any stratum result.

**Challenge reporting** is permitted, with the frozen evidence levels and
denominators of the metrics protocol, and never merged into PRIMARY:

| Subset | Windows | Evidence level |
|---|---|---|
| Rate-related | 4,973 | `quantitative_secondary` |
| Axis shift | 3,000 | `quantitative_secondary` |
| Conduction change | 164 | `exploratory_descriptive` |

Reported per subset: accepted false-positive rate at the retained routing point,
escalation fraction, false-positive count, contributing-subject count and
denominator. Conduction change has one contributing subject: it is descriptive
`FP / N` only, receives **no** bootstrap interval, and is **never** a selection
input or a headline result. Challenge selection digest
`49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a`.

## 12. TEST firewall — absolute

TEST remains **sealed and unopened**. For U1 specifically:

- no TEST path is resolved;
- no TEST subject enters calibration, fitting, fold assignment or evaluation;
- no TEST prediction, label, score or metric is read;
- no B4 sealed TEST result is reopened;
- no calibrator is fitted on TEST;
- no routing threshold is derived from TEST;
- no TEST reliability or risk-coverage analysis is produced.

The 12 TEST subjects are excluded **by construction**: the permitted calibration
subject set is the frozen VALIDATION set, the two are disjoint in the frozen
split manifest, and any subject outside the VALIDATION set is refused by the
protocol validator rather than filtered silently.

Every U1 artifact records `test_accessed: false` and
`sealed_test_state: "unopened"`. **Completion of U1 does not authorise TEST.**

## 13. Relation to M2 — no reopening, no retroactive insertion

- U1 consumes persisted M2-G output **read-only**.
- Calibration does **not** alter patient-memory history, prototypes, gate
  decisions, admission decisions or any M2 trajectory.
- Routing decisions do **not** retroactively affect the M2 replay that produced
  the development evidence. Routing is applied strictly downstream of frozen
  per-window output.
- **U1 uncertainty is never inserted into frozen M2-G.** If a future online
  architecture requires uncertainty to participate in memory admission, that is
  a separate, versioned, separately authorised experiment — not a U1 change and
  not a retro-fit of this one.
- **No M2 rerun is permitted by this protocol.**

## 14. Provenance required of the later U1 execution run

Execution Git SHA and dirty state; interpreter; package count and dependency
digest; runtime START / COMPLETION / PRE_PROMOTION observations under the
existing runtime-integrity sentinel; this protocol's SHA-256; the M2 retention
decision SHA; the M2-G arm-result and lock SHAs; the M2 suite SHA; the split
hash; the PRIMARY, CHALLENGE and FULL-REPLAY population identities; the frozen
fold assignment and its digest; the per-fold fitted parameters; the selected
calibrator family and the criterion value that selected it; the saturation
census; `test_accessed: false`; `sealed_test_state: "unopened"`.

## 15. Success / retention criterion — prespecified

U1 produces **no automatic retention**. A human performs the retention review,
exactly as for M1 and M2. The prespecified conditions the human will weigh:

1. the saturation census is within its frozen bound (§3.1);
2. pooled out-of-fold **Brier and NLL are both lower** than the uncalibrated
   baseline;
3. accepted risk at `c_star = 0.90` is strictly lower than the full-coverage
   (`c = 1.00`) error rate;
4. no asymmetric-abstention flag (§10.2);
5. no routing-calibration inadequacy flag (§10.2).

Failing any condition does **not** authorise a repair, a re-fit, a different
calibrator, a different coverage target or a different uncertainty definition.
It produces a reported result and a human decision.

Improved ECE alone is **not** a success criterion, and no U1 result may be
described as clinical safety, statistical significance, or generalisation.

## 16. No automatic transition

Completion of U1 does **not** begin U2 conformal prediction, T1, T2, edge
benchmarking, or episode-level work. U1 is scientifically complete without
conformal prediction. Whether U2 justifies its schedule is a separate human
decision made after U1 closes.

## 17. Implementation scope

This protocol is frozen in a design-only change set: the protocol document, a
structural validator module incapable of touching real development data, and
synthetic tests. The reviewed execution implementation is a **separate** later
change set, because calibration methodology is a new scientific decision that
deserves its own review.
