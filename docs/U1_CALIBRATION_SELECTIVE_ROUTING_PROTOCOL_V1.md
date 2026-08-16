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

`M2-0` is control/ablation evidence. It is **not** a U1 calibration input.

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
false`, and the two thresholds differ by three orders of magnitude.

**U1 must not calibrate, reinterpret, rescale or route on the G4 admission
score.** Any U1 artifact that consumes the G4 quantity as classifier confidence
is invalid by construction.

### The appropriate calibration input

The only permitted U1 calibration input is the persisted per-window M2-G
classifier score, recorded in the frozen evidence store under schema
`m2_v1_evidence_store/1` as the column:

```
score : float64
```

with the frozen semantics recorded in the M2-G arm result:

> `uncalibrated sigmoid model score; not calibrated probability`

## 3. What is actually persisted — a binding numerical finding

The retained M1L head is declared `"output": "single_raw_logit"`. The scorer
computes, per window:

```
score = float64( sigmoid( head(features) ) )
```

where the head parameters and the feature matrix are **float32**, so the
sigmoid is evaluated in **float32** and only afterwards widened to float64.

Two consequences follow, and both bind U1:

1. **True logits are not persisted.** Only the post-sigmoid probability is
   stored. Re-deriving genuine logits would require re-running M2, which is
   **forbidden**. Therefore *temperature scaling in its proper sense — a rescale
   of true logits — is not available to U1.*
2. **Recovered logits are quantized, and saturate.** A logit may be recovered as
   `z = log(p) - log1p(-p)`, but only up to float32 sigmoid resolution. The
   float32 spacing immediately below `1.0` is `2^-24 ~= 5.96e-8`, so
   `sigmoid(z)` rounds to exactly `1.0` for `z` above roughly `16.6`, and
   distinct large logits collapse onto a few representable probabilities.

No U1 output may describe the approximate recovered-logit comparator as
true-logit temperature scaling. U1 does not repair this; it measures it (§3.1),
reports it, and stops if it is material.

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
another calibrator, does not widen the clamp, and does not re-run M2.

This census is a **precondition**, not a scientific result.

## 4. Permitted partitions and population

| Partition | U1 access |
|---|---|
| TRAIN | **none** — no M2-G replay evidence exists for it, and producing some would require a forbidden M2 rerun |
| VALIDATION | **permitted** — the sole development partition |
| TEST | **NONE — sealed, see §13** |

**Calibration and evaluation population:** the frozen PRIMARY population of the
retained M2-G arm — **473,897** labelled rows (21,628 ischemic positive, 452,269
background negative) across 12 subjects, identity digest
`a671d35a354748e47c9ce77726462c59dfdc82c14249c204ff6ef00d35a27f1c`.

The FULL REPLAY population (492,904 rows) is **never** a calibration or metric
denominator. The CHALLENGE population (8,137 rows) is reported separately under
§12 and never merged into the PRIMARY denominator.

U1 operates strictly on the **post-replay evaluation side**. The frozen M2 label
firewall — `artifacts -> replay -> evidence -> (then) evaluation` — is
preserved.

### 4.1 Development optimism, recorded not hidden

The frozen classification threshold `tau` was itself selected on VALIDATION
predictions under the frozen metrics protocol. U1 calibrates on the same
partition because no other M2-G evidence exists.

Cross-fitting (§5) solves **subject self-calibration only**. It does **not**
make VALIDATION a new independent holdout. Therefore **all U1 development
evidence is development-optimistic and is not an independent generalisation
estimate.**

This is scientifically acceptable because it is explicitly DEVELOPMENT
evidence, TEST remains sealed, and later TEST access — if ever separately
authorised — is the generalisation check.

## 5. Subject-disjoint cross-fitting

### 5.1 Design — leave-one-subject-out, K = 12

The 12 permitted VALIDATION subjects, in frozen canonical ascending order:

```
ltstdb:s2004  ltstdb:s2005  ltstdb:s2019  ltstdb:s2020
ltstdb:s2023  ltstdb:s2031  ltstdb:s2057  ltstdb:s2058
ltstdb:s2059  ltstdb:s3068  ltstdb:s3072  ltstdb:s3073
```

Fold `k` (0-indexed) holds out exactly the `k`-th subject in that order. For
each fold the calibrator is fitted **only** on the PRIMARY rows of the other 11
subjects and applied **only** to the held-out subject's PRIMARY rows, yielding
that subject's **out-of-fold (OOF) calibrated probabilities**.

Every subject is evaluated exactly once, by a calibrator that never saw any row
of that subject. Pooled OOF evidence covers all 473,897 PRIMARY rows exactly
once, with no overlap and no gap.

### 5.2 Why leave-one-subject-out

- **It removes the grouping degree of freedom entirely.** Any `K < 12` requires
  an additional rule deciding *which* subjects share a fold — a free parameter
  that can be tuned, even unintentionally, toward a flattering result. LOSO's
  partition is forced by subject identity alone.
- **It maximises fit-set subject diversity** — 11 of 12 subjects per fold.
- **It gives complete subject-disjoint OOF coverage**, so pooled development
  evaluation uses the entire frozen PRIMARY population.
- **Every subject is evaluated exactly once.**

### 5.3 What the row count does and does not buy — dependence statement

Each LOSO fit uses **11 subjects** and a low-dimensional 1–2 parameter monotonic
calibrator. The large row count provides **numerical support for optimisation**,
but it does **not** remove within-subject dependence or between-subject
calibration variability.

Longitudinal ECG windows from one subject are correlated: 10-second windows at a
5-second stride overlap by construction, and a subject's morphology, baseline
and noise regime persist across hours. **The effective independent support
remains subject-level — 12 subjects.**

**No U1 output may claim that hundreds of thousands of windows constitute
independent evidence**, nor treat the row count as grounds for asserting that
between-fold or between-subject calibration variability is small.

### 5.4 A fixed calibrate/evaluate split was considered and rejected

A fixed split — for example 6 subjects to fit and 6 to evaluate — would evaluate
calibration on 6 subjects, discard roughly half the labelled evidence from the
fit, reintroduce the grouping degree of freedom in its sharpest form (*which* 6),
and produce no OOF probability for the fitting half. Cross-fitting is superior
here. Recorded now so it cannot later be presented as convenience.

### 5.5 Fold assignment depends only on identity

Fold assignment is a pure deterministic function of **frozen subject identity**:
ascending lexicographic order of the canonical namespaced subject id.

It must **not** depend on — and structurally cannot depend on — M2-G
performance, prediction errors, calibration error, subject AUPRC, subject
false-positive rate, subject prevalence, or any desired result.

## 6. Calibrator family and its selection

U1 is deliberately simple. No neural uncertainty model is introduced. No
flexible calibrator is adopted merely to lower ECE.

**Primary method — Platt scaling (2-parameter logistic) on the recovered,
clamped logit:**

```
z    = log(p_clamped) - log1p(-p_clamped),   p_clamped = clip(score, delta, 1 - delta)
g(s) = sigmoid(a * z + b)
```

fitted by maximum likelihood on the fold's fit rows.

**Predeclared comparator — temperature-only scaling**, the nested special case
`b = 0`. It is labelled an *approximation* throughout, because true logits are
not persisted (§3).

Platt is primary because a single scale parameter cannot correct an
intercept/prevalence offset, and the recovered logit is a quantized
reconstruction whose zero point is not guaranteed meaningful.

### 6.1 Family selection is OUT-OF-FOLD ONLY

Selection is by **mean pooled out-of-fold negative log-likelihood**, computed
with the frozen clamp `delta`, over the OOF probabilities of §7A.

- Lower pooled OOF NLL wins.
- **Tie-break:** if the two differ by less than `nll_tie_tolerance = 1e-4` in
  absolute pooled NLL, the **simpler nested model (temperature-only)** is
  retained. Simplicity breaks ties, never the more flexible model.
- Selection uses **development OOF evidence only**. It never uses TEST, never
  uses ECE, and never uses a weighted combination invented after results.

**No final all-subject fit result may reopen this decision** (§7B). Both fitted
families are reported side by side regardless of which is selected.

### 6.2 Numerical safeguards, frozen

- clamp `delta = 1e-7`, applied identically in logit recovery, NLL and
  reporting;
- fits use float64 throughout;
- optimiser: L-BFGS-B, deterministic, initial `(a, b) = (1.0, 0.0)`,
  `maxiter = 500`, gradient tolerance `1e-10`;
- non-convergence, a non-finite parameter, or a non-monotonic fitted map
  (`a <= 0`) is a **hard failure that stops the run for human review** — never a
  silent retry, re-initialisation, or substitution;
- every fitted `(a, b)` is persisted, per fold and for the final fit.

Strict monotonicity (`a > 0`) is required and checked, because §8 depends on it.

**If the final fit of the selected family fails (§7B), the run STOPS FOR HUMAN
REVIEW. It must NOT fall back to the other family.**

## 7. Two distinct calibration artifacts

Cross-fitting produces correct development evidence but does not define which
calibrator a *future unseen subject* uses. U1 therefore freezes **two** distinct
artifacts with different purposes and different claim status.

```
OOF evidence               = DEVELOPMENT evaluation
final all-validation fit   = deployable configuration
```

### 7A. U1 OOF development calibration

**Twelve** LOSO calibrators (§5.1). For each VALIDATION subject: fit on the
other 11 subjects, apply only to the held-out subject.

Purpose:

- calibrator-family selection (§6.1);
- calibration metrics;
- risk-coverage evidence;
- the human U1 retention decision;
- later DEVELOPMENT-side temporal experiments requiring subject-disjoint
  calibration.

**All U1 DEVELOPMENT performance evidence MUST use these OOF probabilities.**
An all-subject fitted calibrator is **never** substituted into DEVELOPMENT
metrics.

### 7B. U1 final deployment calibrator

**Only after** the calibrator family has been selected from frozen OOF NLL
(§6.1), fit **exactly one** calibrator *of that already-selected family* on:

- all 12 frozen VALIDATION subjects, and
- all frozen PRIMARY rows belonging to them.

**This is not a new model-selection step.** No alternative family may be
reconsidered during this fit.

Persist: selected family; all final fitted parameters; fit subject ids; fit
population identity; optimiser status; clamp; this protocol's SHA-256; the M2-G
identities; and the final calibrator artifact SHA-256.

The purpose of this artifact is **only**:

- future unseen subjects;
- later sealed TEST, if separately authorised;
- final physical edge/cloud deployment configuration.

**It must NOT be used to report U1 DEVELOPMENT calibration performance.** The
final-fit pass is **parameterisation, not evaluation** — its in-sample
behaviour is not a U1 scientific result and no in-sample performance claim from
it is authorised.

## 8. Classification remains frozen — provable, not asserted

The frozen decision is, and remains, on the **raw persisted score**:

```
y_hat = 1  if and only if  score >= tau,   tau = 0.7554003000259399
```

Because every permitted calibrator `g` is strictly monotonic increasing
(`a > 0`, enforced in §6.2), for a calibrated boundary `pi = g(tau)`:

```
score >= tau   <==>   g(score) >= pi
```

so the calibrated decision is **identical** to the frozen decision, row for row.
Calibration cannot silently redefine the ischemia classifier. **No new
classification threshold is optimised in U1.**

## 9. Uncertainty definition — frozen

Let `p` be the calibrated probability and `y_hat` the frozen decision. The
**primary U1 uncertainty** is the calibrated probability that the frozen
decision is wrong:

```
u = 1 - p   if y_hat == 1
u =     p   if y_hat == 0
```

`u` lies in `[0, 1]`. It is decision-aware and directly interpretable as an
expected error probability, so calibration is genuinely load-bearing: a
miscalibrated `p` produces a wrong risk estimate, not merely a rescaled
ordering.

Raw sigmoid output is **never** called uncertainty.

**Secondary, descriptive only:** calibrated binary entropy
`H(p) = -p ln p - (1 - p) ln(1 - p)`. It is **not** used for the retained
routing rule. Fixed now so it cannot be substituted later.

### 9.1 Routing semantics

```
accept locally    if  u <= u_star
escalate to cloud if  u >  u_star
```

U1 is **window-level selective routing only**. Episode persistence, onset delay
and false-alarms-per-hour are **not** implemented or claimed here.

## 10. Risk, coverage and required evidence

**Risk** is the error rate of the frozen decision among locally accepted
windows:

```
risk = ( FP_accepted + FN_accepted ) / N_accepted
```

**Coverage** is the accepted fraction; **escalation fraction** is `1 - coverage`.

### 10.1 Frozen coverage grid

```
0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00
```

`1.00` is the no-routing reference. The full risk-coverage curve is reported on
pooled **OOF** evidence.

### 10.2 Calibration evidence (required, OOF only)

- **Brier score** `mean((p - y)^2)`;
- **NLL** `-mean[ y ln p + (1 - y) ln(1 - p) ]` with clamp `delta`;
- **ECE**, both binnings, constructed exactly as in §10.3;
- **reliability evidence** per bin;
- the same three quantities for the **uncalibrated baseline** (raw `score`
  treated as a probability).

### 10.3 Exact ECE bin construction — frozen, not library-delegated

Let `p` be the calibrated probability and `stable_id` the frozen per-row
identity `(record_id, channel_index, start_sample)`.

**Equal-width — 15 intervals over `[0, 1]`.** Bin `b` for `b = 0 .. 14`:

```
bin b  covers  [ b/15 , (b+1)/15 )     for b = 0 .. 13
bin 14 covers  [ 14/15 , 1.0 ]         inclusive of 1.0
```

Lower edge inclusive, upper edge exclusive, **except** the final bin, whose
upper edge is inclusive so that `p = 1.0` is always binned.

**Equal-mass — 15 contiguous groups.** Sort all eligible rows ascending by the
deterministic key `(p, stable_id)`. Split the ordered rows into exactly 15
contiguous groups whose sizes differ by at most one: with `N = 15 * q + r`, the
first `r` groups have `q + 1` rows and the remaining `15 - r` groups have `q`
rows. If calibrated-probability ties span a group boundary, `stable_id` decides
deterministically which side each tied row falls on.

**Equal-mass semantics are never delegated to a library-default quantile
implementation.**

For both binnings, persist per bin: `count`, `minimum probability`,
`maximum probability`, `mean probability`, `empirical positive fraction`.

```
ECE = sum_b ( n_b / N ) * | empirical_positive_fraction_b - mean_probability_b |
```

### 10.4 Selective-routing evidence (required)

At every grid point: coverage, escalation fraction, accepted risk, accepted
window count.

**Class-aware evidence, mandatory** — so the system cannot appear safe merely by
abstaining disproportionately from difficult positives:

- accepted sensitivity, specificity, PPV, NPV;
- escalation fraction among true positives;
- escalation fraction among true negatives;
- accepted positive and negative counts.

**Routing-calibration consistency diagnostic (required):** predicted accepted
risk `mean(u | accepted)` versus observed accepted risk.

### 10.5 Subject-level evidence and the bootstrap claim boundary

Per subject: coverage, escalation fraction, accepted risk, and accepted
sensitivity/specificity where defined. Subject-macro summaries follow the frozen
metrics protocol — a subject lacking a class is **undefined, never zero**, with
contributing/non-contributing counts reported.

Confidence intervals reuse the frozen convention: **1000 subject-level bootstrap
replicates, seed `2026`**, 2.5th/97.5th percentiles, degenerate and undefined
replicates reported. Window-level bootstrap is prohibited.

**Claim boundary.** Intervals computed from the frozen OOF predictions quantify
**between-subject variation conditional on the fitted OOF calibration
procedure**. They are **not** a complete bootstrap of calibrator re-fitting
uncertainty unless a future execution protocol explicitly performs fold
re-fitting inside each bootstrap replicate. **No U1 output may imply that
millions of windows provide independent inferential support**; the inferential
unit is the subject, and there are 12.

## 11. Routing thresholds — exact rule, two artifacts

**No routing threshold is chosen in this task.** The rule is frozen here.

### 11.1 The frozen empirical order statistic

Let `N` be the number of eligible uncertainty values and `c_star` the target
coverage. Then:

```
k = ceil( c_star * N )
```

Sort the eligible rows ascending by the deterministic key `(u, stable_id)`, and
take

```
u_star = value of the k-th ordered row, 1-based
```

Local acceptance remains `u <= u_star`. Because acceptance is inclusive at
`u_star` and `k >= c_star * N`, **achieved empirical coverage is always
`>= c_star`.**

If several rows tie at `u_star`, every tied row satisfying `u <= u_star` is
accepted, so achieved coverage **may exceed** `c_star`. Ties can only increase
achieved coverage, never decrease it below target.

**Worked check at the frozen PRIMARY size.** With `N = 473,897` and
`c_star = 0.90`, `c_star * N = 426,507.3`. A "lower"-style convention lands on
`426,507`, giving coverage `0.8999993669...`, which is **below** target. The
frozen rule takes `k = ceil(426,507.3) = 426,508`, giving
`0.9000014771... >= 0.90`. This is exactly why the earlier `numpy` `'lower'`
quantile wording was insufficient and has been **removed**; no library-default
quantile convention governs `u_star`.

**Required reporting** for every derived threshold:

```
target_coverage
achieved_coverage
accepted_count
threshold_tie_count
```

### 11.2 `u_star_dev` — development routing threshold

Derived by §11.1 from the **pooled OOF calibrated uncertainties**.

Used for: U1 DEVELOPMENT routing evidence; accepted-risk evidence; class-aware
escalation evidence; the human retention review; and later subject-disjoint
DEVELOPMENT temporal work.

### 11.3 `u_star_deploy` — deployment configuration threshold

After the selected family is fitted once on all 12 VALIDATION subjects (§7B),
that final calibrator is applied to its own fit-development population **solely
to instantiate** the prespecified `c_star = 0.90` deployment configuration.
`u_star_deploy` is derived by the same rule in §11.1.

**This pass is parameterisation, not evaluation.** Calibration or routing
performance produced by the final all-subject fit **must not** be reported as U1
scientific DEVELOPMENT evidence. Persist the threshold and its provenance only.

Future TEST or unseen-subject coverage **is allowed to differ from 0.90**;
`c_star` constrains the configuration, not the world.

`u_star_dev` and `u_star_deploy` are **different artifacts with different claim
status** and are never interchanged.

### 11.4 `c_star = 0.90` — what it is and is not

`c_star = 0.90` is retained because it was selected **prospectively, before any
U1 result**. It is described as:

> an **a-priori operational design assumption / reference operating point**

and **not** as measured deployment capacity.

With a 5-second stride, 10% window escalation corresponds to roughly **72
escalated windows per hour per channel before temporal aggregation**. Actual
network cost depends on payload, routing and temporal logic, and will be
**measured later in E1**, not asserted here.

**No change to `c_star` is authorised by later U1 results.**

### 11.5 Why a fixed point on a reported grid, not an optimised scalar

Optimising `risk + lambda * escalation` requires a `lambda` that cannot be
justified before results exist and would in practice be chosen after seeing
them. With 12 subjects the argmin is genuinely noisy. A frozen grid plus one
a-priori retained point lets the human judge the whole trade-off while the
retained point cannot have been tuned.

### 11.6 Reporting guards — flags, never automatic re-selection

Both guards **report and stop for human review**; neither adjusts any threshold.

- **Asymmetric abstention:** if at `c_star` the positive escalation fraction
  exceeds the negative escalation fraction by a factor greater than
  `asymmetric_abstention_ratio = 3.0`.
- **Routing-calibration inadequacy:** if predicted and observed accepted risk
  differ by more than `accepted_risk_agreement_tolerance = 0.02` absolute.

## 12. Cold-start and challenge reporting

**Cold start.** M2's 0–5 minute limitation is frozen and inherited. U1
introduces **no** special cold-start threshold and performs **no** post-hoc
cold-start repair. U1 *reports* OOF calibration and selective behaviour by the
already frozen strata:

| Stratum | PRIMARY windows |
|---|---|
| 0–5 minutes | 1,798 |
| 5–60 minutes | 19,637 |
| over 60 minutes | 452,462 |

The 0–5 minute stratum contains a single ischemic positive, so its
discrimination-dependent quantities are undefined and reported as counts without
a confidence interval. The retained routing point is **not** altered because of
any stratum result.

**Challenge reporting** is permitted with the frozen evidence levels and
denominators of the metrics protocol, never merged into PRIMARY:

| Subset | Windows | Evidence level |
|---|---|---|
| Rate-related | 4,973 | `quantitative_secondary` |
| Axis shift | 3,000 | `quantitative_secondary` |
| Conduction change | 164 | `exploratory_descriptive` |

Reported per subset: accepted false-positive rate at `u_star_dev`, escalation
fraction, false-positive count, contributing-subject count and denominator.
Conduction change has one contributing subject: descriptive `FP / N` only, **no**
bootstrap interval, **never** a selection input or headline. Challenge selection
digest `49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a`.

## 13. TEST firewall — absolute

TEST remains **sealed and unopened**. For U1 specifically:

- no TEST path is resolved;
- no TEST subject enters calibration, fitting, fold assignment or evaluation —
  including the final all-validation fit of §7B, which uses **only** the 12
  VALIDATION subjects;
- no TEST prediction, label, score or metric is read;
- no B4 sealed TEST result is reopened;
- no calibrator is fitted on TEST;
- no routing threshold is derived from TEST;
- no TEST reliability or risk-coverage analysis is produced.

The 12 TEST subjects are excluded **by construction**: the permitted calibration
subject set is the frozen VALIDATION set, the two are disjoint in the frozen
split manifest, and any subject outside the VALIDATION set is **refused** by the
protocol validator rather than filtered silently.

Every U1 artifact records `test_accessed: false` and
`sealed_test_state: "unopened"`. **Completion of U1 does not authorise TEST.**

## 14. Relation to M2 and to downstream experiments

- U1 consumes persisted M2-G output **read-only**.
- Calibration does **not** alter patient-memory history, prototypes, gate
  decisions, admission decisions or any M2 trajectory.
- Routing decisions do **not** retroactively affect the M2 replay that produced
  the development evidence.
- **U1 uncertainty is never inserted into frozen M2-G.** If a future online
  architecture requires uncertainty to participate in memory admission, that is
  a separate, versioned, separately authorised experiment.
- **No M2 rerun is permitted by this protocol.**

### 14.1 Downstream T1/T2 development calibration rule — frozen

For later **T1/T2 DEVELOPMENT** experiments on the existing VALIDATION subjects,
the calibrated probabilities and uncertainties supplied by U1 **must be the OOF
U1 probabilities** (§7A).

**A VALIDATION subject must never be given a probability produced by a
calibrator that was fitted using that subject.** The final all-validation
calibrator (§7B) is reserved for previously unseen subjects and for future,
separately authorised TEST or deployment use.

## 15. Provenance required of the later U1 execution run

Execution Git SHA and dirty state; interpreter; package count and dependency
digest; runtime START / COMPLETION / PRE_PROMOTION observations under the
existing runtime-integrity sentinel; this protocol's SHA-256; the M2 retention
decision SHA; the M2-G arm-result and lock SHAs; the M2 suite SHA; the split
hash; the PRIMARY, CHALLENGE and FULL-REPLAY population identities; the frozen
fold assignment and its digest; per-fold fitted parameters; the selected family
and the OOF NLL values that selected it; the final deployment calibrator record
of §7B including its artifact SHA; the saturation census; `u_star_dev` and
`u_star_deploy` each with `target_coverage`, `achieved_coverage`,
`accepted_count` and `threshold_tie_count`; `test_accessed: false`;
`sealed_test_state: "unopened"`.

## 16. Success / retention criterion — prespecified

U1 produces **no automatic retention**. A human performs the retention review.
The prespecified conditions the human will weigh, all computed on **OOF**
evidence:

1. the saturation census is within its frozen bound (§3.1);
2. pooled OOF **Brier and NLL are both lower** than the uncalibrated baseline;
3. accepted risk at `c_star = 0.90` (via `u_star_dev`) is strictly lower than
   the full-coverage (`c = 1.00`) error rate;
4. no asymmetric-abstention flag (§11.6);
5. no routing-calibration inadequacy flag (§11.6).

Failing any condition does **not** authorise a repair, a re-fit, a different
calibrator, a different coverage target or a different uncertainty definition.

Improved ECE alone is **not** a success criterion, and no U1 result may be
described as clinical safety, statistical significance, or generalisation.

## 17. No automatic transition

Completion of U1 does **not** begin U2 conformal prediction, T1, T2, E1 edge
benchmarking, or episode-level work. U1 is scientifically complete without
conformal prediction. Whether U2 justifies its schedule is a separate human
decision made after U1 closes.

## 18. Implementation scope

This protocol is frozen in a design-only change set: the protocol document, a
structural validator module incapable of touching real development data, and
synthetic tests. The reviewed execution implementation is a **separate** later
change set.
