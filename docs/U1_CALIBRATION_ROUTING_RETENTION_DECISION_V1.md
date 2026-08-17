# U1 Calibration / Selective-Routing Retention Decision V1

## 0. Nature of this document

**THIS IS A HUMAN GOVERNANCE DECISION, NOT A NEW SCIENTIFIC EXPERIMENT.**

It records a **split retention judgement** over the frozen U1-v1 development
evidence produced by the completed canonical attempt `u1-v1-development`
(scientific identity `U1_selective_v1`). No metric here was recomputed; every
value is read from immutable promoted artifacts.

This document is not a U1 rerun, not threshold tuning, not coverage tuning, not
recalibration, not a new calibrator-family comparison, not statistical
significance testing, and not a TEST decision.

The canonical run deliberately did not make this decision and continues to
record `automatic_retention: false`, `human_review_required: true`,
`automatic_u2_transition: false`, `test_accessed: false` and
`sealed_test_state: "unopened"`. **This document does not modify it.**

## 1. Bound identities

| Artifact | SHA-256 |
|---|---|
| Git execution tree | `233a474aca14dac4bad7d213eae46cd07836928a` |
| U1-v1 protocol | `d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b` |
| M2 update-policy retention decision | `da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47` |
| **`U1_RESULT.json` (file)** | **`649631cbf5188731d006f533997cfe28df4f5acb79e7693514e86ad0cef0cb12`** |
| **`U1_EXPERIMENT_LOCK` (canonical self-digest)** | **`7f4dd1505919e23a598773736dc57e2d1b4d360f496b45acdf2028ed0574b1b6`** |
| `U1_EXPERIMENT_LOCK.json` (file) | `eca664ced24cdbc3f28b1ef339c99f0e37ec7185a034a7c7ed28b7f773d1ebfc` |
| `U1_SATURATION_CENSUS.json` | `0ee3e80dc86d48d89dbb2e3a9f3d1ddb8263670a636335853e36bd91a710e5de` |
| `U1_FOLD_MANIFEST.json` | `6de92e8d86f8fed03357a5daf6a5c33a5c97df06d5cefa846ee2d453e49ed82a` |
| `U1_OOF_CALIBRATION.json` | `c6a48fcd5e14cbe9d543eaa1d81328a8eade41343cc9629c8f3f8b78eee47da2` |
| `U1_FAMILY_SELECTION.json` | `cbf8dec21defa18143050cd74b5c08916a17f07279578541e234fac3cdce70d1` |
| `U1_OOF_RESULT.json` | `dbe546ecb4da1b6a974ace6549803ac9a6894db321707da25cff39d9bca0e7e6` |
| `U1_DEPLOYMENT_CALIBRATOR.json` | `acec97c1ebd3bed459ad2d75204b6c82f274b248edbb1d779b844bd46c62fdc1` |
| OOF evidence-store **content** (canonical payload) | `b95f484c9a7b08447f5a5d4330528136e040cf05acb9e2f7e54305e20bdffcba` |
| PRIMARY OOF row evidence (473,897 rows) | `d30ee58f72e88f09ec940b6a2b284a5c2030f32c2fb8045e1c64b2fb08e60de2` |
| CHALLENGE OOF row evidence (8,137 rows) | `52fffe2fbef91da55679615d480da2de600ad9acd05b173fe89f0673297e5bec` |
| Fold assignment | `f0f5d8e93a757c0975f3613879d11f53970befa6c6bc57578b1a084c92c85b9a` |

Canonical attempt id: `u1-v1-development`. Scientific identity
`U1_selective_v1`. Final status `COMPLETE`, `git_dirty: false`.
Partition accessed: `validation`.

**Sealed TEST: unopened.** `test_accessed: false`. The 12 TEST subjects were
refused by name, not filtered, and no TEST subject appears in the promoted
per-row evidence.

Some entries above are **file** digests and some are the artifact's own
**canonical payload** self-digest. They differ by design.

## 2. Decision

This is a **split** retention decision. Calibration is retained. The
window-level symmetric selective router is **not** retained.

| Component | Retained |
|---|---|
| Platt calibration on the recovered / clamped logit (`platt_logistic_on_recovered_logit`) | **true** |
| Subject-disjoint OOF calibrated probabilities, for downstream DEVELOPMENT | **true** |
| Final all-VALIDATION Platt calibrator, for genuinely unseen subjects / separately authorised TEST / deployment parameterisation | **true** |
| Frozen symmetric window-level selective-routing policy at `c_star = 0.90`, as the downstream or final operational router | **false** |
| `u_star_dev` as a final routing threshold | **false** |
| `u_star_deploy` as a final routing threshold | **false** |

| Property | Value |
|---|---|
| Selection basis | **DEVELOPMENT evidence only** |
| `test_accessed` | **false** |
| `statistical_significance_claim` | **false** |
| U1 rerun permitted | **no** |

**No artifact is deleted, rewritten or reset.** The rejected router is not
erased: `u_star_dev`, `u_star_deploy` and the entire risk-coverage curve remain
**immutable U1 DEVELOPMENT / ablation evidence**.

## 3. Calibration retention evidence

Read from `U1_OOF_CALIBRATION.json`, 473,897 rows, 12 subjects, clamp
`1e-07`, 15 bins in both binnings, no library quantile used.

| Family | NLL | Brier | ECE equal-width | ECE equal-mass |
|---|---|---|---|---|
| Uncalibrated raw score (reference) | 0.23170495211589118 | 0.0635671818303644 | 0.06384438607391933 | 0.062464338934517576 |
| **Platt on recovered logit (OOF)** | **0.14370784818131235** | **0.040344375976781484** | **0.016990579896181784** | **0.018603649015666395** |
| Approximate temperature-only (OOF) | 0.19169200154056643 | 0.05864710970657676 | 0.07404013328988358 | 0.07404013328988358 |

Frozen family-selection result: **`platt_logistic_on_recovered_logit`**.
Selection basis: **lower pooled OOF NLL only**
(`nll_difference_platt_minus_temperature = -0.04798415335925407`,
`tie_within_tolerance: false`, tolerance `1e-4`). AUPRC, Brier, ECE, routing
risk and challenge evidence were **not** selection inputs, and no weighted score
was used.

Two honest qualifications, recorded rather than smoothed over:

- The uncalibrated baseline is a **reference, not out-of-fold evidence**
  (`out_of_fold: false`, `development_evidence: false`). It is the raw persisted
  M2-G score treated as a probability. The Platt row is out-of-fold; the
  baseline row is not, and the two are not a matched comparison.
- The temperature-only comparator is **approximate**: true logits are not
  persisted, only `sigmoid(logit)` in float32-then-widened form, so proper
  temperature scaling is unavailable
  (`true_logit_temperature_scaling_performed: false`). Its two ECEs are
  identical because it over-predicts in every bin of both binnings, so both
  collapse to the same global mean gap. That is real behaviour, not a defect.

**No statistical-significance claim is made.** No hypothesis test, confidence
interval or paired subject-level test was performed on the family comparison,
and none is implied by the arithmetic.

## 4. Decision equivalence

Frozen classification threshold `0.7554003000259399`, inherited from the
retained M1L arm through M2-G and **not** selected here.

Total calibration-induced classification disagreements: **0** — across all 12
folds and in the final all-validation calibrator over 473,897 rows
(`row_for_row_identical: true`, `threshold_selected_here: false`,
`calibrated_boundary_is_a_new_threshold: false`).

Calibration is therefore retained as a **probability / uncertainty
transformation only**. It does **not** replace, retune or reinterpret the frozen
detector threshold, and it changes no detection decision.

## 5. Routing result at the frozen `c_star`

Frozen `c_star = 0.90`. Threshold rule
`empirical_order_statistic_ceil_1_based`, sort key `(uncertainty, stable_id)`,
acceptance `u <= u_star`, `library_quantile_used: false`.

| Quantity | Value |
|---|---|
| `u_star_dev` | `0.12763774358328017` |
| rank / eligible | 426,508 / 473,897 |
| achieved coverage | 0.9000014771142253 |
| threshold tie count | 1 |
| accepted count | 426,508 |
| escalation fraction | 0.09999852288577471 |
| accepted risk (observed) | 0.024770930439757286 |
| predicted accepted risk | 0.018087238783122118 |
| risk-agreement absolute error | 0.006683691656635168 |

The **calibration-agreement guard PASSED**: `0.006683691656635168` against the
frozen tolerance `0.02`.

However:

| Quantity | Value |
|---|---|
| positive-label escalation fraction | 0.5167375624190864 |
| negative-label escalation fraction | 0.0800696045937263 |
| **ratio** | **6.453604523726777** |
| frozen asymmetric-abstention limit | 3.0 |
| result | **GUARD RAISED** |

Accepted-population discrimination at `u_star_dev`, exactly as persisted:

| Quantity | Value |
|---|---|
| accepted sensitivity | `0.0007654037504783774` |
| accepted specificity | `0.9997091737650701` |
| accepted PPV | `0.06201550387596899` |
| accepted NPV | `0.9755053602546092` |
| **accepted positive-label windows** (the sensitivity denominator) | **10,452** |
| accepted true-positive detections | 8 |
| accepted false negatives | 10,444 |
| positive-label windows in the PRIMARY population | 21,628 |

Accepted sensitivity is computed over the **accepted** positive-label windows:
`8 / (8 + 10,444) = 8 / 10,452 = 0.0007654037504783774`. The PRIMARY total of
**21,628 positive-label windows is the population count, not this denominator** —
the remaining 11,176 positive-label windows were escalated and are therefore not
part of the accepted population at all.

Subject-level (n = 12, the inferential unit): macro coverage 0.8789353968217225,
macro escalation 0.1210646031782775, macro accepted risk 0.025130228890384968,
macro accepted sensitivity 0.001035733757051165 (9 of 12 subjects contributing),
macro accepted specificity 0.9995026715988491. Bootstrap 1000 replicates, seed
2026, subject unit, 0 undefined: accepted-sensitivity 95 % interval
[0.0, 0.003795162292345004], accepted-risk [0.005127610957336603,
0.04904173942575431], coverage [0.8263853590782775, 0.9524000003363469]. The
claim scope is **between-subject variation conditional on fitted OOF
calibration**; folds were not refitted per replicate.

**The window router does not satisfy the human retention requirement.**

## 6. Interpretation — what the low accepted risk does and does not mean

The low overall accepted risk **MUST NOT be interpreted in isolation.** The
DEVELOPMENT population is strongly class imbalanced: 21,628 positive-label
windows against 452,269 negative-label windows. An accepted population that is
almost entirely negative will show low error rate and high specificity whether
or not the router is useful.

At `c_star = 0.90` the router disproportionately escalates positive-label
windows — 51.67 % of positive-label windows against 8.01 % of negative-label
windows. Of the **21,628 positive-label windows in the PRIMARY population**, the
router escalates 51.67 %, leaving **10,452 positive-label windows locally
accepted**. Only **8** of those accepted positive-label windows are
**true-positive detections** and **10,444** are false negatives, yielding
accepted sensitivity `8 / 10,452 = 0.0007654037504783774`.

A positive-label window is not the same thing as a true-positive detection: the
label says ischemia is present, the detection says the frozen classifier fired.
Almost all detected ischemia is routed away; what stays local is overwhelmingly
the easy negative mass.

Therefore this decision does **NOT** claim:

- balanced selective safety;
- improved local ischemia detection;
- safe autonomous local classification;
- universally improved decision quality.

The prespecified class-aware guard **successfully exposed this behaviour**. It
did what a prespecified guard exists to do, and the correct response is to
decline retention of the router, not to weaken, retune or reinterpret the guard.

## 7. Positive secondary finding — preserved, not overridden

Challenge routing evidence at `u_star_dev`, out-of-fold, not merged into the
primary denominator, `is_selection_input: false`, `is_headline_metric: false`.

| Subset | N | all-window FP | accepted | accepted FP | accepted FPR | escalation fraction |
|---|---|---|---|---|---|---|
| Rate-related | 4,973 | 2,029 | 2,254 | 1 | `0.00044365572315882877` | `0.5467524633018299` |
| Axis shift | 3,000 | 236 | 2,428 | 0 | `0.0` | `0.19066666666666665` |
| Conduction change | 164 | 5 | 157 | 0 | `0.0` | `0.04268292682926833` |

Conduction change is **exploratory / descriptive only** — a single contributing
subject, 164 windows — and carries no weight.

Interpretation: **calibrated uncertainty is useful for identifying many
confounder false alarms for escalation.** Rate-related windows, the worst
confounder family in M2, escalate at 54.68 % and leave only one accepted false
positive out of 2,029 all-window false positives.

**This benefit does NOT override the asymmetric-abstention failure.** A router
that escalates most confounder false alarms *and* most true positives has not
been shown to be operationally sound; the same uncertainty ordering produces
both behaviours.

## 8. The routing curve remains an ablation

The entire risk-coverage curve — grid `0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
0.90, 0.95, 0.99, 1.00` — remains **frozen descriptive DEVELOPMENT evidence**.

**No post-hoc coverage point is chosen here.** Not 0.95, not 0.99, not any
other. **No alternative `u_star` is chosen here.** `u_star_dev` and
`u_star_deploy` are not selected downstream merely because some other coverage
point would look more favourable. Choosing a coverage point after seeing the
curve is exactly the selection this protocol forbids.

## 9. Downstream DEVELOPMENT input

For future T1 / T2 DEVELOPMENT work on the existing VALIDATION subjects:

**USE** the selected-family **OOF Platt calibrated probability** for each row.

**DO NOT USE** the all-validation final calibrator on VALIDATION subjects — on
those 12 subjects it is in-sample.

**DO NOT USE** `u_star_dev` as an operational routing gate.
**DO NOT USE** `u_star_deploy` as an operational routing gate.

The retained downstream DEVELOPMENT input is therefore:

> frozen detector decision
> **+** subject-disjoint OOF Platt calibrated probability / uncertainty
> **+** retained M2-G patient state

as applicable to the future temporal protocol.

U1 development evidence remains **development-optimistic**: VALIDATION already
selected `tau` upstream, and cross-fitting corrects subject self-calibration
only.

## 10. Final deployment calibrator status

Retained as the **prospective calibration mapping for genuinely unseen
subjects**, and for a separately authorised TEST or deployment.

| Parameter | Value |
|---|---|
| `a` | `0.3715906808641229` |
| `b` | `-1.7662772879067046` |
| fit subjects | 12 |
| fit rows | 473,897 |
| optimizer status | 0 (`CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`) |
| calibrated boundary | `0.20631829355583678` |
| decision disagreements | 0 |

Its in-sample calibration performance remains **NON-EVIDENCE**
(`in_sample_performance_reported: false`,
`in_sample_performance_claim_authorised: false`, `is_evaluation: false`,
`is_parameterisation: true`). It is **parameterisation, not evaluation**, and
must never be reported as DEVELOPMENT evidence.

Its accompanying `u_star_deploy = 0.12914217081334087` is
**configuration provenance only** and is **NOT retained as the final routing
policy**.

## 11. Future routing direction

**No new routing rule is implemented by this decision.**

The human scientific direction is: window-level calibrated probability is
retained; **final edge/cloud routing should be reconsidered prospectively AFTER
temporal reasoning**, where longitudinal WATCH / EVENT / RECOVERY evidence can
participate in routing. A single-window uncertainty ordering is not the right
object to route on when the downstream system reasons over time.

This decision does **NOT** freeze any future routing algorithm.

Nothing here authorises a new U1 threshold, class-specific thresholds,
positive-always-cloud logic, temporal routing or conformal routing. Each would
require its own separate prospective protocol if and when undertaken.

## 12. U2

**U2 conformal prediction does NOT automatically begin.** U2 remains optional.

The next core scientific block after this retention decision is the prospective
temporal-reasoning design. No automatic transition occurs here.

## 13. Scope

No U1 artifact is modified by this decision. **No U1 rerun is permitted** — no
`recovery1`, no alternate root, no timestamp/uuid/random suffix, no repair of
the promoted artifacts. **No M2 rerun is permitted.** M2-0 and M2-G remain
immutable.

The rejected router is preserved, not deleted. It remains the frozen ablation
that makes the retained calibration's scope legible: calibration is retained
*because* it is a faithful probability transformation that changes no decision,
and the router is rejected *because* the class-aware guard showed what routing
on that probability actually does at `c_star = 0.90`.

The sealed TEST partition remains **unopened**, and completion of U1
DEVELOPMENT does **not** authorise TEST.
