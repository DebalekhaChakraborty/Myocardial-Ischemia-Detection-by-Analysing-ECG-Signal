# M1 Physical Observation Decision V1

## 0. Nature of this document

A **human governance decision**, not a scientific result. It records the policy
selected after the complete physical census of all 19,007 VALIDATION-extra
windows, and before any M1 arm claim or metric existed.

**Zero** M1 scientific arm claims and **zero** M1 results exist. Both prior
authorizations are consumed. The B4 sealed test is unopened.

## 1. Evidence used

`docs/M1_ATTEMPT2_VALIDATION_ADMISSIBILITY_CENSUS.md`
(`8170068ee3f40875428a28374c8bb1accf4b6fbfd3cc510195f6851f954ce1ee`) and
`docs/M1_STAGE1_ATTEMPT2_FAILURE.md`
(`1bf9539f89d179e8cbf6adb7e578d9f78a9e990fbbf906e5ae3679b93ec1310a`).

**The policy was selected using observable physical waveform properties only.**
No target label, challenge label, quality/event annotation, M1 metric or
sealed-test evidence was used, and none could have been: no M1 metric exists.

## 2. Decision — POLICY B is FROZEN

> An exact-flat 10-second single-channel physical ECG interval that fails the
> existing hard dynamic-variation criterion is a
> **PHYSICALLY UNAVAILABLE SENSOR OBSERVATION**.

It is **not** a valid physiological observation, and **not** merely a
low-confidence physiological observation.

## 3. Why the distinction matters

"Sensor unavailable" and "low-confidence physiological observation" differ in
**kind, not degree**.

- A low-confidence observation still carries physiological information. Deciding
  whether to admit it is a quality/uncertainty question.
- An unavailable sensor carries **no** physiological information. Admitting it at
  any confidence weight is a category error.

Collapsing the two would push a physical-availability question into statistical
quality machinery and leave M1 unable to state whether an observation exists at
all.

**This physical-availability decision is conceptually prior to M2.** M2 remains
responsible for contamination-safe admission of **available** physiological
observations using quality, normality, uncertainty and confounder logic.

## 4. What follows for M1

For a physically unavailable row **no observation occurs**. Therefore:

- **`past_observed_count` MUST NOT increment.**
- **`past_update_count` MUST NOT increment.**
- Both remain unchanged across the unavailable timeline slot.
- No B4-B inference, no representation, no deviation score, no prototype update.
- Real elapsed time and the row's timeline position are preserved.

M1-v2 remains **AVAILABLE + FINITE OBSERVATION → ALWAYS UPDATE**, and remains
**explicitly NOT contamination-safe**: no SQI threshold, no uncertainty gate, no
event gate, no `morphology_valid` gate, no rollback, no M2 behaviour.

An earlier census draft suggested the two counters should diverge across
unavailable slots. **That wording is superseded by this decision** and is not
implemented. Counter divergence belongs to M2, where an observation may be
available yet not admitted.

## 5. Scope limits

This decision admits **exactly one** new state, defined by the **existing** B4
hard criterion `np.ptp(values) <= np.finfo(np.float64).eps` on an individually
read 2500-sample single-channel physical mV segment that already passed interval
validity, header calibration, unit support, mV conversion and finiteness.

It creates **no** near-flat threshold, variance threshold, SQI threshold,
amplitude threshold, morphology threshold or learned availability classifier,
and does **not** use `morphology_valid` to decide availability. That all six
census rows had `morphology_valid = 0` is corroborative evidence only.

All other failure classes — non-finite, miscalibrated, unsupported unit, invalid
interval, wrong channel, wrong dataset identity — **remain fatal** and must never
be reclassified as unavailable.

The frozen **B4-B input contract is not weakened**: for an unavailable row the
encoder is simply never invoked. M1-v2 decides whether a physical observation
exists **before** encoder inference.

## 6. Prospective supersession

`docs/M1_DUAL_MEMORY_PROTOCOL_V1.md` remains **immutable historical evidence**
bound to `08f71c5b54ebd0fcc9c1f26f05d7df2c5a1b0ca5253b8821435a65673ad65253`.
It is superseded prospectively by `docs/M1_DUAL_MEMORY_PROTOCOL_V2.md`.

The supersession is **result-independent**: it was triggered by a physical
sensor property discovered before any arm claim or scientific metric existed.
