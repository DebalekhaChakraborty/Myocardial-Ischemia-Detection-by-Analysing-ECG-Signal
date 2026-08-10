# B4 Sealed-Test Deferral Decision V1

## 1. Status

Phase 3B-2 architecture selection is **complete**. The global short-window
encoder is **B4-B** (`B4B_cnn_transformer_v1` / `B4BTransformerCNN`), frozen by
`docs/B4_GLOBAL_ENCODER_SELECTION_V1.md`
(SHA-256 `1300e7ad641df9137e1722771e5d3932cae0fc4d244047b7c8a5070f151f74bb`).

Under the earlier Handbook choreography, one-shot sealed-test access for the
selected encoder became **eligible** at that point.

## 2. Decision

**B4-B sealed-test access is eligible but is intentionally NOT authorized now.**

The B4 sealed test **remains unopened**. No `TEST_ATTEMPT` exists.

## 3. Reason

The remaining claim-bearing architecture still contains prospective development
choices that are not yet frozen:

- physiology fusion (P1);
- patient memory (M1);
- contamination-safe adaptation (M2);
- calibration and selective routing;
- longitudinal temporal reasoning (T1/T2);
- episode logic.

The sealed test draws on a fixed, small set of held-out subjects. Observing
their outcomes now — even once, even only for B4-B — would put that information
inside the design loop for every component listed above. Each subsequent design
decision would then be partly informed by test subjects, and the final test
result would no longer be a clean estimate of generalisation.

Deferring costs nothing scientifically. Spending the test early cannot be undone.

## 4. Binding consequences

1. **No P1, M1, M2, U1, U2, T1 or T2 design decision may use test information**,
   directly or indirectly, including as a tie-breaker or sanity check.
2. **B0–B3 historical test evidence remains closed** and must not be used to
   tune, select, initialise or motivate any later component.
3. There is **no automatic test access** at any future milestone. Access requires
   an explicit, separate governance decision.

## 5. Preferred future choreography

One **coordinated final test event**, executed only after the claim-bearing
system architecture and all development-only decisions are frozen.

A future explicit governance decision may alter this choreography if
scientifically justified — for example if an intermediate result makes a staged
evaluation genuinely necessary. Such a change must be argued and recorded before
any access, never assumed.

## 6. Scope limits of this document

This document does **not** modify the frozen test subjects, the test labels, the
benchmark definition, any protocol, or any existing scientific artifact. It
records a governance choice about *when* the sealed test may be opened, and
nothing else.

| | |
|---|---|
| B4 sealed test | **UNOPENED** |
| `TEST_ATTEMPT` | absent |
| Authorizes test access | **no** |
