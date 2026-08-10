# B4 Global Encoder Selection V1

## 1. Purpose

This document records and freezes the Phase 3B-2 selection of the CardioSentinel
**global short-window encoder** from the three prospectively frozen candidates
B4-A, B4-B and B4-C.

It is a **decision and provenance record**. It generates no new scientific
evidence: every number below is read from an already-frozen, digest-bound
artifact. Arithmetic differences and ratios are labelled as **derived
descriptive comparisons** and carry no inferential claim.

**No B4 sealed-test evidence was used, viewed or produced.**

## 2. Evidence hierarchy applied

Per `docs/B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md` and Handbook v1.1 §10.2:

1. pooled validation AUPRC (primary discrimination);
2. subject-macro behaviour;
3. rate-related challenge FPR (quantitative secondary);
4. axis-shift challenge FPR (quantitative secondary);
5. parameters / model size;
6. latency and RAM feasibility;
7. training stability.

Conduction-change challenge evidence is **exploratory/descriptive only** and is
excluded from selection weight (§7).

## 3. Candidate identities

| Official | Experiment | Architecture | Params | FP32 payload |
|---|---|---|---:|---:|
| B4-A | `B4_raw_compact_cnn_v1` | `B4CompactCNN` | 87,089 | 348,356 B |
| B4-B | `B4B_cnn_transformer_v1` | `B4BTransformerCNN` | 309,809 | 1,239,236 B |
| B4-C | `B4C_cnn_ssm_v1` | `B4CSSMCNN` | 155,313 | 621,252 B |

## 4. Frozen evidence sources

| Artifact | SHA-256 |
|---|---|
| B4 protocol | `f6f5e9ed728c86a9b2bd75b2327b9199f0e097b91387525a192c212e6771b28b` |
| Architecture-selection protocol | `986bc166f7f4a787423e1ac33cad65342ae7a700f85bfd8bb9d0291f64d2a0dc` |
| Resource-benchmark protocol | `9184f54eb2b80fd495460d0a5c8989cdc6b923ed992a87ea18253e836f4c4b98` |
| Validation-challenge protocol | `44df775f43301a782d3acd48fc4b3cd9358c07d4ed45d270fcb2763200761926` |
| **Official resource suite** | `2292dbf102091ca002e6a8fb5acc12a72aa69374eba57bf8a45b56f780333f6d` |
| **Official validation-challenge suite** | `f063c67dc00e85fb38bd20fc98ecb333bf17d27540ff96c7a47de47eb1c0bedb` |
| Challenge selection identity | `49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a` |
| B4-A experiment lock | `ea1e1d76365b0cd52ba1b7f022f22f85af848bbdc002beeae806eda9c39a78fa` |
| B4-B experiment lock | `58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b` |
| B4-C experiment lock | `22ba491b7219ee94af7bb64b5cab57e96b4a0aea84a237d19e6608712738e959` |
| B4-A checkpoint | `3a33cfb3c05e0f26fc8bc9c3bb826710215921da11b1ecd3a7ea92c3c57e9175` |
| B4-B checkpoint | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |
| B4-C checkpoint | `d590a3ab657ef4dec9c63b01754b04851937547f1bc85083b3c085ab529ab3ec` |

Development split `66e25d77…ced7`, feature corpus `f18785d5…6eb5`. All three
locks record `test: null`.

## 5. Candidate comparison

Primary validation partition: 473,897 windows, 21,628 positive, 452,269
negative, 12 subjects. Scores are **uncalibrated sigmoid model scores**, not
calibrated probabilities. Each candidate is evaluated at its own locked
threshold.

| Metric | B4-A | **B4-B** | B4-C |
|---|---|---|---|
| Pooled validation AUPRC | 0.3156014611186772 | **0.38053499010488423** | 0.3377705149052735 |
| Pooled AUROC | 0.8675598293803359 | 0.892761910201445 | 0.8906614915366206 |
| Pooled F1 | 0.3740814436848638 | 0.3914730391076231 | 0.3903703569421533 |
| Pooled sensitivity | 0.4389680044386906 | 0.41053264287035324 | 0.4978268910671352 |
| Pooled specificity | 0.9565811497139977 | 0.9671545031828404 | 0.9496582785908386 |
| Pooled MCC | 0.3436873632525397 | 0.3613835902960735 | 0.3641926396794557 |
| Subject-macro AUPRC (9 contributing) | 0.3658236963081271 | 0.40063630025780333 | **0.4033236569167703** |
| Subject-macro F1 (12) | 0.22402765846288297 | 0.2202080648100302 | 0.2697452138232212 |
| Rate-related FPR (n=4,973; 4 subj) | 0.34566659963804547 | **0.331188417454253** | 0.4651116026543334 |
| Axis-shift FPR (n=3,000; 8 subj) | 0.102 | **0.06166666666666667** | 0.11766666666666667 |
| Conduction FPR (n=164; **1 subj**) | 0.036585365853658534 | 0.0 | 0.018292682926829267 |
| Parameters | **87,089** | 309,809 | 155,313 |
| FP32 payload | **348,356 B** | 1,239,236 B | 621,252 B |
| Median latency | **3.274761 ms** | 4.1613225 ms | 14.4363955 ms |
| p95 latency | **3.527782 ms** | 4.33681 ms | 15.315093 ms |
| Peak RSS | **301,044 KiB** | 305,340 KiB | 304,280 KiB |
| Selected epoch | 4 | 2 | 2 |
| Locked threshold | 0.8274613618850708 | 0.8329097628593445 | 0.7442968487739563 |

Bold marks the best observed value per row; it is a reading aid, not a claim of
significance.

## 6. Decision

**Selected global encoder: B4-B.**

| | |
|---|---|
| Official model | `B4-B` |
| Experiment ID | `B4B_cnn_transformer_v1` |
| Architecture | `B4BTransformerCNN` |
| Experiment-lock SHA | `58e44a09ce3ebffecfcd49d957acfa368fc03b534fdcd990aedb9b6b0e9bda7b` |
| Checkpoint SHA | `b1301723909c641a0014c31f6daa9549d47ab231f0b07483e0de729aff5591c9` |
| Locked validation threshold | `0.8329097628593445` |

This is a **development architecture-selection decision** made under a frozen
benchmark and a specific implementation. It is **not** a claim of statistical
superiority, and **not** a claim that Transformers are generally superior to CNNs
or state-space models. No hypothesis test, confidence interval or paired
analysis supports these comparisons; none was performed.

## 7. Rationale

### A. Primary discrimination

B4-B has the highest pooled validation AUPRC.

*Derived descriptive comparisons:*
- B4-B − B4-C = **+0.0427644752** absolute
- B4-B − B4-A = **+0.0649335290** absolute

These margins are materially larger than the Handbook's illustrative
0.001–0.003 AUPRC band, which would not justify additional edge cost. **No
statistical significance is implied by this arithmetic.**

### B. Subject-macro behaviour

B4-C has the numerically highest subject-macro AUPRC (0.4033236569167703 versus
B4-B's 0.40063630025780333), a derived difference of **+0.0026873567** for B4-C
across 9 contributing subjects.

This is treated as a **small numerical macro advantage for B4-C**, not as
evidence sufficient to overturn B4-B's substantially higher pooled AUPRC and its
better rate and axis robustness. The two are **not** described as statistically
equivalent: no paired subject-level analysis has been performed, and such a
statement would require one.

### C. Rate-related challenge

B4-B has the lowest rate-related validation FPR: 0.331188417454253, versus B4-A
0.34566659963804547 (derived Δ 0.0144781822) and B4-C 0.4651116026543334
(derived Δ 0.1339231852). Quantitative secondary evidence over 4,973 windows and
4 subjects.

### D. Axis-shift challenge

B4-B has the lowest axis-shift validation FPR: 0.06166666666666667, versus B4-A
0.102 (derived Δ 0.0403333333) and B4-C 0.11766666666666667 (derived Δ
0.0560000000). Quantitative secondary evidence over 3,000 windows and 8
subjects.

### E. Conduction-change — reported, not weighted

All values are reported in §5 for completeness. **Conduction evidence is
exploratory/descriptive only because the validation challenge stratum is
supported by exactly one subject (164 windows).** It was not bootstrapped.

**B4-B's zero observed conduction false positives is explicitly NOT a reason for
this selection** and must not be cited as one.

### F. Resource trade-off — stated honestly

**B4-A is the smallest and fastest candidate**, at 87,089 parameters, a 348,356-byte
FP32 payload and 3.274761 ms median latency. That advantage is real.

B4-B costs 309,809 parameters (derived 3.557× B4-A), a 1,239,236-byte payload,
and 4.1613225 ms median latency — a derived increase of **≈0.887 ms/window** over
B4-A. Peak RSS differs by ~4.3 MiB across all three (301.0–305.3 MiB), so memory
does not separate them at batch size 1.

These are measurements on the **frozen resource benchmark host** (Intel Xeon
@2.20 GHz, single intra-op thread, batch 1). They are **not** Raspberry Pi or any
other edge-device latency, and must not be quoted as such. Edge feasibility
remains an open question for a later deployment study.

The judgement recorded here is that ≈0.887 ms/window on the benchmark host is
accepted in exchange for materially higher pooled AUPRC and better rate/axis
robustness.

### G. Why not B4-C

B4-C has fewer parameters than B4-B (155,313; derived 1.995× smaller) and the
small subject-macro advantage of §B. Against that it shows lower pooled AUPRC,
materially worse rate FPR (+0.134 absolute over B4-B), materially worse axis FPR
(+0.056 absolute), and **≈3.47× B4-B's median latency** under the frozen
implementation. Its parameter advantage does not justify selecting it as the
short-window encoder.

Note the latency figure reflects the deliberately unoptimised 79-step Python
recurrence frozen by protocol §18; it is a property of this implementation, not
an inherent bound on diagonal SSMs.

### H. Training stability

| Candidate | Epochs | Selected | Validation AUPRC trajectory |
|---|---:|---:|---|
| B4-A | 8 | 4 | 0.3039, 0.2723, 0.2868, **0.3156**, 0.2327, 0.2403, 0.1985, 0.2384 |
| B4-B | 6 | 2 | 0.3795, **0.3805**, 0.3028, 0.3728, 0.3204, 0.3344 |
| B4-C | 6 | 2 | 0.2869, **0.3378**, 0.3099, 0.2873, 0.2361, 0.2451 |

All three peaked early and then oscillated or declined in validation AUPRC while
training loss continued to fall — the classic overfitting shape. B4-B and B4-C
both selected epoch 2 and degraded afterwards; B4-A selected epoch 4.

The frozen early-stopping policy (patience 4, delta 1e-6) behaved exactly as
intended in all three runs. **No model was tuned, rescued or retrained
retrospectively**, and none may be.

## 8. SSM interpretation — important

**Rejecting B4-C as the short-window global encoder does NOT reject state-space
models from the CardioSentinel architecture.**

B4-C applied an **S4D-inspired diagonal gated SSM** across the 79 local tokens
*within a single 10-second window*, discarding its recurrent state after every
forward call. It is **not Mamba**: the transition is diagonal, time-invariant and
input-independent, so there is no selective mechanism.

The planned **T2 longitudinal SSM** is a distinct experiment operating over
*successive window embeddings* to model minutes of patient history. It remains a
**core planned CardioSentinel component** and is untouched by this decision.

## 9. Limitations

- Single dataset (LTSTDB), 12 validation subjects; 9 contribute to macro AUPRC.
- Conduction challenge rests on one subject and is descriptive only.
- No confidence intervals, hypothesis tests or paired analyses were computed for
  the primary comparison; margins are descriptive.
- Scores are uncalibrated; no calibration study has been performed.
- Latency and RAM are frozen-host measurements, not edge-device measurements.
- Selection reflects this frozen implementation of each architecture, including
  B4-C's deliberately unoptimised recurrence.
- Generalisation beyond this benchmark is unestablished and awaits the sealed
  test.

## 10. Test-access state

| | |
|---|---|
| `test_evidence_used` | **false** |
| B4 sealed test | **UNOPENED** |
| `TEST_ATTEMPT` | absent |

All three candidate locks record `test: null`. **This document does not
authorize sealed-test access.** One-shot test access for the selected encoder is
a separate governance decision to be taken after this decision is reviewed and
merged.

## 11. Phase 3B-2 closeout

**Phase 3B-2 exit gate: SATISFIED upon merge of this reviewed selection
decision.**

- **Selected global encoder:** B4-B — CNN + Tiny Transformer.
- **B4-A:** retained as the efficient required CNN reference.
- **B4-C:** retained as a scientifically useful negative/alternative
  architecture result.
- **T2 longitudinal SSM:** remains planned and unaffected.

## 12. Next gate

The next step is a separate, separately authorised governance decision on
one-shot sealed-test access for the selected encoder. Physiology fusion (P1),
personalization, calibration and longitudinal temporal modelling all remain out
of scope until that gate is resolved.
