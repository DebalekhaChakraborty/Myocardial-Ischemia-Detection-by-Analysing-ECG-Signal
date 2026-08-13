# M2 TRAIN-only receipt canonicalization — provenance record

> **STATUS: PROVENANCE RECORD OF A COMPLETED CANONICALIZATION.**
> This document explains why `docs/M2_GATE_DERIVATION_RECEIPT_V1.json` was
> revised, and proves that no M2 scientific choice changed. It is history,
> not a design proposal — contrast with `RUNTIME_INTEGRITY_SENTINEL_V1.md`
> (design only, not implemented).

## 1. Background

`docs/M2_GATE_DERIVATION_RECEIPT_V1.json` was originally generated
(git commit `510eea0`) while the shared scientific interpreter transiently
carried five distributions (`jmespath`, `botocore`, `s3transfer`, `boto3`,
`awscrt`) installed by a concurrent, unrelated application-side agent
session. That interpreter has since been separated from application work
(`/home/AI_POC/venvs/tactics/bin/python`, isolated); see
`docs/RUNTIME_INTEGRITY_SENTINEL_V1.md` for the full incident record and the
(not-yet-authorized) prospective control it proposes.

The original receipt's `environment.dependency_digest` recorded that mutated
identity, `78e838d2d41a0239f16dbfbaabdddc7efeaffac391ca13a8bbf1475c080cdc25`,
rather than the frozen `tactics` identity,
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

## 2. What was done

A read-only, TRAIN-only derivation verifier
(`src/cardiosentinel/neural/m2_gate_derivation.py`) independently recomputed
every constant in the receipt under the canonical `tactics` runtime, using
only already-materialized TRAIN-partition caches: the M1-v2 stream memory
store, the COMBINED_V1 feature corpus, and the frozen retained M1L
checkpoint. No B4-B encoder inference, no training, and no memory replay was
performed — the causal history was materialized once by the canonical M1-v2
run and only read back.

## 3. Four discrepancies, fully explained

An initial strict-equality comparison found four descriptive/derived
statistics that differed from the committed receipt at floating-point
precision (relative differences ~1e-16 to ~4e-6). Their exact origin was
recovered read-only from the Claude Code session transcript that produced
the original receipt (the derivation scripts were run ad hoc and never
committed to git). All four are **arithmetic-path or forward-pass
reduction-order artifacts, not scientific, row-membership, or population
differences**:

| Field | Root cause | Integer evidence |
|---|---|---|
| `g3_sqi.combined_train_rejection_fraction` | Original computed `1.0 − mean(pass_mask)`; a same-session, separately-written sanity script computed `mean(fail_mask)` for the analogous `refusal_fractions.sqi` field — the two formulas round differently by a few ULP. The **committed receipt already contained both values, disagreeing with each other**, before this investigation. | Both read the identical 86,061 / 2,208,431 rejected-row count. |
| `train_only_sanity.refusal_fractions.normal_evidence` | Same `1−pass` vs `mean(fail)` relationship. | Identical 1,187,523 / 2,208,431 count. |
| `train_only_sanity.refusal_fractions.morphology` | Same `1−pass` vs `mean(fail)` relationship. | Identical 53 / 2,208,431 count. |
| `g4_normal_evidence.descriptive_distribution.min` | Original scored the 280,839-row PRIMARY TRAIN background-negative population in a **dedicated** batched forward pass (batch size 4096, via `m1_store.locate_rows`/`store.gather`). Scoring that same population as a subset of a full-timeline pass changes PyTorch/BLAS GEMM reduction order, which is amplified by `sigmoid` near its tail. | The same row, `ltstdb:s30801:0:8271250:8273750`, attains the minimum in every configuration tested (batch sizes 1, 4096, 8192, and the dedicated pass); batch sizes 1, 4096 and 8192 all reproduce the frozen minimum bit-exactly. |

`m2_gate_derivation.py` now reproduces every receipt field using the exact
historical arithmetic path (not a cleaner-looking equivalent), and two
independent runs of it produce bit-identical output.

## 4. What did not change

Every frozen scientific decision quantity reproduced **bit-exactly** and is
unchanged: all six G3 Q99 bounds, `NORMAL_EVIDENCE_THRESHOLD =
0.0002997174742631614`, the G5 60-second re-armable refractory semantics,
G6 inclusion, the retained arm (`M1L_long_memory_v2`), the core arms
(M2-0/M2-G), and rollback's exclusion from the claim-bearing core. No row
membership and no population count differs anywhere. No M1 artifact was
touched and M1 was not rerun. No M2 scientific execution occurred. No
VALIDATION or TEST partition was accessed at any point in this
investigation or canonicalization.

## 5. Result

- Superseded receipt: `3befd05dc7e9c51ddfed99078d3020375fd610b328d19e64fc7ee3cc745f398e`
- Canonical receipt: `5b14c1a72f34945d59d73f152e8fdeaf929a3be56ad47d94a698bc4bfabd3f24`
- Canonical dependency digest: `b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`
- Scientific values changed: **none**

This is a provenance-only canonicalization, not a new derivation and not a
correction of the earlier receipt's science.
