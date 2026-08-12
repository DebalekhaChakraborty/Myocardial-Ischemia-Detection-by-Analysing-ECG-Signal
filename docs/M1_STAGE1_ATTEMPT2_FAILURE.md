# M1 Stage-1 — Attempt 2 Failure Record

## 0. Nature of this document

A **historical execution-governance record**, not a scientific result. Attempt 2
was the one replacement invocation permitted by Authorization 2. It was consumed
without producing any M1 scientific artifact.

**This was not a resource failure.** The bounded-memory implementation merged in
PR #19 worked at full TRAIN scale. Attempt 2 stopped on a **waveform
admissibility** refusal.

## 1. Authorized state

| | |
|---|---|
| Master / scientific tree | `f8abf535cdf7a1ec0abcfac00b9b56d9279ccf72` |
| M1-v1 protocol | `08f71c5b54ebd0fcc9c1f26f05d7df2c5a1b0ca5253b8821435a65673ad65253` |
| Authorization | **2** |
| Preflight receipt | `c11769cbf00161f36da0dbe71265f0a65e364ffc67e7802b226a1192e3a933ca` |
| Attempt number | **2** |
| Replacement invocation count | **1** |

Exact command, invoked once inside tmux `cardiosentinel_m1_stage1_attempt2`
(PID 33231) under a single-shot wrapper with no retry, loop, restart-on-exit,
fallback or resume:

```
/home/AI_POC/venvs/debalekha/bin/python -m cardiosentinel m1 run-stage1 \
  --run-root cardiosentinel-runs/phase5-m1-dual-memory-v1 \
  --stream-cache-root cardiosentinel-features/m1-stream-memory-v1 \
  --p1-run-root cardiosentinel-runs/phase4-p1-physiology-v1 \
  --cache-root cardiosentinel-features/p1-b4b-embeddings-v1 \
  --source cardiosentinel-data/ltstdb/1.0.0 \
  --feature-root cardiosentinel-features/ltstdb-baseline-v1 \
  --b4b-run cardiosentinel-runs/phase3b2-architecture-v1/B4B_cnn_transformer_v1
```

Runtime: Python 3.12.6, torch 2.13.0+cpu, numpy 2.3.2, sklearn 1.9.0,
scipy 1.18.0, wfdb 4.3.1, CPU, CUDA off, AMP off, dependency digest
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

## 2. Timeline and outcome

| | |
|---|---|
| Launch (UTC) | `2026-08-11T13:32:21Z` |
| End (UTC) | `2026-08-11T15:14:09Z` |
| Wall clock | **1 h 41 m 48 s** |
| **Exit code** | **1** — an ordinary Python exception with a full traceback |

```
cardiosentinel.signal.errors.SignalValidationError:
Waveform segment has no dynamic variation in channels [0].
```

Raised through
`execute_m1_stage1 -> materialize_stream_store("validation") ->
_fill_embeddings -> flush_batch -> canonical_waveform_batches ->
B4WaveformDataset.read_waveform -> read_local_segment -> _read_segment ->
validate_waveform_segment`.

Exit 1 with a traceback is categorically different from Attempt 1's exit 137
(SIGKILL, no traceback).

## 3. What Attempt 2 completed

The bounded-memory design behaved as designed at full scale:

| | |
|---|---|
| TRAIN full-stream rows | **2,208,431** (132 streams, 60 records) |
| TRAIN extra rows newly extracted through frozen B4-B | **1,833,979** |
| TRAIN extraction waveform source reads | **1,833,979** — exact |
| Primary overlap audit rows / reads | **64 / 64** — exact |
| Total TRAIN waveform reads | **1,834,043** |
| TRAIN schema-2 store | **built and promoted** by rename |
| TRAIN manifest | `67349720ce0ecc712905177868b604e172de46e287f50c30fbf9571fa3cbe894` |
| Distance standardizer | **persisted**, `380f5f6bf83ceef230486b7485253f427714764ab355e31dbfd3e5dd5631c7f9`, 374,452 rows, train-only |
| **Peak RSS** | **≈ 3.1 GB** (observed band 1.9–3.1 GB over 1h41m) |
| MemAvailable at end | 27.2 GiB |

Attempt 1 by contrast reached 26.6 GB RSS and was killed. The corrected
waveform-read accounting from PR #19 is confirmed exact on real data.

## 4. Where it stopped

During **VALIDATION extra-row waveform read**, at extra-row ordinal **11446**
of 19,007 (`ltstdb:s20571:1:8921250:8923750`), inside extraction flush #44 at
batch size 256. **11,264** extra rows had been written before fail-fast.

## 5. Artifact state after failure — preserved untouched

| Artifact | State |
|---|---|
| TRAIN schema-2 store | **PRESENT and promoted** |
| `M1_DISTANCE_STANDARDIZER.json` | **PRESENT** |
| `.staging-validation` | **PRESENT** (13 files, no manifest, never promoted) |
| VALIDATION store | **absent** |
| M1 canonical run root | **ABSENT** |
| `M1S_short_memory_v1` / `M1L` / `M1D` | **NONE CLAIMED** |
| `M1_STAGE1_RESULTS.json` / `RUN_STATUS.json` | absent |
| Any M1 scientific metric | **none** |
| `TEST_ATTEMPT.json` / `TEST_*` | absent / none |

The staging claim records `is_a_valid_cache: false`, `resume_permitted: false`,
`automatic_repair_permitted: false`, `automatic_deletion_permitted: false`.

Nothing was deleted, renamed, promoted, resumed or repaired. **No retry was
performed.**

## 6. Test firewall

The **B4 sealed test remained unopened** throughout. No test waveform, label,
prediction or metric was accessed.

## 7. Governance consequence

- **Authorization 2 is CONSUMED.**
- Combined with Attempt 1, **two** authorized invocations have occurred and
  **zero** scientific M1 arm claims or results exist.
- All Attempt-1 and Attempt-2 artifacts remain historical M1-v1 evidence.
- Any further execution requires a **new** protocol decision, a **new**
  read-only preflight and a **new explicit human authorization**.

## 8. Engineering finding

The failure is a genuine, previously unexercised contract interaction, not a
defect in the bounded-memory work. It is characterised in
`docs/M1_ATTEMPT2_VALIDATION_ADMISSIBILITY_CENSUS.md` and resolved
prospectively by `docs/M1_PHYSICAL_OBSERVATION_DECISION_V1.md` and
`docs/M1_DUAL_MEMORY_PROTOCOL_V2.md`.
