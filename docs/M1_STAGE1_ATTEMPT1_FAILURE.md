# M1 Stage-1 — Attempt 1 Failure Record

## 0. Nature of this document

This is a **historical execution-governance record**, not a scientific result.
It records that the first human-authorized M1 Stage-1 execution attempt was
consumed without producing any scientific artifact.

Nothing in this document is M1 evidence. No metric, threshold, comparison or
retention decision appears here, because none was ever produced.

## 1. Authorized state

| | |
|---|---|
| Master SHA | `229fb6ecda98adf88d947a2c7bc9de3a80028e1d` |
| M1 protocol SHA | `08f71c5b54ebd0fcc9c1f26f05d7df2c5a1b0ca5253b8821435a65673ad65253` |
| Authorization / preflight receipt | `3f00ddfbd7a74156679771c2311ad494b1bc073c1c229476bfd1df29ee4b4ad6` |
| Preflight status at authorization | `stream_cache_materialization_required` |

Exact authorized command, invoked **once**:

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

Launched in tmux session `cardiosentinel_m1_stage1` under a single-shot wrapper
with no retry, loop, restart-on-exit or fallback command.

## 2. Timeline

| | |
|---|---|
| Launch (UTC) | `2026-08-10T23:45:14Z` |
| End (UTC) | `2026-08-11T06:26:46Z` |
| Wall clock | **6 h 41 m 32 s** |
| Exit code | **137** (= 128 + 9, SIGKILL) |
| Python traceback | **none emitted** |
| stdout | 0 bytes — the canonical JSON report was never produced |

## 3. Observed resource trajectory

Read-only monitoring at 12–15 minute cadence recorded resident set size:

| Elapsed | RSS |
|---|---|
| ~2 min | ~2.95 GB |
| ~21 min | ~10.6 GB |
| ~36 min | ~16.3 GB |
| ~51 min | ~21.8 GB |
| ~68 min | ~26.6 GB |

Host: **~31 GB physical RAM**, **swap = 0** (`VmSwap: 0 kB` throughout). Last
reported available memory before the process entered uninterruptible sleep:
**~277 MB**. At that point the sampled CPU-time delta collapsed from ~1997 to 52
ticks per 20 s and the process state changed to `D`, consistent with page
reclaim pressure rather than compute.

## 4. Cause — conservative wording

The kernel `dmesg` / journal records were **not readable** from the unprivileged
account used for execution and monitoring, so the kernel's own termination
record was never directly observed.

> **Failure is strongly consistent with process termination under host memory
> exhaustion / OOM pressure.**

This wording is deliberate and permanent. It is **not** asserted that the kernel
OOM killer was confirmed, because the kernel record was not directly observed.

## 5. Exact point reached

The attempt terminated during the **extra-row waveform read + frozen B4-B
extraction** phase, which is the first and longest stage of the canonical route.
It never reached the distance standardizer, and therefore never reached stream
cache materialization, arm claiming or training.

## 6. Artifact state after failure — nothing was created

| Artifact | State |
|---|---|
| M1 run root `phase5-m1-dual-memory-v1` | **absent** |
| M1 stream-cache root `m1-stream-memory-v1` | **absent** |
| `M1_DISTANCE_STANDARDIZER.json` | **absent** |
| TRAIN stream cache | **absent** |
| VALIDATION stream cache | **absent** |
| Partial `.npz` / staging files | **absent** |
| `M1S_short_memory_v1` claim | **never created** |
| `M1L_long_memory_v1` claim | **never created** |
| `M1D_dual_memory_v1` claim | **never created** |
| M1 `RUN_STATUS.json` | **absent** |
| `M1_STAGE1_RESULTS.json` | **absent** |
| Any M1 scientific metric | **none** |
| `TEST_ATTEMPT.json` | **absent** |
| `TEST_*` artifacts | **none** |

The termination landed before the first persistence point, so **no orphaned
partial scientific artifact exists** and **no arm claim was consumed**.

Nothing was deleted, renamed, repaired, reset or overwritten in response to the
failure.

## 7. Test firewall

The **B4 sealed test remained unopened** throughout. No test waveform, label,
prediction or metric was accessed at any point. The frozen P1 Stage-1 suite
continues to record `test_accessed: false`.

## 8. Upstream integrity after failure

| Artifact | State |
|---|---|
| M1 protocol `08f71c5b…65253` | unchanged |
| P1 retention decision `7b403709…97f68` | unchanged |
| P1 Stage-1 suite `cc354ef6…be772` | unchanged, validates |
| P1-B lock `796f00e3…0676d0` | unchanged |
| B4-B checkpoint `b1301723…5591c9` | unchanged |
| Git | `master` @ `229fb6ec…`, clean |

## 9. Operational logs

Non-scientific wrapper logs, deliberately **not** versioned (the whole
`cardiosentinel-runs/` tree is Git-ignored):

Directory:
`/home/AI_POC/tactics/Myocardial-Ischemia-Detection-by-Analysing-ECG-Signal/cardiosentinel-runs/phase5-m1-dual-memory-v1-logs`

| Relative path | Bytes | SHA-256 |
|---|---|---|
| `cardiosentinel-runs/phase5-m1-dual-memory-v1-logs/M1_STAGE1_STDOUT.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `cardiosentinel-runs/phase5-m1-dual-memory-v1-logs/M1_STAGE1_STDERR.log` | 127 | `9d78e1ab8695840a8c84fa4e0282b82b124d062780af5bcde7dc83cc96479afc` |

The stderr log contains only the wrapper header and footer:
`LAUNCH_UTC`, `GIT_SHA`, `FINISH_UTC`, `EXIT_CODE=137`. There is no Python
traceback because the process received SIGKILL.

`/tmp/cardiosentinel_m1_stage1.exit` did not survive; the stderr `EXIT_CODE=137`
line is the authoritative exit record.

## 10. Governance consequence

- **The first human authorization is CONSUMED.**
- **No automatic retry is permitted**, and none was performed.
- The canonical run directory was never claimed, so no arm attempt is consumed.
- A replacement scientific execution requires **all** of the following, in
  order: an implementation correction to bound peak memory; a **new**
  real-environment read-only preflight; and a **new explicit human
  authorization**.

No replacement execution is authorized by this document.

## 11. Engineering finding

The failure was not environmental noise. The canonical implementation at
`229fb6ec…` materializes the entire full-stream representation in memory before
persisting anything: the frozen primary embedding lookup, one `ndarray` object
per newly extracted row (~1.83 M for TRAIN), the complete raw physiology
mapping, ~2.2 M `B4WindowReference` objects, and the stacked matrices are all
live simultaneously. On a 31 GB host with no swap this cannot fit, so re-running
the same code on the same host would be expected to fail the same way.

This is a real limitation of the implementation, not of the frozen scientific
protocol. The protocol does not require any particular in-memory or on-disk
representation.
