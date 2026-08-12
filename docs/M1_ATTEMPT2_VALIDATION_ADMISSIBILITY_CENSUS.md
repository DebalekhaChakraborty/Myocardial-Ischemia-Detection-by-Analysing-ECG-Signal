# M1 Attempt-2 Validation Waveform Admissibility Census

## 0. Nature of this document

A **read-only forensic record** of the complete physical census of every
VALIDATION-extra window, performed after Attempt 2 stopped on a waveform
admissibility refusal and **before** any admissibility policy was chosen.

Every finding here rests on **observable physical waveform properties and row
identity/chronology only**. No target label, target family, challenge context,
quality annotation, event annotation or sealed-test information was used.

No M1 arm claim or scientific metric existed when this census was taken, and
none exists now.

## 1. Population, by identity only

| | |
|---|---|
| FULL VALIDATION full-stream rows | **492,904** |
| Frozen P1 primary VALIDATION cache rows | **473,897** |
| **VALIDATION-extra rows** (set difference) | **19,007** |
| P1-cache IDs outside the full stream | **0** |
| Extra IDs unique | yes |
| Extra streams / records | 30 / 13 |

## 2. Contract diagnosis

| Element | Finding |
|---|---|
| `validate_waveform_segment` | when `require_dynamic` and the segment is ≥ 1 s, raises on `np.ptp(values, axis=0) <= np.finfo(np.float64).eps` |
| `_read_segment` / `read_local_segment` | call the validator with defaults, i.e. `require_dynamic=True` |
| `B4WaveformDataset.read_waveform` | reads the exact single-channel 10-second interval and therefore inherits the hard requirement |
| Feature materialization | reads larger record-level chunks and windows them afterwards |
| `StreamingPreprocessor.process` | explicitly calls `validate_waveform_segment(waveform, require_dynamic=False)` |

**Why the mismatch was latent.** The dynamic check is evaluated per *read*, not
per *window*. Materialization reads a long chunk containing ample variation and
disables the check regardless, so a flat 10-second subwindow inside it is never
refused. B4 reads each 10-second window as its own segment with the check on. A
window can therefore be a legitimate feature-corpus row and simultaneously
inadmissible to B4.

## 3. Complete physical census — all 19,007 windows

Read through a diagnostic composition of existing repository helpers preserving
the same WFDB source, sample interval, channel, header calibration, source
units and mV conversion, with every existing check applied **except** the
dynamic-variation refusal, which was observed rather than raised.

| | |
|---|---|
| Rows censused | **19,007** |
| Source read OK | **19,007** |
| **Exact-flat, `ptp ≤ eps`** | **6** |
| Exactly constant | **6** |
| Non-finite | **0** |
| Calibration / unit / interval failures | **0** |
| Sample count == 2500 everywhere | true |

Nonzero-PTP distribution over the remaining **19,001** rows — **diagnostic
only, never a threshold-selection exercise**:

| min | p1 | p5 | p25 | median | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.030 | 0.465 | 0.535 | 0.715 | **2.265** | 2.800 | 3.425 | 4.800 | 7.910 mV |

There is no near-flat cluster; the gap between 0 and 0.030 mV is clean.

## 4. The six hard-invalid windows

All in **record `s20571`, channel 1, lead MLIII**, source unit mV, adc_gain
200.0, 2500 samples, finite, **1 unique value**, min = median = max =
**−5.120000 mV**, **ptp = 0.0**, **exactly constant after physical
calibration**.

| Ordinal | stable_id | Samples | Seconds |
|---|---|---|---|
| 11446 | `ltstdb:s20571:1:8921250:8923750` | 8921250–8923750 | 35685.0–35695.0 |
| 11447 | `ltstdb:s20571:1:8922500:8925000` | 8922500–8925000 | 35690.0–35700.0 |
| 11448 | `ltstdb:s20571:1:8923750:8926250` | 8923750–8926250 | 35695.0–35705.0 |
| 11449 | `ltstdb:s20571:1:8925000:8927500` | 8925000–8927500 | 35700.0–35710.0 |
| 11450 | `ltstdb:s20571:1:8926250:8928750` | 8926250–8928750 | 35705.0–35715.0 |
| 11451 | `ltstdb:s20571:1:8927500:8930000` | 8927500–8930000 | 35710.0–35720.0 |

**No other hard physical-failure class exists.**

## 5. Physical nature

Raw ADC is a single value, **−1024**, with baseline 0 and gain 200, giving
−5.120 mV. −1024 is **not** the 12-bit format minimum (−2048); it is a
**persistent clipping floor** for this channel. Across a 50,000-sample
neighbourhood **22.3 %** of samples sit exactly at −1024 and **none** go below
it. Neighbouring windows at −10 s and +0 s also bottom at exactly −5.120 mV
while still varying (ptp 6.65 and 5.00 mV).

This is signal clipping/saturation that becomes total for 35 s — an
instrumentation artefact, not physiology.

**Corroborative only:** all six rows carry `morphology_valid = 0` with 3 of 18
morphology descriptors finite. This is independent evidence, **not** the
availability criterion.

## 6. Physical outage grouping

Because windows are 10 s with a 5 s stride, the six overlapping failures are
**one** physical event:

| record | channel | first start | last end | windows | union |
|---|---|---|---|---|---|
| `s20571` | 1 | 8921250 (35685.0 s) | 8930000 (35720.0 s) | 6 | **35.0 s** |

affected records **1** · affected record-channel streams **1** of 30 · distinct
flat intervals **1** · failed windows **6** · total unique flat time **35.0 s**
(0.032 % of the extra population).

## 7. First canonical failure

`ltstdb:s20571:1:8921250:8923750`, ordinal **11446** of 19,007 (0-based),
extraction flush **#44** at batch 256, position 182, with **11,264** extra rows
written before fail-fast. Consistent with fail-fast at the first inadmissible
row: all six are contiguous.

## 8. Prior exposure — precise wording

These samples **were** previously read, as part of larger record-level chunks
during feature materialization, and all six windows exist as rows in the frozen
feature corpus with morphology features computed. They were **not** individually
subjected to the exact 10-second single-channel hard-dynamic check, because that
path uses `require_dynamic=False`. P1 required no B4-B embedding for these rows,
which lie outside the primary cache. M1 Attempt 2 was the first execution to
apply this specific check to them.

It is **not** claimed that these waveforms had never been read before.

## 9. TRAIN

TRAIN extraction completed all 1,833,979 extra rows under the same hard check,
so no exact-flat window exists among them. This is execution evidence, and the
M1-v2 implementation recomputes it algorithmically rather than assuming it.
