# M2-v1 stress-interval eligibility — frozen prospective human decision

> **STATUS: FROZEN PROSPECTIVE HUMAN DECISION — MADE BEFORE ANY M2 DEVELOPMENT
> ACCESS.**
> Decided 2026-08-13, against parent master
> `e246c84ac7bc3f922cfa2615aba96249356f5957`.
> **No DEVELOPMENT or VALIDATION data was accessed, enumerated or measured in
> reaching this decision, and no TEST data was accessed.** The decision was
> reached from the frozen LTSTDB annotation semantics and the parsed source
> representations alone.

## 0. What this decides, and what it does not

`docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md` §7.1 names five longitudinal
stress families for prototype-contamination evaluation. Implementation review
established that the frozen LTSTDB V1 corpus supplies a **source-defined
interval** for only three of them: the remaining two exist in the source as
*instantaneous markers with no paired end annotation*.

This document freezes which families are eligible for the §7.2 prototype-drift
metric. It **changes no M2 gate condition, no threshold, no memory policy and
no challenge definition**, and it does not modify the frozen M2 protocol, which
remains immutable historical science.

## 1. The eligibility policy

CardioSentinel uses only stress intervals whose **start and end are both
explicitly source-defined** by the frozen LTSTDB annotation semantics. Duration
is never invented for an instantaneous source marker.

| # | Family | Source form | Parsed representation | Interval | Eligible |
|---|---|---|---|---|---|
| 1 | Ischemic ST event | `.stb` `(stX…` / `astX…` / `stX…)` | `STEvent(event_subtype="ischemic")` | `[onset_sample, end_sample]` | **YES** |
| 2 | Heart-rate-related ST event | `.stb` rate-marked episode forms | `STEvent(event_subtype="heart_rate_related")` | `[onset_sample, end_sample]` | **YES** |
| 3 | Unreadable quality | `.stb` paired `(urdX` … `urdX)` | `SignalQualityInterval(state="unreadable")` | `[start_sample, end_sample]` | **YES** |
| 4 | Point noise | `.stb` `noiX-…` | `AnnotationMarker(subtype="point_noise")` | none | **NO** |
| 5 | Axis shift | `.stb` `sstX` | `AnnotationMarker(category="st_shift", subtype="axis_related")` | none | **NO** |
| 6 | Conduction change | `.stb` `sccstX` | `AnnotationMarker(category="st_shift", subtype="conduction_related")` | none | **NO** |

Only **complete canonical source episodes** are eligible for families 1 and 2.
A source-censored episode that the ingestion layer already excludes from the
canonical event set does **not** gain a fabricated onset or endpoint for M2.

### Canonical exclusion reasons

```
point_noise_marker_has_no_source_defined_interval
axis_shift_marker_has_no_source_defined_interval
conduction_change_marker_has_no_source_defined_interval
```

## 2. Why ±30 seconds is NOT reused as a stress duration

`MARKER_VICINITY_SECONDS = 30.0` is frozen in
`src/cardiosentinel/evaluation/protocol.py` and documented in
`docs/METRICS_PROTOCOL.md` for exactly one purpose: **axis-shift window-level
FPR challenge membership** — "axis-shift FPR in the plus/minus 30-second marker
vicinity".

It is a *window-selection radius*, not a claim about how long an axis shift
stresses a patient prototype. Repurposing it for §7.2 would silently redefine
"drift at stress end" as 30 s after the marker, and the "≥5 minute" and
"≥30 minute" residual origins as 5.5 and 30.5 minutes after the marker. That is
a new scientific assertion the frozen protocol never made.

The ±30-second vicinity therefore **must not** be used as a stress duration, a
persistence duration, a prototype-drift interval, a recovery origin, or a
drift-at-stress-end definition. Axis-shift FPR remains `quantitative_secondary`
exactly as frozen.

## 3. Why marker-to-next-marker / stream-end persistence is NOT used

An axis or conduction shift is a step change with no annotated end. Defining
its interval as "marker until the next `st_shift` marker" or "marker until the
end of the stream" would invent a persistence duration from annotation spacing
— an artifact of how densely an expert happened to mark a record, not a
source-traceable physiological duration. No such rule is defined, implemented
or permitted.

## 4. Scientific interpretation

This is **not** a claim that axis or conduction changes have zero duration, and
these families have **not** failed.

The correct statement is:

> The LTSTDB V1 source does not provide a source-traceable duration suitable
> for the prespecified prototype-drift metric.

Those interval-based statistics are therefore reported as:

```
not_estimable_from_source_defined_LTSTDB_intervals
```

rather than as zero, as a silent omission, or as an estimate from an invented
duration. No stress end, drift-at-stress-end, +5 minute origin or +30 minute
origin is fabricated for them.

## 5. Evidence consequences

Prototype-contamination/drift evidence is canonical for ischemic ST events,
heart-rate-related ST events, and source-defined unreadable-quality intervals.

Axis shift and conduction change remain evaluated through their already-frozen
**challenge false-positive evidence**, unchanged: axis `quantitative_secondary`,
conduction `exploratory_descriptive` with its frozen one-subject limitation. No
inferential or drift-duration claim is created for either.

The M2 exit decision may therefore weigh: source-defined prototype
contamination/drift; background false-alarm evidence; rate challenge FPR; axis
challenge FPR; conduction descriptive FP/N; primary detectability; cold-start
evidence; and update-admission/freeze coverage. An axis or conduction
prototype-drift number is **not** created merely to complete a table.

## 6. Naming

The source-defined quality family is called **LTSTDB unreadable-quality
intervals**. It must not be described in a way implying that every artifact or
noise class has a source-defined duration: point `noi` markers are not
longitudinal quality intervals.

## 7. Separation from follow-up exclusion

Source-interval **eligibility** (this document) and trajectory-support
**eligibility** (frozen protocol §7.2) are independent. A family excluded here
never enters drift evaluation at all. An *eligible* interval may still be
excluded from a specific statistic afterwards for lack of a valid pre-stress
prototype or lack of eligible ≥5 / ≥30 minute causal follow-up, and those
reasons are recorded separately. Follow-up is never fabricated.

## 8. Scope

This decision changes no M2 gate condition (G1–G6), no G3 bound, no
`NORMAL_EVIDENCE_THRESHOLD`, no `M1L_CLASSIFICATION_THRESHOLD`, no refractory
semantics, no memory policy, no challenge definition, no cold-start bin and no
prototype-drift formula. No frozen scientific document is rewritten.
