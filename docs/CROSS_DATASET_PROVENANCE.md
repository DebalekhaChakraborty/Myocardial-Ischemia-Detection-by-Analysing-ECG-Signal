# Cross-Dataset Provenance

This audit governs any comparison between Long-Term ST Database (LTSTDB) v1.0.0
and European ST-T Database (EDB) v1.0.0. It uses only authoritative PhysioNet
release documentation and record headers. Age, sex, diagnosis, medication, lead
configuration, morphology, and approximate timing are never identity evidence.

## Release-level evidence

The official [LTSTDB release page](https://physionet.org/content/ltstdb/1.0.0/)
states that ten recordings came from the Pisa collection originally gathered
for EDB, that EDB contains two-hour excerpts of some of the same recordings,
and that the analog recordings were redigitized and rescaled for LTSTDB. Direct
annotation comparison is therefore not valid merely by aligning amplitudes or
times across releases.

## Verified correspondences

Each mapping below is stated directly in the corresponding official LTSTDB
header. These are source-recording correspondences, not waveform-file identity.

| LTSTDB | EDB | Confidence | Official evidence |
|---|---|---|---|
| `s20021` | `e0113` | verified | [`s20021.hea`](https://physionet.org/files/ltstdb/1.0.0/s20021.hea) |
| `s20151` | `e0103` | verified | [`s20151.hea`](https://physionet.org/files/ltstdb/1.0.0/s20151.hea) |
| `s20161` | `e0105` | verified | [`s20161.hea`](https://physionet.org/files/ltstdb/1.0.0/s20161.hea) |
| `s20171` | `e0127` | verified | [`s20171.hea`](https://physionet.org/files/ltstdb/1.0.0/s20171.hea) |
| `s20181` | `e0162` | verified | [`s20181.hea`](https://physionet.org/files/ltstdb/1.0.0/s20181.hea) |
| `s20291` | `e0104` | verified | [`s20291.hea`](https://physionet.org/files/ltstdb/1.0.0/s20291.hea) |
| `s20301` | `e0125` | verified | [`s20301.hea`](https://physionet.org/files/ltstdb/1.0.0/s20301.hea) |
| `s20311` | `e0129` | verified | [`s20311.hea`](https://physionet.org/files/ltstdb/1.0.0/s20311.hea) |
| `s20581` | `e0603` | verified | [`s20581.hea`](https://physionet.org/files/ltstdb/1.0.0/s20581.hea) |
| `s20591` | `e0604` | verified | [`s20591.hea`](https://physionet.org/files/ltstdb/1.0.0/s20591.hea) |

The release-level statement identifies ten Pisa-origin LTSTDB recordings, and
the ten direct header mappings account for all ten. No additional pair is
inferred.

## Conservative evaluation policy

EDB is a Phase-3 secondary benchmark, not a fully independent external-
validation cohort. Any later evaluation claiming independence from LTSTDB must
exclude the ten verified EDB excerpts and every additional EDB record grouped
with one of those records under the documented EDB subject mapping. The
conservative affected EDB record set is:

```text
e0103 e0104 e0105 e0113 e0123 e0124 e0125 e0126 e0127
e0129 e0133 e0162 e0163 e0603 e0604
```

This exclusion addresses known record and same-EDB-subject overlap. It does not
prove that every remaining EDB subject is independent of every LTSTDB subject.
Any new authoritative correspondence requires a reviewed registry update and a
new benchmark protocol version if it changes a frozen evaluation cohort.

The benchmark exposes two explicit secondary cohorts:

- `full`: all 90 EDB records, for dataset description, with
  `contains_known_source_overlap = true`;
- `overlap_clean`: 75 EDB records after the conservative 15-record exclusion,
  with `contains_known_source_overlap = false` for the known registry only.

When model training includes LTSTDB, the recommended EDB secondary evaluation
cohort is `overlap_clean`; policy validation rejects `full` for that use. The
full cohort may still be enumerated descriptively. Neither cohort may be called
fully independent external validation because independence of every remaining
subject has not been proven.

The typed registry is implemented in
`cardiosentinel.evaluation.provenance`. Its confidence vocabulary is
`verified`, `collection-level-risk`, or `unknown`; demographic similarity can
never promote a record to `verified`.
