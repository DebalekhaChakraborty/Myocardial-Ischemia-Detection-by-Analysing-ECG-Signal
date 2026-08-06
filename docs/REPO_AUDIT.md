# Repository Audit

Audit date: 2026-08-06. The pre-reboot repository contained 16 tracked files,
all last updated in June 2020, with no package metadata, dependency lock, test
suite, CI workflow, or reproducible experiment command.

## Historical contents

The single script was `SOURCE CODE.py` (110 lines). It read an absent,
Windows-specific path, `C:\\Users\\MAHABHARAT\\Desktop\\ECG\\FINAL PROJECT 8
SEM\\data.csv`, rather than a repository file. It hard-coded `fs = 250`, a 0.2
second rolling window despite a 0.75 second comment, a 1.2 amplitude multiplier,
fixed ST timing, and an ischemia decision threshold of `avg_slope > 0.35`.

The script did not validate input metadata, detect failures, preserve a final
peak window, enforce a refractory period, use subject partitions, or record
provenance. It used a global average BPM to place ST points and raw sample
amplitudes without verified units or baseline reference.

`Dataset/` contained three unlabelled `hart` CSV columns: 1,580, 1,265, and 819
samples. They had no source record IDs, patient IDs, leads, units, sampling
metadata, annotations, or licensing provenance. `Results/` contained three
plots and three IDLE screenshots, including manual local paths and outcomes, but
no reproducible commands or metrics. `REPORT.pdf` is a 19-page 2020 academic
report. The root README used third-party Imgur assets and did not link a
reproducible workflow.

## Disposition

All original files are preserved under `legacy/college-v1/` for traceability.
They are excluded from the active pipeline. The historical sample data must not
be used until provenance, rights, and suitability are independently established.
Git history also contains deleted CSV and XLSX data blobs; a separate, reviewed
data-governance task must decide whether history remediation is needed.

