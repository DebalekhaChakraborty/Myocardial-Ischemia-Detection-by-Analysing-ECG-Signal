# Dataset Contract

CardioSentinel Phase 1 supports only the local, explicitly versioned sources
below. Raw files, derived patient data, and generated manifests remain outside
Git.

## European ST-T Database

- Canonical ID/version: `edb` / `1.0.0`.
- Source: <https://physionet.org/content/edb/1.0.0/>; official file digests are
  supplied in `SHA256SUMS.txt`.
- Expected scope: 90 two-hour, two-signal records from 79 subjects; nominally
  250 Hz. The adapter reads frequency and signal names from each WFDB header and
  treats a header mismatch as a validation failure.
- Annotations: `.atr` reference beat, rhythm, ST/T change, and quality stream.
  The primary CardioSentinel ground-truth source is the documented EDB reference
  ST-change stream, not a model-derived label.
- Role: preserve expert transient ST-change and positional axis-shift semantics
  for later evaluation. Limitations: reference waveform location is not
  available, record-level clinical metadata is sensitive, and this database is
  not a diagnosis dataset.
- Citation: Taddei et al., *ST-T change analysis in ECG ambulatory monitoring*,
  Computers in Cardiology 14:63-68 (1987), plus the current PhysioNet citation.

## Long-Term ST Database

- Canonical ID/version: `ltstdb` / `1.0.0`.
- Source: <https://physionet.org/content/ltstdb/1.0.0/>; official file digests
  are supplied in `SHA256SUMS.txt`.
- Expected scope: 86 records from 80 subjects, each about 21--24 hours with two
  or three signals and nominal 250 Hz. Header metadata remains authoritative.
- Annotations: `.atr` beat, `.16a` measurements, `.sta`, `.stb`, `.stc` episode
  definitions, `.stf` reference/deviation functions, and quality annotations.
  `.stb` (100 uV, 30 s) is pre-specified as the primary episode definition;
  `.sta` and `.stc` remain separate sensitivity definitions.
- Role: long-duration evaluation of ischemic, axis-related, rate-related,
  conduction-related, drift, noise, and mixed phenomena. Reference functions
  define expert ground truth only and must never become future-leaking features.
- Citation: Jager et al., *Long-term ST database*, Med Biol Eng Comput
  41(2):172-183 (2003), plus the current PhysioNet citation.

PhysioNet access and attribution terms apply. Before using either resource,
review the source page, any applicable credentialing or data-use conditions, and
the repository `NOTICE.md`; the MIT license does not grant dataset rights.

## Acquisition Boundary

`cardiosentinel data download-metadata` retrieves only `RECORDS`, WFDB headers,
one requested annotation stream, and the official checksum manifest. It verifies
every downloaded file against its corresponding official SHA-256 entry and never
downloads `.dat` waveform files. Its URLs are explicitly pinned to the declared
dataset version rather than a library's current-release lookup. Remote probe and
validation commands use WFDB `pn_dir` access for headers and annotations only.

Phase 2 waveform probes are a separate, explicitly bounded operation. They use
WFDB physical calibration and request only the declared half-open sample
interval and selected channels. Probe output contains metadata and aggregate
amplitude summaries, not raw arrays. Waveform segments and derived outputs must
remain outside Git.
