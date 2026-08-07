# CardioSentinel

CardioSentinel is a research-software foundation for patient-adaptive,
uncertainty-aware, edge-oriented investigation of transient ischemic ST episodes
in ambulatory ECG. It is not a medical device and does not provide diagnosis,
treatment, or medical recommendations.

## Project evolution

This repository previously hosted a 2020 B.Tech prototype based on fixed ECG
thresholds. That work is retained unchanged in
[`legacy/college-v1/`](legacy/college-v1/README.md) for historical traceability.
It is not part of the CardioSentinel pipeline and its outputs are not clinical
evidence.

The current research objective is a reproducible system that can eventually
support the physical-system loop:

`ECG acquisition or replay -> edge processing -> patient-adaptive inference -> uncertainty -> local decision or escalation -> temporal episode reasoning -> evidence-grounded alert`.

No clinical effectiveness claim is made. Physiological data and experiment
outputs remain external to this repository.

## Repository structure

- `src/cardiosentinel/`: package and future research domains.
- `configs/`: versioned, validated configuration profiles.
- `docs/`: scope, integrity contract, audit, and implementation roadmap.
- `tests/`: offline unit, contract, and integration tests.
- `data/` and `artifacts/`: documented local locations; their contents are not
  committed.
- `legacy/college-v1/`: preserved academic prototype.

## Setup

Python 3.11 is the initial supported version.

```bash
python -m pip install -e ".[dev]"
python -m cardiosentinel --help
python -m cardiosentinel info
```

Raw or processed physiological data is not included. Read
[`docs/RESEARCH_SCOPE.md`](docs/RESEARCH_SCOPE.md) and
[`docs/EXPERIMENT_CONTRACT.md`](docs/EXPERIMENT_CONTRACT.md) before conducting
research changes.

## Development status

Phase 1 implementation and annotation-semantic validation are complete for
header and annotation metadata from EDB and LTSTDB. Phase 2 implements a bounded
physical-waveform reader, canonical mV representation, raw identity profile,
optional stateful causal filters, causal windows, descriptive signal-quality
metrics, and filter audits. Bounded physical-waveform integration validation is
complete for the first 60 seconds of EDB `e0113`, EDB `e0161`, and LTSTDB
`s20011`. Phase 3A freezes the LTSTDB `.stb` benchmark protocol, 56/12/12
subject split, causal 10-second/5-second window targets, leakage controls,
training-sampling policy, and metrics protocol. Phase 3B-1 is complete: each
frozen B0--B3 global classical baseline received one sealed-test evaluation,
with no test-guided tuning. The compact evidence and limitations are recorded in
[`docs/PHASE3B1_CLASSICAL_BASELINE_RESULTS.md`](docs/PHASE3B1_CLASSICAL_BASELINE_RESULTS.md).
The next planned baseline is B4, a prospective compact raw-waveform neural
baseline whose contract must be frozen before training or testing.

Data commands require the optional `data` dependency group and never download
data during import or tests:

```bash
python -m pip install -e ".[dev,data]"
python -m cardiosentinel data --help
```

Signal commands require the optional `signal` dependency group. High-pass,
low-pass, and notch filtering are disabled by default:

```bash
python -m pip install -e ".[dev,data,signal]"
python -m cardiosentinel signal --help
```

See [`docs/SIGNAL_PROCESSING_CONTRACT.md`](docs/SIGNAL_PROCESSING_CONTRACT.md)
for the causality, physical-unit, filtering, quality, and ground-truth boundary.

Benchmark commands inspect metadata and annotations without training or full
waveform downloads:

```bash
python -m cardiosentinel benchmark --help
python -m cardiosentinel benchmark split-info \
  --split protocols/splits/ltstdb_v1.json
```

The frozen rules are in [`docs/BENCHMARK_PROTOCOL_V1.md`](docs/BENCHMARK_PROTOCOL_V1.md),
with metrics in [`docs/METRICS_PROTOCOL.md`](docs/METRICS_PROTOCOL.md) and known
EDB/LTSTDB overlap in
[`docs/CROSS_DATASET_PROVENANCE.md`](docs/CROSS_DATASET_PROVENANCE.md).

Classical baseline commands require the `ml` extras. Raw and derived
physiological data and experiment outputs are never committed to Git. They may
use explicit roots outside the repository filesystem or the approved Git-ignored
local roots `cardiosentinel-data/`, `cardiosentinel-features/`, and
`cardiosentinel-runs/`:

```bash
python -m pip install -e ".[dev,data,signal,ml]"
python -m cardiosentinel baseline --help
python -m cardiosentinel baseline acquire \
  --destination /external/data/ltstdb/1.0.0
```

The acquisition command is plan-only unless `--execute` is supplied. The frozen
feature, model, preprocessing, sampling, test-access, and artifact rules are in
[`docs/BASELINE_PROTOCOL_V1.md`](docs/BASELINE_PROTOCOL_V1.md).

### Monitoring Phase 3B materialization

The read-only Phase 3B monitor reports progress from an existing external
feature root. It does not access waveform source data, run models, or change
feature caches or manifests:

```bash
python scripts/monitor_phase3b.py
watch -n 30 python scripts/monitor_phase3b.py
```

To inspect a different external feature root:

```bash
python scripts/monitor_phase3b.py --feature-root /path/to/features
```

### Auditing a completed Phase 3B corpus

The read-only corpus audit validates every cache and the persisted corpus hash
before reporting frozen primary counts, descriptive challenge/exclusion families,
and algorithmic morphology-feature validity:

```bash
python scripts/audit_phase3b_corpus.py \
  --feature-root /path/to/ltstdb-baseline-v1
```

## License and attribution

The repository code is licensed under the MIT License. See `NOTICE.md` before
adding third-party data, annotations, models, or documentation.
