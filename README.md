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

No model, dataset, performance result, or clinical effectiveness claim is
provided in Phase 0.

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
`s20011`. Models, personalization, calibration, episode reasoning, and edge
benchmarking remain planned work. No performance or clinical results are
reported.

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

## License and attribution

The repository code is licensed under the MIT License. See `NOTICE.md` before
adding third-party data, annotations, models, or documentation.
