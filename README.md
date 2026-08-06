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
header and annotation metadata from EDB and LTSTDB. CardioSentinel preserves
subject identity, episode semantics, unknown-form accounting, and provenance;
it does not download ECG waveforms during remote probes or validation. Signal
processing, models, personalization, calibration, episode reasoning, and edge
benchmarking remain planned work. No results are reported.

Data commands require the optional `data` dependency group and never download
data during import or tests:

```bash
python -m pip install -e ".[dev,data]"
python -m cardiosentinel data --help
```

## License and attribution

The repository code is licensed under the MIT License. See `NOTICE.md` before
adding third-party data, annotations, models, or documentation.
