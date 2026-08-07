# Signal-Processing Contract

CardioSentinel Phase 2 defines a causal physical-signal path for engineering
research. It is not a clinical ECG standard, diagnostic pipeline, or medical
device claim.

## Physical path and boundaries

The implemented path is:

```text
bounded WFDB waveform
-> WFDB physical calibration validation
-> explicit source-unit conversion to mV
-> optional stateful causal filtering
-> completed causal windows
-> descriptive waveform-only quality metrics
```

These stages have separate responsibilities:

1. **Physical signal acquisition** reads only a requested sample interval and
   preserves source indices, channel order, source units, and calibration
   metadata.
2. **Causal preprocessing** may transform a sample using that sample and prior
   state only. Raw physical ECG remains a first-class identity profile.
3. **Quality measurement** describes finite values, variation, derivatives, and
   spectral power using waveform samples only.
4. **Future morphology extraction** may eventually locate QRS, J points, ST
   levels, and T-wave morphology, but none of this is implemented in Phase 2.
5. **Ground-truth annotations** may later define targets and evaluation
   intervals. They are never preprocessing, window, or quality-feature inputs.

Expert episode annotations, expert-corrected ST measurements, `.stf`, GRST/LRST,
future event endpoints, and future samples are outside the signal package. No
window is centered, shifted, or selected using an annotation.

## Physical representation

`WaveformSegment.values` has shape `[samples, channels]`, uses floating-point
physical amplitude in canonical `mV`, and retains absolute half-open source
indices `[start_sample, end_sample)`. Signal names, lead names, and channel order
match WFDB metadata exactly. Source units remain in `source_physical_units`.

Supported explicit conversions are `V -> 1000 mV`, `mV -> 1 mV`, and
`uV`/`µV`/`μV -> 0.001 mV`. Unknown units, invalid gain metadata, missing units,
non-finite physical reads, invalid sampling rates, and sufficiently long
constant segments fail hard validation. CardioSentinel does not reapply EDB's
historical ADC scaling after WFDB has produced calibrated physical samples. It
does not normalize, center, or z-score a patient or record.

Verified WFDB sources may use explicitly supported legacy lexical aliases. The
LTSTDB v1.0.0 spelling `mv` is mapped to dimensional `mV` with factor 1.0 at the
reader boundary; the exact source spelling remains in `source_physical_units`
and source files are never rewritten. Unit matching is an explicit mapping, not
case folding, so unknown or dimensionally ambiguous spellings still fail.

## Causal filter profiles

The base configuration uses the explicit `raw` profile. High-pass, low-pass,
and notch filters are all disabled. There is no assumed 50 or 60 Hz mains
frequency. A configured chain uses SciPy's causal Butterworth SOS sections and
`iirnotch`; it never uses `filtfilt`, forward-backward filtering, centered
smoothing, or future samples.

High-pass filtering can alter low-frequency ECG morphology through attenuation
and phase shift. Published studies show that real-time 0.5 Hz high-pass filters
can distort the ST segment and discuss 0.05 Hz as the conservative cutoff for
causal/real-time filtering. Accordingly, a profile named `st_preserving`
rejects a high-pass cutoff above 0.05 Hz. This is a research guardrail, not a
claim that a configured filter is clinically compliant or morphology-neutral.

The streaming initial state is all zeros: no preceding waveform history is
invented. Filter warm-up is estimated from the slowest SOS pole as the samples
required for its residual to fall below `1e-3`. Every processed chunk exposes
the initial-state strategy, current state, processed count, warm-up samples in
the chunk, remaining warm-up, and whether the chain is warm. Reset explicitly
returns to zero history.

Continuous one-shot processing and sequential chunk processing with retained
state must agree within the numerical tolerance tested by the repository.
Already emitted output must be independent of all future samples.

## Causal windows

Window length and stride have no Phase 2 defaults. They must map exactly to
source samples at the segment's validated sampling frequency. A window is
available only at its exclusive `end_sample`, after all required samples have
arrived. Each single-channel window retains record, subject, channel, lead,
absolute indices, times, unit, and whether it contains filter warm-up. This
module does not assign labels.

## Descriptive quality metrics

For each completed window, CardioSentinel reports:

- `finite_sample_fraction`
- `flatline_fraction`
- `repeated_value_fraction`
- `robust_amplitude_range_mv` (95th minus 5th percentile)
- `robust_derivative_scale_mv_per_s` (scaled first-derivative MAD in mV/s)
- `derivative_outlier_fraction`
- `low_frequency_power_ratio` for 0.01--0.5 Hz
- `high_frequency_power_ratio` for 40--100 Hz when below Nyquist
- `powerline_ratio_50hz` for 49--51 Hz
- `powerline_ratio_60hz` for 59--61 Hz

Welch spectral ratios are relative to finite non-DC power. Each requested band
requires at least two usable frequency bins; unsupported bands, insufficient
spectral resolution, non-finite windows, and near-zero total power return
`null`.
There is no composite or purported universal signal-quality score. Expert EDB
and LTSTDB quality labels are not imported or consumed.

## Filter audit and provenance

The filter audit emits JSON containing the profile, component type/order/
frequencies, sampling frequency, SOS coefficients and SHA-256 digest, warm-up,
initialization strategy, and magnitude/phase/gain at relevant frequencies below
Nyquist. Runtime provenance includes Git SHA and dirty state, CardioSentinel,
Python, NumPy, SciPy, and WFDB versions. Waveform processing provenance also
records dataset/version, record/subject, sample interval, sampling frequency,
source/canonical units, schema version, and filter configuration.

Generated waveform-derived artefacts remain outside Git.

## Phase 2 integration validation

Bounded remote validation used the mechanically selected interval `[0, 15000)`
(the first 60 seconds) from EDB `e0113`, EDB `e0161`, and LTSTDB `s20011`.
WFDB reported 250 Hz, two channels, and `mV` for every selected source channel.
Each interval was finite with shape `[15000, 2]`. Raw processing was bit-exact;
the tested causal high-pass/low-pass/50 Hz notch chain remained finite and gave
zero maximum absolute difference between one-shot and one-second chunks. These
checks establish engineering behavior only, not morphology or clinical validity.

## Scientific rationale

- García-Niebla et al., *High-Bandpass Filters in Electrocardiography: Source of
  Error in the Interpretation of the ST Segment*:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3388307/>
- Abächerli et al., *Meet the challenge of high-pass filter and ST-segment
  requirements with a DC-coupled digital electrocardiogram amplifier*:
  <https://pubmed.ncbi.nlm.nih.gov/19700169/>
- Isaksen et al., *Quantification of the first-order high-pass filter's
  influence on the automatic measurements of the electrocardiogram*:
  <https://pubmed.ncbi.nlm.nih.gov/28187886/>

These sources motivate conservative safeguards; they do not validate
CardioSentinel's implementation clinically.

## Not implemented

Phase 2 does not implement QRS detection, ECG delineation, J-point detection,
ST-level estimation, baseline-wander correction as ground truth, T-wave
morphology, predictive feature engineering, machine learning, AI inference,
episode classification, or clinical recommendations.
