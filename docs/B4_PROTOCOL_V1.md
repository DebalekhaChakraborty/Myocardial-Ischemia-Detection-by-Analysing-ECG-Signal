# B4 Protocol V1

## Scope

This document prospectively freezes the CardioSentinel Phase 3B-2 B4 compact
raw-waveform neural baseline. B4 is a global, single-channel comparator to the
frozen B0-B3 classical baselines. It is not the CardioSentinel contribution and
contains no personalization, temporal episode reasoning, foundation-model
knowledge, or cloud inference.

B4 investigates transient ischemic ST episode detection for research and
monitoring decision support using a public dataset. It is not a diagnostic
system, medical device, or clinical-effectiveness claim.

## Research question

Does a compact end-to-end neural representation learned directly from the same
causal single-channel ECG windows improve subject-disjoint discrimination
relative to the frozen global classical baselines, without personalization,
temporal episode reasoning, foundation-model knowledge, or cloud inference?

## Frozen benchmark identity

- Dataset: Long-Term ST Database (`ltstdb`) v1.0.0.
- Primary annotation definition: `ltstdb.stb`, unchanged from Benchmark V1.
- Split: frozen subject-disjoint 56/12/12 train/validation/test assignment.
- Split SHA-256:
  `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`.
- Window geometry: completed causal 10-second windows every 5 seconds.
- Target precedence and challenge definitions: unchanged Benchmark V1 rules.
- Existing Phase 3B-1 feature corpus SHA-256:
  `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5`.
- Feature materialization commit:
  `4b20a284aac91155bbeefc134fd5fb448028fa8a`.
- B0-B3 experiment implementation commit:
  `4f57ba38d4df593abd9fdd77d5544931b8255534`.

The Phase 3B-1 feature corpus and B0-B3 results remain frozen historical
evidence. B4 neither changes nor rematerializes that corpus and does not permit
any B0-B3 rerun or retrospective change.

## Unit of analysis

Repository inspection confirms that `CausalWindowGenerator` emits one
`CausalWindow` for each record, channel, and completed start/end interval. Each
window stores one one-dimensional physical-mV array. Benchmark targets and
feature rows use the same channel-specific identity, with stable ID:

```text
dataset:record_id:channel_index:window_start_sample:window_end_sample
```

B4 V1 preserves this experimental unit. It does not fuse, concatenate, or pool
the two or three LTSTDB channels. Subject ID, record ID, channel index, channel
count, signal name, and lead name remain provenance/grouping metadata only and
are never predictive inputs.

## Raw waveform input contract

Each model input is one channel-specific completed 10-second waveform:

```text
sampling frequency: 250 Hz
samples:            2500
physical unit:      mV
batch tensor:       [B, 1, 2500]
training dtype:     IEEE 754 float32
processing profile: raw identity
```

The implementation must verify the authoritative WFDB header frequency is
exactly 250 Hz and the half-open window interval contains exactly 2,500 samples.
It must fail rather than resample a mismatch. `read_local_segment` must obtain
source-calibrated values from the verified pinned LTSTDB source and convert
supported physical units to canonical mV using the existing validation path.
Values must be finite. The canonical float64 reader values are cast once to
float32 for the FP32 model; no other numeric transformation is permitted.
Malformed, non-finite, miscalibrated, or shape-invalid values cause failure and
are never silently imputed.

The B4 input contains no high-pass or low-pass filtering, notch filtering,
`filtfilt`, future samples, patient/record normalization, per-window z-scoring,
amplitude normalization, handcrafted `signal_v1` or `morphology_v1` values,
R peaks, J points, ST measurements, expert beats, annotations, labels, or
context flags. Labels and context may be joined by stable ID only after waveform
geometry exists and only for loss construction or evaluation.

The existing feature corpus may supply provenance and row metadata: stable ID,
subject, record, channel, window bounds, partition, target family, and the
frozen training selection. Its numeric feature matrix is prohibited as a B4
model input.

## Eligible train and validation populations

The existing metadata-only sampler was rechecked without opening any test
cache. B4 uses the exact B1-B3 selection policy with seed `2026`: every eligible
training ischemic-positive row and the deterministic subject-aware selection of
at most three background negatives per positive.

| Population | Ischemic positive | Background negative | Total |
| --- | ---: | ---: | ---: |
| Unsampled train metadata | 93,613 | 2,049,986 | 2,143,599 |
| Selected B4 train | 93,613 | 280,839 | 374,452 |
| Full primary validation | 21,628 | 452,269 | 473,897 |

The selected training set contains 56 subjects. Rate-related, axis-shift,
conduction-change, boundary-ambiguous, quality-excluded, and
source-censored-or-unknown rows do not train B4. Validation checkpoint selection
and threshold selection use every primary validation row without sampling.
Validation challenge rows may be reported under the frozen evidence policy but
may not select or change the architecture, checkpoint, or threshold.

No B4 test-row inspection is permitted during protocol design, implementation,
training, checkpoint selection, threshold selection, or experiment locking.

## Architecture

All convolutions are one-dimensional. Every GroupNorm uses 8 groups,
`eps=1e-5`, and trainable affine scale and bias. Every convolution has
`bias=False`; both Linear layers have bias. Padding is symmetric and equals
`dilation * (kernel_size - 1) // 2` where specified.

```text
Input: [B, 1, 2500]

Stem:
  Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
  GroupNorm(8, 32)
  SiLU

Downsampling block 1: 32 -> 48, kernel=9, stride=2, dilation=1
Downsampling block 2: 48 -> 64, kernel=7, stride=2, dilation=1
Downsampling block 3: 64 -> 96, kernel=5, stride=2, dilation=1
Downsampling block 4: 96 -> 128, kernel=5, stride=2, dilation=1

Each downsampling block:
  depthwise Conv1d(Cin, Cin, kernel, stride, padding, dilation,
                   groups=Cin, bias=False)
  GroupNorm(8, Cin)
  SiLU
  pointwise Conv1d(Cin, Cout, kernel_size=1, bias=False)
  GroupNorm(8, Cout)
  SiLU

Context block 1: 128 -> 128, kernel=5, stride=1, dilation=2
Context block 2: 128 -> 128, kernel=5, stride=1, dilation=4
Context block 3: 128 -> 128, kernel=5, stride=1, dilation=8

Each context block computes x + F(x), where F is:
  depthwise Conv1d(128, 128, kernel=5, stride=1, padding=2*dilation,
                   dilation=dilation, groups=128, bias=False)
  GroupNorm(8, 128)
  SiLU
  pointwise Conv1d(128, 128, kernel_size=1, bias=False)
  GroupNorm(8, 128)
  SiLU

There is no post-addition activation or dropout in a context block.

Head:
  AdaptiveAvgPool1d(1)
  flatten
  Dropout(p=0.10)
  Linear(128, 64)
  SiLU
  Dropout(p=0.10)
  Linear(64, 1)

Output: one raw binary logit; no sigmoid is part of the model.
```

No architecture search, attention, transformer, alternate width/depth, or
multi-lead variant belongs to B4 V1.

## Parameter, shape, and receptive-field calculation

For a bias-free grouped convolution, parameters are
`Cout * (Cin / groups) * kernel`. GroupNorm contributes `2 * channels` affine
parameters, and a Linear layer contributes `out * in + out`.

| Component | Trainable parameters |
| --- | ---: |
| Stem convolution + GroupNorm | 544 |
| Downsampling block 1 | 1,984 |
| Downsampling block 2 | 3,632 |
| Downsampling block 3 | 6,784 |
| Downsampling block 4 | 13,216 |
| Context block, dilation 2 | 17,536 |
| Context block, dilation 4 | 17,536 |
| Context block, dilation 8 | 17,536 |
| Head | 8,321 |
| **Total** | **87,089** |

The total is below the hard ceiling of 1,000,000 trainable parameters. The raw
FP32 parameter payload is `87,089 * 4 = 348,356` bytes, approximately
0.348 MB or 0.332 MiB. A serialized state dictionary will be slightly larger
because of tensor and container metadata; the future implementation must record
its actual file size and SHA-256.

Starting with receptive field `r=1` and sample jump `j=1`, each convolution
updates `r <- r + (kernel - 1) * dilation * j` and
`j <- j * stride`.

| Stage | Temporal length | Sample jump | Local receptive field |
| --- | ---: | ---: | ---: |
| Input | 2500 | 1 | 1 |
| Stem | 1250 | 2 | 15 |
| Downsampling block 1 | 625 | 4 | 31 |
| Downsampling block 2 | 313 | 8 | 55 |
| Downsampling block 3 | 157 | 16 | 87 |
| Downsampling block 4 | 79 | 32 | 151 |
| Context dilation 2 | 79 | 32 | 407 |
| Context dilation 4 | 79 | 32 | 919 |
| Context dilation 8 | 79 | 32 | 1943 |

The final local feature receptive field is 1,943 samples, or 7.772 seconds at
250 Hz. Adaptive average pooling aggregates all 79 final positions, so the
logit can aggregate evidence spanning the complete 10-second input. Symmetric
zero padding supplies no observed or future ECG samples.

## Training configuration

- Framework: PyTorch; exact resolved version must be recorded.
- Seed: `2026` for Python, NumPy, PyTorch, and CUDA when applicable.
- Numerical mode: FP32 only; no mixed precision.
- Loss: `BCEWithLogitsLoss(reduction="mean")` with no class or sample weights.
- Optimizer: one AdamW parameter group containing every trainable parameter,
  with learning rate `1e-3`, weight decay `1e-4`, `betas=(0.9, 0.999)`,
  `eps=1e-8`, `amsgrad=False`, `foreach=False`, and `fused=False`.
- Batch size: 256; retain the final incomplete batch (`drop_last=False`).
- Maximum epochs: 15 completed training epochs.
- Training order: deterministic epoch-wise shuffle controlled by the frozen
  seed; validation order does not alter results.
- Learning-rate scheduler: none.
- Early stopping: stop after four consecutive completed epochs without a
  validation AUPRC increase strictly greater than `1e-6` over the
  early-stopping reference.
- No hyperparameter search, oversampling beyond the frozen selection, data
  augmentation, random noise, amplitude scaling, or lead dropout.

PyTorch module initialization defaults from the recorded resolved environment
are used once after all seeds and deterministic settings are established. No
test-informed initialization or restart selection is permitted.

## Checkpoint selection

After each completed epoch, score every primary validation row using
`sigmoid(raw_logit)` in evaluation mode and compute pooled-window AUPRC. Store
the checkpoint with the numerically maximum validation AUPRC; an exact tie keeps
the earliest epoch. Any strict increase updates the maximum checkpoint, while
only an increase greater than `1e-6` resets early-stopping patience. Training
ends after the fourth non-improving completed epoch or after epoch 15, whichever
occurs first.

Restore the selected checkpoint before threshold selection. Do not average
checkpoints, refit on validation, or choose a restart using validation results.
The sigmoid score is an uncalibrated model score, not calibrated probability or
confidence.

## Threshold selection

Using only full primary validation labels and scores from the selected
checkpoint, choose the exact observed score threshold that maximizes validation
F1. A row is positive when `score >= threshold`; an F1 tie selects the highest
threshold. This is the unchanged B1-B3 validation rule. No approximate grid,
test value, challenge result, or subject-specific threshold is permitted.

## Determinism and reproducibility

The future implementation must establish all seeds before model or data-loader
construction, enable deterministic PyTorch algorithms where supported, disable
cuDNN benchmarking, and enable deterministic cuDNN behavior when applicable.
CUDA determinism requirements, including required workspace configuration, must
be applied before execution. Worker initialization and data-order generation
must derive deterministically from seed `2026`.

A deterministic-algorithm failure must stop execution and must not be silently
ignored or downgraded. A platform/library exception requires review before any
training proceeds.

Record at minimum the exact Python, PyTorch, NumPy, CUDA/cuDNN, and dependency
environment; CPU/GPU device; deterministic settings; worker count; Git SHA and
dirty state; commands; epoch history; selected epoch; and training duration.
Worker count and I/O strategy must not change selected IDs, input values, batch
order semantics, or outputs.

## Raw-waveform data-access feasibility

The current repository already provides the required scientific identity path:

1. Validate the pinned local LTSTDB v1.0.0 source against the official PhysioNet
   checksum manifest.
2. Validate the frozen feature-corpus and split identities, then read only its
   metadata arrays to recover stable IDs, partition, target family, record,
   channel, and exact sample bounds. Never load the handcrafted numeric matrix
   for model input.
3. Reproduce and digest the exact frozen training selection; expose all primary
   validation IDs without sampling. Development code must not enumerate or open
   test caches.
4. Group requested IDs by record and channel. Use bounded local WFDB reads and
   `read_local_segment`, selecting only the requested channel and exact half-open
   intervals. Record-grouped reads, bounded contiguous chunks, and an in-memory
   cache may reduce repeated I/O without changing values.
5. Validate stable-ID correspondence, source calibration, interval length,
   physical units, and finiteness before the single float32 cast.

The preferred first implementation is a metadata index plus bounded,
record-aware source reads, avoiding a persistent duplicate of every overlapping
window. Performance profiling may justify an external derived waveform cache,
but that cache must remain outside Git and bind the official source SHA-256,
dataset/version, split, B4 protocol digest, input dtype semantics, window
geometry, and every record/channel/start/end stable ID. It must be content
hashed and reproduce elementwise the float32 cast of the canonical reader
values. Cache layout, compression, worker count, streaming, and record grouping
are I/O choices only and cannot alter scientific inputs or ordering semantics.

No annotation value, context flag, handcrafted feature, future sample, or test
partition access may enter waveform retrieval or predictive input construction.

## Experiment lock and test-access policy

B4 development may access train and validation only. Before any B4 test access,
training must be complete and a clean-checkout experiment lock must bind:

- clean Git SHA and `git_dirty=false`;
- SHA-256 of the exact committed `B4_PROTOCOL_V1.md` bytes;
- frozen split SHA-256 and verified source/dataset provenance;
- input, physical-unit, dtype, channel, window, and processing contract;
- canonical digest of sorted selected training stable IDs and sampling policy;
- complete architecture/configuration and trainable parameter count;
- seed and resolved environment;
- selected epoch, validation AUPRC checkpoint-selection record, and duration;
- checkpoint filename, exact SHA-256, and serialized size;
- validation-selected threshold and threshold-selection rule;
- command/provenance record; and
- `test = null`.

The research team has observed historical B0-B3 test results. B4 test evaluation
must therefore be described as a **predeclared one-shot B4 test evaluation under
a test-access firewall**, not as evaluation on a globally unseen test set. B4
architecture or training configuration cannot change because of B0-B3 outcomes.

Future test evaluation must validate the complete lock first, write an attempt
receipt before loading any test waveform row, reject repeat evaluation for that
locked experiment, use only the locked checkpoint and threshold, and never
update the model. This policy is specified here but is not implemented in this
protocol-only phase.

## Evaluation metrics

The future primary metric is pooled-window AUPRC. Secondary metrics are AUROC,
F1, sensitivity, specificity, PPV, NPV, balanced accuracy, and MCC. Report the
same subject-macro metrics as B0-B3. Future test uncertainty uses 1,000
subject-bootstrap replicates with seed `2026`.

At the validation-frozen threshold, rate-related and axis-shift false-positive
fractions are quantitative secondary results. Conduction-change evidence is
exploratory and descriptive only; it is never bootstrapped or headlined. B4 has
no episode-level metric because it has no temporal episode model.

## Explicit exclusions

B4 V1 contains no patient personalization, patient memory, dual-timescale
memory, patient-specific normalization, uncertainty calibration, uncertainty
routing, risk-coverage optimization, edge/cloud routing, temporal episode state
machine, foundation model, knowledge distillation, teacher model, LLM, cloud
inference, clinical ST/J delineation, handcrafted morphology input, multi-lead
fusion, or hardware benchmarking. These belong to later CardioSentinel phases.

## Scientific limitations

- B4 is a global single-channel model and cannot represent patient adaptation,
  lead relationships, or episode evolution.
- Its sigmoid scores are uncalibrated and support neither confidence nor
  uncertainty-routing claims.
- Overlapping windows are correlated observations; the effective inferential
  unit remains the subject.
- The fixed training sampler changes training prevalence, while primary
  validation and future test evaluation remain unsampled.
- The morphology-free design is a comparator, not evidence that clinical ST
  delineation is unnecessary.
- LTSTDB public-dataset results alone cannot establish diagnosis, clinical
  utility, generalization to another cohort, or hardware readiness.
- The B0-B3 test results are historically observed, so only the predeclared
  one-shot B4 firewall claim is valid.

## Freeze statement

`B4_PROTOCOL_V1` defines the prospective B4 experiment before B4
implementation, training, validation performance, or B4 test access. Any
scientific change after training begins requires a new protocol version and
cannot be justified using B4 test results.
