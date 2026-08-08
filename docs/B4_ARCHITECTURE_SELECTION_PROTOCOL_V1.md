# B4 Architecture Selection Protocol V1

## Scope

This document prospectively freezes the CardioSentinel Phase 3B-2C architecture
comparison. It defines exactly one B4-B architecture and exactly one B4-C
architecture, their shared input and training contract, the evidence used to
choose between them, and the decision rule. It is written before either
architecture is implemented, trained, or validated.

B4-A, B4-B and B4-C are global single-channel comparator baselines. None of them
is the CardioSentinel contribution. None contains personalization, patient
memory, uncertainty calibration, temporal episode reasoning, longitudinal
across-window context, foundation-model knowledge, or cloud inference.

This is research software protocol. It is not a diagnostic, clinical
effectiveness, or medical-device claim.

## Research-handbook mapping

| Handbook item | Meaning | Status in this protocol |
| --- | --- | --- |
| B4-A | Compact CNN/TCN global raw-waveform reference | Complete and frozen; historical comparator |
| B4-B | CNN local stem + tiny Transformer short-window attention | Frozen here; not implemented |
| B4-C | CNN local stem + compact short-window state-space model | Frozen here; not implemented |
| B4-D | CNN + tiny Transformer + longitudinal SSM hybrid | Conditional; **not authorized** |
| T2 | Longitudinal SSM over successive window embeddings | Reserved; separate later phase |
| T1 | Episode state machine | Reserved; separate later phase |
| M1 | Patient dual-memory | Reserved; separate later phase |
| M2 | Contamination-safe adaptation | Reserved; separate later phase |
| U1 / U2 | Calibrated/selective uncertainty and conformal routing | Reserved; separate later phase |
| E1 | Edge and hardware-in-the-loop benchmarking | Reserved; separate later phase |

### B4-C does not satisfy T2

This distinction is load-bearing and must not be lost.

**B4-C** applies a state-space recurrence **inside one completed 10-second
window**, over the 79 temporal tokens produced by the convolutional front end.
Its state is created at the start of a window and discarded at the end of that
window. It carries no information between windows.

**T2** is a different experiment: a longitudinal state-space model over the
sequence of *successive 10-second window embeddings*, spanning tens of seconds
to minutes, whose purpose is temporal episode evidence accumulation.

Implementing B4-C therefore does **not** implement, approximate, or discharge
T2. No later work may claim that intra-window state-space modelling already
covered longitudinal temporal or episode modelling. T1 and T2 remain required.

The preferred final research story is unchanged: compact encoder, physiology,
patient dual-memory, contamination-safe adaptation, calibrated and selective
uncertainty, longitudinal temporal modelling, NORMAL/WATCH/EVENT/RECOVERY, and
edge/cloud execution.

## Research questions

- **B4-B**: does short-window relational attention add value beyond convolution
  on the same completed 10-second window?
- **B4-C**: can a compact state-space temporal representation provide a better
  predictive/resource trade-off than short-window attention?

Both are questions about the temporal block only. The front end and the
classifier head are held constant so that the temporal block is the changed
factor.

## Frozen benchmark identity

B4-B and B4-C evaluate the identical scientific population as B4-A.

- Dataset: Long-Term ST Database (`ltstdb`) v1.0.0.
- Primary annotation definition: `ltstdb.stb`, unchanged from Benchmark V1.
- Split SHA-256:
  `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`.
- Feature corpus SHA-256:
  `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5`.
- Training-selection SHA-256:
  `318da148da5d638af44e73c06c00cc4df2815017d4ce8bb1a1b864e53eda8009`.
- Waveform-cache SHA-256:
  `dcac70260c92a8a4934dfcaa120e22fee939a976b63bf880f75ae176993d3ed2`.
- Window geometry: completed causal 10-second windows every 5 seconds.

| Population | Rows | Ischemic positive | Background negative | Subjects |
| --- | ---: | ---: | ---: | ---: |
| Selected train | 374,452 | 93,613 | 280,839 | 56 |
| Full primary validation | 473,897 | 21,628 | 452,269 | 12 |

The existing lossless train/validation waveform cache is reused unchanged
because the scientific waveform population is identical. No new cache is
materialized, and the cache identity above is verified before any training.

## Common input contract

```text
channels:            1
window:              10 seconds
sampling frequency:  250 Hz
samples:             2500
physical unit:       mV
batch tensor:        [B, 1, 2500]
training dtype:      IEEE 754 float32
processing profile:  raw identity
```

Prohibited as predictive input for every candidate: high-pass, low-pass or notch
filtering; per-window, per-record or per-patient normalization; z-scoring;
amplitude rescaling; handcrafted `signal_v1` or `morphology_v1` values; subject
ID; record ID; channel index or lead identity; expert ST or J-point
measurements; annotations; labels; context flags; and any sample beyond the
window endpoint.

Identity metadata remains provenance and grouping information only, exactly as
in B4-A.

## Common training contract

To isolate architecture as the changed factor, B4-B and B4-C reuse the frozen
B4-A training semantics without exception. No architecture-specific training
difference is required by either design, and none is authorized.

- Seed: `2026` for Python, NumPy, PyTorch and CUDA where applicable.
- Numerical mode: FP32 only; automatic mixed precision is off.
- Loss: `BCEWithLogitsLoss(reduction="mean")`, no class or sample weighting.
- Optimizer: one AdamW group over every trainable parameter, `lr=1e-3`,
  `weight_decay=1e-4`, `betas=(0.9, 0.999)`, `eps=1e-8`, `amsgrad=False`,
  `foreach=False`, `fused=False`.
- Batch size 256; `drop_last=False`.
- Maximum 15 completed training epochs; no learning-rate scheduler.
- Early stopping: stop after four consecutive completed epochs without a
  validation AUPRC increase strictly greater than `1e-6` over the early-stopping
  reference.
- Checkpoint selection: maximum full-primary-validation AUPRC; an exact tie
  keeps the earliest epoch.
- Threshold: maximum F1 over exact observed full-primary-validation scores, with
  the highest threshold winning an exact tie. Selected on validation only.
- No augmentation, oversampling beyond the frozen selection, hyperparameter
  search, restart selection, or test-informed choice.

Model scores are **uncalibrated sigmoid model scores**. They are not calibrated
probabilities and not confidence. Calibration remains a later phase.

Determinism, environment capture, execution-argument provenance, crash safety,
one-canonical-run enforcement and lock creation follow the existing frozen
runner semantics used for B4-A.

## Shared convolutional front end

B4-B and B4-C reuse the **exact** B4-A stem and four downsampling blocks,
unchanged, including kernel sizes, strides, padding, GroupNorm groups and SiLU
placement:

```text
Input: [B, 1, 2500]

Stem:
  Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False)
  GroupNorm(8, 32); SiLU

Downsampling block 1: 32 -> 48, kernel=9, stride=2
Downsampling block 2: 48 -> 64, kernel=7, stride=2
Downsampling block 3: 64 -> 96, kernel=5, stride=2
Downsampling block 4: 96 -> 128, kernel=5, stride=2

Each downsampling block:
  depthwise Conv1d(Cin, Cin, kernel, stride=2, padding=(kernel-1)//2,
                   groups=Cin, bias=False)
  GroupNorm(8, Cin); SiLU
  pointwise Conv1d(Cin, Cout, kernel_size=1, bias=False)
  GroupNorm(8, Cout); SiLU

Output: [B, 128, 79]
```

Temporal lengths are `2500 -> 1250 -> 625 -> 313 -> 157 -> 79`, so the token
sequence length is exactly **79** and the token dimension is exactly **128**.
Front-end trainable parameters: **26,160**.

The classifier head is likewise reused unchanged from B4-A:

```text
Head:
  AdaptiveAvgPool1d(1); flatten
  Dropout(p=0.10)
  Linear(128, 64); SiLU
  Dropout(p=0.10)
  Linear(64, 1)
```

Head trainable parameters: **8,321**. Output is one raw binary logit; no sigmoid
is part of any model.

Holding the front end and head identical means B4-A, B4-B and B4-C differ only
in the temporal block that consumes the 79x128 token sequence. B4-A's temporal
block is three dilated residual convolution blocks (52,608 parameters).

Exact front-end reuse is technically sensible for both candidates, so no
front-end difference is authorized.

## Causality interpretation

A window becomes available only at its exclusive end sample, after every
required sample has arrived. Once available, the complete 10-second window is
in hand.

Self-attention in B4-B and the state-space recurrence in B4-C therefore operate
across tokens **within one completed window**, which uses no sample later than
that window's endpoint. This is the same treatment already frozen for B4-A,
whose dilated context blocks use symmetric padding and are likewise
bidirectional within the completed window. The comparison is therefore fair and
consistent with existing precedent.

No candidate may consume any sample, token, embedding, state or label from a
future window. No candidate carries state across windows.

## B4-B: CNN local stem + tiny Transformer

### Architecture

```text
Front end (shared, frozen)            -> [B, 128, 79]
transpose                             -> [B, 79, 128]
add learned positional embedding P    -> [B, 79, 128],  P has shape [79, 128]

Encoder block x 2 (pre-norm):
  h <- x + Dropout_0.10(MHSA(LayerNorm(x)))
  x <- h + Dropout_0.10(FFN(LayerNorm(h)))

Final LayerNorm(128)
transpose back                        -> [B, 128, 79]
Head (shared, frozen)                 -> one raw logit
```

- `FFN`: `Linear(128, 256)` then GELU then `Linear(256, 128)`. The expansion
  factor is 2, chosen for compactness rather than the conventional 4.
- Positional representation: a learned absolute embedding of shape `[79, 128]`,
  added once before the first block. This is exact only because the token count
  is frozen at 79 by the frozen window geometry.

Two encoder blocks, 4 heads, and an expansion factor of 2 are the single frozen
configuration. No depth, head-count, width, expansion or dropout sweep is
authorized.

### Frozen multi-head attention semantics

```text
embed_dim        = 128
num_heads        = 4
head_dim         = 32
bias             = true          (input and output projections)
batch_first      = true
attention span   = full 79 in-window tokens (bidirectional)
attn_mask        = none
key_padding_mask = none
attention dropout = 0.0
```

### Frozen dropout placement

The **only** dropout inside a B4-B encoder block is the external residual-branch
dropout already shown above, at `p=0.10`, inherited from B4-A's frozen head
rather than tuned:

```text
h = x + Dropout_0.10( MHSA(LayerNorm(x)) )
x = h + Dropout_0.10( FFN(LayerNorm(h)) )
```

Internal attention-weight dropout is **0.0**. Both internal attention dropout and
external branch dropout must not be applied together: a `torch.nn.MultiheadAttention`
must be constructed with `dropout=0.0`, and a prebuilt
`torch.nn.TransformerEncoderLayer` must not be used unless its internal dropout
is likewise driven to `0.0`, because its default couples several dropout sites.

### Frozen initialization

Every initialization below is drawn once, after `initialize_determinism(seed=2026)`
has established all seeds and deterministic settings, and before any optimizer
step.

- Positional embedding: `P ~ Normal(mean=0.0, std=0.02)`, elementwise and
  independent, shape `[79, 128]`.
- Every `LayerNorm`: `weight = 1`, `bias = 0`, `eps = 1e-5`, trainable affine.
- Every `Linear` and every `MultiheadAttention` projection weight and bias: the
  **PyTorch module default initialization of the resolved PyTorch version**,
  applied after `initialize_determinism(seed=2026)`. The scientific run records
  the exact resolved PyTorch version in its environment provenance, so the
  initialization is reproducible from the recorded environment plus the seed.
  The reference environment is PyTorch 2.13; a different resolved version is
  permitted only if it is recorded, and the run is then reproducible against
  that recorded version.

No initialization is tuned, searched, or revised after any validation result.

### Trainable parameters

Linear layers count `out * in + out`. `LayerNorm(d)` contributes `2d`.
Multi-head attention contributes `3 * (d*d) + 3d` for the packed input
projection and `d*d + d` for the output projection.

| Component | Trainable parameters |
| --- | ---: |
| Shared front end | 26,160 |
| Positional embedding `[79, 128]` | 10,112 |
| Per block: LayerNorm | 256 |
| Per block: multi-head self-attention | 66,048 |
| Per block: LayerNorm | 256 |
| Per block: feed-forward `128 -> 256 -> 128` | 65,920 |
| **Per block subtotal** | **132,480** |
| Encoder blocks (x2) | 264,960 |
| Final LayerNorm | 256 |
| Shared head | 8,321 |
| **B4-B total** | **309,809** |

Raw FP32 parameter payload: `309,809 * 4 = 1,239,236` bytes, approximately
1.182 MiB. The serialized state dictionary will be slightly larger because of
tensor and container metadata; the implementation must record its actual file
size and SHA-256.

309,809 is below the hard ceiling of 1,000,000 trainable parameters.

## B4-C: CNN local stem + compact diagonal state-space model

### Terminology

This block is a **diagonal, gated, S4D-inspired state-space model**. It is not
Mamba and must not be called Mamba: its state transition is input-independent,
so it implements no selective state-space mechanism. The name used throughout
the implementation must be `DiagonalGatedSSMBlock` or an equally explicit
equivalent.

### Why this is a state-space model and not an RNN with a label

The recurrence is derived from a continuous-time linear state-space system and
then discretized, and its transition matrix is diagonal, time-invariant and
independent of the input.

The claims below are made about the **linear SSM core** `y = SSM(u)`, not about
the complete block. Three consequences follow that no gated RNN provides:

1. The state transition of the SSM core is a fixed diagonal linear operator, so
   the core is a linear time-invariant (LTI) system per channel. Its impulse
   response has a closed form, and the **SSM core alone** therefore has an exact
   causal-convolution representation.
2. Because the core recurrence is linear and input-independent, it is associative
   and is in principle evaluable by parallel scan or in the frequency domain. A
   gated RNN, whose transition depends on the input, is not.
3. Stability is guaranteed by construction through the parameterization below,
   not by clipping, spectral normalization or gradient tricks.

### Scope of the LTI claim

The **complete `DiagonalGatedSSMBlock` is not LTI and is not
convolution-equivalent.** It contains `LayerNorm`, the learned `Linear_in` and
`Linear_out` projections, `SiLU` gating that multiplies the core output by a
data-dependent signal, and a residual pathway. Those components are nonlinear or
input-dependent, so no exact convolution representation exists for the block as a
whole.

The architecture is nonetheless legitimately a state-space architecture because
its **temporal memory operator** — the only component that carries information
across tokens — is explicitly state-space-derived, diagonal, time-invariant and
input-independent.

Restated for the avoidance of doubt: this is **not Mamba**, and there is **no
selective or input-dependent state transition** anywhere in the design.

### Equations

Let the front end produce tokens `z[k] in R^128` for `k = 1..79`. Write `H = 128`
for the channel count and `N = 16` for the state dimension per channel.

Per block, with pre-normalization and a residual connection:

```text
v      = LayerNorm(z)
(u, g) = split(Linear_in(v)),  Linear_in: R^128 -> R^256, u,g in R^128
y      = SSM(u)                                     (per-channel, see below)
o      = Linear_out( y  *  SiLU(g) )                Linear_out: R^128 -> R^128
z_out  = z + Dropout(o)
```

The state-space operator `SSM` acts independently on each channel
`c in {1..H}`. For channel `c` and state index `n in {1..N}`, the continuous
system is

```text
dx_{c,n}(t)/dt = lambda_{c,n} * x_{c,n}(t) + B_{c,n} * u_c(t)
y_c(t)         = sum_n Re( C_{c,n} * x_{c,n}(t) ) + D_c * u_c(t)
```

with `lambda_{c,n}` and `C_{c,n}` complex and `B_{c,n}`, `D_c` real.

Stability parameterization. The real part is forced strictly negative and the
step strictly positive:

```text
lambda_{c,n} = -exp( a_{c,n} ) + i * w_{c,n}          a, w learned real
Delta_c      =  exp( d_c )                            d learned real
```

Zero-order-hold discretization with step `Delta_c` gives the discrete recurrence

```text
Abar_{c,n} = exp( Delta_c * lambda_{c,n} )
Bbar_{c,n} = ( Abar_{c,n} - 1 ) / lambda_{c,n} * B_{c,n}

x_{c,n}[k] = Abar_{c,n} * x_{c,n}[k-1] + Bbar_{c,n} * u_c[k]
y_c[k]     = sum_n Re( C_{c,n} * x_{c,n}[k] ) + D_c * u_c[k]
```

with `x_{c,n}[0] = 0`. Because `Re(lambda_{c,n}) = -exp(a_{c,n}) < 0` strictly
and `Delta_c > 0`, the discrete pole magnitude satisfies

```text
| Abar_{c,n} | = exp( Delta_c * Re(lambda_{c,n}) ) < 1
```

for every channel and state, so the recurrence is unconditionally stable with no
clipping or renormalization. `lambda_{c,n} = 0` cannot occur, so `Bbar` is well
defined.

### Frozen numerically stable ZOH computation

The mathematics above is unchanged. The **implementation** must evaluate `Bbar`
with a complex `expm1` rather than forming `exp(z) - 1` explicitly:

```text
z          = Delta_c * lambda_{c,n}
Abar_{c,n} = exp( z )
Bbar_{c,n} = expm1( z ) / lambda_{c,n} * B_{c,n}
```

`expm1(z)` and `exp(z) - 1` are algebraically identical. They differ numerically:
when `|z|` is small, `exp(z)` is close to `1`, so `exp(z) - 1` cancels the leading
digits and loses relative precision, while `expm1` is designed to retain it. Small
`|z|` is the expected regime here because `Delta_c` is initialized as low as
`1e-3`. `lambda_{c,n}` is nonzero by parameterization, so the division is safe.

This freezes the numerically stable implementation of an unchanged scientific
model before training.

### Frozen initialization

All draws happen once, after `initialize_determinism(seed=2026)` has established
every seed and deterministic setting, and before any optimizer step. State index
`n` runs over `n = 1, 2, ..., 16` (one-based; `N = 16`).

```text
w_{c,n} = pi * n                    for n = 1..16     (linear frequency grid)
a_{c,n} = log(1/2)                                    (so Re(lambda) = -1/2)
d_c     ~ Uniform( log(1e-3), log(1e-1) )
Delta_c = exp( d_c )                                  (log-uniform on [1e-3, 1e-1])
B_{c,n}     ~ Normal(0, 1) / sqrt(N)
Re(C_{c,n}) ~ Normal(0, 1) / sqrt(N)
Im(C_{c,n}) ~ Normal(0, 1) / sqrt(N)
D_c     = 0
```

`d_c` is the learned **log** step, so it is drawn uniformly on
`[log(1e-3), log(1e-1)]`; the resulting step `Delta_c = exp(d_c)` is therefore
log-uniform on `[1e-3, 1e-1]`. The earlier phrasing "`d_c` log-uniform over
`[log(0.001), log(0.1)]`" was ambiguous about which quantity was log-uniform and
is superseded by the contract above.

The frequency grid `w_{c,n} = pi * n` is a **linear frequency grid in the spirit
of S4D**. It is used because it is simple, deterministic and reproducible. No
claim is made that it reproduces any specific published initializer exactly.

The implementation must record the exact initialization it used.

The state exists only within one window. It is created at `k = 1` and discarded
after `k = 79`.

### Frozen numerical dtype contract

The shared scientific input remains `float32`. No `float64` or `complex128`
scientific forward path is authorized.

| Quantity | dtype |
| --- | --- |
| Block input `z`, `u`, `g` | `torch.float32` |
| Real trainable tensors `a`, `w`, `d`, `B`, `Re(C)`, `Im(C)`, `D` | `torch.float32` |
| `lambda` | `torch.complex64` |
| `C` | `torch.complex64` |
| `Abar` | `torch.complex64` |
| `Bbar` | `torch.complex64` |
| Recurrent state `x` | `torch.complex64` |
| SSM core output `y` | `torch.float32` |
| Complete model output raw logit | `torch.float32` |

Complex tensors are assembled from the real `float32` parameters, so every
trainable parameter is stored and counted as real `float32`. The parameter table
below is unaffected.

### Frozen canonical forward evaluation

The V1 scientific implementation uses exactly one evaluation of the SSM core: the
straightforward discrete recurrence over the fixed 79-token sequence.

```text
x[0] = 0
for k = 1..79:
    x[k] = Abar * x[k-1] + Bbar * u[k]
    y[k] = sum_n Re( C * x[k] ) + D * u[k]
```

The `H x N` state is held as a tensor and only the temporal dimension is
iterated, so the loop runs exactly 79 steps regardless of batch size.

For the scientific B4-C V1 run this **excludes** FFT or frequency-domain
convolution, associative or parallel scan, custom CUDA kernels, and any external
state-space library. The sequence is only 79 tokens, so the plain recurrence is
inexpensive, and fixing it makes numerical and provenance semantics unambiguous.

The LTI core is theoretically parallelizable, as noted above, but every reported
B4-C V1 resource measurement refers to this frozen recurrence implementation. The
implementation must not be changed after any validation or latency result is
observed.

### Architecture

```text
Front end (shared, frozen)            -> [B, 128, 79]
transpose                             -> [B, 79, 128]

DiagonalGatedSSMBlock x 2  (pre-norm, residual, as above)

Final LayerNorm(128)
transpose back                        -> [B, 128, 79]
Head (shared, frozen)                 -> one raw logit
```

- Two blocks, `N = 16`, `H = 128` is the single frozen configuration. No block
  count, state dimension, expansion or dropout sweep is authorized.
- Dropout `p=0.10` on the block residual branch, inherited from B4-A's frozen
  head rate, not tuned.
- No positional embedding is used or needed: the recurrence is inherently
  ordered, which is a genuine architectural difference from B4-B and is recorded
  as such rather than corrected.

### Trainable parameters

| Component | Shape | Trainable parameters |
| --- | --- | ---: |
| Shared front end | | 26,160 |
| Per block: LayerNorm | `[128] x 2` | 256 |
| Per block: `Linear_in` `128 -> 256` | | 33,024 |
| Per block: `a` (state decay) | `[128, 16]` | 2,048 |
| Per block: `w` (state frequency) | `[128, 16]` | 2,048 |
| Per block: `B` | `[128, 16]` | 2,048 |
| Per block: `Re(C)` | `[128, 16]` | 2,048 |
| Per block: `Im(C)` | `[128, 16]` | 2,048 |
| Per block: `D` (skip) | `[128]` | 128 |
| Per block: `d` (log step) | `[128]` | 128 |
| Per block: SSM core subtotal | | 10,496 |
| Per block: `Linear_out` `128 -> 128` | | 16,512 |
| **Per block subtotal** | | **60,288** |
| SSM blocks (x2) | | 120,576 |
| Final LayerNorm | | 256 |
| Shared head | | 8,321 |
| **B4-C total** | | **155,313** |

Raw FP32 parameter payload: `155,313 * 4 = 621,252` bytes, approximately
0.592 MiB. The implementation must record the actual serialized size and
SHA-256.

155,313 is below the hard ceiling of 1,000,000 trainable parameters.

## Candidate summary

| Candidate | Temporal block | Trainable parameters | FP32 payload bytes |
| --- | --- | ---: | ---: |
| B4-A (frozen, complete) | 3 dilated residual conv blocks | 87,089 | 348,356 |
| B4-B | 2 tiny Transformer encoder blocks | 309,809 | 1,239,236 |
| B4-C | 2 diagonal gated SSM blocks | 155,313 | 621,252 |

The parameter counts above are predictions from the frozen arithmetic. Each
implementation must verify its constructed model against them before training
and must fail if they differ.

## Architecture-selection evidence

Architecture selection uses **development validation evidence only**. No test
partition is opened, enumerated, hashed, scored, or consulted at any point in
the selection.

Primary comparison dimension:

- pooled full-primary-validation AUPRC.

Mandatory supporting dimensions:

- subject-macro AUPRC, with contributing and non-contributing subject counts;
- pooled and subject-macro AUROC;
- F1, sensitivity and PPV at each candidate's own validation-selected threshold;
- rate-related challenge false-positive fraction, quantitative secondary;
- axis-shift challenge false-positive fraction, quantitative secondary.

Resource dimensions:

- trainable parameter count;
- serialized FP32 artifact size;
- CPU inference latency under the recorded environment;
- peak inference memory where it can be measured reliably.

Conduction-change evidence remains `exploratory_descriptive`. It is reported as
`FP / N` with contributing subjects, is never bootstrapped, and is never a
selection criterion.

Every candidate reports its evidence under the frozen metrics protocol,
including undefined metrics preserved as undefined.

## Pareto decision rule

The selected architecture is not "the fanciest model" and not the winner of a
scalar score invented after results are seen.

### Resource vector

Define, for each candidate, the predeclared resource vector

```text
R = ( trainable_parameter_count,
      serialized_FP32_bytes,
      measured_CPU_inference_latency,
      measured_peak_inference_memory )
```

Lower is better in every component of `R`.

### Formal dominance

Candidate `X` **Pareto-dominates** candidate `Y` if and only if all three hold:

1. `AUPRC(X) >= AUPRC(Y)` on pooled full-primary validation; **and**
2. `X` is no worse than `Y` in **every available** predeclared resource
   dimension of `R`; **and**
3. `X` is strictly better than `Y` in at least one of: validation AUPRC, or some
   resource dimension of `R`.

If the resource measurements trade off against each other, that is, `X` is better
in one component of `R` and worse in another, then **neither candidate is
Pareto-dominant on resources** and no dominance claim may be made in either
direction.

A Pareto-dominated candidate must not be selected.

### Review dimensions that are not part of dominance

Subject-macro AUPRC and the rate-related and axis-shift challenge false-positive
fractions are mandatory safety and support review dimensions. They are examined
before the decision is recorded, and a material robustness concern in them may
block an otherwise attractive selection. They are deliberately **not** folded
into the dominance test and must not be converted into a post-hoc scalar score.

Conduction-change evidence remains descriptive and is never a selection criterion.

### Trade-offs

A small validation AUPRC improvement does not automatically justify a materially
larger, slower or more memory-hungry edge model. Where candidates trade
predictive performance against resource use, the selection record must state the
size of the predictive difference alongside the size of the resource difference
and justify the choice against the edge-oriented research goal. Stating a
trade-off is a permitted outcome; it must not trigger any additional tuning,
retraining or new configuration.

### Prohibited

Constructing a weighted scalar "selection score" after results are known;
breaking a tie with test performance; consulting the historical B0-B3 test
results to justify a B4 architecture choice; and re-running any candidate because
its first validation result was disappointing.

## Resource tie-break

If the predictive and supporting evidence genuinely does not separate the
candidates, and no candidate Pareto-dominates the other across the resource
vector, the following **lexicographic** resource preference applies. It is frozen
here, before any result is observed, and reflects the edge-oriented research
goal:

1. lower measured **median CPU inference latency**;
2. lower measured **peak inference memory**;
3. fewer **trainable parameters**;
4. smaller **serialized FP32 model size**.

Each item is consulted only if every higher-priority item is tied.

All measurements must use the same recorded CPU environment, the same batch
convention and the same benchmark procedure across candidates; otherwise they are
not comparable and must not be used.

If a higher-priority resource measure is unavailable or unreliable in the
recorded environment, the selection record must state that explicitly and move to
the next predeclared item. A missing measurement is never imputed, estimated or
substituted, and no resource score may be invented after results.

## Experiment identifiers

| Identifier | Meaning |
| --- | --- |
| `B4A_cnn_v1` | Logical name for the completed compact-CNN reference |
| `B4_raw_compact_cnn_v1` | **Historical run ID actually on disk for B4-A** |
| `B4B_cnn_transformer_v1` | B4-B canonical run |
| `B4C_cnn_ssm_v1` | B4-C canonical run |
| `B4_architecture_selection_v1` | Validation-only architecture-selection record |

`B4A_cnn_v1` is an alias for documentation only. The completed B4-A run keeps
its historical identifier `B4_raw_compact_cnn_v1`. Its run directory, lock,
checkpoint, threshold and artifacts are never renamed, moved or rewritten.

Each of B4-B and B4-C is a single canonical run under the existing
one-canonical-run enforcement: a completed or interrupted run refuses any repeat,
restart or fresh-seed retry without documented human review.

## Predeclared run sequence

- **A.** Merge this protocol.
- **B.** Implement and harden B4-B and B4-C with synthetic fixtures and smoke
  checks only. No scientific training.
- **C.** Review and CI.
- **D.** Scientific B4-B train, validation and lock. No test access.
- **E.** Scientific B4-C train, validation and lock. No test access.
- **F.** Validation-only architecture-selection report. No retraining.
- **G.** Freeze the selected global encoder.
- **H.** Only then review the sealed-test path for the selected architecture.

No architecture receives an iterative second configuration because its first
validation result was disappointing. A scientific change after training begins
requires a new protocol version and cannot be justified using any test result.

## Test-access policy

The B4-A sealed test is unopened and remains unopened. The one-shot sealed-test
evaluator merged in Phase 3B-2B.2 exists on `master` but must not be executed
during architecture selection.

For the whole of B4-B and B4-C development and selection, the test partition is
unavailable: no test metadata, no test feature caches, no test waveforms, no
test predictions and no test metrics.

After B4-A, B4-B and B4-C validation evidence is frozen and the architecture
selection is documented, only the selected global architecture is considered for
the architecture family's one-shot sealed-test evaluation. This protocol does
**not** authorize that evaluation.

The existing evaluator is specific to the B4-A architecture. If the selected
architecture is not B4-A, that evaluator requires a separate reviewed
generalization before any test access, and the generalization must be reviewed
before the test is opened, not during.

## Scientific limitations

- B4-B and B4-C remain global single-channel models. Neither represents patient
  adaptation, lead relationships, or episode evolution.
- Their sigmoid scores are uncalibrated and support no confidence or
  uncertainty-routing claim.
- Overlapping windows are correlated observations; the effective inferential unit
  remains the subject.
- Validation evidence is used for checkpoint selection, threshold selection and
  architecture selection, so every validation figure carries selection bias.
  Architecture-selection results are therefore not generalization estimates.
- The fixed training sampler changes training prevalence, while primary
  validation remains unsampled.
- An architecture comparison on one public dataset cannot establish diagnosis,
  clinical utility, generalization to another cohort, or hardware readiness.
- B4-A's observed development result is validation AUPRC `0.3156014611186772`,
  validation AUROC `0.8675598293803359`, selected epoch 4. See the historical
  provenance statement below for exactly what was and was not known when the
  B4-B and B4-C configurations were frozen.

## B4-A historical provenance

This statement is deliberately precise rather than flattering, because the
sequence matters for how the architecture comparison may be interpreted.

The B4-B and B4-C architecture **families** were predeclared in the research
handbook before any B4 sealed-test access. Their **exact V1 configurations** were
frozen in this document *after* the B4-A development validation result had
already been observed, but *before* either candidate was implemented or trained,
and while every B4 test artifact remained unopened.

The B4-A validation result is therefore historical context. It is **not** an
optimization objective, **not** a hyperparameter search signal, and **not** an
authorization for iterative tuning of B4-B or B4-C. No candidate configuration in
this document may be revised because of it, and no candidate may receive a second
configuration because its own first validation result is disappointing.

No claim of blindness to the B4-A validation result is made, because that
blindness did not exist. What is claimed, and what the run sequence enforces, is
that every B4-B and B4-C configuration is fixed before those candidates are
trained, and that no test artifact informed any of it.

## B4-D status

B4-D, a CNN plus tiny Transformer plus longitudinal SSM hybrid, is **conditional
and not authorized**. It is not part of this architecture-selection run and must
not be implemented under this protocol.

The high-value longitudinal state-space model is reserved primarily for the
later T2 temporal and episode phase. B4-D must not delay patient memory,
contamination-safe adaptation, uncertainty, temporal episode reasoning, or edge
validation.

## Freeze statement

`B4_ARCHITECTURE_SELECTION_PROTOCOL_V1` defines exactly one B4-B architecture
and exactly one B4-C architecture, their shared contract, the selection evidence
and the decision rule, before either is implemented, trained or validated.

There is no grid search, layer-count sweep, hidden-size sweep, dropout sweep or
learning-rate search anywhere in this protocol.

Any architecture or hyperparameter change after this document is merged requires
an explicit protocol revision, recorded **before** the corresponding validation
results are observed. `B4_PROTOCOL_V1` is unchanged by this document and
continues to govern B4-A.
