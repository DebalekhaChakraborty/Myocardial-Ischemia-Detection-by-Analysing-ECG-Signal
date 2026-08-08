# B4 Resource Benchmark V1

## Scope

This document freezes the official resource-measurement procedure used by the
B4 architecture comparison. It is frozen **before** any B4-B or B4-C validation
result exists, and before any candidate has been trained.

It applies to exactly three models:

- **B4-A** `B4_raw_compact_cnn_v1` (`B4CompactCNN`)
- **B4-B** `B4B_cnn_transformer_v1` (`B4BTransformerCNN`)
- **B4-C** `B4C_cnn_ssm_v1` (`B4CSSMCNN`)

The measurement runs **only on locked checkpoints**. A model whose experiment
lock does not validate is refused.

This procedure produces the resource half of the Pareto evidence defined in
`docs/B4_ARCHITECTURE_SELECTION_PROTOCOL_V1.md`. It produces no predictive
metric and never touches the dataset.

## Relationship to earlier engineering numbers

The engineering-preflight throughput and RSS figures recorded during candidate
implementation are **not** official evidence and must not be cited as such. They
used process-wide cumulative `ru_maxrss` in a shared process and a different
batch convention. Only measurements produced by this frozen procedure count.

## Frozen environment

The benchmark runs in the exact B4-A scientific software environment, the same
requirement the canonical candidate runs enforce:

```text
python              3.12.6
torch               2.13.0+cpu
numpy               2.3.2
scikit-learn        1.9.0
scipy               1.18.0
wfdb                4.3.1
dependency digest   b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a
device              CPU
```

A differing environment refuses the benchmark. Packages are never installed,
upgraded or downgraded to satisfy it.

## Frozen input

The benchmark is dataset-independent. It uses a deterministic synthetic tensor
and never opens a train, validation or test waveform, cache or metadata file.

```text
generator      torch.Generator seeded with 2026
input          torch.randn -> shape [1, 1, 2500], dtype float32
batch size     1
```

Batch 1 is chosen deliberately: the edge-oriented research goal is per-window
latency, not bulk throughput.

## Frozen inference procedure

```text
model.eval()
model.requires_grad_(False)
with torch.no_grad():
    ... measured calls ...
```

No optimizer is constructed, no backward pass runs, and no weight is modified.

## Frozen threading

```text
torch.set_num_threads(1)          # intra-op threads pinned to 1
```

The inter-op thread setting and the CPU identity are recorded as observed and
are not pinned. Pinning intra-op threads to 1 makes the comparison a property of
the architecture rather than of the host's core count.

## Frozen timing

```text
warm-up calls   50
measured calls  500
timer           time.perf_counter_ns
```

Each measured call times exactly one forward pass on one window. Reported:

- **median latency, milliseconds per window**
- **p95 latency, milliseconds per window**

The predeclared resource tie-break in the architecture protocol uses the
**median** latency. p95 is reported for transparency and is not a tie-break
input.

## Frozen model-size measurement

- `trainable_parameter_count`
- `raw FP32 payload bytes` = `trainable_parameter_count * 4`
- `locked model_selected.pt size in bytes`, the actual serialized artifact

The serialized size is read from the locked checkpoint bound by the experiment
lock, not estimated.

## Frozen peak-memory measurement

`resource.getrusage(RUSAGE_SELF).ru_maxrss` is a **high-water mark for the whole
process and never decreases**. Measuring several models in one process therefore
attributes the largest model's peak to every later model.

Each model is consequently benchmarked in a **fresh subprocess**. The child
process loads exactly one model, runs the procedure, and reports its own
`ru_maxrss` at exit.

Recorded fields:

- `peak_rss` as reported by the child process
- `peak_rss_units`, platform dependent; on Linux `ru_maxrss` is **kibibytes**
- `measurement_method`, naming `ru_maxrss` and the fresh-subprocess isolation
- `peak_rss_available`, a boolean

If peak RSS cannot be measured reliably and comparably across all three models,
it is recorded as **unavailable** and the predeclared Pareto and tie-break rules
skip that dimension, moving to the next predeclared item. No value is invented,
imputed, estimated or substituted.

## Result integrity

Every benchmark result records the resolved environment, threading, CPU
identity, procedure constants, model identity, locked experiment lock SHA-256
and locked checkpoint SHA-256, and carries its own canonical
`benchmark_result_sha256` over the immutable scientific fields.

Timing values are excluded from that digest's immutable-identity portion only
where they are genuinely measurement outputs; the digest covers the full
reported payload so a result cannot be edited after the fact.

## Prohibitions

The benchmark must never:

- open any train, validation or test waveform, cache, metadata or prediction;
- construct an optimizer, call backward, or modify any weight;
- load a model whose experiment lock fails validation;
- run several models in one process for peak-memory purposes;
- be re-run selectively for one candidate to obtain a more favourable number.

The implementation is frozen before results exist and must not be changed after
any candidate's latency or memory figure is observed.

## Freeze statement

`B4_RESOURCE_BENCHMARK_V1` defines the official resource evidence procedure for
the B4-A/B4-B/B4-C comparison, frozen before B4-B and B4-C were trained. Any
change requires a new version recorded before the corresponding measurements are
observed.
