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

The check runs **inside each child benchmark process**, before model warm-up,
before any timed call, and before any memory evidence is accepted. It reuses the
same `require_exact_scientific_environment` semantics the canonical candidate
runs enforce, so there is exactly one definition of "the scientific environment"
in the repository.

## Frozen protocol identity

Before any official measurement, the implementation hashes the exact bytes of
this document and requires the frozen digest. Every benchmark result records:

```text
resource_benchmark_protocol         B4_RESOURCE_BENCHMARK_V1
resource_benchmark_protocol_sha256  <validated digest of this file>
```

The protocol digest is covered by `benchmark_result_sha256`, so a result cannot
be detached from the procedure that produced it.

## Official locked-model requirements

The official path accepts exactly three models and requires each to match its
frozen mapping:

| Model | `experiment_id` | Architecture | Trainable parameters |
| --- | --- | --- | ---: |
| B4-A | `B4_raw_compact_cnn_v1` | `B4CompactCNN` | 87,089 |
| B4-B | `B4B_cnn_transformer_v1` | `B4BTransformerCNN` | 309,809 |
| B4-C | `B4C_cnn_ssm_v1` | `B4CSSMCNN` | 155,313 |

Each official lock must additionally satisfy:

- `status = locked_for_one_shot_test`
- `test = null`
- `git_dirty = false`
- `model.verified_against_constructed_model = true`
- locked checkpoint SHA-256 matches the lock
- locked checkpoint byte size matches the lock where the lock binds one
- trainable parameter count exactly equal to the frozen value above
- the frozen `B4_PROTOCOL_V1` digest
- for B4-B and B4-C, additionally the frozen
  `B4_ARCHITECTURE_SELECTION_PROTOCOL_V1` digest
- an environment recording the frozen dependency digest

B4-A's lock predates the candidate-specific fields. Where a field genuinely does
not exist in that historical lock, the strongest historically available
equivalent is validated instead; nothing is fabricated to make it pass, and no
requirement that B4-A does satisfy is weakened.

## Official A/B/C suite

Selective reruns are prohibited, so the official path is a **single suite
invocation covering all three models**. It accepts exactly three locked run
directories, one per model. No model may be omitted, no fourth model may be
added, and no single model may be benchmarked officially on its own.

The suite executes in this fixed predeclared order:

```text
B4-A  ->  B4-B  ->  B4-C
```

Each model still runs in its own fresh subprocess. The combined result contains
every individual result payload and digest.

### Same-host comparability

Resource comparison is only meaningful on one host. After all three children
finish, the suite requires exact equality across all three child results for:

```text
python version, torch version, numpy version, dependency digest,
platform, CPU model, device, intra-op threads, inter-op threads
```

and requires intra-op threads to be exactly 1. Any difference **refuses** the
official suite result before it can be accepted as completed.

### Suite repeat protection

The official suite writes its evidence to a Git-ignored directory, for example
`cardiosentinel-runs/phase3b2-architecture-v1/B4_architecture_resource_benchmark_v1/`,
holding `RESOURCE_BENCHMARK_ATTEMPT.json` and `RESOURCE_BENCHMARK_RESULTS.json`.

Before the first model is measured, the suite atomically and exclusively claims
the attempt. An existing attempt or result **refuses** another automatic official
benchmark. There is no `--force`, `--best-of`, `--retry-one`, `--rerun-candidate`
or `--overwrite`.

If the suite fails after the attempt is claimed, it records
`FAILED_OR_INTERRUPTED` with `human_review_required = true`. The failed model is
never selectively retried. Any human-authorized recovery reruns the **full**
suite under explicit documented governance, never one model alone.

Synthetic unit tests in temporary fixture directories are unaffected.

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

The p95 definition is frozen here as the **nearest-rank** percentile, so no
rounding convention can drift:

```text
rank = ceil(0.95 * N)          with N = 500 measured samples
p95  = sorted_samples[rank - 1]
```

For `N = 500` this is `rank = 475`, that is the 475th smallest sample. The
implementation computes exactly this and does not rely on any library rounding
behaviour.

The predeclared resource tie-break in the architecture protocol uses the
**median** latency. p95 is descriptive only and is never a tie-break input.

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

Every individual benchmark result records the resolved environment, threading,
CPU identity, procedure constants, model identity, this protocol's digest, the
locked experiment-lock SHA-256 and the locked checkpoint SHA-256, and carries its
own canonical `benchmark_result_sha256` over the full reported payload, so a
result cannot be edited after the fact.

The combined official suite result additionally binds:

```text
resource benchmark protocol SHA-256
architecture protocol SHA-256
B4 protocol SHA-256
suite attempt identity
the fixed candidate order
three experiment-lock SHA-256 values
three locked checkpoint SHA-256 values
three individual benchmark-result SHA-256 values
the exact shared environment and host identity
timing procedure and memory procedure
model sizes, median and p95 values
suite duration
```

and carries its own canonical `resource_benchmark_suite_sha256`.

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
