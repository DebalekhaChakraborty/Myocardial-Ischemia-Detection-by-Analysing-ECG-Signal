# T2 Canonical Training Execution Specification V1

## 0. Nature of this document

**THIS FREEZES NON-SCIENTIFIC TRAINING MECHANICS, PROSPECTIVELY, BEFORE ANY REAL
TRAINING.**

The science is already frozen in
`docs/T2_LONGITUDINAL_TEMPORAL_PROTOCOL_V1.md`, SHA-256
`6546086a55fe2c9c109f4121cdb6b42d4d53ce0112c9611eb895bd8c805cfefb`. **Nothing
here may change it.** No representation, width, layer count, state dimension,
activation, normalisation, discretisation, optimiser, learning rate, TBPTT
length, model family, selection rule, threshold rule, input signal, loss family
or sampling scheme is decided in this document.

What is decided here is only the engineering required to make the frozen
protocol executable: how independent streams are batched, how the direct loss is
reduced, how seeds and devices are handled, what an epoch means operationally,
and what the canonical claim and artifacts look like.

**This specification is created before execution and is never result-derived.**
No real TRAIN optimisation, no real internal-dev scoring, no real outer
VALIDATION scoring and no TEST access occurs in the change set that introduces
it.

## 1. Synchronized stream-batch TBPTT

Every stream is preserved independently. The stream key is
`(record_id, channel_index)`, ordered by the persisted `start_sample`.

There is **no mini-batch-size hyperparameter**. The active streams at each
synchronized frontier *are* the batch:

- every eligible fitting stream owns its own recurrent state;
- at TBPTT frontier `k`, each still-active stream contributes its local rows
  `[k·256 : (k+1)·256]`;
- those independent chunks are batched together along the batch dimension;
- shorter final chunks may be padded **for tensor shape only**;
- padded positions carry no score evidence, no loss, no metric and no state
  meaning once the stream has ended;
- every real stream remains chronologically intact;
- after a chunk, the optimiser steps **once** if that chunk contained at least
  one PRIMARY direct-loss row;
- carried state is **detached** before the next chunk;
- state is **not** reset at chunk boundaries;
- state resets only at real stream boundaries.

The batch dimension therefore contains independent streams. There is no
cross-stream state and no cross-stream feature interaction.

## 2. Loss reduction

For a synchronized training chunk, binary cross entropy with logits is computed
**only** over `AVAILABLE + PRIMARY` rows, using the frozen
`pos_weight = N_negative / N_positive` derived from the 48-subject FIT partition
only.

The exact reduction is:

1. weighted BCE with `reduction="sum"` over the direct-loss rows;
2. divided by the number of direct-loss rows in that optimiser step.

Every eligible PRIMARY window therefore contributes equally to the pooled
training objective, regardless of which stream or chunk it landed in.

Challenge rows, other non-primary rows and unavailable rows receive **no direct
loss**.

If a chunk contains **zero** direct-loss rows: causal context and state are
processed correctly, **no optimiser step is performed**, state is detached at
the normal TBPTT boundary, and the zero-loss-step event is recorded.

## 3. Deterministic initialisation

Before constructing **each** candidate independently, all relevant generators
are reset to the frozen seed **2026** — at minimum `random`, `numpy` and
`torch`. Each arm is built from a fresh identical seed origin, so construction
order cannot alter its initialisation. Constructing the GRU, consuming RNG, then
constructing the S4D would make the S4D's initialisation depend on the other arm
existing; that is forbidden.

The initialisation seed is persisted per arm.

Deterministic execution controls are enabled where the installed PyTorch
supports them. **Nothing is installed.** If the runtime cannot satisfy a
required deterministic operation, execution **STOPS FOR HUMAN REVIEW**;
determinism is never silently disabled.

## 4. Device policy

There is **no user-selectable scientific device flag**. The canonical runtime
uses the host-supported torch device, and persists: device type; device model or
name where available; torch version; CUDA version where applicable;
deterministic-algorithm state; thread configuration; and relevant backend
settings.

**Both arms must run on the same device and runtime environment.** If the device
or runtime changes between arms, execution **STOPS**. No mixed CPU/GPU
scientific comparison is admissible.

## 5. Epoch semantics

At most **10 completed epochs**. For each epoch:

1. every FIT stream starts at zero temporal state;
2. one full chronological pass runs through every FIT stream;
3. synchronized TBPTT frontiers of 256 timeline windows are used;
4. state carries across frontiers;
5. carried state is detached at each frontier;
6. optimiser state persists across chunks **and** epochs;
7. at the next epoch, temporal state resets to zero at every stream start.

Temporal state is **not** preserved from the end of epoch *N* into epoch *N+1*.
A new epoch is another pass over the recordings, not a physical continuation of
the patient's timeline.

## 6. Internal-dev evaluation

After every completed training epoch the model switches to `eval` mode and runs
**one** causal full-timeline pass over all internal-dev streams: zero state at
each real stream start, the complete timeline, no direct loss, no optimiser, no
gradient. Challenge and other non-primary rows remain causal context.

Only the **PRIMARY mask** feeds the checkpoint metric: pooled PRIMARY AUPRC.

## 7. Early stopping and checkpoint selection

`best_epoch` starts undefined. An epoch is an improvement **iff**

```
internal_dev_pooled_auprc > best_internal_dev_pooled_auprc
```

Exact equality is **not** an improvement, and the earlier epoch remains best.

The patience counter is the number of consecutive completed epochs after the
current best that fail to improve. Training stops when
`patience_counter >= 3`, or after 10 completed epochs.

Every completed epoch result is persisted. A non-finite AUPRC is a **hard
failure**: no retry, no substitution.

Exactly one TRAIN-development checkpoint is retained per arm — the one from the
best internal-dev pooled AUPRC. A later epoch is **never** retained because its
training loss looked better.

## 8. Internal-dev threshold

After the best checkpoint is frozen, exactly one causal internal-dev scoring
pass is run for that checkpoint. On internal-dev **PRIMARY rows only**, the
exact maximum-F1 threshold is derived with the **highest-threshold tie-break**,
matching repository convention.

Persisted: threshold; TP; FP; FN; TN; F1; sensitivity; specificity; PPV; NPV;
balanced accuracy; MCC; internal-dev pooled AUPRC; internal-dev pooled AUROC.

These are **TRAIN-development evidence**, not outer-VALIDATION evidence. The
threshold is not modified after this point.

## 9. No arm selection during training

GRU and S4D are **not** compared on TRAIN or internal-dev evidence. The
protocol's arm selection consumes outer-VALIDATION pooled AUPRC, then
subject-macro AUPRC, then parameter count.

After canonical TRAIN-only execution both arms remain candidates, and every
artifact records:

```
arm_selection_status = "pending_one_shot_outer_validation"
```

No winner is declared.

## 10. Outer-VALIDATION activation state

The outer-VALIDATION evaluator is implemented so it can be reviewed **before**
scientific exposure, but its execution is **structurally disabled**:

```
T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED = False
```

The canonical evaluator entry point refuses **before** opening VALIDATION
representation arrays, reading VALIDATION labels or constructing VALIDATION
timeline evidence. There is no environment variable, no hidden flag and no
manual override. A separate future activation change set will authorise the
one-shot outer VALIDATION after the TRAIN-only artifacts are human-reviewed.

## 11. Canonical claim

| | |
|---|---|
| Experiment identity | `T2_temporal_v1` |
| Attempt / claim id | `t2-v1-training` |
| Run root | `cardiosentinel-runs/phase8-t2-development-v1/` |

No timestamp, no UUID, no random suffix, no automatic `recovery1`, no retry
name. **The claim directory is consumed once created.** A future authorised
attempt that fails after the claim preserves its failure evidence and STOPS; no
automatic retry occurs.

## 12. Memory safety

M1 already demonstrated that careless handling of the full corpus can exhaust
host memory. Therefore: large representation arrays are memory-mapped read-only;
the complete 2 208 431 × 146 representation is never duplicated or converted
into one torch tensor; only active TBPTT chunks are materialised; temporary
graphs are released after optimiser steps; only detached states are carried; and
compact arrays or views are used instead of per-window Python objects for the
full TRAIN population.

## 13. Runtime integrity

The existing runtime-integrity sentinel is reused unchanged. Canonical
interpreter `/home/AI_POC/venvs/tactics/bin/python`; expected dependency digest
`b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`.

Enforcement points: **START**, **pre-model-construction**,
**pre-checkpoint-promotion**, **COMPLETION**. No automatic repair, no install,
no upgrade, no downgrade, no alternate interpreter.

## 14. TEST firewall

TEST is absolutely sealed. The implementation structurally refuses
`partition == "test"`, TEST subject identities, TEST representation paths, TEST
labels, TEST checkpoint evaluation and TEST metric calculation — before path
resolution wherever practicable. No TEST option exists in the canonical CLI, and
every T2 artifact records `test_accessed: false` and
`sealed_test_state: "unopened"`.
