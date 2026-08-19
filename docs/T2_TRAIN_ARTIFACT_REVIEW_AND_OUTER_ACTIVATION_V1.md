# T2 TRAIN-artifact review and outer-VALIDATION activation, V1

This is a **governance and admissibility record**, not a scientific protocol. It
freezes no question, no architecture, no metric, no threshold and no selection
rule. Everything scientific about T2 was frozen before the TRAIN run executed,
in `T2_LONGITUDINAL_TEMPORAL_PROTOCOL_V1.md` and
`T2_CANONICAL_TRAINING_EXECUTION_SPEC_V1.md`, and neither document is touched by
the change set that carries this file.

What this document does is narrow: it records that a human reviewed the
completed canonical TRAIN artifacts, states exactly which artifacts were
reviewed by digest, and binds the outer-VALIDATION activation to those exact
bytes. Its own SHA-256 is bound in the executable governance layer, so the
activation cannot be separated from the review that justified it.

## A. Reviewed source identity

| | |
|---|---|
| Master used for TRAIN | `f4759e2a97d17db26cb6a6b7c0e9b6207eb0b045` |
| T2 protocol SHA-256 | `6546086a55fe2c9c109f4121cdb6b42d4d53ce0112c9611eb895bd8c805cfefb` |
| T2 canonical training execution-spec SHA-256 | `af6ebf1a6314edb86cce7aa88a6260dd1bd155fd0aebe472d3745b6c823b8054` |

Both frozen documents are byte-identical to those digests at the commit that
carries this record. The activation change set does not modify either one.

## B. Exact canonical TRAIN artifact identities

Experiment `T2_temporal_v1`, attempt `t2-v1-training`, run root
`cardiosentinel-runs/phase8-t2-development-v1`.

| Artifact | SHA-256 |
|---|---|
| Top-level result `T2_TRAINING_RESULT.json` | `ff9258f95631405b6705811d638d754400a067be4c1a43bb9d52021bb246adb8` |
| Experiment lock, file bytes | `37e633a38b7162c53e733b973f79395336623beb1e45c46411b398aafce9cfc8` |
| Experiment lock, self-digest | `d8de03554931fe65a6f1c1242d80c1c95f1a6a26f93b8013cff5bc221a92202f` |
| GRU checkpoint | `027048c5b3fedb13d1c695f2550b352ff81d447fc2f4dc4bbbb617dd420fa82b` |
| GRU checkpoint lock, file bytes | `61c5091125060c90ff52b51a3c8c3f0673688845787f2449578fe2057d1274ad` |
| GRU checkpoint lock, self-digest | `fab35e12016b8a2d10dd3ba29eca4d9c2df05af83fd40e85d653f996183bd9a5` |
| S4D checkpoint | `63ccfbe00c209f94124610f1a22b25d84a2ad2b7e941ecaa3f0c8e9684a6722e` |
| S4D checkpoint lock, file bytes | `a9807515736abfeb9bcc34a3d98a8bdc766b1bae73a53eb3c2a8acc38259f8c7` |
| S4D checkpoint lock, self-digest | `a51ad25ed6cbef266b282954097e623b9666f82e0b25e84f7ced175cf46f5139` |

The lock **file** digest and the lock **self**-digest are different quantities
and both are recorded deliberately. The self-digest is computed over the lock's
own content; the file digest covers the persisted bytes. Binding only one would
leave the other free to drift.

Frozen internal-dev thresholds, derived from each retained best checkpoint by
the frozen rule (exact maximum F1, highest-threshold tie-break, internal-dev
PRIMARY rows only):

| Arm | Threshold |
|---|---|
| `causal_gru_longitudinal_v1` | `0.8328019380569458` |
| `causal_s4d_longitudinal_v1` | `0.8972153067588806` |

## C. TRAIN result status

- Both frozen candidates were successfully trained and frozen.
- Canonical verifier `validate_canonical_t2_attempt`: **PASS**
  (`verified`, `checkpoint_locks_verified`, `git_identity_verified` all true).
- No automatic retry was performed (`automatic_retry_performed: false`,
  `repeat_attempt_permitted: false`).
- No recovery attempt of any kind exists; there is exactly one T2 attempt
  directory and it is `t2-v1-training`.
- Outer VALIDATION: **unopened** (`outer_validation_accessed: false`,
  `validation_accessed: false`).
- TEST: **unopened** (`test_accessed: false`, `sealed_test_state: "unopened"`).
- **No arm selected** (`arm_selected: null`,
  `arm_selection_status: "pending_one_shot_outer_validation"`,
  `arm_compared_on_train_evidence: false`).

## D. TRAIN-development evidence

Recorded descriptively, for review completeness only.

| Arm | Best epoch | Internal-dev pooled AUPRC |
|---|---|---|
| `causal_gru_longitudinal_v1` | 1 | `0.6285039007027243` |
| `causal_s4d_longitudinal_v1` | 10 | `0.6402892809361228` |

> **THESE VALUES ARE TRAIN-DEVELOPMENT EVIDENCE ONLY.**
>
> **THEY DO NOT SELECT THE T2 ARM.**

The internal-dev partition is 8 TRAIN subjects held out of fitting inside the
TRAIN partition. It is checkpoint and threshold evidence, nothing else. It is
not outer evidence, it is not generalization evidence, and the numeric ordering
of the two AUPRC values above carries no selection authority whatsoever. The
frozen selection rule reads outer VALIDATION alone, and it is not permitted to
be anticipated here.

The GRU arm early-stopped after 4 epochs with its best checkpoint at epoch 1;
the S4D arm ran its full 10-epoch budget with its best checkpoint at epoch 10.
Both facts are recorded because they describe what the retained checkpoints are,
not because either is an argument for either arm.

## E. Temporal-cadence forensic audit

A read-only forensic audit resolved an apparent inconsistency in the persisted
epoch histories. Every epoch of both arms records `frontier_count: 135`, while a
nominal 24-hour stream at the frozen 5-second stride yields roughly 68
frontiers. The audit performed no scientific execution.

Findings:

- Persisted frontier count is genuinely **135** in every epoch of both arms; it
  is not a reporting artifact.
- FIT stream count: **116**.
- Maximum FIT stream: record `s20611`, channel 0 and channel 1.
- Each of those two streams holds **34,439** windows.
- `start_sample` cadence is exactly **1,250 samples** everywhere in FIT.
- Non-1,250 FIT deltas: **0**, over 1,938,899 consecutive differences. No
  duplicate start samples and no non-increasing transitions anywhere.
- `s20611` header: **43,050,000 samples at 250 Hz**, and the header's own
  comment records "This is a 48-hour record."
- Duration: **172,200 s ≈ 47.83 h**.
- The final window ends exactly at sample **43,050,000**, matching the header
  sample count exactly.
- `ceil(34,439 / 256) = 135`, reconciling the persisted frontier count exactly.
- Both channels of `s20611` agree in length, range and delta distribution, so
  the property is record-wide rather than channel-specific.
- Stable IDs agree across all 68,878 rows of the record: window length is
  exactly 2,500 samples and record, channel and start agree with the row.
- No duplicated timeline.
- No cadence defect.
- No scientific execution occurred during the audit; no model was constructed,
  no score recomputed, no threshold rederived, and `representation.npy` was
  never opened.
- VALIDATION remained unopened throughout the audit.
- TEST remained unopened throughout the audit.

The frontier count is set by the longest FIT stream because the schedule is
synchronized across streams, so one documented 48-hour record legitimately
governs the frontier budget for the whole FIT partition.

**Decision:**

**T2 TRAIN TEMPORAL-CADENCE AUDIT PASS.**

**CANONICAL T2 TRAIN ARTIFACTS ARE ADMISSIBLE FOR THE FROZEN ONE-SHOT
OUTER-VALIDATION EVALUATION.**

This review is an admissibility finding about artifact integrity and provenance.
**It is not independent generalization evidence**, it is not a performance
claim, and it does not anticipate any outer result. The one-shot outer
VALIDATION remains the sole arm-selection evidence, and at the time this record
is written it has not been executed.

## F. What activation does and does not do

The change set carrying this document flips exactly one source-controlled
constant, `T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED`, from `False` to `True`,
and binds the reviewed TRAIN identities above into the outer preflight. There is
no environment variable, no setter, no CLI activation flag and no alternate
activation path.

Activation does not execute anything. The one-shot outer VALIDATION still
requires a human to invoke the canonical route against a merged commit, and the
outer preflight now additionally refuses unless the verified TRAIN attempt is
byte-for-byte the one reviewed here: the same result digest, the same experiment
lock self-digest, the same authorized TRAIN commit, still with no arm selected
and with TEST still sealed. If any of those differ, the route refuses **before**
the outer claim and before any VALIDATION artifact is opened.
