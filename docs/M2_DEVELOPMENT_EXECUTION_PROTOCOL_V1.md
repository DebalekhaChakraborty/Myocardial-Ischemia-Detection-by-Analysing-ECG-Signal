# M2-v1 development execution protocol

> **STATUS: FROZEN IMPLEMENTATION/EXECUTION SEMANTICS.**
> This document freezes *how* the canonical M2 development experiment is
> executed and what its evidence must bind. It freezes **no scientific rule**.
>
> **No canonical M2 development scientific execution has occurred.** This
> document describes the activated route; running it requires its own human
> authorization.

## 0. Scientific rules are inherited unchanged

Every M2 scientific choice remains exactly as frozen in
`docs/M2_CONTAMINATION_SAFE_MEMORY_PROTOCOL_V1.md` and the M2 gate derivation
receipt. Nothing in this protocol, and nothing in the execution route it
describes, changes:

M2-0; M2-G; G1–G6; the G3 bounds; the `finite_sample_fraction` rule;
`NORMAL_EVIDENCE_THRESHOLD = 0.0002997174742631614`;
`M1L_CLASSIFICATION_THRESHOLD = 0.7554003000259399`; the 60-second re-armable
refractory; score-before-update; post-decision re-arming; the retained
`M1L_long_memory_v2` memory; the scorer weights; the scorer feature vector; the
physiology representation; the memory alpha; the cold-start bins; the challenge
family definitions; the prototype-drift formula; or the rollback exclusion.

No retraining occurs. No threshold is selected. No arm is chosen automatically.

The stress-family eligibility rule is frozen separately in
`docs/M2_STRESS_INTERVAL_ELIGIBILITY_DECISION_V1.md`
(SHA-256 `078acb3d72a11513010c88a03b0143a2be43da5da807c72d3d7433f98031f8f6`),
which this route binds into every claim-bearing artifact.

## 1. Four populations, and they are not the same thing

The frozen M1/P1 route separates *what the memory saw* from *what a metric was
computed over*. M2 reproduces that separation exactly. **Full replay ≠ primary
metric ≠ challenge metric ≠ stress-interval selection.**

| Population | Authority | Carries |
|---|---|---|
| **Full replay** | the verified full VALIDATION M1 stream cache | causal memory evolution, the M2-0/M2-G update policy, policy evidence, the prototype trajectory |
| **Primary metric** | the frozen P1 VALIDATION population/cache | AUPRC, AUROC, sensitivity, specificity, PPV, MCC, subject-macro, background FPR, subject FPR distribution, cold start |
| **Challenge metric** | `build_validation_challenge_index(...)` and its frozen selection digest | rate-related FPR, axis-shift FPR, conduction-change FP/N |
| **Stress interval** | source-defined LTSTDB `.stb` intervals only | prototype contamination/drift |

**Full replay** keeps every frozen timeline row — including challenge, quality,
boundary and censored rows — because dropping any of them would corrupt causal
history. It is never a metric denominator.

**Primary metric** membership is fixed upstream by P1/M1 and is never derived
from an M2 score. Challenge-only, quality-only, boundary and censored rows never
enter the primary classification denominator. Its frozen identity is
473,897 total / 21,628 positive / 452,269 negative / 12 subjects.

**Challenge metric** membership is the frozen validation challenge selection:
4,973 rate-related + 3,000 axis-shift + 164 conduction-change = 8,137 windows,
selection digest `49899d1b…`. No binary primary label is invented for a
challenge-only row; the challenge annotation structure has nowhere to put one.

Primary and challenge are proven **disjoint**, and both are proven **exact
subsets of the full replay population by stable identity**, before any metric is
computed.

## 2. Which evidence uses which population

| Evidence | Population |
|---|---|
| Policy / update-admission evidence | FULL REPLAY |
| Window discrimination | PRIMARY |
| Background and subject-wise FPR | PRIMARY |
| Cold-start strata | PRIMARY |
| Challenge FPR | CHALLENGE |
| Prototype contamination / drift | FULL REPLAY trajectory + STRESS INTERVAL selection |

`false_alarm_evidence` remains one top-level section, but it is internally
explicit: `background_and_subject_fpr` (PRIMARY) and `challenge_fpr`
(CHALLENGE), each carrying its own validated population identity. The result
never implies that one denominator served both.

## 3. VALIDATION is the development evidence partition; TEST stays sealed

Canonical M2 development evidence is computed on **VALIDATION**, and a
claim-bearing arm result therefore records

```
partition_accessed  = "validation"
validation_accessed = true
test_accessed       = false
sealed_test_state   = "unopened"
```

The canonical partition is hard-fixed by
`require_canonical_development_partition`, which is deliberately a *separate*
firewall from the permissive smoke guard: widening that one would have let the
bounded TRAIN engineering smoke become claim-bearing development evidence. TRAIN
remains permitted only for that explicitly non-claim-bearing smoke.

TEST is rejected by both firewalls, before any metadata, stream-cache,
source-path, waveform or annotation access. No B4 sealed-test utility is
imported by the canonical route, and no `TEST_ATTEMPT` is ever created.

## 4. One suite, two independent canonical attempts

> **Two attempts are consumed.** Recovery1
> (`m2-v1-development-two-arm-recovery1`, master `d77fbdc`) also claimed both
> arms and failed before scoring, on a *different* defect: the feature join used
> `isnan(output)` as proof that a row was never written, but NaN is also the
> legitimate representation of an upstream source null, so a valid corpus raised
> a structural error. Exactly one further recovery is permitted:
> `m2-v1-development-two-arm-recovery2`. See
> `docs/M2_DEVELOPMENT_RECOVERY1_FAILURE_AND_RECOVERY2_DECISION_V1.md`
> (SHA-256 `93e53d3c8281d922823d48b73712a2a1ede1c5b0f5bc9f41694af563e1a2fca4`).
> Every recovery2 artifact binds **both** prior lineages.
>
> **Attempt #1 is consumed.** The first canonical execution, suite
> `m2-v1-development-two-arm` at master `3c1ba4c`, claimed both arms and then
> failed during full label-blind replay, before a single row was scored, on a
> TRAIN-vs-VALIDATION feature-join defect. It is permanently preserved and never
> re-run. Exactly one recovery is permitted:
> `m2-v1-development-two-arm-recovery1`. See
> `docs/M2_DEVELOPMENT_ATTEMPT1_FAILURE_AND_RECOVERY_DECISION_V1.md`
> (SHA-256 `e9d55d7a047e9610c6e156afc9e1a98aafbca86a3131c02a8e56624da7ad57d6`),
> whose digest every recovery artifact binds along with the lineage fields
> `recovery_from_suite_id`, `recovery_suite_id`, `recovery_reason_class`,
> `prior_attempt_scoring_started=false`,
> `prior_attempt_metrics_computed=false` and
> `prior_attempt_test_accessed=false`.

The production suite identity is **immutable**:
`CANONICAL_SUITE_ID = "m2-v1-development-two-arm-recovery2"`. There is no public
`suite_id` parameter and no CLI option that selects one, because a
caller-chosen suite name would let a second "canonical" suite run under a
different directory after the first was consumed. A non-canonical id is refused
before claim checking, before any filesystem creation and before any VALIDATION
access. A private `_suite_id` seam exists for synthetic tests only, and it still
cannot bypass a consumed attempt: the pair preflight refuses any suite whose
paths already exist.

Each arm gets its **own** immutable claim directory:

```
<run_root>/<suite_id>__M2-0     the M2-0 canonical attempt
<run_root>/<suite_id>__M2-G     the M2-G canonical attempt
<run_root>/<suite_id>          the suite result
<run_root>/<suite_id>__evidence the disk-backed evidence workspace
```

A single shared experiment id would make M2-0 claim the directory and M2-G
collide with it, so the two-arm run could never start. The identities are
deterministic: never random, never timestamped, and never auto-renamed on
collision, because any of those would let a consumed attempt be silently re-run.

The evidence workspace is derived from the suite attempt rather than chosen by
the caller, so a generic root holding a previous attempt's evidence can never be
silently reused. It is created with `exist_ok=False` and is never cleaned,
overwritten or retried.

## 5. Execution order

The canonical runner is `src/cardiosentinel/neural/m2_development_run.py`. It is
the only public route to a claim-bearing M2 development result, it always runs
both arms in the frozen order `M2-0` then `M2-G`, and it never selects between
them. Nothing executes on import, and `__main__` dispatch is the last statement
in the file so module execution can never enter the run with an undefined
runtime helper.

1. **PRE-CLAIM ARTIFACT READINESS.** In this order, and all of it before any
   VALIDATION access: the exact Git SHA on a clean checkout; the frozen
   dependency runtime; the M2 protocol digest; the M2 gate receipt digest; the
   stress-eligibility decision document digest; the retained M1L lock; the
   retained M1L checkpoint; the P1-B lock; the B4-B checkpoint identity; the
   frozen TRAIN-only distance standardizer; the label firewall; the TEST
   firewall; and the pair-claim absence check. The scorer and standardizer are
   readied here because neither requires VALIDATION — and discovering a missing
   or altered checkpoint *after* claiming two canonical attempts would consume
   them for nothing. Any failure means no arm claim, no VALIDATION access and
   no retry.
2. **START / CLAIM.** An independent `RuntimeIntegrityRecord` per arm; a
   successful START recorded for **both** arms; then M2-0 claimed and M2-G
   claimed. **Only after BOTH claims succeed may VALIDATION be opened.** A
   failed claim stops for human review without opening validation.
3. **DEVELOPMENT SOURCE INTEGRITY.** The stress selection later reads raw
   LTSTDB `.stb`, so an arbitrary local directory is never trusted. The
   repository's existing `validate_development_feature_integrity` and
   `validate_development_source_integrity` bind the `.hea`/`.dat`/`.stb` files
   to the official pinned manifest, the frozen per-record source digests and the
   frozen feature-corpus identity, over the train/validation development
   partitions only. TEST files are never hashed. The resulting receipt is bound
   into the arm provenance, the experiment lock, the stress-selection identity
   and the suite provenance.
4. **FULL LABEL-BLIND REPLAY.** The validation input is loaded exactly once and
   the canonical full replay identity is proven. Both arms replay the identical
   frozen rows with the identical frozen scorer, each keeping its own stream
   state. No annotation is loaded until both trajectories are complete, so no
   M2-0 result can alter M2-G's replay.
5. **POST-REPLAY.** Only then: primary membership, challenge membership,
   source-defined stress intervals, identity-keyed joins, frozen evidence.
6. **PERSIST.** Per arm: validate the result payload, stage, COMPLETION check,
   separate PRE_PROMOTION observations for `M2_ARM_RESULT.json` and
   `M2_EXPERIMENT_LOCK.json`, atomic promotion.
7. **SUITE.** One aggregating two-arm suite that expresses no retention
   decision. It computes no new scientific metric and applies no preference: it
   aggregates two already-frozen arm results. The sequence exists so the
   promoted artifact can *prove* its own promotion gate rather than assert it:

   1. create the suite `RuntimeIntegrityRecord` and record START;
   2. build the aggregation body **unsigned**;
   3. verify every declaration against the ACTUAL arm artifacts (below);
   4. take a PRE_PROMOTION observation specifically for
      `M2_SUITE_RESULT.json` — never a reused arm observation;
   5. require every suite observation GREEN;
   6. embed the complete suite runtime block into the payload;
   7. compute `m2_suite_sha256` only **after** that block exists, so the
      signature covers the promotion evidence;
   8. promote atomically and re-validate from the persisted bytes.

   No observation is ever fabricated after hashing. If either arm is not
   COMPLETE there is no canonical suite; if suite promotion fails, both arm
   artifacts are retained for human review and nothing is re-run automatically.

   The suite is the immutable aggregation of two frozen arms, so its validator
   proves the declarations against the files rather than trusting them: for
   each arm the deterministic experiment id, the presence of the result and
   lock, `sha256(M2_ARM_RESULT.json)` equal to the declared
   `arm_result_sha256`, the lock's own `experiment_lock_sha256` equal to the
   declared value, a full `validate_canonical_run_lock` against the real
   directory, the lock's `artifact_sha256[M2_ARM_RESULT.json]` equal to the
   promoted result's digest, the lock's arm and experiment id, the same
   execution Git SHA as the suite, the same four population identities, the
   same development-source identity, `test_accessed=false`,
   `sealed_test_state="unopened"` and no arm selection.

## 6. Execution consent, the Git gate, and the deterministic roots

Execution requires **both** `--execute-canonical-development` and
`--expected-git-sha <HUMAN_REVIEWED_MASTER_SHA>`. HEAD must equal that SHA on a
clean checkout, checked before any data access, so a run against an unreviewed
tree stops without consuming an attempt.

The expected SHA is deliberately **not defaulted or hard-coded**: the scientific
run happens only after the activation PR is merged and the resulting master SHA
is human-verified.

Those two flags are the **only** CLI options. Every root and identity is
deterministic, from the repository's existing conventions:

| Root | Path |
|---|---|
| source | `cardiosentinel-data/ltstdb/1.0.0` |
| feature | `cardiosentinel-features/ltstdb-baseline-v1` |
| stream cache | `cardiosentinel-features/m1-stream-memory-v2` |
| P1 cache | `cardiosentinel-features/p1-b4b-embeddings-v1` |
| M1 run | `cardiosentinel-runs/phase5-m1-dual-memory-v2` |
| M2 run | `cardiosentinel-runs/phase6-m2-development-v1` |

The P1 embedding cache and the M1 stream-memory cache are **different
artifacts**. The primary metric population lives only in the former, and there
is no fallback from one to the other anywhere on the route.

There is no partition option, no arm option, no threshold option, no retry
option, no seed option and no alternative data-source option. A private
`_roots`/`_loaders` dependency-injection seam exists for synthetic tests only;
it is absent from the CLI and from the public scientific contract.

## 6a. Partition-aware feature assembly

Scientific timeline assembly joins COMBINED_V1 columns through
`m2_feature_join.join_sqi_and_morphology_for_partition`, which takes the
partition explicitly and names it in every refusal. The frozen M2 **TRAIN**
gate derivation keeps its own TRAIN-only helper unchanged, so the frozen
receipt continues to mean exactly what it meant.

The join requires exact record-set equality between the COMBINED_V1 corpus and
the M1 stream-cache manifest **for the same partition**, refuses duplicate
record ids and escaping cache paths, and treats a missing, unmatched or extra
row as fatal rather than inner-joining it away. TEST is refused before the
feature manifest is opened.

Per record it additionally requires **exact stable-ID set equality in both
directions**: the NPZ's stable-ID count matches its feature-row count, neither
the NPZ nor the stream cache repeats an identity, every stream row has a
feature row, and every feature row has a stream row. Requiring only the first
direction would silently accept a feature cache holding extra rows — a corpus
that is not the one the stream cache was built from, quietly reduced to a
subset at join time. Order is not asserted, because the join realigns by stable
identity and the frozen contract fixes the *stream* chronology.

## 6a-bis. Attempt #1 is proven from artifacts, never inferred

`consumed_failed_pre_scoring` is a scientific claim — that *this* attempt
failed, that it failed before any row was scored, that no metric was produced
and that the sealed test stayed shut. A claim directory proves none of that, so
`validate_original_attempt1_failure_lineage()` verifies the preserved artifacts
against frozen digests before a recovery may be claimed:

* both original arm directories and their `M2_RUN_STATUS.json` files exist and
  hash to `3699e656…6365d` (M2-0) and `7908130758…251` (M2-G);
* neither original arm holds `M2_ARM_RESULT.json` or `M2_EXPERIMENT_LOCK.json`,
  and the original suite holds no `M2_SUITE_RESULT.json`;
* the additive receipt exists at the frozen failure-review path, its file
  digest is `8c3a0734…b278`, its canonical `receipt_sha256` recomputes and
  equals `31345512…92eb`;
* the receipt names the original suite and execution SHA, is
  `claim_bearing=false` / `canonical=false`, records the frozen pre-scoring
  replay stage and the partition-alignment exception, and records
  `validation_opened=true`, `scoring_started=false`, `metrics_computed=false`,
  `test_accessed=false`, `sealed_test_state="unopened"`;
* it binds the frozen preserved status digests and the recovery decision digest.

Any absent or mutated artifact **stops for human review**. Verification is
read-only: nothing is repaired, replaced, normalised or inferred.

## 6a-ter. Structural missingness is not a source null

The join tracks **structural assignment** separately from feature **values**. An
explicit `written` mask records every row a block assigns, and
`require_all_rows_written()` refuses any row the join never wrote, naming its
stream position and stable id.

A legitimate source null survives the join **unchanged** as NaN. It is never
replaced by zero, a TRAIN median, a bound or an infinity; its row is never
dropped; its observation is never marked physically unavailable; no SQI
threshold is created; and neither SIGNAL_V1 nor COMBINED_V1 is regenerated.

The join's responsibility is identity alignment. The **existing frozen policy**
owns what such a value means, and is unchanged:

* `UNAVAILABLE_EXACT_FLAT` → `G1 = false`, G2–G6 **not applicable**; a
  physically unavailable row is *not* counted as a G3 refusal.
* `AVAILABLE` with a non-finite G3 feature → that feature fails the frozen
  `np.isfinite(value) and value <= bound` rule, so `G3 = false` and the memory
  update is refused. No imputation.
* **M2-0** does not operate G3–G6, so an M2-G-only null changes nothing for the
  naive control; an unavailable row remains unavailable to it too.

## 6b. Post-claim failure accounting

Once any arm claim exists, an uncaught canonical-run exception writes
deterministic non-claim-bearing accounting: one additive
`M2_ATTEMPT_FAILURE_RECEIPT.json` in `<suite_id>__failure_review/`, plus the
established `FAILED_OR_INTERRUPTED` status on each existing claim. The receipt
records the exact stage and exception, whether VALIDATION had been opened, the
promotion state, the runtime observations so far, and `canonical=false`.

Nothing is deleted, cleaned, renamed, reseeded or retried; staged and evidence
files are preserved exactly as the failure left them; and a partially failed
attempt is never made to look COMPLETE. A failure *before* any claim writes
nothing, because no attempt was consumed.

**The receipt reports REAL exposure.** Attempt #1's `scoring_started=false` is
a frozen determination about that attempt, not a template: a future failure
after the scorer has been invoked must say so. A transparent wrapper flags
scoring the moment the frozen scorer is called and returns its output
unchanged; `post_replay_evaluation_started` and per-arm `metrics_completed` are
set at the corresponding points. Where an abrupt exception leaves a fact
genuinely unknowable it is recorded as `indeterminate` rather than as a
flattering `false`.

**Promotion accounting is per arm.** `arm_result_promoted` and
`experiment_lock_promoted` are maps keyed by arm, so M2-0 completing while
M2-G fails is preserved exactly and one arm's promotion never implies the
other's.

Historical claim files from a previous attempt are never rewritten to make a
failed state look cleaner: the forensic classification lives beside the claim,
in its own additive receipt.

## 7. Bounded memory

The M1 host-memory incident established that corpus-scale Python row retention
is unsafe. The canonical route therefore replays **stream by stream**, one
`(record_id, channel_index)` at a time, and folds each stream's bounded batch of
`M2RowEvidence` into compact typed arrays and integer counters before releasing
it. No whole-corpus row list, no whole-corpus duplicate representation matrix
and no two-arm whole-corpus object duplication exists at any point.

The full causal prototype trajectory is written to a disk-backed evidence store
per stream, and **drift evaluation loads one stream at a time**: one trajectory
is read, only that stream's frozen stress intervals are evaluated, the compact
drift entries are appended and the trajectory is released before the next stream
is read. The process never holds the prototype matrices of several streams at
once. The streaming and whole-dictionary forms share one implementation, so
`sqrt(mean((mu_long(t) - mu_ref) ** 2))` and the follow-up semantics are
identical and no approximation is introduced.

The store's schema and content digest are bound in canonical provenance, and the
manifest is re-validated against the actual persisted files after finalization,
before it may enter a claim-bearing result.

Precision is preserved end to end: scores, availability times and prototypes are
stored and read back as float64, so `sqrt(mean((mu_long(t) - mu_ref) ** 2))` is
reproduced exactly. No downcast, quantization or lossy conversion occurs.

Prototype persistence is label-blind: the whole trajectory is written first, and
stress annotations select points from it only afterwards. Annotations never
decide which prototypes are kept.

## 8. Development-source identity coherence

The receipt proving the raw `.hea`/`.dat`/`.stb` were the official frozen
source is one identity, carried identically by every claim-bearing artifact:

```
arm result   development_source_identity
  == stress_interval_selection_identity.development_source_identity
  == arm lock development_source_identity      (explicit top-level provenance)
  == suite     development_source_identity
```

It is validated as `identity_class = "m2_v1_development_source_integrity"`,
`annotation_set = "stb"`, `test_partition_hashed = false`,
`verified_before_stress_selection = true`, and both the feature and source
receipts reporting `verification_result = "passed"`. The frozen split,
feature-corpus and official-manifest identities remain bound inside those
receipts; no second source-identity algorithm is invented.

## 9. The result and lock contract

A canonical arm result binds four separate identities — the single
`evaluated_population_identity` that once stood for all of them is gone, because
it let the full causal replay population masquerade as a metric denominator:

```
replay_population_identity
primary_evaluation_population_identity
challenge_evaluation_population_identity
stress_interval_selection_identity
```

Every headline section must declare the population it was computed over, and
that declaration must equal the arm result's own identity for that population. A
section that borrows another's denominator, or declares none, is fatal. The
experiment lock binds all four and must agree with the result exactly.

## 10. Execution history is read, never asserted

No source constant records whether a canonical run has happened: source code
cannot rewrite itself, so such a boolean could only ever become a lie.
`canonical_execution_history()` reports run history from the canonical claim
directories, the run-status files, the experiment locks and the suite result.

## 11. What this protocol does not do

It defines no metric, computes no value, selects no threshold and expresses no
retention or rollback decision. It does not modify the frozen M2 gate protocol,
and the frozen protocol was not adjusted to fit the runner.

**No M2 development scientific execution has occurred. No VALIDATION data has
been read. The B4 sealed test remains unopened.**
