# T1 Execution Recovery Amendment V1.1

## 0. Nature of this document

**THIS IS A HUMAN GOVERNANCE AMENDMENT, NOT A NEW SCIENTIFIC EXPERIMENT AND NOT
A RETRY.**

**Version note.** V1 of this amendment was drafted, reviewed and approved in
principle, but was corrected before merge and therefore never froze. **No V1
artifact exists in this repository and no V1 digest was ever pinned.** V1.1 is
the first and only frozen version. It differs from the reviewed draft in exactly
two respects, both narrowing: §9.1 records that the approved continuation
exercises less than §9 permits, and §13.6–§13.7 make that narrowing a
mechanically proven authorization gate carried by the evidence itself, rather
than an implementation convention. Nothing was widened, and no clause of either
governing document is amended beyond the three named in §4.

It records a human decision to permit **one measurement continuation** of the
consumed canonical T1 attempt `t1-v1-development` (scientific identity
`T1_state_machine_v1`), which failed after stage 24 of 29 and can neither be
completed nor repeated under the specification as frozen.

This document amends **two clauses of
`T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1`** and **one clause of
`T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1`**, named exactly in §4. It amends nothing
else. Every other clause of both documents remains in force verbatim.

No metric in this document was computed. Every value is read from immutable
promoted artifacts of the consumed attempt, or from the failure log.

This document is not a fold retry, not a re-selection, not a threshold
re-derivation, not a new candidate set, not a second canonical attempt, not an
upstream rerun, and not a TEST decision. **It does not modify the consumed
attempt.**

## 1. Bound identities

### 1.1 Consumed canonical attempt

| | |
|---|---|
| Experiment identity | `T1_state_machine_v1` |
| Attempt id | `t1-v1-development` |
| Run root | `cardiosentinel-runs/phase9-t1-development-v1` |
| Authorized commit | `c538181eb93884f4583a8bd328e50573efbcf3df` |
| Claimed | `2026-08-21T19:47:24Z` |
| Failed | `2026-08-21T19:57:57Z` |
| Terminal stage | `promote_oof_result` (24 of 29) |
| Working tree at claim | clean, `git_dirty: false` |
| Partition accessed | `validation` |
| Sealed TEST | **unopened**, `test_accessed: false` |

### 1.2 Governing documents

| Document | SHA-256 |
|---|---|
| `T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1.md` | `ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4` |
| `T1_CANONICAL_DEVELOPMENT_EXECUTION_SPEC_V1.md` | `11b6a9aff2f1d928a9f33516db2ea764cf0553a949cd79c14562bafe34f090bf` |
| `RUNTIME_INTEGRITY_SENTINEL_V1.md` | `cd5c2e6d0b5dbc4ea35b319f98e9b9e678256c391491839d3f1745247eeb4075` |

### 1.3 Preserved artifacts of the consumed attempt

File digests. Where a manifest also carries a canonical payload self-digest, both
are given; **they differ by design.**

| Artifact | SHA-256 (file) |
|---|---|
| `T1_PREFLIGHT.json` | `917b5421c9c7731eb185821ed279564c65fed5737153316cfa410811ea4f25da` |
| `T1_RUN_STATUS.json` | `f305da7ad3d465c4500124fe4d4422dfc471580a01afe7b9d424e866e9e2c59d` |
| `T1_INPUT_LINEAGE.json` | `e307bdd3ad244f6440ad437f66d5f7b4e2af3072b6b1833e74552095ede3c555` |
| `T1_INPUT_EVIDENCE.json` | `bf36ac0e538b0cee61a97109de413c52ec942356d974930e5de64bc32b86423b` |
| **`t1_input_evidence.npz`** | **`4391b4e7cda5ac5d70c93663563cc37954afdfc7b28092ef65c2d351006c2f5c`** |
| `T1_FOLD_SELECTIONS.json` | `71e0da62ad2a86fd6bb2561137e0a152df2d5b894bd9fecfb67ad762a5682f6d` |
| `T1_OOF_STATE_EVIDENCE.json` | `aefc922a5224b7c857b9bf99b12441e55e46fdc71def373c043ffb112e5e2405` |
| **`t1_oof_state_evidence.npz`** | **`72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232`** |

| Canonical payload (content) self-digest | SHA-256 |
|---|---|
| Label-blind input evidence store | `57d434d9b4eee9fa3d37f581397d89aca6a5bbd3188aa35f907801766be6a8ac` |
| OOF state-evidence store | `cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f` |
| Fold-selection binding carried by the OOF manifest | `32bab16ca6ec4d8ab7d3b6f2d9a3c8782ae97f3e58a84eb900357df1d881451d` |

### 1.4 The twelve promoted fold selections

Each was promoted, re-read and digest-verified **before** its held-out labels
were authorized, per specification §16. Each attests `held_out_labels_opened:
false` at selection time and `test_accessed: false`.

| Fold | Held-out subject | Selected policy | Selection SHA-256 |
|---|---|---|---|
| 00 | `ltstdb:s2004` | `qw0.9_qe0.99_FAST` | `02ffccd4eb546a7d07017f7234aec9f3c3f189819f4f90ca5663e1d4cf11467c` |
| 01 | `ltstdb:s2005` | `qw0.9_qe0.99_BALANCED` | `f08799a205200c7b1d22a26f1d8354848149828c4eb6e68beb51c6eebde5a786` |
| 02 | `ltstdb:s2019` | `qw0.9_qe0.99_FAST` | `e5d4967a45eb891a11294d640ab6a5e5de77cffdb60cf0c1338be5ff8e3558a1` |
| 03 | `ltstdb:s2020` | `qw0.9_qe0.99_FAST` | `daa0e1def15d45cc826516b8478369c92755ec77634429014580161ed7d6d7ed` |
| 04 | `ltstdb:s2023` | `qw0.9_qe0.99_FAST` | `fa3cec3519513d7681100bb701f38d988af3bacb504cb9dc4702bd6432559dc0` |
| 05 | `ltstdb:s2031` | `qw0.9_qe0.99_FAST` | `6c07098b90548fe03eddf8437ac56bedccd3b1a39abdaa90aa077694b2fb0d0f` |
| 06 | `ltstdb:s2057` | `qw0.9_qe0.99_FAST` | `c9c1b0fb345693ff07f95073d92afb3cacaa72cfb05e290c36dd07ee2c5a6c9a` |
| 07 | `ltstdb:s2058` | `qw0.9_qe0.99_FAST` | `9a6ee4e4e33372e5208234c38c7830015ddccb23ee166140a1a78c83eb68a72d` |
| 08 | `ltstdb:s2059` | `qw0.9_qe0.99_FAST` | `602c9d1f09a3af4f46b234e483592ccb3eb56a9f78251f95f030ce150630a07e` |
| 09 | `ltstdb:s3068` | `qw0.9_qe0.99_FAST` | `31bca60e10377c7bfc77f9fb6a9c54340b6b91e9c048b006f8e198781df99961` |
| 10 | `ltstdb:s3072` | `qw0.9_qe0.99_FAST` | `696e99527caf33c9798721dffd92d12a5d98ef98720579f006c9a96aae4c26a8` |
| 11 | `ltstdb:s3073` | `qw0.9_qe0.99_FAST` | `3384a1261c7e069d8276eb5fe35a66dd7589c953fd37d3ae902ab0e496e03050` |

### 1.5 Upstream chain, unchanged

| Arm | Identity | Retention decision SHA-256 |
|---|---|---|
| B4-B global encoder | `B4B_cnn_transformer_v1` | `1300e7ad641df9137e1722771e5d3932cae0fc4d244047b7c8a5070f151f74bb` |
| P1-B physiology | retained | `7b403709fa0fb12eef65423d830c121fc3ada904266a1b47931d438f5e797d68` |
| M1L dual memory | retained | `a3685fc0f8ff1fa0dce2bf9954bb28a925787070c021f3e80ca5716a4fa5f0ed` |
| M2 update policy | `M2-G`, suite `m2-v1-development-two-arm-recovery2` | `da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47` |
| U1 calibration | `U1_selective_v1`, attempt `u1-v1-development` | `9d8436f2b7d2c303aeeb03e438c60fb8110f7d06d0bbd589f5be65ea8f80cb7b` |
| T2 longitudinal | `causal_s4d_longitudinal_v1` | `4846921135b0ac83ceb40a0db063c2e4a3b2520971f279abe4f0c517c4f7dd20` |

No upstream arm is reopened, re-selected, re-calibrated or rerun by this
amendment. `m2_rerun_permitted: false`, `u1_rerun_permitted: false`,
`t2_rerun_permitted: false` all remain true statements.

## 2. The human decision

**Option B — measurement continuation under amendment — is authorized.**

The continuation is **not** classified as a fold retry. The stated grounds:

1. All twelve fold selections were completed before any held-out evaluation.
2. All twelve selections were promoted, re-read and digest-verified.
3. **No held-out result was observed by any process, artifact, log or human
   before the failure.**
4. No candidate policy, threshold, persistence rule, transition rule or fold
   assignment will change.
5. The continuation exists only to recover held-out evaluation evidence that
   specification §17 already required to be persisted, and which was lost to an
   engineering defect rather than to any scientific choice.

Ground 3 is the load-bearing one, and it is verifiable rather than asserted: the
held-out traces existed only in an in-process mapping, no receipt recorded them,
no artifact was written from them, the run log contains only a traceback, and
the twelve selections were promoted and digest-verified before any held-out
trace existed. The adaptivity hazard that §17's prohibitions exist to prevent
therefore had no channel through which to act.

## 3. The original failure — record

### 3.1 What happened

The canonical attempt claimed its run directory at `2026-08-21T19:47:24Z`,
completed stages 1 through 23, and raised at stage 24 `promote_oof_result`:

```
KeyError: 'true_positive'
  t1_canonical_driver.execute:512  -> collaborators.assemble_oof_result(...)
  t1_assembly:612                  -> _build_assemble_oof_result(...)
  t1_assembly:336                  -> tp = int(primary_confusion["true_positive"])
  t1_composition:225               -> return self._resolved()[key]
```

Elapsed: 10 min 32.8 s from claim to failure.

### 3.2 Classification

**Software integration defect.** A producer-consumer key-vocabulary mismatch
between two first-party modules:

| Layer | Vocabulary |
|---|---|
| `t1_fold_evaluator` produces | `tp` · `fp` · `tn` · `fn` |
| `t1_composition` pools and forwards unchanged | `tp` · `fp` · `tn` · `fn` |
| `t1_assembly` consumes | `true_positive` · `false_positive` · `false_negative` · `true_negative` |

The mapping was wrapped lazily, so the mismatch first surfaced when stage 24
read it — after the claim and after all twelve folds. The pre-claim capability
gate proves callable shape, attestation and structural production of a value; it
does not compare key vocabularies, which is a runtime data contract rather than a
callable contract.

Not a scientific failure: no protocol rule was violated. Not a data failure:
both evidence arrays verify against their recorded digests. Not a contamination
failure: the fold barrier held in all twelve folds and TEST was never touched.
Not a runtime failure: the frozen 335-package dependency digest matched at every
enforcement point.

### 3.3 The compounding defect

Specification §17 requires the held-out **state trace and evaluation evidence**
to be persisted once. The harness persisted only the state trace, by widening it
into the stage-23 OOF store. The per-fold confusion counts, episode evidence and
onset latencies were held in memory and destroyed with the process.

**This, and not the key mismatch, is why the failure was unrecoverable.** A
one-line naming defect at stage 24 would have been a resumable inconvenience had
§17 been implemented in full.

### 3.4 The missing failure receipt

Specification §25 requires a failure receipt with fourteen named fields. None
was written. `PERSIST.write_failure_receipt` and
`T1DevelopmentRun.failure_receipt()` are both implemented, but no caller invokes
them and the driver's `execute` contains no exception handler, so the exception
propagated to the interpreter.

`T1_RUN_STATUS.json` was likewise written once at claim and never refreshed. It
still reads `status: STARTED`, `label_blind_input_opened: false` and
`held_out_labels_opened_for_folds: []` — none of which was true at the moment of
failure.

A **reconstructed** failure receipt carrying all fourteen fields is required by
§13.2 of this amendment. It is stored **outside** the consumed run directory and
is marked reconstructed. It is not recovered evidence; it is a post-hoc record,
and it must never be presented as an artifact the run produced.

## 4. What this amendment changes

Exactly three clauses. Each is quoted as frozen, then amended.

### 4.1 Specification §1 — alternate run root

> "The attempt name is **deterministic**. There is no timestamp, no UUID, no
> random suffix, no automatic retry, no `recovery1` / `recovery2`, no fresh-seed
> attempt and no alternate run root."

**Amended, narrowly.** One alternate run root is authorized, for the single named
identity in §7 of this amendment and for no other. The prohibitions on
timestamps, UUIDs, random suffixes, automatic retry, numbered recovery series
and fresh-seed attempts are **unamended and remain absolute**. No successor
identity is predeclared.

The remainder of §1 is unamended, including:

> "The claim directory is consumed once created. A failure after the claim
> requires documented human review. No automatic recovery identity is
> predeclared, because predeclaring one is how a second attempt becomes
> reachable without a human deciding it should be."

This amendment **is** that documented human review. It does not make a second
attempt automatically reachable; it authorizes one continuation by name.

### 4.2 Specification §17 — "once"

> "Persist the held-out state trace and evaluation evidence **once**. There is no
> fold retry."

**Interpreted and amended.** "Once" is held to constrain
**decision-informing evaluation**, not **evidence persistence**. Where a held-out
evaluation has been performed but its evidence was never persisted and never
observed, one further evaluation under the already-promoted selection is a
completion of §17's persistence obligation, not a second decision-informing
evaluation.

Every prohibition listed in §17 is **unamended and remains absolute**:

> Forbidden: running the 11 rejected candidate policies on the held-out subject ·
> deriving a new threshold · changing persistence · changing cold-start logic ·
> **reselecting after seeing held-out results** · retrying the fold.

### 4.3 Protocol §14 — "exactly once"

> "the selected policy is then evaluated **exactly once** on the held-out
> subject. No fold retry and no fold-specific manual override."

**Interpreted identically to §4.2** and only in that respect. The frozen
12-fold leave-one-subject-out design, the FIT/HELD-OUT partition, the
FIT-only threshold and selection rule, and the prohibition on fold-specific
manual override are **unamended**.

## 5. What this amendment does not change

All other clauses of both governing documents remain in force verbatim.
Explicitly and without limitation:

- Protocol §14 development split — the same twelve frozen VALIDATION subjects.
- Protocol §15 development-optimism disclosure — see §12 below.
- Specification §3 canonical interpreter and runtime integrity.
- Specification §12 fold-scoped label firewall — **the most important execution
  rule in the specification**, unamended in every particular.
- Specification §13–§16 FIT threshold generation, FIT policy evaluation, exact
  FIT selection, and the fold-selection promotion barrier.
- Specification §18 OOF state-evidence store.
- Specification §19–§23 result, exposure semantics, subject evidence and
  bootstrap, challenge reporting, final all-VALIDATION configuration.
- Specification §24 persistence and artifact plan.
- Specification §25 failure semantics, including **"No failed attempt is deleted
  or rewritten to look clean."**
- Specification §26 TEST firewall — **absolute, and see §11 below**.
- The runtime integrity sentinel document in full.
- Every upstream retention decision.

## 6. Preserved evidence boundary

### 6.1 Reusable — consumed as verified input by the continuation

| Artifact | Basis for reuse |
|---|---|
| Label-blind input evidence + array | Digest-verified. `contains_label`, `contains_target_family`, `contains_challenge_identity` all false. Row census 492,904 (492,898 scored, 6 unavailable). Built from frozen upstream artifacts and frozen U1 calibrators only. |
| Input lineage | Binds the input store to `ltstdb_baseline_v1_feature_corpus` and records `forbidden_members_never_opened: [target_family, label, primary_mask, cold_start_bin]` — the proof that §12 was honoured during assembly. |
| Twelve fold selections + aggregate | All twelve digests verify. Each is a §16 barrier artifact, promoted and verified before its held-out labels were authorized. |
| OOF state-evidence store + array | Digest-verified. Exactly one held-out trace per subject, verified subject↔fold bijection over 492,904 rows, `cross_fitted: true`, no label or target family present — which §18 explicitly prefers. |
| Upstream M2 / U1 / T2 artifacts | Untouched by the failure; separately frozen and separately forbidden from rerun. |

**The unifying reason for reuse:** every artifact above was promoted before the
failure, was digest-verified at promotion, and encodes no decision that the
missing measurement could influence. Every choice they contain was made under
FIT-only evidence.

### 6.2 Not reusable — must be produced by the continuation

| Missing | Why it cannot be reconstructed from what survives |
|---|---|
| Held-out evaluation evidence (per-fold confusion, episode evidence, onset latencies) | Does not exist in any persisted form. This is the sole item requiring re-measurement. |
| `T1_OOF_RESULT.json` (§19) | Never written. Its state-burden and state-flow blocks are derivable from the persisted label-free array; its episode, onset, PRIMARY-window and descriptive blocks require held-out labels. |
| `T1_SUBJECT_EVIDENCE.json` (§21) | Never written. Requires per-subject held-out evidence; the OOF store is label-free by design. |
| `T1_BOOTSTRAP.json` (§21) | Never run. Resamples subjects over frozen per-subject statistics — exactly what was lost. |
| `T1_CHALLENGE_EVIDENCE.json` (§22) | Never written. Requires the state traces joined to target family at stage 26. |
| `T1_FINAL_CONFIGURATION.json` (§23) | Never written. §23 permits it only after the complete OOF result is promoted and verified. |
| `T1_RESULT.json`, `T1_EXPERIMENT_LOCK.json` | Never written. Terminal artifacts of stages never reached. |

## 7. Continuation identity rules

| Field | Value |
|---|---|
| Experiment identity | `T1_state_machine_v1` — **unchanged** |
| Run class | `t1_continuation_measurement` — new, distinct from both `canonical_t1_development` and `harness_verification` |
| Attempt id | `t1-v1-measurement-continuation` |
| Run root | `cardiosentinel-runs/phase9-t1-continuation-v1` |

Rules:

1. **The attempt id must not begin with `t1-v1-development`, and the run root
   must not begin with or sit inside `phase9-t1-development-v1`.** Both strings
   are canonical reserved prefixes, matched case-insensitively by prefix. A name
   such as `t1-v1-development-continuation` is reserved and must be refused.
2. The run class is neither of the two existing classes, because neither is
   honest. The continuation is not a canonical first attempt and is not
   synthetic verification output.
3. `protocol_evidence: true`. Its output **is** T1 development evidence, subject
   to §12 of this amendment.
4. No successor continuation identity is predeclared, for the reason
   specification §1 gives.
5. The continuation must **refuse to start** unless the consumed attempt's run
   directory exists and every digest in §1.3, §1.4 and the fold-selection binding
   re-verifies. A continuation that could run without its predecessor would be a
   fresh experiment wearing a continuation's name.
6. The continuation's stage set must contain **no selection stage and no
   threshold-generation stage**. This is a structural requirement on the stage
   list, not a prose undertaking.
7. A human must name the continuation commit at execution time, exactly as
   specification §1 requires for the canonical attempt.

## 8. Provenance rules

Every continuation artifact must bind, at minimum:

```
continues:
  predecessor_attempt_id           t1-v1-development
  predecessor_run_root             cardiosentinel-runs/phase9-t1-development-v1
  predecessor_authorized_git_sha   c538181eb93884f4583a8bd328e50573efbcf3df
  predecessor_terminal_stage       promote_oof_result
  predecessor_outcome              post_claim_failure
  amendment_document_sha256        <this document>

consumed_evidence:
  input_evidence_content_sha256    57d434d9...
  input_evidence_array_sha256      4391b4e7...
  oof_state_content_sha256         cf74f00a...
  oof_state_array_sha256           72f13a8b...
  fold_selection_binding_sha256    32bab16c...
  fold_selection_sha256[0..11]     as tabulated in §1.4

is_canonical_first_attempt         false
is_retry_of_consumed_attempt       false
first_attempt_consumed_by          t1-v1-development
selection_performed_here           false
thresholds_generated_here          false
```

Upstream identities carry forward from the consumed attempt's preflight and
lineage and must be **re-validated, not copied**. The continuation must also bind
Git identity, runtime identity, the protocol and specification digests, and the
digest of this amendment, and must record the exact label-access choreography,
per specification §24.

## 9. Permitted continuation

The continuation is permitted to, and only to:

1. Verify the predecessor's artifacts by digest before anything else.
2. Re-verify each fold's promoted selection digest immediately before opening
   that fold's held-out labels, re-proving the §16 barrier rather than assuming
   it.
3. Open the held-out subject's evaluation labels and target-family metadata for
   one fold at a time, under the selection already promoted for that fold.
4. Run the **already-selected policy exactly once** per held-out subject, per
   specification §17.
5. Persist the held-out evaluation evidence that §17 required and the consumed
   attempt omitted.
6. Execute stages 24 through 29: OOF development result (§19), subject evidence
   and the 1000-replicate bootstrap at seed 2026 with subject as sampling unit
   and no reselection inside the bootstrap (§21), challenge reporting join
   (§22), final all-VALIDATION configuration (§23), experiment lock, completion.

### 9.1 The approved continuation exercises less than this section permits

**The authorized continuation does not perform item 4 above, and must not.**

§9 states the outer envelope of what this amendment would permit. The human
decision that accompanies it narrows the exercise of that envelope further, and
the narrower form is what is authorized:

> The continuation consumes the **persisted OOF state trace** of the consumed
> attempt. It does not regenerate it. Persisted predictions plus held-out labels
> yield evaluation evidence; nothing else runs.

Consequently the continuation does **not** re-run the selected policy, does not
re-execute the frozen state machine, does not re-derive a transition, and does
not re-enter a fold. The emitted states, state durations and transitions it
measures against are the ones already promoted under content digest
`cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f` and array
digest `72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232`.

Three consequences follow, and all three are improvements on the envelope:

1. **The scientific claim rests on the original immutable state trace.** There is
   no second trace that could differ from the first, so no question of
   determinism, ordering or floating-point reproducibility arises.
2. **"The already-selected policy is not re-run" becomes literally true**, not
   merely true in effect.
3. **Episode grouping and matching are performed by the frozen protocol
   functions** the consumed attempt itself used. The continuation introduces no
   new science; it supplies one new input, the held-out labels, to code that is
   byte-identical to what ran before.

Permission exceeding exercise is safe in this direction and only this direction.
§13.6 makes the narrowing mechanically provable so that it cannot quietly widen
back to §9's envelope without a further documented human decision.

## 10. Forbidden actions

Absolutely forbidden to the continuation:

1. **Any policy selection.** No candidate evaluation, no selection metrics, no
   tie-breaking. The twelve selections are input, never output.
2. **Any threshold generation or derivation.** Thresholds are read verbatim from
   the promoted selections.
3. **Any change to the candidate set, persistence profile, cold-start logic or
   transition rule.**
4. **Any change to fold assignment or to the subject↔fold bijection.**
5. **Reselection after seeing held-out results** — specification §17, unamended.
6. **Running the eleven rejected candidates on any held-out subject.**
7. **Any TEST access of any kind.**
8. **Any change to upstream arms.** No M2, U1 or T2 rerun, re-selection or
   re-calibration.
9. **Any write into the consumed run directory**, which is immutable:
   `T1_FAILED_ATTEMPT_MAY_BE_DELETED_OR_REWRITTEN = false`. It is not deleted,
   renamed, re-rooted, extended, tidied or made to look clean.
10. **Any automatic retry.** If the continuation fails post-claim, that is a
    further documented human decision. No second continuation is authorized by
    this document.
11. **Any change to code in response to scientific results.**

## 11. TEST rules

Specification §26 is restated and **unamended**:

> Absolute. The T1 execution package exposes **no TEST option**. Before any path
> resolution it refuses `partition == "test"`, TEST subjects, TEST result paths,
> TEST target metadata, TEST upstream row evidence and TEST metrics.

Every continuation artifact records `test_accessed: false` and
`sealed_test_state: "unopened"`. The B4/neural sealed-test firewall governed by
`B4_TEST_DEFERRAL_DECISION_V1` is untouched by this amendment. No TEST work is
authorized here, and none may be inferred from the completion of T1.

## 12. Claim scope and reporting rules

Protocol §15 is restated and **unamended**:

> **T1 LOSO is subject-disjoint for T1 POLICY SELECTION only. It is not
> independent end-to-end validation.** Future T1 OOF evidence must therefore be
> described as **cross-fitted T1 development evidence conditional on frozen
> upstream components** — never as unseen generalization, external validation,
> independent validation or clinical validation.

Additionally, and specific to this amendment: **any report, paper, figure or
summary derived from the continuation's evidence must disclose that the
canonical attempt failed at stage 24, that the evidence was completed by an
amended measurement continuation, and that this amendment authorized it.** The
continuation's provenance makes this discoverable; the disclosure requirement
makes it unavoidable.

## 13. Acceptance criteria

The continuation may not be authorized to execute until every item below is
satisfied and independently checked.

### 13.1 Governance

1. This amendment is merged and frozen, and its SHA-256 is pinned in the
   codebase and in the continuation's provenance block.
2. A human names the continuation commit at execution time.

### 13.2 Record of the consumed attempt

3. A reconstructed failure receipt exists **outside** the consumed run
   directory, carries all fourteen fields named by specification §25, and is
   explicitly marked as reconstructed rather than recovered.
4. The consumed run directory is byte-unchanged since `2026-08-21T19:57:57Z`.

### 13.3 Evidence integrity

5. Every digest in §1.3, §1.4 and the fold-selection binding re-verifies.
6. The OOF store still shows twelve distinct subjects, twelve distinct fold
   indices, a verified subject↔fold bijection, and 492,904 rows.
7. Upstream validators pass: M2 with 17 bound fields, U1 with 35, T2 with 36.

### 13.4 Engineering

8. The confusion-key contract is repaired at the junction, with a contract test
   asserting that producer keys cover consumer keys for every collaborator the
   composition root wires.
9. The §17 held-out evaluation evidence persistence is implemented, so that a
   completed held-out evaluation survives process exit.
10. Failure-receipt generation is wired into the execution path and proven by
    test, so that a future post-claim failure produces the artifact §25
    requires.
11. Run-status updates are wired, so `T1_RUN_STATUS.json` reflects the state that
    actually holds.
12. The continuation's stage set is proven, structurally, to contain no
    selection or threshold-generation stage.

### 13.5 Environment and firewall

13. The runtime dependency digest matches the frozen 335-package set
    `b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a`, under
    Python 3.12.6.
14. The working tree is clean and HEAD equals the named continuation commit.
15. TEST firewall tests pass unchanged, and no `TEST_ATTEMPT*` artifact exists.

### 13.6 Mechanical continuation constraints

The narrowing recorded in §9.1 is an **authorization gate**, not an
implementation convention. A constraint that exists only in code is a constraint
the code can forget, which is the reason specification §16 required its own
barrier to be structural. Each constraint below must be proven at three
independent layers, and a proof at one layer does not substitute for another.

| # | Constraint | Required value |
|---|---|---|
| 16 | Zero state-machine regeneration | `state_machine_invocations = 0` |
| 17 | Zero threshold generation | `threshold_generation_calls = 0` |
| 18 | Zero policy selection | `policy_selection_calls = 0` |
| 19 | Zero fold evaluator execution | `fold_evaluations = 0` |

**Layer 1 — structural.** The continuation's stage set contains no selection
stage and no threshold-generation stage, and its import surface reaches no
transition entry point, no threshold generator, no candidate evaluator and no
fold evaluator. Proven by import surface and syntax tree, **never by scanning
source text**: a module's own refusal list contains the words it refuses, and a
receipt asserting an action did not occur contains that action's name.

**Layer 2 — runtime.** Each of the four entry points is instrumented with a
counter. The counters are read at completion and every one must be zero. This is
positive proof that nothing ran, which absence of code is not: a counter that
reads zero is evidence, whereas an unreferenced import is only an argument.

**Layer 3 — evidence.** The measured trace must be the predecessor's trace:

- the consumed `emitted_state`, `state_elapsed_seconds`, `transition_from`,
  `transition_to` and `transition_occurred` columns digest to the predecessor's
  array digest `72f13a8b…`;
- each fold's `p_watch`, `s_watch`, `p_event`, `s_event` equal the values in that
  fold's promoted selection artifact exactly;
- each fold's `selected_policy_id` equals the value in that fold's promoted
  selection artifact exactly;
- no continuation artifact carries a `policy_runs` counter, because no policy
  was run.

### 13.7 The continuation execution attestation

The continuation must promote an **execution attestation artifact** carrying at
minimum:

```
artifact_class                    t1_v1_continuation_execution_attestation
state_machine_invocations         0
threshold_generation_calls        0
policy_selection_calls            0
fold_evaluations                  0
state_trace_source                predecessor_oof_state_evidence
state_trace_content_sha256        cf74f00a6eb38471e80ce008dc6b88d16aa5c36b110bce87c7c37dba6d7d835f
state_trace_array_sha256          72f13a8b29eafdd99801bb64dbf8b61f19717f3d7af777d74f21c9709dd28232
selection_performed_here          false
thresholds_generated_here         false
state_transitions_regenerated     false
predecessor_digests_verified      true
test_accessed                     false
sealed_test_state                 unopened
```

The attestation travels **with the evidence**, not in a test log, for the same
reason every other artifact records `test_accessed: false`: a claim that lives
only where the evidence does not is a claim a future reader cannot check.

A non-zero counter, a failed digest comparison, or a missing attestation is a
**refusal**, not a warning. The continuation must stop, and the failure is
governed by §14.

## 14. If the continuation fails

A post-claim failure of the continuation consumes the continuation attempt. Its
run directory is then immutable on the same terms as the consumed canonical
attempt: not deleted, not rewritten, not retried automatically, and not
succeeded by any predeclared identity.

This amendment authorizes **one** continuation. A second would require a further
human decision, documented on the same footing as this one.

## 15. Scope

This document authorizes exactly one measurement continuation of one consumed
canonical attempt, and amends exactly three named clauses to permit it.

It is not a precedent for retrying failed experiments. The grounds recorded in
§2 are unusually narrow — selections promoted and verified before evaluation,
and results never observed by any process or person — and an amendment resting
on different grounds would be a different decision requiring its own review.
