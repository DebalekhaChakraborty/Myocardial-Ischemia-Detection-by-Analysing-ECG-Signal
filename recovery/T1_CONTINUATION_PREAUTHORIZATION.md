# T1 Continuation Pre-Authorization Record

This is a **pre-authorization record**. It states what was verified, against
which clauses, at which commit, so that the human deciding whether to arm the one
authorized measurement continuation decides against checked facts rather than a
summary of them.

**It authorizes nothing.** `T1_CONTINUATION_AUTHORIZED` is `False` and is
unchanged by this document. The block in §1 is deliberately unsigned: the
amendment reserves that decision for a human, and a record that signed on their
behalf would be the opposite of what this file exists for.

It lives in `recovery/`, which is tracked, unlike `cardiosentinel-runs/`. That is
deliberate for the same reason the reconstructed failure receipt lives here: a
governance record that exists on one disk only is not a record.

| | |
|---|---|
| Prepared | 2026-08-22 |
| Governing document | `T1_EXECUTION_RECOVERY_AMENDMENT_V1_1` |
| Amendment SHA-256 | `d3ea7734c93be8f59796e03e8c0210778716327f7adc033cb2d3dcfff7f92c96` |
| Method | read-only audit — nothing executed, no held-out label opened |

---

## 1. Authorization status

```
Technical readiness    : GO
Execution authorization: AUTHORIZED
Continuation executed  : NO
```

```
Authorized by       : Debalekha Chakraborty
Date                : 2026-08-22
Authorization commit: b40b4acac16893dcb1af1f1fa91feb0d74c8a78d
Execution commit    : 61704aa7259d91eaf9d4dfc2502bf78881a05d61
```

**The authorization enables exactly one continuation execution. Failure after
claim consumes the authorization. No retry or successor identity is authorized.**

The flag is now `True` on disk. The continuation will proceed past stage 1 when
an operator invokes it. It has not been invoked: `Continuation executed: NO`, and
`cardiosentinel-runs/phase9-t1-continuation-v1` does not exist.

The test suite cannot invoke it. `tests/neural/conftest.py` forces the flag back
to `False` for the duration of a test session, because arming is an operator
decision and pytest is not the operator.

---

## 2. Repository state

| | |
|---|---|
| `master` | `467272220ea4d757a09f91797a19cb66ff177ea7` |
| Working tree | clean |
| Open PRs at preparation | 0 (#48–#56 all merged) |
| Continuation run root | **absent** — `cardiosentinel-runs/phase9-t1-continuation-v1` does not exist |
| `TEST_ATTEMPT.json` | absent |

---

## 3. Governance readiness

| Item | State |
|---|---|
| Amendment digest `d3ea7734…` | ✅ validated; pinned in `t1_recovery_amendment` and re-derived from the document |
| Reconstructed failure receipt | ✅ `recovery/T1_FAILURE_RECEIPT_RECONSTRUCTED.json`, **14/14 §25 fields**, `receipt_type: "reconstructed"`, outside the consumed run directory |
| Consumed attempt immutable | ✅ 8/8 §1.3 artifacts, 12/12 §1.4 fold selections, 20 files, newest mtime `2026-08-21T19:57:57Z` — the recorded failure timestamp |
| TEST unopened | ✅ `test_accessed: false` on every attempt artifact; `sealed_test_state: "unopened"`; no `TEST_ATTEMPT*` artifact |

The receipt names the three lost quantities — per-fold PRIMARY confusion counts,
episode evidence, onset latencies — as `not_reconstructed` with a reason. No
fabricated measurement appears in it: every numeric value it carries is a count,
byte size, index, stage number or elapsed time.

---

## 4. Scientific boundary

### The continuation performs

- **persisted OOF trace consumption** — the promoted store is read and digest-checked; array `72f13a8b…`, content `cf74f00a…`, 492,904 rows, `contains_label: false`
- **held-out label measurement** — one fold at a time, through the existing §16 authority, under the selection already promoted for that fold
- **evidence generation** — the six run-level artifacts the consumed attempt never wrote

```
persisted OOF state trace  +  held-out labels  ->  measurement
```

### The continuation does NOT perform

| Forbidden | How it is prevented |
|---|---|
| Training | no checkpoint load reachable in the continuation graph |
| Checkpoint loading | structural — no `load` / `load_state_dict` call site |
| State-trace regeneration | `state_machine_invocations = 0`; the trace is consumed, never regenerated |
| Threshold generation | `threshold_generation_calls = 0`; thresholds read verbatim from promoted selections |
| Policy selection | `policy_selection_calls = 0`; the twelve selections are input, never output |
| Fold evaluation rerun | `fold_evaluations = 0`; `t1_fold_evaluator` is never loaded |

The scientific claim continues to rest on the original immutable trace. There is
no second trace, so no question of determinism, ordering or floating-point
reproducibility arises. Episode grouping and matching run through the frozen
protocol functions the consumed attempt itself used.

---

## 5. Challenge strata decision

```
challenge_strata = none
join_performed   = false
```

**No post-result subgroup definition will be introduced.**

Recorded here, before execution, because that is the only point at which it can
be recorded honestly. Choosing challenge strata after seeing the measurement
would be selection on the evidence — §17 forbids it, and no amount of later
documentation would repair it.

The continuation therefore emits `T1_CHALLENGE_EVIDENCE` with `join_performed:
false` and an empty strata map, and records that as an **absent join rather than
an empty subgroup**. An absent join and an empty join are different facts, and
only one of them is evidence.

---

## 6. One-shot acknowledgement

- The continuation is **single-use**: `T1_CONTINUATION_ATTEMPTS_AUTHORIZED = 1`.
- A **post-claim failure consumes** it. Its run directory then becomes immutable
  on the same terms as the consumed canonical attempt — not deleted, not
  rewritten, not tidied.
- **No retry identity exists.** `T1_CONTINUATION_AUTOMATIC_RETRY_PERMITTED = False`,
  and no retry, resume, reset or force path exists in any continuation module.
- **No successor continuation is pre-authorized.** Under §14, a second would
  require a further human decision documented on the same footing as the
  amendment itself.

### Known residue

`execute_continuation` has never run end-to-end and cannot until armed. Its
stages are individually tested and its measurement arithmetic verified on
synthetic evidence with answers known by inspection, but the assembled
claim-to-lock path executes exactly once. That is the amendment's design rather
than a gap in it, and it is why PRs #48–#56 exist.

The reducible part of that residue has been reduced. The label path is proven
against the real authority's signatures and refusals, and the identity artifact
is validated to carry 492,904 rows across all twelve promoted held-out subjects —
a row count equal to the persisted trace — with no label array materialized.

---

## 7. Technical gate summary

| Clause | Result |
|---|---|
| **§13.1** Governance | **PASS**, except the human naming of the continuation commit, which is satisfied *by* authorizing rather than before it |
| **§13.2** Record of the consumed attempt | **PASS** |
| **§13.3** Evidence integrity | **PASS** — see §8 |
| **§13.4** Engineering | **PASS** |
| **§13.5** Environment and firewall | **PASS** — 335 packages, `b0fd6eaa…`, Python 3.12.6, clean tree |
| **§13.6** Negative capability | **PASS** — three independent layers |
| **§13.7** Attestation capability | **PASS** |

### §13.6, in more detail

**Layer 1, structural.** Eight continuation modules proven: zero forbidden
modules, zero forbidden bound symbols, zero forbidden call sites. A fresh
interpreter importing the runner leaves every never-loaded module clean.

**Layer 2, runtime.** Seven instrumented entry points across `t1_protocol` and
`t1_development_run`; five never-loaded modules. The two sets partition the
forbidden set exactly — nothing double-covered, nothing uncovered.

Non-vacuity was **proven, not assumed**. An earlier version of this gate carried
counters no production path could increment; they read zero because nothing could
touch them, which is precisely the substitution §13.6 forbids. The audit drove a
real entry point through the instrumentation: the counter incremented, recorded
`state_machine_invocations:next_state`, `require_all_zero` refused, and the frozen
protocol was restored intact.

**Layer 3, evidence.** Both trace digests match; all twelve folds' thresholds and
`selected_policy_id` equal their promoted selection artifacts exactly;
`policy_runs` is forbidden and absent from the continuation contract.

---

## 8. Known qualification

The three upstream retention validators pass and return the frozen digests
`da4a05b4…` (M2), `9d8436f2…` (U1) and `4846921135…` (T2).

Specification §13.3 item 7 additionally annotates these as "M2 with 17 bound
fields, U1 with 35, T2 with 36". **Those numerical field-count annotations were
not independently re-derived during this authorization preparation.** The
validators return a digest string rather than a field collection, and the counts
could not be mapped to a constant in the codebase during this audit.

This is recorded rather than glossed, and it is **not an execution blocker**: the
substantive requirement — that the upstream validators pass and bind the correct
frozen retention decisions — is verified directly. The specification is not
changed by this record.

---

## 9. Tests

Normal suite only. No continuation was executed, no held-out label was accessed,
and the consumed attempt directory was unchanged throughout.

| | |
|---|---|
| Result | **3056 passed, 1 skipped** |
| Skip | `test_m1_memory_scaling.py:117` — opt-in large-N stress behind `M1_STRESS_ROWS>=500000`; environmental, unrelated to T1 |
| Continuation executed | **NO** |
| Held-out labels accessed | **NONE** — enforced by test, not asserted |
| Attempt directory changes | **NONE** — mtime unchanged at `2026-08-21T19:57:57Z` |

---

## Verdict

**Technical readiness: GO.** Every §13 gate is satisfied except those satisfied by
the act of authorizing. Only human authorization remains.

The signature block in §1 is unsigned.

---

## 10. Post-execution completion

**Added 2026-08-22, after the authorized continuation was executed. Everything
above this section is preserved exactly as it was written before execution.**

The only in-place change made to the pre-execution text is the `Execution
commit` field in §1, which was written as `pending` precisely so that it could
be filled once the commit existed. Every other pre-execution statement stands as
recorded, including `Continuation executed: NO` in §1 and §9 and the sentence in
§1 noting that `cardiosentinel-runs/phase9-t1-continuation-v1` did not exist.
Those were true when written. They are superseded by this section, not edited —
a pre-authorization record that is rewritten after the fact stops being one.

### 10.1 Outcome

```
Continuation executed  : YES
Outcome                : COMPLETED
Authorization          : CONSUMED by a completed run
```

| | |
|---|---|
| `attempt_id` | `t1-v1-measurement-continuation` |
| `run_class` | `t1_continuation_measurement` |
| Execution commit | `61704aa7259d91eaf9d4dfc2502bf78881a05d61` |
| Authorization commit | `b40b4acac16893dcb1af1f1fa91feb0d74c8a78d` |
| Started | 2026-08-22T16:18:39Z |
| Completed | 2026-08-22T16:18:49Z |
| Folds measured | 12 / 12 |
| Artifacts promoted | 19 files (6 run-level, 1 attestation, 12 per-fold) |
| Attestation SHA-256 | `b5a557dd40927999e00516e982c2f1619fdbeb3e5ebdd3ad108037b474eca588` |

The artifacts carry no wall-clock field by design; the times above are the
operator record of the run, and the attestation is the authority on everything
else in this table.

### 10.2 It took two launches, and the first did not consume the attempt

The first invocation raised
`TypeError: git_provenance() missing 1 required positional argument` at
`runner.py:282`, inside `_authorized_git_sha()` — six lines before `_claim()` at
`runner.py:288`. The attempt was therefore **not** claimed and, per §25, the
authorization survived. PR #59 fixed the argument and added the seam test that
should have preceded the first launch. The second launch crossed the claim and
completed.

This is recorded rather than glossed. The refusal was a real defect in the
assembled path, caught by luck of ordering rather than by test, and the lesson
belongs in this record: the stages were each tested and the junctions were not,
which is the same defect class that consumed the canonical attempt at stage 24.

### 10.3 Firewall state after execution

| Counter | Value |
|---|---|
| `fold_evaluations` | `0` |
| `policy_selection_calls` | `0` |
| `state_machine_invocations` | `0` |
| `threshold_generation_calls` | `0` |
| `state_transitions_regenerated` | `false` |
| `test_accessed` | `false` |
| `sealed_test_state` | `unopened` |

The measurement consumed a persisted state trace
(`state_trace_source: predecessor_oof_state_evidence`,
`state_trace_content_sha256: cf74f00a…`) and ran no model. All three negative
capability gate layers passed. Predecessor verification re-verified 8 §1.3
artifact digests and 12 §1.4 fold-selection digests. No file in the run contains
`policy_runs`.

**The consumed attempt directory was unchanged by the continuation**, still 20
files with mtime `2026-08-21T19:57:57`. The continuation wrote only into its own
run root.

### 10.4 Standing after execution

**The authorization is spent.** It was consumed by a completed run rather than a
failed one. `T1_CONTINUATION_AUTHORIZED` remains `True` on disk; that flag is now
a spent token, not a live permission. §14 authorizes no second continuation and
none is predeclared, so the continuation run directory is immutable on the same
terms as the consumed attempt.

TEST was never opened by either run.
