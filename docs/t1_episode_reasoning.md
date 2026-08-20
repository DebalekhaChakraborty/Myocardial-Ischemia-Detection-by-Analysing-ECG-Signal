# T1 episode reasoning — the execution harness

This document explains the harness that runs the frozen T1 causal episode-state
protocol. The protocol itself is
[`T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1.md`](T1_CAUSAL_EPISODE_STATE_PROTOCOL_V1.md),
digest `ef044754020b1756ea7aae5fa1b747c5ba6fc0c8cd70d52e73185555897d70d4`, and it
is immutable. Nothing here changes it; this is the machinery that executes it.

---

## 1. Why temporal reasoning is separate from neural classification

A window classifier answers one question: *does this ten seconds of signal look
ischemic?* That is a genuinely different question from the one a clinician or an
operator actually has, which is *is this patient having an episode right now?*

Four reasons the second question gets its own layer rather than a bigger model.

**A classifier has no memory, and an episode is made of memory.** Ischemia is not
a property of a ten-second window; it is a sustained physiological state with an
onset, a duration and a resolution. Asking a per-window model to also decide
"this is the third minute of an ongoing episode" requires it to carry state
across windows — which either makes it non-causal, or makes it a state machine
wearing a neural network's clothes.

**Thresholded per-window output is unstable in exactly the way alerts must not
be.** The retained upstream temporal arm is measurably fragmented at its own
frozen operating point: 1787 predicted runs against the comparator's 1081, a
median run of 10 s against 25 s, and roughly half its positive windows isolated
singletons. Feeding that directly to an operator produces an alarm every few
minutes. Persistence and hysteresis are what convert a noisy per-window signal
into a state a human can act on, and they are policy, not perception.

**The two layers fail differently and should be debuggable separately.** When an
episode is missed, the useful question is *which layer missed it* — did the
detector never fire, or did it fire and the persistence budget swallow it? A
fused model gives one number and no answer. A separate state machine gives a
transition log where the streak counters are visible at every window.

**A state machine is explainable in a way a network is not.** Every transition
here is a comparison against a named threshold and a named consecutive-window
budget. The reason string in the transition log is not a post-hoc rationalisation
of a learned decision; it is a restatement of the arithmetic that produced it.

There is a fifth reason, specific to this program. The upstream components are
frozen and were selected on the full development population. Building the
temporal layer as a separate deterministic machine means its own development can
be cross-fitted *without retraining or re-selecting anything upstream* — and,
just as importantly, means the harness cannot accidentally launder an upstream
choice into a new one.

---

## 2. What T1 is not

Repeating the protocol's own list, because it is the most common source of
confusion when reading harness code:

T1 is **not** another neural model, **not** another temporal model, **not**
another calibrator, **not** a post-hoc smoother, **not** a relabelled binary
threshold from the temporal arm, **not** an LLM, and **not** an edge/cloud
router. It is a deterministic causal state machine over already-frozen evidence.

The harness reflects this structurally: `t1_stream.py`, `t1_engine.py` and
`t1_execution.py` import no model machinery at all, and a test walks their import
graphs to prove it.

---

## 3. The state space and the transition rules

Four states, initial state `NORMAL`, held **per stream** — one stream is one
`(record_id, channel_index)` pair. State never crosses a stream, and therefore
never crosses a record, a channel or a subject. Patient-level multi-channel
fusion is undefined in V1 and the harness does not invent it.

```
                 watch evidence (one window, immediate)
        NORMAL ──────────────────────────────────────────▶ WATCH
           │                                                 │
           │  event evidence × confirm budget                │  event evidence
           │  (escalation has priority)                      │  × confirm budget
           ▼                                                 ▼
         EVENT ◀───────────────────────────────────────────────
           │  ▲
           │  │ event evidence × re_event_confirm_windows
           │  │
           │  └──────────────── RECOVERY
           │                        │
           │ normal evidence ×      │ normal evidence ×
           │ event_release_windows  │ recovery_clear_windows
           ▼                        ▼
        RECOVERY                  NORMAL
```

`WATCH` also returns to `NORMAL` on `watch_clear_windows` consecutive windows of
NORMAL evidence. **`RECOVERY` never automatically becomes `WATCH`**: a recovering
stream either re-escalates on EVENT evidence or clears all the way down.

### Evidence levels

Per window, from the calibrated probability `p`, the temporal score `s` and the
detector decision `d`:

| Level | Condition |
|---|---|
| WATCH evidence | `d` **or** `p ≥ p_watch` **or** `s ≥ s_watch` |
| EVENT evidence (mature) | `d` **and** `p ≥ p_event` **and** `s ≥ s_event` |
| EVENT evidence (cold start) | `d` **and** `p ≥ p_event` |
| NORMAL evidence | `not d` **and** `p < p_watch` **and** `s < s_watch` |
| ambiguous | anything else — treated at WATCH level |

### Persistence, hysteresis and the three profiles

Persistence is a count of **consecutive available windows**, not elapsed
wall-clock time, so a dropout cannot be counted as agreement. Hysteresis is the
gap between the escalation budget and the release budget: entering EVENT and
leaving it are deliberately not symmetric, which is what stops a stream sitting
on a threshold from oscillating.

| | FAST | BALANCED | CONSERVATIVE |
|---|---|---|---|
| `watch_clear_windows` | 2 | 3 | 6 |
| `event_confirm_windows` | 2 | 3 | 6 |
| `event_release_windows` | 2 | 3 | 6 |
| `re_event_confirm_windows` | 1 | 2 | 3 |
| `recovery_clear_windows` | 3 | 6 | 12 |
| `cold_event_confirm_windows` | 4 | 6 | 12 |

`WATCH` entry is immediate on a single window; there is no watch-confirmation
budget to tune.

### Cold start

Below 300 s of stream age, EVENT evidence drops the temporal term and uses the
longer `cold_event_confirm_windows` budget. The upstream temporal arm recorded
**0.0** thresholded sensitivity in its first five minutes, so requiring the
temporal term there would make early EVENT unreachable by construction rather
than by evidence. This relaxes a T1 rule. It repairs nothing upstream and
changes no upstream state.

### Missing windows

A window with unusable signal quality or no score is **not evidence of
anything**. On such a window the harness holds the state, advances state time,
resets *every* confirmation streak, and fires no transition. Nothing is imputed,
forward-filled or zero-filled — the config keys that would enable those exist
only so that setting one to `true` produces a refusal rather than silence.

Resetting the streaks is the important part. A gap must not be able to confirm an
escalation or a release across itself: five hot windows, a dropout, five more hot
windows is *not* ten consecutive confirmations, and the harness has a test that
says so.

---

## 4. Causal guarantees

**Definition.** No window may influence any decision at or before its own
position. Concretely: truncating the stream at position *k* must produce exactly
the trace that the full run produced for its first *k* windows.

That is not an aspiration in this harness; it is the test. `test_a_prefix_of_the_stream_produces_a_prefix_of_the_trace`
runs the full stream and seven truncations of it and compares the serialised
traces. A second test mutates a *future* window and asserts the earlier trace is
byte-identical.

Three mechanisms make it true:

1. **The stream is consumed in the order given and is never re-sorted.** A window
   that does not strictly follow its predecessor on that stream is refused. This
   matters more than it looks: sorting a batch would let a later window decide
   where an earlier one belongs, which is future dependence with extra steps.
2. **No lookahead buffer exists.** The adapter and the engine iterate the input
   once, pulling one window per decision. A test parses their ASTs and asserts
   neither calls `list()`, `sorted()`, `tuple()` or `len()` on the input stream —
   a lookahead would have to be constructed before it could be consulted.
3. **The transition function is pure and sees one row.** `next_state` in the
   frozen protocol module takes the current state, the current streaks and the
   current row. There is no argument through which the future could arrive.

**What the transition function may see:** `stable_id`, the detector score, the
detector decision `d_t`, the calibrated probability `p_t`, the derived
uncertainty `u_t`, the temporal score `s_t`, availability, elapsed stream time,
elapsed state time.

**What it may never see:** any label, `target_family`, subject outcome, episode
identity, any future row or score, the comparator arm's score, the temporal arm's
binary decision or reporting threshold, `u_star_dev`, `u_star_deploy`, challenge
identity, the upstream gate outcome, `m2_update_admitted`, or any test-derived
quantity. `require_no_forbidden_fields` refuses a payload carrying any of them.

Signal quality is not in the forbidden list because it is genuinely available on
a live stream — it decides *availability*, never the transition itself. Context
and confounder flags are carried into the outputs for stratified reporting and
are structurally barred from the transition path; a test runs the same stream
with and without flags and asserts the states are identical.

---

## 5. Configuration is selection, not tuning

`configs/t1_episode.yaml` drives the harness, and nothing operational is
hardcoded in Python. But config-driven does not mean freely tunable, because the
protocol's central guarantee is that its thresholds were **generated
prospectively**, from FIT-subject background negatives, by exact empirical order
statistic `k = ceil(q·N)` — never chosen by a human who had already seen results.
A config file that could simply state `watch_threshold: 0.8` would dissolve that
guarantee completely.

So the config selects *which frozen option applies*, and the harness recomputes
the rest:

| Config genuinely chooses | Harness derives or enforces |
|---|---|
| quantile pair `q_watch`, `q_event` | the four numeric thresholds, by the frozen order-statistic rule |
| persistence profile by name | the six window budgets, from the frozen profile |
| alerting policy and refractory | every state transition, from the frozen machine |
| run class, run root, attempt id | whether the run may claim to be protocol evidence |

Declared values that duplicate frozen ones — the six `expected_windows`, the
protocol digest, the cold-start rule — are **assertions the harness checks**, not
sources it reads. A disagreement corrects the config, never the protocol.

### Two run classes

| | `canonical_t1_development` | `harness_verification` |
|---|---|---|
| Thresholds | must be derived | may be literal |
| Frozen values | must match exactly | may deviate |
| Authorization | requires a T1 execution specification | none |
| Artifacts stamped | `protocol_evidence: true` | `protocol_evidence: false` |

**No T1 execution specification has been authorized**, so the harness refuses to
construct a canonical run at all, and additionally refuses to claim any attempt
id in the reserved canonical namespace. A synthetic run cannot consume the
canonical attempt by accident.

---

## 6. Where the refractory period lives

The protocol has no refractory concept. Adding one to the transition function
would be a different protocol, and one that could hold back a genuine escalation.

So the refractory period applies to **alert emission only**. Run the same stream
with a refractory of zero and of an hour: the state trace, the episodes, the
transitions and the recovery spans are byte-identical, and only the `suppressed`
flag on alerts differs. Suppressed alerts are still recorded, so nothing
disappears from the audit trail — an operator sees one notification, a reviewer
sees both.

The config key `refractory_applies_to` must read `alert_emission_only`. Any other
value is refused, which is the point of having the key at all.

---

## 7. Outputs

| Artifact | Contents |
|---|---|
| `T1_STATE_TRACE.json` | one entry per window: evidence, state before and after, streaks before and after, cold-start flag, gap seconds |
| `T1_EPISODES.json` | maximal EVENT runs per stream, with onset, offset, window count and duration |
| `T1_TRANSITIONS.json` | only the windows where the state changed, each with a reason |
| `T1_ALERTS.json` | notifications, with refractory suppression applied and recorded |
| `T1_RECOVERY_SPANS.json` | each RECOVERY period and its outcome: cleared, re-escalated, or open at stream end |
| `T1_RUN_MANIFEST.json` | git SHA and dirtiness, config digest, runtime metadata, input stream digest, protocol digest |
| `T1_RUN_RESULT.json` | output digests, thresholds used, counts, explicit non-claims |

Episode duration is `window_count × stride`, and episodes are physically
contiguous: a cadence gap is never bridged.

The run manifest treats **input order as part of input identity** — the same
windows in a different order produce a different digest, because they are a
different causal stream.

---

## 8. What the harness does not do

It does not optimize thresholds. It does not tune against test data. It makes no
performance claim: `T1_RUN_RESULT.json` records counts and digests, and carries
`performance_claimed: false`, `thresholds_optimized: false` and
`tuned_against_test_data: false` as explicit fields rather than as absences.

It does not open the sealed test partition, define a router, or involve a
language model in any state decision.

It does not calibrate. `p_t` must arrive already calibrated by the frozen,
already-fitted upstream calibrator; a window that is available but carries no
calibrated probability is refused rather than back-filled with a raw model score.

---

## 9. Feeding it a model

The harness is model-agnostic by construction: `T1WindowEvidence` is the only
input type, and it names no model. Anything that can emit a score, a calibrated
probability and a temporal score per window is a valid producer — a future B4
neural output, an ensemble, or a synthetic reviewer stream.

```python
from cardiosentinel.neural.t1_config import load_t1_episode_config
from cardiosentinel.neural.t1_execution import execute_t1_run
from cardiosentinel.neural.t1_stream import T1WindowEvidence

config = load_t1_episode_config("configs/t1_episode.yaml")

windows = [
    T1WindowEvidence(
        window_id=stable_id,
        subject_id=subject,
        record_id=record,
        channel_index=channel,
        start_sample=start,
        model_score=score,                    # any model
        calibrated_probability=probability,   # from the frozen calibrator
        temporal_evidence=temporal,
        signal_quality="good",
        context_flags=(),                     # reporting only
    )
    for ... in ...
]

result = execute_t1_run(windows, config)
```

Connecting a real model requires no change to any module in this harness. What it
requires is a config, a calibrated probability, and a chronological stream.
