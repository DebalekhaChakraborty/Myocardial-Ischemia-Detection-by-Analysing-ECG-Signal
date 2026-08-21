# Current State

This is a living document, not a frozen protocol record. Unlike the `_V1`
documents elsewhere in this folder, it carries no digest and no freeze ritual —
it is meant to be regenerated wholesale, not amended. Do not hand-edit the
data sections; ask Claude to refresh this file (a fresh read-only pass against
`git`, `gh`, and `cardiosentinel-runs/`) and it will be rewritten in place.
Commentary can go in a `Notes` subsection if needed, but treat everything else
here as disposable output, not source of truth — **the repository is the
source of truth; this file is a cache of it.**

`docs/IMPLEMENTATION_PLAN.md` and `docs/RESEARCH_SCOPE.md` are the project's
narrative plan and have not been revised since 2026-08-07. This file exists
because those two drifted far enough from reality that a 2026-08-21 audit had
to reconstruct actual state from `cardiosentinel-runs/` and git history by
hand. Read this file for "where are we," and the `_V1` docs for "what did we
decide and why."

---

**As of:** `origin/master` `64d5fc9` (local checkout: `3c8d433` on
`research/t1-challenge-and-composition-v1`, now merged remotely), 2026-08-21
**Working tree:** clean (aside from this file)
**Open PRs:** 1 (#39)
**Canonical T1 attempt:** not consumed
**TEST partition:** sealed

## Live flag — read before touching `t1_development_run.py` or PR #39

**Update, same session, second revision:** the `t1_development_run.py`
rewrite tracked below went uncommitted → committed locally (`3c8d433`) →
pushed and merged to master, all within this one audit session, as **PR #45,
"T1: wire canonical composition root"** (`64d5fc91`, merged
2026-08-21T18:18:23Z). `main()` in `src/cardiosentinel/neural/t1_development_run.py`
now really does call `T1CanonicalDevelopmentExecutor.execute(...)` on
`master`, composed by the new `t1_composition.py`, which — per PR #45's own
description — "resolves the frozen artifacts from repository-defined
locations, reconstructs the twelve frozen U1 out-of-fold fits without
refitting them, binds the collaborators the driver threads."

This is the reassuring version of the outcome this section was warning
about. PR #45's merged body states outright: *"This does not authorize
execution. PR #39 remains the separate and final authorization event, and
needs a rewrite rather than a rebase: its refusal prose asserts that nothing
sequences the 29 stages and that no caller supplies a fold evaluator, both of
which this PR makes false."* Whoever merged #45 already reached, in writing,
the same conclusion this audit reached independently: PR #39 cannot be
merged as it stands.

Traced by hand on this pass, against the new master:
`T1_EXECUTION_SPECIFICATION_AUTHORIZED` is still `False`,
`require_canonical_execution_capability()` still checks it, and
`cardiosentinel-runs/phase9-t1-development-v1` is still absent — so the
single canonical T1 attempt is **not** consumed. PR #39 is still open,
unmerged, unrewritten, and — by its own sibling PR's admission — not safe to
merge as written. See §7.

---

## 1. Repository identity

| | |
|---|---|
| Repository | `tactics/Myocardial-Ischemia-Detection-by-Analysing-ECG-Signal` (GitHub: `DebalekhaChakraborty/Myocardial-Ischemia-Detection-by-Analysing-ECG-Signal`, renamed `CardioSentinel-AI`) |
| `origin/master` | `64d5fc91c225266fb958c4f99752afc17714786d` — merge of PR #45, 2026-08-21T18:18:23Z |
| Local checkout (this pass) | branch `research/t1-challenge-and-composition-v1` at `3c8d433` — pushed and merged remotely since; local branch ref not yet fast-forwarded to reflect it |
| local `master` ref | `bbb78d8` — well behind `origin/master` now (stale local ref only; `git fetch` fixes it) |
| Last fetch | 2026-08-21, this refresh |
| Outer repo (`/home/AI_POC`) | HEAD `086ee2813…`, untouched by this pass |

### Working tree — clean

As of this refresh, clean aside from this file. The rewrite of
`t1_development_run.py` and the new `t1_composition.py` /
`test_t1_composition.py` that were uncommitted a few minutes earlier in this
session went local commit (`3c8d433`) → pushed → merged as PR #45 — see
"Live flag" above.

### Open PR — #39 "T1: authorize canonical development execution"

Opened 2026-08-21 01:36 UTC, last touched 02:06 UTC, CI green (both checks),
0 reviews, `gh`-reported mergeable state unknown. Scope as written: flips
`T1_EXECUTION_SPECIFICATION_AUTHORIZED` `False → True` only; no protocol,
spec, model, metric, or fold-logic bytes touched. Its own description:
*"nothing sequences [the 29 stages] end to end … a verified preflight is
therefore followed by an honest stop naming a missing capability, never a
withheld permission."* That description is now confirmed false by the
project's own later work: PRs #40–#45 built exactly that missing capability,
and PR #45's merged body says outright that #39 "needs a rewrite rather than
a rebase" before it can merge safely. Still open, still unrewritten, as of
this refresh.

### Recent history (last 20 commits)

```
64d5fc9 origin/master  Merge PR #45 — wire canonical composition root
3c8d433 (local HEAD)   T1: compose the canonical execution graph and delegate
c87be5d                T1: implement the final all-VALIDATION configuration selection
9e16e32                T1: assemble subject evidence from the held-out evaluations
b202840                T1: derive challenge membership from the canonical identity
95254b7 origin/master  Merge PR #44 — t1-fold-evaluator-v1
34abdc8                T1: implement the canonical fold evaluator
a545666                Merge PR #43 — t1-assembly-collaborators-v1
c578f21                T1: make the assembly collaborators answerable to the capability gate
68478af                T1: add the label-bearing assembly collaborators
e72c93a                T1: prove capability before the claim, not after it
c7a458a                Merge PR #42 — t1-fold-evaluation-capability-v1
74f4c94                T1: add the controlled fold evaluation capability
bbb78d8 master (local) Merge PR #41 — t1-fold-authority-v1
e6f5dfe                Merge PR #40 — t1-canonical-driver-v1
bbefa38                T1: add the fold-scoped evaluation authority
0639c9e                T1: add the canonical development execution driver
5804e66                Merge PR #38 — t1-canonical-development-harness-v1
f91c417                T1: skip canonical-claim tests outside the frozen interpreter
2feb76c                T1: implement the canonical development harness
2672a72                Merge PR #36 — t1-execution-harness-v1
c472fba                T1: harden the episode-state engine against the merged execution spec
```

CI has been green on the last 5 verified pushes to master (#38, #40, #41,
#43, #44 — ~6 min each, two jobs). PR #45's own merged description reports
full-suite 2,764 passed / 1 skipped and a clean `ruff check .`, not
independently re-run by this audit.

## 2. Where this stands vs. the plan docs

No file named `HANDBOOK` exists anywhere under `/home/AI_POC`. The closest
things are `docs/IMPLEMENTATION_PLAN.md` / `docs/RESEARCH_SCOPE.md` (written
2026-08-07, never revised) and the `CARDIOSENTINEL_HANDOFF_ECG{3…10}.md`
session logs at the repo root's parent (freshest: ECG 10, itself written
earlier on 2026-08-21, before the 5 PRs and the commit described above).

`research/phase-3b-classical-baselines` is a real, long-closed branch
(`87b5d39`, remote already deleted). The repository has gone through nine
further phase boundaries since: 3B-2 (B4 architecture selection), P1
(physiology), M1 (patient memory), M2 (contamination-safe update), U1
(calibration/routing), T2 (longitudinal temporal), and three T1 sub-phases
(frozen protocol → canonical harness → execution driver/evaluator, now
"challenge and composition").

`docs/IMPLEMENTATION_PLAN.md`, item by item:

| # | Item | Doc says | Reality |
|---|---|---|---|
| 1 | Dataset ingestion & annotation validation | complete | matches |
| 2 | Signal-processing pipeline | complete | matches |
| 3 | Reproducible baselines (B0–B3) | complete | matches |
| 4 | Patient-adaptive memory | future work | **done** — M1L selected |
| 5 | Physiology-guided model | future work | **done** — P1-B selected |
| 6 | Uncertainty calibration | future work | **done** — U1 Platt selected |
| 7 | Temporal episode reasoning | future work | **partial** — T2 done; T1 in progress, unexecuted |
| 8 | Edge/cloud routing | future work | **done** — U1 selective routing implemented |
| 9 | Edge benchmarking | future work | **partial** — latency/params measured on a benchmark host, not an edge device |
| 10 | Final ablation & external validation | future work | matches — not started |

`docs/RESEARCH_SCOPE.md` still reads *"no approved dataset integration, no
validated labels, no model, no measured performance, no clinical validation,
and no selected edge hardware target"* — every clause except the last two is
now false. Neither doc is wrong about what it asserts; both are silent about
six phases of work that happened after they were written.

**Verdict:** the repository is ahead of every written plan on every axis, by
roughly nine phase boundaries.

## 3. Experiment ladder

| ID | Exists | Status | Git SHA | Metrics | Notes |
|---|---|---|---|---|---|
| B0 | yes | complete, TEST opened | `4f57ba3` | val AUPRC 0.0461 / AUROC 0.500 | constant-prior floor; v3 is canonical |
| B1 | yes | complete, TEST opened | `4f57ba3` | val AUPRC 0.1173 / AUROC 0.790 | signal-only logreg |
| B2 | yes | complete, TEST opened | `4f57ba3` | val AUPRC 0.1640 / AUROC 0.823 | + morphology, logreg |
| B3 | yes | complete, TEST opened | `4f57ba3` | val AUPRC 0.1683 / AUROC 0.836 | morphology + HGB, best classical |
| B4-A (CNN/TCN) | yes, as `B4_raw_compact_cnn_v1` | complete, rejected | `21a38ec` | pooled val AUPRC 0.316 | compact CNN; no TCN exists in-repo. Kept as reference, 87k params |
| B4-B (CNN Transformer) | yes | **selected — official model** | `b27d528` | pooled val AUPRC 0.381 / subj-macro 0.401 | locked threshold 0.8329, 310k params |
| B4-C (CNN SSM) | yes | complete, rejected | `b27d528` | pooled val AUPRC 0.338 | negative result, short-window only; 155k params |
| B4-D (Hybrid) | not found | not started | — | — | only 3 candidates ever scoped (A/B/C) |
| P1 | yes — P1A, P1B | **P1-B selected** | `7e02c22` | files present | physiology fusion beat plain neural head |
| M1 | yes — v1 (2 failures) + v2 (M1S/D/L) | **M1L selected** | `8260b71` | M1S 0.365/0.912, M1D 0.381/0.912, M1L 0.385/0.908 (AUPRC/AUROC) | M1L wins on AUPRC, not AUROC — reads as pre-specified metric |
| M2 | yes — 3 attempts | **M2-G selected** (recovery2) | `cdc3379` | M2-0 vs M2-G closely matched (~0.386/0.911 vs 0.386/0.912) | 2 earlier attempts failed with documented receipts |
| U1 | yes | complete | `233a474` | files present, ECE not re-extracted | Platt calibration + selective routing selected |
| U2 | not found | not started | — | — | no U2 anywhere; U-phase is U1 only |
| T1 | harness + composition root yes, evidence no | **0 canonical attempts** | merged at `64d5fc9` (PR #45) | zero — VALIDATION unread, TEST sealed | full execution wiring now on master; sole remaining gate is PR #39 (needs a rewrite, see "Live flag") |
| T2 | yes — training + one-shot outer validation | **causal_s4d_longitudinal_v1 selected** | `f4759e2` / `b0f189a` | training 0.629/0.972 → 0.640/0.972; outer val 0.00195/0.715 | outer validation is a consumed one-shot artifact |
| E1 | not found | not started | — | — | only a benchmark-host latency measurement exists under B4; `edge/` is an empty package |

Also present: three `cardiosentinel-runs/phase-3b-smoke-*` folders — CI/pipeline
smoke fixtures, not scientific experiments.

## 4. Scientific lock audit

Sampled every `EXPERIMENT_LOCK.json` across B0–T2 (~16 files):

- **Git SHA** — 100%, every lock records one.
- **Dirty state** — 100%, every lock records `"git_dirty": false`.
- **Split SHA** — uniform: `66e25d77b6aaa25502974b4b60667b4c4b24d649bf9493666827c747a385ced7`, B0 through T2.
- **Feature/dataset hash** — uniform: `f18785d520828cb171482926922346dda824c8868ed4b7f9be45897cd71d6eb5` (`ltstdb-baseline-v1`).
- **Seed** — confirmed (`"seed": 2026`) for B4-A/B/C, P1A/B, M1S/D/L. Not found by a top-level key scan for B0–B3, M2, U1, T2 — may be under a different key; needs a manual check, not asserted missing.
- **Config** — no lock exposes a literal `"config"` key; parameters appear inlined directly.
- **Validation metrics** — present for every completed experiment.
- **Test access** — opened only for B0–B3 (v3). Everything B4 onward carries explicit `"test_evidence_used": false` / `"sealed_test_state": "unopened"`. T1 has not read even VALIDATION yet.

## 5. Code maturity

| Layer | Location | Maturity |
|---|---|---|
| Models | `models/` | thin stub — only `baselines.py`; neural architectures actually live under `neural/` |
| Trainers | `neural/*_experiment.py`, `*_development_run.py` | mature — one harness per phase |
| Pipelines | `signal/` (11), `features/` (4), `data/` (8) | mature |
| Inference path | — none found — | **not started** — no `predict()`/`infer()`/`serve()` anywhere; deploying B4-B today means reusing harness-internal scoring code |
| Evaluation framework | `evaluation/` (8 files) | mature, shared across phases |
| `neural/` package | 74 files | has absorbed what `edge/`, `episodes/`, `personalization/`, `uncertainty/` (each an empty `__init__.py`) look like they were meant to hold |
| Test suite | `tests/` | 2,410 passed / 1 skipped (frozen interpreter) as of PR #39; 5 more PRs merged since, count not re-verified here |

**Can T1 canonical execution start immediately?** Mechanically, essentially
all of it now: the 29-stage harness, driver, fold authority, fold evaluation
capability, assembly collaborators, fold evaluator, and — as of PR #45 — the
composition root itself are all merged to master. Procedurally, still no:
the sole remaining gate is PR #39, and, per PR #45's own merged description,
its refusal prose is now factually false and it "needs a rewrite rather than
a rebase" before it can merge safely. It has not been rewritten as of this
refresh.

## 6. Architecture maturity

| Stage | Status | Evidence |
|---|---|---|
| Signal pipeline | done | `signal/`, used by every phase |
| CNN encoder | done | B4-A, also feeds B4-B/C |
| Transformer / SSM | done | B4-B selected (short-window); T2 causal S4D (longitudinal) trained + outer-validated |
| Physiology | done | P1-B selected |
| Patient memory | done | M1L selected |
| Calibration | done | U1 Platt + selective routing |
| Temporal reasoning | partial | T2 (longitudinal) done; T1 (episodic/alerting) harness built, zero executions |
| Edge deployment | not started | `edge/` empty; only benchmark-host latency numbers exist, explicitly not on-device |

T1 doesn't map onto one box above — it's a causal episode/alerting layer
downstream of the model score stream, and it's the actual current frontier.

## 7. Next steps

**A. Immediate:** PR #39 cannot be merged as written — that is now PR #45's
own merged position, not just this audit's. Rewrite or replace #39 so its
refusal prose matches a repository where the composition root already
exists, before anyone merges it.

**B. Required audits:** rewrite PR #39 against current master (`64d5fc9`)
rather than rebasing it; re-run the ECG 9 eight-item authorization checklist
against whatever commit ends up authorized; sync the local checkout and the
local `master` ref, both now behind `origin/master`; the stage-recorder
granularity note and the ECG 3 outer-repo index reconstruction are still
open from earlier handoffs.

**C. Experiments remaining:** T1 canonical execution (0 of 1 attempts used);
final ablation / external validation (plan item 10); genuine edge/on-device
benchmarking (item 9); U2, E1, B4-D as such don't exist — confirm whether
they belong to a different plan version or were never scoped.

**D. Risks:** the single canonical T1 attempt sits behind exactly one gate
now — PR #39 — and that PR's own safety narrative is confirmed stale by its
sibling PR's merged description, not just by this audit; TEST stays sealed
until the T1 execution question is resolved; planning docs were silently
stale enough to actively mislead if trusted over the repo (README.md,
IMPLEMENTATION_PLAN.md, and RESEARCH_SCOPE.md were brought current this
session); seed capture is unconfirmed for B0–B3/M2/U1/T2 locks.

---

_Last refreshed: 2026-08-21, read-only pass against `origin/master` `64d5fc9`
(local checkout `3c8d433`). To refresh, ask Claude to re-run the audit and
rewrite this file — nothing here is meant to be trusted past its own "As of"
line._
