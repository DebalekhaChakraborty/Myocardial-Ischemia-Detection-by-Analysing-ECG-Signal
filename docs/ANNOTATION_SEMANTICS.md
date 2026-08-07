# Annotation Semantics

Ingestion preserves source annotations; it does not create a single
`ischemia: bool` label.

## EDB `.atr`

Upper-case `(ST`, `AST`, and `ST...)` sequences are reference ST-change episodes
with onset, extremum, and end. Lower-case `(st`, `ast`, and `st...)` sequences
are apparent ST deviations associated with positional/axis shift. They are
stored as `axis_shift` / `apparent_st_change`, never silently converted to
ischemia. EDB `NOISE` annotations encode clean, noisy, or unreadable state for
each of two channels and are represented as quality intervals. The source
amplitude is already in microvolts, so `peak_deviation_uv` retains it unchanged.

The official WFDB mapping is enforced: upper-case ST forms use symbol `s`,
lower-case axis-shift forms use comment symbol `"`, and the source `chan` is 0;
the lead encoded in the auxiliary text is authoritative. Beat, rhythm, button,
tape-slippage, and T-wave annotations are explicitly accounted for but are not
ST targets. EDB quality uses WFDB's two-signal bitmask: unreadable bits take
precedence over noise bits. This covers released subtype `0x23` as noisy lead 0
and unreadable lead 1 even though that redundant combination is omitted from the
short EDB table.

EDB ST deviations are relative to an expert reference waveform from the first
30 seconds. The source does not preserve its exact reference beat; this is an
important limitation for future causal feature design.

## LTSTDB `.sta`, `.stb`, `.stc`

Each set is parsed separately. Episode start, extremum, and end markers become
`st_episode` events with subtype `ischemic` or `heart_rate_related`. Significant
ST shifts are retained as axis-related or conduction-related markers; `GRST` and
`LRST` remain reference markers; `noi` is a noise marker; paired `urd` markers
become unreadable intervals. The verified v1.0.0 syntax is `(st0-100`,
`ast0-200`, `st0-49)`, with `rt` for rate-related episodes, `sst0` for an axis
shift, `sccst0` for a conduction shift, `noi1-187` for noise, `(urd0` / `urd0)`
for an unreadable interval, and `GRST0` / `LRST1+20` for references. LTSTDB
source deviations are already microvolts: `ast0-200` maps to
`peak_deviation_uv == -200.0`, with no scale conversion.

The updated annotation text renders unreadable as `und`, but the released
v1.0.0 `.sta`, `.stb`, and `.stc` streams use `urd`; only the observed release
spelling is accepted. All these forms use WFDB symbol `s`. The auxiliary-note
lead is authoritative because released annotations can retain a different valid
WFDB `chan` value at the same sample; `chan` is preserved and range-validated.
Slow drift and mixed phenomena are not inferred from annotations that do not
explicitly encode them; their interpretation stays in source-traceable markers
and future protocol work.

Benchmark windows preserve source marker context independently of primary target
precedence. Axis/conduction marker vicinity and directly contained point-noise
markers remain traceable on ischemic-positive windows without changing the
binary target, creating a new disease class, inventing point-noise duration, or
automatically excluding the window.

For both datasets, lead identifiers are validated against WFDB header signal
count. Peak deviation, direction, source extension, definition, and raw WFDB
annotation fields are retained for traceability.

Unknown forms are classified and counted. Unknown potentially ST-related forms
are validation errors. Complete source episodes require onset, peak, and end.
Released sequences that are demonstrably source-censored (terminal onset/peak
without an endpoint, or peak/end without an onset) are reported as warnings and
excluded from the canonical event set. CardioSentinel never fabricates an event
endpoint, onset, or binary ischemia label from incomplete expert annotation.
