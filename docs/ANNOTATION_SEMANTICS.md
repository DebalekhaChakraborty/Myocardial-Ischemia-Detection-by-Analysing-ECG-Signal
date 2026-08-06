# Annotation Semantics

Ingestion preserves source annotations; it does not create a single
`ischemia: bool` label.

## EDB `.atr`

Upper-case `(ST`, `AST`, and `ST...)` sequences are reference ST-change episodes
with onset, extremum, and end. Lower-case `(st`, `ast`, and `st...)` sequences
are apparent ST deviations associated with positional/axis shift. They are
stored as `axis_shift` / `apparent_st_change`, never silently converted to
ischemia. EDB `NOISE` annotations encode clean, noisy, or unreadable state for
each of two channels and are represented as quality intervals.

EDB ST deviations are relative to an expert reference waveform from the first
30 seconds. The source does not preserve its exact reference beat; this is an
important limitation for future causal feature design.

## LTSTDB `.sta`, `.stb`, `.stc`

Each set is parsed separately. Episode start, extremum, and end markers become
`st_episode` events with subtype `ischemic` or `heart_rate_related`. Significant
ST shifts are retained as axis-related or conduction-related markers; `GRST` and
`LRST` remain reference markers; `noi` is a noise marker; paired `urd` markers
become unreadable intervals. Slow drift and mixed phenomena are not inferred
from annotations that do not explicitly encode them; their interpretation stays
in source-traceable markers and future protocol work.

For both datasets, lead identifiers are validated against WFDB header signal
count. Peak deviation, direction, source extension, definition, and raw WFDB
annotation fields are retained for traceability.

Incomplete onset/peak/end sequences are reported as malformed and are not
silently closed at a record boundary. CardioSentinel never fabricates an event
endpoint or a binary ischemia label from an incomplete expert annotation.
