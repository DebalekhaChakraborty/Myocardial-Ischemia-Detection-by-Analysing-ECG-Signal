# Data Split Policy

The split unit is the subject, never an ECG window or record. Every record from
one person must map to exactly one partition. EDB uses the explicitly documented
shared-subject groups; LTSTDB uses the documented record-name relationship in
which records from one subject differ only in their final digit.

Before any V1 model result, the deterministic LTSTDB split balances source-only
subject burden including ischemic/rate episode counts and durations, recording
metadata, and axis/conduction marker counts and presence. All normalized burden
features have equal objective treatment; model outputs and waveform-derived
features are prohibited. The manifest separately records assignment, source
metadata, and generator-code hashes.

Once a test subject list is established it is immutable for that study. Test
subjects must not influence model selection, thresholds, calibration strategy,
or personalization hyperparameters. The reusable leakage validator rejects a
record assignment that differs from its subject assignment.

For final streaming evaluation, any patient adaptation may use only information
strictly preceding the evaluation instant and permitted by the finalized
protocol. Future windows, reference functions, or post-event annotations must
never leak backward into causal inputs. Expert annotations remain ground truth,
not input features.

## Ground-Truth Boundary

Expert episode annotations, `.stf` reference functions, global/local-reference
annotations, expert-corrected ST-deviation values, future episode endpoints, and
future samples may define targets and evaluation intervals only. They must never
be used as predictive inputs, baseline features, personalization state, or
model-selection signals.
