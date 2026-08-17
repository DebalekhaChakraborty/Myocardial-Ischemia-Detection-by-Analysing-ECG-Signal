"""The ONE canonical U1-v1 DEVELOPMENT invocation.

This is the only public route to a claim-bearing U1 development result. It
consumes the retained M2-G evidence **read-only** and is post-replay by
construction: it never invokes the B4 scorer, never builds a P1 embedding,
never replays M1 or M2, never touches a prototype, never regenerates a score or
a gate, and never reopens M2 arm selection. `m2_replay_firewall()` proves that
from this package's own source rather than asserting it in prose.

**Nothing runs on import**, and `__main__` dispatch sits at the very end of the
file, after every helper the run needs is defined.

**Execution order** (frozen, §3 of the authorization):

1. PRE-CLAIM READINESS -- Git SHA and clean checkout, frozen runtime, the U1
   protocol digest, the M2 retention binder, the M2-G result/lock identity, the
   split identity, both firewalls, and the claim-absence check. **No per-window
   VALIDATION evidence is opened here.**
2. CLAIM -- the one deterministic attempt directory. Only afterwards may the
   permitted M2-G DEVELOPMENT evidence be opened.
3. PRIMARY population proven exactly, then the SATURATION CENSUS -- **before**
   any calibrator is fitted. Outside its frozen bound the run promotes the
   census, writes the stop receipt and STOPS.
4. Twelve LOSO folds; BOTH frozen families fitted in every fold; one OOF
   probability per PRIMARY row per family, exactly once.
5. Family selection from pooled OOF NLL alone.
6. OOF calibration evidence, uncertainty, risk-coverage, `u_star_dev`,
   class-aware routing, subject evidence, the frozen subject bootstrap,
   cold-start strata and challenge routing -- all OOF only.
7. Exactly ONE final deployment calibrator of the already-selected family, then
   `u_star_deploy` as configuration provenance.
8. Persist immutable artifacts and STOP for the human retention review.

**A raised reporting guard is not an infrastructure failure.** The guards fire
only after the scientific evidence exists, so the complete result is persisted
with the flags beside it; no threshold is re-selected, nothing is re-fitted and
nothing is retried.

**One shot.** If the claim directory exists in any state the attempt is
consumed and this stops. There is no recovery identity defined in advance, no
timestamp, no uuid, no random suffix and no automatic retry.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.neural.u1_protocol import (
    U1_CALIBRATION_SUBJECTS,
    U1_CLASSIFICATION_THRESHOLD,
    U1_COLD_START_STRATA,
    U1_FORBIDDEN_CALIBRATION_INPUTS,
    U1_FULL_REPLAY_ROW_COUNT,
    U1_PRIMARY_ROW_COUNT,
    U1_RETAINED_COVERAGE,
    U1ProtocolError,
)

EXECUTION_FLAG: Final = "--execute-canonical-development"
EXPECTED_GIT_SHA_FLAG: Final = "--expected-git-sha"

U1_EXPERIMENT_IDENTITY: Final = "U1_selective_v1"
"""The frozen scientific identity of the one U1 development experiment."""

CANONICAL_RUN_ID: Final = "u1-v1-development"
"""The ONE permitted canonical attempt directory name. Deterministic: no
timestamp, no uuid, no random suffix, and no `recovery1` invented in advance.
If this attempt is ever claimed and fails, execution STOPS FOR HUMAN REVIEW."""

PLANNED_EXECUTION_ORDER: Final = (
    "pre_claim_artifact_readiness",
    "claim_canonical_attempt",
    "open_retained_m2g_evidence_read_only",
    "prove_exact_primary_population",
    "saturation_census_before_any_fit",
    "build_leave_one_subject_out_folds",
    "fit_both_families_in_every_fold",
    "pooled_out_of_fold_probabilities",
    "select_family_from_pooled_oof_nll",
    "out_of_fold_calibration_evidence",
    "uncertainty_from_the_frozen_decision",
    "risk_coverage_evidence",
    "derive_u_star_dev",
    "class_aware_routing_evidence",
    "subject_evidence_and_frozen_bootstrap",
    "cold_start_evidence",
    "challenge_routing_evidence",
    "final_deployment_calibrator",
    "derive_u_star_deploy_for_configuration_only",
    "persist_immutable_u1_artifacts",
    "stop_for_human_retention_review",
)

# Symbols whose presence anywhere in the U1 execution package would mean U1 had
# reached back into the M2 production side. U1 is post-replay: it consumes
# promoted evidence and produces calibration, nothing else.
U1_FORBIDDEN_REPLAY_SYMBOLS: Final = (
    "replay_stream",
    "replay_both_arms",
    "M2StreamState",
    "iter_timeline_streams",
    "load_frozen_m1l_scorer",
    "M2EvidenceStore",
    "canonical_replay_population",
    "build_p1_embedding_cache",
    "execute_canonical_development",
    "read_annotations",
    "read_record",
    "evaluate_locked_test",
    "load_sealed_test",
)

U1_FORBIDDEN_TEST_SYMBOLS: Final = (
    "sealed_test",
    "evaluate_locked_test",
    "load_sealed_test",
    "TEST_ATTEMPT",
)

U1_EXECUTION_MODULES: Final = (
    "u1_protocol.py",
    "u1_calibration.py",
    "u1_evidence_store.py",
    "u1_persistence.py",
    "u1_development_run.py",
)


class U1DevelopmentRunError(RuntimeError):
    """Raised when the canonical U1 development route refuses to proceed."""


# --------------------------------------------------------------------------
# Canonical roots -- the repository's existing conventions, not new ones
# --------------------------------------------------------------------------


def canonical_roots() -> dict[str, Path]:
    """The deterministic inputs and the one output root of the U1 run."""
    from cardiosentinel.neural.m2_gate_derivation import (
        DEFAULT_FEATURE_ROOT,
        DEFAULT_P1_CACHE_ROOT,
        DEFAULT_STREAM_CACHE_ROOT,
    )
    from cardiosentinel.neural.m2_persistence import evidence_workspace
    from cardiosentinel.neural.m2_policy import M2_ARM_GATED
    from cardiosentinel.neural.m2_selection import M2_RETAINED_SUITE_ID
    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    m2_run_root = REPOSITORY_ROOT / "cardiosentinel-runs" / "phase6-m2-development-v1"
    return {
        "m2_run_root": m2_run_root,
        # The retained arm's promoted per-window evidence. M2-0 is deliberately
        # NOT reachable from here: it is control/ablation, never a U1 input.
        "m2g_evidence_root": (
            evidence_workspace(m2_run_root, M2_RETAINED_SUITE_ID) / M2_ARM_GATED
        ),
        "feature_root": Path(DEFAULT_FEATURE_ROOT),
        "p1_cache_root": Path(DEFAULT_P1_CACHE_ROOT),
        "stream_cache_root": Path(DEFAULT_STREAM_CACHE_ROOT),
        "split_manifest": REPOSITORY_ROOT / "protocols" / "splits" / "ltstdb_v1.json",
        "run_root": (
            REPOSITORY_ROOT / "cardiosentinel-runs" / "phase7-u1-development-v1"
        ),
    }


# --------------------------------------------------------------------------
# Firewalls, proven from source rather than asserted
# --------------------------------------------------------------------------


def _u1_source_paths() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in U1_EXECUTION_MODULES)


def _identifiers_in(path: Path) -> set[str]:
    """Every name, attribute and imported symbol mentioned in one module."""
    tree = ast.parse(Path(path).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            found.add(node.name.split(".")[-1])
            if node.asname:
                found.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.update(node.module.split("."))
    return found


def m2_replay_firewall(
    *, forbidden: tuple[str, ...] = U1_FORBIDDEN_REPLAY_SYMBOLS
) -> dict[str, Any]:
    """Prove from source that U1 cannot invoke any M2 production path.

    An identifier check on the package's own AST, not a runtime hope: if a
    future edit imported the scorer, the replay or the evidence writer, this
    refuses before anything is claimed. `U1_FORBIDDEN_REPLAY_SYMBOLS` is quoted
    in this module as data, so the scan deliberately ignores that one
    definition and looks only at how the modules actually use names.
    """
    violations: dict[str, list[str]] = {}
    for path in _u1_source_paths():
        used = _identifiers_in(path)
        hits = sorted(symbol for symbol in forbidden if symbol in used)
        if hits:
            violations[path.name] = hits
    if violations:
        raise U1DevelopmentRunError(
            f"The U1 execution package references M2 production symbols "
            f"{violations}. U1 is post-replay: it consumes promoted evidence "
            "read-only. It never replays M1 or M2, never invokes the B4 "
            "scorer, never regenerates a score or a gate and never reopens M2 "
            "arm selection."
        )
    return {
        "firewall_class": "u1_m2_replay_firewall",
        "checked_modules": [path.name for path in _u1_source_paths()],
        "forbidden_symbols": list(forbidden),
        "violations": {},
        "m2_replay_invoked": False,
        "m2_rerun_performed": False,
        "b4_scorer_invoked": False,
        "p1_embeddings_constructed": False,
        "prototypes_modified": False,
        "gates_regenerated": False,
        "m2_arm_selection_reopened": False,
        "checked_by": "abstract_syntax_tree_identifier_scan",
    }


def assert_test_firewall(split_manifest: Path) -> dict[str, Any]:
    """Prove TEST cannot enter U1 -- by refusal, never by silent filtering."""
    from cardiosentinel.neural.m2_execution import (
        M2ExecutionError,
        require_canonical_development_partition,
    )
    from cardiosentinel.neural.u1_protocol import (
        require_calibration_subjects,
        validate_against_frozen_split,
    )

    split = validate_against_frozen_split(Path(split_manifest))
    manifest = json.loads(Path(split_manifest).read_text())
    test_subjects = sorted(manifest["partitions"]["test"]["subjects"])
    if not test_subjects:
        raise U1DevelopmentRunError(
            "The split manifest names no TEST subjects; the firewall cannot be "
            "proven against an empty set."
        )
    for subject in test_subjects:
        try:
            require_calibration_subjects([subject])
        except U1ProtocolError:
            continue
        raise U1DevelopmentRunError(
            f"The calibration subject firewall accepted the TEST subject "
            f"{subject!r}; refusing to run."
        )
    for forbidden in ("test", "TEST", " test "):
        try:
            require_canonical_development_partition(forbidden)
        except M2ExecutionError:
            continue
        raise U1DevelopmentRunError(
            f"The partition firewall accepted {forbidden!r}; refusing to run."
        )

    violations: dict[str, list[str]] = {}
    for path in _u1_source_paths():
        used = _identifiers_in(path)
        hits = sorted(symbol for symbol in U1_FORBIDDEN_TEST_SYMBOLS if symbol in used)
        if hits:
            violations[path.name] = hits
    if violations:
        raise U1DevelopmentRunError(
            f"The U1 execution package references sealed-TEST symbols {violations}."
        )
    return {
        "firewall_class": "u1_test_firewall",
        **split,
        "test_subject_count": len(test_subjects),
        "test_subjects_refused_not_filtered": True,
        "test_path_resolved": False,
        "test_score_read": False,
        "test_label_read": False,
        "test_prediction_read": False,
        "test_metric_computed": False,
        "sealed_test_result_reopened": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


def calibration_input_firewall() -> dict[str, Any]:
    """The only calibration input is the persisted `score` column.

    The G4 normal-evidence quantity is a memory-admission gate, not classifier
    confidence, and is never calibrated, rescaled or routed on.
    """
    from cardiosentinel.neural.m2_evidence_store import ROW_EVIDENCE_COLUMNS

    if "score" not in ROW_EVIDENCE_COLUMNS:
        raise U1DevelopmentRunError(
            "The frozen M2 evidence schema carries no `score` column; the only "
            "permitted U1 calibration input is absent."
        )
    intruders = sorted(
        name for name in U1_FORBIDDEN_CALIBRATION_INPUTS if name in ROW_EVIDENCE_COLUMNS
    )
    if intruders:
        raise U1DevelopmentRunError(
            f"The M2 evidence schema exposes forbidden calibration inputs "
            f"{intruders}; U1 must never calibrate the G4 admission quantity."
        )
    return {
        "firewall_class": "u1_calibration_input_firewall",
        "calibration_input_field": "score",
        "calibration_input_semantics": (
            "uncalibrated sigmoid model score; not calibrated probability"
        ),
        "forbidden_calibration_inputs": list(U1_FORBIDDEN_CALIBRATION_INPUTS),
        "g4_normal_evidence_calibrated": False,
        "m2_0_used_as_calibration_input": False,
    }


# --------------------------------------------------------------------------
# PRE-CLAIM readiness: everything provable WITHOUT per-window evidence
# --------------------------------------------------------------------------


def require_expected_git_sha(expected_git_sha: str | None) -> str:
    """HEAD must equal the human-authorized SHA, on a clean checkout."""
    from cardiosentinel.data.provenance import git_provenance
    from cardiosentinel.neural.patient_memory import REPOSITORY_ROOT

    if not expected_git_sha:
        raise U1DevelopmentRunError(
            f"{EXPECTED_GIT_SHA_FLAG} is required. The canonical U1 development "
            "run executes only against the exact human-reviewed master SHA, "
            "which is known only after the activation PR is merged and verified."
        )
    expected = str(expected_git_sha).strip().lower()
    provenance = git_provenance(REPOSITORY_ROOT)
    actual = str(provenance["git_sha"]).lower()
    if provenance["git_dirty"]:
        raise U1DevelopmentRunError(
            "Canonical U1 development evidence requires a clean Git checkout; "
            "the working tree is dirty. No data was opened."
        )
    if actual != expected:
        raise U1DevelopmentRunError(
            f"HEAD is {actual}, but the human authorization names {expected}. "
            "Execution stops BEFORE any data access; nothing was opened and no "
            "attempt was consumed."
        )
    return actual


def m2g_input_identity(roots: dict[str, Path]) -> dict[str, Any]:
    """Prove the retained M2-G identity from its promoted artifacts.

    Static frozen metadata only: the suite result, the arm result and the lock.
    No per-window row is read here, so this is safely pre-claim.
    """
    from cardiosentinel.neural.m2_persistence import (
        ARM_RESULT_NAME,
        EXPERIMENT_LOCK_NAME,
        arm_experiment_id,
    )
    from cardiosentinel.neural.m2_policy import M2_ARM_GATED
    from cardiosentinel.neural.m2_selection import (
        M2_RETAINED_SUITE_ID,
        validate_retained_m2_arm,
    )

    retention = validate_retained_m2_arm(Path(roots["m2_run_root"]))
    arm_dir = Path(roots["m2_run_root"]) / arm_experiment_id(
        M2_RETAINED_SUITE_ID, M2_ARM_GATED
    )
    arm_result_path = arm_dir / ARM_RESULT_NAME
    arm_result = json.loads(arm_result_path.read_text())
    arm_lock = json.loads((arm_dir / EXPERIMENT_LOCK_NAME).read_text())
    primary = arm_result["primary_evaluation_population_identity"]
    challenge = arm_result["challenge_evaluation_population_identity"]
    replay = arm_result["replay_population_identity"]
    if int(primary["row_count"]) != U1_PRIMARY_ROW_COUNT:
        raise U1DevelopmentRunError(
            f"The retained PRIMARY population holds {primary['row_count']} rows, "
            f"not the frozen {U1_PRIMARY_ROW_COUNT}."
        )
    if int(replay["row_count"]) != U1_FULL_REPLAY_ROW_COUNT:
        raise U1DevelopmentRunError(
            f"The retained FULL REPLAY population holds {replay['row_count']} "
            f"rows, not the frozen {U1_FULL_REPLAY_ROW_COUNT}."
        )
    if arm_result.get("test_accessed") is not False:
        raise U1DevelopmentRunError("The retained M2-G result records TEST access.")

    # The M1 stream cache U1 reads the per-row recording-age strata from is
    # already bound by M2, in two independent places. Both are carried, and a
    # disagreement between them stops the run rather than picking one.
    lock_cache = arm_lock.get("stream_cache_sha256")
    replay_cache = replay.get("stream_cache_sha256")
    if not lock_cache or lock_cache != replay_cache:
        raise U1DevelopmentRunError(
            f"The retained M2-G lock binds stream_cache_sha256 {lock_cache!r} "
            f"while its replay population identity binds {replay_cache!r}. The "
            "exact stream-cache identity the cold-start strata depend on cannot "
            "be established. STOP FOR HUMAN REVIEW."
        )
    return {
        "identity_class": "u1_m2g_input_identity",
        "retention": retention,
        "retained_arm": M2_ARM_GATED,
        "control_arm_is_calibration_input": False,
        "primary_population_identity": primary,
        "challenge_population_identity": challenge,
        "full_replay_population_identity": replay,
        "m2g_evidence_store_identity": arm_result["evidence_store_identity"],
        "m2g_experiment_lock_sha256": arm_lock["experiment_lock_sha256"],
        "stream_cache_sha256": lock_cache,
        "stream_cache_identity_source": (
            "m2g_experiment_lock.stream_cache_sha256, cross-checked against "
            "m2g_arm_result.replay_population_identity.stream_cache_sha256"
        ),
        # The frozen recording-age strata M2 already reported. U1 introduces no
        # cold-start rule of its own, so its strata must reproduce these counts
        # exactly -- which is how a drifted stream cache is caught rather than
        # silently re-stratifying the population.
        "cold_start_strata_window_counts": {
            stratum: int(entry["window_count"])
            for stratum, entry in arm_result["cold_start_evidence"]["strata"].items()
        },
        "read_only": True,
    }


def require_m2g_evidence_store_lineage(
    observed: dict[str, Any], frozen: dict[str, Any]
) -> dict[str, Any]:
    """A self-consistent store is not enough: it must be THE frozen store.

    The retained M2-G arm result already carries the store's own manifest
    verbatim, so the comparison is against that existing canonical identity --
    no second digest scheme is invented here. A store that validates perfectly
    against its own manifest but differs from the one the frozen arm result
    binds is refused: no fit, no regeneration, no replay, no repair.
    """
    from cardiosentinel.neural.integrity import canonical_sha256

    if not isinstance(frozen, dict) or not frozen:
        raise U1DevelopmentRunError(
            "The retained M2-G arm result binds no evidence-store identity; the "
            "opened store cannot be authenticated. STOP FOR HUMAN REVIEW."
        )
    observed_identity = canonical_sha256(observed)
    frozen_identity = canonical_sha256(frozen)
    if observed_identity != frozen_identity:
        differing = sorted(
            key
            for key in set(observed) | set(frozen)
            if observed.get(key) != frozen.get(key)
        )
        raise U1DevelopmentRunError(
            "The observed M2-G evidence-store identity does not equal the "
            "identity bound by the frozen retained arm result: observed "
            f"content_sha256 {observed.get('content_sha256')!r} against frozen "
            f"{frozen.get('content_sha256')!r}, differing fields {differing}. "
            "The store is internally consistent but it is not the frozen store. "
            "No calibrator is fitted, nothing is regenerated, no M2 replay is "
            "performed and nothing is repaired. STOP FOR HUMAN REVIEW."
        )
    return {
        "lineage_class": "u1_m2g_evidence_store_lineage",
        "identity_source": "m2g_arm_result.evidence_store_identity",
        "content_sha256": observed["content_sha256"],
        "row_evidence_sha256": observed["row_evidence_sha256"],
        "schema": observed["schema"],
        "arm": observed["arm"],
        "row_count": observed["row_count"],
        "canonical_identity_sha256": observed_identity,
        "self_consistent": True,
        "matches_frozen_arm_result_identity": True,
        "second_digest_scheme_introduced": False,
    }


def require_population_identity_lineage(
    observed: dict[str, Any], frozen: dict[str, Any], *, name: str
) -> dict[str, Any]:
    """Every field the observed authority issues must match the M2-bound one.

    The frozen bundle identity in the M2-G arm result is the authority's own
    identity payload plus a handful of evaluation-side keys, so requiring every
    OBSERVED key to be present and equal in the frozen identity compares the
    complete scientific identity -- population, partition, authority, row and
    class counts, ordered stable-id digest, and the upstream cache or selection
    digest -- rather than a single field. Nothing is normalised away.
    """
    if not isinstance(observed, dict) or not isinstance(frozen, dict) or not frozen:
        raise U1DevelopmentRunError(
            f"The {name} identity cannot be cross-linked to the retained M2-G "
            "result. STOP FOR HUMAN REVIEW."
        )
    missing = sorted(key for key in observed if key not in frozen)
    differing = sorted(
        key for key in observed if key in frozen and observed[key] != frozen[key]
    )
    if missing or differing:
        raise U1DevelopmentRunError(
            f"The observed {name} identity does not match the identity bound by "
            f"the frozen retained M2-G arm result: fields absent upstream "
            f"{missing}, fields differing {differing}. U1 calibrates exactly the "
            "frozen population; a disagreement is never normalised away."
        )
    return {
        "lineage_class": f"u1_{name}_identity_lineage",
        "identity_source": f"m2g_arm_result.{name}_identity",
        "compared_fields": sorted(observed),
        "compared_field_count": len(observed),
        "matches_frozen_m2g_identity": True,
    }


def require_stream_cache_identity(
    observed_sha256: Any, expected_sha256: str
) -> dict[str, Any]:
    """The cold-start bins must come from THE stream cache M2 replayed.

    Aggregate stratum counts cannot detect a permutation of bin assignments
    across rows; the cache's own content-bound digest can. The expected value
    is the one M2 already froze -- it is never derived here, and it is never
    re-derived by inspecting VALIDATION rows.
    """
    observed = str(observed_sha256)
    if observed != str(expected_sha256):
        raise U1DevelopmentRunError(
            f"The M1 stream cache supplying the per-row recording-age strata "
            f"digests to {observed!r}, not the {expected_sha256!r} the frozen "
            "retained M2-G evidence binds. Identical stratum totals do not make "
            "it the same artifact. No cache is derived from source, no bin is "
            "regenerated and M1 is not replayed. STOP FOR HUMAN REVIEW."
        )
    return {
        "provenance_class": "u1_cold_start_stream_cache_identity",
        "stream_cache_sha256": observed,
        "identity_source": "m2g_experiment_lock.stream_cache_sha256",
        "matches_frozen_m2_identity": True,
        "cache_derived_from_source": False,
        "m1_replayed": False,
        "bins_regenerated": False,
    }


def preflight(
    *,
    expected_git_sha: str | None,
    roots: dict[str, Path] | None = None,
    loaders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity checks that open no per-window VALIDATION evidence at all."""
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        require_runtime_identity,
    )
    from cardiosentinel.neural.u1_persistence import runtime_provenance
    from cardiosentinel.neural.u1_protocol import (
        u1_protocol_identity,
        validate_u1_protocol_document,
    )

    resolved = dict(roots or canonical_roots())
    git_sha = require_expected_git_sha(expected_git_sha)
    start = require_runtime_identity(
        EnforcementPoint.START, detail="u1_development_preflight"
    )
    protocol_sha = validate_u1_protocol_document()
    identity = u1_protocol_identity()
    replay_firewall = m2_replay_firewall()
    firewall = assert_test_firewall(resolved["split_manifest"])
    inputs = calibration_input_firewall()
    m2g = (loaders or {}).get("m2g_input_identity", m2g_input_identity)(resolved)
    return {
        "preflight_class": "u1_v1_canonical_development_preflight",
        "experiment_identity": U1_EXPERIMENT_IDENTITY,
        "canonical_run_id": CANONICAL_RUN_ID,
        "git_sha": git_sha,
        "git_dirty": False,
        "runtime_identity": start.as_dict(),
        "runtime_provenance": runtime_provenance(),
        "u1_protocol_sha256": protocol_sha,
        "u1_protocol_identity": identity,
        "m2_replay_firewall": replay_firewall,
        "test_firewall": firewall,
        "calibration_input_firewall": inputs,
        "m2g_input_identity": m2g,
        "per_window_evidence_opened": False,
        "calibrator_fitting_started": False,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


def pre_claim_readiness(
    *,
    expected_git_sha: str | None,
    roots: dict[str, Path],
    loaders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete readiness gate, plus the claim-absence check."""
    from cardiosentinel.neural.u1_persistence import require_unclaimed_u1_attempt

    readiness = preflight(
        expected_git_sha=expected_git_sha, roots=roots, loaders=loaders
    )
    readiness["readiness_class"] = "u1_v1_pre_claim_artifact_readiness"
    readiness["claim_check"] = require_unclaimed_u1_attempt(
        roots["run_root"], CANONICAL_RUN_ID
    )
    return readiness


# --------------------------------------------------------------------------
# Post-claim inputs: the permitted M2-G DEVELOPMENT evidence, read-only
# --------------------------------------------------------------------------


def load_m2g_score_table(evidence_root: Path) -> Any:
    """Read the retained arm's promoted per-window scores. Read-only.

    The store's own manifest is re-validated with the frozen M2 validator, so a
    mutated or truncated artifact is refused rather than silently consumed. No
    row is regenerated: if the evidence cannot be consumed faithfully the run
    stops.
    """
    from cardiosentinel.neural.m2_evidence_store import (
        ROW_EVIDENCE_NAME,
        STORE_MANIFEST_NAME,
        M2ScoreTable,
        validate_evidence_store_manifest,
    )
    from cardiosentinel.neural.m2_policy import M2_ARM_GATED

    root = Path(evidence_root)
    manifest_path = root / STORE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise U1DevelopmentRunError(
            f"No promoted M2 evidence store manifest at {manifest_path}. The "
            "required per-window evidence cannot be consumed faithfully; it is "
            "never regenerated. STOP FOR HUMAN REVIEW."
        )
    manifest = validate_evidence_store_manifest(
        json.loads(manifest_path.read_text()), root=root
    )
    if manifest["arm"] != M2_ARM_GATED:
        raise U1DevelopmentRunError(
            f"The evidence store belongs to arm {manifest['arm']!r}. Only the "
            f"retained {M2_ARM_GATED} is a U1 calibration input; the control "
            "arm never is."
        )
    with np.load(root / ROW_EVIDENCE_NAME, allow_pickle=False) as payload:
        table = M2ScoreTable(
            arm=str(manifest["arm"]),
            stable_ids=np.asarray(payload["stable_id"]),
            scores=np.asarray(payload["score"], dtype=np.float64),
            scored=np.asarray(payload["scored"], dtype=np.bool_),
        )
    if table.row_count != int(manifest["row_count"]):
        raise U1DevelopmentRunError(
            f"The evidence store holds {table.row_count} rows, not the "
            f"{manifest['row_count']} its manifest binds."
        )
    return table, manifest


def load_cold_start_bins(stream_cache_root: Path, stable_ids: tuple[str, ...]):
    """Per-row recording-age strata, from the frozen persisted M1 stream cache.

    Label-free by construction and read from a persisted array -- this reads a
    frozen artifact, it does not replay a stream. The repository's existing
    `load_stream_store` is the reader, so the cache's manifest self-digest and
    every array's content digest are re-verified by already-reviewed code
    before a single bin is joined.

    Returns the bins AND the cache's own identity. The identity check itself
    lives in the caller, so an injected loader cannot bypass it.
    """
    from cardiosentinel.neural.m1_experiment import load_stream_store
    from cardiosentinel.neural.m1_store import COLD_START_BIN_FILE, STABLE_ID_FILE

    store, manifest = load_stream_store(Path(stream_cache_root), "validation")
    try:
        cache_ids = np.asarray(store.array(STABLE_ID_FILE))
        bins = np.asarray(store.array(COLD_START_BIN_FILE))
    finally:
        store.close()
    index = {str(value): position for position, value in enumerate(cache_ids)}
    missing = [sid for sid in stable_ids if sid not in index]
    if missing:
        raise U1DevelopmentRunError(
            f"{len(missing)} PRIMARY rows have no persisted recording-age "
            f"stratum, beginning {missing[:3]}."
        )
    joined = np.asarray([str(bins[index[sid]]) for sid in stable_ids], dtype=np.str_)
    return joined, str(manifest["stream_cache_sha256"])


# --------------------------------------------------------------------------
# §9 Folds, §10 fitting, §11 selection
# --------------------------------------------------------------------------


def build_fold_manifest(
    *, subject_ids: np.ndarray, fold_assignment_sha256: str
) -> dict[str, Any]:
    """The frozen LOSO assignment, with real per-fold row counts."""
    from cardiosentinel.neural.u1_protocol import assign_calibration_folds

    folds = assign_calibration_folds()
    entries = []
    for fold in folds:
        held_out = subject_ids == fold.held_out_subject
        entries.append(
            {
                "fold_index": fold.fold_index,
                "held_out_subject": fold.held_out_subject,
                "fit_subjects": list(fold.fit_subjects),
                "fit_subject_count": len(fold.fit_subjects),
                "fit_row_count": int(np.count_nonzero(~held_out)),
                "evaluation_row_count": int(np.count_nonzero(held_out)),
            }
        )
    empty = [entry for entry in entries if entry["evaluation_row_count"] == 0]
    if empty:
        raise U1DevelopmentRunError(
            f"{len(empty)} folds hold out a subject with no PRIMARY rows: "
            f"{[entry['held_out_subject'] for entry in empty]}."
        )
    return {
        "manifest_class": "u1_fold_manifest",
        "fold_design": "leave_one_subject_out",
        "fold_count": len(entries),
        "fold_assignment_basis": "frozen_subject_identity_ascending_only",
        "fold_assignment_sha256": str(fold_assignment_sha256),
        "performance_dependent_assignment": False,
        "folds": entries,
    }


def fit_out_of_fold(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    fold_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fit BOTH frozen families in every fold and emit complete OOF coverage.

    Every PRIMARY row receives exactly one probability per family, from a
    calibrator that never saw that row's subject. A gap or a duplicate is
    fatal: no row is scored twice and none is left unscored.
    """
    from cardiosentinel.neural.u1_calibration import (
        FAMILY_PLATT,
        FAMILY_TEMPERATURE,
        fit_calibrator,
    )

    total = int(labels.shape[0])
    probabilities = {
        family: np.full(total, np.nan, dtype=np.float64)
        for family in (FAMILY_PLATT, FAMILY_TEMPERATURE)
    }
    assigned = np.zeros(total, dtype=np.int64)
    fold_index = np.full(total, -1, dtype=np.int64)
    calibrators: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for entry in fold_manifest["folds"]:
        held_out = subject_ids == entry["held_out_subject"]
        fit_rows = ~held_out
        record = dict(entry)
        record["fitted"] = {}
        calibrators[entry["fold_index"]] = {}
        for family in (FAMILY_PLATT, FAMILY_TEMPERATURE):
            calibrator = fit_calibrator(
                logits=logits[fit_rows],
                labels=labels[fit_rows],
                family=family,
                fit_subjects=entry["fit_subjects"],
            )
            probabilities[family][held_out] = calibrator.apply_to_logits(
                logits[held_out]
            )
            calibrators[entry["fold_index"]][family] = calibrator
            record["fitted"][family] = calibrator.as_dict()
        assigned[held_out] += 1
        fold_index[held_out] = entry["fold_index"]
        records.append(record)

    if int(np.count_nonzero(assigned != 1)):
        gaps = int(np.count_nonzero(assigned == 0))
        duplicates = int(np.count_nonzero(assigned > 1))
        raise U1DevelopmentRunError(
            f"Out-of-fold coverage is not exact: {gaps} rows received no "
            f"calibrated probability and {duplicates} received more than one. "
            "Pooled OOF evidence must cover every PRIMARY row exactly once."
        )
    for family, values in probabilities.items():
        if not bool(np.all(np.isfinite(values))):
            raise U1DevelopmentRunError(
                f"The {family} OOF probabilities contain a non-finite value."
            )
    return {
        "probabilities": probabilities,
        "fold_index": fold_index,
        "calibrators": calibrators,
        "records": records,
    }


def pooled_oof_calibration(
    *,
    labels: np.ndarray,
    stable_ids: tuple[str, ...],
    scores: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Calibration evidence for both families and the uncalibrated baseline."""
    from cardiosentinel.neural.u1_calibration import U1_FAMILIES, calibration_evidence

    evidence = {
        family: calibration_evidence(
            labels=labels,
            probabilities=probabilities[family],
            stable_ids=stable_ids,
            name=family,
            is_out_of_fold=True,
        )
        for family in U1_FAMILIES
    }
    evidence["uncalibrated_baseline"] = calibration_evidence(
        labels=labels,
        probabilities=scores,
        stable_ids=stable_ids,
        name="uncalibrated_raw_score",
        is_out_of_fold=False,
    )
    evidence["uncalibrated_baseline"]["baseline_semantics"] = (
        "the raw persisted M2-G score treated as a probability; it is a "
        "reference, not an out-of-fold artifact"
    )
    return {
        "artifact_class": "u1_oof_development_calibration",
        "development_evidence": True,
        "out_of_fold": True,
        "calibrator_count": 12,
        "families": evidence,
        "comparator_is_approximate": True,
        "true_logit_temperature_scaling_performed": False,
    }


# --------------------------------------------------------------------------
# §12 Cold start and §12/§21 challenge routing
# --------------------------------------------------------------------------


def cold_start_evidence(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    decisions: np.ndarray,
    uncertainties: np.ndarray,
    stable_ids: tuple[str, ...],
    cold_start_bins: np.ndarray,
    u_star: float,
) -> dict[str, Any]:
    """OOF calibration and selective behaviour by the frozen recording-age strata.

    No new threshold, no post-hoc repair, no retention change. A stratum whose
    discrimination-dependent quantities are undefined reports counts and leaves
    them null.
    """
    from cardiosentinel.neural.u1_calibration import (
        accepted_evidence,
        brier_score,
        negative_log_likelihood,
    )

    strata: dict[str, Any] = {}
    for stratum in U1_COLD_START_STRATA:
        mask = cold_start_bins == stratum
        count = int(np.count_nonzero(mask))
        positives = int(np.count_nonzero(labels[mask] == 1))
        entry: dict[str, Any] = {
            "window_count": count,
            "positive_count": positives,
            "negative_count": count - positives,
            "confidence_interval_reported": False,
        }
        if count:
            entry["routing"] = accepted_evidence(
                labels=labels[mask],
                decisions=decisions[mask],
                uncertainties=uncertainties[mask],
                accepted=uncertainties[mask] <= float(u_star),
            )
            entry["brier"] = brier_score(labels[mask], probabilities[mask])
            entry["negative_log_likelihood"] = negative_log_likelihood(
                labels[mask], probabilities[mask]
            )
        if positives <= 1:
            entry["discrimination_dependent_quantities"] = (
                "undefined_single_or_no_positive"
            )
        strata[stratum] = entry
    return {
        "evidence_class": "u1_cold_start_evidence",
        "out_of_fold": True,
        "u_star": float(u_star),
        "strata": strata,
        "cold_start_threshold_defined": False,
        "post_hoc_cold_start_repair_performed": False,
        "retained_routing_point_altered_by_stratum_result": False,
        "inherited_limitation": (
            "M2's zero sensitivity in the 0-5 minute stratum at the frozen "
            "thresholds is inherited; U1 introduces no cold-start threshold and "
            "performs no cold-start repair"
        ),
    }


def challenge_routing_evidence(
    *,
    target_families: np.ndarray,
    subject_ids: np.ndarray,
    decisions: np.ndarray,
    uncertainties: np.ndarray,
    u_star: float,
) -> dict[str, Any]:
    """Challenge routing at `u_star_dev`, per subset, never merged into PRIMARY.

    Challenge rows carry no binary label by construction, so a "false positive"
    is a positive frozen decision on a confounder window -- exactly the frozen
    metrics-protocol convention. Conduction change is descriptive `FP / N` with
    one contributing subject: no interval, never a selection input.
    """
    from cardiosentinel.evaluation.protocol import challenge_evidence_policy

    accepted = uncertainties <= float(u_star)
    report: dict[str, Any] = {}
    for challenge in ("rate_related", "axis_shift", "conduction_change"):
        policy = challenge_evidence_policy(challenge)
        selected = target_families == policy.target_family
        denominator = int(np.count_nonzero(selected))
        accepted_here = selected & accepted
        accepted_count = int(np.count_nonzero(accepted_here))
        false_positives = int(np.count_nonzero(accepted_here & decisions))
        subjects = sorted(set(subject_ids[selected].tolist()))
        report[challenge] = {
            "evidence_level": policy.evidence_level,
            "denominator": denominator,
            "contributing_subject_count": len(subjects),
            "accepted_count": accepted_count,
            "escalated_count": denominator - accepted_count,
            "escalation_fraction": (
                None if denominator == 0 else 1.0 - accepted_count / denominator
            ),
            "accepted_false_positive_count": false_positives,
            "accepted_false_positive_rate": (
                None if accepted_count == 0 else false_positives / accepted_count
            ),
            "all_windows_false_positive_count": int(
                np.count_nonzero(selected & decisions)
            ),
            "bootstrap_interval_reported": False,
            "is_selection_input": False,
            "is_headline_metric": policy.is_headline_metric,
            "merged_into_primary": False,
            "binary_labels_invented": False,
        }
    return {
        "evidence_class": "u1_challenge_routing_evidence",
        "out_of_fold": True,
        "u_star": float(u_star),
        "subsets": report,
        "merged_into_primary_denominator": False,
    }


# --------------------------------------------------------------------------
# The canonical orchestration
# --------------------------------------------------------------------------


def require_canonical_run_id(run_id: str) -> str:
    """Production canonical execution is ALWAYS the one canonical attempt."""
    if str(run_id) != CANONICAL_RUN_ID:
        raise U1DevelopmentRunError(
            f"The canonical U1 development attempt is {CANONICAL_RUN_ID!r}; "
            f"{run_id!r} is refused. A second attempt name would let a consumed "
            "canonical attempt be re-run under another directory. Nothing was "
            "claimed, created or opened."
        )
    return CANONICAL_RUN_ID


def execute_canonical_u1_development(
    *,
    expected_git_sha: str | None,
    execute: bool = False,
    _roots: dict[str, Path] | None = None,
    _loaders: dict[str, Any] | None = None,
    _run_id: str | None = None,
) -> dict[str, Any]:
    """The canonical U1 development run.

    Without `execute=True` this performs the identity preflight and returns the
    plan; it opens no per-window evidence and consumes no attempt.

    There is deliberately **no public run-id parameter**: production always runs
    the one `CANONICAL_RUN_ID`. `_roots`, `_loaders` and `_run_id` are private
    TEST-ONLY seams, absent from the CLI and from the public scientific
    contract, and `_run_id` still cannot bypass a consumed attempt because the
    claim-absence check refuses any attempt whose paths already exist.
    """
    roots = dict(_roots or canonical_roots())
    if not execute:
        plan = preflight(
            expected_git_sha=expected_git_sha,
            roots=roots,
            loaders=dict(_loaders or {}),
        )
        plan["planned_execution_order"] = list(PLANNED_EXECUTION_ORDER)
        plan["executed"] = False
        return plan
    run_id = (
        require_canonical_run_id(CANONICAL_RUN_ID) if _run_id is None else str(_run_id)
    )
    return _run(
        expected_git_sha=expected_git_sha,
        run_id=run_id,
        roots=roots,
        loaders=dict(_loaders or {}),
    )


def _run(
    *,
    expected_git_sha: str | None,
    run_id: str,
    roots: dict[str, Path],
    loaders: dict[str, Any],
) -> dict[str, Any]:
    """The frozen execution order, end to end."""
    from cardiosentinel.neural.runtime_sentinel import (
        EnforcementPoint,
        RuntimeIntegrityRecord,
        require_runtime_identity,
    )
    from cardiosentinel.neural.u1_persistence import (
        DEPLOYMENT_CALIBRATOR_NAME,
        FAMILY_SELECTION_NAME,
        FOLD_MANIFEST_NAME,
        OOF_CALIBRATION_NAME,
        OOF_RESULT_NAME,
        RESULT_CLASS,
        SATURATION_CENSUS_NAME,
        claim_u1_run_directory,
        finalize_and_promote_u1_result,
        promote_component,
        record_saturation_stop,
        record_u1_attempt_failure,
    )

    stage = "pre_claim_artifact_readiness"
    readiness = pre_claim_readiness(
        expected_git_sha=expected_git_sha, roots=roots, loaders=loaders
    )

    stage = "claim_canonical_attempt"
    runtime = RuntimeIntegrityRecord()
    require_runtime_identity(
        EnforcementPoint.START, record=runtime, detail=f"u1_development:{run_id}"
    )
    claimed = claim_u1_run_directory(roots["run_root"], run_id, runtime=runtime)
    exposure = {
        "per_window_evidence_opened": False,
        "calibrator_fitting_started": False,
        "u1_metrics_produced": False,
        "routing_threshold_derived": False,
        "deployment_calibrator_fitted": False,
        "test_accessed": False,
    }
    try:
        return _run_after_claim(
            readiness=readiness,
            claimed=claimed,
            runtime=runtime,
            roots=roots,
            loaders=loaders,
            exposure=exposure,
            names={
                "census": SATURATION_CENSUS_NAME,
                "folds": FOLD_MANIFEST_NAME,
                "calibration": OOF_CALIBRATION_NAME,
                "selection": FAMILY_SELECTION_NAME,
                "oof_result": OOF_RESULT_NAME,
                "deployment": DEPLOYMENT_CALIBRATOR_NAME,
                "result_class": RESULT_CLASS,
            },
            promote=promote_component,
            saturation_stop=record_saturation_stop,
            finalize=finalize_and_promote_u1_result,
        )
    except BaseException as error:
        record_u1_attempt_failure(
            roots["run_root"],
            claimed,
            exception=error,
            stage=exposure.get("stage", stage),
            exposure=exposure,
            runtime=runtime,
        )
        raise


def _run_after_claim(
    *,
    readiness: dict[str, Any],
    claimed: Any,
    runtime: Any,
    roots: dict[str, Path],
    loaders: dict[str, Any],
    exposure: dict[str, Any],
    names: dict[str, str],
    promote: Any,
    saturation_stop: Any,
    finalize: Any,
) -> dict[str, Any]:
    """Everything the claim authorises, in the frozen order."""
    from cardiosentinel.neural.u1_calibration import (
        FAMILY_PLATT,
        FAMILY_TEMPERATURE,
        binary_entropy,
        derive_routing_threshold,
        derive_u_star_dev,
        fit_calibrator,
        frozen_decisions,
        negative_log_likelihood,
        prove_decision_equivalence,
        recover_logits,
        risk_coverage_curve,
        routing_guards,
        saturation_census,
        select_calibrator_family,
        subject_bootstrap,
        subject_level_evidence,
        uncertainty_from_decision,
    )
    from cardiosentinel.neural.u1_evidence_store import (
        U1_STORE_MANIFEST_NAME,
        write_u1_evidence_store,
    )
    from cardiosentinel.neural.u1_persistence import (
        u1_evidence_workspace,
    )
    from cardiosentinel.neural.u1_protocol import (
        U1_CLAMP_DELTA,
        assign_calibration_folds,
        fold_assignment_digest,
        require_calibration_subjects,
    )

    def _use(name: str, default: Any) -> Any:
        return loaders.get(name, default)

    # -- 3. Permitted M2-G DEVELOPMENT evidence, read-only. ------------------
    exposure["stage"] = "open_retained_m2g_evidence_read_only"
    exposure["per_window_evidence_opened"] = True
    frozen_inputs = readiness["m2g_input_identity"]
    table, store_manifest = _use("load_m2g_score_table", load_m2g_score_table)(
        roots["m2g_evidence_root"]
    )
    # A store that validates against its own manifest is only self-consistent.
    # This is the LINEAGE check: it must be the exact store the frozen retained
    # arm result binds. Performed in the caller, so an injected reader cannot
    # bypass it.
    store_lineage = require_m2g_evidence_store_lineage(
        store_manifest, frozen_inputs["m2g_evidence_store_identity"]
    )
    primary = _use("primary_population", _default_primary_population)(
        roots["p1_cache_root"]
    )
    challenge = _use("challenge_population", _default_challenge_population)(
        roots["feature_root"]
    )

    exposure["stage"] = "prove_exact_primary_population"
    observed = primary.identity()
    primary_lineage = require_population_identity_lineage(
        observed,
        frozen_inputs["primary_population_identity"],
        name="primary_population",
    )
    if observed["row_count"] != U1_PRIMARY_ROW_COUNT:
        raise U1DevelopmentRunError(
            f"The PRIMARY population holds {observed['row_count']} rows, not "
            f"the frozen {U1_PRIMARY_ROW_COUNT}."
        )
    challenge_identity = challenge.identity()
    challenge_lineage = require_population_identity_lineage(
        challenge_identity,
        frozen_inputs["challenge_population_identity"],
        name="challenge_population",
    )
    require_calibration_subjects(sorted(set(primary.subject_ids)))

    stable_ids = tuple(primary.stable_ids)
    labels = np.asarray(primary.labels, dtype=np.int64)
    subject_ids = np.asarray(primary.subject_ids, dtype=np.str_)
    scores = table.scores_for(stable_ids)
    challenge_scores = table.scores_for(challenge.stable_ids)

    # -- 4. SATURATION CENSUS, before any calibrator is fitted. --------------
    exposure["stage"] = "saturation_census_before_any_fit"
    census = saturation_census(scores)
    census["population_identity"] = observed
    census_digest = promote(claimed, names["census"], census, runtime=runtime)
    if not census["within_review_bound"]:
        receipt = saturation_stop(
            roots["run_root"], claimed, census=census, census_sha256=census_digest
        )
        return {
            "executed": True,
            "stopped_for_human_review": True,
            "stop_reason": receipt["stop_reason"],
            "saturation_census": census,
            "saturation_stop_receipt": receipt,
            "calibrator_fitting_started": False,
            "u1_metrics_produced": False,
            "test_accessed": False,
            "sealed_test_state": "unopened",
        }

    logits = recover_logits(scores, delta=U1_CLAMP_DELTA)
    challenge_logits = recover_logits(challenge_scores, delta=U1_CLAMP_DELTA)

    # -- 5. Twelve LOSO folds, both families in every fold. ------------------
    exposure["stage"] = "build_leave_one_subject_out_folds"
    folds = assign_calibration_folds()
    fold_manifest = build_fold_manifest(
        subject_ids=subject_ids,
        fold_assignment_sha256=fold_assignment_digest(folds),
    )
    exposure["stage"] = "fit_both_families_in_every_fold"
    exposure["calibrator_fitting_started"] = True
    fitted = fit_out_of_fold(
        logits=logits,
        labels=labels,
        subject_ids=subject_ids,
        fold_manifest=fold_manifest,
    )
    fold_manifest["folds"] = fitted["records"]
    fold_digest = promote(claimed, names["folds"], fold_manifest, runtime=runtime)

    # -- 6. Calibration evidence, then family selection from OOF NLL alone. --
    exposure["stage"] = "out_of_fold_calibration_evidence"
    exposure["u1_metrics_produced"] = True
    calibration = pooled_oof_calibration(
        labels=labels,
        stable_ids=stable_ids,
        scores=scores,
        probabilities=fitted["probabilities"],
    )
    calibration_digest = promote(
        claimed, names["calibration"], calibration, runtime=runtime
    )

    exposure["stage"] = "select_family_from_pooled_oof_nll"
    selection = select_calibrator_family(
        platt_pooled_oof_nll=negative_log_likelihood(
            labels, fitted["probabilities"][FAMILY_PLATT]
        ),
        temperature_pooled_oof_nll=negative_log_likelihood(
            labels, fitted["probabilities"][FAMILY_TEMPERATURE]
        ),
    )
    selection_digest = promote(claimed, names["selection"], selection, runtime=runtime)
    selected_family = selection["selected_family"]
    probabilities = fitted["probabilities"][selected_family]

    # -- 7. Uncertainty, routing, subject evidence, strata, challenge. -------
    exposure["stage"] = "uncertainty_from_the_frozen_decision"
    decisions = frozen_decisions(scores)
    equivalence = [
        prove_decision_equivalence(
            scores=scores[subject_ids == entry["held_out_subject"]],
            probabilities=probabilities[subject_ids == entry["held_out_subject"]],
            calibrated_boundary=fitted["calibrators"][entry["fold_index"]][
                selected_family
            ].calibrated_boundary(),
        )
        for entry in fold_manifest["folds"]
    ]
    uncertainties = uncertainty_from_decision(
        probabilities=probabilities, decisions=decisions
    )

    exposure["stage"] = "risk_coverage_evidence"
    risk_coverage = risk_coverage_curve(
        labels=labels,
        decisions=decisions,
        uncertainties=uncertainties,
        stable_ids=stable_ids,
    )
    exposure["stage"] = "derive_u_star_dev"
    u_star_dev = derive_u_star_dev(uncertainties=uncertainties, stable_ids=stable_ids)
    exposure["routing_threshold_derived"] = True
    retained_point = next(
        point
        for point in risk_coverage["points"]
        if point["target_coverage"] == U1_RETAINED_COVERAGE
    )
    if retained_point["threshold"]["u_star"] != u_star_dev["u_star"]:
        raise U1DevelopmentRunError(
            "The retained grid point and u_star_dev disagree; one frozen rule "
            "governs both."
        )
    guards = routing_guards(retained_point)

    exposure["stage"] = "subject_evidence_and_frozen_bootstrap"
    subjects = subject_level_evidence(
        labels=labels,
        decisions=decisions,
        uncertainties=uncertainties,
        subject_ids=subject_ids,
        u_star=u_star_dev["u_star"],
    )
    bootstrap = subject_bootstrap(
        labels=labels,
        decisions=decisions,
        uncertainties=uncertainties,
        subject_ids=subject_ids,
        u_star=u_star_dev["u_star"],
    )

    exposure["stage"] = "cold_start_evidence"
    cold_start_bins, observed_cache_sha256 = _use(
        "cold_start_bins", load_cold_start_bins
    )(roots["stream_cache_root"], stable_ids)
    # The artifact identity FIRST: identical stratum totals do not make it the
    # same cache, so a permuted bin assignment is refused here, before the
    # counts are ever compared.
    cache_provenance = require_stream_cache_identity(
        observed_cache_sha256, frozen_inputs["stream_cache_sha256"]
    )
    cold_start = cold_start_evidence(
        labels=labels,
        probabilities=probabilities,
        decisions=decisions,
        uncertainties=uncertainties,
        stable_ids=stable_ids,
        cold_start_bins=cold_start_bins,
        u_star=u_star_dev["u_star"],
    )
    expected_strata = frozen_inputs["cold_start_strata_window_counts"]
    observed_strata = {
        stratum: entry["window_count"]
        for stratum, entry in cold_start["strata"].items()
    }
    if observed_strata != expected_strata:
        raise U1DevelopmentRunError(
            f"The recording-age strata {observed_strata} differ from the frozen "
            f"M2-G strata {expected_strata}. U1 inherits M2's cold-start "
            "definition and never re-stratifies the population."
        )
    cold_start["strata_match_frozen_m2g_counts"] = True
    cold_start["stream_cache_provenance"] = cache_provenance

    exposure["stage"] = "challenge_routing_evidence"
    challenge_subjects = np.asarray(challenge.subject_ids, dtype=np.str_)
    challenge_families = np.asarray(challenge.target_families, dtype=np.str_)
    challenge_fold = np.full(challenge_subjects.shape[0], -1, dtype=np.int64)
    challenge_probabilities = {
        family: np.full(challenge_subjects.shape[0], np.nan, dtype=np.float64)
        for family in (FAMILY_PLATT, FAMILY_TEMPERATURE)
    }
    for entry in fold_manifest["folds"]:
        mask = challenge_subjects == entry["held_out_subject"]
        if not bool(np.any(mask)):
            # Only 4 + 8 + 1 of the twelve subjects contribute challenge
            # windows at all, so most folds have nothing to calibrate here.
            # An empty subset is not an error and is never padded with a
            # substitute calibrator.
            continue
        challenge_fold[mask] = entry["fold_index"]
        for family in (FAMILY_PLATT, FAMILY_TEMPERATURE):
            challenge_probabilities[family][mask] = fitted["calibrators"][
                entry["fold_index"]
            ][family].apply_to_logits(challenge_logits[mask])
    if int(np.count_nonzero(challenge_fold < 0)):
        raise U1DevelopmentRunError(
            "Some CHALLENGE rows belong to no frozen fold; every challenge row "
            "must be calibrated by the fold that held its subject out."
        )
    challenge_decisions = frozen_decisions(challenge_scores)
    challenge_uncertainty = uncertainty_from_decision(
        probabilities=challenge_probabilities[selected_family],
        decisions=challenge_decisions,
    )
    challenge_evidence = challenge_routing_evidence(
        target_families=challenge_families,
        subject_ids=challenge_subjects,
        decisions=challenge_decisions,
        uncertainties=challenge_uncertainty,
        u_star=u_star_dev["u_star"],
    )

    # -- 8. Per-row OOF evidence store. --------------------------------------
    exposure["stage"] = "persist_immutable_u1_artifacts"
    workspace = u1_evidence_workspace(roots["run_root"], claimed.experiment_id)
    store = write_u1_evidence_store(
        workspace,
        primary={
            "stable_id": np.asarray(stable_ids, dtype=np.str_),
            "subject_id": subject_ids,
            "fold_index": fitted["fold_index"],
            "label": labels,
            "score": scores,
            "recovered_logit": logits,
            "oof_probability_platt": fitted["probabilities"][FAMILY_PLATT],
            "oof_probability_temperature": fitted["probabilities"][FAMILY_TEMPERATURE],
            "frozen_decision": decisions,
            "cold_start_bin": cold_start_bins,
        },
        challenge={
            "stable_id": np.asarray(challenge.stable_ids, dtype=np.str_),
            "subject_id": challenge_subjects,
            "fold_index": challenge_fold,
            "target_family": challenge_families,
            "score": challenge_scores,
            "recovered_logit": challenge_logits,
            "oof_probability_platt": challenge_probabilities[FAMILY_PLATT],
            "oof_probability_temperature": challenge_probabilities[FAMILY_TEMPERATURE],
            "frozen_decision": challenge_decisions,
        },
        selected_family=selected_family,
        fold_assignment_sha256=fold_manifest["fold_assignment_sha256"],
        clamp_delta=U1_CLAMP_DELTA,
        classification_threshold=U1_CLASSIFICATION_THRESHOLD,
    )

    oof_result = {
        "artifact_class": "u1_oof_development_result",
        "development_evidence": True,
        "out_of_fold": True,
        "selected_family": selected_family,
        "family_selection": selection,
        "primary_population_identity": observed,
        "challenge_population_identity": challenge_identity,
        "input_lineage": {
            "m2g_evidence_store": store_lineage,
            "primary_population": primary_lineage,
            "challenge_population": challenge_lineage,
            "cold_start_stream_cache": cache_provenance,
            "self_consistency_alone_accepted": False,
        },
        "decision_equivalence_per_fold": equivalence,
        "risk_coverage": risk_coverage,
        "u_star_dev": u_star_dev,
        "routing_guards": guards,
        "subject_evidence": subjects,
        "subject_bootstrap": bootstrap,
        "cold_start_evidence": cold_start,
        "challenge_routing_evidence": challenge_evidence,
        "descriptive_entropy_summary": {
            "mean_calibrated_binary_entropy": float(
                np.mean(binary_entropy(probabilities))
            ),
            "descriptive_only": True,
            "used_for_routing": False,
        },
        "windows_are_independent_evidence": False,
        "inferential_unit": "subject",
        "inferential_unit_count": 12,
        "development_optimistic": True,
        "development_optimism_note": (
            "tau was itself selected on VALIDATION; cross-fitting removes "
            "subject self-calibration only and does not make VALIDATION an "
            "independent holdout"
        ),
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    oof_digest = promote(claimed, names["oof_result"], oof_result, runtime=runtime)

    # -- 9. ONE final deployment calibrator, then u_star_deploy. -------------
    exposure["stage"] = "final_deployment_calibrator"
    exposure["deployment_calibrator_fitted"] = True
    final_calibrator = fit_calibrator(
        logits=logits,
        labels=labels,
        family=selected_family,
        fit_subjects=U1_CALIBRATION_SUBJECTS,
    )
    deployment_probabilities = final_calibrator.apply_to_logits(logits)
    deployment_uncertainty = uncertainty_from_decision(
        probabilities=deployment_probabilities, decisions=decisions
    )
    exposure["stage"] = "derive_u_star_deploy_for_configuration_only"
    u_star_deploy = derive_routing_threshold(
        uncertainties=deployment_uncertainty,
        stable_ids=stable_ids,
        target_coverage=U1_RETAINED_COVERAGE,
        name="u_star_deploy",
    )
    deployment = {
        "artifact_class": "u1_final_deployment_calibrator",
        "purpose": "unseen_subjects_and_separately_authorised_test_or_deployment_only",
        "is_evaluation": False,
        "is_parameterisation": True,
        "in_sample_performance_claim_authorised": False,
        "in_sample_performance_reported": False,
        "family_reselected": False,
        "fallback_to_other_family_performed": False,
        "selected_family": selected_family,
        "family_selection_source": "frozen_out_of_fold_nll",
        "calibrator": final_calibrator.as_dict(),
        "fit_subjects": list(U1_CALIBRATION_SUBJECTS),
        "fit_population_identity": observed,
        "clamp_delta": U1_CLAMP_DELTA,
        "u1_protocol_sha256": readiness["u1_protocol_sha256"],
        "m2g_arm_result_sha256": readiness["m2g_input_identity"]["retention"][
            "retained_arm_result_sha256"
        ],
        "m2g_lock_sha256": readiness["m2g_input_identity"]["retention"][
            "retained_lock_sha256"
        ],
        "calibrated_boundary": final_calibrator.calibrated_boundary(),
        "decision_equivalence": prove_decision_equivalence(
            scores=scores,
            probabilities=deployment_probabilities,
            calibrated_boundary=final_calibrator.calibrated_boundary(),
        ),
        "u_star_deploy": u_star_deploy,
        "u_star_deploy_is_scientific_evidence": False,
        "u_star_deploy_semantics": "configuration_provenance_only",
        "test_subjects_in_fit": [],
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    deployment_digest = promote(
        claimed, names["deployment"], deployment, runtime=runtime
    )

    # -- 10. ONE top-level result binding every component. -------------------
    result = {
        "artifact_class": names["result_class"],
        "experiment_id": claimed.experiment_id,
        "experiment_identity": U1_EXPERIMENT_IDENTITY,
        "pre_claim_readiness": readiness,
        "m2g_evidence_store_identity": store_manifest,
        "input_lineage": {
            "m2g_evidence_store": store_lineage,
            "primary_population": primary_lineage,
            "challenge_population": challenge_lineage,
            "cold_start_stream_cache": cache_provenance,
            "self_consistency_alone_accepted": False,
        },
        "component_sha256": {
            names["census"]: census_digest,
            names["folds"]: fold_digest,
            names["calibration"]: calibration_digest,
            names["selection"]: selection_digest,
            names["oof_result"]: oof_digest,
            names["deployment"]: deployment_digest,
        },
        "oof_evidence_store": {
            "root": str(workspace),
            "manifest_name": U1_STORE_MANIFEST_NAME,
            "content_sha256": store["content_sha256"],
            "row_groups": store["row_groups"],
        },
        "selected_family": selected_family,
        "pooled_oof_nll": selection["pooled_oof_nll"],
        "u_star_dev": u_star_dev,
        "u_star_deploy": u_star_deploy,
        "routing_guards": guards,
        "development_evidence_source": "u1_oof_development_calibration",
        "deployment_calibrator_in_sample_performance_reported": False,
        "human_review_required": True,
        "automatic_retention": False,
        "automatic_u2_transition": False,
        "m2_replay_invoked": False,
        "m2_rerun_performed": False,
        "classification_threshold": U1_CLASSIFICATION_THRESHOLD,
        "classification_threshold_selected_here": False,
        "partition_accessed": "validation",
        "validation_accessed": True,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    promoted = finalize(
        claimed,
        result=result,
        provenance={
            "primary_population_identity": observed,
            "challenge_population_identity": challenge_identity,
            "full_replay_population_identity": readiness["m2g_input_identity"][
                "full_replay_population_identity"
            ],
            "fold_assignment_sha256": fold_manifest["fold_assignment_sha256"],
            "fold_parameters": [
                {
                    "fold_index": entry["fold_index"],
                    "held_out_subject": entry["held_out_subject"],
                    "fitted": entry["fitted"],
                }
                for entry in fold_manifest["folds"]
            ],
            "pooled_oof_nll": selection["pooled_oof_nll"],
            "selected_family": selected_family,
            "oof_evidence_store_sha256": store["content_sha256"],
            "u_star_dev": u_star_dev,
            "final_deployment_calibrator_sha256": deployment_digest,
            "u_star_deploy": u_star_deploy,
        },
        runtime=runtime,
    )
    return {
        "executed": True,
        "stopped_for_human_review": False,
        "experiment_id": claimed.experiment_id,
        "planned_execution_order": list(PLANNED_EXECUTION_ORDER),
        "selected_family": selected_family,
        "u_star_dev": u_star_dev,
        "u_star_deploy": u_star_deploy,
        "routing_guards": guards,
        "human_review_required": True,
        "automatic_retention": False,
        "automatic_u2_transition": False,
        **promoted,
    }


def _default_primary_population(p1_cache_root: Path):
    from cardiosentinel.neural.m2_populations import primary_evaluation_population

    return primary_evaluation_population(Path(p1_cache_root))


def _default_challenge_population(feature_root: Path):
    from cardiosentinel.neural.m2_populations import challenge_evaluation_population

    return challenge_evaluation_population(Path(feature_root))


# --------------------------------------------------------------------------
# CLI: only the deliberate execution controls
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Every root and identity is deterministic, so nothing else.

    There is deliberately no subject, fold, calibrator, coverage, threshold,
    partition, retry or TEST option: each would be a way to run something other
    than the one reviewed canonical experiment.
    """
    parser = argparse.ArgumentParser(
        prog="u1_development_run",
        description=(
            "Canonical U1-v1 DEVELOPMENT run (VALIDATION only, retained M2-G "
            "evidence read-only, no automatic retention). Requires explicit "
            "execution consent and the exact human-authorized Git SHA."
        ),
    )
    parser.add_argument(
        EXECUTION_FLAG,
        action="store_true",
        help="Explicit consent to run the canonical U1 development experiment.",
    )
    parser.add_argument(
        EXPECTED_GIT_SHA_FLAG,
        required=True,
        help="The human-reviewed master SHA this authorization names. HEAD must match.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Never runs on import."""
    args = build_parser().parse_args(argv)
    result = execute_canonical_u1_development(
        expected_git_sha=getattr(args, "expected_git_sha"),
        execute=getattr(args, "execute_canonical_development"),
    )
    print(U1_EXPERIMENT_IDENTITY, CANONICAL_RUN_ID, result.get("executed"))
    return 0


# `__main__` dispatch is LAST on purpose: every helper the canonical run needs
# is defined above it, so module execution can never enter the run with an
# undefined runtime helper.
if __name__ == "__main__":  # pragma: no cover - CLI dispatch
    raise SystemExit(main())
