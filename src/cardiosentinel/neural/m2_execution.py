"""Canonical M2 execution harness: label-blind input assembly and replay.

This module assembles `M2TimelineRow`s from the already-frozen canonical
artifacts and drives `m2_policy.replay_stream` for the two claim-bearing arms.

**Label firewall by module boundary.** Nothing here imports, reads or accepts
an ischemia, rate, axis, conduction, quality or episode annotation, and no
subject identifier reaches the runtime row. Annotations belong exclusively to
`m2_evaluation`, which runs strictly AFTER replay and which this module never
imports. The dependency therefore points one way only:

    frozen artifacts -> m2_execution (label-blind) -> m2_policy (label-blind)
                                                   -> evidence
                                                          |
                                    ONLY AFTER REPLAY:    v
                                                     m2_evaluation (+labels)

`assert_label_firewall()` enforces that direction structurally.

**Partition firewall.** TEST is hard-rejected before any path resolution. This
authorization permits TRAIN only, and only for a bounded, explicitly
non-claim-bearing engineering integration smoke; VALIDATION is refused here and
awaits its own separate authorization.

This module performs no canonical scientific execution on its own: the bounded
smoke entry point is explicitly named and refuses to emit claim-bearing output.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from cardiosentinel.neural import m2_gate as GATE
from cardiosentinel.neural.m1_experiment import load_stream_store
from cardiosentinel.neural.m1_store import (
    CHANNEL_INDEX_FILE,
    OBSERVATION_STATE_FILE,
    RECORD_ID_FILE,
    REPRESENTATION_FILE,
    STABLE_ID_FILE,
    START_SAMPLE_FILE,
)
from cardiosentinel.neural.m2_gate_derivation import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_M1_RUN_ROOT,
    DEFAULT_STREAM_CACHE_ROOT,
    join_sqi_and_morphology,
)
from cardiosentinel.neural.m2_policy import (
    M2_ARMS,
    M2RowEvidence,
    M2TimelineRow,
    replay_stream,
    require_m2_arm,
)
from cardiosentinel.neural.m2_scorer import FrozenM1LScorer, load_frozen_m1l_scorer
from cardiosentinel.neural.patient_memory import (
    OBSERVATION_AVAILABLE,
    REPOSITORY_ROOT,
    STANDARDIZER_NAME,
    M1DistanceStandardizer,
)
from cardiosentinel.neural.runtime_sentinel import (
    EnforcementPoint,
    RuntimeIntegrityRecord,
    require_runtime_identity,
)

FORBIDDEN_PARTITIONS: Final = ("test",)
PARTITIONS_PERMITTED_HERE: Final = ("train",)

SMOKE_ARTIFACT_CLASS: Final = "NON_CLAIM_BEARING_TRAIN_INTEGRATION_SMOKE"


class M2ExecutionError(RuntimeError):
    """Raised when a canonical M2 execution step cannot proceed with integrity."""


# --------------------------------------------------------------------------
# Partition firewall
# --------------------------------------------------------------------------


def require_permitted_partition(partition: str) -> str:
    """Hard-reject TEST, and refuse anything this authorization does not cover.

    The firewall runs before any root resolution or file access, so a forbidden
    partition can never reach a path. The B4 sealed-test evaluator is not
    reused, imported or reachable from here; future M2 test evidence, if it is
    ever wanted, needs its own separate authorization and its own reviewed path.
    """
    evaluated = str(partition).strip().lower()
    if evaluated in FORBIDDEN_PARTITIONS:
        raise M2ExecutionError(
            f"The M2 execution harness hard-rejects the {evaluated!r} partition. "
            "The B4 sealed test remains unopened and no M2 path may open it."
        )
    if evaluated not in PARTITIONS_PERMITTED_HERE:
        raise M2ExecutionError(
            f"Partition {evaluated!r} is not permitted by this authorization; "
            f"only {PARTITIONS_PERMITTED_HERE} may be touched, and only for a "
            "bounded non-claim-bearing engineering smoke."
        )
    return evaluated


# --------------------------------------------------------------------------
# Structural label firewall
# --------------------------------------------------------------------------

_ANNOTATION_IDENTIFIERS: Final = (
    "target_family",
    "target_families",
    "binary_label",
    "ischemic_positive",
    "rate_related",
    "axis_shift",
    "conduction_change",
    "challenge",
    "annotation",
    "episode",
)


FIREWALL_DEFINITION_NAME: Final = "_ANNOTATION_IDENTIFIERS"


def _module_identifiers(path: Path) -> set[str]:
    """Identifiers and non-docstring literals a module's CODE references.

    Two constructs are excluded, both because they are declarations rather
    than usages: docstrings (a module must stay free to document in prose
    which annotations it refuses to consult) and the firewall's own
    `_ANNOTATION_IDENTIFIERS` list (a definition of what is banned is not a
    use of it -- otherwise this check could never name what it forbids).
    """
    tree = ast.parse(Path(path).read_text())
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                excluded.add(id(body[0].value))
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == FIREWALL_DEFINITION_NAME
            for target in targets
        ):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    excluded.add(id(inner))
    docstrings = excluded
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            found.add(node.arg)
        elif isinstance(node, ast.alias):
            found.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.add(node.value)
    return found


def _module_imports(path: Path) -> set[str]:
    """Every module name a file actually imports.

    Checked via import statements rather than a text/identifier search, so
    naming a module in a guard string is not mistaken for depending on it.
    """
    tree = ast.parse(Path(path).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.add(module)
            for alias in node.names:
                imports.add(f"{module}.{alias.name}" if module else alias.name)
    return imports


def assert_label_firewall() -> dict[str, Any]:
    """Prove annotations cannot flow backward into the label-blind replay.

    Checked structurally rather than by convention: the replay-side modules
    must neither name an annotation quantity in code nor import the evaluation
    module that legitimately handles them.
    """
    replay_side = {
        "m2_execution": Path(__file__),
        "m2_policy": REPOSITORY_ROOT / "src/cardiosentinel/neural/m2_policy.py",
    }
    for name, path in replay_side.items():
        identifiers = _module_identifiers(path)
        leaked = sorted(identifiers & set(_ANNOTATION_IDENTIFIERS))
        if leaked:
            raise M2ExecutionError(
                f"Label firewall breach: {name} names annotation quantities "
                f"{leaked} in code."
            )
        imported = _module_imports(path)
        if any(module.endswith("m2_evaluation") for module in imported):
            raise M2ExecutionError(
                f"Label firewall breach: {name} imports the post-replay "
                "evaluation module, which would let annotations feed backward."
            )
    return {
        "label_firewall": "enforced_structurally",
        "replay_side_modules": sorted(replay_side),
        "annotation_identifiers_absent": True,
        "replay_imports_evaluation": False,
        "direction": "artifacts -> replay -> evidence -> (then) evaluation",
    }


# --------------------------------------------------------------------------
# Canonical input assembly
# --------------------------------------------------------------------------


SELECTION_SCOPE_FULL: Final = "full_partition"
SELECTION_SCOPE_BOUNDED: Final = "bounded_subset"


@dataclass(frozen=True, slots=True)
class M2CanonicalPopulation:
    """The AUTHORITY for what the full canonical evaluation population is.

    Produced only by `M2InputBundle.canonical_input_population_identity()`,
    which proves the bundle is the complete frozen partition before returning
    one. Evaluation accepts this token instead of a caller-supplied digest, so
    the reference can never be authored by the very rows being evaluated.
    """

    partition: str
    row_count: int
    ordered_stable_id_sha256: str
    stream_cache_sha256: str
    source: str = "verified_full_input_bundle"

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "row_count": self.row_count,
            "ordered_stable_id_sha256": self.ordered_stable_id_sha256,
            "stream_cache_sha256": self.stream_cache_sha256,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class M2InputBundle:
    """Row-aligned, identity-checked, label-blind canonical inputs."""

    partition: str
    rows: tuple[M2TimelineRow, ...]
    stable_ids: tuple[str, ...]
    standardizer: M1DistanceStandardizer
    stream_cache_manifest: dict[str, Any]
    selection_scope: str = SELECTION_SCOPE_FULL

    @property
    def is_full_partition(self) -> bool:
        return self.selection_scope == SELECTION_SCOPE_FULL

    def selected_ordered_stable_id_sha256(self) -> str:
        """The digest of THIS bundle's own selected rows.

        Deliberately distinct from the manifest's identity: when `record_ids`
        filtered the bundle, reporting the manifest digest would describe rows
        the bundle does not contain.
        """
        from cardiosentinel.neural.p1_experiment import ordered_stable_id_digest

        return ordered_stable_id_digest(self.stable_ids)

    def identity(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "rows": len(self.rows),
            "selection_scope": self.selection_scope,
            "selected_row_count": len(self.stable_ids),
            "selected_ordered_stable_id_sha256": (
                self.selected_ordered_stable_id_sha256()
            ),
            "stream_cache_sha256": self.stream_cache_manifest["stream_cache_sha256"],
            "ordered_chronology_sha256": self.stream_cache_manifest[
                "ordered_chronology_sha256"
            ],
            "manifest_ordered_stable_id_sha256": self.stream_cache_manifest[
                "ordered_stable_id_sha256"
            ],
            "split_sha256": self.stream_cache_manifest["split_sha256"],
            "feature_corpus_sha256": self.stream_cache_manifest[
                "feature_corpus_sha256"
            ],
            "representation_dim": self.stream_cache_manifest["representation_dim"],
            "distance_standardizer_sha256": self.stream_cache_manifest[
                "distance_standardizer_sha256"
            ],
            "labels_present": False,
            "subject_identifiers_present": False,
        }

    def canonical_input_population_identity(self) -> M2CanonicalPopulation:
        """Prove this bundle IS the full frozen partition, then vouch for it.

        Refuses a bounded/selected bundle, refuses a row count that disagrees
        with the frozen manifest's `full_stream_row_count`, and refuses a
        selected-row digest that differs from the manifest's frozen
        `ordered_stable_id_sha256`. Only a bundle that survives all three may
        define the canonical evaluation population.
        """
        if not self.is_full_partition:
            raise M2ExecutionError(
                f"A canonical population requires the full partition; this "
                f"bundle is {self.selection_scope!r}. A bounded or "
                "record-filtered bundle can never define canonical scope."
            )
        expected_rows = int(self.stream_cache_manifest["full_stream_row_count"])
        if len(self.stable_ids) != expected_rows:
            raise M2ExecutionError(
                f"Full-partition bundle holds {len(self.stable_ids)} rows, but "
                f"the frozen manifest records {expected_rows}."
            )
        observed = self.selected_ordered_stable_id_sha256()
        frozen = self.stream_cache_manifest["ordered_stable_id_sha256"]
        if observed != frozen:
            raise M2ExecutionError(
                "The bundle's ordered stable-ID digest does not match the "
                f"frozen manifest identity: observed {observed}, frozen "
                f"{frozen}."
            )
        return M2CanonicalPopulation(
            partition=self.partition,
            row_count=expected_rows,
            ordered_stable_id_sha256=frozen,
            stream_cache_sha256=self.stream_cache_manifest["stream_cache_sha256"],
        )


def load_distance_standardizer(cache_root: Path) -> M1DistanceStandardizer:
    """The frozen TRAIN-only distance standardizer, never refitted."""
    from cardiosentinel.baseline.cache import read_json

    payload = read_json(Path(cache_root) / STANDARDIZER_NAME)
    # `fitted_on_partition` is a serialized provenance key, not a dataclass
    # attribute, so it is checked on the payload it was persisted in.
    if payload.get("fitted_on_partition") != "train":
        raise M2ExecutionError("The M2 distance standardizer must be TRAIN-only.")
    return M1DistanceStandardizer.from_dict(payload)


def assemble_timeline_rows(
    partition: str,
    *,
    stream_cache_root: Path = DEFAULT_STREAM_CACHE_ROOT,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    record_ids: tuple[str, ...] | None = None,
) -> M2InputBundle:
    """Join the frozen artifacts into label-blind runtime rows.

    Every join is deterministic and identity-checked. A missing or misaligned
    row is a fatal integrity error: this never silently inner-joins a
    disagreement away, because doing so would quietly change the evaluated
    population.

    `record_ids` bounds the assembly to selected records for a bounded
    engineering smoke. It is an engineering convenience only and must never be
    chosen from any observed result.
    """
    evaluated = require_permitted_partition(partition)

    store, manifest = load_stream_store(Path(stream_cache_root), evaluated)
    if manifest["partition"] != evaluated:
        raise M2ExecutionError("The stream cache manifest partition disagrees.")

    stable_ids = np.asarray(store.array(STABLE_ID_FILE))
    record_column = np.asarray(store.array(RECORD_ID_FILE))
    channel_column = np.asarray(store.array(CHANNEL_INDEX_FILE))
    start_column = np.asarray(store.array(START_SAMPLE_FILE))
    state_column = np.asarray(store.array(OBSERVATION_STATE_FILE))
    representation = np.asarray(store.array(REPRESENTATION_FILE))

    columns = join_sqi_and_morphology(store, manifest, Path(feature_root))
    for name, values in columns.items():
        if values.shape[0] != stable_ids.shape[0]:
            raise M2ExecutionError(
                f"COMBINED_V1 column {name!r} is not row-aligned with the "
                "stream cache; this is a fatal integrity error."
            )

    selected = np.ones(stable_ids.shape[0], dtype=bool)
    if record_ids is not None:
        wanted = set(record_ids)
        missing = wanted - {str(value) for value in np.unique(record_column)}
        if missing:
            raise M2ExecutionError(
                f"Requested records {sorted(missing)} are absent from the "
                f"{evaluated} stream cache."
            )
        selected = np.isin(record_column, np.asarray(sorted(wanted)))

    rows: list[M2TimelineRow] = []
    for index in np.flatnonzero(selected):
        available = int(state_column[index]) == OBSERVATION_AVAILABLE
        rows.append(
            M2TimelineRow(
                record_id=str(record_column[index]),
                channel_index=int(channel_column[index]),
                start_sample=int(start_column[index]),
                observation_state=int(state_column[index]),
                representation=(
                    np.asarray(representation[index], dtype=np.float64)
                    if available
                    else None
                ),
                finite_sample_fraction=float(columns["finite_sample_fraction"][index]),
                sqi={name: float(columns[name][index]) for name in GATE.G3_SQI_COLUMNS},
                morphology_valid=float(columns["morphology_valid"][index]),
            )
        )

    standardizer = load_distance_standardizer(Path(stream_cache_root))
    store.close()
    return M2InputBundle(
        partition=evaluated,
        rows=tuple(rows),
        stable_ids=tuple(str(value) for value in stable_ids[selected]),
        standardizer=standardizer,
        stream_cache_manifest=dict(manifest),
        selection_scope=(
            SELECTION_SCOPE_FULL if record_ids is None else SELECTION_SCOPE_BOUNDED
        ),
    )


# --------------------------------------------------------------------------
# Canonical replay harness
# --------------------------------------------------------------------------


def replay_arm(
    bundle: M2InputBundle,
    *,
    arm: str,
    scorer: FrozenM1LScorer,
    collect_prototypes: bool = False,
) -> dict[str, Any]:
    """Replay every stream of one bundle under one arm.

    State is reset for every `(record_id, channel_index)` and never mixed
    across streams; rows are consumed in strict `start_sample` chronology by
    `replay_stream` itself.
    """
    evaluated_arm = require_m2_arm(arm)
    grouped: dict[tuple[str, int], list[M2TimelineRow]] = {}
    for row in bundle.rows:
        grouped.setdefault(row.stream_key, []).append(row)

    evidence: dict[tuple[str, int], list[M2RowEvidence]] = {}
    trajectories: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = {}
    for key in sorted(grouped):
        collected: list[tuple[float, np.ndarray]] = []
        observer = None
        if collect_prototypes:

            def observer(_index, available_time, mu_long, _sink=collected):
                _sink.append((float(available_time), np.asarray(mu_long)))

        evidence[key] = replay_stream(
            grouped[key],
            arm=evaluated_arm,
            standardizer=bundle.standardizer,
            scorer=scorer,
            prototype_observer=observer,
        )
        if collect_prototypes:
            trajectories[key] = collected

    return {
        "arm": evaluated_arm,
        "partition": bundle.partition,
        "streams": len(evidence),
        "rows": sum(len(items) for items in evidence.values()),
        "evidence": evidence,
        "prototype_trajectories": trajectories,
    }


def m2_execution_identity(
    bundle: M2InputBundle,
    scorer: FrozenM1LScorer,
    runtime: RuntimeIntegrityRecord,
) -> dict[str, Any]:
    """Everything a future canonical run must bind about its execution path."""
    return {
        "arms": list(M2_ARMS),
        "third_arm_present": False,
        "rollback": False,
        "input_identity": bundle.identity(),
        "scorer_identity": scorer.identity(),
        "gate_identity": GATE.m2_gate_identity(),
        "label_firewall": assert_label_firewall(),
        "runtime_identity_checks": runtime.as_dict(),
        "partition_accessed": bundle.partition,
        "validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }


# --------------------------------------------------------------------------
# Bounded, explicitly NON-CLAIM-BEARING TRAIN integration smoke
# --------------------------------------------------------------------------


def train_integration_smoke(
    record_ids: tuple[str, ...],
    *,
    max_rows_per_stream: int = 64,
    stream_cache_root: Path = DEFAULT_STREAM_CACHE_ROOT,
    feature_root: Path = DEFAULT_FEATURE_ROOT,
    m1_run_root: Path = DEFAULT_M1_RUN_ROOT,
) -> dict[str, Any]:
    """Prove the real-artifact plumbing works. This is NOT a scientific run.

    It verifies only that the checkpoint loads, identities align, the scorer
    emits finite values, rows assemble, replay completes causally, the schema
    can be staged and the runtime sentinel stays GREEN. It computes no metric,
    makes no arm comparison, writes no canonical artifact and supports no
    performance extrapolation whatsoever.

    `record_ids` and `max_rows_per_stream` bound the work and are chosen for
    engineering convenience BEFORE anything is observed; no selection here is
    ever made from a result.
    """
    runtime = RuntimeIntegrityRecord()
    require_runtime_identity(
        EnforcementPoint.START, record=runtime, detail=SMOKE_ARTIFACT_CLASS
    )

    bundle = assemble_timeline_rows(
        "train",
        stream_cache_root=stream_cache_root,
        feature_root=feature_root,
        record_ids=record_ids,
    )
    # Bound the work per stream, keeping each retained row's stable ID with the
    # row itself. Slicing the first N stable IDs would be WRONG: streams are
    # interleaved in the timeline, so the retained row indices are not a
    # prefix, and the identities would silently misalign with the rows.
    bounded: list[M2TimelineRow] = []
    bounded_ids: list[str] = []
    per_stream: dict[tuple[str, int], int] = {}
    for row, stable_id in zip(bundle.rows, bundle.stable_ids, strict=True):
        taken = per_stream.get(row.stream_key, 0)
        if taken >= max_rows_per_stream:
            continue
        per_stream[row.stream_key] = taken + 1
        bounded.append(row)
        bounded_ids.append(stable_id)
    bounded_bundle = M2InputBundle(
        partition=bundle.partition,
        rows=tuple(bounded),
        stable_ids=tuple(bounded_ids),
        standardizer=bundle.standardizer,
        stream_cache_manifest=bundle.stream_cache_manifest,
        selection_scope=SELECTION_SCOPE_BOUNDED,
    )

    scorer = load_frozen_m1l_scorer(Path(m1_run_root))
    arms: dict[str, Any] = {}
    for arm in M2_ARMS:
        replayed = replay_arm(
            bounded_bundle, arm=arm, scorer=scorer, collect_prototypes=True
        )
        rows = [item for items in replayed["evidence"].values() for item in items]
        scores = [
            item.decision.score for item in rows if item.decision.score is not None
        ]
        if scores and not np.all(np.isfinite(np.asarray(scores))):
            raise M2ExecutionError("The smoke observed a non-finite frozen M1L score.")
        arms[arm] = {
            "streams": replayed["streams"],
            "rows": replayed["rows"],
            "scored_rows": len(scores),
            "replay_completed": True,
        }

    require_runtime_identity(
        EnforcementPoint.COMPLETION, record=runtime, detail=SMOKE_ARTIFACT_CLASS
    )
    scorer.assert_unmutated()

    return {
        "artifact_class": SMOKE_ARTIFACT_CLASS,
        "claim_bearing": False,
        "scientific_evidence": False,
        "metrics_computed": False,
        "arm_comparison_performed": False,
        "retention_decision_performed": False,
        "performance_extrapolation_permitted": False,
        "partition_accessed": "train",
        "validation_accessed": False,
        "test_accessed": False,
        "record_ids": list(record_ids),
        "max_rows_per_stream": int(max_rows_per_stream),
        "selection_basis": "engineering_convenience_chosen_before_any_observation",
        "arms": arms,
        "scorer_identity": scorer.identity(),
        "runtime_identity_checks": runtime.as_dict(),
    }
