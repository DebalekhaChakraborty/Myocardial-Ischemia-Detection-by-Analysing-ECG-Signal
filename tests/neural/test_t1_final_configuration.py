"""Tests for the final all-VALIDATION configuration selection (spec section 23).

This is a separate scientific selection event, not a summary of the folds. It
uses the same candidate set, the same threshold rule, the same transition logic
and the same lexicographic order over a different population -- all twelve
subjects rather than eleven.

Nothing here authorizes execution, claims the canonical attempt, creates the
canonical run directory or reaches TEST. The corpus is synthetic and built in
this module, so no assertion below is a claim about ischemia detection.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import t1_config as CFG
from cardiosentinel.neural import t1_execution_spec as SPEC
from cardiosentinel.neural import t1_final_configuration as F
from cardiosentinel.neural import t1_fold_authority as AUTH
from cardiosentinel.neural import t1_persistence as PERSIST
from cardiosentinel.neural.t1_fold_authority import (
    T1SubjectTargets,
    fit_evaluation_authority,
    require_active_scoped_request,
    require_validation_partition,
)
from cardiosentinel.neural.t1_protocol import (
    T1_VALIDATION_SUBJECTS,
    candidate_policies,
    t1_folds,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CADENCE = 1250
ROWS = 90
EPISODE = range(30, 48)


def _canonical_root() -> Path:
    return PERSIST.canonical_run_directory(REPOSITORY_ROOT)


def _corpus(seed: int = 5):
    rng = np.random.default_rng(seed)
    names = (
        "stable_id",
        "record_id",
        "channel_index",
        "start_sample",
        "subject_id",
        "score_present",
        "m2g_detector_score",
        "detector_decision_d_t",
        "oof_calibrated_probability_p_t",
        "decision_error_uncertainty_u_t",
        "s4d_temporal_evidence_s_t",
        "elapsed_stream_seconds",
    )
    columns: dict[str, list] = {name: [] for name in names}
    targets: dict[str, T1SubjectTargets] = {}
    for number, subject in enumerate(T1_VALIDATION_SUBJECTS):
        identifiers, positives, masks = [], [], []
        for position in range(ROWS):
            stable_id = f"{subject}:{position}"
            positive = position in EPISODE
            probability = float(
                rng.uniform(0.82, 0.99) if positive else rng.uniform(0.0, 0.30)
            )
            temporal = float(
                rng.uniform(0.78, 0.99) if positive else rng.uniform(0.0, 0.40)
            )
            columns["stable_id"].append(stable_id)
            columns["record_id"].append(f"synthetic{number:02d}")
            columns["channel_index"].append(0)
            columns["start_sample"].append(position * CADENCE)
            columns["subject_id"].append(subject)
            columns["score_present"].append(True)
            columns["m2g_detector_score"].append(probability)
            columns["detector_decision_d_t"].append(
                bool(probability >= SPEC.T1_DETECTOR_THRESHOLD)
            )
            columns["oof_calibrated_probability_p_t"].append(probability)
            columns["decision_error_uncertainty_u_t"].append(1.0 - probability)
            columns["s4d_temporal_evidence_s_t"].append(temporal)
            columns["elapsed_stream_seconds"].append(position * CADENCE / 250.0)
            identifiers.append(stable_id)
            positives.append(positive)
            masks.append(True)
        targets[subject] = T1SubjectTargets(
            subject_id=subject,
            stable_id=tuple(identifiers),
            primary_positive=tuple(positives),
            primary_mask=tuple(masks),
        )
    return {name: np.asarray(values) for name, values in columns.items()}, targets


class _Source:
    def __init__(self, targets):
        self._targets = targets
        self.asked: list[str] = []

    def read_subject_targets(self, subject_id, *, partition):
        partition = require_validation_partition(partition)
        require_active_scoped_request(subject_id, partition)
        self.asked.append(subject_id)
        return self._targets[subject_id]


@pytest.fixture(scope="module")
def corpus():
    return _corpus()


def _code_only() -> str:
    """Source with docstrings and prose literals stripped.

    The module legitimately names `policy_sort_key` and "fold selections" in
    its prose -- explaining what it does not do is not doing it -- so a raw
    substring scan would fail on the very sentences that state the guarantee.
    """
    tree = ast.parse(Path(F.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]

    class _DropProse(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and " " in node.value:
                return ast.Constant(value="")
            return node

    return ast.unparse(_DropProse().visit(tree))


# ---------------------------------------------------------------------------
# 1. All twelve subjects, exactly once
# ---------------------------------------------------------------------------


def test_the_authority_carries_the_complete_frozen_roster(corpus):
    _, targets = corpus
    authority = F.final_validation_authority(source=_Source(targets))
    assert sorted(authority.authorized_subjects) == sorted(T1_VALIDATION_SUBJECTS)
    assert len(authority.authorized_subjects) == 12
    assert len(set(authority.authorized_subjects)) == 12


def test_the_selection_asks_for_every_subject_exactly_once(corpus):
    columns, targets = corpus
    source = _Source(targets)
    F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=source)
    )
    assert sorted(source.asked) == sorted(T1_VALIDATION_SUBJECTS)
    assert len(source.asked) == len(set(source.asked)) == 12


def test_a_partial_roster_is_refused(corpus):
    _, targets = corpus
    with pytest.raises(F.T1FinalConfigurationError, match="all twelve"):
        F.FinalValidationAuthority(
            subjects=tuple(T1_VALIDATION_SUBJECTS[:11]), source=_Source(targets)
        )


def test_a_repeated_subject_is_refused(corpus):
    _, targets = corpus
    doubled = tuple(T1_VALIDATION_SUBJECTS[:11]) + (T1_VALIDATION_SUBJECTS[0],)
    with pytest.raises(F.T1FinalConfigurationError, match="more than once"):
        F.FinalValidationAuthority(subjects=doubled, source=_Source(targets))


def test_an_unknown_subject_is_refused(corpus):
    _, targets = corpus
    authority = F.final_validation_authority(source=_Source(targets))
    with pytest.raises(F.T1FinalConfigurationError, match="not a VALIDATION subject"):
        authority.require_authorized("ltstdb:s9999")


def test_the_roster_is_not_a_constructor_parameter():
    """A caller that could choose eleven would have built a fold."""
    import inspect

    parameters = inspect.signature(F.final_validation_authority).parameters
    assert list(parameters) == ["source"]


# ---------------------------------------------------------------------------
# 2. The authority cannot represent TEST
# ---------------------------------------------------------------------------


def test_the_partition_is_fixed_and_is_validation(corpus):
    _, targets = corpus
    authority = F.final_validation_authority(source=_Source(targets))
    assert authority.partition == AUTH.T1_PERMITTED_PARTITION == "validation"
    assert authority.as_dict()["test_accessed"] is False


def test_a_sealed_partition_cannot_become_an_authority(corpus):
    _, targets = corpus
    for sealed in AUTH.T1_SEALED_PARTITIONS:
        with pytest.raises(Exception):
            F.FinalValidationAuthority(
                subjects=tuple(T1_VALIDATION_SUBJECTS),
                source=_Source(targets),
                partition=sealed,
            )


def test_the_scope_cannot_be_renamed(corpus):
    _, targets = corpus
    with pytest.raises(F.T1FinalConfigurationError, match="one scope"):
        F.FinalValidationAuthority(
            subjects=tuple(T1_VALIDATION_SUBJECTS),
            source=_Source(targets),
            scope="fit_subjects_only",
        )


def test_the_module_reaches_no_test_path():
    code = _code_only()
    assert "require_no_test_access" in code
    for forbidden in ("np.load", "open(", "mkdir", "write_text", "read_bytes"):
        assert forbidden not in code, f"the module calls {forbidden}"


# ---------------------------------------------------------------------------
# 3. It is not a fold, and a fold is not it
# ---------------------------------------------------------------------------


def test_the_fold_scopes_are_unchanged():
    """A third scope there would let a fold-path caller reach all twelve."""
    assert AUTH.T1_AUTHORITY_SCOPES == (AUTH.SCOPE_FIT, AUTH.SCOPE_HELD_OUT)
    assert F.SCOPE_FINAL_VALIDATION not in AUTH.T1_AUTHORITY_SCOPES


def test_a_fold_authority_cannot_drive_the_final_selection(corpus):
    columns, targets = corpus
    fold_authority = fit_evaluation_authority(t1_folds()[0], source=_Source(targets))
    with pytest.raises(
        F.T1FinalConfigurationError, match="takes a FinalValidationAuthority"
    ):
        F.select_final_validation_configuration(
            columns=columns, authority=fold_authority
        )


def test_the_final_authority_cannot_drive_a_fold(corpus):
    from cardiosentinel.neural import t1_fold_evaluator as V

    columns, targets = corpus
    authority = F.final_validation_authority(source=_Source(targets))
    with pytest.raises(V.T1FoldEvaluatorError, match="looser type"):
        V.T1CanonicalFoldEvaluator()(t1_folds()[0], authority, columns)


def test_the_selection_reads_no_fold_output():
    code = _code_only()
    # The artifact key `derived_from_fold_selections` is the attestation that
    # this consumes none of them, so scan for reads rather than for the word.
    for forbidden in ("held_out_traces", "oof_columns", "require_held_out_bijection"):
        assert forbidden not in code, f"the selection consumes {forbidden}"
    tree = ast.parse(Path(F.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "cardiosentinel.neural.t1_assembly" not in imported


def test_the_artifact_says_it_is_not_derived_from_folds(corpus):
    columns, targets = corpus
    result = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert result["derived_from_fold_selections"] is False
    assert result["is_development_evidence"] is False
    assert result["reports_performance"] is False


# ---------------------------------------------------------------------------
# 4. Frozen candidates, frozen rules, frozen ordering
# ---------------------------------------------------------------------------


def test_the_candidate_set_is_the_frozen_twelve(corpus):
    columns, targets = corpus
    result = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    expected = [policy.name for policy in candidate_policies()]
    assert result["candidate_order"] == expected
    assert result["candidate_count"] == SPEC.T1_CANDIDATE_POLICIES_PER_FOLD == 12
    assert result["selected_policy_id"] in expected


def test_the_seven_frozen_fields_are_persisted(corpus):
    columns, targets = corpus
    result = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert set(result["configuration"]) == set(F.FINAL_CONFIGURATION_FIELDS)
    configuration = result["configuration"]
    assert configuration["p_event"] >= configuration["p_watch"]
    assert configuration["s_event"] >= configuration["s_watch"]


def test_the_threshold_population_is_the_frozen_one(corpus):
    from cardiosentinel.neural import t1_fold_evaluator as V

    columns, targets = corpus
    result = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert result["threshold_population"] == V.T1_THRESHOLD_POPULATION


def test_the_selection_is_deterministic(corpus):
    columns, targets = corpus
    first = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    second = F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert first == second


def test_no_tuning_knob_is_introduced():
    code = _code_only()
    for forbidden in (
        "policy_sort_key",
        "Q_WATCH =",
        "Q_EVENT =",
        "random",
        "default_rng",
        "seed",
        "grid",
        "tune",
        "optimi",
    ):
        assert forbidden not in code, f"the module introduces {forbidden}"
    assert "select_policy" in code
    assert "generate_thresholds" in code
    assert "run_policy_over_streams" in code


def test_the_frozen_ranking_rules_are_unchanged():
    from cardiosentinel.neural.t1_protocol import T1_SELECTION_ORDER

    assert T1_SELECTION_ORDER[0] == "pooled_episode_f1_desc"
    assert T1_SELECTION_ORDER[1] == "pooled_primary_window_mcc_desc"
    assert len(T1_SELECTION_ORDER) == 7


# ---------------------------------------------------------------------------
# 5. Nothing authorized, claimed, created or opened
# ---------------------------------------------------------------------------


def test_selecting_a_configuration_does_not_change_authorization(corpus):
    """Selecting over VALIDATION is science, not a permission event."""
    before = CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED
    columns, targets = corpus
    F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert CFG.T1_EXECUTION_SPECIFICATION_AUTHORIZED is before


def test_the_canonical_attempt_is_untouched(corpus):
    columns, targets = corpus
    assert not _canonical_root().exists()
    F.select_final_validation_configuration(
        columns=columns, authority=F.final_validation_authority(source=_Source(targets))
    )
    assert not _canonical_root().exists()


def test_the_entrypoint_asks_permission_before_executing():
    """Ordering, not state, is what protects the attempt.

    Earlier PRs asserted `main()` never reached the executor, which was true
    of the code they shipped. The composition-root PR wires it on purpose and
    the authorization PR grants permission, so the surviving property is the
    one that still protects the attempt whatever the constant reads: the gate
    is asked strictly before the executor is called.
    """
    from cardiosentinel.neural import t1_development_run as _R

    source = Path(_R.__file__).read_text(encoding="utf-8")
    body = source[source.index("def main(") :]
    assert "T1CanonicalDevelopmentExecutor" in body
    gate = body.index("require_canonical_execution_capability()")
    assert body.index("executor.execute(") > gate


def test_the_frozen_sources_are_byte_identical():
    import hashlib

    frozen = {
        "t1_protocol.py": (
            "b0df6ea2ade450037e94e5ab3b193694fea980337851a2458b3f43873450b192"
        ),
        "t1_execution_spec.py": (
            "edb0cbf1afe43dee48b5d2d0ed190e0939530fc026fd2f09d3312b929ab1fbe3"
        ),
        "t1_evidence_store.py": (
            "464ca1607191aa02042a6dcbb8cfeda4d4f3aced1eae2e29ae4b77be8cf6d39c"
        ),
    }
    for name, digest in frozen.items():
        path = REPOSITORY_ROOT / "src" / "cardiosentinel" / "neural" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
