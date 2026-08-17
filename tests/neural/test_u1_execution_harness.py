"""The assembled canonical U1 route, executed end to end on synthetic fixtures.

`test_u1_protocol.py` proves the frozen design refuses what it must. This file
proves the EXECUTION HARNESS: the real orchestration runs, the real fits
converge, the real artifacts are promoted and the real firewalls refuse.

**No real VALIDATION scientific evidence is opened.** The per-window store the
run consumes is a synthetic on-disk M2 evidence store in `tmp_path`, written in
the frozen `m2_v1_evidence_store/1` format and read by the REAL reader, so the
reader is exercised rather than stubbed. The populations come from the REAL
`verify_primary_population` / `verify_challenge_population` authorities with
their frozen expectations substituted in one place -- the same TEST-ONLY seam
`test_m2_canonical_runner.py` established. No calibrator is fitted on real
VALIDATION rows, no real saturation census is derived, no real routing
threshold is produced, and the sealed TEST partition is untouched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from cardiosentinel.neural import m2_populations as PP
from cardiosentinel.neural import runtime_sentinel as S
from cardiosentinel.neural import u1_calibration as C
from cardiosentinel.neural import u1_development_run as R
from cardiosentinel.neural import u1_evidence_store as E
from cardiosentinel.neural import u1_persistence as PS
from cardiosentinel.neural import u1_protocol as U
from cardiosentinel.neural.integrity import canonical_sha256

FROZEN_DIGEST = "b0fd6eaa592537b7e4d5574ca68b675e85e923ae3c4a5ba411028ba6fcd7297a"
GIT_SHA = "0" * 40

SUBJECTS = U.U1_CALIBRATION_SUBJECTS
ROWS_PER_SUBJECT = 40
PRIMARY_ROWS = len(SUBJECTS) * ROWS_PER_SUBJECT
CHALLENGE_PER_FAMILY = 6
TAU = U.U1_CLASSIFICATION_THRESHOLD

CHALLENGE_FAMILY_NAMES = (
    "rate_related_confounder",
    "axis_shift_confounder",
    "conduction_change_confounder",
)


# --------------------------------------------------------------------------
# Frozen-runtime seam -- the convention `test_m2_canonical_runner.py` uses
# --------------------------------------------------------------------------


def _frozen_check(point, detail="test"):
    return S.RuntimeCheck(
        enforcement_point=S.EnforcementPoint(point).value,
        observed_digest=FROZEN_DIGEST,
        expected_digest=FROZEN_DIGEST,
        matches=True,
        package_count=335,
        observed_at="2026-01-01T00:00:00Z",
        detail=detail,
    )


@pytest.fixture()
def frozen_runtime(monkeypatch):
    """Drive the real production path with synthetic frozen observations."""

    def fake_observe(point, *, expected_digest=FROZEN_DIGEST, detail=None):
        return _frozen_check(point, detail or "test")

    def fake_require(point, *, record=None, detail=None):
        check = _frozen_check(point, detail or "test")
        if record is not None:
            record.record(check)
        return check

    monkeypatch.setattr(PS, "observe_runtime_identity", fake_observe)
    monkeypatch.setattr(PS, "require_runtime_identity", fake_require)
    monkeypatch.setattr(S, "require_runtime_identity", fake_require)
    monkeypatch.setattr(S, "observe_runtime_identity", fake_observe)
    monkeypatch.setattr(
        PS, "git_provenance", lambda _root: {"git_sha": GIT_SHA, "git_dirty": False}
    )
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance",
        lambda _root: {"git_sha": GIT_SHA, "git_dirty": False},
    )
    monkeypatch.setattr(
        PS,
        "runtime_provenance",
        lambda: {
            "interpreter": "/synthetic/python",
            "python_version": "3.12.0",
            "package_count": 335,
            "dependency_digest": FROZEN_DIGEST,
        },
    )
    return fake_require


@pytest.fixture()
def synthetic_frozen_populations(monkeypatch):
    """Substitute the frozen population expectations for the fixture corpus.

    The real 473,897-row primary identity and the real challenge selection
    digest cannot be met by a synthetic corpus, and the production validators
    rightly refuse anything else. The expectation is replaced in ONE place so
    the authorities still issue their real tokens and the assembled ROUTE is
    what is under test.
    """
    monkeypatch.setattr(
        PP,
        "PRIMARY_VALIDATION_POPULATION",
        {
            "total": PRIMARY_ROWS,
            "positive": None,  # filled by the corpus fixture
            "negative": None,
            "subjects": len(SUBJECTS),
        },
    )
    monkeypatch.setattr(R, "U1_PRIMARY_ROW_COUNT", PRIMARY_ROWS)
    monkeypatch.setattr(
        R, "U1_FULL_REPLAY_ROW_COUNT", PRIMARY_ROWS + 3 * CHALLENGE_PER_FAMILY
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_SELECTION_SHA256",
        "d" * 64,
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_TOTAL_WINDOWS",
        3 * CHALLENGE_PER_FAMILY,
    )
    monkeypatch.setattr(
        "cardiosentinel.neural.validation_challenge.CHALLENGE_EXPECTED_COUNTS",
        {
            family: {"windows": CHALLENGE_PER_FAMILY, "subjects": 1}
            for family in CHALLENGE_FAMILY_NAMES
        },
    )
    return monkeypatch


# --------------------------------------------------------------------------
# The synthetic corpus
# --------------------------------------------------------------------------


def _corpus(seed: int = 11, *, saturate: float = 0.0):
    """A deterministic, genuinely miscalibrated synthetic development corpus."""
    rng = np.random.default_rng(seed)
    stable_ids: list[str] = []
    subjects: list[str] = []
    latent: list[float] = []
    for index, subject in enumerate(SUBJECTS):
        for row in range(ROWS_PER_SUBJECT):
            stable_ids.append(f"s{index:05d}:0:{row:06d}")
            subjects.append(subject)
            latent.append(float(rng.normal(0.1 * index - 0.5, 1.8)))
    z = np.asarray(latent, dtype=np.float64)
    labels = (rng.random(z.shape[0]) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    # Persisted scores are over-confident: a > 1 and a non-zero intercept, so
    # both families have something real to correct.
    scores = 1.0 / (1.0 + np.exp(-(1.9 * z + 0.35)))
    if saturate:
        count = int(round(saturate * scores.shape[0]))
        scores[:count] = 1.0
    cold_start = np.asarray(
        [U.U1_COLD_START_STRATA[index % 3] for index in range(z.shape[0])],
        dtype=np.str_,
    )

    challenge_ids: list[str] = []
    challenge_subjects: list[str] = []
    challenge_families: list[str] = []
    for family_index, family in enumerate(CHALLENGE_FAMILY_NAMES):
        for row in range(CHALLENGE_PER_FAMILY):
            challenge_ids.append(f"c{family_index:02d}:0:{row:06d}")
            challenge_subjects.append(SUBJECTS[family_index])
            challenge_families.append(family)
    challenge_scores = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.5, len(challenge_ids))))
    return {
        "stable_ids": tuple(stable_ids),
        "subjects": tuple(subjects),
        "labels": labels,
        "scores": scores,
        "cold_start": cold_start,
        "challenge_ids": tuple(challenge_ids),
        "challenge_subjects": tuple(challenge_subjects),
        "challenge_families": tuple(challenge_families),
        "challenge_scores": challenge_scores,
    }


def _write_m2g_store(root: Path, corpus, *, arm: str = "M2-G", scores=None) -> dict:
    """A synthetic store in the frozen `m2_v1_evidence_store/1` format on disk.

    Returns the manifest, which is exactly what a real M2-G arm result carries
    as `evidence_store_identity` -- so a fixture can bind STORE A and point the
    runner at STORE B.
    """
    from cardiosentinel.data.provenance import sha256_file
    from cardiosentinel.neural.m2_evidence_store import (
        ROW_EVIDENCE_NAME,
        STORE_MANIFEST_NAME,
    )

    root.mkdir(parents=True, exist_ok=True)
    identities = np.asarray(
        list(corpus["stable_ids"]) + list(corpus["challenge_ids"]), dtype=np.str_
    )
    if scores is None:
        scores = np.concatenate([corpus["scores"], corpus["challenge_scores"]])
    total = int(identities.shape[0])
    arrays = {
        "stable_id": identities,
        "record_id": np.asarray(["s00001"] * total, dtype=np.str_),
        "channel_index": np.zeros(total, dtype=np.int64),
        "start_sample": np.arange(total, dtype=np.int64),
        "available_time": np.arange(total, dtype=np.float64),
        "score": scores,
        "scored": np.ones(total, dtype=np.bool_),
        "update_admitted": np.zeros(total, dtype=np.bool_),
    }
    rows_path = root / ROW_EVIDENCE_NAME
    with rows_path.open("wb") as handle:
        np.savez(handle, **arrays)
    manifest = {
        "schema": "m2_v1_evidence_store/1",
        "arm": arm,
        "row_count": total,
        "stream_count": 1,
        "ordered_stream_keys": [["s00001", 0]],
        "row_evidence_columns": list(arrays),
        "row_evidence_dtypes": {
            name: str(values.dtype) for name, values in sorted(arrays.items())
        },
        "row_evidence_sha256": sha256_file(rows_path),
        "prototype_trajectory_count": 0,
        "prototype_trajectory_sha256": {},
        "score_dtype": "float64",
        "prototype_dtype": "float64",
        "lossy_conversion_applied": False,
        "labels_present": False,
        "subject_identifiers_present": False,
        "annotations_present": False,
        "trajectory_points_selected_by_annotation": False,
        "corpus_scale_row_objects_retained": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    (root / STORE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def _primary_population(corpus):
    counts = {
        "total": int(corpus["labels"].shape[0]),
        "positive": int(np.count_nonzero(corpus["labels"] == 1)),
        "negative": int(np.count_nonzero(corpus["labels"] == 0)),
        "subjects": len(SUBJECTS),
    }
    return PP.verify_primary_population(
        stable_ids=corpus["stable_ids"],
        labels=corpus["labels"].tolist(),
        subject_ids=corpus["subjects"],
        cache_sha256="a" * 64,
        expected_counts=counts,
    )


def _challenge_population(corpus):
    return PP.verify_challenge_population(
        stable_ids=corpus["challenge_ids"],
        target_families=corpus["challenge_families"],
        subject_ids=corpus["challenge_subjects"],
        selection_sha256="d" * 64,
        counts={
            family: {"windows": CHALLENGE_PER_FAMILY, "subjects": 1}
            for family in CHALLENGE_FAMILY_NAMES
        },
        expected_selection_sha256="d" * 64,
        expected_counts={
            family: {"windows": CHALLENGE_PER_FAMILY, "subjects": 1}
            for family in CHALLENGE_FAMILY_NAMES
        },
        expected_total=3 * CHALLENGE_PER_FAMILY,
    )


def _roots(tmp_path: Path) -> dict[str, Path]:
    real = R.canonical_roots()
    return {
        "m2_run_root": tmp_path / "m2-runs",
        "m2g_evidence_root": tmp_path / "m2-evidence" / "M2-G",
        "feature_root": tmp_path / "features",
        "p1_cache_root": tmp_path / "p1",
        "stream_cache_root": tmp_path / "streams",
        # The frozen split manifest is static committed metadata: the TEST
        # firewall is proven against the REAL subject sets, never a fixture.
        "split_manifest": real["split_manifest"],
        "run_root": tmp_path / "u1-runs",
    }


SYNTHETIC_STREAM_CACHE_SHA256 = "9" * 64


def _loaders(corpus, primary, challenge, store_manifest, **overrides):
    identity = {
        "identity_class": "u1_m2g_input_identity",
        "retention": {
            "retained_arm_result_sha256": U.U1_M2G_ARM_RESULT_SHA256,
            "retained_lock_sha256": U.U1_M2G_LOCK_SHA256,
        },
        "retained_arm": "M2-G",
        "control_arm_is_calibration_input": False,
        "primary_population_identity": primary.identity(),
        "challenge_population_identity": challenge.identity(),
        "full_replay_population_identity": {
            "population": "full_replay",
            "row_count": PRIMARY_ROWS + 3 * CHALLENGE_PER_FAMILY,
        },
        "cold_start_strata_window_counts": {
            stratum: int(np.count_nonzero(corpus["cold_start"] == stratum))
            for stratum in U.U1_COLD_START_STRATA
        },
        "m2g_evidence_store_identity": store_manifest,
        "m2g_experiment_lock_sha256": U.U1_M2G_LOCK_SHA256,
        "stream_cache_sha256": SYNTHETIC_STREAM_CACHE_SHA256,
        "read_only": True,
    }
    loaders = {
        "m2g_input_identity": lambda _roots: identity,
        "primary_population": lambda _root: primary,
        "challenge_population": lambda _root: challenge,
        "cold_start_bins": lambda _root, stable_ids: (
            corpus["cold_start"],
            SYNTHETIC_STREAM_CACHE_SHA256,
        ),
    }
    loaders.update(overrides)
    return loaders


def _execute(tmp_path, corpus, *, store_manifest=None, **overrides):
    roots = _roots(tmp_path)
    written = _write_m2g_store(roots["m2g_evidence_root"], corpus)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    return R.execute_canonical_u1_development(
        expected_git_sha=GIT_SHA,
        execute=True,
        _roots=roots,
        _loaders=_loaders(
            corpus,
            primary,
            challenge,
            written if store_manifest is None else store_manifest,
            **overrides,
        ),
    )


@pytest.fixture()
def corpus():
    return _corpus()


@pytest.fixture()
def executed(tmp_path, corpus, frozen_runtime, synthetic_frozen_populations):
    return _execute(tmp_path, corpus), tmp_path, corpus


# ==========================================================================
# A. CLAIM / EXECUTION
# ==========================================================================


def test_nothing_runs_on_import():
    source = Path(R.__file__).read_text()
    tree = ast.parse(source)
    executable = [
        node
        for node in tree.body
        if not isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
            ),
        )
    ]
    assert [type(node).__name__ for node in executable] == ["If"]
    assert source.rstrip().endswith("raise SystemExit(main())")


def test_execution_requires_explicit_consent(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    roots = _roots(tmp_path)
    written = _write_m2g_store(roots["m2g_evidence_root"], corpus)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    plan = R.execute_canonical_u1_development(
        expected_git_sha=GIT_SHA,
        _roots=roots,
        _loaders=_loaders(corpus, primary, challenge, written),
    )
    assert plan["executed"] is False
    assert plan["planned_execution_order"] == list(R.PLANNED_EXECUTION_ORDER)
    assert not (roots["run_root"]).exists()


def test_an_absent_expected_git_sha_is_refused(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    roots = _roots(tmp_path)
    written = _write_m2g_store(roots["m2g_evidence_root"], corpus)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    with pytest.raises(R.U1DevelopmentRunError, match="is required"):
        R.execute_canonical_u1_development(
            expected_git_sha=None,
            execute=True,
            _roots=roots,
            _loaders=_loaders(corpus, primary, challenge, written),
        )
    assert not roots["run_root"].exists()


def test_a_mismatched_git_sha_stops_before_any_claim(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    roots = _roots(tmp_path)
    written = _write_m2g_store(roots["m2g_evidence_root"], corpus)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    with pytest.raises(R.U1DevelopmentRunError, match="human authorization names"):
        R.execute_canonical_u1_development(
            expected_git_sha="f" * 40,
            execute=True,
            _roots=roots,
            _loaders=_loaders(corpus, primary, challenge, written),
        )
    assert not roots["run_root"].exists()


def test_a_dirty_tree_is_refused(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations, monkeypatch
):
    monkeypatch.setattr(
        "cardiosentinel.data.provenance.git_provenance",
        lambda _root: {"git_sha": GIT_SHA, "git_dirty": True},
    )
    with pytest.raises(R.U1DevelopmentRunError, match="clean Git checkout"):
        _execute(tmp_path, corpus)


def test_a_second_canonical_attempt_is_refused(executed):
    result, tmp_path, corpus = executed
    assert result["executed"] is True
    with pytest.raises(PS.U1PersistenceError, match="already claimed"):
        _execute(tmp_path, corpus)


def test_no_alternate_canonical_run_id_is_permitted():
    with pytest.raises(R.U1DevelopmentRunError, match="is refused"):
        R.require_canonical_run_id("u1-v1-development-recovery1")


def test_no_per_window_evidence_is_opened_before_the_claim(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    """The access choreography, proven by absence rather than asserted.

    The per-window store is deleted before the run. Pre-claim readiness must
    still complete -- it never touches it -- and the failure that follows must
    happen only AFTER the claim directory exists.
    """
    roots = _roots(tmp_path)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    # The frozen identity exists; the STORE does not. Written somewhere the
    # runner never looks, purely to supply a realistic bound identity.
    written = _write_m2g_store(tmp_path / "never-opened" / "M2-G", corpus)
    loaders = _loaders(corpus, primary, challenge, written)

    readiness = R.pre_claim_readiness(
        expected_git_sha=GIT_SHA, roots=roots, loaders=loaders
    )
    assert readiness["per_window_evidence_opened"] is False
    assert readiness["calibrator_fitting_started"] is False
    assert readiness["validation_accessed"] is False
    assert not roots["m2g_evidence_root"].exists()

    with pytest.raises(R.U1DevelopmentRunError, match="never regenerated"):
        R.execute_canonical_u1_development(
            expected_git_sha=GIT_SHA,
            execute=True,
            _roots=roots,
            _loaders=loaders,
        )
    run_dir = roots["run_root"] / R.CANONICAL_RUN_ID
    assert run_dir.is_dir()
    receipt = json.loads(
        (
            roots["run_root"]
            / f"{R.CANONICAL_RUN_ID}__review"
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["failed_stage"] == "open_retained_m2g_evidence_read_only"
    assert receipt["exposure"]["calibrator_fitting_started"] is False
    assert receipt["promoted_artifacts"] == {}


def test_the_route_never_retries_automatically():
    source = Path(R.__file__).read_text()
    for banned in ("--force", "--retry", "--reset", "--overwrite", "--fresh-seed"):
        assert banned not in source


# ==========================================================================
# B. INPUT INTEGRITY
# ==========================================================================


def test_only_the_retained_arm_may_be_a_calibration_input(tmp_path, corpus):
    root = tmp_path / "control" / "M2-0"
    _write_m2g_store(root, corpus, arm="M2-0")
    with pytest.raises(R.U1DevelopmentRunError, match="control arm never is"):
        R.load_m2g_score_table(root)


def test_a_mutated_evidence_store_is_refused_never_regenerated(tmp_path, corpus):
    from cardiosentinel.neural.m2_evidence_store import (
        ROW_EVIDENCE_NAME,
        M2EvidenceStoreError,
    )

    root = tmp_path / "evidence" / "M2-G"
    _write_m2g_store(root, corpus)
    (root / ROW_EVIDENCE_NAME).write_bytes(b"corrupted")
    with pytest.raises(M2EvidenceStoreError):
        R.load_m2g_score_table(root)


def test_a_missing_evidence_store_stops_rather_than_regenerating(tmp_path):
    with pytest.raises(R.U1DevelopmentRunError, match="never regenerated"):
        R.load_m2g_score_table(tmp_path / "absent")


def test_the_g4_admission_quantity_is_not_a_calibration_input():
    firewall = R.calibration_input_firewall()
    assert firewall["calibration_input_field"] == "score"
    assert firewall["g4_normal_evidence_calibrated"] is False
    for forbidden in U.U1_FORBIDDEN_CALIBRATION_INPUTS:
        assert forbidden in firewall["forbidden_calibration_inputs"]


def test_primary_membership_must_match_the_retained_identity(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    roots = _roots(tmp_path)
    written = _write_m2g_store(roots["m2g_evidence_root"], corpus)
    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    loaders = _loaders(corpus, primary, challenge, written)
    identity = loaders["m2g_input_identity"](roots)
    identity["primary_population_identity"] = dict(
        identity["primary_population_identity"], ordered_stable_id_sha256="e" * 64
    )
    loaders["m2g_input_identity"] = lambda _roots: identity
    with pytest.raises(
        R.U1DevelopmentRunError, match="observed primary_population identity"
    ):
        R.execute_canonical_u1_development(
            expected_git_sha=GIT_SHA,
            execute=True,
            _roots=roots,
            _loaders=loaders,
        )


def test_duplicate_stable_ids_are_fatal_and_never_deduplicated():
    with pytest.raises(U.U1ProtocolError, match="Duplicate stable_ids"):
        U.equal_mass_groups([0.1, 0.2], ["a", "a"], bins=2)


def test_stable_id_uniqueness_is_linear_not_quadratic():
    """The frozen PRIMARY size must actually be executable.

    `identities.count(i)` per row is O(N^2); at 473,897 rows it never returns.
    The refusal is unchanged -- the same duplicates, in the same sorted order.
    """
    import time

    size = 200_000
    identities = [f"row{index:07d}" for index in range(size)]
    values = [index / size for index in range(size)]
    started = time.perf_counter()
    order = U.equal_mass_sort_order(values, identities)
    assert time.perf_counter() - started < 20.0
    assert len(order) == size

    identities[5] = identities[9]
    with pytest.raises(U.U1ProtocolError) as error:
        U.equal_mass_sort_order(values, identities)
    assert "row0000009" in str(error.value)


def test_an_empty_stable_id_is_refused_before_duplicates_are_reported():
    with pytest.raises(U.U1ProtocolError, match=r"stable_id\[2\] is empty"):
        U.equal_mass_sort_order([0.1, 0.2, 0.3], ["a", "a", "  "])


def test_a_missing_stable_id_is_fatal():
    with pytest.raises(U.U1ProtocolError, match="the tie-break is not"):
        U.equal_mass_sort_order([0.1, 0.2, 0.3], ["a", "b"])


def test_the_linear_uniqueness_check_reports_exactly_what_the_quadratic_one_did():
    """§13 permits hardening only at byte-for-byte equal semantics."""
    identities = ["c", "a", "b", "a", "c", "d", "c"]

    def quadratic(values):
        return sorted({value for value in values if values.count(value) > 1})

    with pytest.raises(U.U1ProtocolError) as error:
        U.equal_mass_sort_order([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], identities)
    expected = quadratic(identities)
    assert expected == ["a", "c"]
    assert f"Duplicate stable_ids {expected[:5]}" in str(error.value)


def test_the_real_retained_m2g_identity_verifies_against_the_frozen_artifacts():
    """Static frozen metadata only -- no per-window row is opened here."""
    roots = R.canonical_roots()
    if not (roots["m2_run_root"]).is_dir():
        pytest.skip("the retained M2 run root is not on this filesystem")
    identity = R.m2g_input_identity(roots)
    assert identity["retained_arm"] == "M2-G"
    assert identity["control_arm_is_calibration_input"] is False
    assert identity["read_only"] is True
    assert (
        identity["retention"]["retained_arm_result_sha256"]
        == U.U1_M2G_ARM_RESULT_SHA256
    )
    assert identity["retention"]["retained_lock_sha256"] == U.U1_M2G_LOCK_SHA256
    assert identity["primary_population_identity"]["row_count"] == 473_897
    assert identity["full_replay_population_identity"]["row_count"] == 492_904
    assert (
        identity["primary_population_identity"]["ordered_stable_id_sha256"]
        == "a671d35a354748e47c9ce77726462c59dfdc82c14249c204ff6ef00d35a27f1c"
    )
    assert (
        identity["challenge_population_identity"]["challenge_selection_sha256"]
        == "49899d1b59430ff22f70cdf509184e98caedbe0e2a8756939ee77e25210ee72a"
    )
    assert identity["cold_start_strata_window_counts"] == {
        "0_5_minutes": 1_798,
        "5_60_minutes": 19_637,
        "over_60_minutes": 452_462,
    }


# ==========================================================================
# C. SATURATION
# ==========================================================================


def test_the_census_counts_exactly_what_the_protocol_asks():
    scores = np.asarray([0.0, 1.0, 0.5, 0.5, 1e-9, 1.0 - 1e-9], dtype=np.float64)
    census = C.saturation_census(scores)
    assert census["score_equal_zero_count"] == 1
    assert census["score_equal_one_count"] == 1
    assert census["score_outside_clamp_count"] == 4
    assert census["distinct_persisted_score_count"] == 5
    assert census["clamp_delta"] == U.U1_CLAMP_DELTA
    assert census["is_scientific_result"] is False


def test_a_saturated_population_stops_before_any_fit(
    tmp_path, frozen_runtime, synthetic_frozen_populations
):
    saturated = _corpus(saturate=0.5)
    result = _execute(tmp_path, saturated)
    assert result["stopped_for_human_review"] is True
    assert result["calibrator_fitting_started"] is False
    assert result["u1_metrics_produced"] is False

    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    promoted = sorted(path.name for path in run_dir.iterdir())
    assert PS.SATURATION_CENSUS_NAME in promoted
    assert PS.RESULT_NAME not in promoted
    assert PS.EXPERIMENT_LOCK_NAME not in promoted
    assert PS.FOLD_MANIFEST_NAME not in promoted

    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_STOPPED
    assert status["calibrator_fitting_started"] is False

    receipt = json.loads(
        (
            _roots(tmp_path)["run_root"]
            / f"{R.CANONICAL_RUN_ID}__review"
            / PS.SATURATION_STOP_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["calibrators_fitted"] == 0
    assert receipt["clamp_widened"] is False
    assert receipt["fallback_calibrator_used"] is False
    assert receipt["m2_rerun_performed"] is False
    assert receipt["test_accessed"] is False


def test_the_review_bound_is_the_frozen_one():
    assert U.U1_SATURATED_FRACTION_REVIEW_BOUND == 0.01
    just_inside = np.concatenate([np.full(1, 1.0), np.full(99, 0.5)])
    assert C.saturation_census(just_inside)["within_review_bound"] is True
    just_outside = np.concatenate([np.full(2, 1.0), np.full(98, 0.5)])
    assert C.saturation_census(just_outside)["within_review_bound"] is False


# ==========================================================================
# D. LOSO
# ==========================================================================


def test_every_row_receives_exactly_one_out_of_fold_probability(executed):
    result, tmp_path, corpus = executed
    workspace = _roots(tmp_path)["run_root"] / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest = json.loads((workspace / E.U1_STORE_MANIFEST_NAME).read_text())
    group = E.read_u1_row_group(workspace, manifest, "primary_metric")
    assert group.row_count == PRIMARY_ROWS
    folds = group.arrays["fold_index"]
    assert set(folds.tolist()) == set(range(len(SUBJECTS)))
    for family in ("oof_probability_platt", "oof_probability_temperature"):
        assert np.all(np.isfinite(group.arrays[family]))


def test_a_fold_never_fits_on_its_own_held_out_subject(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    manifest = json.loads((run_dir / PS.FOLD_MANIFEST_NAME).read_text())
    assert manifest["fold_count"] == 12
    assert manifest["performance_dependent_assignment"] is False
    for entry in manifest["folds"]:
        assert entry["held_out_subject"] not in entry["fit_subjects"]
        assert entry["fit_subject_count"] == 11
        assert entry["evaluation_row_count"] == ROWS_PER_SUBJECT
        assert entry["fit_row_count"] == PRIMARY_ROWS - ROWS_PER_SUBJECT
        for family in (C.FAMILY_PLATT, C.FAMILY_TEMPERATURE):
            fitted = entry["fitted"][family]
            assert fitted["fit_subjects"] == entry["fit_subjects"]
            assert fitted["a"] > 0.0


def test_a_fold_gap_is_fatal():
    fold_manifest = {
        "folds": [
            {
                "fold_index": 0,
                "held_out_subject": SUBJECTS[0],
                "fit_subjects": list(SUBJECTS[1:]),
            }
        ]
    }
    subjects = np.asarray(list(SUBJECTS) * 4, dtype=np.str_)
    logits = np.linspace(-2.0, 2.0, subjects.shape[0])
    labels = (logits > 0).astype(np.int64)
    with pytest.raises(R.U1DevelopmentRunError, match="not exact"):
        R.fit_out_of_fold(
            logits=logits,
            labels=labels,
            subject_ids=subjects,
            fold_manifest=fold_manifest,
        )


# ==========================================================================
# E. FITTING
# ==========================================================================


def test_the_platt_fit_recovers_a_known_calibration():
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 2.0, 60_000)
    labels = (rng.random(z.shape[0]) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    scores = 1.0 / (1.0 + np.exp(-(2.0 * z + 0.5)))
    fitted = C.fit_calibrator(
        logits=C.recover_logits(scores), labels=labels, family=C.FAMILY_PLATT
    )
    assert fitted.a == pytest.approx(0.5, abs=0.05)
    assert fitted.b == pytest.approx(-0.25, abs=0.05)
    assert fitted.optimizer["method"] == "L-BFGS-B"
    assert fitted.optimizer["maxiter"] == 500
    assert fitted.optimizer["gtol"] == 1e-10
    assert fitted.optimizer["automatic_retry_performed"] is False


def test_the_temperature_only_fit_fixes_the_intercept_at_zero():
    rng = np.random.default_rng(4)
    z = rng.normal(0.0, 2.0, 40_000)
    labels = (rng.random(z.shape[0]) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    scores = 1.0 / (1.0 + np.exp(-(2.5 * z)))
    fitted = C.fit_calibrator(
        logits=C.recover_logits(scores), labels=labels, family=C.FAMILY_TEMPERATURE
    )
    assert fitted.b == 0.0
    assert fitted.parameter_count == 1
    assert fitted.a == pytest.approx(0.4, abs=0.05)


def test_the_analytic_gradient_matches_numerical_differentiation():
    from scipy.optimize import check_grad

    rng = np.random.default_rng(5)
    z = rng.normal(0.0, 2.0, 5_000)
    labels = (rng.random(z.shape[0]) < 0.4).astype(np.float64)

    def objective(parameters):
        return C._objective_and_gradient(
            parameters, logits=z, labels=labels, free_intercept=True, delta=1e-7
        )[0]

    def gradient(parameters):
        return C._objective_and_gradient(
            parameters, logits=z, labels=labels, free_intercept=True, delta=1e-7
        )[1]

    assert check_grad(objective, gradient, np.asarray([1.3, -0.2])) < 1e-5


def test_a_non_monotonic_map_is_a_hard_failure():
    with pytest.raises(C.U1CalibrationError, match="not strictly increasing"):
        C.U1Calibrator(
            family=C.FAMILY_PLATT,
            a=-0.5,
            b=0.0,
            clamp_delta=1e-7,
            fit_row_count=10,
            fit_subjects=(),
            optimizer={},
        )


def test_a_non_finite_parameter_is_a_hard_failure():
    with pytest.raises(C.U1CalibrationError, match="non-finite"):
        C.U1Calibrator(
            family=C.FAMILY_PLATT,
            a=float("nan"),
            b=0.0,
            clamp_delta=1e-7,
            fit_row_count=10,
            fit_subjects=(),
            optimizer={},
        )


def test_non_convergence_stops_without_retry_or_substitution(monkeypatch):
    class _Failed:
        success = False
        status = 1
        message = "ABNORMAL_TERMINATION"
        x = np.asarray([1.0, 0.0])
        fun = 0.5
        nit = 500
        nfev = 600

    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs: _Failed())
    with pytest.raises(C.U1CalibrationError, match="no fallback to the other family"):
        C.fit_calibrator(
            logits=np.asarray([0.1, 0.2]),
            labels=np.asarray([0, 1]),
            family=C.FAMILY_PLATT,
        )


def test_the_optimiser_leaves_the_scale_unbounded():
    """A bound at `a > 0` would park a degenerate fit and hide §6.2's failure."""
    rng = np.random.default_rng(6)
    z = rng.normal(0.0, 1.0, 4_000)
    # Labels ANTI-correlated with the score: the maximum-likelihood scale is
    # negative, and the fit must surface that rather than sit on a bound.
    labels = (rng.random(z.shape[0]) < 1.0 / (1.0 + np.exp(z))).astype(np.int64)
    scores = 1.0 / (1.0 + np.exp(-z))
    with pytest.raises(C.U1CalibrationError, match="not strictly increasing"):
        C.fit_calibrator(
            logits=C.recover_logits(scores), labels=labels, family=C.FAMILY_PLATT
        )


# ==========================================================================
# F. FAMILY SELECTION
# ==========================================================================


def test_lower_pooled_oof_nll_wins():
    decision = C.select_calibrator_family(
        platt_pooled_oof_nll=0.40, temperature_pooled_oof_nll=0.55
    )
    assert decision["selected_family"] == C.FAMILY_PLATT
    assert decision["tie_within_tolerance"] is False


def test_a_tie_within_tolerance_retains_the_simpler_nested_model():
    decision = C.select_calibrator_family(
        platt_pooled_oof_nll=0.500000, temperature_pooled_oof_nll=0.50005
    )
    assert decision["selected_family"] == C.FAMILY_TEMPERATURE
    assert decision["tie_within_tolerance"] is True
    assert decision["nll_tie_tolerance"] == 1e-4


def test_family_selection_structurally_cannot_consult_ece_or_brier():
    import inspect

    signature = inspect.signature(C.select_calibrator_family)
    assert set(signature.parameters) == {
        "platt_pooled_oof_nll",
        "temperature_pooled_oof_nll",
        "tie_tolerance",
    }
    decision = C.select_calibrator_family(
        platt_pooled_oof_nll=0.4, temperature_pooled_oof_nll=0.5
    )
    for banned in (
        "ece_used",
        "brier_used",
        "auprc_used",
        "routing_risk_used",
        "challenge_evidence_used",
        "weighted_score_used",
        "test_accessed",
    ):
        assert decision[banned] is False
    assert decision["is_u1_retention_decision"] is False


# ==========================================================================
# G. CALIBRATION METRICS
# ==========================================================================


def test_brier_and_nll_are_the_frozen_definitions():
    labels = np.asarray([1, 0, 1, 0])
    probabilities = np.asarray([0.9, 0.2, 0.6, 0.4])
    assert C.brier_score(labels, probabilities) == pytest.approx(
        float(np.mean((probabilities - labels) ** 2))
    )
    expected = -float(
        np.mean(
            labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities)
        )
    )
    assert C.negative_log_likelihood(labels, probabilities) == pytest.approx(expected)


def test_equal_width_binning_always_bins_probability_one():
    labels = np.asarray([1] * 15 + [0] * 15)
    probabilities = np.concatenate([np.linspace(0.0, 0.99, 15), np.full(15, 1.0)])
    report = C.equal_width_reliability(labels=labels, probabilities=probabilities)
    assert report["bin_count"] == 15
    assert report["bins"][14]["upper_edge_inclusive"] is True
    assert report["bins"][14]["count"] >= 15
    assert report["library_quantile_used"] is False


def test_equal_mass_groups_partition_every_row_exactly_once():
    rng = np.random.default_rng(8)
    probabilities = rng.random(1_003)
    labels = (rng.random(1_003) < probabilities).astype(np.int64)
    identities = [f"row{index:05d}" for index in range(1_003)]
    report = C.equal_mass_reliability(
        labels=labels, probabilities=probabilities, stable_ids=identities
    )
    assert sum(report["group_sizes"]) == 1_003
    assert max(report["group_sizes"]) - min(report["group_sizes"]) <= 1
    assert report["sort_key"] == ["calibrated_probability", "stable_id"]


def test_calibration_metrics_are_row_order_invariant():
    rng = np.random.default_rng(9)
    probabilities = rng.random(300)
    labels = (rng.random(300) < probabilities).astype(np.int64)
    identities = [f"row{index:05d}" for index in range(300)]
    forward = C.calibration_evidence(
        labels=labels,
        probabilities=probabilities,
        stable_ids=identities,
        name="forward",
        is_out_of_fold=True,
    )
    order = rng.permutation(300)
    shuffled = C.calibration_evidence(
        labels=labels[order],
        probabilities=probabilities[order],
        stable_ids=[identities[index] for index in order],
        name="shuffled",
        is_out_of_fold=True,
    )
    for binning in ("reliability_equal_width", "reliability_equal_mass"):
        assert forward[binning]["expected_calibration_error"] == pytest.approx(
            shuffled[binning]["expected_calibration_error"]
        )
    assert forward["brier"] == pytest.approx(shuffled["brier"])


def test_the_uncalibrated_baseline_is_reported_beside_both_families(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    calibration = json.loads((run_dir / PS.OOF_CALIBRATION_NAME).read_text())
    families = calibration["families"]
    assert set(families) == {
        C.FAMILY_PLATT,
        C.FAMILY_TEMPERATURE,
        "uncalibrated_baseline",
    }
    assert families["uncalibrated_baseline"]["out_of_fold"] is False
    assert families[C.FAMILY_PLATT]["out_of_fold"] is True
    assert calibration["true_logit_temperature_scaling_performed"] is False


# ==========================================================================
# H. CLASSIFICATION
# ==========================================================================


def test_the_frozen_decision_is_identical_before_and_after_calibration(executed):
    result, tmp_path, corpus = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    assert len(oof["decision_equivalence_per_fold"]) == 12
    for proof in oof["decision_equivalence_per_fold"]:
        assert proof["disagreement_count"] == 0
        assert proof["row_for_row_identical"] is True
        assert proof["threshold_selected_here"] is False
        assert proof["calibrated_boundary_is_a_new_threshold"] is False


def test_u1_may_not_classify_at_any_other_threshold():
    with pytest.raises(C.U1CalibrationError, match="may not classify"):
        C.frozen_decisions(np.asarray([0.5]), threshold=0.5)


def test_a_broken_calibrated_boundary_is_caught_row_for_row():
    scores = np.asarray([TAU - 1e-3, TAU + 1e-3])
    with pytest.raises(C.U1CalibrationError, match="disagrees with the frozen"):
        C.prove_decision_equivalence(
            scores=scores,
            probabilities=np.asarray([0.9, 0.1]),
            calibrated_boundary=0.5,
        )


# ==========================================================================
# I. ROUTING
# ==========================================================================


def test_the_coverage_grid_is_exactly_the_frozen_one(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    grid = [point["target_coverage"] for point in oof["risk_coverage"]["points"]]
    assert grid == list(U.U1_COVERAGE_GRID)
    for point in oof["risk_coverage"]["points"]:
        assert point["coverage"] >= point["target_coverage"]
        assert point["threshold"]["library_quantile_used"] is False


def test_the_order_statistic_is_the_frozen_ceil_rule():
    assert U.routing_threshold_rank(473_897, 0.90) == 426_508
    lower = 426_507 / 473_897
    assert lower < 0.90
    assert 426_508 / 473_897 >= 0.90


def test_ties_can_only_raise_achieved_coverage():
    # The worked example from the frozen protocol review: four rows tied at
    # `u_star = 0.5` with a target of 0.60 accepts 8 of 10 rows.
    uncertainties = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.9, 0.99]
    identities = [f"row{index}" for index in range(10)]
    derived = C.derive_routing_threshold(
        uncertainties=uncertainties,
        stable_ids=identities,
        target_coverage=0.60,
    )
    assert derived["rank"] == 6
    assert derived["u_star"] == 0.5
    assert derived["threshold_tie_count"] == 4
    assert derived["accepted_count"] == 8
    assert derived["achieved_coverage"] == pytest.approx(0.80)


def test_the_vectorised_routing_path_matches_the_frozen_rule_on_every_grid_point():
    rng = np.random.default_rng(12)
    size = 2_000
    uncertainties = np.round(rng.random(size), 3)  # deliberate ties
    identities = [f"row{index:05d}" for index in range(size)]
    for target in U.U1_COVERAGE_GRID:
        frozen = U.select_routing_threshold(uncertainties.tolist(), identities, target)
        derived = C.derive_routing_threshold(
            uncertainties=uncertainties,
            stable_ids=identities,
            target_coverage=target,
        )
        assert derived["u_star"] == frozen.u_star
        assert derived["accepted_count"] == frozen.accepted_count
        assert derived["threshold_tie_count"] == frozen.threshold_tie_count
        assert derived["achieved_coverage"] == frozen.achieved_coverage


def test_c_star_is_fixed_at_the_frozen_reference_operating_point():
    assert U.U1_RETAINED_COVERAGE == 0.90
    assert U.U1_RETAINED_COVERAGE_IS_MEASURED_CAPACITY is False


def test_class_aware_and_agreement_evidence_is_present_at_every_grid_point(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    for point in oof["risk_coverage"]["points"]:
        for field in (
            "accepted_sensitivity",
            "accepted_specificity",
            "accepted_ppv",
            "accepted_npv",
            "true_positive_escalation_fraction",
            "true_negative_escalation_fraction",
            "accepted_positive_count",
            "accepted_negative_count",
            "predicted_accepted_risk",
            "observed_accepted_risk",
            "accepted_risk_absolute_agreement_error",
        ):
            assert field in point


def test_u_star_dev_agrees_with_the_frozen_protocol_helper(executed):
    result, tmp_path, _corpus_ = executed
    assert result["u_star_dev"]["derived_by"] == "u1_protocol.select_routing_threshold"
    assert result["u_star_dev"]["vectorised_cross_check_agreed"] is True
    assert result["u_star_dev"]["achieved_coverage"] >= 0.90


# ==========================================================================
# J. GUARDS
# ==========================================================================


def test_an_asymmetric_abstention_ratio_above_three_raises_a_flag():
    guards = C.routing_guards(
        {
            "target_coverage": 0.90,
            "true_positive_escalation_fraction": 0.40,
            "true_negative_escalation_fraction": 0.10,
            "accepted_risk_absolute_agreement_error": 0.001,
            "predicted_accepted_risk": 0.05,
            "observed_accepted_risk": 0.049,
        }
    )
    assert guards["asymmetric_abstention_ratio"] == pytest.approx(4.0)
    assert guards["flags"]["asymmetric_abstention"] is True
    assert guards["threshold_reselected"] is False
    assert guards["refit_performed"] is False
    assert guards["scientific_evidence_discarded"] is False


def test_a_risk_disagreement_above_two_percent_raises_a_flag():
    guards = C.routing_guards(
        {
            "target_coverage": 0.90,
            "true_positive_escalation_fraction": 0.10,
            "true_negative_escalation_fraction": 0.10,
            "accepted_risk_absolute_agreement_error": 0.031,
            "predicted_accepted_risk": 0.08,
            "observed_accepted_risk": 0.049,
        }
    )
    assert guards["flags"]["routing_calibration_inadequacy"] is True
    assert guards["accepted_risk_agreement_tolerance"] == 0.02


def test_the_guard_bounds_are_the_frozen_ones():
    assert U.U1_ASYMMETRIC_ABSTENTION_RATIO == 3.0
    assert U.U1_ACCEPTED_RISK_AGREEMENT_TOLERANCE == 0.02


def test_escalating_only_positives_is_surfaced_not_hidden_by_a_zero_divisor():
    guards = C.routing_guards(
        {
            "target_coverage": 0.90,
            "true_positive_escalation_fraction": 0.5,
            "true_negative_escalation_fraction": 0.0,
            "accepted_risk_absolute_agreement_error": 0.0,
        }
    )
    assert guards["flags"]["asymmetric_abstention"] is True
    assert guards["asymmetric_abstention_ratio"] is None


def test_a_raised_guard_still_persists_the_complete_scientific_result(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations, monkeypatch
):
    # The REAL guard evaluator, with tightened bounds, so what is under test is
    # the persistence path a raised flag takes -- not a stubbed verdict.
    real_guards = C.routing_guards
    monkeypatch.setattr(
        C,
        "routing_guards",
        lambda point, **_: real_guards(
            point, abstention_ratio_bound=1.0000001, risk_agreement_tolerance=0.0
        ),
    )
    result = _execute(tmp_path, corpus)
    assert result["routing_guards"]["any_flag_raised"] is True
    assert result["human_review_required"] is True
    assert result["automatic_retention"] is False

    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    assert (run_dir / PS.RESULT_NAME).is_file()
    assert (run_dir / PS.EXPERIMENT_LOCK_NAME).is_file()
    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_COMPLETE
    assert status["routing_guard_flags_raised"]


# ==========================================================================
# K. BOOTSTRAP
# ==========================================================================


def test_the_bootstrap_resamples_subjects_at_the_frozen_seed(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    bootstrap = oof["subject_bootstrap"]
    assert bootstrap["unit"] == "subject"
    assert bootstrap["seed"] == 2026
    assert bootstrap["requested_replicates"] == 1000
    assert bootstrap["window_bootstrap_performed"] is False
    assert bootstrap["calibrators_refitted_per_replicate"] is False
    assert bootstrap["claim_scope"] == U.U1_BOOTSTRAP_CLAIM
    assert bootstrap["windows_are_independent_evidence"] is False
    for interval in bootstrap["intervals"].values():
        assert interval["requested_replicates"] == 1000
        assert (
            interval["successful_replicates"] + interval["undefined_replicates"] == 1000
        )


def test_the_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(13)
    size = 600
    subjects = np.asarray(
        [SUBJECTS[index % 12] for index in range(size)], dtype=np.str_
    )
    uncertainties = rng.random(size)
    labels = (rng.random(size) < 0.3).astype(np.int64)
    decisions = rng.random(size) < 0.3
    first = C.subject_bootstrap(
        labels=labels,
        decisions=decisions,
        uncertainties=uncertainties,
        subject_ids=subjects,
        u_star=0.9,
        replicates=50,
    )
    second = C.subject_bootstrap(
        labels=labels,
        decisions=decisions,
        uncertainties=uncertainties,
        subject_ids=subjects,
        u_star=0.9,
        replicates=50,
    )
    assert first["intervals"] == second["intervals"]


# ==========================================================================
# L. FINAL DEPLOYMENT CALIBRATOR
# ==========================================================================


def test_the_final_calibrator_reuses_the_selected_family_on_all_twelve_subjects(
    executed,
):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    deployment = json.loads((run_dir / PS.DEPLOYMENT_CALIBRATOR_NAME).read_text())
    assert deployment["selected_family"] == result["selected_family"]
    assert deployment["family_reselected"] is False
    assert deployment["fallback_to_other_family_performed"] is False
    assert deployment["fit_subjects"] == list(SUBJECTS)
    assert deployment["test_subjects_in_fit"] == []
    assert deployment["calibrator"]["fit_row_count"] == PRIMARY_ROWS
    assert deployment["is_evaluation"] is False
    assert deployment["is_parameterisation"] is True
    assert deployment["in_sample_performance_reported"] is False
    assert deployment["in_sample_performance_claim_authorised"] is False


def test_the_deployment_artifact_reports_no_in_sample_performance(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    deployment = json.loads((run_dir / PS.DEPLOYMENT_CALIBRATOR_NAME).read_text())
    text = json.dumps(deployment)
    for banned in ("brier", "expected_calibration_error", "reliability"):
        assert banned not in text


def test_u_star_deploy_is_configuration_provenance_only(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    deployment = json.loads((run_dir / PS.DEPLOYMENT_CALIBRATOR_NAME).read_text())
    assert deployment["u_star_deploy_is_scientific_evidence"] is False
    assert deployment["u_star_deploy_semantics"] == "configuration_provenance_only"
    threshold = deployment["u_star_deploy"]
    assert threshold["target_coverage"] == 0.90
    assert threshold["achieved_coverage"] >= 0.90
    assert threshold["name"] == "u_star_deploy"
    assert result["u_star_dev"]["name"] == "u_star_dev"


def test_a_failed_final_fit_stops_rather_than_falling_back(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations, monkeypatch
):
    calls = {"count": 0}
    real = C.fit_calibrator

    def failing(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 24:  # the 12 folds x 2 families are already fitted
            raise C.U1CalibrationError("synthetic final-fit failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(C, "fit_calibrator", failing)
    with pytest.raises(C.U1CalibrationError, match="synthetic final-fit failure"):
        _execute(tmp_path, corpus)

    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    assert not (run_dir / PS.RESULT_NAME).exists()
    assert not (run_dir / PS.DEPLOYMENT_CALIBRATOR_NAME).exists()
    status = json.loads((run_dir / PS.RUN_STATUS_NAME).read_text())
    assert status["status"] == PS.STATUS_FAILED
    assert status["automatic_retry_performed"] is False
    receipt = json.loads(
        (
            _roots(tmp_path)["run_root"]
            / f"{R.CANONICAL_RUN_ID}__review"
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["failed_stage"] == "final_deployment_calibrator"
    assert receipt["attempt_consumed"] is True
    assert receipt["promotion_state_source"] == "filesystem"
    assert PS.OOF_RESULT_NAME in receipt["promoted_artifacts"]


# ==========================================================================
# M. ARTIFACT / LOCK
# ==========================================================================


def test_the_result_binds_every_component_and_the_lock_binds_the_result(executed):
    result, tmp_path, _corpus_ = executed
    from cardiosentinel.data.provenance import sha256_file

    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    payload = json.loads((run_dir / PS.RESULT_NAME).read_text())
    for name, digest in payload["component_sha256"].items():
        assert sha256_file(run_dir / name) == digest
    assert sorted(payload["component_sha256"]) == sorted(PS.COMPONENT_ARTIFACTS)

    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    PS.validate_u1_run_lock(lock, run_dir=run_dir)
    assert lock["artifact_sha256"][PS.RESULT_NAME] == sha256_file(
        run_dir / PS.RESULT_NAME
    )


def test_the_lock_binds_every_identity_the_protocol_requires(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    assert lock["u1_protocol_sha256"] == U.U1_PROTOCOL_SHA256
    assert lock["m2_suite_sha256"] == U.U1_M2_SUITE_SHA256
    assert lock["m2g_arm_result_sha256"] == U.U1_M2G_ARM_RESULT_SHA256
    assert lock["m2g_lock_sha256"] == U.U1_M2G_LOCK_SHA256
    assert lock["split_sha256"] == U.U1_SPLIT_SHA256
    assert lock["m2_retention_decision_sha256"] == (
        "da4a05b4e2e3dd633493b87a08ed369010fa91c9cac21d906980a658fcf2be47"
    )
    assert lock["package_count"] == 335
    assert lock["dependency_digest"] == FROZEN_DIGEST
    assert len(lock["fold_parameters"]) == 12
    assert set(lock["pooled_oof_nll"]) == {C.FAMILY_PLATT, C.FAMILY_TEMPERATURE}
    assert lock["m2_replay_invoked"] is False
    assert lock["m2_rerun_performed"] is False
    assert lock["test_accessed"] is False
    assert lock["sealed_test_state"] == "unopened"
    assert lock["automatic_retention"] is False
    assert lock["automatic_u2_transition"] is False


def test_a_mutated_lock_fails_digest_validation(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    lock["selected_family"] = "something_else"
    with pytest.raises(PS.U1PersistenceError, match="failed digest validation"):
        PS.validate_u1_run_lock(lock)


def test_a_mutated_component_breaks_the_lock_binding(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    lock = json.loads((run_dir / PS.EXPERIMENT_LOCK_NAME).read_text())
    (run_dir / PS.FOLD_MANIFEST_NAME).write_text("{}")
    with pytest.raises(PS.U1PersistenceError, match="does not match its lock digest"):
        PS.validate_u1_run_lock(lock, run_dir=run_dir)


def test_an_incomplete_result_cannot_be_promoted():
    with pytest.raises(PS.U1PersistenceError, match="does not bind every component"):
        PS.validate_u1_result_payload(
            {
                "artifact_class": PS.RESULT_CLASS,
                "experiment_id": "x",
                "component_sha256": {},
                "oof_evidence_store": {},
                "selected_family": C.FAMILY_PLATT,
                "u_star_dev": {},
                "u_star_deploy": {},
                "routing_guards": {},
                "human_review_required": True,
                "automatic_retention": False,
                "development_evidence_source": "u1_oof_development_calibration",
            }
        )


def test_a_result_claiming_automatic_retention_is_refused():
    payload = {
        "artifact_class": PS.RESULT_CLASS,
        "experiment_id": "x",
        "component_sha256": {name: "0" * 64 for name in PS.COMPONENT_ARTIFACTS},
        "oof_evidence_store": {},
        "selected_family": C.FAMILY_PLATT,
        "u_star_dev": {},
        "u_star_deploy": {},
        "routing_guards": {},
        "human_review_required": True,
        "automatic_retention": True,
        "development_evidence_source": "u1_oof_development_calibration",
    }
    with pytest.raises(PS.U1PersistenceError, match="no automatic retention"):
        PS.validate_u1_result_payload(payload)


def test_the_per_row_store_is_digest_bound_and_out_of_fold_only(executed):
    result, tmp_path, _corpus_ = executed
    workspace = _roots(tmp_path)["run_root"] / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest = json.loads((workspace / E.U1_STORE_MANIFEST_NAME).read_text())
    E.validate_u1_evidence_store(manifest, root=workspace)
    assert manifest["out_of_fold_only"] is True
    assert manifest["primary_and_challenge_merged"] is False
    assert manifest["deployment_calibrator_probabilities_present"] is False
    assert manifest["test_rows_present"] is False
    assert manifest["selected_family"] == result["selected_family"]
    assert set(manifest["row_groups"]) == {"primary_metric", "challenge_metric"}
    assert (
        result["result"]["oof_evidence_store"]["content_sha256"]
        == (manifest["content_sha256"])
    )


def test_a_mutated_row_group_is_refused_on_read(executed):
    result, tmp_path, _corpus_ = executed
    workspace = _roots(tmp_path)["run_root"] / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest = json.loads((workspace / E.U1_STORE_MANIFEST_NAME).read_text())
    (workspace / E.U1_PRIMARY_ROWS_NAME).write_bytes(b"corrupted")
    with pytest.raises(E.U1EvidenceStoreError, match="does not match its digest"):
        E.read_u1_row_group(workspace, manifest, "primary_metric")


# ==========================================================================
# Firewalls, cold start, challenge and the CLI
# ==========================================================================


def test_the_replay_firewall_refuses_a_production_symbol():
    with pytest.raises(R.U1DevelopmentRunError, match="M2 production symbols"):
        # `Path` is genuinely used by the package, so this proves the scan
        # detects a real reference rather than passing vacuously.
        R.m2_replay_firewall(forbidden=("Path",))


def test_the_replay_firewall_passes_on_the_real_package():
    firewall = R.m2_replay_firewall()
    assert firewall["violations"] == {}
    assert firewall["m2_replay_invoked"] is False
    assert firewall["b4_scorer_invoked"] is False
    assert firewall["p1_embeddings_constructed"] is False
    assert set(firewall["checked_modules"]) == set(R.U1_EXECUTION_MODULES)


def test_every_real_test_subject_is_refused_by_name():
    firewall = R.assert_test_firewall(R.canonical_roots()["split_manifest"])
    assert firewall["test_subject_count"] == 12
    assert firewall["test_subjects_refused_not_filtered"] is True
    assert firewall["test_accessed"] is False
    assert firewall["sealed_test_state"] == "unopened"
    assert firewall["calibration_test_intersection"] == []


def test_cold_start_strata_are_the_frozen_three_and_carry_no_repair(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    cold = oof["cold_start_evidence"]
    assert set(cold["strata"]) == set(U.U1_COLD_START_STRATA)
    assert cold["cold_start_threshold_defined"] is False
    assert cold["post_hoc_cold_start_repair_performed"] is False
    assert cold["retained_routing_point_altered_by_stratum_result"] is False
    assert cold["strata_match_frozen_m2g_counts"] is True
    for entry in cold["strata"].values():
        assert entry["confidence_interval_reported"] is False


def test_restratified_cold_start_bins_are_refused(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    """U1 inherits M2's strata; it never re-stratifies the population."""
    drifted = np.asarray(
        [U.U1_COLD_START_STRATA[0]] * corpus["cold_start"].shape[0], dtype=np.str_
    )
    with pytest.raises(R.U1DevelopmentRunError, match="never re-stratifies"):
        _execute(
            tmp_path,
            corpus,
            cold_start_bins=lambda _root, stable_ids: (
                drifted,
                SYNTHETIC_STREAM_CACHE_SHA256,
            ),
        )


def test_challenge_subsets_are_reported_separately_and_never_merged(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    challenge = oof["challenge_routing_evidence"]
    assert challenge["merged_into_primary_denominator"] is False
    assert set(challenge["subsets"]) == {
        "rate_related",
        "axis_shift",
        "conduction_change",
    }
    conduction = challenge["subsets"]["conduction_change"]
    assert conduction["evidence_level"] == "exploratory_descriptive"
    assert conduction["bootstrap_interval_reported"] is False
    assert conduction["is_selection_input"] is False
    for subset in challenge["subsets"].values():
        assert subset["binary_labels_invented"] is False
        assert subset["merged_into_primary"] is False
        assert subset["denominator"] == CHALLENGE_PER_FAMILY


def test_challenge_rows_are_calibrated_by_the_fold_that_held_their_subject_out(
    executed,
):
    result, tmp_path, corpus = executed
    workspace = _roots(tmp_path)["run_root"] / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest = json.loads((workspace / E.U1_STORE_MANIFEST_NAME).read_text())
    group = E.read_u1_row_group(workspace, manifest, "challenge_metric")
    assert group.row_count == 3 * CHALLENGE_PER_FAMILY
    for subject, fold in zip(
        group.arrays["subject_id"].tolist(), group.arrays["fold_index"].tolist()
    ):
        assert SUBJECTS[fold] == subject


def test_the_cli_carries_no_scientific_override(monkeypatch):
    parser = R.build_parser()
    options = {action.dest for action in parser._actions if action.dest != "help"}
    assert options == {"execute_canonical_development", "expected_git_sha"}
    for banned in (
        "subject",
        "fold",
        "calibrator",
        "coverage",
        "threshold",
        "partition",
        "retry",
        "test",
    ):
        assert not any(banned in option for option in options)


def test_the_cli_reaches_orchestration_with_only_the_two_flags(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"executed": True}

    monkeypatch.setattr(R, "execute_canonical_u1_development", fake)
    assert R.main([R.EXECUTION_FLAG, R.EXPECTED_GIT_SHA_FLAG, GIT_SHA]) == 0
    assert seen == {"expected_git_sha": GIT_SHA, "execute": True}


def test_canonical_identity_is_deterministic():
    assert R.U1_EXPERIMENT_IDENTITY == "U1_selective_v1"
    assert R.CANONICAL_RUN_ID == "u1-v1-development"
    roots = R.canonical_roots()
    assert roots["run_root"].name == "phase7-u1-development-v1"
    assert roots["m2g_evidence_root"].name == "M2-G"
    assert R.canonical_roots() == roots


def test_the_completed_run_stops_for_human_review(executed):
    result, tmp_path, _corpus_ = executed
    assert result["human_review_required"] is True
    assert result["automatic_retention"] is False
    assert result["automatic_u2_transition"] is False
    assert result["result"]["test_accessed"] is False
    assert result["result"]["sealed_test_state"] == "unopened"
    assert result["result"]["validation_accessed"] is True


# ==========================================================================
# Provenance closure: input lineage, not merely self-consistency
# ==========================================================================


def test_a_self_consistent_but_non_frozen_m2g_store_is_refused(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    """STORE A is bound; STORE B validates perfectly and must still be refused.

    Both stores are internally valid, carry the same arm, the same schema and
    the same row count. Only the score bytes differ. Self-consistency is not
    lineage.
    """
    store_a = _write_m2g_store(tmp_path / "store-a" / "M2-G", corpus)
    perturbed = np.concatenate([corpus["scores"], corpus["challenge_scores"]])
    perturbed[0] = float(np.nextafter(perturbed[0], 0.0))
    roots = _roots(tmp_path)
    store_b = _write_m2g_store(roots["m2g_evidence_root"], corpus, scores=perturbed)

    assert store_a["arm"] == store_b["arm"] == "M2-G"
    assert store_a["schema"] == store_b["schema"]
    assert store_a["row_count"] == store_b["row_count"]
    assert store_a["content_sha256"] != store_b["content_sha256"]
    # STORE B is genuinely valid in its own right.
    from cardiosentinel.neural.m2_evidence_store import validate_evidence_store_manifest

    validate_evidence_store_manifest(store_b, root=roots["m2g_evidence_root"])

    primary = _primary_population(corpus)
    challenge = _challenge_population(corpus)
    with pytest.raises(
        R.U1DevelopmentRunError,
        match="does not equal the identity bound by the frozen retained arm result",
    ):
        R.execute_canonical_u1_development(
            expected_git_sha=GIT_SHA,
            execute=True,
            _roots=roots,
            _loaders=_loaders(corpus, primary, challenge, store_a),
        )
    run_dir = roots["run_root"] / R.CANONICAL_RUN_ID
    assert not (run_dir / PS.FOLD_MANIFEST_NAME).exists()
    receipt = json.loads(
        (
            roots["run_root"]
            / f"{R.CANONICAL_RUN_ID}__review"
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["exposure"]["calibrator_fitting_started"] is False


def test_the_exact_frozen_m2g_store_identity_is_accepted(executed):
    result, tmp_path, _corpus_ = executed
    lineage = result["result"]["input_lineage"]["m2g_evidence_store"]
    assert lineage["matches_frozen_arm_result_identity"] is True
    assert lineage["self_consistent"] is True
    assert lineage["second_digest_scheme_introduced"] is False
    assert lineage["identity_source"] == "m2g_arm_result.evidence_store_identity"
    assert result["result"]["input_lineage"]["self_consistency_alone_accepted"] is False


def test_the_store_lineage_check_compares_the_whole_canonical_identity():
    frozen = {
        "schema": "s",
        "arm": "M2-G",
        "row_count": 3,
        "content_sha256": "a" * 64,
        "row_evidence_sha256": "b" * 64,
    }
    same = dict(frozen)
    assert R.require_m2g_evidence_store_lineage(same, frozen)["row_count"] == 3
    # A field OUTSIDE content_sha256 still breaks lineage.
    drifted = dict(frozen, row_evidence_sha256="c" * 64)
    with pytest.raises(R.U1DevelopmentRunError, match="differing fields"):
        R.require_m2g_evidence_store_lineage(drifted, frozen)


def test_an_absent_frozen_store_identity_stops_rather_than_trusting_the_store():
    with pytest.raises(R.U1DevelopmentRunError, match="cannot be authenticated"):
        R.require_m2g_evidence_store_lineage({"content_sha256": "a" * 64}, {})


def test_the_primary_identity_cross_link_compares_every_authority_field(executed):
    result, tmp_path, corpus = executed
    lineage = result["result"]["input_lineage"]["primary_population"]
    primary = _primary_population(corpus)
    assert lineage["compared_fields"] == sorted(primary.identity())
    assert lineage["compared_field_count"] == len(primary.identity())
    for field in (
        "population",
        "partition",
        "authority",
        "row_count",
        "counts",
        "ordered_stable_id_sha256",
        "p1_embedding_cache_sha256",
    ):
        assert field in lineage["compared_fields"]


def test_a_primary_identity_differing_on_the_p1_cache_digest_is_refused(corpus):
    primary = _primary_population(corpus)
    observed = primary.identity()
    frozen = dict(observed, p1_embedding_cache_sha256="f" * 64)
    with pytest.raises(R.U1DevelopmentRunError, match="fields differing"):
        R.require_population_identity_lineage(
            observed, frozen, name="primary_population"
        )


def test_a_primary_identity_field_absent_upstream_is_refused(corpus):
    primary = _primary_population(corpus)
    observed = primary.identity()
    frozen = {k: v for k, v in observed.items() if k != "counts"}
    with pytest.raises(R.U1DevelopmentRunError, match="fields absent upstream"):
        R.require_population_identity_lineage(
            observed, frozen, name="primary_population"
        )


def test_the_frozen_m2g_bundle_identity_is_a_superset_of_the_authority_identity():
    """The cross-link is exact because M2 recorded the authority payload verbatim."""
    roots = R.canonical_roots()
    if not roots["m2_run_root"].is_dir():
        pytest.skip("the retained M2 run root is not on this filesystem")
    identity = R.m2g_input_identity(roots)
    authority_fields = {
        "population",
        "partition",
        "authority",
        "authority_detail",
        "row_count",
        "counts",
        "ordered_stable_id_sha256",
        "p1_embedding_cache_sha256",
        "membership_derived_from_m2_scores",
        "binary_labels_present",
    }
    assert authority_fields <= set(identity["primary_population_identity"])
    challenge_fields = {
        "population",
        "partition",
        "authority",
        "authority_detail",
        "row_count",
        "counts",
        "challenge_selection_sha256",
        "ordered_stable_id_sha256",
        "binary_labels_invented",
        "membership_derived_from_m2_scores",
    }
    assert challenge_fields <= set(identity["challenge_population_identity"])


def test_the_challenge_identity_cross_link_compares_every_authority_field(executed):
    result, tmp_path, corpus = executed
    lineage = result["result"]["input_lineage"]["challenge_population"]
    challenge = _challenge_population(corpus)
    assert lineage["compared_fields"] == sorted(challenge.identity())
    for field in (
        "partition",
        "row_count",
        "counts",
        "challenge_selection_sha256",
        "ordered_stable_id_sha256",
    ):
        assert field in lineage["compared_fields"]


def test_a_matching_selection_sha_alone_does_not_satisfy_the_challenge_cross_link(
    corpus,
):
    challenge = _challenge_population(corpus)
    observed = challenge.identity()
    frozen = dict(observed, ordered_stable_id_sha256="f" * 64)
    assert (
        observed["challenge_selection_sha256"] == frozen["challenge_selection_sha256"]
    )
    with pytest.raises(R.U1DevelopmentRunError, match="fields differing"):
        R.require_population_identity_lineage(
            observed, frozen, name="challenge_population"
        )


# --------------------------------------------------------------------------
# Cold start: counts are not an identity
# --------------------------------------------------------------------------


def test_the_stream_cache_identity_comes_from_the_frozen_m2g_lock():
    roots = R.canonical_roots()
    if not roots["m2_run_root"].is_dir():
        pytest.skip("the retained M2 run root is not on this filesystem")
    identity = R.m2g_input_identity(roots)
    assert identity["stream_cache_sha256"] == (
        "a3e39137a04ebebb3b97ef6c6c614339c990a6041cf649a0ba6e3c2d43baae18"
    )
    assert "m2g_experiment_lock" in identity["stream_cache_identity_source"]
    assert "replay_population_identity" in identity["stream_cache_identity_source"]


def _mirror_retained_m2_artifacts(destination: Path) -> Path:
    """Copy ONLY the promoted JSON identity artifacts the binder reads.

    Static already-promoted metadata, no per-window evidence, no trajectory.
    """
    import shutil

    source = R.canonical_roots()["m2_run_root"]
    suite = "m2-v1-development-two-arm-recovery2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / suite).mkdir(exist_ok=True)
    shutil.copy2(
        source / suite / "M2_SUITE_RESULT.json",
        destination / suite / "M2_SUITE_RESULT.json",
    )
    for arm in ("M2-0", "M2-G"):
        arm_dir = destination / f"{suite}__{arm}"
        arm_dir.mkdir(exist_ok=True)
        for name in ("M2_ARM_RESULT.json", "M2_EXPERIMENT_LOCK.json"):
            shutil.copy2(source / f"{suite}__{arm}" / name, arm_dir / name)
    return destination


def test_a_disagreeing_stream_cache_identity_stops_rather_than_choosing_one(tmp_path):
    """If M2's two records of the cache disagree, the proof cannot be made.

    The lock's `experiment_lock_sha256` is left untouched, so the existing
    retention binder still accepts the arm; only the cross-check between the
    lock and the replay identity catches the drift.
    """
    roots = R.canonical_roots()
    if not roots["m2_run_root"].is_dir():
        pytest.skip("the retained M2 run root is not on this filesystem")
    mirrored = _mirror_retained_m2_artifacts(tmp_path / "m2-identity")

    intact = dict(roots, m2_run_root=mirrored)
    assert R.m2g_input_identity(intact)["stream_cache_sha256"]

    lock_path = (
        mirrored
        / "m2-v1-development-two-arm-recovery2__M2-G"
        / "M2_EXPERIMENT_LOCK.json"
    )
    lock = json.loads(lock_path.read_text())
    lock["stream_cache_sha256"] = "1" * 64
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(R.U1DevelopmentRunError, match="cannot be established"):
        R.m2g_input_identity(intact)


def test_same_counts_with_a_permuted_cold_start_mapping_is_refused(
    tmp_path, corpus, frozen_runtime, synthetic_frozen_populations
):
    """A different artifact with identical stratum totals must not pass.

    The permuted cache is honestly resealed, so it is internally valid and its
    aggregate counts are byte-identical. Only its content-bound identity
    differs -- which is exactly what the provenance gate compares.
    """
    rng = np.random.default_rng(21)
    permuted = np.asarray(rng.permutation(corpus["cold_start"]).tolist(), dtype=np.str_)
    assert not np.array_equal(permuted, corpus["cold_start"])
    for stratum in U.U1_COLD_START_STRATA:
        assert int(np.count_nonzero(permuted == stratum)) == int(
            np.count_nonzero(corpus["cold_start"] == stratum)
        )
    # An honestly resealed cache: its identity is the digest of its content.
    resealed = canonical_sha256(
        {"cold_start_bin": permuted.tolist(), "partition": "validation"}
    )
    assert resealed != SYNTHETIC_STREAM_CACHE_SHA256

    with pytest.raises(
        R.U1DevelopmentRunError, match="do not make it the same artifact"
    ):
        _execute(
            tmp_path,
            corpus,
            cold_start_bins=lambda _root, stable_ids: (permuted, resealed),
        )
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    assert not (run_dir / PS.RESULT_NAME).exists()
    receipt = json.loads(
        (
            _roots(tmp_path)["run_root"]
            / f"{R.CANONICAL_RUN_ID}__review"
            / PS.ATTEMPT_FAILURE_RECEIPT_NAME
        ).read_text()
    )
    assert receipt["failed_stage"] == "cold_start_evidence"


def test_the_stream_cache_gate_precedes_the_aggregate_count_check():
    provenance = R.require_stream_cache_identity("a" * 64, "a" * 64)
    assert provenance["matches_frozen_m2_identity"] is True
    assert provenance["m1_replayed"] is False
    assert provenance["bins_regenerated"] is False
    assert provenance["cache_derived_from_source"] is False
    with pytest.raises(R.U1DevelopmentRunError, match="Identical stratum totals"):
        R.require_stream_cache_identity("b" * 64, "a" * 64)


def test_the_cold_start_reader_returns_the_cache_identity_it_verified():
    """The join is separate from the gate, so an injected reader cannot skip it."""
    import inspect

    source = inspect.getsource(R.load_cold_start_bins)
    assert "load_stream_store" in source
    assert 'manifest["stream_cache_sha256"]' in source
    run_source = inspect.getsource(R._run_after_claim)
    assert "require_stream_cache_identity(" in run_source


def test_the_completed_run_records_the_cold_start_cache_provenance(executed):
    result, tmp_path, _corpus_ = executed
    run_dir = _roots(tmp_path)["run_root"] / R.CANONICAL_RUN_ID
    oof = json.loads((run_dir / PS.OOF_RESULT_NAME).read_text())
    provenance = oof["cold_start_evidence"]["stream_cache_provenance"]
    assert provenance["stream_cache_sha256"] == SYNTHETIC_STREAM_CACHE_SHA256
    assert provenance["identity_source"] == "m2g_experiment_lock.stream_cache_sha256"
    assert oof["cold_start_evidence"]["strata_match_frozen_m2g_counts"] is True


# --------------------------------------------------------------------------
# Output evidence is part of canonical lock validation
# --------------------------------------------------------------------------


def test_canonical_validation_succeeds_with_intact_sibling_evidence(executed):
    result, tmp_path, _corpus_ = executed
    verified = PS.validate_canonical_u1_attempt(
        _roots(tmp_path)["run_root"], R.CANONICAL_RUN_ID
    )
    assert verified["verified"] is True
    assert (
        verified["oof_evidence_store_sha256"]
        == (result["lock"]["oof_evidence_store_sha256"])
    )
    assert sorted(verified["component_sha256"]) == sorted(PS.COMPONENT_ARTIFACTS)


def test_canonical_lock_validation_fails_after_primary_npz_mutation(executed):
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    (workspace / E.U1_PRIMARY_ROWS_NAME).write_bytes(b"mutated")
    with pytest.raises(PS.U1PersistenceError, match="not intact"):
        PS.validate_u1_run_lock(result["lock"], run_dir=run_root / R.CANONICAL_RUN_ID)
    with pytest.raises(PS.U1PersistenceError, match="not intact"):
        PS.validate_canonical_u1_attempt(run_root, R.CANONICAL_RUN_ID)


def test_canonical_lock_validation_fails_after_challenge_npz_mutation(executed):
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    (workspace / E.U1_CHALLENGE_ROWS_NAME).write_bytes(b"mutated")
    with pytest.raises(PS.U1PersistenceError, match="not intact"):
        PS.validate_canonical_u1_attempt(run_root, R.CANONICAL_RUN_ID)


def test_canonical_lock_validation_fails_after_manifest_mutation(executed):
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest_path = workspace / E.U1_STORE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_family"] = C.FAMILY_PLATT
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PS.U1PersistenceError, match="not intact"):
        PS.validate_canonical_u1_attempt(run_root, R.CANONICAL_RUN_ID)


def test_canonical_lock_validation_fails_when_the_store_digest_moves(executed):
    """A resealed manifest is still refused: the LOCK binds the old digest."""
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest_path = workspace / E.U1_STORE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["clamp_delta"] = 1e-6
    manifest.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        PS.U1PersistenceError, match="has changed since the lock was written"
    ):
        PS.validate_canonical_u1_attempt(run_root, R.CANONICAL_RUN_ID)


def test_canonical_lock_validation_fails_when_the_evidence_store_is_absent(executed):
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    (workspace / E.U1_STORE_MANIFEST_NAME).unlink()
    with pytest.raises(PS.U1PersistenceError, match="but none is present"):
        PS.validate_canonical_u1_attempt(run_root, R.CANONICAL_RUN_ID)


def test_the_lock_digest_equals_the_validated_manifest_content_identity(executed):
    result, tmp_path, _corpus_ = executed
    run_root = _roots(tmp_path)["run_root"]
    workspace = run_root / f"{R.CANONICAL_RUN_ID}__evidence"
    manifest = E.validate_u1_evidence_store(
        json.loads((workspace / E.U1_STORE_MANIFEST_NAME).read_text()), root=workspace
    )
    assert manifest["content_sha256"] == result["lock"]["oof_evidence_store_sha256"]
    binding = PS.validate_u1_evidence_binding(
        result["lock"], run_dir=run_root / R.CANONICAL_RUN_ID
    )
    assert binding["matches_lock"] is True
    assert sorted(binding["row_group_sha256"]) == [
        "challenge_metric",
        "primary_metric",
    ]


def test_the_evidence_workspace_path_is_derived_not_supplied():
    import inspect

    signature = inspect.signature(PS.validate_u1_evidence_binding)
    assert set(signature.parameters) == {"lock", "run_dir"}
    source = inspect.getsource(PS.validate_u1_evidence_binding)
    assert "u1_evidence_workspace(" in source
    assert 'lock["experiment_id"]' in source


def test_provenance_closure_changed_no_scientific_rule():
    assert U.U1_PROTOCOL_SHA256 == (
        "d6235b477af278fe051822bdcccb54f985e4eceb0c6e92c1424f5e9d7d79b33b"
    )
    assert U.validate_u1_protocol_document() == U.U1_PROTOCOL_SHA256
    assert U.U1_RETAINED_COVERAGE == 0.90
    assert U.U1_CLAMP_DELTA == 1e-7
    assert U.U1_SATURATED_FRACTION_REVIEW_BOUND == 0.01
    assert U.U1_NLL_TIE_TOLERANCE == 1e-4
    assert U.U1_BOOTSTRAP_REPLICATES == 1000 and U.U1_BOOTSTRAP_SEED == 2026
    assert U.U1_CLASSIFICATION_THRESHOLD == 0.7554003000259399
    assert U.U1_FOLD_COUNT == 12
    assert C.U1_OPTIMIZER == "L-BFGS-B" and C.U1_OPTIMIZER_MAXITER == 500
    assert C.U1_OPTIMIZER_GTOL == 1e-10
