"""The T2 training and execution harness, proven synthetically (§37 D-P).

No real TRAIN optimisation, no real internal-dev scoring, no real outer
VALIDATION scoring and no TEST access happens anywhere in this file. Every
timeline is a **genuinely valid** synthetic M1 stream cache built in `tmp_path`
by `t2_fixtures`, every tensor is random, and the one real artifact ever read is
a frozen manifest digest.

The fixture being genuinely valid is what gives these tests their force: the
loader is `m1_experiment.load_stream_store`, so a fixture that merely looked
right would have forced a weaker check into the production path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cardiosentinel.neural import t2_evaluation as EV
from cardiosentinel.neural import t2_persistence as PS
from cardiosentinel.neural import t2_timeline as TL
from cardiosentinel.neural import t2_training as TR
from cardiosentinel.neural.t2_models import build_t2_model
from cardiosentinel.neural.t2_protocol import (
    ROLE_CHALLENGE_CONTEXT,
    ROLE_OTHER_NONPRIMARY_CONTEXT,
    ROLE_PRIMARY_DIRECT_LOSS,
    ROLE_UNAVAILABLE_NO_STATE_UPDATE,
    T2_ARM_GRU,
    T2_ARM_S4D,
    T2_INPUT_DIM,
    T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT,
    T2_TBPTT_LENGTH,
)
from tests.neural.t2_fixtures import (
    PRIMARY_NEGATIVE,
    SyntheticStream,
    build_environment,
)

WINDOW = 2500
STRIDE = 1250


def _timeline_environment(
    root: Path,
    layout: list[tuple[str, int, int]],
    *,
    unavailable: dict[tuple[str, int], set[int]] | None = None,
):
    """A synthetic environment from a `(record, channel, rows)` layout.

    Repeated keys are given a start offset so their stable ids stay unique: the
    interleaving under test is a re-appearing stream key, not a duplicated row.
    """
    gaps = unavailable or {}
    offsets: dict[tuple[str, int], int] = {}
    streams: list[SyntheticStream] = []
    for record, channel, count in layout:
        key = (record, channel)
        offset = offsets.get(key, 0)
        offsets[key] = offset + count
        streams.append(
            SyntheticStream(
                record_id=record,
                channel_index=channel,
                families=tuple(PRIMARY_NEGATIVE for _ in range(count)),
                unavailable=frozenset(gaps.get(key, set())),
                start_offset=offset,
            )
        )
    return build_environment(root, streams=tuple(streams))


@pytest.fixture()
def synthetic_train(tmp_path):
    """A small three-stream synthetic TRAIN timeline.

    Deliberately not 2,208,431 rows: materialising the frozen count would write
    1.3 GB per test. The frozen-count gate is proven directly against
    `require_frozen_row_count`, and the fixture path records
    `frozen_row_count_enforced: False` so it can never be mistaken for corpus
    evidence.
    """
    environment = _timeline_environment(
        tmp_path / "env",
        [("s20011", 0, 600), ("s20011", 1, 300), ("s20021", 0, 120)],
    )
    return environment.stream_cache_root


# --- D. full timeline -----------------------------------------------------


def test_timeline_opens_and_reports_its_identity(synthetic_train):
    with TL.T2Timeline("train", root=synthetic_train) as timeline:
        identity = timeline.identity()
        assert identity["row_count"] == 1020
        assert identity["representation_dim"] == 146
        assert identity["canonical_source"] is False
        assert identity["frozen_row_count_enforced"] is False
        assert identity["stream_count"] == 3
        assert identity["p1_embedding_cache_used_as_timeline"] is False
        assert identity["negative_sampling_applied"] is False
        assert identity["test_accessed"] is False
        assert identity["sealed_test_state"] == "unopened"


def test_a_thinned_timeline_is_refused():
    """The canonical route accepts exactly the frozen count and nothing else."""
    assert TL.require_frozen_row_count("train", 2_208_431) == 2_208_431
    assert TL.require_frozen_row_count("validation", 492_904) == 492_904
    for short in (2_208_430, 374_452, 50):
        with pytest.raises(TL.T2TimelineError, match="thinned by nothing at all"):
            TL.require_frozen_row_count("train", short)


def test_the_three_to_one_selection_count_is_refused_as_a_train_timeline():
    """374,452 is the P1 3:1 selection; it is not a timeline length."""
    with pytest.raises(TL.T2TimelineError, match="thinned by nothing at all"):
        TL.require_frozen_row_count("train", 374_452)


def test_a_wrong_stream_cache_digest_is_refused():
    manifest = {
        "stream_cache_sha256": "0" * 64,
        "representation_content_sha256": TL.EXPECTED_REPRESENTATION_SHA256["train"],
    }
    with pytest.raises(TL.T2TimelineError, match="stream cache digests to"):
        TL.require_frozen_stream_identity("train", manifest)


def test_a_wrong_representation_digest_is_refused():
    manifest = {
        "stream_cache_sha256": TL.EXPECTED_STREAM_CACHE_SHA256["train"],
        "representation_content_sha256": "0" * 64,
    }
    with pytest.raises(TL.T2TimelineError, match="representation content digests"):
        TL.require_frozen_stream_identity("train", manifest)


def test_the_p1_three_to_one_cache_is_refused_as_a_timeline_source():
    with pytest.raises(TL.T2TimelineError, match="3:1"):
        TL.refuse_forbidden_source("cardiosentinel-features/p1-b4b-embeddings-v1/train")
    with pytest.raises(TL.T2TimelineError, match="3:1"):
        TL.refuse_forbidden_source(Path("somewhere/p1_embeddings.npz"))
    for digest in TL.FORBIDDEN_TIMELINE_DIGESTS:
        with pytest.raises(TL.T2TimelineError, match="P1 embedding cache"):
            TL.require_not_forbidden_digest(digest)


def test_streams_are_contiguous_and_chronological(synthetic_train):
    with TL.T2Timeline("train", root=synthetic_train) as timeline:
        streams = timeline.streams()
        assert [stream.key for stream in streams] == [
            ("s20011", 0),
            ("s20011", 1),
            ("s20021", 0),
        ]
        assert [stream.row_count for stream in streams] == [600, 300, 120]
        assert timeline.subjects() == ("ltstdb:s2001", "ltstdb:s2002")


def test_an_interleaved_stream_is_refused(tmp_path):
    """A key that reappears after another stream is not one causal stream."""
    environment = _timeline_environment(
        tmp_path / "env",
        [("s20011", 0, 5), ("s20021", 0, 5), ("s20011", 0, 30)],
    )
    with pytest.raises(TL.T2TimelineError, match="more than one span"):
        TL.T2Timeline("train", root=environment.stream_cache_root).streams()


def test_primary_only_and_challenge_only_replays_are_refused_by_role_semantics():
    """Masks select scores; they never shorten the replay."""
    roles = np.array(
        [
            ROLE_PRIMARY_DIRECT_LOSS,
            ROLE_CHALLENGE_CONTEXT,
            ROLE_OTHER_NONPRIMARY_CONTEXT,
            ROLE_PRIMARY_DIRECT_LOSS,
        ]
    )
    context = TL.context_mask(roles)
    assert context.tolist() == [True, True, True, True]
    assert TL.direct_loss_mask(roles).tolist() == [True, False, False, True]
    # every available role is context: a challenge-only or primary-only replay
    # would drop rows the context mask keeps
    assert int(context.sum()) > int(TL.direct_loss_mask(roles).sum())


def test_unavailable_row_is_a_state_no_op(tmp_path):
    environment = _timeline_environment(
        tmp_path / "env", [("s20011", 0, 40)], unavailable={("s20011", 0): {3}}
    )
    with TL.T2Timeline("train", root=environment.stream_cache_root) as timeline:
        states = timeline.observation_state(0, 6)
        assert states[3] == T2_OBSERVATION_UNAVAILABLE_EXACT_FLAT
        roles = TL.assign_row_roles(
            categories=np.array(["background_negative"] * 6),
            observation_state=states,
        )
        assert roles[3] == ROLE_UNAVAILABLE_NO_STATE_UPDATE
        assert TL.context_mask(roles).tolist() == [
            True,
            True,
            True,
            False,
            True,
            True,
        ]


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_skipping_an_unavailable_row_equals_carrying_state_unchanged(arm):
    """AVAILABLE A, UNAVAILABLE, AVAILABLE B == A then B with state carried."""
    model = build_t2_model(arm)
    model.eval()
    generator = torch.Generator().manual_seed(3)
    block_a = torch.randn(1, 4, T2_INPUT_DIM, generator=generator)
    block_b = torch.randn(1, 4, T2_INPUT_DIM, generator=generator)
    with torch.no_grad():
        _, state_after_a = model(block_a)
        via_carry, _ = model(block_b, state_after_a)
        compacted = torch.cat([block_a, block_b], dim=1)
        whole, _ = model(compacted)
    assert torch.allclose(whole[:, 4:], via_carry, atol=1e-5)


def test_compaction_drops_only_unavailable_rows():
    representation = np.arange(5 * T2_INPUT_DIM, dtype=np.float32).reshape(5, -1)
    available = np.array([1, 1, 0, 1, 0], dtype=bool)
    values, positions = TR.compact_available(representation, available)
    assert positions.tolist() == [0, 1, 3]
    assert values.shape == (3, T2_INPUT_DIM)
    assert np.array_equal(values[2], representation[3])


# --- E. loss masking ------------------------------------------------------


def test_only_primary_rows_receive_direct_loss():
    logits = torch.tensor([[0.5, -0.5, 2.0, 1.0]])
    targets = torch.tensor([[1, 1, 0, 1]])
    roles = np.array(
        [
            ROLE_PRIMARY_DIRECT_LOSS,
            ROLE_CHALLENGE_CONTEXT,
            ROLE_OTHER_NONPRIMARY_CONTEXT,
            ROLE_PRIMARY_DIRECT_LOSS,
        ]
    )
    mask = torch.from_numpy(TL.direct_loss_mask(roles)).unsqueeze(0)
    total, count = TR.direct_loss_sum(logits, targets, mask, pos_weight=3.0)
    assert count == 2
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0, [0, 3]],
        targets[0, [0, 3]].float(),
        pos_weight=torch.tensor(3.0),
        reduction="sum",
    )
    assert torch.allclose(total, expected)


def test_a_chunk_with_no_primary_rows_produces_no_loss():
    logits = torch.tensor([[0.5, -0.5]])
    targets = torch.tensor([[1, 0]])
    mask = torch.zeros_like(logits, dtype=torch.bool)
    total, count = TR.direct_loss_sum(logits, targets, mask, pos_weight=2.0)
    assert count == 0
    assert float(total) == 0.0


def test_challenge_context_can_causally_alter_a_later_primary_output():
    """Label-blind context reaches later PRIMARY rows -- the honest claim."""
    model = build_t2_model(T2_ARM_GRU)
    model.eval()
    generator = torch.Generator().manual_seed(7)
    prefix = torch.randn(1, 3, T2_INPUT_DIM, generator=generator)
    other_prefix = torch.randn(1, 3, T2_INPUT_DIM, generator=generator)
    primary = torch.randn(1, 2, T2_INPUT_DIM, generator=generator)
    with torch.no_grad():
        _, state_one = model(prefix)
        first, _ = model(primary, state_one)
        _, state_two = model(other_prefix)
        second, _ = model(primary, state_two)
    assert not torch.allclose(first, second), (
        "different causal context must be able to change a later PRIMARY score"
    )


def test_role_identity_never_becomes_a_model_input():
    """There is no code path from a role or category into the z_t vector."""
    import ast

    source = Path(TL.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    # the role helpers return masks, never a feature matrix
    for name in ("assign_row_roles", "direct_loss_mask", "context_mask"):
        assert name in functions
    assert "representation" not in {
        getattr(node, "attr", None) for node in ast.walk(functions["assign_row_roles"])
    }


# --- F. synchronized TBPTT ------------------------------------------------


def test_tbptt_length_is_exactly_256_and_refuses_substitutes():
    assert T2_TBPTT_LENGTH == 256
    with pytest.raises(TR.T2TrainingError, match="frozen TBPTT length is 256"):
        TR.synchronized_frontiers([300], tbptt=128)


def test_active_streams_batch_together_at_each_frontier():
    frontiers = TR.synchronized_frontiers([600, 256, 100])
    assert len(frontiers) == 3
    assert [len(frontier) for frontier in frontiers] == [3, 1, 1]
    first = TR.group_by_length(frontiers[0])
    assert sorted(first) == [100, 256]
    assert len(first[256]) == 2, "equal-length chunks share one batch"


def test_finished_streams_leave_the_batch(tmp_path):
    frontiers = TR.synchronized_frontiers([256, 512])
    assert {item.stream_index for item in frontiers[0]} == {0, 1}
    assert {item.stream_index for item in frontiers[1]} == {1}


def test_no_padding_is_required_because_groups_have_equal_length():
    for frontier in TR.synchronized_frontiers([600, 431, 100, 256]):
        for length, group in TR.group_by_length(frontier).items():
            assert all(item.length == length for item in group)


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_no_cross_stream_state_leakage(arm):
    """Two streams in one batch must score exactly as they would alone."""
    model = build_t2_model(arm)
    model.eval()
    generator = torch.Generator().manual_seed(13)
    stream_a = torch.randn(1, 5, T2_INPUT_DIM, generator=generator)
    stream_b = torch.randn(1, 5, T2_INPUT_DIM, generator=generator)
    states = TR.initial_states(model, 2)
    with torch.no_grad():
        batched, _ = TR.run_stream_group(
            model, torch.cat([stream_a, stream_b], dim=0), states, [0, 1]
        )
        alone_a, _ = model(stream_a)
        alone_b, _ = model(stream_b)
    assert torch.allclose(batched[0:1], alone_a, atol=1e-5)
    assert torch.allclose(batched[1:2], alone_b, atol=1e-5)


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_state_carries_across_frontiers_and_detaches(arm):
    model = build_t2_model(arm)
    model.train()
    states = TR.initial_states(model, 1)
    values = torch.randn(1, 4, T2_INPUT_DIM)
    TR.run_stream_group(model, values, states, [0])
    carried = TR.detach_all(states)
    tensors = [carried[0]] if isinstance(carried[0], torch.Tensor) else list(carried[0])
    assert all(tensor.grad_fn is None for tensor in tensors)


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_each_epoch_starts_from_zero_state(arm):
    model = build_t2_model(arm)
    first = TR.initial_states(model, 2)
    second = TR.initial_states(model, 2)
    for left, right in zip(first, second, strict=True):
        tensors_l = [left] if isinstance(left, torch.Tensor) else list(left)
        tensors_r = [right] if isinstance(right, torch.Tensor) else list(right)
        for a, b in zip(tensors_l, tensors_r, strict=True):
            assert torch.equal(a, b)
            assert bool(torch.all(a == 0))


# --- G. loss reduction ----------------------------------------------------


def test_loss_reduction_is_sum_over_direct_rows_divided_by_their_count():
    logits = torch.tensor([[0.2, 1.5], [-0.7, 0.9]])
    targets = torch.tensor([[1, 0], [0, 1]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    total, count = TR.direct_loss_sum(logits, targets, mask, pos_weight=2.5)
    assert count == 4
    mean = total / count
    reference = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1),
        targets.reshape(-1).float(),
        pos_weight=torch.tensor(2.5),
        reduction="mean",
    )
    assert torch.allclose(mean, reference)


def test_positive_class_weight_is_negatives_over_positives():
    assert TR.positive_class_weight(negative_count=300, positive_count=100) == 3.0
    with pytest.raises(TR.T2TrainingError, match="both classes"):
        TR.positive_class_weight(negative_count=0, positive_count=10)


# --- H. internal split ----------------------------------------------------


def test_internal_split_is_48_8_and_touches_no_outer_partition():
    from cardiosentinel.neural.t2_protocol import (
        T2_INTERNAL_SPLIT_SHA256,
        assign_internal_split,
        validate_internal_split,
    )

    split_path = Path("protocols/splits/ltstdb_v1.json")
    if not split_path.is_file():
        pytest.skip("the frozen split manifest is not on this filesystem")
    manifest = json.loads(split_path.read_text())
    partitions = manifest["partitions"]
    assignment = assign_internal_split(partitions["train"]["subjects"])
    assert assignment["split_sha256"] == T2_INTERNAL_SPLIT_SHA256
    validate_internal_split(
        assignment,
        validation_subjects=partitions["validation"]["subjects"],
        test_subjects=partitions["test"]["subjects"],
    )
    fit = set(assignment["fit_subjects"])
    dev = set(assignment["internal_dev_subjects"])
    assert len(fit) == 48 and len(dev) == 8 and not (fit & dev)


# --- J. checkpoint selection ----------------------------------------------


def _epoch(number: int, auprc: float) -> TR.T2EpochResult:
    return TR.T2EpochResult(
        epoch=number,
        optimizer_step_count=10,
        zero_direct_loss_frontier_count=0,
        direct_loss_row_count=100,
        weighted_loss_sum=100.0,
        mean_weighted_loss_per_direct_row=1.0,
        internal_dev_pooled_auprc=auprc,
    )


def test_higher_internal_dev_auprc_wins():
    selector = TR.T2CheckpointSelector()
    assert selector.offer(_epoch(1, 0.30)) is True
    assert selector.offer(_epoch(2, 0.42)) is True
    assert selector.best_epoch == 2


def test_exact_tie_keeps_the_earlier_epoch():
    selector = TR.T2CheckpointSelector()
    selector.offer(_epoch(1, 0.40))
    assert selector.offer(_epoch(2, 0.40)) is False
    assert selector.best_epoch == 1


def test_patience_is_exactly_three_consecutive_non_improvements():
    selector = TR.T2CheckpointSelector()
    selector.offer(_epoch(1, 0.50))
    assert selector.should_stop is False
    for number in (2, 3):
        selector.offer(_epoch(number, 0.10))
        assert selector.should_stop is False
    selector.offer(_epoch(4, 0.10))
    assert selector.should_stop is True


def test_an_improvement_resets_patience():
    selector = TR.T2CheckpointSelector()
    selector.offer(_epoch(1, 0.50))
    selector.offer(_epoch(2, 0.10))
    selector.offer(_epoch(3, 0.60))
    assert selector.patience == 0
    assert selector.best_epoch == 3


def test_non_finite_auprc_is_a_hard_failure():
    selector = TR.T2CheckpointSelector()
    with pytest.raises(TR.T2TrainingError, match="hard failure"):
        selector.offer(_epoch(1, float("nan")))


def test_selector_records_the_frozen_limits():
    selector = TR.T2CheckpointSelector()
    selector.offer(_epoch(1, 0.4))
    payload = selector.as_dict()
    assert payload["criterion"] == "internal_development_pooled_auprc"
    assert payload["tie_break"] == "earlier_epoch"
    assert payload["patience_epochs"] == 3
    assert payload["max_epochs"] == 10


@pytest.mark.parametrize("arm", [T2_ARM_GRU, T2_ARM_S4D])
def test_checkpoint_reload_reproduces_predictions_and_refuses_mutation(tmp_path, arm):
    model = build_t2_model(arm)
    model.eval()
    values = torch.randn(1, 6, T2_INPUT_DIM)
    with torch.no_grad():
        before, _ = model(values)
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    from cardiosentinel.data.provenance import sha256_file

    digest = sha256_file(path)

    restored = build_t2_model(arm)
    restored.load_state_dict(PS.load_checkpoint(path, expected_sha256=digest))
    restored.eval()
    with torch.no_grad():
        after, _ = restored(values)
    assert torch.equal(before, after)

    with pytest.raises(PS.T2PersistenceError, match="mutated checkpoint is refused"):
        PS.load_checkpoint(path, expected_sha256="0" * 64)


# --- K. threshold ---------------------------------------------------------


def test_maximum_f1_threshold_matches_the_repository_convention():
    from cardiosentinel.evaluation.metrics import select_validation_f1_threshold

    generator = np.random.default_rng(4)
    labels = generator.integers(0, 2, 300).tolist()
    scores = generator.random(300).round(3).tolist()
    assert TR.maximum_f1_threshold(labels, scores) == (
        select_validation_f1_threshold(labels, scores, partition="validation")
    )


def test_threshold_tie_break_keeps_the_highest_threshold():
    """Two thresholds give the same F1; the higher one must win."""
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    # every threshold in (0.2, 0.8] yields F1 = 1.0; the sweep must keep 0.8
    assert TR.maximum_f1_threshold(labels, scores) == 0.8


def test_threshold_evidence_is_train_development_not_outer_validation():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    evidence = TR.internal_dev_threshold_evidence(labels, scores)
    assert evidence["partition"] == "t2_internal_dev_8_subjects"
    assert evidence["rule"] == "exact_maximum_f1_highest_threshold_tie_break"
    assert evidence["derived_before_outer_validation"] is True
    assert evidence["outer_validation_may_alter"] is False
    assert evidence["is_outer_validation_evidence"] is False
    assert evidence["is_train_development_evidence"] is True
    assert evidence["auprc"] is not None


def test_pooled_auprc_refuses_a_single_class():
    with pytest.raises(TR.T2TrainingError, match="hard failure"):
        TR.pooled_auprc([0, 0, 0], [0.1, 0.2, 0.3])


# --- L. arm selection -----------------------------------------------------


def test_training_declares_no_winner():
    status = EV.training_selection_status()
    assert status["arm_selection_status"] == "pending_one_shot_outer_validation"
    assert status["selected_arm"] is None
    assert set(status["candidates"]) == {T2_ARM_GRU, T2_ARM_S4D}


def test_selection_is_delegated_to_the_frozen_protocol_rule():
    import inspect

    from cardiosentinel.neural.t2_protocol import select_t2_arm

    source = inspect.getsource(EV.select_from_outer_validation)
    assert "select_t2_arm(" in source
    decision = EV.select_from_outer_validation(
        pooled_auprc={T2_ARM_GRU: 0.40, T2_ARM_S4D: 0.41},
        subject_macro_auprc={T2_ARM_GRU: 0.5, T2_ARM_S4D: 0.5},
        parameter_counts={T2_ARM_GRU: 59_521, T2_ARM_S4D: 45_313},
    )
    reference = select_t2_arm(
        pooled_auprc={T2_ARM_GRU: 0.40, T2_ARM_S4D: 0.41},
        subject_macro_auprc={T2_ARM_GRU: 0.5, T2_ARM_S4D: 0.5},
        parameter_counts={T2_ARM_GRU: 59_521, T2_ARM_S4D: 45_313},
    )
    assert decision == reference


def test_challenge_and_latency_cannot_influence_selection():
    decision = EV.select_from_outer_validation(
        pooled_auprc={T2_ARM_GRU: 0.40, T2_ARM_S4D: 0.41},
        subject_macro_auprc={T2_ARM_GRU: 0.9, T2_ARM_S4D: 0.1},
        parameter_counts={T2_ARM_GRU: 59_521, T2_ARM_S4D: 45_313},
    )
    assert decision["challenge_evidence_used"] is False
    assert decision["latency_used"] is False


# --- M. outer-validation firewall -----------------------------------------


def test_every_outer_validation_entry_point_refuses():
    assert EV.OUTER_VALIDATION_ENTRY_POINTS == (EV.execute_canonical_outer_validation,)
    for entry in EV.OUTER_VALIDATION_ENTRY_POINTS:
        with pytest.raises(PS.T2ActivationError, match="not authorized"):
            entry("0" * 40)


def test_the_refusal_fires_before_any_validation_path_is_resolved():
    """Called with a nonsense commit, it still refuses rather than resolving.

    The raw loaders that used to sit on this surface are gone: the only public
    outer route is the claim-bearing canonical one.
    """
    with pytest.raises(PS.T2ActivationError):
        EV.execute_canonical_outer_validation("0" * 40)
    with pytest.raises(PS.T2ActivationError):
        EV.execute_canonical_outer_validation(None)
    assert not hasattr(EV, "open_validation_timeline")
    assert not hasattr(EV, "load_validation_labels")


def test_activation_is_false_and_has_no_override():
    import ast

    assert PS.T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED is False
    source = Path(PS.__file__).read_text()
    assert "os.environ" not in source, "no environment-variable bypass"
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign | ast.Assign)
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        )
        if getattr(target, "id", None) == "T2_OUTER_VALIDATION_EXECUTION_AUTHORIZED"
    ]
    assert len(assignments) == 1, "exactly one place defines the activation state"


def test_the_cli_outer_validation_route_refuses():
    from cardiosentinel.neural.t2_development_run import main

    assert main(["--execute-canonical-outer-validation"]) == 3


def test_the_cli_exposes_no_scientific_option():
    from cardiosentinel.neural.t2_development_run import (
        FORBIDDEN_OPTIONS,
        build_parser,
    )

    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert not (options & set(FORBIDDEN_OPTIONS)), options & set(FORBIDDEN_OPTIONS)
    assert "--execute-canonical-training" in options
    assert "--expected-git-sha" in options


def test_the_cli_requires_the_expected_git_sha():
    from cardiosentinel.neural.t2_development_run import main

    assert main(["--execute-canonical-training"]) == 2


# --- N. TEST firewall -----------------------------------------------------


def test_test_partition_is_refused_by_name():
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.refuse_sealed_partition("test")
    with pytest.raises(TL.T2TimelineError, match="sealed TEST partition"):
        TL.T2Timeline("test")


def test_no_test_route_exists_anywhere_in_the_harness():
    from cardiosentinel.neural.t2_development_run import build_parser

    options = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert "--test" not in options
    assert TL.PERMITTED_PARTITIONS == ("train", "validation")


# --- O. artifacts ---------------------------------------------------------


def test_execution_spec_is_frozen():
    assert PS.validate_t2_execution_spec() == PS.T2_EXECUTION_SPEC_SHA256


def test_claim_refuses_a_consumed_attempt(tmp_path):
    root = tmp_path / "runs"
    (root / PS.T2_TRAINING_ATTEMPT_ID).mkdir(parents=True)
    with pytest.raises(PS.T2PersistenceError, match="already claimed"):
        PS.require_unclaimed_t2_attempt(root, PS.T2_TRAINING_ATTEMPT_ID)


def test_claim_identity_has_no_timestamp_or_random_suffix():
    assert PS.T2_TRAINING_ATTEMPT_ID == "t2-v1-training"
    assert PS.T2_OUTER_VALIDATION_ATTEMPT_ID == "t2-v1-outer-validation"
    assert PS.T2_EXPERIMENT_IDENTITY == "T2_temporal_v1"
    # Absolute, and anchored to the repository rather than to the shell's cwd.
    assert PS.T2_RUN_ROOT.is_absolute()
    assert PS.T2_RUN_ROOT.parts[-2:] == (
        "cardiosentinel-runs",
        "phase8-t2-development-v1",
    )
    for attempt in PS.T2_ATTEMPT_IDS:
        assert not any(character.isdigit() for character in attempt.split("-")[-1])


def test_a_result_that_selects_an_arm_is_refused():
    payload = {
        "artifact_class": PS.RESULT_CLASS,
        "attempt_id": PS.T2_TRAINING_ATTEMPT_ID,
        "authorized_git_sha": "a" * 40,
        "git_sha": "a" * 40,
        "execution_device_proof": {
            "declared_execution_device": "cpu",
            "model_parameter_device": "cpu",
            "execution_device_agrees": True,
        },
        "component_sha256": {name: "0" * 64 for name in PS.COMPONENT_ARTIFACTS},
        "checkpoint_sha256": {},
        "checkpoint_lock_sha256": {},
        "internal_dev_thresholds": {},
        "target_authority_identity": {},
        "arm_selection_status": PS.ARM_SELECTION_PENDING,
        "arm_selected": T2_ARM_GRU,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    with pytest.raises(PS.T2PersistenceError, match="may not select an arm"):
        PS.validate_t2_result_payload(payload)


def test_a_result_recording_validation_or_test_access_is_refused():
    base = {
        "artifact_class": PS.RESULT_CLASS,
        "attempt_id": PS.T2_TRAINING_ATTEMPT_ID,
        "authorized_git_sha": "a" * 40,
        "git_sha": "a" * 40,
        "execution_device_proof": {
            "declared_execution_device": "cpu",
            "model_parameter_device": "cpu",
            "execution_device_agrees": True,
        },
        "component_sha256": {name: "0" * 64 for name in PS.COMPONENT_ARTIFACTS},
        "checkpoint_sha256": {},
        "checkpoint_lock_sha256": {},
        "internal_dev_thresholds": {},
        "target_authority_identity": {},
        "arm_selection_status": PS.ARM_SELECTION_PENDING,
        "arm_selected": None,
        "outer_validation_accessed": False,
        "test_accessed": False,
        "sealed_test_state": "unopened",
    }
    with pytest.raises(PS.T2PersistenceError, match="VALIDATION access"):
        PS.validate_t2_result_payload({**base, "outer_validation_accessed": True})
    with pytest.raises(PS.T2PersistenceError, match="test_accessed=false"):
        PS.validate_t2_result_payload({**base, "test_accessed": True})


def test_an_incomplete_attempt_cannot_be_verified_as_complete(tmp_path):
    root = tmp_path / "runs"
    (root / PS.T2_TRAINING_ATTEMPT_ID).mkdir(parents=True)
    with pytest.raises(PS.T2PersistenceError, match="No canonical T2 artifact"):
        PS.validate_canonical_t2_attempt(root, PS.T2_TRAINING_ATTEMPT_ID)


def test_mixed_runtimes_between_arms_are_refused():
    first = {
        "device_type": "cpu",
        "device_name": None,
        "torch_version": "2.13.0+cpu",
        "cuda_version": None,
        "dependency_digest": "a" * 64,
        "deterministic_algorithms": True,
    }
    PS.require_single_runtime(first, dict(first))
    with pytest.raises(PS.T2PersistenceError, match="different device_type"):
        PS.require_single_runtime(first, {**first, "device_type": "cuda"})


def test_outer_validation_result_schema_requires_both_arms():
    from cardiosentinel.neural.t2_protocol import T2_VALIDATION_PRIMARY_ROW_COUNT

    payload = {name: {} for name in EV.REQUIRED_OUTER_VALIDATION_FIELDS}
    payload.update(
        {
            "artifact_class": EV.OUTER_VALIDATION_RESULT_CLASS,
            "per_arm_evidence": {T2_ARM_GRU: {}},
            "selected_arm": None,
            "test_accessed": False,
            "sealed_test_state": "unopened",
            "primary_population_identity": {
                "row_count": T2_VALIDATION_PRIMARY_ROW_COUNT
            },
        }
    )
    with pytest.raises(EV.T2EvaluationError, match="No outer-validation evidence"):
        EV.validate_outer_validation_result(payload)
    payload["per_arm_evidence"][T2_ARM_S4D] = {}
    assert EV.validate_outer_validation_result(payload) is payload


# --- descriptive metrics, cold start, challenge, bootstrap ---------------


def _descriptor_stream(
    predictions, *, record="s20011", channel=0, present=None, primary=None, labels=None
):
    size = len(predictions)
    return EV.T2DescriptorStream(
        record_id=record,
        channel_index=channel,
        predictions=np.asarray(predictions, dtype=bool),
        score_present=np.asarray(
            [True] * size if present is None else present, dtype=bool
        ),
        primary_mask=np.asarray(
            [True] * size if primary is None else primary, dtype=bool
        ),
        labels=np.asarray([0] * size if labels is None else labels, dtype=np.int64),
    )


def test_temporal_descriptors_are_descriptive_only():
    descriptors = EV.temporal_descriptors(
        [_descriptor_stream([0, 1, 1, 0, 1, 0], labels=[0, 1, 1, 0, 0, 0])]
    )
    assert descriptors["is_selection_input"] is False
    assert descriptors["may_alter_threshold"] is False
    assert descriptors["positive_prediction_run_count"] == 2
    assert descriptors["isolated_single_window_positive_fraction"] == 0.5
    assert descriptors["median_positive_run_duration_seconds"] == 7.5
    # Physical exposure counts every raw position, not just the scored ones.
    assert descriptors["physical_exposure_seconds"] == 30.0
    assert descriptors["stream_count"] == 1


def test_cold_start_reports_the_inherited_strata_without_repair():
    evidence = EV.cold_start_strata_evidence(
        ["0_5_minutes", "0_5_minutes", "over_60_minutes", "over_60_minutes"],
        [1, 0, 1, 0],
        [0.9, 0.2, 0.8, 0.1],
        0.5,
    )
    assert evidence["warmup_threshold_applied"] is False
    assert evidence["cold_start_repair_applied"] is False
    assert set(evidence["strata"]) == {
        "0_5_minutes",
        "5_60_minutes",
        "over_60_minutes",
    }
    assert evidence["strata"]["5_60_minutes"]["row_count"] == 0


def test_challenge_evidence_is_never_a_selection_input():
    evidence = EV.challenge_family_evidence(
        ["rate_related", "axis_shift", "conduction_change", "rate_related"],
        [0, 0, 0, 0],
        [0.9, 0.1, 0.9, 0.2],
        0.5,
    )
    assert evidence["is_selection_input"] is False
    assert evidence["merged_into_primary"] is False
    assert evidence["subsets"]["rate_related"]["false_positive_count"] == 1
    assert (
        evidence["subsets"]["conduction_change"]["evidence_level"]
        == "exploratory_descriptive"
    )


def test_subject_bootstrap_uses_subjects_never_windows():
    subjects = [f"ltstdb:s{2000 + index // 4}" for index in range(24)]
    labels = [index % 2 for index in range(24)]
    scores = [0.9 if index % 2 else 0.1 for index in range(24)]
    evidence = EV.subject_bootstrap_evidence(subjects, labels, scores, 0.5)
    assert evidence["replicates"] == 1000
    assert evidence["seed"] == 2026
    assert evidence["unit"] == "subject"
    assert evidence["window_bootstrap_performed"] is False
    assert evidence["model_refitted_per_replicate"] is False
    assert evidence["claim_scope"] == (
        "between_subject_variation_conditional_on_fitted_temporal_model"
    )


# --- P. memory safety -----------------------------------------------------


def test_the_loader_uses_read_only_memory_maps(synthetic_train):
    with TL.T2Timeline("train", root=synthetic_train) as timeline:
        array = timeline.store.array("representation.npy")
        assert isinstance(array, np.memmap) or array.base is not None
        assert array.flags.writeable is False


def test_bounded_reads_never_materialise_the_whole_representation(synthetic_train):
    with TL.T2Timeline("train", root=synthetic_train) as timeline:
        chunk = timeline.representation(0, T2_TBPTT_LENGTH)
        assert chunk.shape == (T2_TBPTT_LENGTH, T2_INPUT_DIM)
        assert chunk.nbytes < timeline.row_count * T2_INPUT_DIM * 4


def test_multi_stream_synthetic_stress_stays_bounded(synthetic_train):
    """Walk every stream in TBPTT chunks; peak resident slice stays small."""
    with TL.T2Timeline("train", root=synthetic_train) as timeline:
        largest = 0
        for stream in timeline.streams():
            frontiers = TR.synchronized_frontiers([stream.row_count])
            for frontier in frontiers[:3]:
                for item in frontier:
                    start = stream.start_index + item.local_start
                    stop = stream.start_index + item.local_stop
                    chunk = timeline.representation(start, stop)
                    largest = max(largest, chunk.nbytes)
        assert largest <= T2_TBPTT_LENGTH * T2_INPUT_DIM * 4
