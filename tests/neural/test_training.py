import random

import numpy as np
import pytest
import torch

from cardiosentinel.neural.determinism import initialize_determinism
from cardiosentinel.neural.model import B4CompactCNN
from cardiosentinel.neural.training import (
    CheckpointTracker,
    build_loss,
    build_optimizer,
    build_training_loader,
    build_validation_loader,
    run_frozen_training,
    validation_f1_threshold,
)


def test_optimizer_and_loss_match_frozen_configuration() -> None:
    model = B4CompactCNN()
    optimizer = build_optimizer(model)
    group = optimizer.param_groups[0]

    assert len(optimizer.param_groups) == 1
    assert group["lr"] == 1e-3
    assert group["weight_decay"] == 1e-4
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["amsgrad"] is False
    assert group["foreach"] is False
    assert group["fused"] is False
    assert build_loss().reduction == "mean"


def test_loaders_freeze_batch_shuffle_and_final_batch_semantics() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(257))
    training_loader = build_training_loader(dataset, workers=0)
    validation_loader = build_validation_loader(dataset, workers=0)

    assert training_loader.batch_size == 256
    assert training_loader.drop_last is False
    assert isinstance(training_loader.sampler, torch.utils.data.RandomSampler)
    assert validation_loader.batch_size == 256
    assert validation_loader.drop_last is False
    assert isinstance(
        validation_loader.sampler, torch.utils.data.SequentialSampler
    )
    assert [batch[0].numel() for batch in validation_loader] == [256, 1]


def test_checkpoint_exact_tie_keeps_earliest_epoch() -> None:
    tracker = CheckpointTracker()
    first = tracker.update(1, 0.4)
    tied = tracker.update(2, 0.4)

    assert first.save_checkpoint is True
    assert tied.save_checkpoint is False
    assert tied.best_epoch == 1


def test_small_strict_increase_saves_but_does_not_reset_patience() -> None:
    tracker = CheckpointTracker()
    tracker.update(1, 0.4)
    result = tracker.update(2, 0.4000005)

    assert result.save_checkpoint is True
    assert result.best_epoch == 2
    assert result.patience == 1


def test_early_stopping_after_four_completed_nonimprovements() -> None:
    tracker = CheckpointTracker()
    tracker.update(1, 0.5)
    decisions = [tracker.update(epoch, 0.5) for epoch in range(2, 6)]
    assert [item.stop_training for item in decisions] == [False, False, False, True]


def test_threshold_reuses_validation_only_highest_tie_rule() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert validation_f1_threshold(labels, scores) == 0.7


def test_determinism_repeats_python_numpy_and_torch() -> None:
    first_state = initialize_determinism(requested_device="cpu")
    first = (random.random(), np.random.random(), torch.rand(1).item())
    second_state = initialize_determinism(requested_device="cpu")
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
    assert first_state.deterministic_algorithms is True
    assert second_state.cudnn_benchmark is False


def test_determinism_rejects_nonfrozen_seed() -> None:
    with pytest.raises(ValueError, match="frozen seed"):
        initialize_determinism(seed=1, requested_device="cpu")


def test_frozen_training_stops_and_restores_earliest_best(
    monkeypatch, tmp_path
) -> None:
    import cardiosentinel.neural.training as training

    model = torch.nn.Linear(1, 1)
    labels = np.array([1.0, 0.0])
    scores = np.array([0.9, 0.1])
    monkeypatch.setattr(training, "train_one_epoch", lambda *args: 0.25)
    monkeypatch.setattr(training, "validation_scores", lambda *args: (labels, scores))
    monkeypatch.setattr(training, "validation_auprc", lambda *args: 0.5)

    result = run_frozen_training(
        model,
        (),
        (),
        torch.device("cpu"),
        tmp_path / "checkpoint.pt",
    )

    assert len(result.history) == 5
    assert result.selected_checkpoint_epoch == 1
    assert result.selected_validation_auprc == 0.5
    assert result.validation_threshold == 0.9
