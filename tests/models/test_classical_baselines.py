"""SYNTHETIC fixed-model, prior, and train-only transform tests."""

import numpy as np
import pytest

from cardiosentinel.models.baselines import (
    HGB_PARAMETERS,
    LOGISTIC_PARAMETERS,
    ConstantPriorClassifier,
    build_estimator,
    model_parameters,
    positive_scores,
    transform_snapshot,
)


def test_b0_is_constant_and_uses_training_labels_only() -> None:
    estimator = ConstantPriorClassifier().fit(
        np.empty((4, 0)), np.asarray([0, 0, 0, 1])
    )
    scores = positive_scores(estimator, np.empty((3, 0)))
    assert np.array_equal(scores, np.full(3, 0.25))
    assert estimator.positive_prior_ == 0.25


def test_b1_and_b2_use_the_same_fixed_logistic_pipeline() -> None:
    for name in ("B1_signal_logreg", "B2_morphology_logreg"):
        estimator = build_estimator(name)
        assert tuple(estimator.named_steps) == ("imputer", "scaler", "classifier")
        classifier = estimator.named_steps["classifier"]
        for key, expected in LOGISTIC_PARAMETERS.items():
            assert classifier.get_params()[key] == expected
        assert model_parameters(name)["class_weight"] is None


def test_b3_parameters_are_frozen_without_early_stopping() -> None:
    estimator = build_estimator("B3_morphology_hgb")
    classifier = estimator.named_steps["classifier"]
    for key, expected in HGB_PARAMETERS.items():
        assert classifier.get_params()[key] == expected


def test_validation_extreme_cannot_change_training_imputer_or_scaler() -> None:
    training = np.asarray([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0], [4.0, 30.0]])
    labels = np.asarray([0, 0, 1, 1])
    estimator = build_estimator("B1_signal_logreg").fit(training, labels)
    before = transform_snapshot(estimator)
    positive_scores(estimator, np.asarray([[1e12, -1e12], [np.nan, np.nan]]))
    after = transform_snapshot(estimator)
    assert after == before
    assert before["imputer_statistics"] == [2.5, 20.0]


def test_positive_scores_reject_non_probability_output() -> None:
    class InvalidEstimator:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.full((len(features), 2), np.inf)

    with pytest.raises(ValueError, match="finite"):
        positive_scores(InvalidEstimator(), np.zeros((2, 1)))
