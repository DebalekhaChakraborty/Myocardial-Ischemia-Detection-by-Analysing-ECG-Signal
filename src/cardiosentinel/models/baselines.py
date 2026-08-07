"""Pre-specified Phase 3B-1 classical baseline estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from cardiosentinel.evaluation.protocol import DEFAULT_SEED
from cardiosentinel.features.schema import COMBINED_V1, SIGNAL_V1, FeatureSchema

BASELINE_NAMES: Final = (
    "B0_constant_prior",
    "B1_signal_logreg",
    "B2_morphology_logreg",
    "B3_morphology_hgb",
)

LOGISTIC_PARAMETERS: Final = {
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": None,
    "random_state": DEFAULT_SEED,
}
HGB_PARAMETERS: Final = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": DEFAULT_SEED,
}


@dataclass
class ConstantPriorClassifier:
    """A control estimator that emits the unsampled training positive prior."""

    positive_prior_: float | None = None

    def fit(
        self, features: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> ConstantPriorClassifier:
        del features
        if labels.size == 0 or not np.isin(labels, (0, 1)).all():
            raise ValueError("B0 requires non-empty binary training labels.")
        return self.fit_from_counts(int(np.sum(labels)), int(labels.size))

    def fit_from_counts(
        self, positive_count: int, total_count: int
    ) -> ConstantPriorClassifier:
        """Fit the same prior without allocating unsampled training arrays."""
        if total_count <= 0 or not 0 <= positive_count <= total_count:
            raise ValueError("B0 requires valid unsampled primary training counts.")
        self.positive_prior_ = positive_count / total_count
        return self

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.positive_prior_ is None:
            raise ValueError("B0 must be fitted before prediction.")
        positive = np.full(features.shape[0], self.positive_prior_, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def feature_schema_for_baseline(name: str) -> FeatureSchema:
    """Return the frozen input schema for one named baseline."""
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {name!r}.")
    return SIGNAL_V1 if name == "B1_signal_logreg" else COMBINED_V1


def model_parameters(name: str) -> dict[str, object]:
    """Return the protocol parameters before estimator construction."""
    if name == "B0_constant_prior":
        return {"algorithm": "constant_unsampled_training_primary_prior"}
    if name in {"B1_signal_logreg", "B2_morphology_logreg"}:
        return {
            "algorithm": "sklearn.linear_model.LogisticRegression",
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "scaler": "StandardScaler",
            **LOGISTIC_PARAMETERS,
        }
    if name == "B3_morphology_hgb":
        return {
            "algorithm": "sklearn.ensemble.HistGradientBoostingClassifier",
            "imputer": {"strategy": "median", "keep_empty_features": True},
            **HGB_PARAMETERS,
        }
    raise ValueError(f"Unknown baseline {name!r}.")


def build_estimator(name: str) -> Any:
    """Construct one fixed estimator without any hyperparameter search."""
    if name == "B0_constant_prior":
        return ConstantPriorClassifier()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name in {"B1_signal_logreg", "B2_morphology_logreg"}:
        return Pipeline(
            (
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**LOGISTIC_PARAMETERS)),
            )
        )
    if name == "B3_morphology_hgb":
        return Pipeline(
            (
                ("imputer", imputer),
                ("classifier", HistGradientBoostingClassifier(**HGB_PARAMETERS)),
            )
        )
    raise ValueError(f"Unknown baseline {name!r}.")


def positive_scores(
    estimator: Any, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return finite positive-class scores from a fitted binary estimator."""
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    if probabilities.shape != (features.shape[0], 2):
        raise ValueError("Baseline predict_proba output must have two columns.")
    scores = probabilities[:, 1]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Baseline predictions must be finite probabilities.")
    return scores


def transform_snapshot(estimator: Any) -> dict[str, object]:
    """Serialize learned preprocessing state separately from model parameters."""
    if isinstance(estimator, ConstantPriorClassifier):
        return {"preprocessing": "none"}
    snapshot: dict[str, object] = {}
    imputer = estimator.named_steps["imputer"]
    snapshot["imputer_statistics"] = np.asarray(imputer.statistics_).tolist()
    if "scaler" in estimator.named_steps:
        scaler = estimator.named_steps["scaler"]
        snapshot["scaler_mean"] = np.asarray(scaler.mean_).tolist()
        snapshot["scaler_scale"] = np.asarray(scaler.scale_).tolist()
    return snapshot
