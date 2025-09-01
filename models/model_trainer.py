"""Core fold training functionality for cross-validation."""
from __future__ import annotations

from time import perf_counter
from typing import Any
import numpy as np

from pydantic import BaseModel, ConfigDict
from sklearn.preprocessing import StandardScaler

from experiment_configs.model import is_regressor
from models.scalers import IdentityScaler


class Model:

    def __init__(self,
                 ModelClass,  # sklearn-like model class
                 model_params: dict[str, Any]):
        self.ModelClass = ModelClass
        self.model_params = model_params
        self.estimator = ModelClass(**model_params)
        self.X_scaler = StandardScaler()
        if is_regressor(self.estimator):
            self.y_scaler = StandardScaler()
        else:
            self.y_scaler = IdentityScaler()

    def ensure_numpy(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Ensure inputs are numpy arrays."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y).ravel()
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        X, y = self.ensure_numpy(X, y)
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        self.estimator.fit(X_scaled, y_scaled)

    def predict(self, X: np.ndarray,
                inverse_transform: bool = True
                ) -> np.ndarray:
        """Predict using the fitted model."""

        X, _ = self.ensure_numpy(X, None)

        # Ensure X is 2D - reshape if it's 1D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.X_scaler.transform(X)
        if not is_regressor(self.estimator):
            if hasattr(self.estimator, 'predict_proba'):
                # For classification, return probabilities
                y_pred_scaled = self.estimator.predict_proba(X_scaled)[:, 1]
            else:
                y_pred_scaled = self.estimator.predict(X_scaled)
        else:
            y_pred_scaled = self.estimator.predict(X_scaled)

        if inverse_transform:
            return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred_scaled

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and transform the data."""
        X, y = self.ensure_numpy(X, y)
        self.fit(X, y)
        return self.predict(X)


def run_train_and_predict(
    ModelClass,  # sklearn-like model class
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_idx: int,
    verbose: int = 0,
) -> ModelPred:
    """Train one fold and return prediction record.

    Notes:
    - Applies StandardScaler independently to X and y (regression); y is inverse-transformed.
    - Test features are transformed with the training-fitted X scaler (bug fix).
    """
    t0 = perf_counter()
    est = Model(ModelClass, model_params)
    train_X = X[train_idx]
    train_y = y[train_idx]
    est.fit(train_X, train_y)
    y_pred = est.predict(X[test_idx])
    duration = perf_counter() - t0
    if verbose:
        print(f'Fold {fold_idx} trained in {duration:.3f}s')
    return ModelPred(
        model_class=ModelClass.__name__,
        model_params=model_params,
        test_indices=test_idx,
        y_true=y[test_idx],
        y_pred=y_pred,
        duration_sec=duration,
    )


class ModelPred(BaseModel):
    """Pure prediction record for a single fold (no metrics)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_class: str  # sklearn-like model class
    model_params: dict[str, Any]
    test_indices: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    duration_sec: float
