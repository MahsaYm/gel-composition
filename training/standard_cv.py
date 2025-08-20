"""Standard K-fold cross-validation implementation."""
from __future__ import annotations

from time import perf_counter
from typing import Any
import numpy as np
from pydantic import BaseModel, ConfigDict

from models.measurements import compute_classification_metrics, compute_regression_metrics
from experiment_configs.model import is_regressor
from models.model_trainer import ModelPred
from training.cv_helpers import ensure_numpy, iterate_splits, make_kfold
from models.model_trainer import run_train_and_predict


def cross_validate_model(
    ModelClass,  # sklearn-like model class
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: bool = True,
    n_splits: int | None = None,
    random_state: int | None = 42,
    verbose: int = 0,
) -> CVSummary:
    """Execute K-Fold producing only predictions (no metrics yet)."""
    X_arr, y_arr = ensure_numpy(X, y)
    n_splits = n_splits if n_splits is not None else y.shape[0]  # leave-one-out

    splitter = make_kfold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds: list[ModelPred] = []
    global_start = perf_counter()
    for fold_idx, train_idx, test_idx in iterate_splits(splitter, X_arr, y_arr):
        fold_preds = run_train_and_predict(
            ModelClass=ModelClass,
            model_params=model_params,
            X=X_arr,
            y=y_arr,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_idx=fold_idx,
            verbose=verbose,
        )
        folds.append(fold_preds)
    total_time = perf_counter() - global_start
    print(f'K-Fold CV completed in {total_time:.3f}s with {len(folds)} folds') if verbose else None
    return CVRun(
        folds=folds,
        n_splits=n_splits,
        model_class=ModelClass.__name__,  # match dataclass field type (str)
        model_params=model_params,
        total_time_sec=total_time,
        target_type='regression' if is_regressor(ModelClass) else 'classification'
    ).to_summary(verbose=verbose)


class CVSummary(BaseModel):
    """(Legacy) Summary including aggregated metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_splits: int
    metrics: dict[str, float]
    model_class: str
    model_params: dict[str, Any]
    target_type: str
    total_time_sec: float


class CVRun(BaseModel):
    """Container returned by the new CV runner (predictions only)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    folds: list[ModelPred]
    n_splits: int
    model_class: str
    model_params: dict[str, Any]
    total_time_sec: float
    target_type: str = 'regression'

    def reconstruct_full_predictions(self, y_length: int) -> dict[str, Any]:
        """Reconstruct y_true and y_pred arrays in original sample order.

        Positions not present in any test split (should not happen in standard CV) will be np.nan.
        """
        import numpy as np
        y_true_full = np.full(shape=(y_length,), fill_value=np.nan)
        y_pred_full = np.full(shape=(y_length,), fill_value=np.nan)
        for fold in self.folds:
            y_true_full[fold.test_indices] = fold.y_true
            y_pred_full[fold.test_indices] = fold.y_pred
        return {'y_true': y_true_full, 'y_pred': y_pred_full}

    # ------------------- New global metric summarization ------------------- #
    def to_summary(self, verbose: int = 0) -> CVSummary:
        """Compute global metrics on concatenated fold predictions.

        Per-fold metric reporting is intentionally skipped; returned CVSummary has
        an empty folds list (or could carry timing metadata only if extended later).
        metric_stds are set to 0.0 for backward compatibility with existing schema.
        """
        if self.target_type == 'regression':
            metrics_fn = compute_regression_metrics
        else:
            metrics_fn = compute_classification_metrics

        if not self.folds:
            metrics: dict[str, float] = {}
        else:
            y_true_all = np.concatenate([f.y_true for f in self.folds])
            y_pred_all = np.concatenate([f.y_pred for f in self.folds])
            metrics = metrics_fn(y_true_all, y_pred_all)
        summary = CVSummary(
            n_splits=self.n_splits,
            metrics=metrics,
            model_class=self.model_class,
            model_params=self.model_params,
            target_type=self.target_type,
            total_time_sec=self.total_time_sec,
        )
        if verbose and metrics:
            print('Global CV Metrics: ' + ', '.join(f"{k}={v:.4f}" for k, v in metrics.items()))
            print(f'Total time: {self.total_time_sec:.2f}s')
        return summary

    def get_output(self) -> tuple[np.ndarray, np.ndarray]:
        """Get concatenated y_true and y_pred arrays from all folds."""
        y_true_all = np.concatenate([f.y_true for f in self.folds])
        y_pred_all = np.concatenate([f.y_pred for f in self.folds])
        return y_true_all, y_pred_all
