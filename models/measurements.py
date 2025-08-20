"""Small, pure regression metric functions.

All functions accept array-like inputs and internally convert to NumPy arrays.
Returned metrics are floats (Python scalars).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from typing import Any, Iterable, Literal, cast
from sklearn.metrics import average_precision_score

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

AverageType = Literal['micro', 'macro', 'samples', 'weighted', 'binary']
RocAverageType = Literal['micro', 'macro', 'samples', 'weighted']
MultiClassType = Literal['raise', 'ovr', 'ovo']

LOWER_IS_BETTER = {
    'mse': True,
    'rmse': True,
    'mae': True,
    'r2': False,
    'explained_variance': False,
    'accuracy': False,
    'balanced_accuracy': False,
    'precision': False,
    'recall': False,
    'f1': False,
    'roc_auc': False,
    'cross_entropy': True,
}

# ------------------------- regression metrics ---------------------------- #


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(mean_squared_error(y_t, y_p))


def average_precisione(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (average_precisionE).
    Note: average_precisionE can be undefined if y_true contains zeros.
    """
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_t - y_p) / y_t))) if np.any(y_t) else float('nan')


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(mean_absolute_error(y_t, y_p))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(r2_score(y_t, y_p))


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t, y_p = np.asarray(y_true), np.asarray(y_pred)
    return float(explained_variance_score(y_t, y_p))

# ------------------------- classification metrics ---------------------------- #


def _to_arrays(y_true: Iterable[Any], y_pred: Iterable[Any]):
    return np.asarray(y_true), np.asarray(y_pred)


def proba_to_labels(y_proba: np.ndarray) -> np.ndarray:
    """Convert probabilities to binary labels (0 or 1) based on a threshold of 0.5."""
    return (y_proba >= 0.5).astype(int)


def accuracy(y_true, y_pred) -> float:
    y_t, y_p = _to_arrays(y_true, y_pred)
    return float(accuracy_score(y_t, proba_to_labels(y_p)))


def balanced_accuracy(y_true, y_pred) -> float:
    y_t, y_p = _to_arrays(y_true, y_pred)
    return float(balanced_accuracy_score(y_t, proba_to_labels(y_p)))


def precision(y_true, y_pred, average: AverageType = 'binary', zero_division: int = 0) -> float:
    y_t, y_p = _to_arrays(y_true, y_pred)
    y_p = proba_to_labels(y_p)
    return float(precision_score(y_t, y_p, average=average, zero_division=zero_division))


def recall(y_true, y_pred, average: AverageType = 'binary', zero_division: int = 0) -> float:
    y_t, y_p = _to_arrays(y_true, y_pred)
    y_p = proba_to_labels(y_p)
    return float(recall_score(y_t, y_p, average=average, zero_division=zero_division))


def f1(y_true, y_pred, average: AverageType = 'binary', zero_division: int = 0) -> float:
    y_t, y_p = _to_arrays(y_true, y_pred)
    y_p = proba_to_labels(y_p)
    return float(f1_score(y_t, y_p, average=average, zero_division=zero_division))


def average_precision(y_true, y_pred, average: AverageType = 'binary') -> float:
    """Mean Average Precision (average_precision) for binary classification."""
    y_t, y_p = _to_arrays(y_true, y_pred)
    return float(average_precision_score(y_t, y_p))


def roc_auc(
    y_true,
    y_proba,
    multi_class: MultiClassType = 'ovr',
    average: RocAverageType | None = 'macro'
) -> float:
    """ROC AUC for binary or multi-class.

    y_proba can be:
      - shape (n_samples,) : probability/score for positive class (binary)
      - shape (n_samples, n_classes) : probabilities per class (multi-class)
    """
    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_proba)
    return float(roc_auc_score(y_t, y_pr, multi_class=multi_class, average=average))


def cross_entropy(y_true, y_proba) -> float:
    """Log loss (a.k.a. cross entropy)."""
    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_proba)
    return float(log_loss(y_t, y_pr))


# ------------------------- bundled computation --------------------------- #

def _infer_average(y_true) -> AverageType:
    y_t = np.asarray(y_true)
    classes = np.unique(y_t)
    return 'binary' if classes.shape[0] == 2 else 'macro'


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Any | None = None,
    average: AverageType | Literal['auto'] | None = 'auto',
    zero_division: int = 0,
) -> dict[str, float]:
    """Return a dictionary of common classification metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground truth labels and predicted labels.
    y_proba : array-like, optional
        Probabilities or decision scores. If provided enables ROC AUC and
        log loss metrics. For binary classification shape can be (n_samples,)
        or (n_samples, 2). For multi-class expected shape: (n_samples, n_classes).
    average : str or 'auto'
        Averaging strategy for precision/recall/F1. If 'auto', picks 'binary'
        for 2 classes else 'macro'.
    zero_division : int
        Passed to precision/recall/F1 to control behavior on zero division.
    """
    if average == 'auto' or average is None:
        avg: AverageType = _infer_average(y_true)
    else:
        avg = cast(AverageType, average)

    metrics: dict[str, float] = {
        'accuracy': accuracy(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred, average=avg, zero_division=zero_division),
        'recall': recall(y_true, y_pred, average=avg, zero_division=zero_division),
        'f1': f1(y_true, y_pred, average=avg, zero_division=zero_division),
        'average_precision': average_precision(y_true, y_pred, average=avg),
    }

    if y_proba is not None:
        y_pr = np.asarray(y_proba)
        try:
            # Determine multi-class vs binary for ROC AUC call
            if y_pr.ndim == 1 or (y_pr.ndim == 2 and y_pr.shape[1] == 1):
                # binary probability for positive class
                metrics['roc_auc'] = roc_auc(y_true, y_pr, average=None)
            else:
                # multi-class or binary with both columns
                metrics['roc_auc'] = roc_auc(y_true, y_pr, multi_class='ovr', average='macro')
        except Exception:
            # Silently skip if not computable (e.g., single class present)
            pass
        try:
            metrics['log_loss'] = cross_entropy(y_true, y_pr)
        except Exception:
            pass

    return metrics
# ------------------------- bundled computation --------------------------- #


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dictionary of standard regression metrics.

    Can be extended easily; all call tiny metric functions above.
    """
    return {
        'mse': mse(y_true, y_pred),
        'average_precisione': average_precisione(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'explained_var': explained_variance(y_true, y_pred),
    }
