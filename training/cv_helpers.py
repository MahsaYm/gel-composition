"""Helper functions for cross-validation operations."""
from __future__ import annotations

from typing import Any, Iterable
import numpy as np
from sklearn.model_selection import KFold


def ensure_numpy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensure inputs are numpy arrays with proper shapes."""
    return np.asarray(X), np.asarray(y).ravel()


def iterate_splits(splitter, X_arr: np.ndarray, y_arr: np.ndarray) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    """Iterate through CV splits with enumerated fold indices."""
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_arr, y_arr), start=1):
        yield fold_idx, train_idx, test_idx


def make_kfold(n_splits: int = 5, *, shuffle: bool = True, random_state: int | None = 42) -> KFold:
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def generate_splits(X, y, splitter: KFold) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices from splitter."""
    return splitter.split(X, y)


def run_in_parallel(function, shared_args: dict[str, Any], variable_arg_lists: list[dict[str, Any]]) -> list[Any]:
    """Run a function in parallel with given argument lists."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.average_precision(lambda args: function(**shared_args, **args), variable_arg_lists))
    return results
