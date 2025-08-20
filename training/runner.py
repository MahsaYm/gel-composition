"""Lightweight experiment runner functions for training/evaluating models.

Keep orchestration logic here so notebooks remain minimal (variables + function calls).
"""
from __future__ import annotations

from typing import Any
from dataclasses import asdict

from sklearn.model_selection import ParameterGrid

from data.dataloader import GelDataLoader
from training.nested_cv import NestedCVRun
from results.cache_results import disk_cache
from training.nested_cv import nested_cross_validate_model
from training.standard_cv import CVSummary, cross_validate_model


@disk_cache()
def run_cv_experiment(
    data_path: str,
    ModelClass,  # sklearn-like model class
    model_params: dict[str, Any],
    dataset_types: list[str],
    input_columns: list[str],
    target_column: str,
    *,
    n_splits: int | None = None,
    random_state: int | None = 42,
    verbose: int = 1,
) -> CVSummary:
    """Run K-fold CV with an SVR model on the specified target (default Uniformity).

    Returns a dictionary with keys: summary (CVSummary dataclass as dict) and meta (data metadata).
    """

    loader = GelDataLoader(data_path)
    X_df, y_df, meta = loader.load(
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column,
        drop_na_rows=True,
    )

    # Cast to Any to bypass protocol mismatch (SVR fit signature includes sample_weight)
    summary = cross_validate_model(
        ModelClass,  # type: ignore[arg-type]
        model_params,
        X=X_df.values,
        y=y_df[target_column].to_numpy().ravel(),
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
        verbose=verbose,
    )

    return summary


@disk_cache()
def run_nested_cv_experiment(
    data_path: str,
    ModelClass,  # sklearn-like model class
    model_params: dict[str, list],
    dataset_types: list[str],
    input_columns: list[str],
    target_column: str,
    *,
    outer_loop_n_splits: int | None = None,
    inner_loop_n_splits: int | None = None,
    random_state: int | None = 42,
    verbose: int = 0,
) -> NestedCVRun:
    """Run nested cross-validation with the specified model and parameters.

    Returns a dictionary with keys: summary (CVSummary dataclass as dict) and meta (data metadata).
    """

    loader = GelDataLoader(data_path)
    X_df, y_df, meta = loader.load(
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column,
        drop_na_rows=True,
    )

    # create a joint list from param grid
    list_model_params = list(ParameterGrid(model_params))

    # Cast to Any to bypass protocol mismatch (SVR fit signature includes sample_weight)
    return nested_cross_validate_model(
        ModelClass,
        list_model_params,
        X=X_df.values,
        y=y_df[target_column].to_numpy().ravel(),
        outer_loop_n_splits=outer_loop_n_splits,
        inner_loop_n_splits=inner_loop_n_splits,
        random_state=random_state,
        verbose=verbose,
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column
    )
