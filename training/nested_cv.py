"""Nested cross-validation implementation for hyperparameter tuning."""
from __future__ import annotations

import ast
from time import perf_counter
from typing import Any
import numpy as np
import pandas
from pydantic import BaseModel, ConfigDict

from experiment_configs.model import is_regressor
import tqdm
from results.cache_results import disk_cache
from training.cv_helpers import ensure_numpy, iterate_splits, make_kfold, run_in_parallel
from models.model_trainer import run_train_and_predict
from training.standard_cv import cross_validate_model, ModelPred, CVRun, CVSummary
from models.measurements import LOWER_IS_BETTER


@disk_cache()
def single_param_nested_cross_validate_model(
    ModelClass,  # sklearn-like model class
    params: dict[str, Any],
    param_idx: int,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    outer_loop_n_splits: int,
    *,
    inner_loop_n_splits: int | None = None,  # None is leave-one-out
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: int = 0,
    dataset_types: list[str],
    input_columns: list[str],
    target_column: str,
) -> NestedCVParamResult:
    param_start = perf_counter()

    # For this parameter, run full nested CV
    outer_splitter = make_kfold(n_splits=outer_loop_n_splits, shuffle=shuffle, random_state=random_state)
    outer_folds: list[ModelPred] = []
    inner_summaries: list[CVSummary] = []

    for outer_fold_idx, outer_train_idx, outer_test_idx in iterate_splits(outer_splitter, X_arr, y_arr):
        if verbose >= 2:
            print(f"  Outer fold {outer_fold_idx}/{outer_loop_n_splits}")

        # Get outer training data
        X_outer_train = X_arr[outer_train_idx]
        y_outer_train = y_arr[outer_train_idx]

        # Inner CV for validation performance
        inner_summary = cross_validate_model(
            ModelClass,
            params,
            X=X_outer_train,
            y=y_outer_train,
            n_splits=inner_loop_n_splits,
            shuffle=shuffle,
            random_state=random_state,
            verbose=max(0, verbose - 3),
        )
        inner_summaries.append(inner_summary)

        # Train on full outer training set and predict on outer test set for test performance
        outer_fold_pred = run_train_and_predict(
            ModelClass=ModelClass,
            model_params=params,
            X=X_arr,
            y=y_arr,
            train_idx=outer_train_idx,
            test_idx=outer_test_idx,
            fold_idx=outer_fold_idx,
            verbose=max(0, verbose - 2),
        )
        outer_folds.append(outer_fold_pred)

    # Aggregate validation metrics across outer folds
    if inner_summaries:
        # Compute mean and std for all available metrics
        all_metric_names = set()
        for summary in inner_summaries:
            all_metric_names.update(summary.metrics.keys())

        aggregated_validation_metrics = {}
        for metric_name in all_metric_names:
            values = [summary.metrics.get(metric_name, 0.0) for summary in inner_summaries]
            aggregated_validation_metrics[metric_name] = np.mean(values)
            aggregated_validation_metrics[f'{metric_name}_std'] = np.std(values)

        # Create aggregated validation summary
        validation_summary = CVSummary(
            n_splits=inner_loop_n_splits or y_arr.shape[0],
            metrics=aggregated_validation_metrics,
            model_class=ModelClass.__name__,
            model_params=params,
            target_type='regression' if is_regressor(ModelClass) else 'classification',
            total_time_sec=sum(s.total_time_sec for s in inner_summaries)
        )
    else:
        # Empty validation summary if no inner CV was performed
        validation_summary = CVSummary(
            n_splits=0,
            metrics={},
            model_class=ModelClass.__name__,
            model_params=params,
            target_type='regression' if is_regressor(ModelClass) else 'classification',
            total_time_sec=0.0
        )

    # Create test CVRun for this parameter
    test_run = CVRun(
        folds=outer_folds,
        n_splits=outer_loop_n_splits,
        model_class=ModelClass.__name__,
        model_params=params,
        total_time_sec=0.0,  # Will be set in NestedCVRun
        target_type='regression' if is_regressor(ModelClass) else 'classification'
    )

    param_duration = perf_counter() - param_start

    # Store parameter result
    return NestedCVParamResult(
        param_index=param_idx,
        model_params=params,
        validation_summary=validation_summary,
        test_run=test_run,
        duration_sec=param_duration,
        dataset_types=dataset_types,
        input_columns=input_columns,
        target_column=target_column
    )


def nested_cross_validate_model(
    ModelClass,  # sklearn-like model class
    model_params: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    outer_loop_n_splits: int | None = None,  # None is leave-one-out
    inner_loop_n_splits: int | None = None,  # None is leave-one-out
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: int = 0,
    dataset_types: list[str],
    input_columns: list[str],
    target_column: str,
) -> NestedCVRun:

    X_arr, y_arr = ensure_numpy(X, y)
    outer_loop_n_splits = outer_loop_n_splits if outer_loop_n_splits is not None else y.shape[0]
    inner_loop_n_splits = inner_loop_n_splits if inner_loop_n_splits is not None else y.shape[0]

    global_start = perf_counter()

    if verbose:
        print(f"Starting parameter-first nested CV with {len(model_params)} parameter sets...")
        print(f"Outer CV: {outer_loop_n_splits} splits, Inner CV: {inner_loop_n_splits} splits")

    param_results: list[NestedCVParamResult] = []

    for param_idx, params in tqdm.tqdm(enumerate(model_params), total=len(model_params), desc="Nested CV Progress"):
        if verbose:
            print(f"\nParameter set {param_idx + 1}/{len(model_params)}: {params}")

        param_result = single_param_nested_cross_validate_model(
            ModelClass=ModelClass,
            params=params,
            param_idx=param_idx,
            X_arr=X_arr,
            y_arr=y_arr,
            outer_loop_n_splits=outer_loop_n_splits,
            inner_loop_n_splits=inner_loop_n_splits,
            shuffle=shuffle,
            random_state=random_state,
            verbose=verbose,
            dataset_types=dataset_types,
            input_columns=input_columns,
            target_column=target_column
        )

        param_results.append(param_result)

        if verbose >= 2:
            val_metric_name = 'r2' if is_regressor(ModelClass) else 'accuracy'
            val_score_mean = param_result.validation_summary.metrics.get(val_metric_name, 0.0)
            print(f"  Validation {val_metric_name}: {val_score_mean:.4f}")

    total_time = perf_counter() - global_start

    if verbose:
        print(f'\nParameter-first nested CV completed in {total_time:.3f}s')

    # Create NestedCVRun
    nested_cv_run = NestedCVRun(
        param_results=param_results,
        n_outer_splits=outer_loop_n_splits,
        n_inner_splits=inner_loop_n_splits,
        model_class=ModelClass.__name__,
        total_time_sec=total_time,
        target_type='regression' if is_regressor(ModelClass) else 'classification'
    )

    return nested_cv_run


class NestedCVParamResult(BaseModel):
    """Results for a single parameter set in nested CV."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    param_index: int
    model_params: dict[str, Any]
    validation_summary: CVSummary  # Inner CV performance with all metrics
    test_run: CVRun  # Outer CV performance for this parameter
    duration_sec: float
    dataset_types: list[str]
    input_columns: list[str]
    target_column: str


class NestedCVRun(BaseModel):
    """Container for parameter-first nested CV results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    param_results: list[NestedCVParamResult]
    n_outer_splits: int
    n_inner_splits: int
    model_class: str
    total_time_sec: float
    target_type: str = 'regression'

    def get_best_param_index(self, metric_name: str | None = None) -> int:
        """Get index of parameter with best validation performance."""
        if not self.param_results:
            raise ValueError("No parameter results available")

        if metric_name is None:
            # Default metric selection
            metric_name = 'r2' if self.target_type == 'regression' else 'accuracy'

        # Check if lower values are better for this metric
        lower_is_better = LOWER_IS_BETTER.get(metric_name, False)

        best_idx = 0
        if lower_is_better:
            best_score = self.param_results[0].validation_summary.metrics.get(metric_name, float('inf'))
            for i, result in enumerate(self.param_results[1:], 1):
                score = result.validation_summary.metrics.get(metric_name, float('inf'))
                if score < best_score:
                    best_score = score
                    best_idx = i
        else:
            best_score = self.param_results[0].validation_summary.metrics.get(metric_name, float('-inf'))
            for i, result in enumerate(self.param_results[1:], 1):
                score = result.validation_summary.metrics.get(metric_name, float('-inf'))
                if score > best_score:
                    best_score = score
                    best_idx = i

        return best_idx

    def to_summary(self, metric_name: str | None = None, verbose: int = 0) -> 'NestedCVSummary':
        """Create summary with best parameter selection."""
        if not self.param_results:
            raise ValueError("No parameter results available")

        best_idx = self.get_best_param_index(metric_name)
        best_result = self.param_results[best_idx]

        # Get test performance of best parameter
        test_summary = best_result.test_run.to_summary(verbose=0)

        if verbose:
            if metric_name is None:
                metric_name = 'r2' if self.target_type == 'regression' else 'accuracy'
            print(
                f"Best parameter (index {best_idx}) validation {metric_name}: {best_result.validation_summary.metrics.get(metric_name, 'N/A'):.4f}")
            print(f"Best parameter test performance:")
            print('Test Metrics: ' + ', '.join(f"{k}={v:.4f}" for k, v in test_summary.metrics.items()))
            print(f'Total nested CV time: {self.total_time_sec:.2f}s')

        return NestedCVSummary(
            best_param_index=best_idx,
            best_params=best_result.model_params,
            validation_metrics=best_result.validation_summary.metrics,
            test_metrics=test_summary.metrics,
            all_param_results=self.param_results,
            n_outer_splits=self.n_outer_splits,
            n_inner_splits=self.n_inner_splits,
            model_class=self.model_class,
            target_type=self.target_type,
            total_time_sec=self.total_time_sec
        )

    def best_hyperparameter_summary(
        self: NestedCVRun,
    ) -> pandas.DataFrame:
        """Return a DataFrame with columns model_param and best results for every metric."""

        if self.target_type == 'regression':
            metric_names = ['r2', 'mae', 'average_precisione', 'mse', 'rmse', 'explained_var']
        elif self.target_type == 'classification':
            metric_names = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'average_precision']
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")
        # Extract the best parameter result

        if not self.param_results:
            return pandas.DataFrame(columns=['model_name', 'model_params'] + metric_names)

        param_results = self.param_results

        # Extract metrics and model params
        all_metrics = [param_result.test_run.to_summary().metrics for param_result in param_results]
        model_params = [param_result.model_params for param_result in param_results]

        # Filter metric names to only those present in the data
        available_metrics = set()
        for metrics_dict in all_metrics:
            available_metrics.update(metrics_dict.keys())
        metric_names = [m for m in metric_names if m in available_metrics]

        # Find best parameter for each metric
        best_param_indices = set()

        for metric_name in metric_names:
            # Check if lower is better for this metric (imported from models.measurements)
            from models.measurements import LOWER_IS_BETTER
            lower_is_better = LOWER_IS_BETTER.get(metric_name, False)

            # Find best parameter index for this metric
            metric_values = [metrics_dict.get(metric_name, float('inf') if lower_is_better else float('-inf'))
                             for metrics_dict in all_metrics]

            if lower_is_better:
                best_idx = min(range(len(metric_values)), key=lambda i: metric_values[i])
            else:
                best_idx = max(range(len(metric_values)), key=lambda i: metric_values[i])

            best_param_indices.add(best_idx)

        # Create DataFrame with rows for parameters that are best for at least one metric
        rows = []
        for idx in best_param_indices:
            row_data = {'model_name': self.model_class,
                        'model_params': str(model_params[idx])}  # Convert dict to string for display

            # Add all metric values for this parameter
            for metric_name in metric_names:
                row_data[metric_name] = all_metrics[idx].get(metric_name, None)

            rows.append(row_data)

        result = pandas.DataFrame(rows)

        # If no rows were added, return an empty DataFrame
        if result.empty:
            return pandas.DataFrame(columns=['model_name', 'model_params'] + metric_names)

        return result

    def get_output(self, model_params: dict[str, float] | str) -> tuple[np.ndarray, np.ndarray]:
        """Get prediction error for the given model parameters."""

        # Find the parameter result that matches the given model_params
        for param_result in self.param_results:
            if f'{param_result.model_params}' == f'{model_params}':
                return param_result.test_run.get_output()
        raise ValueError(f"Model parameters {model_params} not found in nested CV results for {self.model_class}")


class NestedCVSummary(BaseModel):
    """Summary of parameter-first nested CV with best parameter selection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_param_index: int
    best_params: dict[str, Any]
    validation_metrics: dict[str, float]  # Best param's validation performance
    test_metrics: dict[str, float]  # Best param's test performance
    all_param_results: list[NestedCVParamResult]  # All parameter results for analysis
    n_outer_splits: int
    n_inner_splits: int
    model_class: str
    target_type: str
    total_time_sec: float
