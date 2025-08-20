"""
Utility functions for visualization and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any
import json


class Visualizer:
    """Visualization utilities for model results and Optimization."""

    @staticmethod
    def plot_model_comparison(results: dict[str, Any],
                              save_path: str | None = None,
                              figsize: tuple[int, int] = (12, 8)) -> None:
        """
        Plot model performance comparison.

        Args:
            results (dict[str, Any]): Nested CV results
            save_path (Optional[str]): Path to save the plot
            figsize (tuple[int, int]): Figure size
        """
        model_performances = results['model_performances']

        # Extract model names and MAE values
        models = list(model_performances.keys())
        mae_values = [model_performances[model]['mae'] for model in models]
        r2_values = [model_performances[model]['r2'] for model in models]

        # Sort by MAE (best to worst)
        sorted_indices = np.argsort(mae_values)
        models = [models[i] for i in sorted_indices]
        mae_values = [mae_values[i] for i in sorted_indices]
        r2_values = [r2_values[i] for i in sorted_indices]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # MAE plot
        bars1 = ax1.barh(models, mae_values, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Mean Absolute Error (MAE)')
        ax1.set_title('Model Performance - MAE')
        ax1.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, mae_values)):
            ax1.text(value + max(mae_values) * 0.01, i, f'{value:.3f}',
                     va='center', fontsize=9)

        # R² plot
        bars2 = ax2.barh(models, r2_values, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('R² Score')
        ax2.set_title('Model Performance - R²')
        ax2.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars2, r2_values)):
            ax2.text(value + max(r2_values) * 0.01, i, f'{value:.3f}',
                     va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_prediction_vs_actual(results: dict[str, Any],
                                  model_name: str,
                                  save_path: str | None = None,
                                  figsize: tuple[int, int] = (8, 8)) -> None:
        """
        Plot predictions vs actual values for a specific model.

        Args:
            results (dict[str, Any]): Nested CV results
            model_name (str): Name of the model to plot
            save_path (Optional[str]): Path to save the plot
            figsize (tuple[int, int]): Figure size
        """
        y_true = []
        y_pred = []

        for fold_result in results['fold_results']:
            if model_name in fold_result['model_predictions']:
                pred = fold_result['model_predictions'][model_name]
                if not np.isnan(pred):
                    y_true.append(fold_result['y_true'])
                    y_pred.append(pred)

        if not y_true:
            print(f"No valid predictions found for model: {model_name}")
            return

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2

        # Create plot
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Actual\nMAE: {mae:.3f}, R²: {r2:.3f}')
        plt.grid(True, alpha=0.3)

        # Make plot square
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_hyperparameter_importance(results: dict[str, Any],
                                       model_name: str,
                                       save_path: str | None = None) -> None:
        """
        Plot hyperparameter usage frequency across folds.

        Args:
            results (dict[str, Any]): Nested CV results
            model_name (str): Name of the model
            save_path (Optional[str]): Path to save the plot
        """
        # Collect hyperparameters from all folds
        hyperparams = {}

        for fold_result in results['fold_results']:
            if model_name in fold_result['model_best_params']:
                params = fold_result['model_best_params'][model_name]
                for param, value in params.items():
                    if param not in hyperparams:
                        hyperparams[param] = []
                    hyperparams[param].append(value)

        if not hyperparams:
            print(f"No hyperparameters found for model: {model_name}")
            return

        # Create subplots for each hyperparameter
        n_params = len(hyperparams)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, (param, values) in enumerate(hyperparams.items()):
            ax = axes[i]

            # Handle different types of values
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric values - histogram
                ax.hist(values, bins=min(10, len(set(values))), alpha=0.7, edgecolor='black')
                ax.set_xlabel(param)
                ax.set_ylabel('Frequency')
            else:
                # Categorical values - bar plot
                unique_values, counts = np.unique(values, return_counts=True)
                ax.bar(range(len(unique_values)), counts, alpha=0.7)
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels(unique_values, rotation=45)
                ax.set_xlabel(param)
                ax.set_ylabel('Frequency')

            ax.set_title(f'{param} Distribution')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'{model_name} - Hyperparameter Usage Across Folds')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_cross_target_comparison(all_results: dict[str, dict[str, Any]],
                                     save_path: str | None = None,
                                     figsize: tuple[int, int] = (15, 10)) -> None:
        """
        Plot model performance comparison across different target variables.

        Args:
            all_results (dict[str, dict[str, Any]]): Results for all targets
            save_path (Optional[str]): Path to save the plot
            figsize (tuple[int, int]): Figure size
        """
        # Extract data for plotting
        targets = list(all_results.keys())
        models = set()
        for target_results in all_results.values():
            models.update(target_results['model_performances'].keys())
        models = sorted(list(models))

        # Create performance matrix
        mae_matrix = np.full((len(models), len(targets)), np.nan)
        r2_matrix = np.full((len(models), len(targets)), np.nan)

        for j, target in enumerate(targets):
            target_results = all_results[target]
            for i, model in enumerate(models):
                if model in target_results['model_performances']:
                    mae_matrix[i, j] = target_results['model_performances'][model]['mae']
                    r2_matrix[i, j] = target_results['model_performances'][model]['r2']

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # MAE heataverage_precision
        im1 = ax1.imshow(mae_matrix, aspect='auto', caverage_precision='Reds')
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels(targets, rotation=45)
        ax1.set_yticks(range(len(models)))
        ax1.set_yticklabels(models)
        ax1.set_title('Mean Absolute Error (MAE)\nAcross Targets and Models')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(targets)):
                if not np.isnan(mae_matrix[i, j]):
                    ax1.text(j, i, f'{mae_matrix[i, j]:.3f}',
                             ha='center', va='center', fontsize=8)

        plt.colorbar(im1, ax=ax1)

        # R² heataverage_precision
        im2 = ax2.imshow(r2_matrix, aspect='auto', caverage_precision='Greens')
        ax2.set_xticks(range(len(targets)))
        ax2.set_xticklabels(targets, rotation=45)
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels(models)
        ax2.set_title('R² Score\nAcross Targets and Models')

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(targets)):
                if not np.isnan(r2_matrix[i, j]):
                    ax2.text(j, i, f'{r2_matrix[i, j]:.3f}',
                             ha='center', va='center', fontsize=8)

        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_Optimization_surface_3d(optimizer_results: dict[str, Any],
                                     param1: str, param2: str,
                                     save_path: str | None = None) -> None:
        """
        Plot 3D Optimization surface for two parameters.

        Args:
            optimizer_results (dict[str, Any]): Optimization results
            param1 (str): First parameter name
            param2 (str): Second parameter name
            save_path (Optional[str]): Path to save the plot
        """
        # Extract data
        history = optimizer_results['Optimization_history']

        # Get parameter indices
        best_formulation = optimizer_results['best_formulation']
        param_names = list(best_formulation.keys())

        try:
            idx1 = param_names.index(param1)
            idx2 = param_names.index(param2)
        except ValueError:
            print(f"Parameters {param1} or {param2} not found in Optimization results")
            return

        # Extract values
        x_values = [h['parameters'][idx1] for h in history]
        y_values = [h['parameters'][idx2] for h in history]
        z_values = [h['objective_value'] for h in history]

        # Create 3D plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='markers',
            marker=dict(
                size=5,
                color=z_values,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Objective Value")
            ),
            text=[f"Iteration {h['iteration']}" for h in history],
            hovertemplate=f"{param1}: %{{x}}<br>{param2}: %{{y}}<br>Objective: %{{z}}<br>%{{text}}<extra></extra>"
        )])

        # Mark best point
        best_x = optimizer_results['best_formulation'][param1]
        best_y = optimizer_results['best_formulation'][param2]
        best_z = optimizer_results['best_objective_value']

        fig.add_trace(go.Scatter3d(
            x=[best_x],
            y=[best_y],
            z=[best_z],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Best Solution'
        ))

        fig.update_layout(
            title=f'Optimization Surface: {param1} vs {param2}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title='Objective Value'
            )
        )

        if save_path:
            fig.write_html(save_path)
        fig.show()


class ResultsAnalyzer:
    """Utility class for analyzing and comparing results."""

    @staticmethod
    def load_results(filepath: str) -> dict[str, Any]:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def compare_datasets(results1: dict[str, Any],
                         results2: dict[str, Any],
                         dataset1_name: str = "Dataset 1",
                         dataset2_name: str = "Dataset 2") -> pd.DataFrame:
        """
        Compare model performance between two datasets.

        Args:
            results1 (dict[str, Any]): Results from first dataset
            results2 (dict[str, Any]): Results from second dataset
            dataset1_name (str): Name for first dataset
            dataset2_name (str): Name for second dataset

        Returns:
            pd.DataFrame: Comparison table
        """
        # Extract model performances
        perf1 = results1['model_performances']
        perf2 = results2['model_performances']

        # Get common models
        common_models = set(perf1.keys()) & set(perf2.keys())

        comparison_data = []
        for model in common_models:
            comparison_data.append({
                'Model': model,
                f'{dataset1_name}_MAE': perf1[model]['mae'],
                f'{dataset1_name}_R2': perf1[model]['r2'],
                f'{dataset2_name}_MAE': perf2[model]['mae'],
                f'{dataset2_name}_R2': perf2[model]['r2'],
                'MAE_Improvement': perf1[model]['mae'] - perf2[model]['mae'],
                'R2_Improvement': perf2[model]['r2'] - perf1[model]['r2']
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values(f'{dataset2_name}_MAE')

        return df

    @staticmethod
    def get_best_models_summary(all_results: dict[str, dict[str, Any]],
                                top_n: int = 3) -> pd.DataFrame:
        """
        Get summary of best performing models for each target.

        Args:
            all_results (dict[str, dict[str, Any]]): Results for all targets
            top_n (int): Number of top models to show per target

        Returns:
            pd.DataFrame: Summary of best models
        """
        summary_data = []

        for target, results in all_results.items():
            model_performances = results['model_performances']

            # Sort models by MAE
            sorted_models = sorted(model_performances.items(),
                                   key=lambda x: x[1]['mae'])

            for rank, (model_name, metrics) in enumerate(sorted_models[:top_n], 1):
                summary_data.append({
                    'Target': target,
                    'Rank': rank,
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'R2': metrics['r2'],
                    'RMSE': metrics['rmse']
                })

        return pd.DataFrame(summary_data)

    @staticmethod
    def calculate_ensemble_performance(results: dict[str, Any],
                                       ensemble_models: list[str]) -> dict[str, float]:
        """
        Calculate performance of an ensemble of models.

        Args:
            results (dict[str, Any]): Nested CV results
            ensemble_models (list[str]): list of models to include in ensemble

        Returns:
            dict[str, float]: Ensemble performance metrics
        """
        y_true = []
        y_pred_ensemble = []

        for fold_result in results['fold_results']:
            y_true.append(fold_result['y_true'])

            # Calculate ensemble prediction (mean of selected models)
            fold_predictions = []
            for model_name in ensemble_models:
                if model_name in fold_result['model_predictions']:
                    pred = fold_result['model_predictions'][model_name]
                    if not np.isnan(pred):
                        fold_predictions.append(pred)

            if fold_predictions:
                y_pred_ensemble.append(np.mean(fold_predictions))
            else:
                y_pred_ensemble.append(np.nan)

        # Filter out NaN predictions
        valid_indices = ~np.isnan(y_pred_ensemble)
        y_true_valid = np.array(y_true)[valid_indices]
        y_pred_valid = np.array(y_pred_ensemble)[valid_indices]

        if len(y_true_valid) > 0:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            mse = mean_squared_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_valid, y_pred_valid)

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'n_valid_predictions': len(y_true_valid)
            }
        else:
            return {
                'mae': np.inf,
                'mse': np.inf,
                'rmse': np.inf,
                'r2': -np.inf,
                'n_valid_predictions': 0
            }
