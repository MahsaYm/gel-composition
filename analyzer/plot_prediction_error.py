from training.nested_cv import NestedCVRun
import pandas
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from analyzer.utils import color_from_label_name, get_marker_from_label_name


def regression_plot_prediction_error(
    title: str,
    summary: pandas.DataFrame,
    results_to_plot: int,
    results: dict[str, NestedCVRun],
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot prediction error for hyperparameter fitting results."""

    # Create subplots once for all models
    n_plots = min(results_to_plot, len(summary))
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))  # Increased from (15, 12)
    fig.suptitle(title)

    # Reduce space between subplots
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.25)

    # Handle the case where we have only one subplot
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for plot_idx, (i, row) in enumerate(summary.iterrows()):
        if plot_idx >= results_to_plot:
            break

        ax = axes[plot_idx]
        ax.set_title(f"{row['model_name']}\n{row['model_params']}", fontsize=10, pad=10)

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('True Values')

        model_name = row['model_name']
        model_param = row['model_params']
        r2 = row['r2']
        y_true, y_pred = results[model_name].get_output(model_param)
        color = color_from_label_name(model_name)
        marker = get_marker_from_label_name(model_name)
        ax.set_xlim(0.7*y_true.min(), 1.3*y_true.max())
        ax.set_ylim(0.7*y_true.min(), 1.3*y_true.max())
        ax.grid(True)
        # Plot the prediction error
        ax.scatter(y_pred, y_true, c=color, alpha=0.6, label=model_name, marker=marker)
        # Plot the identity line
        identity_line = pandas.Series([min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())])
        ax.plot(identity_line, identity_line, color='black', linestyle='--', label='Identity Line')

        # Plot the line that best fits the data
        coeffs = pandas.Series(np.polyfit(y_pred, y_true, 1))
        fit_line = coeffs[0] * y_pred + coeffs[1]
        ax.plot(y_pred, fit_line, color=color, linestyle='-', label='Best Fit Line')

        # Add R² annotation to bottom right
        ax.annotate(f'R² = {r2:.3f}', xy=(0.95, 0.05), xycoords='axes fraction',
                    fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add a legend to top left
        ax.legend(loc='upper left')

    # Hide any unused subplots
    for j in range(len(summary), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()  # Close the figure if not showing to avoid memory leaks


def classification_plot_prediction_error(
    title: str,
    summary: pandas.DataFrame,
    results_to_plot: int,
    results: dict[str, NestedCVRun],
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot prediction error for classification hyperparameter fitting results."""

    # Create subplots once for all models
    n_plots = min(results_to_plot, len(summary))
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))  # Increased from (15, 12)
    fig.suptitle(title)

    # Reduce space between subplots
    plt.subplots_adjust(top=0.92, hspace=0.25, wspace=0.25)

    # Handle the case where we have only one subplot
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for plot_idx, (i, row) in enumerate(summary.iterrows()):
        if plot_idx >= results_to_plot:
            break

        ax = axes[plot_idx]
        ax.set_title(f"{row['model_name']}\n{row['model_params']}", fontsize=8, pad=10)

        ax.set_xlabel('Predicted Leakage')
        ax.set_ylabel('Percentage of Samples (%)')

        model_name = row['model_name']
        model_param = row['model_params']
        f1 = row['f1']
        y_true, y_pred = results[model_name].get_output(model_param)

        # Separate predictions by true class
        y_pred_class_0 = y_pred[y_true == 0]
        y_pred_class_1 = y_pred[y_true == 1]

        # Plot histograms for each class with percentage normalization
        bins = np.linspace(0, 1, 101)  # 101 equal-width bins from 0 to 1
        weights_0 = np.ones_like(y_pred_class_0) / len(y_pred_class_0) * 100
        weights_1 = np.ones_like(y_pred_class_1) / len(y_pred_class_1) * 100
        ax.hist(y_pred_class_0, bins=bins, weights=weights_0, alpha=0.6, label=f'No Leakage', color='red', edgecolor='black')
        ax.hist(y_pred_class_1, bins=bins, weights=weights_1, alpha=0.6, label=f'Leakage', color='blue', edgecolor='black')

        # Set x-axis limits for binary classification scores
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)  # Percentage scale
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Leakage', 'Leakage'])

        # Add accuracy annotation to bottom right
        ax.annotate(f'F1 = {f1:.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=9, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add a legend to top left
        ax.legend(loc='upper left')

    # Hide any unused subplots
    for j in range(len(summary), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()  # Close the figure if not showing to avoid memory leaks


def plot_prediction_error(
    title: str,
    summary: pandas.DataFrame,
    results_to_plot: int,
    results: dict[str, NestedCVRun],
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot prediction error for hyperparameter fitting results."""

    if list(results.values())[0].target_type == 'regression':
        regression_plot_prediction_error(title, summary, results_to_plot, results, show, save_path)
    else:
        classification_plot_prediction_error(title, summary, results_to_plot, results, show, save_path)
