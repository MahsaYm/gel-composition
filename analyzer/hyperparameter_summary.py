
from matplotlib import pyplot as plt
import pandas
from training.nested_cv import NestedCVRun, NestedCVParamResult, NestedCVSummary
import os
from analyzer.utils import color_from_label_name, get_marker_from_label_name


def visualize_performance_graphs(
    nested_cv_run: NestedCVRun,
    metric_1: str,
    metric_2: str,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    label: str | None = None,
) -> None:
    """Visualize R2 and MAE graphs for a all the parameter combinations in the nested CV run."""

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Add a title
        plt.title(f'{metric_1.capitalize()} vs {metric_2.capitalize()} for Hyperparameter Combinations')

        # Show grid
        ax.grid(True)

    if not isinstance(nested_cv_run, NestedCVRun):
        raise TypeError(f"Expected NestedCVRun, got {type(nested_cv_run)}")

    if not nested_cv_run.param_results:
        return

    # Prepare data for plotting
    param_results = nested_cv_run.param_results
    metric_1_values = [param_result.test_run.to_summary().metrics[metric_1] for param_result in param_results]
    metric_2_values = [param_result.test_run.to_summary().metrics[metric_2] for param_result in param_results]
    model_params = [param_result.model_params for param_result in param_results]

    # Create a figure
    if label is None:
        label = nested_cv_run.model_class
    color = color_from_label_name(label)
    marker = get_marker_from_label_name(label)
    # plot R2 vs average_precisionE with model parameters hovering
    ax.scatter(metric_1_values, metric_2_values, c=color, alpha=0.6, label=label, marker=marker)
    ax.set_xlabel(metric_1.capitalize())
    ax.set_ylabel(metric_2.capitalize())
    ax.tick_params(axis='y', labelcolor=color)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()


def visualize_full_performance_graphs(
    results: dict[str, NestedCVRun],
    dataset_name: str,
    figsize: tuple[int, int] = (16, 8),
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualize performance graphs for a list of NestedCVRun."""
    for result in results.values():
        if result.target_type == 'regression':
            metric_1 = 'r2'
            metric_2 = 'average_precisione'
            xlim = (-0.5, 1)
            ylim = (0, 0.5)
        else:
            metric_1 = 'balanced_accuracy'
            metric_2 = 'average_precision'
            xlim = (0, 1)
            ylim = (0, 1)
        break

    fig, ax1 = plt.subplots(figsize=figsize)
    plt.suptitle(f'{metric_1.capitalize()} vs {metric_2.capitalize()} for Hyperparameter Combinations')

    plt.title(dataset_name)
    # Show grid
    ax1.grid(True)

    # Adjust subplot parameters to make room for annotations on the right
    plt.subplots_adjust(right=0.7)

    for model_name, result in results.items():
        visualize_performance_graphs(result, metric_1=metric_1, metric_2=metric_2, fig=fig,
                                     ax=ax1, show=False, save_path=None, label=model_name)

    # Add legend to show labels and colors in the top right
    ax1.legend(loc='best')

    # Fix the range:
    ax1.set_ylim(ylim[0], ylim[1])  # Set lower limit to 90% of min F1
    ax1.set_xlim(xlim[0], xlim[1])

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()

    plt.close(fig)


def best_hyperparameter_experiment_summary(
    nested_cv_results: dict[str, NestedCVRun],
    experiment_name: str | None = None,
) -> pandas.DataFrame:
    """Return a DataFrame with the best hyperparameters for each model."""

    if experiment_name is not None:
        save_path_filename = f'results/tables/{experiment_name}.csv'
        if os.path.exists(save_path_filename):
            return pandas.read_csv(save_path_filename)

    all_results = []
    for model_name, run in nested_cv_results.items():
        best_summary = run.best_hyperparameter_summary()

        all_results.append(best_summary)

    if not all_results:
        return pandas.DataFrame()

    all_results = pandas.concat(all_results, ignore_index=True)

    if 'r2' in all_results.columns:
        all_results = all_results.sort_values(by='r2', ascending=False)
    elif 'f1' in all_results.columns:
        all_results = all_results.sort_values(by='f1', ascending=False)

    all_results = all_results.reset_index(drop=True)

    if experiment_name is not None:
        os.makedirs(os.path.dirname(save_path_filename), exist_ok=True)
        all_results.to_csv(save_path_filename, index=False)

    return all_results
