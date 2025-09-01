# %load_ext cuml.accel
import os
import pandas as pd
from analyzer.plot_prediction_error import plot_prediction_error
from analyzer.hyperparameter_summary import best_hyperparameter_experiment_summary
from analyzer.hyperparameter_summary import visualize_full_performance_graphs
from training.runner import run_nested_cv_experiment
from experiment_configs.model import CLASSIFICATION_MODELS, REGRESSION_MODELS
from experiment_configs.dataset import DatasetConfig, CLASSIFICATION_TARGET_COLUMNS
import warnings
import pandas
warnings.filterwarnings("ignore")


def run_nested_training(dataset_config: DatasetConfig, model_config: dict):

    experiment_configs = {
        **dataset_config.model_dump(),
        **model_config,
        'outer_loop_n_splits': None,
        'inner_loop_n_splits': 5,
        'random_state': 42,
    }
    return run_nested_cv_experiment(**experiment_configs)


def run_all_nested_training(dataset: DatasetConfig,
                            visualize: bool = True,
                            ) -> pandas.DataFrame:

    results = {}
    save_name = f'{dataset.target_column}/{dataset.dataset_types}-{dataset.input_columns}'
    if not visualize and os.path.exists(f'results/tables/{save_name}.csv'):
        summary = pd.read_csv(f'results/tables/{save_name}.csv')
    else:
        if dataset.target_column in CLASSIFICATION_TARGET_COLUMNS:
            models = CLASSIFICATION_MODELS
        else:
            models = REGRESSION_MODELS

        for model in models:
            result = run_nested_training(dataset, model)
            results[model['ModelClass'].__name__] = result

        summary = best_hyperparameter_experiment_summary(results, experiment_name=save_name)

    if visualize:
        dataset_name = f"{dataset.target_column}-{dataset.dataset_types}\n{dataset.input_columns}"
        visualize_full_performance_graphs(results,
                                          dataset_name=dataset_name,
                                          show=True,
                                          save_path=f'results/hyperparameter_fitting_results/{save_name}.png')
        plot_prediction_error(
            title=f"Prediction Error for {dataset.dataset_types} - {dataset.target_column} \n {dataset.input_columns}",
            summary=summary,
            results_to_plot=9,
            results=results,
            show=True,
            save_path=f'results/prediction_error/{save_name}.png'
        )

    return summary
