# %load_ext cuml.accel
from training.runner import run_nested_cv_experiment
from experiment_configs.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from experiment_configs.dataset import DatasetConfig, REGRESSION_DATASETS, CLASSIFICATION_DATASETS
import warnings
warnings.filterwarnings("ignore")


results = {}


def run_nested_training(dataset_config: DatasetConfig, model_config: dict):

    experiment_configs = {
        **dataset_config.model_dump(),
        **model_config,
        'outer_loop_n_splits': None,
        'inner_loop_n_splits': 5,
        'random_state': 42,
    }
    return run_nested_cv_experiment(**experiment_configs)


for dataset in CLASSIFICATION_DATASETS:
    for model in CLASSIFICATION_MODELS:
        print(f"Running nested training for model:")
        print(f"Model: {model['ModelClass'].__name__}")
        print(f"Dataset: {dataset.dataset_types}, Target: {dataset.target_column}, Inputs: {dataset.input_columns}")
        result = run_nested_training(dataset, model)
        results[model['ModelClass'].__name__] = result


for dataset in REGRESSION_DATASETS:
    for model in REGRESSION_MODELS:
        print(f"Running nested training for model:")
        print(f"Model: {model['ModelClass'].__name__}")
        print(f"Dataset: {dataset.dataset_types}, Target: {dataset.target_column}, Inputs: {dataset.input_columns}")
        result = run_nested_training(dataset, model)
        results[model['ModelClass'].__name__] = result
