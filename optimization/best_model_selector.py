"""Extract and prepare the best models from hyperparameter fitting results."""

from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from analyzer.hyperparameter_analysis import run_all_nested_training
from data.dataloader import GelDataLoader
from experiment_configs.dataset import DatasetConfig
from experiment_configs.model import MODEL_CLASSES, is_regressor
from models.model_trainer import Model


def get_best_models(
    dataset: DatasetConfig,
    top_models: int = 1,
    metric: str | None = None
) -> list[Model]:

    summary = run_all_nested_training(dataset, visualize=False)
    metric = metric or ('r2' if 'r2' in summary.columns else 'f1')

    best_models = []

    loader = GelDataLoader(dataset.data_path)
    X_df, y_df, meta = loader.load(
        dataset_types=dataset.dataset_types,
        input_columns=dataset.input_columns,
        target_column=dataset.target_column,
        drop_na_rows=True,
    )

    for row_index, row in summary.iterrows():
        if row_index >= top_models:
            break

        model_name = row['model_name']
        model_params = row['model_params']
        if isinstance(model_params, str):
            model_params = eval(model_params)  # Convert string representation to dict

        model = Model(MODEL_CLASSES[model_name], model_params)
        model.fit(X_df.values, y_df[dataset.target_column].to_numpy().ravel())
        best_models.append(model)

    return best_models
