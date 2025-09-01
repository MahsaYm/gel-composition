import os
import matplotlib.pyplot as plt
from experiment_configs.dataset import (GEL_BOUNDS, DatasetConfig, FORMULATION_COLUMNS,
                                        INPUT_COLUMNS,
                                        REGRESSION_TARGET_COLUMNS,
                                        CLASSIFICATION_TARGET_COLUMNS,
                                        DATASET_TYPES)
from optimization.bayesian_optimizer import optimize_value
from optimization.best_model_selector import get_best_models
from results.cache_results import disk_cache
from optimization.composit_objective import composit_objective_function
from optimization.grid_operation import create_3d_parameter_grid_centered_on_optimal
from optimization.visualizer import visualize_penalties
import tqdm
import pandas as pd


def optimize(top_models: int = 3,
             polymer_penalty_weight: float = 0.0,
             leakage_penalty_weight: float = 0.0,
             input_columns: list[str] = FORMULATION_COLUMNS,  # + PRINTING_CONDITION_COLUMNS
             target_columns: list[str] = REGRESSION_TARGET_COLUMNS + CLASSIFICATION_TARGET_COLUMNS,
             dataset_type: list[str] = ['Optimization'],
             verbose: int = 1,
             num_grid_points: int = 35,
             visualize_grid: bool = True,
             ):
    top_models = top_models
    best_models = {}

    input_columns = input_columns
    gel_bounds = [GEL_BOUNDS[k]
                  for k in input_columns
                  if k in GEL_BOUNDS]

    for target_column in target_columns:
        dataset = DatasetConfig(
            data_path=r"data/Data_average_new.csv",
            dataset_types=dataset_type,
            input_columns=input_columns,
            target_column=target_column
        )
        best_models[target_column] = get_best_models(dataset, top_models=top_models, verbose=verbose)

    optimal_gel_composition = optimize_value(best_models=best_models,
                                             gel_bounds=gel_bounds,
                                             polymer_penalty_weight=polymer_penalty_weight,
                                             leakage_penalty_weight=leakage_penalty_weight)

    if verbose > 0:
        print("Optimal Gel Composition (including printing conditions if any):")
        print(f'Gel Bounds: \n{gel_bounds}')
        print(f'Optimal Gel Composition: \n{optimal_gel_composition}')

    if visualize_grid:
        # Code for visualizing the grid goes here

        if "Pressure" not in input_columns or "Speed" not in input_columns:
            pressure_fixed = None
            speed_fixed = None
        else:
            pressure_fixed = optimal_gel_composition[-2]
            speed_fixed = optimal_gel_composition[-1]

        a4c, a4m, ac_disol, X = create_3d_parameter_grid_centered_on_optimal(
            num_points=num_grid_points,
            optimal_gel_composition=optimal_gel_composition,
            pressure_fixed=pressure_fixed,
            speed_fixed=speed_fixed
        )

        # Since models were trained with only FORMULATION_COLUMNS (4 features),
        # we need to use only the first 4 columns from the grid

        penalties = composit_objective_function(best_models=best_models,
                                                X=X,
                                                polymer_penalty_weight=polymer_penalty_weight
                                                )
        visualize_penalties(penalties=penalties,
                            optimal_gel_composition=optimal_gel_composition)

    return optimal_gel_composition


def get_all_optimization_results(path='optimization/results.csv'):

    if os.path.exists(path):
        df = pd.read_csv(path)
        return df

    optimization_results = []

    for dataset_type in tqdm.tqdm(DATASET_TYPES, desc="Dataset Types"):
        for input_column in tqdm.tqdm(INPUT_COLUMNS, desc="Input Columns"):
            for target_column in [REGRESSION_TARGET_COLUMNS,
                                  REGRESSION_TARGET_COLUMNS + CLASSIFICATION_TARGET_COLUMNS]:
                for top_models in [1, 2, 3, 4]:
                    for polymer_penalty_weight in [0.1, 0.316, 1, 3.16, 10]:
                        if "Leakage" in target_column:
                            leakage_penalty_weights = [0.1, 0.316, 1, 3.16, 10]
                        else:
                            leakage_penalty_weights = [1]

                        for leakage_penalty_weight in leakage_penalty_weights:

                            optimal_gel_composition = optimize(
                                top_models=top_models,
                                polymer_penalty_weight=0.001 * polymer_penalty_weight,
                                leakage_penalty_weight=leakage_penalty_weight,
                                input_columns=input_column,
                                target_columns=target_column,
                                dataset_type=dataset_type,
                                verbose=0,  # 1 for print, 0 for silent
                                num_grid_points=35,
                                visualize_grid=False
                            )

                            data = {
                                'dataset_type': dataset_type,
                                'input_column': input_column,
                                'target_column': target_column,
                                'polymer_penalty_weight': polymer_penalty_weight,
                                'leakage_penalty_weight': leakage_penalty_weight,
                                'top_models': top_models,
                                'optimal_gel_composition': optimal_gel_composition,
                                'a4c': optimal_gel_composition[0],
                                'a4m': optimal_gel_composition[1],
                                'ac-disol': optimal_gel_composition[2]
                            }

                            optimization_results.append(data)

    df = pd.DataFrame(optimization_results)
    df.to_csv("optimization/results.csv", index=False)
    return df
