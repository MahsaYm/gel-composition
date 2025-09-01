from optimization.optimize import optimize
import tqdm
import pandas as pd
from experiment_configs.dataset import (FORMULATION_COLUMNS,
                                        PRINTING_CONDITION_COLUMNS,
                                        INPUT_COLUMNS,
                                        REGRESSION_TARGET_COLUMNS,
                                        CLASSIFICATION_TARGET_COLUMNS,
                                        DATASET_TYPES
                                        )
from optimization.scatter_arrow_plots import create_scatter_plot, load_and_prepare_data
from optimization.scatter_arrow_plots import create_arrow_plots
import matplotlib.pyplot as plt

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

df = load_and_prepare_data("optimization/results.csv")

print("\nCreating enhanced scatter plot...")
fig1 = create_scatter_plot(df)

# Create arrow plots
print("\nCreating arrow plots...")
fig2 = create_arrow_plots(df)

plt.show()
