
from analyzer.hyperparameter_analysis import run_all_nested_training
from data.dataloader import GelDataLoader
from experiment_configs.dataset import DatasetConfig
from experiment_configs.model import MODEL_CLASSES
from models.model_trainer import Model
from sklearn.gaussian_process import kernels
import re


def safe_eval_params(model_params_str: str) -> dict:
    """Safely evaluate model parameters string with proper kernel handling."""
    if not isinstance(model_params_str, str):
        return model_params_str

    try:
        # First, try to create a safe evaluation environment with necessary objects
        safe_dict = {
            '__builtins__': {},
            'RBF': kernels.RBF,
            'Matern': kernels.Matern,
            'RationalQuadratic': kernels.RationalQuadratic,
            'DotProduct': kernels.DotProduct,
            'ConstantKernel': kernels.ConstantKernel,
            'kernels': kernels,
        }

        # Try to evaluate with the safe dictionary
        return eval(model_params_str, safe_dict)
    except Exception as e:
        print(f"Error in safe_eval_params: {e}")

        # If safe evaluation fails, try to parse manually for kernel objects
        try:
            # Handle common kernel patterns
            if 'RBF(' in model_params_str:
                # Extract parameters from RBF kernel
                length_scale_match = re.search(r'RBF\(length_scale=([^)]+)\)', model_params_str)
                if length_scale_match:
                    length_scale = float(length_scale_match.group(1))
                    # Replace the kernel string with actual kernel object
                    modified_str = re.sub(r'RBF\([^)]*\)', 'kernels.RBF(length_scale=' + str(length_scale) + ')', model_params_str)
                    return eval(modified_str, {'kernels': kernels})
                else:
                    # Default RBF kernel
                    modified_str = re.sub(r'RBF\([^)]*\)', 'kernels.RBF()', model_params_str)
                    return eval(modified_str, {'kernels': kernels})

            # Handle other kernel types similarly
            for kernel_name in ['Matern', 'RationalQuadratic', 'DotProduct', 'ConstantKernel']:
                if f'{kernel_name}(' in model_params_str:
                    modified_str = re.sub(f'{kernel_name}\\([^)]*\\)', f'kernels.{kernel_name}()', model_params_str)
                    return eval(modified_str, {'kernels': kernels})

            # If no kernel patterns found, try standard eval
            return eval(model_params_str)

        except Exception as e2:
            print(f"Failed to parse model parameters: {model_params_str}, Error: {e2}")
            return {}


def get_best_models(
    dataset: DatasetConfig,
    top_models: int = 1,
    metric: str | None = None,
    verbose: int = 1,
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

    for idx, (row_index, row) in enumerate(summary.iterrows()):
        if idx >= top_models:
            break

        model_name = row['model_name']
        model_params = row['model_params']
        if isinstance(model_params, str):
            model_params = safe_eval_params(model_params)  # Use safe evaluation

        model = Model(MODEL_CLASSES[model_name], model_params)
        model.fit(X_df.values, y_df[dataset.target_column].to_numpy().ravel())
        best_models.append(model)

        if verbose > 0:
            print(f"Trained model {model_name} with params: {model_params}")

    return best_models
