from models.model_trainer import Model
import numpy as np
from experiment_configs.dataset import Target, IDEAL_TARGETS


def penalty_function(predictions: np.ndarray,
                     target: Target,
                     ) -> np.ndarray:
    """ Create a piecewise penalty function based on the target's ideal and unacceptable ranges. 
        If the prediction is within the ideal range, the penalty is 0.
        If it exceeds the ideal max, the penalty increases linearly.
        If it falls below the ideal min, the penalty increases linearly.
        If it exceeds the unacceptable max or falls below the unacceptable min, the penalty grows quadratically.
    """
    penalty = np.zeros_like(predictions, dtype=float)
    # Penalty for exceeding the ideal max
    penalty += np.maximum(0, predictions - target.ideal_max)

    # Penalty for falling below the ideal min
    penalty += np.maximum(0, target.ideal_min - predictions)

    # Penalty for exceeding the unacceptable max
    penalty += np.where(predictions > target.unacceptable_max,
                        (predictions - target.unacceptable_max) ** 2, 0)

    # Penalty for falling below the unacceptable min
    penalty += np.where(predictions < target.unacceptable_min,
                        (target.unacceptable_min - predictions) ** 2, 0)

    return penalty


def composit_objective_function(best_models: dict[str, list[Model]],
                                X: np.ndarray,  # 100xn_input
                                polymer_penalty_weight: float = 0.01
                                ) -> np.ndarray:  # # 100x1, <penalty>
    penalties = {}

    for target_name, model_list in best_models.items():
        preds = np.mean([model.predict(X, inverse_transform=False) for model in model_list], axis=0)
        penalties[target_name] = penalty_function(
            predictions=preds,
            target=IDEAL_TARGETS[target_name].scaled_target(best_models[target_name][0].y_scaler)
        )

        penalties[target_name] *= IDEAL_TARGETS[target_name].importance

    total_penalty = np.sum(list(penalties.values()), axis=0)

    polymer = np.sum(X[:, :3], axis=1)  # Assuming the first 3 columns are polymer properties

    return total_penalty + polymer_penalty_weight * polymer
