import logging
from optimization.composit_objective import composit_objective_function
from skopt import Optimizer
import numpy as np

logger = logging.getLogger(__name__)


def optimize_value(
    best_models,
    gel_bounds: list[tuple[float, float]],
    base_estimator: str = "GP",
    n_initial_points: int = 10,
    acq_func: str = "gp_hedge",
    polymer_penalty_weight: float = 1,
    leakage_penalty_weight: float = 1
):

    optimizer = Optimizer(dimensions=[(bound[0] * 1.05, bound[1] * 0.95) for bound in gel_bounds],
                          base_estimator=base_estimator,
                          n_initial_points=n_initial_points,
                          acq_func=acq_func
                          )
    for _ in range(150):
        X_raw = optimizer.ask()  # This gives us [A4C, A4M, Ac-Di-Sol] - length 3 or [A4C, A4M, Ac-Di-Sol, Pressure, Speed] - length 5

        # Calculate emulsion from the formula: emulsion = (6000 - a4c - a4m - ac_disol) / 1000
        emulsion = (6000 - X_raw[0] - X_raw[1] - X_raw[2]) / 1000

        if len(X_raw) == 3:
            X_full = np.array([X_raw[0], X_raw[1], X_raw[2], emulsion])
        elif len(X_raw) == 5:
            X_full = np.array([X_raw[0], X_raw[1], X_raw[2], emulsion, X_raw[3], X_raw[4]])
        else:
            raise BaseException(f"Unexpected number of input features. {len(X_raw)}, {gel_bounds}")

        penalty = composit_objective_function(best_models=best_models,
                                              X=X_full,
                                              polymer_penalty_weight=polymer_penalty_weight,
                                              leakage_penalty_weight=leakage_penalty_weight,
                                              )

        # Ensure penalty is a scalar for the optimizer
        if isinstance(penalty, np.ndarray):
            penalty = penalty.item() if penalty.size == 1 else penalty[0]

        optimizer.tell(X_raw, penalty)  # Tell optimizer about the raw parameters (3 features)

    optimal_index = np.argmin(optimizer.yi)
    # Extract the optimal composition and its score
    optimal_gel_composition_raw = optimizer.Xi[optimal_index]

    # Calculate emulsion for the optimal composition
    optimal_emulsion = (6000 - optimal_gel_composition_raw[0] - optimal_gel_composition_raw[1] - optimal_gel_composition_raw[2]) / 1000

    # Return full composition including calculated emulsion
    if len(optimal_gel_composition_raw) == 3:
        optimal_gel_composition = np.array([
            optimal_gel_composition_raw[0],  # A4C
            optimal_gel_composition_raw[1],  # A4M
            optimal_gel_composition_raw[2],  # Ac-Di-Sol
            optimal_emulsion                 # Emulsion
        ])
    elif len(optimal_gel_composition_raw) == 5:
        optimal_gel_composition = np.array([
            optimal_gel_composition_raw[0],  # A4C
            optimal_gel_composition_raw[1],  # A4M
            optimal_gel_composition_raw[2],  # Ac-Di-Sol
            optimal_emulsion,                 # Emulsion
            optimal_gel_composition_raw[3],  # Pressure
            optimal_gel_composition_raw[4]   # Speed
        ])

    # penalty score of the optimal composition
    optimal_score = optimizer.yi[optimal_index]

    return optimal_gel_composition
