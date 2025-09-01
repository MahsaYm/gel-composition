import numpy as np
from experiment_configs.dataset import GEL_BOUNDS


def create_range(gel_bounds, num_points) -> np.ndarray:
    """Create a range of values from start to end."""
    return np.linspace(gel_bounds[0], gel_bounds[1], num_points)


def create_range_centered_on_optimal(gel_bounds, num_points, optimal_value: float) -> np.ndarray:
    """
    Create a range of values that includes the optimal value as one of the grid points.

    Args:
        gel_bounds: Tuple of (min, max) bounds
        num_points: Number of points in the range
        optimal_value: The optimal value that should be included in the grid

    Returns:
        Array of values that includes the optimal value
    """
    min_bound, max_bound = gel_bounds

    # Ensure optimal value is within bounds
    optimal_value = np.clip(optimal_value, min_bound, max_bound)

    # Calculate the range span
    range_span = max_bound - min_bound
    step_size = range_span / (num_points - 1)

    # Find the closest grid point to the optimal value in a standard grid
    standard_range = np.linspace(min_bound, max_bound, num_points)
    closest_idx = np.argmin(np.abs(standard_range - optimal_value))

    # Calculate the offset needed to place optimal value exactly at that grid point
    target_position = min_bound + closest_idx * step_size
    offset = optimal_value - target_position

    # Create the adjusted range
    adjusted_range = standard_range + offset

    # Ensure the range stays within bounds
    if adjusted_range[0] < min_bound:
        adjustment = min_bound - adjusted_range[0]
        adjusted_range += adjustment
    elif adjusted_range[-1] > max_bound:
        adjustment = max_bound - adjusted_range[-1]
        adjusted_range += adjustment

    return adjusted_range


def create_3d_parameter_grid(num_points,
                             pressure_fixed: float | None = None,
                             speed_fixed: float | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D grid of parameter values for visualization.

    Args:
        num_points: Number of points per dimension

    Returns:
        Tuple of (a4c_grid, a4m_grid, ac_disol_grid, X_input_array)
    """
    # Create 3D ranges for the three variables we want to visualize
    a4c_range = create_range(GEL_BOUNDS['Methocel A4C'], num_points)
    a4m_range = create_range(GEL_BOUNDS['Methocel A4M'], num_points)
    ac_disol_range = create_range(GEL_BOUNDS['Ac-Di-Sol'], num_points)

    # Create 3D meshgrid
    a4c, a4m, ac_disol = np.meshgrid(a4c_range, a4m_range, ac_disol_range)

    # Calculate emulsion for each point
    emulsion = (6000 - a4c - a4m - ac_disol) / 1000

    # Create input array for all points in the 3D grid
    if pressure_fixed is None:
        X = np.array([
            a4c.ravel(),
            a4m.ravel(),
            ac_disol.ravel(),
            emulsion.ravel(),
        ]).T
    else:
        X = np.array([
            a4c.ravel(),
            a4m.ravel(),
            ac_disol.ravel(),
            emulsion.ravel(),
            np.full(a4c.size, pressure_fixed),
            np.full(a4c.size, speed_fixed)
        ]).T

    return a4c, a4m, ac_disol, X


def create_3d_parameter_grid_centered_on_optimal(num_points,
                                                 optimal_gel_composition: np.ndarray,
                                                 pressure_fixed: float | None = None,
                                                 speed_fixed: float | None = None
                                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 3D grid of parameter values for visualization, ensuring the optimal composition
    is exactly on one of the grid points.

    Args:
        num_points: Number of points per dimension
        optimal_gel_composition: Array with optimal composition [A4C, A4M, Ac-Di-Sol, ...]
        pressure_fixed: Fixed pressure value (if any)
        speed_fixed: Fixed speed value (if any)

    Returns:
        Tuple of (a4c_grid, a4m_grid, ac_disol_grid, X_input_array)
    """
    # Extract optimal values for the three main components
    optimal_a4c = optimal_gel_composition[0]
    optimal_a4m = optimal_gel_composition[1]
    optimal_ac_disol = optimal_gel_composition[2]

    # Create 3D ranges centered on optimal values
    a4c_range = create_range_centered_on_optimal(GEL_BOUNDS['Methocel A4C'], num_points, optimal_a4c)
    a4m_range = create_range_centered_on_optimal(GEL_BOUNDS['Methocel A4M'], num_points, optimal_a4m)
    ac_disol_range = create_range_centered_on_optimal(GEL_BOUNDS['Ac-Di-Sol'], num_points, optimal_ac_disol)

    # Create 3D meshgrid
    a4c, a4m, ac_disol = np.meshgrid(a4c_range, a4m_range, ac_disol_range)

    # Calculate emulsion for each point
    emulsion = (6000 - a4c - a4m - ac_disol) / 1000

    # Create input array for all points in the 3D grid
    if pressure_fixed is None:
        X = np.array([
            a4c.ravel(),
            a4m.ravel(),
            ac_disol.ravel(),
            emulsion.ravel(),
        ]).T
    else:
        X = np.array([
            a4c.ravel(),
            a4m.ravel(),
            ac_disol.ravel(),
            emulsion.ravel(),
            np.full(a4c.size, pressure_fixed),
            np.full(a4c.size, speed_fixed)
        ]).T

    return a4c, a4m, ac_disol, X
