import numpy as np
from experiment_configs.dataset import GEL_BOUNDS


def create_range(gel_bounds, num_points) -> np.ndarray:
    """Create a range of values from start to end."""
    return np.linspace(gel_bounds[0], gel_bounds[1], num_points)


def create_3d_parameter_grid(num_points,
                             pressure_fixed,
                             speed_fixed) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    X = np.array([
        a4c.ravel(),
        a4m.ravel(),
        ac_disol.ravel(),
        emulsion.ravel(),
        np.full(a4c.size, pressure_fixed),
        np.full(a4c.size, speed_fixed)
    ]).T

    return a4c, a4m, ac_disol, X
