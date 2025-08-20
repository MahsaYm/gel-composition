
import numpy as np
from experiment_configs.dataset import GEL_BOUNDS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.model_trainer import Model
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from optimization.grid_operation import create_range
from typing import cast
from scipy.interpolate import RectBivariateSpline


def create_voxel_colors_and_alpha(penalties: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Create colors and alpha values for voxel visualization with quadratic scaling.

    Args:
        penalties: 3D array of prediction values

    Returns:
        Tuple of (colors_with_alpha, pred_min, pred_max)
    """

    # Normalize penalties for color mapping and transparency
    pred_min, pred_max = penalties.min(), penalties.max()
    pred_normalized = (penalties - pred_min) / (pred_max - pred_min)

    # Create quadratic scaling: most values transparent, only lowest values opaque
    # Apply much more aggressive scaling to make gray/white values completely transparent
    quadratic_scaling = pred_normalized ** 2  # Even more aggressive than quadratic

    # Create alpha values: HIGH penalties are transparent, LOW penalties are opaque
    # Use a threshold approach: only the lowest values get significant opacity
    alpha_values = quadratic_scaling

    # Make the transparency much more aggressive - only very low values are visible
    alpha_values = np.where(pred_normalized < 0.5, 1 - quadratic_scaling, 0.0)  # Only bottom 30% visible
    alpha_values = np.clip(alpha_values, 0.0, 0.95)  # Completely transparent to very opaque

    # Create colors: Use 'Reds_r' (reversed) so low values are red, high values are white
    norm = Normalize(vmin=pred_min, vmax=pred_max)
    colormap = get_cmap('Reds_r')  # Reversed colormap: low=red, high=white
    colors = colormap(norm(penalties))

    # Set alpha channel in colors array (using quadratic scaling)
    colors[..., 3] = alpha_values  # Set alpha channel

    return colors, pred_min, pred_max


def create_smooth_contour_data(data: np.ndarray, x_range: np.ndarray, y_range: np.ndarray, resolution: int = 15):
    """
    Create smooth contour data by interpolating the original data.

    Args:
        data: 2D numpy array to create contours from
        x_range: Range of x values
        y_range: Range of y values  
        resolution: Number of points for interpolation grid

    Returns:
        Tuple of (X_smooth, Y_smooth, Z_smooth) for contour plotting
    """
    # Create original coordinate grids
    x_orig = np.linspace(x_range[0], x_range[-1], data.shape[1])
    y_orig = np.linspace(y_range[0], y_range[-1], data.shape[0])

    # Create high-resolution grids for smooth contours
    x_smooth = np.linspace(x_range[0], x_range[-1], resolution)
    y_smooth = np.linspace(y_range[0], y_range[-1], resolution)
    X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)

    # Interpolate the data to the high-resolution grid
    from scipy.interpolate import RectBivariateSpline
    interpolator = RectBivariateSpline(y_orig, x_orig, data, kx=3, ky=3, s=0)
    Z_smooth = interpolator(y_smooth, x_smooth)

    return X_smooth, Y_smooth, Z_smooth


def visualize_penalties(
    penalties: np.ndarray,
) -> None:
    """
    Visualize the penalties using a 3D voxel plot and three 2D heatmaps.
    """
    fig = plt.figure(figsize=(16, 12))

    num_points = round(penalties.shape[0] ** (1/3))  # Assuming penalties is a cubic grid
    penalties_3d = penalties.reshape((num_points, num_points, num_points))

    # Create 3D voxel plot in subplot (2,2,1)
    ax_3d = cast(Axes3D, fig.add_subplot(2, 2, 1, projection='3d'))
    visualize_3d_voxels(
        penalties=penalties_3d,
        ax=ax_3d,
    )

    # Find the minimum penalty location
    min_idx = np.unravel_index(np.argmin(penalties_3d), penalties_3d.shape)
    min_a4c_idx, min_a4m_idx, min_ac_disol_idx = min_idx

    # Get parameter ranges for axis labels
    a4c_range = create_range(GEL_BOUNDS['Methocel A4C'], num_points)
    a4m_range = create_range(GEL_BOUNDS['Methocel A4M'], num_points)
    ac_disol_range = create_range(GEL_BOUNDS['Ac-Di-Sol'], num_points)

    # Create 2D heatmaps at minimum penalty location

    # Subplot (2,2,2): A4M vs A4C plane (fixed Ac-Di-Sol)
    ax1 = fig.add_subplot(2, 2, 2)
    plane_a4m_a4c = penalties_3d[:, :, min_ac_disol_idx]

    # Create smooth contour data
    X1, Y1, Z1 = create_smooth_contour_data(plane_a4m_a4c.T, a4c_range, a4m_range)

    # Create filled contour plot
    contour1 = ax1.contourf(X1, Y1, Z1, levels=20, cmap='Reds_r', alpha=0.8)
    # Add contour lines for better definition
    ax1.contour(X1, Y1, Z1, levels=20, colors='black', alpha=0.3, linewidths=0.5)

    ax1.set_title(f'A4M vs A4C\n(Ac-Di-Sol = {ac_disol_range[min_ac_disol_idx]:.1f})')
    ax1.set_xlabel('Methocel A4C')
    ax1.set_ylabel('Methocel A4M')
    plt.colorbar(contour1, ax=ax1, label='Penalty')

    # Mark minimum point
    ax1.plot(a4c_range[min_a4c_idx], a4m_range[min_a4m_idx], 'b*',
             markersize=15, markeredgecolor='black', markeredgewidth=1)

    # Subplot (2,2,3): A4M vs Ac-Di-Sol plane (fixed A4C)
    ax2 = fig.add_subplot(2, 2, 3)
    plane_a4m_ac_disol = penalties_3d[min_a4c_idx, :, :]

    # Create smooth contour data
    X2, Y2, Z2 = create_smooth_contour_data(plane_a4m_ac_disol, ac_disol_range, a4m_range)

    # Create filled contour plot
    contour2 = ax2.contourf(X2, Y2, Z2, levels=20, cmap='Reds_r', alpha=0.8)
    # Add contour lines for better definition
    ax2.contour(X2, Y2, Z2, levels=20, colors='black', alpha=0.3, linewidths=0.5)

    ax2.set_title(f'A4M vs Ac-Di-Sol\n(A4C = {a4c_range[min_a4c_idx]:.1f})')
    ax2.set_xlabel('Ac-Di-Sol')
    ax2.set_ylabel('Methocel A4M')
    plt.colorbar(contour2, ax=ax2, label='Penalty')

    # Mark minimum point
    ax2.plot(ac_disol_range[min_ac_disol_idx], a4m_range[min_a4m_idx], 'b*',
             markersize=15, markeredgecolor='black', markeredgewidth=1)

    # Subplot (2,2,4): A4C vs Ac-Di-Sol plane (fixed A4M)
    ax3 = fig.add_subplot(2, 2, 4)
    plane_a4c_ac_disol = penalties_3d[:, min_a4m_idx, :]

    # Create smooth contour data
    X3, Y3, Z3 = create_smooth_contour_data(plane_a4c_ac_disol.T, a4c_range, ac_disol_range)

    # Create filled contour plot
    contour3 = ax3.contourf(X3, Y3, Z3, levels=20, cmap='Reds_r', alpha=0.8)
    # Add contour lines for better definition
    ax3.contour(X3, Y3, Z3, levels=20, colors='black', alpha=0.3, linewidths=0.5)

    ax3.set_title(f'A4C vs Ac-Di-Sol\n(A4M = {a4m_range[min_a4m_idx]:.1f})')
    ax3.set_xlabel('Methocel A4C')
    ax3.set_ylabel('Ac-Di-Sol')
    plt.colorbar(contour3, ax=ax3, label='Penalty')

    # Mark minimum point
    ax3.plot(a4c_range[min_a4c_idx], ac_disol_range[min_ac_disol_idx], 'b*',
             markersize=15, markeredgecolor='black', markeredgewidth=1)

    plt.tight_layout()
    plt.show()


def visualize_3d_voxels(
    penalties: np.ndarray,
    ax: Axes3D,
) -> None:
    """
    Create 3D voxel visualization for a single model's penalties.

    Args:
        penalties: 3D array of penalties values
        pressure_fixed: Fixed pressure value for display
        speed_fixed: Fixed speed value for display
    """

    # Create colors and alpha values
    colors, pred_min, pred_max = create_voxel_colors_and_alpha(penalties)

    # Create a boolean array to show all voxels (filled space)
    filled = np.ones_like(penalties, dtype=bool)

    # Create the 3D plot

    # Create the voxel plot
    ax.voxels(filled, facecolors=colors)

    # Set proper axis scaling and labels to match the actual parameter ranges
    ax.set_xlim(0, penalties.shape[0] - 1)
    ax.set_ylim(0, penalties.shape[1] - 1)
    ax.set_zlim(0, penalties.shape[2] - 1)

    # Create custom tick labels to show actual parameter values
    tick_positions = np.linspace(0, penalties.shape[0] - 1, 5)  # 5 ticks along each axis

    # Get parameter ranges for tick labels
    a4c_range = create_range(GEL_BOUNDS['Methocel A4C'], penalties.shape[0])
    a4m_range = create_range(GEL_BOUNDS['Methocel A4M'], penalties.shape[1])
    ac_disol_range = create_range(GEL_BOUNDS['Ac-Di-Sol'], penalties.shape[2])

    # Map tick positions to actual parameter values
    a4c_tick_labels = [f'{val:.1f}' for val in np.linspace(a4c_range[0], a4c_range[-1], 5)]
    a4m_tick_labels = [f'{val:.1f}' for val in np.linspace(a4m_range[0], a4m_range[-1], 5)]
    ac_disol_tick_labels = [f'{val:.1f}' for val in np.linspace(ac_disol_range[0], ac_disol_range[-1], 5)]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.zaxis.set_ticks(tick_positions)
    ax.set_xticklabels(a4c_tick_labels)
    ax.set_yticklabels(a4m_tick_labels)
    ax.zaxis.set_ticklabels(ac_disol_tick_labels)

    # Create a dummy scatter for colorbar (since voxels doesn't return a mappable)
    dummy_scatter = ax.scatter(0, 0, 0, c=pred_min, cmap='Reds_r', vmin=pred_min, vmax=pred_max, alpha=0)
    plt.colorbar(dummy_scatter, ax=ax, label=f'Penalty', shrink=0.6)

    # Set labels and title
    ax.set_xlabel('Methocel A4C')
    ax.set_ylabel('Methocel A4M')
    ax.set_zlabel('Ac-Di-Sol')
    ax.set_title(
        f'Penalties Visualization\n'
    )

    # Improve the view angle
    ax.view_init(elev=20, azim=45)
