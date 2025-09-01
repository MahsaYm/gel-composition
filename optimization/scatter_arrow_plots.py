import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def load_and_prepare_data(path: str = 'optimization/results.csv'):
    """Load and prepare the optimization results data"""
    # Load the data
    df = pd.read_csv(path)

    # We don't need to clean dataset_type column anymore since we'll use it as-is
    # df['dataset_type_clean'] = df['dataset_type'].str.strip().str.lower()

    # Create input/target condition variables
    # Check for printing parameters (Pressure and Speed indicate printing conditions)
    df['has_printing'] = df['input_column'].str.contains('Pressure|Speed', na=False)
    # Check for leakage in targets (case insensitive)
    df['has_leakage'] = df['target_column'].str.contains('leakage|Leakage', na=False)
    df['symbol_type'] = df['has_printing'].astype(str) + '_' + df['has_leakage'].astype(str)

    # Clean input and target columns for better readability
    df['input_column_clean'] = df['input_column'].str.replace('_', ' ').str.title()
    df['target_column_clean'] = df['target_column'].str.replace('_', ' ').str.title()

    return df


def create_scatter_plot(df):
    """Create an enhanced scatter plot with colors for datasets and symbols for input/target types"""

    # Set up the plot style
    plt.style.use('default')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))

    # Define color mapping for dataset types (4 distinct colors)
    dataset_types = df['dataset_type'].unique()
    dataset_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    color_map = {dataset: dataset_colors[i] for i, dataset in enumerate(sorted(dataset_types))}

    # Define symbol mapping for printing/leakage combinations
    symbol_map = {
        'False_False': 'o',    # No printing, no leakage - circle
        'False_True': 's',     # No printing, with leakage - square
        'True_False': '^',     # With printing, no leakage - triangle up
        'True_True': 'D'       # With printing, with leakage - diamond
    }

    # Create scatter plot for each combination
    for dataset in sorted(dataset_types):
        for symbol_type in symbol_map.keys():
            # Filter data for this specific combination
            mask = (df['dataset_type'] == dataset) & (df['symbol_type'] == symbol_type)
            subset = df[mask]

            if len(subset) == 0:
                continue

            # Decode symbol type for label
            has_printing = symbol_type.split('_')[0] == 'True'
            has_leakage = symbol_type.split('_')[1] == 'True'

            printing_label = "with printing" if has_printing else "no printing"
            leakage_label = "with leakage" if has_leakage else "no leakage"
            label = f"{dataset} ({printing_label}, {leakage_label})"

            # Create scatter plot with more distinct sizes
            scatter = ax.scatter(subset['a4m'], subset['a4c'],
                                 c=color_map[dataset],
                                 marker=symbol_map[symbol_type],
                                 s=subset['top_models'] ** 2 * 30,  # Square scaling for more distinct sizes
                                 alpha=0.7,
                                 label=label,
                                 edgecolors='black',
                                 linewidth=0.5)

    # Customize the plot
    ax.set_xlabel('Methocel A4M', fontsize=14, fontweight='bold')
    ax.set_ylabel('Methocel A4C', fontsize=14, fontweight='bold')
    ax.set_title('Enhanced Scatter Plot: Optimal Gel Compositions\n' +
                 'Colors = Dataset Types, Symbols = Input/Target Conditions, Size = Number of Top Models',
                 fontsize=16, fontweight='bold', pad=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Create legends
    # Legend 1: Dataset colors (bottom left)
    dataset_legend_elements = [mlines.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color_map[dataset],
                                             markersize=10, alpha=0.7,
                                             label=dataset)
                               for dataset in sorted(dataset_types)]
    legend1 = ax.legend(handles=dataset_legend_elements, title='Dataset Types',
                        loc='lower left', bbox_to_anchor=(0, -0.15), ncol=1)

    # Legend 2: Symbol meanings (bottom center)
    symbol_legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=10, alpha=0.7, label='No printing, No leakage'),
        mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                      markersize=10, alpha=0.7, label='No printing, With leakage'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                      markersize=10, alpha=0.7, label='With printing, No leakage'),
        mlines.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                      markersize=10, alpha=0.7, label='With printing, With leakage')
    ]
    legend2 = ax.legend(handles=symbol_legend_elements, title='Input/Target Conditions',
                        loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Legend 3: Size meanings (bottom right) with more distinct sizes
    sizes = [1, 2, 3, 4]
    size_legend_elements = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                          markersize=np.sqrt(size**2 * 30)/4, alpha=0.7,
                                          label=f'Top {size} Models') for size in sizes]
    legend3 = ax.legend(handles=size_legend_elements, title='Number of Top Models',
                        loc='lower right', bbox_to_anchor=(1, -0.15), ncol=1)

    # Add all three legends to the plot as artists
    ax.add_artist(legend1)  # Dataset legend (bottom left)
    ax.add_artist(legend2)  # Symbol legend (bottom center)
    # legend3 is the current legend and will be displayed automatically (bottom right)

    # Adjust layout to accommodate legends at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    return fig


def create_arrow_plots(df):
    """Create scatter plots with arrows showing the effect of different variables"""

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    axes = [ax1, ax2, ax3, ax4]

    # Variables to study
    variables = ['top_models', 'polymer_penalty_weight', 'leakage_penalty_weight', 'dataset_type']

    titles = [
        'Effect of Number of Top Models',
        'Effect of Polymer Penalty Weight',
        'Effect of Leakage Penalty Weight',
        'Effect of Dataset Type'
    ]

    # Define colors and symbols
    dataset_types = df['dataset_type'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(dataset_types))}

    for idx, (ax, variable, title) in enumerate(zip(axes, variables, titles)):

        arrow_count = 0

        # Different grouping strategy based on variable
        if variable == 'dataset_type':
            # For dataset variation, group by everything except dataset_type
            groups = df.groupby(['symbol_type', 'top_models', 'polymer_penalty_weight', 'leakage_penalty_weight'])
        else:
            # For other variables, group by everything except the variable of interest
            group_cols = ['dataset_type', 'symbol_type', 'top_models', 'polymer_penalty_weight', 'leakage_penalty_weight']
            group_cols.remove(variable)
            groups = df.groupby(group_cols)

        for group_key, group_data in groups:
            if len(group_data) < 2:
                continue

            # Sort by the variable and calculate means for each unique value
            if variable == 'dataset_type':
                # Custom ordering for dataset types (with proper capitalization)
                dataset_order = ["['Optimization']", "['Optimization', 'Mahsa']", "['Optimization', 'Robustness', 'Mahsa']"]
                # Filter out the robustness-only dataset for now
                available_datasets = [d for d in dataset_order if d in group_data[variable].unique()]
                unique_values = available_datasets
            else:
                unique_values = sorted(group_data[variable].unique())

            if len(unique_values) < 2:
                continue

            value_means = []
            for value in unique_values:
                value_data = group_data[group_data[variable] == value]
                if len(value_data) > 0:
                    mean_a4m = value_data['a4m'].mean()
                    mean_a4c = value_data['a4c'].mean()
                    value_means.append((mean_a4m, mean_a4c, value))

            # Create arrows between consecutive values
            if len(value_means) > 1:
                for i in range(len(value_means) - 1):
                    x1, y1, val1 = value_means[i]
                    x2, y2, val2 = value_means[i+1]

                    # Only create arrow if points are different enough
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if distance > 0.01:  # Minimum distance threshold
                        # Get source and destination colors based on the variable values
                        if variable == 'top_models':
                            # Use a green-to-red gradient similar to polymer penalty for top_models (1-4)
                            colors_gradient = ['#00cc00', '#66cc00', '#cccc00', '#cc0000']  # Green to red progression
                            try:
                                source_idx = int(val1) - 1  # Convert to 0-based index
                                dest_idx = int(val2) - 1
                                source_color = colors_gradient[min(max(source_idx, 0), 3)]
                                dest_color = colors_gradient[min(max(dest_idx, 0), 3)]
                            except (ValueError, TypeError):
                                source_color = dest_color = 'red'
                        elif variable == 'polymer_penalty_weight':
                            # Use a color gradient from green to red for penalty weights
                            penalty_colors = {0.1: '#00cc00', 0.316: '#66cc00', 1.0: '#cccc00', 3.16: '#cc6600', 10.0: '#cc0000'}
                            source_color = penalty_colors.get(val1, 'red')
                            dest_color = penalty_colors.get(val2, 'red')
                        elif variable == 'leakage_penalty_weight':
                            # Use a green-to-red gradient similar to polymer penalty for leakage penalty
                            penalty_colors = {0.1: '#00cc00', 0.316: '#66cc00', 1.0: '#cccc00', 3.16: '#cc6600', 10.0: '#cc0000'}
                            source_color = penalty_colors.get(val1, 'red')
                            dest_color = penalty_colors.get(val2, 'red')
                        elif variable == 'dataset_type':
                            # Use green-to-red gradient for dataset complexity progression
                            dataset_colors_map = {
                                "['Optimization']": '#00cc00',  # Green for simplest
                                "['Optimization', 'Mahsa']": '#cccc00',  # Yellow for medium
                                "['Optimization', 'Robustness', 'Mahsa']": '#cc0000',  # Red for most complex
                                "['Optimization', 'Robustness']": '#cc6600'  # Orange for robustness-only
                            }
                            source_color = dataset_colors_map.get(val1, 'red')
                            dest_color = dataset_colors_map.get(val2, 'red')
                        else:
                            source_color = dest_color = 'red'

                        # Create gradient arrow using matplotlib's arrow patch
                        from matplotlib.patches import FancyArrowPatch
                        from matplotlib.colors import LinearSegmentedColormap
                        import matplotlib.colors as mcolors

                        # Create a simple gradient effect by drawing multiple connected line segments
                        n_segments = 10
                        for seg in range(n_segments):
                            # Interpolate position and color
                            t = seg / n_segments
                            t_next = (seg + 1) / n_segments

                            # Linear interpolation of positions
                            seg_x1 = x1 + t * (x2 - x1)
                            seg_y1 = y1 + t * (y2 - y1)
                            seg_x2 = x1 + t_next * (x2 - x1)
                            seg_y2 = y1 + t_next * (y2 - y1)

                            # Linear interpolation of colors
                            source_rgb = mcolors.to_rgb(source_color)
                            dest_rgb = mcolors.to_rgb(dest_color)
                            seg_rgb = (source_rgb[0] + t * (dest_rgb[0] - source_rgb[0]),
                                       source_rgb[1] + t * (dest_rgb[1] - source_rgb[1]),
                                       source_rgb[2] + t * (dest_rgb[2] - source_rgb[2]))
                            seg_color = mcolors.to_hex(seg_rgb)

                            # Draw line segment (no arrowhead needed due to gradient effect)
                            ax.plot([seg_x1, seg_x2], [seg_y1, seg_y2],
                                    color=seg_color, linewidth=1, alpha=0.3, solid_capstyle='round')

                        # Add a triangle arrowhead at the end point using the destination color
                        # Make the arrow point inward so it doesn't extend beyond the endpoint

                        # Get data range to scale arrowhead appropriately
                        x_range = ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] != ax.get_xlim()[0] else 100
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 100
                        arrow_scale = min(x_range, y_range) * 0.02  # 2% of the smaller axis range

                        arrow_dx = x2 - x1
                        arrow_dy = y2 - y1
                        arrow_norm = np.sqrt(arrow_dx**2 + arrow_dy**2)

                        if arrow_norm > 0:
                            # Normalize direction vector
                            arrow_dx_norm = arrow_dx / arrow_norm
                            arrow_dy_norm = arrow_dy / arrow_norm

                            # Calculate arrowhead vertices (pointing inward from endpoint)
                            arrow_base_x = x2 - arrow_dx_norm * arrow_scale
                            arrow_base_y = y2 - arrow_dy_norm * arrow_scale

                            # Perpendicular vector for arrowhead width
                            perp_x = -arrow_dy_norm * (arrow_scale * 0.5)
                            perp_y = arrow_dx_norm * (arrow_scale * 0.5)

                            # Triangle vertices
                            triangle_x = [x2, arrow_base_x + perp_x, arrow_base_x - perp_x, x2]
                            triangle_y = [y2, arrow_base_y + perp_y, arrow_base_y - perp_y, y2]

                            # Draw filled triangle arrowhead
                            ax.fill(triangle_x, triangle_y, color=dest_color, alpha=0.4, edgecolor=dest_color, linewidth=0.2)

                        arrow_count += 1

        # Customize subplot
        ax.set_xlabel('Methocel A4M', fontsize=12)
        ax.set_ylabel('Methocel A4C', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add color legend for arrows based on variable
        if variable == 'top_models':
            legend_elements = [
                mlines.Line2D([0], [0], color='#00cc00', linewidth=3, label='1 Model'),
                mlines.Line2D([0], [0], color='#66cc00', linewidth=3, label='2 Models'),
                mlines.Line2D([0], [0], color='#cccc00', linewidth=3, label='3 Models'),
                mlines.Line2D([0], [0], color='#cc0000', linewidth=3, label='4 Models')
            ]
            ax.legend(handles=legend_elements, title='Arrow Colors', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)
        elif variable in ['polymer_penalty_weight', 'leakage_penalty_weight']:
            # Both use the same green-to-red color scheme now
            colors = ['#00cc00', '#66cc00', '#cccc00', '#cc6600', '#cc0000']
            title_text = 'Polymer Penalty' if variable == 'polymer_penalty_weight' else 'Leakage Penalty'

            legend_elements = [
                mlines.Line2D([0], [0], color=colors[0], linewidth=3, label='0.1'),
                mlines.Line2D([0], [0], color=colors[1], linewidth=3, label='0.316'),
                mlines.Line2D([0], [0], color=colors[2], linewidth=3, label='1.0'),
                mlines.Line2D([0], [0], color=colors[3], linewidth=3, label='3.16'),
                mlines.Line2D([0], [0], color=colors[4], linewidth=3, label='10.0')
            ]
            ax.legend(handles=legend_elements, title=f'{title_text} Values',
                      loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=9)
        elif variable == 'dataset_type':
            # Updated legend colors to match the new green-to-red gradient
            dataset_legend_data = [
                ("['Optimization']", '#00cc00', 'Optimization Only'),
                ("['Optimization', 'Mahsa']", '#cccc00', 'Optimization + Mahsa'),
                ("['Optimization', 'Robustness', 'Mahsa']", '#cc0000', 'Optimization + Mahsa + Robustness')
            ]
            legend_elements = [
                mlines.Line2D([0], [0], color=color, linewidth=3, label=label)
                for _, color, label in dataset_legend_data
            ]
            ax.legend(handles=legend_elements, title='Dataset Types', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9)

        # Set reasonable axis limits
        all_a4m = df['a4m'].values
        all_a4c = df['a4c'].values
        margin_m = (all_a4m.max() - all_a4m.min()) * 0.05
        margin_c = (all_a4c.max() - all_a4c.min()) * 0.05

        ax.set_xlim(all_a4m.min() - margin_m, all_a4m.max() + margin_m)
        ax.set_ylim(all_a4c.min() - margin_c, all_a4c.max() + margin_c)

    # Add overall title
    fig.suptitle('Variable Effects on Optimal Gel Composition (Arrows show direction of change)',
                 fontsize=16, fontweight='bold')

    # Adjust layout to accommodate legends at the top
    plt.tight_layout(rect=(0, 0, 1, 0.92))  # Leave space at the top for legends
    plt.subplots_adjust(top=0.85)  # Additional space for legends
    return fig


def visualize_scatter_and_arrow_plots(path: str = 'optimization/results.csv'):
    df = load_and_prepare_data(path=path)

    # Create scatter plot
    fig1 = create_scatter_plot(df)

    # Create arrow plots
    fig2 = create_arrow_plots(df)

    # Show plots
    plt.show()
