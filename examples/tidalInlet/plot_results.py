import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.cm import get_cmap 
import matplotlib.colors as colors
import colorcet as cc  # Required if using 'cet_rainbow4' colormap

# Execution set-up is at the end of this script

# FUNCTIONS:
# ==============================================================================
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

def plot_combined_view(
    timestep_val: int,
    coredirout: str,
    extension: str,
    cmap_obj: plt.cm,
    fig_size: tuple = (18, 16),
    aspect_equal: bool = True
) -> None:
    """
    Plots velocity, sediment, bathymetry, and SSH fields in a 2x2 subplot for a given timestep.
    """
    
    # Configure the min-max color bar values for the fields
    plot_configs = {
        'vel': {
            'VARIABLE_BASE_NAME': 'Velocity2d',
            'FIELD_NAME_IN_VTU': 'Depth averaged velocity',
            'VARIABLE_BASE_DIR': 'Velocity2d',
            'vmin': 0.0, 'vmax': 0.5,
            'is_bathymetry': False, 'is_sediments': False, 'is_ssh': False,
            'title': 'Velocity Magnitude'
        },
        'sed': {
            'VARIABLE_BASE_NAME': 'Sediment2d',
            'FIELD_NAME_IN_VTU': 'Sediment',
            'VARIABLE_BASE_DIR': 'Sediment2d',
            'vmin': 0.0, 'vmax': 0.5,
            'is_bathymetry': False, 'is_sediments': True, 'is_ssh': False,
            'title': 'Sediment Concentration'
        },
        'bathy': {
            'VARIABLE_BASE_NAME': 'Bathymetry2d',
            'FIELD_NAME_IN_VTU': 'Bathymetry',
            'VARIABLE_BASE_DIR': 'Bathymetry2d',
            'vmin': 9.0, 'vmax': 12.0,
            'is_bathymetry': True, 'is_sediments': False, 'is_ssh': False,
            'title': 'Bathymetry Change'
        },
        'ssh': {
            'VARIABLE_BASE_NAME': 'Elevation2d',
            'FIELD_NAME_IN_VTU': 'Elevation',
            'VARIABLE_BASE_DIR': 'Elevation2d',
            'vmin': 0.94, 'vmax': 1.03,
            'is_bathymetry': False, 'is_sediments': False, 'is_ssh': True,
            'title': 'Sea Surface Height (SSH)'
        }
    }

    def get_data_for_plot(config):
        """Helper function to load and process data for a single plot."""
        data_dir = os.path.join(coredirout, config['VARIABLE_BASE_DIR'], config['VARIABLE_BASE_DIR'])
        file_name = f"{config['VARIABLE_BASE_NAME']}_{timestep_val}.{extension}"
        filepath = os.path.join(data_dir, file_name)

        if not os.path.exists(filepath):
            print(f"Warning: File not found for {config['title']} at timestep {timestep_val}. Skipping.")
            return None

        try:
            mesh = pv.read(filepath)
            if config['is_bathymetry']:
                originalbathy_mesh = pv.read(os.path.join(data_dir, f"{config['VARIABLE_BASE_NAME']}_{0}.{extension}"))
            if config['is_ssh']:
                filepath_vel = os.path.join(f'{coredirout}/Velocity2d/Velocity2d', f"Velocity2d_{timestep_val}.{extension}")
                mesh_vel = pv.read(filepath_vel)
        except Exception as e:
            print(f"Error reading mesh for {config['title']}: {e}. Skipping.")
            return None

        field_name = config['FIELD_NAME_IN_VTU']
        if field_name in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()
            if config['is_bathymetry']:
                originalbathy_mesh = originalbathy_mesh.cell_data_to_point_data()

        if field_name not in mesh.point_data:
            print(f"Error: Field '{field_name}' not found for {config['title']}. Skipping.")
            return None

        raw_field_data = mesh.point_data[field_name]
        velocity_x, velocity_y = None, None

        is_vector_field = (field_name == 'Depth averaged velocity' and raw_field_data.ndim > 1 and raw_field_data.shape[-1] >= 2)

        if is_vector_field:
            velocity_x, velocity_y = raw_field_data[:, 0], raw_field_data[:, 1]
            field_data = np.sqrt(velocity_x**2 + velocity_y**2)
        elif config['is_bathymetry']:
            raw_field_data_originalbathy = originalbathy_mesh.point_data[field_name]
            field_data = (raw_field_data_originalbathy - raw_field_data) + 10
        elif config['is_sediments']:
            field_data = raw_field_data * 1024
        elif config['is_ssh']:
            field_data = raw_field_data
            raw_field_data_vel = mesh_vel.point_data['Depth averaged velocity']
            velocity_x, velocity_y = raw_field_data_vel[:, 0], raw_field_data_vel[:, 1]
        else:
            field_data = raw_field_data

        # Triangulation
        try:
            tri_mesh = mesh.extract_surface().extract_all_triangles()
            if tri_mesh.n_cells > 0 and tri_mesh.faces.size > 0:
                 triangle_cells = tri_mesh.faces.reshape(-1, 4)[:, 1:] if tri_mesh.faces[0] == 3 else tri_mesh.faces.reshape(-1, 3)
            else:
                triangle_cells = mesh.cells.reshape((-1, 4))[:, 1:]
        except Exception:
            triangle_cells = mesh.cells.reshape((-1, 4))[:, 1:]

        unique_indices = np.unique(triangle_cells.flatten())
        idx_map = np.full(mesh.points.shape[0], -1, dtype=int)
        idx_map[unique_indices] = np.arange(len(unique_indices))
        
        filtered_points = mesh.points[unique_indices, :2]
        filtered_field_data = field_data[unique_indices]
        x_coords, y_coords = filtered_points[:, 0], filtered_points[:, 1]
        remapped_cells = idx_map[triangle_cells]
        triangulation = tri.Triangulation(x_coords, y_coords, remapped_cells)

        return {
            'triangulation': triangulation,
            'filtered_field_data': filtered_field_data,
            'config': config,
            'is_vector_field': is_vector_field or config['is_ssh'],
            'velocity_x': velocity_x[unique_indices] if velocity_x is not None else None,
            'velocity_y': velocity_y[unique_indices] if velocity_y is not None else None,
            'filtered_points': filtered_points
        }

    # --- Plotting Setup ---
    fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=200)
    fig.suptitle(f"Combined View — Timestep {timestep_val}", fontsize=24)
    
    plot_order = [('vel', axes[0, 0]), ('sed', axes[0, 1]), 
                  ('bathy', axes[1, 0]), ('ssh', axes[1, 1])]

    for key, ax in plot_order:
        plot_data = get_data_for_plot(plot_configs[key])
        if plot_data is None:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            ax.set_title(plot_configs[key]['title'])
            continue

        # --- Plotting Logic ---
        config = plot_data['config']
        vmin, vmax = config['vmin'], config['vmax']
        contour_step = (vmax - vmin) / NUM_CBARS
        levels = np.arange(vmin, vmax + contour_step, contour_step)
        
        im = ax.tricontourf(
            plot_data['triangulation'], 
            plot_data['filtered_field_data'],
            levels=levels, cmap=cmap_obj, extend='both', vmin=vmin, vmax=vmax
        )

        if plot_data['is_vector_field']:
            from scipy.interpolate import griddata
            xi = np.linspace(plot_data['filtered_points'][:, 0].min(), plot_data['filtered_points'][:, 0].max(), 100)
            yi = np.linspace(plot_data['filtered_points'][:, 1].min(), plot_data['filtered_points'][:, 1].max(), 100)
            grid_x, grid_y = np.meshgrid(xi, yi)
            u_grid = griddata(plot_data['filtered_points'], plot_data['velocity_x'], (grid_x, grid_y), method='linear')
            v_grid = griddata(plot_data['filtered_points'], plot_data['velocity_y'], (grid_x, grid_y), method='linear')
            ax.streamplot(grid_x, grid_y, u_grid, v_grid, color='k', linewidth=0.8, arrowsize=1.0, density=1.5)

        # --- Formatting ---
        ax.set_title(config['title'], fontsize=18)
        ax.set_xlabel(r'X, $(km)$', fontsize=16)
        ax.set_ylabel(r'Y, $(km)$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim([0, 15000])
        ax.set_ylim([0, 14000])
        ax.grid(True, linestyle='--', alpha=0.7)
        
        from matplotlib.ticker import FuncFormatter
        def to_km(x, pos): return f"{x / 1000:.0f}"
        ax.xaxis.set_major_formatter(FuncFormatter(to_km))
        ax.yaxis.set_major_formatter(FuncFormatter(to_km))

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', aspect=20, pad=0.08)
        cbar.set_label(config['FIELD_NAME_IN_VTU'], fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        tick_step = contour_step * NUM_CBARS/2
        # cbar.set_ticks(np.arange(vmin, vmax + tick_step, tick_step))
        cbar.set_ticks(np.arange(vmin, vmax + contour_step, contour_step*2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if 'SHOW_FIG' in globals() and SHOW_FIG:
        plt.show()
    
    if 'SAVE_FIG' in globals() and SAVE_FIG:
        FIGURES_DIR = './figures'
        if not os.path.exists(FIGURES_DIR):
            os.makedirs(FIGURES_DIR)
        
        filename = os.path.join(FIGURES_DIR, f"combined_view_{timestep_val}.jpg")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")

    plt.close(fig)




# MAIN execution (combined plot) 
# ==============================================================================

# SET UP parameters below to plot:
    
# - time window to plot (in terms of time indicies)
# TIME_WINDOW = np.arange(47, 50, 1)  #
# TIME_WINDOW = np.arange(4, 6, 1)  #
TIME_WINDOW = np.arange(96, 98, 1)  # ~ 48 hours if half-hourly export (t_export = 1800)

# - output files:
coredirout = './outputs_sed'

# - color map settings 
trunc_cmap_min = 0.1
trunc_cmap_max = 1.0

# - bar plot sections -
NUM_CBARS = 20

# - extension for file reading 
extension = 'vtu' # for single core
extension = 'pvtu' # for multi-core

# - figure params
COLORMAP_NAME = 'cet_rainbow4'
FIGURE_SIZE = (9, 8)
PLOT_ASPECT_EQUAL = True
SAVE_FIG = True
SHOW_FIG = False  # Set to True to show the plot

# Execute
if __name__ == "__main__":
    try:
        selected_colormap = get_cmap(COLORMAP_NAME)
        selected_colormap = truncate_colormap(selected_colormap, trunc_cmap_min, trunc_cmap_max)
    except ValueError:
        print(f"Error: Colormap '{COLORMAP_NAME}' not found. Using 'viridis'.")
        selected_colormap = get_cmap('viridis')

    for ts in TIME_WINDOW:
        plot_combined_view(
            timestep_val=ts,
            coredirout=coredirout,
            extension=extension,
            cmap_obj=selected_colormap
        )
