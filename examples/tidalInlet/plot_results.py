import os
import glob
import numpy as np
import vtkmodules.vtkIOXML as _vtkio
import vtkmodules.util.numpy_support as _vnp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
# import colorcet as cc  # Required if using 'cet_rainbow4' colormap


def read_mesh(filepath):
    """
    Read a .vtu or .pvtu file using the vtk library already present in the firedrake env.
    Returns (points (N,3), tri_cells (M,3), point_data {name: ndarray}).
    """
    if filepath.endswith('.pvtu'):
        reader = _vtkio.vtkXMLPUnstructuredGridReader()
    else:
        reader = _vtkio.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()
    ug = reader.GetOutput()

    points = _vnp.vtk_to_numpy(ug.GetPoints().GetData())          # (N, 3)
    connectivity = _vnp.vtk_to_numpy(ug.GetCells().GetConnectivityArray())
    tri_cells = connectivity.reshape(-1, 3).astype(np.int32)       # pure triangle mesh

    pd = ug.GetPointData()
    point_data = {pd.GetArrayName(i): _vnp.vtk_to_numpy(pd.GetArray(i))
                  for i in range(pd.GetNumberOfArrays())}
    return points, tri_cells, point_data

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
    fig_size: tuple = (16, 15),
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
            'vmin': 0.0, 'vmax': 1.0,
            'is_bathymetry': False, 'is_sediments': False, 'is_ssh': False,
            'title': 'Velocity Magnitude',
            'units': 'm/s',
            'cmap': 'jet'
        },
        'sed': {
            'VARIABLE_BASE_NAME': 'Sediment2d',
            'FIELD_NAME_IN_VTU': 'Sediment',
            'VARIABLE_BASE_DIR': 'Sediment2d',
            'vmin': 0.0, 'vmax': 1.0,
            'is_bathymetry': False, 'is_sediments': True, 'is_ssh': False,
            'title': 'Sediment Concentration',
            'units': '-',
            'cmap': 'viridis'
        },
        'bathy': {
            'VARIABLE_BASE_NAME': 'Bathymetry2d',
            'FIELD_NAME_IN_VTU': 'Bathymetry',
            'VARIABLE_BASE_DIR': 'Bathymetry2d',
            'vmin': -1.0, 'vmax': 1.0,
            'is_bathymetry': True, 'is_sediments': False, 'is_ssh': False,
            'title': 'Bathymetry Change',
            'units': 'm',
            'cmap': 'RdBu_r'
        },
        'ssh': {
            'VARIABLE_BASE_NAME': 'Elevation2d',
            'FIELD_NAME_IN_VTU': 'Elevation',
            'VARIABLE_BASE_DIR': 'Elevation2d',
            'vmin': 0.94, 'vmax': 1.03,
            'is_bathymetry': False, 'is_sediments': False, 'is_ssh': True,
            'title': 'Sea Surface Height (SSH)',
            'units': 'm',
            'cmap': 'RdYlBu_r'
        }
    }

    def get_data_for_plot(config):
        """Load and process mesh + field data for a single subplot."""
        data_dir = os.path.join(coredirout, config['VARIABLE_BASE_DIR'], config['VARIABLE_BASE_DIR'])
        filepath = os.path.join(data_dir, f"{config['VARIABLE_BASE_NAME']}_{timestep_val}.{extension}")

        if not os.path.exists(filepath):
            print(f"Warning: File not found for {config['title']} at timestep {timestep_val}. Skipping.")
            return None

        try:
            points, tri_cells, pdata = read_mesh(filepath)
            if config['is_bathymetry']:
                _, _, pdata_bathy0 = read_mesh(
                    os.path.join(data_dir, f"{config['VARIABLE_BASE_NAME']}_0.{extension}"))
            if config['is_ssh']:
                _, _, pdata_vel = read_mesh(
                    os.path.join(coredirout, 'Velocity2d', 'Velocity2d',
                                 f"Velocity2d_{timestep_val}.{extension}"))
        except Exception as e:
            print(f"Error reading mesh for {config['title']}: {e}. Skipping.")
            return None

        field_name = config['FIELD_NAME_IN_VTU']
        raw = pdata.get(field_name)
        if raw is None:
            print(f"Error: Field '{field_name}' not found for {config['title']}. Skipping.")
            return None

        velocity_x, velocity_y = None, None
        is_vector_field = (field_name == 'Depth averaged velocity'
                           and raw.ndim > 1 and raw.shape[-1] >= 2)

        if is_vector_field:
            velocity_x, velocity_y = raw[:, 0], raw[:, 1]
            field_data = np.sqrt(velocity_x**2 + velocity_y**2)
        elif config['is_bathymetry']:
            field_data = pdata_bathy0.get(field_name) - raw
        elif config['is_sediments']:
            field_data = raw * 1024
        elif config['is_ssh']:
            field_data = raw
            vel_raw = pdata_vel.get('Depth averaged velocity')
            velocity_x, velocity_y = vel_raw[:, 0], vel_raw[:, 1]
        else:
            field_data = raw

        triangulation = tri.Triangulation(points[:, 0], points[:, 1], tri_cells)

        return {
            'triangulation': triangulation,
            'filtered_field_data': field_data,
            'config': config,
            'is_vector_field': is_vector_field or config['is_ssh'],
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'filtered_points': points[:, :2],
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
            levels=levels, cmap=config['cmap'], extend='both',
            extendrect=False,
            vmin=vmin, vmax=vmax
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

        def to_km(x, pos):
            return f"{x / 1000:.0f}"
        ax.xaxis.set_major_formatter(FuncFormatter(to_km))
        ax.yaxis.set_major_formatter(FuncFormatter(to_km))

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', aspect=20, pad=0.08)
        cbar.set_label(f"{config['FIELD_NAME_IN_VTU']} [{config['units']}]", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        cbar.set_ticks(np.arange(vmin, vmax + contour_step, contour_step * 2))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if 'SHOW_FIG' in globals() and SHOW_FIG:
        plt.show()

    if 'SAVE_FIG' in globals() and SAVE_FIG:
        FIGURES_DIR = './figures'
        if not os.path.exists(FIGURES_DIR):
            os.makedirs(FIGURES_DIR)

        filename = os.path.join(FIGURES_DIR, f"combined_view_{timestep_val}.jpg")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {filename}")

    plt.close(fig)


# MAIN execution
# ==============================================================================
# SET UP parameters below to plot:
# - time window to plot (in terms of time indicies)
TIME_WINDOW = np.arange(96, 98, 1)  # ~ 48 hours if half-hourly export (t_export = 1800)

LAST_FRAME = True

# - output files:
coredirout = './outputs_sed_3600s'

# - color map settings
trunc_cmap_min = 0.1
trunc_cmap_max = 1.0

# - bar plot sections -
NUM_CBARS = 20

# - extension for file reading # CHANGE ME DEPENDING ON THE OUTPUT TYPE (single-core = .vtu, multi-core = .pvtu)
extension = 'vtu'  # for single core
# extension = 'pvtu'  # for multi-core

# - figure params
# COLORMAP_NAME = 'cet_rainbow4'
COLORMAP_NAME = 'jet'
FIGURE_SIZE = (9, 8)
PLOT_ASPECT_EQUAL = True
SAVE_FIG = True
SHOW_FIG = False  # Set to True to show the plot


def get_last_timestep(coredirout, extension):
    """Return the highest timestep index found in the Velocity2d output directory."""
    pattern = os.path.join(coredirout, 'Velocity2d', 'Velocity2d', f'Velocity2d_*.{extension}')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No output files found matching: {pattern}')
    indices = [int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) for f in files]
    return max(indices)


# Execute
if __name__ == "__main__":
    try:
        selected_colormap = get_cmap(COLORMAP_NAME)
        selected_colormap = truncate_colormap(selected_colormap, trunc_cmap_min, trunc_cmap_max)
    except ValueError:
        print(f"Error: Colormap '{COLORMAP_NAME}' not found. Using 'viridis'.")
        selected_colormap = get_cmap('viridis')

    if LAST_FRAME:
        last_ts = get_last_timestep(coredirout, extension)
        print(f'LAST_FRAME=True: plotting timestep {last_ts}')
        ts_range = [last_ts]
    else:
        ts_range = TIME_WINDOW

    for ts in ts_range:
        plot_combined_view(
            timestep_val=ts,
            coredirout=coredirout,
            extension=extension,
            cmap_obj=selected_colormap
        )
