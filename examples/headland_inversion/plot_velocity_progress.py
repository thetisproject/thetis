import h5py
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

matplotlib.rc('font', size=18)

# parse user input
parser = argparse.ArgumentParser(description='Plot time series progress for elevation and/or velocity.')
parser.add_argument(
    '-s', '--station', nargs='+',
    help='Station(s) to plot. Multiple can be defined.',
    required=True,
    choices=['stationA', 'stationB', 'stationC', 'stationD', 'stationE', 'stationF', 'stationG']
)
parser.add_argument(
    '--case', nargs='+',
    help='Method of Manning field representation',
    choices=['Uniform', 'Regions', 'IndependentPointsScheme', 'GradientReg', 'HessianReg'],
    default=['IndependentPointsScheme'],
)
parser.add_argument('--ensemble', action='store_true',
                    help='Enable ensemble mode with one station/time offset per ensemble member')
args = parser.parse_args()
station_names = sorted(str(s) for s in args.station)
station_str = '-'.join(station_names)

case_to_output_dir = {
    'Uniform': 'uniform_friction',
    'Regions': 'region_based',
    'IndependentPointsScheme': 'independent_points_scheme',
    'GradientReg': 'gradient_regularised',
    'HessianReg': 'hessian_regularised'
}

selected_case = args.case[0]
output_dir_forward = os.path.join('outputs', 'outputs_forward')
output_dir_invert = os.path.join('outputs', 'outputs_inverse', case_to_output_dir[selected_case])

station_to_member = {
    'stationA': 0,
    'stationB': 1,
    'stationC': 2,
    'stationD': 3,
    'stationE': 4,
    'stationF': 5,
    'stationG': 6,
}


def resolve_forward_file(station_name):
    member = station_to_member[station_name]
    if args.ensemble:
        member_file = os.path.join(output_dir_forward, f'member_{member}', f'diagnostic_timeseries_{station_name}.hdf5')
    else:
        member_file = os.path.join(output_dir_forward, f'diagnostic_timeseries_{station_name}.hdf5')
    return member_file


def resolve_progress_file(station_name, variable_suffix):
    member = station_to_member[station_name]
    if args.ensemble:
        member_file = os.path.join(output_dir_invert, f'member_{member}', f'diagnostic_timeseries_progress_{station_name}_{variable_suffix}.hdf5')
    else:
        member_file = os.path.join(output_dir_invert, f'diagnostic_timeseries_progress_{station_name}_{variable_suffix}.hdf5')
    return member_file


# --- Loop through stations ---
for sta in station_names:
    init = False
    iter_times = None
    elev_iter_vals, u_iter_vals, v_iter_vals = None, None, None
    niter = 0
    f = resolve_forward_file(sta)

    if not os.path.exists(f):
        print(f"Skipping {sta}: forward file not found.")
        continue

    # --- Load forward (observed) data ---
    with h5py.File(f, 'r') as h5file:
        time = h5file['time'][:].flatten()
        var = h5file[sta][:]
        ncols = var.shape[1]

        elev_vals = None
        u_vals = None
        v_vals = None

        if ncols == 1:
            elev_vals = var[:, 0]
        elif ncols == 2:
            u_vals = var[:, 0]
            v_vals = var[:, 1]
        elif ncols == 3:
            elev_vals = var[:, 0]
            u_vals = var[:, 1]
            v_vals = var[:, 2]
        else:
            raise ValueError(f"Unrecognized data format for {sta}: shape {var.shape}")

    # --- Try to load inverse progress files ---
    elev_file = resolve_progress_file(sta, 'elev')
    uv_file = resolve_progress_file(sta, 'uv')

    if os.path.exists(elev_file):
        with h5py.File(elev_file, 'r') as h5file:
            iter_times = h5file['time'][:].flatten()
            elev_iter_vals = h5file['elev'][:]
            niter = max(niter, elev_iter_vals.shape[0] - 1)

    if os.path.exists(uv_file):
        with h5py.File(uv_file, 'r') as h5file:
            iter_times = h5file['time'][:].flatten()
            u_iter_vals = h5file['uv_u_component'][:]
            v_iter_vals = h5file['uv_v_component'][:]
            niter = max(niter, u_iter_vals.shape[0] - 1)

    # --- Determine which panels to plot ---
    plot_elev = elev_vals is not None and elev_iter_vals is not None
    plot_uv = u_vals is not None and v_vals is not None and u_iter_vals is not None and v_iter_vals is not None

    n_panels = (1 if plot_elev else 0) + (2 if plot_uv else 0)
    if n_panels == 0:
        print(f"No data to plot for {sta}, skipping.")
        continue

    fig, ax = plt.subplots(n_panels, 1, figsize=(12, 5 * n_panels))
    if n_panels == 1:
        ax = [ax]  # make iterable for uniform handling

    panel_idx = 0

    # --- Plot elevation ---
    if plot_elev:
        init = True
        assert elev_iter_vals is not None
        elev_series = elev_iter_vals
        a = ax[panel_idx]
        a.plot(time, elev_vals, 'k:', lw=1.3, label='observation', zorder=3)
        for j, eta in enumerate(elev_series):
            a.plot(iter_times, eta, lw=0.5, label=f'iteration {j}')
        a.set_ylabel('Elevation (m)')
        a.set_title(f'{sta} (elevation)', size='small')
        a.grid(True)
        a.legend(prop={'size': 10})
        panel_idx += 1

    # --- Plot velocity ---
    if plot_uv:
        assert u_iter_vals is not None and v_iter_vals is not None
        u_series = u_iter_vals
        v_series = v_iter_vals
        a_u = ax[panel_idx]
        a_v = ax[panel_idx + 1]

        a_u.plot(time, u_vals, 'k:', lw=1.3, label='observation', zorder=3)
        a_v.plot(time, v_vals, 'k:', lw=1.3, label='observation', zorder=3)

        for j, u in enumerate(u_series):
            a_u.plot(iter_times, u, lw=0.5, label=f'iteration {j}' if not init else None)
        for j, v in enumerate(v_series):
            a_v.plot(iter_times, v, lw=0.5)

        a_u.set_ylabel('u (m/s)')
        a_v.set_ylabel('v (m/s)')
        a_u.set_title(f'{sta} (velocity)', size='small')
        a_v.set_xlabel('Time (s)')
        a_u.grid(True)
        a_v.grid(True)
        a_u.legend(prop={'size': 10})

    # --- Annotate and save ---
    ax[0].text(
        0.5, 1.1,
        f'Optimizing Manning, {niter} iterations',
        ha='center', transform=ax[0].transAxes
    )

    imgfile = os.path.join(
        output_dir_invert,
        f'optimization_progress_{selected_case}_{sta}_ts.png'
    )
    print(f'Saving to {imgfile}')
    plt.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
