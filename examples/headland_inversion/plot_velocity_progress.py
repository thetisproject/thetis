import h5py
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

matplotlib.rc('font', size=18)

# parse user input
parser = argparse.ArgumentParser(
    description='Plot velocity time series progress.',
)
parser.add_argument('-s', '--station', nargs='+',
                    help='Station(s) to plot. Multiple can be defined.',
                    required=True,
                    choices=['stationA', 'stationB', 'stationC', 'stationD', 'stationE']
                    )
parser.add_argument('--case', nargs='+',
                    help='Method of Manning field representation',
                    choices=['Uniform', 'Regions', 'IndependentPointsScheme', 'GradientReg', 'HessianReg'],
                    default=['IndependentPointsScheme'],
                    )
parser.add_argument('--optimiser', nargs='+',
                    help='Optimiser to use for inversion problem',
                    choices=['L-BFGS-B', 'ROL'],
                    default=['ROL'],
                    )
args = parser.parse_args()
station_names = sorted(args.station)

station_str = '-'.join(station_names)

case_to_output_dir = {
    'Uniform': 'uniform_friction',
    'Regions': 'region_based',
    'IndependentPointsScheme': 'independent_points_scheme',
    'GradientReg': 'gradient_regularised',
    'HessianReg': 'hessian_regularised'
}

selected_case = args.case[0]
optimiser = args.optimiser[0]
output_dir_forward = os.path.join('outputs', 'outputs_forward')
output_dir_invert = os.path.join('outputs', 'outputs_inverse', optimiser, case_to_output_dir[selected_case])

niter = 0
nplots = len(station_names)

for i, sta in enumerate(station_names):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    f = os.path.join(output_dir_forward, f'diagnostic_timeseries_{sta}_uv.hdf5')
    with h5py.File(f, 'r') as h5file:
        time = h5file['time'][:].flatten()
        var = h5file[sta][:]
        u_vals = var[:, 0]
        v_vals = var[:, 1]

    g = os.path.join(output_dir_invert, f'diagnostic_timeseries_progress_{sta}_uv.hdf5')
    with h5py.File(g, 'r') as h5file:
        iter_times = h5file['time'][:].flatten()
        u_iter_vals = h5file['uv_u_component'][:]
        v_iter_vals = h5file['uv_v_component'][:]

    niter = u_iter_vals.shape[0] - 1

    ax[0].plot(time, u_vals, 'k:', zorder=3, label='observation', lw=1.3)
    ax[1].plot(time, v_vals, 'k:', zorder=3, label='observation', lw=1.3)
    for j, u in enumerate(u_iter_vals):
        ax[0].plot(iter_times, u, label=f'iteration {j}', lw=0.5)
    for j, v in enumerate(v_iter_vals):
        ax[1].plot(iter_times, v, lw=0.5)
    ax[0].set_title(f'{sta}', size='small')
    ax[0].set_ylabel('u component of velocity (m/s)')
    ax[1].set_ylabel('v component of velocity (m/s)')
    ax[0].grid(True), ax[1].grid(True)
    ax[0].legend(ncol=1, prop={'size': 10})
    ax[1].set_xlabel('Time (s)')

    ax[0].text(0.5, 1.1, f'Optimizing Manning, {niter} iterations', ha='center', transform=ax[0].transAxes)

    imgfile = os.path.join(output_dir_invert, f'optimization_progress_{selected_case}_{station_str}_ts.png')
    print(f'Saving to {imgfile}')
    plt.savefig(imgfile, dpi=200, bbox_inches='tight')
