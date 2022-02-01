import h5py
import matplotlib
import matplotlib.pyplot as plt
import argparse
from collections import Iterable

matplotlib.rc('font', size=10)

# parse user input
parser = argparse.ArgumentParser(
    description='Plot elevation time series progress.',
)
parser.add_argument('-c', '--controls', nargs='+',
                    help='Control variable(s) in the inverse problem. Multiple can be defined.',
                    required=True,
                    choices=['Bathymetry', 'Manning', 'InitialElev']
                    )
parser.add_argument('-s', '--station', nargs='+',
                    help='Station(s) to plot. Multiple can be defined.',
                    required=True,
                    choices=['stationA', 'stationB', 'stationC', 'stationD', 'stationE']
                    )
args = parser.parse_args()
controls = sorted(args.controls)
station_names = sorted(args.station)

ctrl_str = '-'.join(controls)
station_str = '-'.join(station_names)
inv_dir = f'outputs_{ctrl_str}-opt'

nplots = len(station_names)

fig = plt.figure(figsize=(6, 3*nplots))
ax_list = fig.subplots(nplots, 1, sharex=True)
if not isinstance(ax_list, Iterable):
    ax_list = [ax_list]
ax_iter = iter(ax_list)

for i, sta in enumerate(station_names):

    f = f'outputs_forward/diagnostic_timeseries_{sta}_elev.hdf5'
    with h5py.File(f, 'r') as h5file:
        time = h5file['time'][:].flatten()
        vals = h5file['elev'][:].flatten()

    g = f'{inv_dir}/diagnostic_timeseries_progress_{sta}_elev.hdf5'
    with h5py.File(g, 'r') as h5file:
        iter_vals = h5file['elev'][:]

    niter = iter_vals.shape[0] - 1

    ax = next(ax_iter)
    ax.plot(time, vals, 'k:', zorder=3, label='observation', lw=1.3)
    for i, v in enumerate(iter_vals):
        ax.plot(time, v, label=f'iteration {i}', lw=0.5)
    ax.set_title(f'{sta}', size='small')
    ax.set_ylabel('Elevation (m)')
    ax.grid(True)
    ax.legend(ncol=1, prop={'size': 6})
    if i == nplots - 1:
        ax.set_xlabel('Time (s)')

ax = ax_list[0]
ctrl_header = ', '.join(controls)
ax.text(0.5, 1.1, (f'Optimizing {ctrl_header}, {niter} iterations'),
        ha='center', transform=ax.transAxes)

imgfile = f'optimization_progress_{ctrl_str}_{station_str}_ts.png'
print(f'Saving to {imgfile}')
plt.savefig(imgfile, dpi=200, bbox_inches='tight')
