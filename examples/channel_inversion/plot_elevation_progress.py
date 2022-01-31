import h5py
import matplotlib.pyplot as plt


station_names = ['stationA', 'stationB']

# control_field = 'Bathymetry'
control_field = 'Manning'
# control_field = 'InitialElev'

# inv_dir = 'outputs'
inv_dir = f'outputs_{control_field}-opt'

nplots = len(station_names)

fig = plt.figure(figsize=(5*nplots, 10))
ax_list = fig.subplots(nplots, 1)
ax_iter = iter(ax_list)

for sta in station_names:

    f = f'outputs_forward/diagnostic_timeseries_{sta}_elev.hdf5'
    with h5py.File(f, 'r') as h5file:
        time = h5file['time'][:].flatten()
        vals = h5file['elev'][:].flatten()

    g = f'{inv_dir}/diagnostic_timeseries_progress_{sta}_elev.hdf5'
    with h5py.File(g, 'r') as h5file:
        # time = h5file['time'][:].flatten()
        iter_vals = h5file['elev'][:]

    niter = iter_vals.shape[0] - 1

    ax = next(ax_iter)
    ax.plot(time, vals, 'k:', zorder=3, label='observation', lw=1.3)
    for i, v in enumerate(iter_vals):
        ax.plot(time, v, label=f'iteration {i}', lw=0.5)
    ax.set_title(f'{sta}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Elevation (m)')
    ax.legend(ncol=1)

ax = ax_list[0]
ax.text(0.5, 1.1, (f'Optimizing {control_field}, {niter} iterations'),
        ha='center', transform=ax.transAxes, size='large')

imgfile = f'optimization_progress_{control_field}_ts.png'
print(f'Saving to {imgfile}')
plt.savefig(imgfile, dpi=200, bbox_inches='tight')
