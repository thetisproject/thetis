import numpy
import h5py
import matplotlib.pyplot as plt

from plot_elevation_ts import epoch_to_datetime, simtime_to_epoch, init_date

d = h5py.File('rigilk/outputs_normal_dt15/diagnostic_vertprofile_rice_salt-temp.hdf5')
time = d['time'][:]
salt = d['salt'][:]
z = d['z_coord'][:]

time = simtime_to_epoch(d['time'][:], init_date)
ntime, nz = z.shape

t = epoch_to_datetime(time)[:, numpy.newaxis]
T = numpy.tile(t, (1, nz))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
p = ax.contourf(T, z, salt, 64)
cb = plt.colorbar(p, label='Salinity [psu]', pad=0.01)

ylim = list(ax.get_ylim())
ylim[0] = -22.0
ax.set_ylim(ylim)

ax.set_ylabel('Z coordinate [m]')
fig.autofmt_xdate()

t_min, t_max = time.min(), time.max()

date_str = '_'.join([epoch_to_datetime(t).strftime('%Y-%m-%d') for t in [t_min, t_max]])
imgfn = 'vprof_salt_rice_{:}.png'.format(date_str)
print('Saving {:}'.format(imgfn))
fig.savefig(imgfn, dpi=200, bbox_inches='tight')
