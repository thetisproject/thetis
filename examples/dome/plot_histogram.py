"""
Plots 2D tracer histogram
"""
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy

fontsize = 18
matplotlib.rcParams['font.size'] = fontsize
clim = [0, 8e10]

export_ix = 150  # the export index to load
outputdir = 'outputs_coarse_dg-dg'
filename = 'diagnostic_histogram_salt_3d.hdf5'

# load data set
hdf5_file = h5py.File(os.path.join(outputdir, filename), 'r')
time = hdf5_file['time'][:]
x_bins = hdf5_file.attrs['x_bins']
rho_bins = hdf5_file.attrs['rho_bins']

nx = len(x_bins) - 1
nrho = len(rho_bins) - 1

hist = hdf5_file['value'][export_ix, :]
hist = numpy.reshape(hist, (nrho, nx))

x_mean = numpy.linspace(x_bins.min(), x_bins.max(), len(x_bins))/1000.
rho_mean = numpy.linspace(rho_bins.min(), rho_bins.max(), len(rho_bins))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

cmap = plt.get_cmap('magma')
cmap.set_over('w')
cmap.set_under('k')
cmap.set_bad('k')

print('Loaded export {:}: {:}'.format(export_ix, numpy.linalg.norm(hist)))

p = ax.pcolormesh(x_mean, rho_mean, hist, cmap=cmap, vmin=clim[0], vmax=clim[1])
ax.invert_yaxis()
ax.set_xlabel('x [km]')
ax.set_ylabel(u'Density anomaly [kg/m\u00B3]')
ax.autoscale(tight=True)
ax.grid(True, color='w')
cb = plt.colorbar(p, label=u'Volume [m\u00B3]', extend='max')
imgname = 'tracer_histogram.png'
print('Saving to {:}'.format(imgname))
fig.savefig(imgname, dpi=200, bbox_inches='tight')
