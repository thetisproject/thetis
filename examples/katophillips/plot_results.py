"""
Plot Kato-Phillips test case results:

- Mixed layer depth versus time
- Vertical profiles of variables at the end of the simulation
"""
from katophillips import *
solver_obj.create_equations()

from thetis.exporter import HDF5Exporter
import numpy


def load_function(var, export_ix):
    func = solver_obj.fields[var]
    fs = func.function_space()
    fname = field_metadata[var]['filename']
    h5reader = HDF5Exporter(fs, outputdir + '/hdf5', fname,
                            next_export_ix=0, verbose=False)
    h5reader.load(export_ix, func)
    return func


# construct plot coordinates
npoints = layers*4
z_max = -(depth - 1e-10)
z = numpy.linspace(0, z_max, npoints)
x = numpy.zeros_like(z)
y = numpy.zeros_like(z)
xyz = numpy.vstack((x, y, z)).T


def get_mixed_layer_depth(export_ix):
    tke_tol = 1e-5
    func = load_function('tke_3d', export_ix)
    tke_arr = numpy.array(func.at(tuple(xyz)))
    ix = tke_arr > tke_tol
    if not ix.any():
        return 0.0
    ml_depth = -z[ix].min()
    return ml_depth


timestamps = np.arange(360 + 1)
ntime = len(timestamps)
time = timestamps*t_export
ml_depth = np.zeros_like(time)
for i in range(ntime):
    ml_depth[i] = get_mixed_layer_depth(timestamps[i])
u_s = 0.01
N0 = 0.01
target = 1.05*u_s*np.sqrt(time/N0)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 16

# plot mixed layer depth
plt.plot(time/3600.0, ml_depth)
plt.plot(time/3600.0, target, 'k:')
plt.xlabel('Time [h]')
plt.ylabel('Mixed layer depth [m]')
plt.savefig('mixed_layer_depth.png', bbox_inches='tight')

ana_data = {}

# plot instantaneous profiles
export_ix = 360
varlist = ['salt_3d', 'buoy_freq_3d', 'shear_freq_3d', 'tke_3d', 'eps_3d', 'len_3d', 'eddy_diff_3d']
nplots = len(varlist)
fig, axlist = plt.subplots(nrows=1, ncols=nplots, sharey=True, figsize=(nplots*2.0, 7))
for v, ax in zip(varlist, axlist):
    func = load_function(v, export_ix)
    arr = numpy.array(func.at(tuple(xyz)))
    if len(arr.shape) == 2:
        # take first component of vectors
        arr = arr[:, 0]
    ax.plot(arr, z, 'k', lw=1.5)
    if v in ana_data:
        zz, uu = ana_data[v]
        ax.plot(uu, zz, 'r', lw=1.7, linestyle='dashed')
    ax.grid(True)
    ax.set_title(field_metadata[v]['shortname'].replace(' ', '\n'))
    ax.set_xlabel('[{:}]'.format(field_metadata[v]['unit']), horizontalalignment='right')
    loc = matplotlib.ticker.MaxNLocator(nbins=3, prune='upper')
    fmt = matplotlib.ticker.ScalarFormatter(useOffset=None, useMathText=None, useLocale=None)
    fmt.set_powerlimits((-2, 3))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
axlist[0].set_ylabel('z [m]')
axlist[0].set_ylim([-depth*1.005, 0])
plt.savefig('profiles.png', bbox_inches='tight', dpi=200)
