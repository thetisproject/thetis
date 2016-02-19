"""
Plot result velocity etc profiles at steady state

Tuomas Karna 2016-03-10
"""
from steadyChannel import *

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
npoints = layers*6
z = numpy.linspace(0, -depth, npoints)
x = numpy.zeros_like(z)
y = numpy.zeros_like(z)
xyz = numpy.vstack((x, y, z)).T

import matplotlib
import matplotlib.pyplot as plt

# compute analytical log profiles
u_max = 0.9  # max velocity in [2] Fig 2.
kappa = solver_obj.gls_model.params['von_karman']
z_0 = physical_constants['z0_friction']
u_b = u_max * kappa / numpy.log((depth + z_0)/z_0)
u_log = u_b / kappa * numpy.log((z + depth + z_0)/z_0)
# and viscosity profile
nu = kappa*u_b*(z + z_0 + depth)*(-z/depth)

ana_data = {'uv_3d': (z, u_log),
            'eddy_visc_3d': (z, nu),
            }

# plot instantaneous profiles
export_ix = 108
varlist = ['uv_3d', 'shear_freq_3d', 'tke_3d', 'eps_3d', 'len_3d', 'eddy_visc_3d']
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
plt.savefig('profiles.png', bbox_inches='tight')
