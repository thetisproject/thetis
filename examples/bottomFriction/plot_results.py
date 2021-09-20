"""
Plot result velocity etc profiles at steady state
"""
from thetis import *
from steadyChannel import bottom_friction_test as run_test
from steadyChannel import depth, surf_slope
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict


closure_abbr = {
    'Generic Length Scale': 'gls',
    'k-epsilon': 'k-epsilon',
    'k-omega': 'k-omega',
}

stab_abbr = {
    'Canuto A': 'CanutoA',
    'Canuto B': 'CanutoB',
    'Cheng': 'Cheng',
}


def unique(input_list):
    """
    Returns unique elements in a list
    """
    return list(OrderedDict.fromkeys(input_list))


def parse_params(output_dir):
    words = output_dir.split('_')
    nz = int(words[1].replace('nz', ''))
    gls_closure = words[2].replace('-', ' ').replace('k ', 'k-')
    stability_func = words[3].replace('-', ' ')
    return nz, gls_closure, stability_func


output_dir_list = [
    'outputs_nz250_Generic-Length-Scale_Canuto-A',
]

export_ix = 108


def load_model(output_dir):
    layers, gls_closure, stability_func = parse_params(output_dir)
    solver_obj = run_test(
        layers, gls_closure=gls_closure, stability_func=stability_func,
        iterate=False, load_export_ix=export_ix, no_exports=True)

    # add depth averaged velocity to uv
    uv_3d = solver_obj.fields.uv_3d
    uv_3d += solver_obj.fields.uv_dav_3d

    entry = {
        'solver': solver_obj,
        'nz': layers,
        'gls_closure': gls_closure,
        'stability_func': stability_func,
        'bottom_roughness': solver_obj.options.bottom_roughness
    }
    return entry


data = []
for odir in output_dir_list:
    entry = load_model(odir)
    data.append(entry)


# construct plot coordinates
def construct_plot_coordinates(layers):
    offset = 1e-3
    layer_h = depth/layers
    z = numpy.arange(0, -depth, -layer_h)
    z = numpy.vstack((z - offset, z - layer_h + offset)).T.ravel()
    x = numpy.zeros_like(z)
    y = numpy.zeros_like(z)
    xyz = numpy.vstack((x, y, z)).T
    return z, xyz


layers = 100
offset = depth/layers/2
z = numpy.linspace(0 - offset, -depth + offset, layers)
# compute analytical log profiles
u_max = 0.9  # max velocity in [2] Fig 2.
kappa = 0.4
z_0 = float(data[0]['bottom_roughness'])
# bottom friction velocity from u_max
u_b = u_max * kappa / numpy.log((depth + z_0)/z_0)
# analytical bottom friction velocity
g = float(physical_constants['g_grav'])
u_b = numpy.sqrt(g * depth * abs(surf_slope))

u_log = u_b / kappa * numpy.log((z + depth + z_0)/z_0)
# and viscosity profile
nu = kappa*u_b*(z + z_0 + depth)*(-z/depth)
# assuming that P - epsilon = 0
eps = u_b**3/kappa/depth * (-z) / (z + depth + z_0)
print('   u_b {:}'.format(u_b))
print('max  u {:}'.format(u_log.max()))
print('max nu {:}'.format(nu.max()))

ana_data = {'uv_3d': (z, u_log),
            'eddy_visc_3d': (z, nu),
            'eps_3d': (z[int(len(z)/3):-1], eps[int(len(z)/3):-1]),
            }

xlim = {
    'uv_3d': [0.0, 1.0],
    'tke_3d': [0.0, 7e-3],
    'eps_3d': [-1e-4, 9e-4],
    'eddy_visc_3d': [0.0, 7e-2],
    'len_3d': [0.0, 2.5],
}

# plot instantaneous profiles
varlist = ['uv_3d', 'tke_3d', 'eps_3d', 'len_3d', 'eddy_visc_3d']
log_variables = []
nplots = len(varlist)
fig, axlist = plt.subplots(nrows=1, ncols=nplots, sharey=True,
                           figsize=(nplots*2.3, 6))

# plot analytical solutions
for v, ax in zip(varlist, axlist):
    if v in ana_data:
        zz, uu = ana_data[v]
        ax.plot(uu, zz, 'r', lw=1.7, linestyle='dashed',
                label='analytical', zorder=10)
    ax.grid(True)
    ax.set_title(field_metadata[v]['shortname'].replace(' ', '\n'))
    ax.set_xlabel('[{:}]'.format(field_metadata[v]['unit']), horizontalalignment='right')
    loc = matplotlib.ticker.MaxNLocator(nbins=3, prune='upper')
    fmt = matplotlib.ticker.ScalarFormatter(useOffset=None, useMathText=None, useLocale=None)
    fmt.set_powerlimits((-2, 3))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    if v in xlim:
        ax.set_xlim(xlim[v])


def model_label(entry):
    closure = entry['gls_closure']
    return closure_abbr[closure] + ' ' + stab_abbr[entry['stability_func']]


# plot_models
arrays = {}
for entry in data:
    for v, ax in zip(varlist, axlist):
        solver_obj = entry['solver']
        gls_closure = entry['gls_closure']
        stability_func = entry['stability_func']
        layers = entry['nz']
        z, xyz = construct_plot_coordinates(layers)
        func = solver_obj.fields[v]
        arr = numpy.array(func.at(tuple(xyz)))
        arrays[v] = arr
        print('field: {:} min {:} max {:}'.format(v, arr.min(), arr.max()))
        if len(arr.shape) == 2:
            # take first component of vectors
            arr = arr[:, 0]
        label = model_label(entry)
        if v in log_variables:
            ax.semilogx(arr, z, lw=1.5, ls='solid', label=label, alpha=0.7)
        else:
            ax.plot(arr, z, lw=1.5, ls='solid', label=label, alpha=0.7)

axlist[0].set_ylabel('z [m]')
axlist[0].set_ylim([-depth*1.005, 0])

# add legend
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))


closure_str = '-'.join(sorted(unique([closure_abbr[e['gls_closure']].replace('-', '') for e in data])))
stab_str = '-'.join(sorted(unique([stab_abbr[e['stability_func']] for e in data])))
nz_str = '-'.join(sorted(unique([str(e['nz']) for e in data])))

imgfile = 'profiles_{:}_{:}_nz{:}.png'.format(
    closure_str, stab_str, nz_str)
print('Saving figure {:}'.format(imgfile))
plt.savefig(imgfile, bbox_inches='tight', dpi=200.)
plt.close()

# additional diagnostics plots

fig, axlist = plt.subplots(nrows=1, ncols=nplots, sharey=True,
                           figsize=(nplots*2.3, 6))
z, xyz = construct_plot_coordinates(layers)
m2 = numpy.array(solver_obj.fields.shear_freq_3d.at(tuple(xyz)))
nu = numpy.array(solver_obj.fields.eddy_visc_3d.at(tuple(xyz)))
dudz = numpy.sqrt(m2)
eps = numpy.array(solver_obj.fields.eps_3d.at(tuple(xyz)))

ax = axlist[0]
ax.plot(dudz, z, lw=1.5, ls='solid', label='dudz', alpha=0.7)
ax.legend()

ax = axlist[1]
ax.plot(nu, z, lw=1.5, ls='solid', label='nu', alpha=0.7)
ax.legend()

ax = axlist[2]
nududz = nu*dudz
ax.plot(nududz, z, lw=1.5, ls='solid', label='nu*dudz', alpha=0.7)
ax.legend()

ax = axlist[3]
ax.plot(m2, z, lw=1.5, ls='solid', label='M2', alpha=0.7)
ax.plot(dudz**2, z, lw=1.5, ls='solid', label='dudz2', alpha=0.7)
ax.legend()

ax = axlist[4]
ax.plot(eps, z, lw=1.5, ls='solid', label='eps', alpha=0.7)
p = nu*m2
ax.plot(p, z, lw=1.5, ls='dashed', label='P', alpha=0.7)
ax.legend()

imgfile = 'diagnostics_{:}_{:}_nz{:}.png'.format(
    closure_str, stab_str, nz_str)
print('Saving figure {:}'.format(imgfile))
plt.savefig(imgfile, bbox_inches='tight', dpi=200.)
plt.close()
