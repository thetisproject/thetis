"""
Plot Kato-Phillips test case results:

- Mixed layer depth versus time
- Vertical profiles of variables at the end of the simulation
"""
from thetis import *
from katophillips import katophillips_test as run_test
from katophillips import depth
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import interp1d
from scipy.optimize import brentq

matplotlib.rcParams['font.size'] = 14


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


def load_model(output_dir, export_ix):
    layers, gls_closure, stability_func = parse_params(output_dir)
    if gls_closure == 'gls':
        gls_closure = 'Generic Length Scale'
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
    }
    return entry


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


def construct_plot_coordinates_centers(layers):
    npoints = layers
    offset = depth/layers/2
    z = numpy.linspace(-offset, -depth+offset, npoints)
    x = numpy.zeros_like(z)
    y = numpy.zeros_like(z)
    xyz = numpy.vstack((x, y, z)).T
    return z, xyz


def model_label(entry, label_attr=None):
    label = {
        'closure': closure_abbr[entry['gls_closure']],
        'stability_func': stab_abbr[entry['stability_func']],
        'nz': '{:} layers'.format(entry['nz']),
    }
    if label_attr is not None:
        return label[label_attr]
    return label['closure'] + ' ' + label['stability_func']


def get_mixed_layer_depth(tke_func, xyz):
    tke_tol = 1e-5
    tke_arr = numpy.array(tke_func.at(tuple(xyz)))
    z = xyz[:, 2]
    if (tke_arr < tke_tol).all():
        return 0.0
    fit = interp1d(z, tke_arr)
    z_zero = 0.0
    if (tke_arr[0] - tke_tol) * (tke_arr[-1] - tke_tol) < 0:
        f = lambda x: fit(x) - tke_tol
        z_zero = brentq(f, z.min(), z.max())
    ml_depth = -z_zero
    return ml_depth


def plot_profiles(data, label_attr=None):

    ana_data = {}

    xlim = {}

    # plot instantaneous profiles
    varlist = ['salt_3d', 'buoy_freq_3d', 'shear_freq_3d', 'tke_3d', 'eps_3d', 'len_3d', 'eddy_diff_3d']
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

    # plot_models
    for entry in data:
        for v, ax in zip(varlist, axlist):
            solver_obj = entry['solver']
            layers = entry['nz']
            z, xyz = construct_plot_coordinates(layers)
            func = solver_obj.fields[v]
            arr = numpy.array(func.at(tuple(xyz)))
            print('field: {:} min {:} max {:}'.format(v, arr.min(), arr.max()))
            if len(arr.shape) == 2:
                # take first component of vectors
                arr = arr[:, 0]
            label = model_label(entry, label_attr=label_attr)
            if v in log_variables:
                ax.semilogx(arr, z, lw=1.5, ls='solid', label=label, alpha=0.7)
            else:
                ax.plot(arr, z, lw=1.5, ls='solid', label=label, alpha=0.7)

    axlist[0].set_ylabel('z [m]')
    axlist[0].set_ylim([-depth*1.005, 0])

    # add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
               prop={'size': 'small'})

    closure_str = '-'.join(sorted(unique([closure_abbr[e['gls_closure']].replace('-', '') for e in data])))
    stab_str = '-'.join(sorted(unique([stab_abbr[e['stability_func']] for e in data])))
    nz_str = '-'.join(map(str, sorted(unique([e['nz'] for e in data]))))

    imgfile = 'kato_profiles_{:}_{:}_nz{:}.png'.format(
        closure_str, stab_str, nz_str)
    print('Saving figure {:}'.format(imgfile))
    plt.savefig(imgfile, bbox_inches='tight', dpi=200.)
    plt.close(fig)


def plot_mixed_layer_depth(data, label_attr=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))

    timestamps = numpy.arange(360 + 1)
    ntime = len(timestamps)

    for entry in data:
        solver_obj = entry['solver']
        layers = entry['nz']
        z, xyz = construct_plot_coordinates_centers(layers)

        ml_depth = numpy.zeros_like(timestamps, dtype=float)
        time = timestamps*solver_obj.options.simulation_export_time

        hdf5_dir = os.path.join(solver_obj.options.output_directory, 'hdf5')
        em = exporter.ExportManager(hdf5_dir, ['tke_3d'], solver_obj.fields,
                                    field_metadata, export_type='hdf5')

        for i in range(ntime):
            # solver_obj.load_state(i)
            tke_func = solver_obj.fields.tke_3d
            em.exporters['tke_3d'].load(i, tke_func)
            ml_depth[i] = get_mixed_layer_depth(tke_func, xyz)

        entry['time'] = time
        entry['ml_depth'] = ml_depth

        label = model_label(entry, label_attr=label_attr)
        ax.plot(time/3600.0, ml_depth, lw=1.5, ls='solid', label=label, alpha=0.7)

    u_s = 0.01
    N0 = 0.01
    target = 1.05*u_s*numpy.sqrt(time/N0)

    # plot mixed layer depth
    ax.plot(time/3600.0, target, 'k:')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Mixed layer depth [m]')
    plt.legend(loc='lower right', prop={'size': 'small'})

    closure_str = '-'.join(sorted(unique([closure_abbr[e['gls_closure']].replace('-', '') for e in data])))
    stab_str = '-'.join(sorted(unique([stab_abbr[e['stability_func']] for e in data])))
    nz_str = '-'.join(map(str, sorted(unique([e['nz'] for e in data]))))

    imgfile = 'kato_mldepth_{:}_{:}_nz{:}.png'.format(
        closure_str, stab_str, nz_str)
    print('Saving figure {:}'.format(imgfile))
    plt.savefig(imgfile, bbox_inches='tight', dpi=200.)
    plt.close(fig)


output_dir_list = [
    'outputs_nz50_k-epsilon_Canuto-A',
]
export_ix = 360


data = []
for odir in output_dir_list:
    entry = load_model(odir, export_ix)
    data.append(entry)

plot_profiles(data)
plot_mixed_layer_depth(data)
