"""
Plotting routines
"""
from thetis import *
import itertools

try:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_INSTALLED = True
except ImportError as e:
    print('Warning: matplotlib is not intalled: plotting is disabled')
    MATPLOTLIB_INSTALLED = False


class PlotCallback(DiagnosticCallback):
    """
    Plots density field at runtime.
    """
    name = 'fieldplot'
    variable_names = ['figfile']

    def __init__(self, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        assert export_to_hdf5 is False
        super(PlotCallback, self).__init__(outputdir,
                                           export_to_hdf5,
                                           append_to_log)
        self.export_count = itertools.count()
        self._initialized = False

    def _initialize(self, solver_obj):
        if MATPLOTLIB_INSTALLED:
            outputdir = self.outputdir
            if outputdir is None:
                outputdir = solver_obj.options.outputdir
            imgdir = os.path.join(outputdir, 'plots')
            self.imgdir = create_directory(imgdir)
            # store density difference
            self.rho = solver_obj.fields.density_3d
            self.rho_lim = [self.rho.dat.data.min(), self.rho.dat.data.max()]

            # construct mesh points for plotting
            layers = solver_obj.mesh.topology.layers - 1
            mesh2d = solver_obj.mesh2d
            depth = solver_obj.fields.bathymetry_2d.dat.data.mean()
            self.xyz = mesh2d.coordinates
            x = self.xyz.dat.data[:, 0]
            x_max = x.max()
            x_min = x.min()
            delta_x = solver_obj.fields.h_elem_size_2d.dat.data.mean()*np.sqrt(2)
            n_x = np.round((x_max - x_min)/delta_x)
            npoints = layers*4
            epsilon = 1e-10  # nudge points to avoid libspatialindex errors
            z_max = -(depth - epsilon)
            z = np.linspace(-epsilon, z_max, npoints)
            npoints = n_x + 1
            x = np.linspace(x_min + epsilon, x_max - epsilon, npoints)
            xx, zz = np.meshgrid(x, z)
            yy = np.zeros_like(xx)
            self.mesh_shape = xx.shape
            self.xyz = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
            self.x_plot = x/1000.0  # to km
            self.z_plot = z

            self.cmap = plt.get_cmap('RdBu_r')
            self.cmap.set_over('#ffa500')
            self.cmap.set_under('#00e639')

    def plot_on_ax(self, ax, color_array, clim, titlestr, cbartitle, cmap):
        """Plots a field on given axis"""
        if MATPLOTLIB_INSTALLED:
            cmin, cmax = clim
            overshoot_tol = 1e-6
            # divide each integer in color array to this many levels
            cbar_reso = max(min(int(np.round(5.0/(cmax - cmin)*4)), 6), 1)
            levels = np.linspace(cmin - overshoot_tol, cmax + overshoot_tol,
                                 cbar_reso*(cmax-cmin) + 1)
            p = ax.contourf(self.x_plot, self.z_plot, color_array, levels, cmap=cmap, extend='both')
            ax.set_xlabel('x [km]')
            ax.set_ylabel('z [m]')
            ax.set_title(titlestr)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.3, pad=0.1)
            cb = plt.colorbar(p, cax=cax, orientation='vertical', format='%4.1f')
            cb.set_label(cbartitle)
            cticks = np.linspace(cmin, cmax, int(np.round((cmax-cmin))) + 1)
            cb.set_ticks(cticks)
            cax.invert_yaxis()
            return p

    def __call__(self, solver_obj):
        if not self._initialized:
            self._initialize(solver_obj)

        if MATPLOTLIB_INSTALLED:
            # evaluate function on regular grid
            func = self.rho
            arr = np.array(func.at(tuple(self.xyz)))
            arr = arr.reshape(self.mesh_shape)

            nplots = 1
            fix, ax = plt.subplots(nrows=nplots, figsize=(12, 3.5*nplots))

            iexp = self.export_count.next()
            title = 'Time {:.2f} h'.format(solver_obj.simulation_time/3600.0)
            fname = 'plot_density_{0:06d}.png'.format(iexp)
            fname = os.path.join(self.imgdir, fname)
            varstr = 'Density'
            unit = 'kg m-3'
            clim = self.rho_lim
            clabel = '{:} [{:}]'.format(varstr, unit)
            self.plot_on_ax(ax, arr, clim, title, clabel, self.cmap)
            plt.savefig(fname, dpi=240, bbox_inches='tight')
            plt.close()
            return fname,

    def __str__(self, args):
        line = 'Saving figure: {:}'.format(args[0])
        return line
