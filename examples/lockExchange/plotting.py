"""
Plotting routines
"""
from thetis import *
import itertools
import copy

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MATPLOTLIB_INSTALLED = True
except ImportError:
    print('Warning: matplotlib is not intalled: plotting is disabled')
    MATPLOTLIB_INSTALLED = False


class PlotCallback(DiagnosticCallback):
    """
    Plots density field at runtime.
    """
    name = 'fieldplot'
    variable_names = ['figfile']

    def __init__(self, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        assert export_to_hdf5 is False
        super(PlotCallback, self).__init__(solver_obj,
                                           outputdir=outputdir,
                                           export_to_hdf5=export_to_hdf5,
                                           append_to_log=append_to_log)
        self.export_count = itertools.count()
        self._initialized = False

    def _initialize(self):
        if MATPLOTLIB_INSTALLED:
            outputdir = self.outputdir
            if outputdir is None:
                outputdir = self.solver_obj.options.output_directory
            imgdir = os.path.join(outputdir, 'plots')
            self.imgdir = create_directory(imgdir)
            # store density difference
            self.rho = self.solver_obj.fields.density_3d
            self.rho_lim = [self.rho.dat.data.min(), self.rho.dat.data.max()]

            # construct mesh points for plotting
            layers = self.solver_obj.mesh.topology.layers - 1
            mesh2d = self.solver_obj.mesh2d
            depth = self.solver_obj.fields.bathymetry_2d.dat.data.mean()
            self.xyz = mesh2d.coordinates
            x = self.xyz.dat.data[:, 0]
            x_max = x.max()
            x_min = x.min()
            delta_x = self.solver_obj.fields.h_elem_size_2d.dat.data.mean()*numpy.sqrt(2)
            n_x = int(numpy.round((x_max - x_min)/delta_x))
            npoints = layers*4
            epsilon = 1e-10  # nudge points to avoid libspatialindex errors
            z_min = -(depth - epsilon)
            z_max = -0.08  # do not include the top 8 cm to avoid surface waves
            z = numpy.linspace(z_max, z_min, npoints)
            npoints = n_x + 1
            x = numpy.linspace(x_min + epsilon, x_max - epsilon, npoints)
            xx, zz = numpy.meshgrid(x, z)
            yy = numpy.zeros_like(xx)
            self.mesh_shape = xx.shape
            self.xyz = numpy.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
            self.x_plot = x/1000.0  # to km
            self.z_plot = z

            self.cmap = copy.copy(plt.get_cmap('RdBu_r'))
            self.cmap.set_over('#ffa500')
            self.cmap.set_under('#00e639')

    def plot_on_ax(self, ax, color_array, clim, titlestr, cbartitle, cmap,
                   ylim=None):
        """Plots a field on given axis"""
        if MATPLOTLIB_INSTALLED:
            cmin, cmax = clim
            overshoot_tol = 1e-6
            # divide each integer in color array to this many levels
            cbar_reso = max(min(int(numpy.round(5.0/(cmax - cmin)*4)), 6), 1)
            nlevels = int(cbar_reso*(cmax-cmin) + 1)
            levels = numpy.linspace(cmin - overshoot_tol, cmax + overshoot_tol,
                                    nlevels)
            p = ax.contourf(self.x_plot, self.z_plot, color_array, levels, cmap=cmap, extend='both')
            ax.set_xlabel('x [km]')
            ax.set_ylabel('z [m]')
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_title(titlestr)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.3, pad=0.1)
            cb = plt.colorbar(p, cax=cax, orientation='vertical', format='%4.1f')
            cb.set_label(cbartitle)
            cticks = numpy.linspace(cmin, cmax, int(numpy.round((cmax-cmin))) + 1)
            cb.set_ticks(cticks)
            cax.invert_yaxis()
            return p

    def __call__(self):
        if not self._initialized:
            self._initialize()

        if MATPLOTLIB_INSTALLED:
            # evaluate function on regular grid
            func = self.rho
            arr = numpy.array(func.at(tuple(self.xyz)))
            arr = arr.reshape(self.mesh_shape)

            nplots = 1
            fix, ax = plt.subplots(nrows=nplots, figsize=(12, 3.5*nplots))

            iexp = next(self.export_count)
            title = 'Time {:.2f} h'.format(self.solver_obj.simulation_time/3600.0)
            fname = 'plot_density_{0:06d}.png'.format(iexp)
            fname = os.path.join(self.imgdir, fname)
            varstr = 'Density'
            unit = 'kg m-3'
            clim = self.rho_lim
            clabel = '{:} [{:}]'.format(varstr, unit)
            ylim = [-20.0, 0.0]
            self.plot_on_ax(ax, arr, clim, title, clabel, self.cmap, ylim=ylim)
            plt.savefig(fname, dpi=240, bbox_inches='tight')
            plt.close()
            return fname,

    def message_str(self, *args):
        line = 'Saving figure: {:}'.format(args[0])
        return line
