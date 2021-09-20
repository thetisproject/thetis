"""
Diagnostic callbacks for DOME test case
"""
from thetis import *
from scipy.stats import binned_statistic_2d


class VerticalProfileCallback(DiagnosticCallback):
    """
    Evaluates a vertical profile of a given field at a given (x,y) location.
    """
    name = 'vertprofile'
    variable_names = ['z_coord', 'value']

    def __init__(self, solver_obj, fieldname, x, y, npoints=48,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        assert export_to_hdf5 is True
        self.fieldname = fieldname
        self.name += '_' + self.fieldname
        attrs = {'x': x, 'y': y}
        super(VerticalProfileCallback, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=npoints,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x = x
        self.y = y
        self.npoints = npoints
        self.field = self.solver_obj.fields[self.fieldname]
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # construct mesh points for plotting
        depth = self.solver_obj.fields.bathymetry_2d.at((self.x, self.y))
        epsilon = 1e-5  # nudge points to avoid libspatialindex errors
        z_min = -(depth - epsilon)
        z_max = -5  # do not include the top x m to avoid surface waves
        self.z = numpy.linspace(z_max, z_min, self.npoints)
        x = numpy.array([self.x])
        xx, zz = numpy.meshgrid(x, self.z)
        yy = numpy.zeros_like(xx)
        yy[:] = self.y
        self.mesh_shape = xx.shape
        self.xyz = numpy.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    def __call__(self):
        if not self._initialized:
            self._initialize()

        # evaluate function on regular grid
        func = self.field
        arr = numpy.array(func.at(tuple(self.xyz)))

        return (self.z, arr)

    def message_str(self, *args):
        minval = args[1].min()
        maxval = args[1].max()

        line = 'Evaluated {:} profile, value range: {:.3g} - {:.3g}'.format(
            self.fieldname, minval, maxval)
        return line


class TracerHistogramCallback(DiagnosticCallback):
    r"""
    Evaluates a 2D (x, rho) histogram of tracer volume field.

    For every (x, rho) bin we compute the integral

    .. math::
        I = \int_{Omega} \lambda C dx

    where :math:`\lambda` is a binary indicator function of the (x, rho) bin
    and :math:`C` is the tracer.
    """
    name = 'histogram'
    variable_names = ['value']

    def __init__(self, solver_obj, fieldname, x_bins, rho_bins,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=False):
        assert export_to_hdf5 is True
        self.fieldname = fieldname
        self.name += '_' + self.fieldname
        npoints = (len(x_bins) - 1)*(len(rho_bins) - 1)
        attrs = {'x_bins': x_bins, 'rho_bins': rho_bins}
        super(TracerHistogramCallback, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=npoints,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x_bins = x_bins
        self.rho_bins = rho_bins
        self.npoints = npoints
        self.field = self.solver_obj.fields[self.fieldname]
        self.density = self.solver_obj.fields.density_3d
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # compute x coords and nodal volume fields
        fs = self.field.function_space()
        assert fs == self.density.function_space()
        xyz = SpatialCoordinate(self.solver_obj.mesh)
        self.x_coords = Function(fs, name='x').interpolate(xyz[0])
        test = TestFunction(fs)
        self.nodal_volume = assemble(test*dx)

    def __call__(self):
        if not self._initialized:
            self._initialize()

        # we want to bin array field*nodal_volume based on x_coords and rho
        ndecimals = 4  # round to avoid jitter in binning from parallelization
        x_arr = numpy.around(self.x_coords.dat.data[:], ndecimals)
        rho_arr = numpy.around(self.density.dat.data[:], ndecimals)
        c_arr = self.field.dat.data[:]*self.nodal_volume.dat.data[:]

        statistic = 'sum'
        hist, rho_edges, x_edges, binnumber = binned_statistic_2d(
            rho_arr, x_arr, c_arr,
            statistic=statistic,
            bins=[self.rho_bins, self.x_bins])
        # compute global histogram by summing
        comm = self.solver_obj.comm
        global_hist = comm.reduce(hist.ravel(), op=MPI.SUM)

        return (global_hist, )

    def message_str(self, *args):
        minval = args[0].min()
        maxval = args[0].max()

        line = 'Evaluated {:} histogram, value range: {:.3g} - {:.3g}'.format(
            self.fieldname, minval, maxval)
        return line
