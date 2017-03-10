"""
Diagnostic callbacks for DOME test case
"""
from thetis import *
import itertools


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
        self.export_count = itertools.count()
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
        self.z = np.linspace(z_max, z_min, self.npoints)
        x = np.array([self.x])
        xx, zz = np.meshgrid(x, self.z)
        yy = np.zeros_like(xx)
        yy[:] = self.y
        self.mesh_shape = xx.shape
        self.xyz = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    def __call__(self):
        if not self._initialized:
            self._initialize()

        # evaluate function on regular grid
        func = self.field
        arr = np.array(func.at(tuple(self.xyz)))

        return (self.z, arr)

    def message_str(self, *args):
        minval = args[1].min()
        maxval = args[1].max()

        line = 'Evaluated {:} profile, value range: {:.3g} - {:.3g}'.format(
            self.fieldname, minval, maxval)
        return line
