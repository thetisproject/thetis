"""
Diagnostic callbacks for the Columbia river plume application
"""
from thetis import *
from scipy.stats import binned_statistic_2d


class TimeSeriesCallback2D(DiagnosticCallback):
    """
    Evaluates a vertical profile of a given field at a given (x,y) location.
    """
    name = 'timeseries'
    variable_names = ['value']

    def __init__(self, solver_obj, fieldname, x, y,
                 location_name,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        assert export_to_hdf5 is True
        self.fieldname = fieldname
        self.name += '_' + self.fieldname
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = location_name
        self.name += '_{:}'.format(location_name)
        super(TimeSeriesCallback2D, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=1,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x = x
        self.y = y
        self.field = self.solver_obj.fields[self.fieldname]
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # construct mesh points
        xx = np.array([self.x])
        yy = np.array([self.y])
        zz = np.zeros_like(xx)
        self.mesh_shape = xx.shape
        self.xyz = np.vstack((xx.ravel(), yy.ravel())).T

    def __call__(self):
        if not self._initialized:
            self._initialize()

        # evaluate function on regular grid
        func = self.field
        arr = np.array(func.at(tuple(self.xyz)))

        return (arr, )

    def message_str(self, *args):
        minval = args[0].min()
        maxval = args[0].max()

        line = 'Evaluated {:} elevation, value range: {:.3g} - {:.3g}'.format(
            self.fieldname, minval, maxval)
        return line
