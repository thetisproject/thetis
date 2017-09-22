"""
Diagnostic callbacks for the Columbia river plume application
"""
from thetis import *


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
        self.location_name = location_name
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = self.location_name
        self.name += '_{:}'.format(self.location_name)
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
        val = args[0][0]

        line = 'Value of {:} at {:}: {:.3g}'.format(
            self.fieldname, self.location_name, val)
        return line
