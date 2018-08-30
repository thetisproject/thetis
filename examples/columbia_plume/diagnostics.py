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

    def __init__(self, solver_obj, fieldnames, x, y,
                 location_name,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = self.location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
        super(TimeSeriesCallback2D, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=1,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x = x
        self.y = y
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        # test evaluation
        try:
            self.solver_obj.fields.bathymetry_2d.at((self.x, self.y))
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e

        # construct mesh points
        self.xyz = np.array([[self.x, self.y]])
        self._initialized = True

    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(self.xyz))
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, value in zip(self.fieldnames, args):
            out += 'Evaluated {:} at {:}: {:.3g}\n'.format(
                fieldname, self.location_name, value[0])
        out = out[:-1]  # suppress last line break
        return out


class TimeSeriesCallback3D(DiagnosticCallback):
    """
    Evaluates a time series of a field at a given (x,y,z) location.
    """
    name = 'timeseries'
    variable_names = ['value']

    def __init__(self, solver_obj, fieldnames, x, y, z,
                 location_name,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y, 'z': z}
        attrs['location_name'] = self.location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
        super(TimeSeriesCallback3D, self).__init__(
            solver_obj,
            outputdir=outputdir,
            array_dim=1,
            attrs=attrs,
            export_to_hdf5=export_to_hdf5,
            append_to_log=append_to_log)
        self.x = x
        self.y = y
        self.z = z
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir

        try:
            min_z = -self.solver_obj.fields.bathymetry_2d.at((self.x, self.y))
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e

        if self.z < min_z:
            new_z = min_z + 0.1
            warning('Water depth too shallow at {:}; replacing z={:} by z={:}'.format(self.location_name, self.z, new_z))
            self.z = new_z

        # construct mesh points
        self.xyz = np.array([[self.x, self.y, self.z]])
        self._initialized = True

    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(self.xyz))
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, value in zip(self.fieldnames, args):
            out += 'Evaluated {:} at {:} {:.2f} m: {:.3g}\n'.format(
                fieldname, self.location_name, self.z, value[0])
        out = out[:-1]  # suppress last line break
        return out


class VerticalProfileCallback(DiagnosticCallback):
    """
    Evaluates a vertical profile of a given field at a given (x,y) location.
    """
    name = 'vertprofile'
    variable_names = ['z_coord', 'value']

    def __init__(self, solver_obj, fieldnames, x, y,
                 location_name,
                 npoints=48,
                 outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        assert export_to_hdf5 is True
        self.fieldnames = fieldnames
        self.location_name = location_name
        attrs = {'x': x, 'y': y}
        attrs['location_name'] = self.location_name
        field_short_names = [f.split('_')[0] for f in self.fieldnames]
        field_str = '-'.join(field_short_names)
        self.variable_names = ['z_coord'] + field_short_names
        self.name += '_' + self.location_name
        self.name += '_' + field_str
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
        self.xy = np.array([self.x, self.y])
        self.xyz = np.zeros((self.npoints, 3))
        self.xyz[:, 0] = self.x
        self.xyz[:, 1] = self.y
        self.epsilon = 1e-2  # nudge points to avoid libspatialindex errors
        self.alpha = np.linspace(0, 1, self.npoints)
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir
        self._initialized = True

    def _construct_z_array(self):
        # construct mesh points for func evaluation
        try:
            depth = self.solver_obj.fields.bathymetry_2d.at(self.xy)
            elev = self.solver_obj.fields.elev_cg_2d.at(self.xy)
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e
        z_min = -(depth - self.epsilon)
        z_max = elev - self.epsilon
        self.xyz[:, 2] = z_max + (z_min - z_max)*self.alpha

    def __call__(self):
        if not self._initialized:
            self._initialize()
        # update time-dependent z array
        self._construct_z_array()

        outvals = [self.xyz[:, 2]]
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(self.xyz))
                outvals.append(arr)
            except PointNotInDomainError as e:
                error('{:}: Cannot evaluate data at station {:}'.format(self.__class__.__name__, self.location_name))
                raise e
        return tuple(outvals)

    def message_str(self, *args):
        out = ''
        for fieldname, prof in zip(self.fieldnames, args[1:]):
            minval = prof.min()
            maxval = prof.max()
            out += 'Evaluated {:} profile at {:}: range {:.3g} - {:.3g}\n'.format(
                fieldname, self.location_name, minval, maxval)
        out = out[:-1]  # suppress last line break
        return out
