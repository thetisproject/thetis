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
        xx = np.array([self.x])
        yy = np.array([self.y])
        self.xyz = np.vstack((xx.ravel(), yy.ravel())).T
        self._initialized = True

    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(tuple(self.xyz)))
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
        xx = np.array([self.x]).ravel()
        yy = np.array([self.y]).ravel()
        zz = np.array([self.z]).ravel()
        self.xyz = np.vstack((xx, yy, zz)).T
        self._initialized = True

    def __call__(self):
        if not self._initialized:
            self._initialize()
        outvals = []
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(tuple(self.xyz)))
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
        self._initialized = False

    def _initialize(self):
        outputdir = self.outputdir
        if outputdir is None:
            outputdir = self.solver_obj.options.outputdir
        self._initialized = True

    def _construct_z_array(self):
        # construct mesh points for func evaluation
        try:
            depth = self.solver_obj.fields.bathymetry_2d.at((self.x, self.y))
            elev = self.solver_obj.fields.elev_cg_2d.at((self.x, self.y))
        except PointNotInDomainError as e:
            error('{:}: Station "{:}" out of horizontal domain'.format(self.__class__.__name__, self.location_name))
            raise e
        epsilon = 1e-2  # nudge points to avoid libspatialindex errors
        z_min = -(depth - epsilon)
        z_max = elev - epsilon
        self.z = np.linspace(z_max, z_min, self.npoints)
        x = np.array([self.x])
        xx, zz = np.meshgrid(x, self.z)
        yy = np.ones_like(xx)*self.y
        self.mesh_shape = xx.shape
        self.xyz = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    def __call__(self):
        if not self._initialized:
            self._initialize()
        # update time-dependent z array
        self._construct_z_array()

        outvals = [self.z]
        for fieldname in self.fieldnames:
            try:
                field = self.solver_obj.fields[fieldname]
                arr = np.array(field.at(tuple(self.xyz)))
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
