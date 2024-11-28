from thetis import *
from thetis.turbines import DiscreteTidalTurbineFarm


class TidalPowerCallback(DiagnosticCallback):
    """
    DERIVED Callback to evaluate tidal stream power at the specified locations
    Based on Thetis' standard `DetectorsCallback`
    """
    def __init__(self, solver_obj,
                 idx,
                 farm_options,
                 field_names,
                 name,
                 turbine_names=None,
                 **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg idx: Index (int), physical ID of farm
        :arg farm_options: Farm configuration (farm_option object)
        :arg field_names: List of fields to be interpolated, e.g. `['pow']`
        :arg name: Unique name for this callback and its associated set of
            locations. This determines the name of the output HDF5 file
            (prefixed with `diagnostic_`).
        :arg turbine_names (optional): List of turbine names (otherwise
            named loca0, loca1, ..., locaN)
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        # printing all location output to log is probably not a useful default:
        kwargs.setdefault('append_to_log', False)
        self.field_dims = []
        for field_name in field_names:
            if field_name != 'pow':
                self.field_dims.append(solver_obj.fields[field_name].function_space().block_size)
        attrs = {
            # use null-padded ascii strings, dtype='U' not supported in hdf5,
            # see http://docs.h5py.org/en/latest/strings.html
            'field_names': numpy.array(field_names, dtype='S'),
            'field_dims': self.field_dims,
        }
        super().__init__(solver_obj, array_dim=sum(self.field_dims), attrs=attrs, **kwargs)

        locations = farm_options.turbine_coordinates
        if turbine_names is None:
            turbine_names = ['loca{:0{fill}d}'.format(i, fill=len(str(len(locations))))
                             for i in range(len(locations))]
        self.loc_names = turbine_names
        self._variable_names = self.loc_names
        self.locations = locations

        # similar to solver2d.py
        p = solver_obj.function_spaces.U_2d.ufl_element().degree()
        quad_degree = 2*p + 1
        fdx = dx(idx, degree=quad_degree)

        self.farm = DiscreteTidalTurbineFarm(solver_obj.mesh2d, fdx, farm_options)
        self.field_names = field_names
        self._name = name

        # Disassemble density field
        xyvec = SpatialCoordinate(self.solver_obj.mesh2d)
        loc_dens = []
        radius = farm_options.turbine_options.diameter / 2
        for (xo, yo) in locations:
            dens = self.farm.turbine_density
            dens = conditional(And(lt(abs(xyvec[0]-xo), radius), lt(abs(xyvec[1]-yo), radius)), dens, 0)
            loc_dens.append(dens)

        self.loc_dens = loc_dens

    @property
    def name(self):
        return self._name

    @property
    def variable_names(self):
        return self.loc_names

    @property
    def get_turbine_coordinates(self):
        """
        Returns a list of turbine locations as x, y coordinates instead
        of Firedrake constants.
        """
        turbine_coordinates = []

        for loc in self.locations:
            x_val, y_val = loc
            turbine_coordinates.append([x_val.values()[0], y_val.values()[0]])

        return numpy.array(turbine_coordinates)

    def _values_per_field(self, values):
        """
        Given all values evaluated in a detector location, return the values per field
        """
        i = 0
        result = []
        for dim in self.field_dims:
            result.append(values[i:i+dim])
            i += dim
        return result

    def message_str(self, *args):
        return '\n'.join(
            'In {}: '.format(name) + ', '.join(
                '{}={}'.format(field_name, field_val)
                for field_name, field_val in zip(self.field_names, self._values_per_field(values)))
            for name, values in zip(self.loc_names, args))

    def _evaluate_field(self, field_name):
        if field_name == 'pow':  # intended use
            _uv, _eta = split(self.solver_obj.fields.solution_2d)
            _depth = self.solver_obj.fields.bathymetry_2d
            farm = self.farm
            self.list_powers = [0] * len(self.locations)
            for j in range(len(self.locations)):
                p1 = assemble(farm.turbine.power(_uv, _depth) * self.loc_dens[j] * farm.dx)
                self.list_powers[j] = p1

            return self.list_powers
        else:
            return self.solver_obj.fields[field_name](self.locations)

    def __call__(self):
        """
        Evaluate all current fields in all detector locations

        Returns a nturbines x ndims array, where ndims is the sum of the
        dimension of the fields.
        """
        nturbines = len(self.locations)
        field_vals = []
        for field_name in self.field_names:
            field_vals.append(numpy.reshape(self._evaluate_field(field_name), (nturbines, -1)))

        return numpy.hstack(field_vals)
