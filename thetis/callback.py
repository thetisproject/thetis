"""
Defines custom callback functions used to compute various metrics at runtime.

"""
from __future__ import absolute_import
from .utility import *
from abc import ABC, abstractproperty, abstractmethod
import h5py
from collections import defaultdict
from .log import *
from .firedrake import *
import numpy as np


class CallbackManager(defaultdict):
    """
    Stores callbacks in different categories and provides methods for
    evaluating them.

    Create callbacks and register them under ``'export'`` mode

    .. code-block:: python

        cb1 = VolumeConservation3DCallback(...)
        cb2 = TracerMassConservationCallback(...)
        cm = CallbackManager()
        cm.add(cb1, 'export')
        cm.add(cb2, 'export')

    Evaluate callbacks, calls :func:`evaluate` method of all callbacks
    registered in the given mode.

    .. code-block:: python

        cm.evaluate('export')

    """
    def __init__(self):
        super(CallbackManager, self).__init__(OrderedDict)

    def add(self, callback, mode):
        """
        Add a callback under the given mode

        :arg callback: a :class:`.DiagnosticCallback` object
        :arg str mode: register callback under this mode
        """
        key = callback.name
        self[mode][key] = callback

    def evaluate(self, mode, index=None):
        """
        Evaluate all callbacks registered under the given mode

        :arg str mode: evaluate all callbacks under this mode
        :kwarg int index: if provided, sets the export index. Default behavior
            is to append to the file or stream.
        """
        for key in sorted(self[mode]):
            self[mode][key].evaluate(index=index)


class DiagnosticHDF5(object):
    """
    A HDF5 file for storing diagnostic time series arrays.
    """
    def __init__(self, filename, varnames, array_dim=1, attrs=None,
                 comm=COMM_WORLD, new_file=True, dtype='d',
                 include_time=True):
        """
        :arg str filename: Full filename of the HDF5 file.
        :arg varnames: List of variable names that the diagnostic callback
            provides
        :kwarg int array_dim: Dimension of the output array. 1 for scalars.
        :kwarg dict attrs: Additional attributes to be saved in the hdf5 file.
        :kwarg comm: MPI communicator
        :kwarg bool new_file: Define whether to create a new hdf5 file or
            append to an existing one (if any)
        """
        self.comm = comm
        self.filename = filename
        self.varnames = varnames
        self.nvars = len(varnames)
        self.array_dim = array_dim
        self.include_time = include_time
        if comm.rank == 0 and new_file:
            # create empty file with correct datasets
            with h5py.File(filename, 'w') as hdf5file:
                if include_time:
                    hdf5file.create_dataset('time', (0, 1),
                                            maxshape=(None, 1), dtype=dtype)
                for var in self.varnames:
                    hdf5file.create_dataset(var, (0, array_dim),
                                            maxshape=(None, array_dim), dtype=dtype)
                if attrs is not None:
                    hdf5file.attrs.update(attrs)

    def _expand_array(self, hdf5file, varname):
        """Expands array varname by 1 entry"""
        arr = hdf5file[varname]
        shape = arr.shape
        arr.resize((shape[0] + 1, shape[1]))

    def _expand(self, hdf5file):
        """Expands data arrays by 1 entry"""
        # TODO is there an easier way for doing this?
        for var in self.varnames:
            self._expand_array(hdf5file, var)
        if self.include_time:
            self._expand_array(hdf5file, 'time')

    def _nentries(self, hdf5file):
        return hdf5file[self.varnames[0]].shape[0]

    def export(self, variables, time=None, index=None):
        """
        Appends a new entry of (time, variables) to the file.

        The HDF5 is updated immediately.

        :arg time: time stamp of entry
        :type time: float
        :arg variables: values of entry
        :type variables: tuple of float
        :kwarg int index: If provided, defines the time index in the file
        """
        if self.comm.rank == 0:
            with h5py.File(self.filename, 'a') as hdf5file:
                if index is not None:
                    nentries = self._nentries(hdf5file)
                    assert index <= nentries, 'time index out of range {:} <= {:} \n  in file {:}'.format(index, nentries, self.filename)
                    expand_required = index == nentries
                    ix = index
                if index is None or expand_required:
                    self._expand(hdf5file)
                    ix = self._nentries(hdf5file) - 1
                if self.include_time:
                    assert time is not None, 'time should be provided as 2nd argument to export()'
                    hdf5file['time'][ix] = time
                for i in range(self.nvars):
                    hdf5file[self.varnames[i]][ix, :] = variables[i]
                hdf5file.close()


class DiagnosticCallback(ABC):
    """
    A base class for all Callback classes
    """

    def __init__(self, solver_obj, array_dim=1, attrs=None,
                 outputdir=None,
                 export_to_hdf5=True,
                 append_to_log=True,
                 include_time=True,
                 hdf5_dtype='d'):
        """
        :arg solver_obj: Thetis solver object
        :kwarg str outputdir: Custom directory where hdf5 files will be stored.
            By default solver's output directory is used.
        :kwarg int array_dim: Dimension of the output array. 1 for scalars.
        :kwarg dict attrs: Additional attributes to be saved in the hdf5 file.
        :kwarg bool export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :kwarg bool append_to_log: If True, callback output messages will be
            printed in log
        :kwarg bool include_time: whether to include time in the hdf5 file
        :kwarg hdf5_dtype: Precision to use in hdf5 output: `d` for double
            precision (default), and `f` for single precision
        """
        self.solver_obj = solver_obj
        if outputdir is None:
            self.outputdir = self.solver_obj.options.output_directory
        else:
            self.outputdir = outputdir
        self.array_dim = array_dim
        self.attrs = attrs
        self.append_to_hdf5 = export_to_hdf5
        self.append_to_log = append_to_log
        self.hdf5_dtype = hdf5_dtype
        self.include_time = include_time
        self._create_new_file = True
        self._hdf5_initialized = False

    def set_write_mode(self, mode):
        """
        Define whether to create a new hdf5 file or append to an existing one

        :arg str mode: Either 'create' (default) or 'append'
        """
        assert mode in ['create', 'append']
        self._create_new_file = mode == 'create'

    def _create_hdf5_file(self):
        """
        Creates an empty hdf5 file with correct datasets.
        """
        if self.append_to_hdf5:
            comm = self.solver_obj.comm
            create_directory(self.outputdir, comm=comm)
            fname = 'diagnostic_{:}.hdf5'.format(self.name.replace(' ', '_'))
            fname = os.path.join(self.outputdir, fname)
            self.hdf_exporter = DiagnosticHDF5(fname, self.variable_names,
                                               array_dim=self.array_dim,
                                               new_file=self._create_new_file,
                                               attrs=self.attrs,
                                               comm=comm, dtype=self.hdf5_dtype,
                                               include_time=self.include_time)
        self._hdf5_initialized = True

    @abstractproperty
    def name(self):
        """The name of the diagnostic"""
        pass

    @abstractproperty
    def variable_names(self):
        """Names of all scalar values"""
        pass

    @abstractmethod
    def __call__(self):
        """
        Evaluate the diagnostic value.

        .. note::
            This method must implement all MPI reduction operations (if any).
        """
        pass

    @abstractmethod
    def message_str(self, *args):
        """
        A string representation.

        :arg args: If provided, these will be the return value from
            :meth:`__call__`.
        """
        return "{} diagnostic".format(self.name)

    def push_to_log(self, time, args):
        """
        Push callback status message to log

        :arg time: time stamp of entry
        :arg args: the return value from :meth:`__call__`.
        """
        print_output(self.message_str(*args))

    def push_to_hdf5(self, time, args, index=None):
        """
        Append values to HDF5 file.

        :arg time: time stamp of entry
        :arg args: the return value from :meth:`__call__`.
        """
        if not self._hdf5_initialized:
            self._create_hdf5_file()
        self.hdf_exporter.export(args, time=time, index=index)

    def evaluate(self, index=None):
        """
        Evaluates callback and pushes values to log and hdf file (if enabled)
        """
        values = self.__call__()
        time = self.solver_obj.simulation_time
        if self.append_to_log:
            self.push_to_log(time, values)
        if self.append_to_hdf5:
            self.push_to_hdf5(time, values, index=index)


class ScalarConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a scalar quantity"""
    variable_names = ['integral', 'relative_difference']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates scalar conservation check callback object

        :arg scalar_callback: Python function that takes the solver object as
            an argument and returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        super(ScalarConservationCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.initial_value = None

    def __call__(self):
        value = self.scalar_callback()
        if self.initial_value is None:
            self.initial_value = value
        rel_diff = (value - self.initial_value)/self.initial_value
        return value, rel_diff

    def message_str(self, *args):
        line = '{0:s} rel. error {1:11.4e}'.format(self.name, args[1])
        return line


class VolumeConservation3DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    name = 'volume3d'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        def vol3d():
            return comp_volume_3d(self.solver_obj.mesh)
        super(VolumeConservation3DCallback, self).__init__(vol3d, solver_obj, **kwargs)


class VolumeConservation2DCallback(ScalarConservationCallback):
    """Checks conservation of 2D volume (integral of water elevation field)"""
    name = 'volume2d'

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """

        def vol2d():
            return comp_volume_2d(self.solver_obj.fields.elev_2d,
                                  self.solver_obj.fields.bathymetry_2d)
        super(VolumeConservation2DCallback, self).__init__(vol2d, solver_obj, **kwargs)


class TracerMassConservation2DCallback(ScalarConservationCallback):
    """
    Checks conservation of depth-averaged tracer mass

    Depth-averaged tracer mass is defined as the integral of 2D tracer
    multiplied by total depth.
    """
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            return comp_tracer_mass_2d(self.solver_obj.fields.elev_2d,
                                       self.solver_obj.fields.bathymetry_2d,
                                       self.solver_obj.fields[tracer_name])
        super(TracerMassConservation2DCallback, self).__init__(mass, solver_obj, **kwargs)


class TracerMassConservationCallback(ScalarConservationCallback):
    """Checks conservation of total tracer mass"""
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            return comp_tracer_mass_3d(self.solver_obj.fields[tracer_name])
        super(TracerMassConservationCallback, self).__init__(mass, solver_obj, **kwargs)


class MinMaxConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a minimum/maximum"""
    variable_names = ['min_value', 'max_value', 'undershoot', 'overshoot']

    def __init__(self, minmax_callback, solver_obj, **kwargs):
        """
        :arg minmax_callback: Python function that takes the solver object as
            an argument and returns a (min, max) value tuple
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        super(MinMaxConservationCallback, self).__init__(solver_obj, **kwargs)
        self.minmax_callback = minmax_callback
        self.initial_value = None

    def __call__(self):
        value = self.minmax_callback()
        if self.initial_value is None:
            self.initial_value = value
        overshoot = max(value[1] - self.initial_value[1], 0.0)
        undershoot = min(value[0] - self.initial_value[0], 0.0)
        return value[0], value[1], undershoot, overshoot

    def message_str(self, *args):
        l = '{0:s} {1:g} {2:g}'.format(self.name, args[2], args[3])
        return l


class TracerOvershootCallBack(MinMaxConservationCallback):
    """Checks overshoots of the given tracer field."""
    name = 'tracer overshoot'

    def __init__(self, tracer_name, solver_obj, **kwargs):
        """
        :arg tracer_name: Name of the tracer. Use canonical field names as in
            :class:`.FieldDict`.
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        self.name = tracer_name + ' overshoot'

        def minmax():
            tracer_min = self.solver_obj.fields[tracer_name].dat.data.min()
            tracer_max = self.solver_obj.fields[tracer_name].dat.data.max()
            tracer_min = self.solver_obj.comm.allreduce(tracer_min, op=MPI.MIN)
            tracer_max = self.solver_obj.comm.allreduce(tracer_max, op=MPI.MAX)
            return tracer_min, tracer_max
        super(TracerOvershootCallBack, self).__init__(minmax, solver_obj, **kwargs)


class DetectorsCallback(DiagnosticCallback):
    """
    Callback that evaluates the specified fields at the specified locations
    """
    def __init__(self, solver_obj,
                 detector_locations,
                 field_names,
                 name,
                 detector_names=None,
                 **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg detector_locations: List of x, y locations in which fields are to
            be interpolated.
        :arg field_names: List of fields to be interpolated.
        :arg name: Unique name for this callback and its associated set of
            detectors. This determines the name of the output h5 file
            (prefixed with `diagnostic_`).
        :arg detector_names: List of names for each of the detectors. If not
            provided automatic names of the form `detectorNNN` are created
            where NNN is the (0-padded) detector number.
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        # printing all detector output to log is probably not a useful default:
        kwargs.setdefault('append_to_log', False)
        self.field_dims = [solver_obj.fields[field_name].function_space().value_size
                           for field_name in field_names]
        attrs = {
            # use null-padded ascii strings, dtype='U' not supported in hdf5, see http://docs.h5py.org/en/latest/strings.html
            'field_names': np.array(field_names, dtype='S'),
            'field_dims': self.field_dims,
        }
        super().__init__(solver_obj, array_dim=sum(self.field_dims), attrs=attrs, **kwargs)

        ndetectors = len(detector_locations)
        if detector_names is None:
            fill = len(str(ndetectors))
            self.detector_names = ['detector{:0{fill}d}'.format(i, fill=fill) for i in range(ndetectors)]
        else:
            assert ndetectors == len(detector_names), "Different number of detector locations and names"
            self.detector_names = detector_names
        self._variable_names = self.detector_names
        self.detector_locations = detector_locations
        self.field_names = field_names
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def variable_names(self):
        return self.detector_names

    def _values_per_field(self, values):
        """
        Given all values evaulated in a detector location, return the values per field"""
        i = 0
        result = []
        for dim in self.field_dims:
            result.append(values[i:i+dim])
            i += dim
        return result

    def message_str(self, *args):
        return '\n'.join(
            'In {}: '.format(name) + ', '.join(
                '{}={}'.format(field_name, field_val) for field_name, field_val in zip(self.field_names, self._values_per_field(values)))
            for name, values in zip(self.detector_names, args))

    def _evaluate_field(self, field_name):
        return self.solver_obj.fields[field_name](self.detector_locations)

    def __call__(self):
        """
        Evaluate all current fields in all detector locations

        Returns a ndetectors x ndims array, where ndims is the sum of the
        dimension of the fields.
        """
        ndetectors = len(self.detector_locations)
        field_vals = []
        for field_name in self.field_names:
            field_vals.append(np.reshape(self._evaluate_field(field_name), (ndetectors, -1)))

        return np.hstack(field_vals)


class AccumulatorCallback(DiagnosticCallback):
    """Callback that evaluates a (scalar) functional involving integrals in both time and space. This callback can also
    be used to assemble time dependent objective functionals for adjoint simulations. Time integration is achieved using
    the trapezium rule."""
    variable_names = ['spatial integral at current timestep']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        :arg scalar_callback: Python function that returns a list of values of an objective functional.
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        kwargs.setdefault('export_to_hdf5', False)
        kwargs.setdefault('append_to_log', False)
        super(AccumulatorCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback      # Evaluate functional
        self.dt = solver_obj.options.timestep
        self.integrant = 0.
        self.old_value = None

    def __call__(self):
        scalar_value = self.scalar_callback()
        if self.old_value is not None:
            self.integrant += 0.5 * (self.old_value + scalar_value) * self.dt
        self.old_value = scalar_value

        return scalar_value

    def get_val(self):
        return self.integrant

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[0])
        return line


class ErrorEstimateCallback(DiagnosticCallback):
    """Base class for callbacks that evaluate an error estimator"""
    variable_names = ['error estimate']

    def __init__(self, error_type, solver_obj, export_to_pvd=False, **kwargs):
        """
        Creates error estimate evaluation callback object

        :arg error_type: Python function that takes the solver object as
            an argument and returns an error estimator
        :arg solver_obj: Thetis solver object
        :arg kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        kwargs.setdefault('export_to_hdf5', False)
        self.export_to_hdf5 = kwargs.get('export_to_hdf5')
        super(ErrorEstimateCallback, self).__init__(solver_obj, **kwargs)
        self.estimator = error_type
        self.directory = solver_obj.options.output_directory
        self.ix = 0
        self.export_to_pvd = export_to_pvd
        if self.export_to_pvd:
            self.plot_file = File(self.directory + 'ErrorEstimator2d.pvd')

    def __call__(self):
        f = self.estimator()

        if self.export_to_hdf5:
            index = (5 - len(str(self.ix))) * '0' + str(self.ix)
            with DumbCheckpoint(self.directory + 'hdf5/' + self.label + index, mode=FILE_CREATE) as dc:
                if isinstance(f, FiredrakeFunction):
                    dc.store(f)
                    if self.export_to_pvd:
                        self.plot_file.write(f, time=float(self.ix))
                else:
                    dc.store(f[0])
                    dc.store(f[1])
                    if self.export_to_pvd:
                        self.plot_file.write(f[0], f[1], time=float(self.ix))
                dc.close()
        self.ix += 1

        if isinstance(f, FiredrakeFunction):        # TODO: Why can't we just have one output?
            return norm(f), 0                       # TODO: Not really the right thing to output for ExplicitErrorCallback
        else:
            return sqrt(assemble(sum(inner(fi, fi) for fi in f) * dx)), 0.

    def message_str(self, *args):
        line = '{0:s} error estimate {1:11.4e}'.format(self.name, args[0])
        return line


class CellResidualCallback(ErrorEstimateCallback):
    """Callback which evaluates strong residual on element cells."""
    name = 'cell residual'
    label = 'CellResidual2d_'

    def __init__(self, solver_obj, **kwargs):
        """
        Creates error estimator corresponding to the strong residual on element cells.

        :param solver_obj: Thetis solver object.
        :param kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        def cell_residual():
            if solver_obj.options.tracer_only:
                R = Function(solver_obj.fields.tracer_2d.function_space(), name="residual")
                tracer_integrator = solver_obj.timestepper.tracer_integrator(solver_obj.eq_tracer,
                                                                             solver_obj.fields.tracer_2d,
                                                                             solver_obj.fields,
                                                                             solver_obj.options.timestep)
                res = tracer_integrator.cell_residual()
                R.interpolate(res)
                return R
            else:
                R = Function(solver_obj.fields.solution_2d.function_space(), name="residual")
                res_u, res_e = solver_obj.timestepper.cell_residual()
                Ru, Re = R.split()
                Ru.rename("momentum residual")
                Re.rename("continuity residual")
                Ru.interpolate(res_u)
                Re.interpolate(res_e)
                return Ru, Re

        super(CellResidualCallback, self).__init__(cell_residual, solver_obj, **kwargs)


class EdgeResidualCallback(ErrorEstimateCallback):
    """Callback which evaluates strong residual across element edges."""
    name = 'edge residual'
    label = 'EdgeResidual2d_'

    def __init__(self, solver_obj, **kwargs):
        """
        Creates error estimator corresponding to the strong residual across element edges.

        :param solver_obj: Thetis solver object.
        :param kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        def edge_residual():
            if solver_obj.options.tracer_only:
                R = Function(solver_obj.fields.tracer_2d.function_space(), name="residual")
                tracer_integrator = solver_obj.timestepper.tracer_integrator(solver_obj.eq_tracer,
                                                                             solver_obj.fields.tracer_2d,
                                                                             solver_obj.fields,
                                                                             solver_obj.options.timestep)
                res = tracer_integrator.edge_residual()
                R.interpolate(res)
                return R
            else:
                res_u1, res_u2, res_e = solver_obj.timestepper.edge_residual()
                fs = solver_obj.fields.solution_2d.function_space()
                mesh = solver_obj.mesh2d
                els = fs.ufl_element().sub_elements()
                V1 = FunctionSpace(mesh, els[0].family(), els[0].degree())
                V2 = FunctionSpace(mesh, els[1].family(), els[1].degree())
                R = Function(V1 * V1 * V2)
                Ru1, Ru2, Re = R.split()
                Ru1.rename("momentum residual 1")
                Ru2.rename("momentum residual 2")
                Re.rename("continuity residual")

                Ru1.interpolate(res_u1)
                Ru2.interpolate(res_u2)
                Re.interpolate(res_e)
                return Ru1, Ru2, Re

        super(EdgeResidualCallback, self).__init__(edge_residual, solver_obj, **kwargs)


class ExplicitErrorCallback(ErrorEstimateCallback):
    r"""
    Estimate error using an a posteriori error indicator [Ainsworth and Oden, 1997], given by

    .. math::
        \|\textbf{R}(\textbf{q}_h)\|_{\mathcal{L}_2(K)}
            + h_K^{-1}\|\textbf{r}(\textbf{q}_h)\|_{\mathcal{L}_2(\partial K)},

    where
    :math:`\textbf{q}_h` is the approximation to the PDE solution,
    :math:`\textbf{R}` denotes the strong residual on cells (element interiors),
    :math:`\textbf{r}` denotes the strong residual on edges,
    :math:`h_K` is the size of mesh element `K`.
    """
    name = 'explicit'
    label = 'ExplicitError2d_'

    def __init__(self, solver_obj, **kwargs):
        """
        Creates error estimator corresponding to the strong residual across edges.

        :param solver_obj: Thetis solver object.
        :param kwargs: any additional keyword arguments, see
            :class:`.DiagnosticCallback`.
        """
        def explicit_error():
            # Create P0 spaces and an associated TestFunction `v`, scaled to take value 1 in each cell. Suppose we have
            # an error estimator `e`. Then this ensures `assemble(assemble(p0_test * e * dx) * dx) = assemble(e * dx)`
            # (for piecewise constant and piecewise linear estimators `e`).
            mesh = solver_obj.mesh2d
            P0 = FunctionSpace(mesh, "DG", 0)
            v = Constant(mesh.num_cells()) * TestFunction(P0)
            ee = Function(P0, name="explicit error")
            h = CellSize(mesh)

            # Take a weighted sum of residual contributions from element cells and element edges
            if solver_obj.options.tracer_only:
                res = solver_obj.timestepper.tracer_integrator.cell_residual()
                bres = solver_obj.timestepper.tracer_integrator.edge_residual()
            else:
                res_u, res_e = solver_obj.timestepper.cell_residual()
                res = inner(res_u, res_u) + res_e * res_e
                bres_u1, bres_u2, bres_e = solver_obj.timestepper.edge_residual()
                bres = bres_u1 * bres_u1 + bres_u2 * bres_u2 + bres_e * bres_e
            print("cell residual norm = {i:.4e}".format(i=norm(res)))
            print("edge residual norm = {b:.4e}".format(b=norm(bres)))
            ee.interpolate(assemble(v * sqrt(assemble(v * (res + bres / h) * dx)) * dx))

            return ee

        super(ExplicitErrorCallback, self).__init__(explicit_error, solver_obj, **kwargs)
