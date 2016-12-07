"""
Defines custom callback functions used to compute various metrics at runtime.

"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty, abstractmethod
import h5py
from collections import defaultdict
from .log import *


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

    Evaluate callbacks, calls :func:`evaluate` method of all callbacks registered
    in the given mode.

    .. code-block:: python

        cm.evaluate('export')

    """
    def __init__(self):
        super(CallbackManager, self).__init__(OrderedDict)

    def add(self, callback, mode):
        """
        Add a callback under the given mode

        :param callback: a :class:`.DiagnosticCallback` object
        :param mode: register callback under this mode
        :type mode: string
        """
        key = callback.name
        self[mode][key] = callback

    def evaluate(self, mode):
        """
        Evaluate all callbacks registered under the given mode
        """
        for key in self[mode]:
            self[mode][key].evaluate()


class DiagnosticHDF5(object):
    """
    A HDF5 file for storing diagnostic time series arrays.
    """
    def __init__(self, filename, varnames, comm=COMM_WORLD):
        """
        :param filename: Full filename of the HDF5 file.
        :type filename: string
        :param varnames: List of variable names that the diagnostic callback
            provides
        :param comm: MPI communicator
        """
        self.comm = comm
        self.filename = filename
        self.varnames = varnames
        self.nvars = len(varnames)
        if comm.rank == 0:
            # create empty file with correct datasets
            with h5py.File(filename, 'w') as hdf5file:
                hdf5file.create_dataset('time', (0, 1),
                                        maxshape=(None, 1))
                for var in self.varnames:
                    hdf5file.create_dataset(var, (0, 1),
                                            maxshape=(None, 1))

    def _expand(self, hdf5file):
        """Expands data arrays by 1 entry"""
        # TODO is there an easier way for doing this?
        for var in self.varnames + ['time']:
            arr = hdf5file[var]
            shape = arr.shape
            arr.resize((shape[0] + 1, shape[1]))

    def export(self, time, variables):
        """
        Appends a new entry of (time, variables) to the file.

        The HDF5 is updated immediately.

        :param time: time stamp of entry
        :type time: float
        :param variables: values of entry
        :type variables: tuple of float
        """
        if self.comm.rank == 0:
            with h5py.File(self.filename, 'a') as hdf5file:
                self._expand(hdf5file)
                ix = hdf5file['time'].shape[0] - 1
                hdf5file['time'][ix] = time
                for i in range(self.nvars):
                    hdf5file[self.varnames[i]][ix] = variables[i]
                hdf5file.close()


class DiagnosticCallback(object):
    """
    A base class for all Callback classes
    """
    __metaclass__ = ABCMeta

    def __init__(self, solver_obj, outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        """
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        self.solver_obj = solver_obj
        if outputdir is None:
            self.outputdir = self.solver_obj.options.outputdir
        else:
            self.outputdir = outputdir
        self.append_to_hdf5 = export_to_hdf5
        self.append_to_log = append_to_log
        self._hdf5_initialized = False

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
                                               comm=comm)
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

        :param args: If provided, these will be the return value from :meth:`__call__`.
        """
        return "{} diagnostic".format(self.name)

    def push_to_log(self, time, args):
        """
        Push callback status message to log

        :param time: time stamp of entry
        :param args: the return value from :meth:`__call__`.
        """
        print_output(self.message_str(*args))

    def push_to_hdf5(self, time, args):
        """
        Append values to HDF5 file.

        :param time: time stamp of entry
        :param args: the return value from :meth:`__call__`.
        """
        if not self._hdf5_initialized:
            self._create_hdf5_file()
        self.hdf_exporter.export(time, args)

    def evaluate(self):
        """Evaluates callback and pushes values to log and hdf file (if enabled)"""
        values = self.__call__()
        time = self.solver_obj.simulation_time
        if self.append_to_log:
            self.push_to_log(time, values)
        if self.append_to_hdf5:
            self.push_to_hdf5(time, values)


class ScalarConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a scalar quantity"""
    variable_names = ['integral', 'relative_difference']

    def __init__(self, scalar_callback, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        Creates scalar conservation check callback object

        :param scalar_callback: Python function that takes the solver object as an argument and
            returns a scalar quantity of interest
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        super(ScalarConservationCallback, self).__init__(solver_obj,
                                                         outputdir,
                                                         export_to_hdf5,
                                                         append_to_log)
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

    def __init__(self, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        def vol3d():
            return comp_volume_3d(self.solver_obj.mesh)
        super(VolumeConservation3DCallback, self).__init__(vol3d,
                                                           solver_obj,
                                                           outputdir,
                                                           export_to_hdf5,
                                                           append_to_log)


class VolumeConservation2DCallback(ScalarConservationCallback):
    """Checks conservation of 2D volume (integral of water elevation field)"""
    name = 'volume2d'

    def __init__(self, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        def vol2d():
            return comp_volume_2d(self.solver_obj.fields.elev_2d,
                                  self.solver_obj.fields.bathymetry_2d)
        super(VolumeConservation2DCallback, self).__init__(vol2d,
                                                           solver_obj,
                                                           outputdir,
                                                           export_to_hdf5,
                                                           append_to_log)


class TracerMassConservationCallback(ScalarConservationCallback):
    """Checks conservation of total tracer mass"""
    name = 'tracer mass'

    def __init__(self, tracer_name, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        :param tracer_name: Name of the tracer. Use canonical field names as in :class:`.FieldDict`.
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        self.name = tracer_name + ' mass'  # override name for given tracer

        def mass():
            return comp_tracer_mass_3d(self.solver_obj.fields[tracer_name])
        super(TracerMassConservationCallback, self).__init__(mass,
                                                             solver_obj,
                                                             outputdir,
                                                             export_to_hdf5,
                                                             append_to_log)


class MinMaxConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a minimum/maximum"""
    variable_names = ['min_value', 'max_value', 'undershoot', 'overshoot']

    def __init__(self, minmax_callback, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        :param minmax_callback: Python function that takes the solver object as
            an argument and returns a (min, max) value tuple
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        super(MinMaxConservationCallback, self).__init__(solver_obj,
                                                         outputdir,
                                                         export_to_hdf5,
                                                         append_to_log)
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

    def __init__(self, tracer_name, solver_obj, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        :param tracer_name: Name of the tracer. Use canonical field names as in :class:`.FieldDict`.
        :param solver_obj: Thetis solver object
        :param outputdir: Custom directory where hdf5 files will be stored. By
            default solver's output directory is used.
        :type outputdir: string
        :param export_to_hdf5: If True, diagnostics will be stored in hdf5
            format
        :type export_to_hdf5: bool
        :param append_to_log: If True, callback output messages will be printed
            in log
        :type append_to_log: bool
        """
        self.name = tracer_name + ' overshoot'

        def minmax():
            tracer_min = self.solver_obj.fields[tracer_name].dat.data.min()
            tracer_max = self.solver_obj.fields[tracer_name].dat.data.max()
            tracer_min = self.solver_obj.comm.allreduce(tracer_min, op=MPI.MIN)
            tracer_max = self.solver_obj.comm.allreduce(tracer_max, op=MPI.MAX)
            return tracer_min, tracer_max
        super(TracerOvershootCallBack, self).__init__(minmax,
                                                      solver_obj,
                                                      outputdir,
                                                      export_to_hdf5,
                                                      append_to_log)
