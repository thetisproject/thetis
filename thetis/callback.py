"""
Callback functions used to compute metrics at runtime.

Tuomas Karna 2016-01-28
"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty, abstractmethod
import h5py
from collections import defaultdict


class CallbackManager(defaultdict):
    """
    Stores callbacks in different categories and provides methods for
    evaluating them.
    """
    def __init__(self):
        super(CallbackManager, self).__init__(OrderedDict)

    def add(self, callback, mode):
        key = callback.name
        self[mode][key] = callback

    def evaluate(self, solver_obj, mode):
        for key in self[mode]:
            self[mode][key].evaluate(solver_obj)


class DiagnosticHDF5(object):
    """Creates hdf5 files for diagnostic time series arrays."""
    def __init__(self, filename, varnames, comm=COMM_WORLD):
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
        Appends a new entry of (time, variables) to the file
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

    :arg outputdir: Custom directory where hdf5 files will be stored. By
        default solver's output directory is used.
    :arg export_to_hdf5: If True, diagnostics will be stored in hdf5 format
    :arg append_to_log: If True, diagnostic will be printed in log
    """
    __metaclass__ = ABCMeta

    def __init__(self, outputdir=None, export_to_hdf5=True,
                 append_to_log=True):
        self.append_to_hdf5 = export_to_hdf5
        self.append_to_log = append_to_log
        self.outputdir = outputdir
        self._hdf5_initialized = False

    def _create_hdf5_file(self, solver_obj):
        """
        Creates an empty hdf5 file with correct datasets.
        """
        if self.append_to_hdf5:
            comm = solver_obj.comm
            outputdir = self.outputdir
            if outputdir is None:
                outputdir = solver_obj.options.outputdir
            create_directory(outputdir)
            fname = 'diagnostic_{:}.hdf5'.format(self.name.replace(' ', '_'))
            fname = os.path.join(outputdir, fname)
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
    def __call__(self, solver_obj):
        """Evaluate the diagnostic value.

        :arg solver_obj: The solver object.

        NOTE: This method must implement all MPI reduction operations (if any).
        """
        pass

    @abstractmethod
    def __str__(self, args):
        """A string representation.

        :arg args: If provided, these will be the return value from :meth:`__call__`.
        """
        return "{} diagnostic".format(self.name)

    def push_to_log(self, time, args):
        """Print diagnostic status message to log"""
        # TODO update to use real logger object
        print_info(self.__str__(args))

    def push_to_hdf5(self, solver_obj, time, args):
        """Append values to export file."""
        if not self._hdf5_initialized:
            self._create_hdf5_file(solver_obj)
        self.hdf_exporter.export(time, args)

    def evaluate(self, solver_obj):
        """Evaluates callback and pushes values to log and hdf stream"""
        values = self.__call__(solver_obj)
        time = solver_obj.simulation_time
        if self.append_to_log:
            self.push_to_log(time, values)
        if self.append_to_hdf5:
            self.push_to_hdf5(solver_obj, time, values)


class ScalarConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a scalar quantity"""
    variable_names = ['integral', 'relative_difference']

    def __init__(self, scalar_callback, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        Creates scalar conservation check callback object

        name : str
            human readable name of the quantity
        scalar_callback : function
            function that takes the FlowSolver object as an argument and
            returns the scalar quantity of interest
        """
        super(ScalarConservationCallback, self).__init__(outputdir,
                                                         export_to_hdf5,
                                                         append_to_log)
        self.scalar_callback = scalar_callback
        self.initial_value = None

    def __call__(self, solver_obj):
        value = self.scalar_callback(solver_obj)
        if self.initial_value is None:
            self.initial_value = value
        rel_diff = (value - self.initial_value)/self.initial_value
        return value, rel_diff

    def __str__(self, args):
        line = '{0:s} rel. error {1:11.4e}'.format(self.name, args[1])
        return line


class VolumeConservation3DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    name = 'volume3d'

    def __init__(self, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        def vol3d(solver_object):
            return comp_volume_3d(solver_object.mesh)
        super(VolumeConservation3DCallback, self).__init__(vol3d,
                                                           outputdir,
                                                           export_to_hdf5,
                                                           append_to_log)


class VolumeConservation2DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    name = 'volume2d'

    def __init__(self, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        def vol2d(solver_object):
            return comp_volume_2d(solver_object.fields.elev_2d,
                                  solver_object.fields.bathymetry_2d)
        super(VolumeConservation2DCallback, self).__init__(vol2d,
                                                           outputdir,
                                                           export_to_hdf5,
                                                           append_to_log)


class TracerMassConservationCallback(ScalarConservationCallback):
    """Checks conservation of total tracer mass"""
    name = 'tracer mass'

    def __init__(self, tracer_name, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        # override name for given tracer
        self.name = tracer_name + ' mass'

        def mass(solver_object):
            return comp_tracer_mass_3d(solver_object.fields[tracer_name])
        super(TracerMassConservationCallback, self).__init__(mass,
                                                             outputdir,
                                                             export_to_hdf5,
                                                             append_to_log)


class MinMaxConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a minimum/maximum"""
    variable_names = ['min_value', 'max_value', 'undershoot', 'overshoot']

    def __init__(self, minmax_callback, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        """
        Creates scalar conservation check callback object

        :arg varname: human readable name of the quantity
        :arg minmax_callback: function that takes the FlowSolver object as
            an argument and returns the minimum and maximum values as a tuple
        """
        super(MinMaxConservationCallback, self).__init__(outputdir,
                                                         export_to_hdf5,
                                                         append_to_log)
        self.minmax_callback = minmax_callback
        self.initial_value = None

    def __call__(self, solver_obj):
        value = self.minmax_callback(solver_obj)
        if self.initial_value is None:
            self.initial_value = value
        overshoot = max(value[1] - self.initial_value[1], 0.0)
        undershoot = min(value[0] - self.initial_value[0], 0.0)
        return value[0], value[1], undershoot, overshoot

    def __str__(self, args):
        l = '{0:s} overshoots {1:g} {2:g}'.format(self.name, args[2], args[3])
        return l


class TracerOvershootCallBack(MinMaxConservationCallback):
    """Checks overshoots of the given tracer field."""
    name = 'tracer overshoot'

    def __init__(self, tracer_name, outputdir=None, export_to_hdf5=False,
                 append_to_log=True):
        self.name = tracer_name + ' overshoot'

        def minmax(solver_object):
            tracer_min = solver_object.fields[tracer_name].dat.data.min()
            tracer_max = solver_object.fields[tracer_name].dat.data.max()
            tracer_min = solver_object.comm.allreduce(tracer_min, op=MPI.MIN)
            tracer_max = solver_object.comm.allreduce(tracer_max, op=MPI.MAX)
            return tracer_min, tracer_max
        super(TracerOvershootCallBack, self).__init__(minmax,
                                                      outputdir,
                                                      export_to_hdf5,
                                                      append_to_log)
