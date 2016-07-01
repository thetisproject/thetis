"""
Callback functions used to compute metrics at runtime.

Tuomas Karna 2016-01-28
"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty, abstractmethod


class DiagnosticCallback(object):
    """A base class for all Callback classes"""
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @property
    def name(self):
        "The name of the diagnostic"
        return type(self).__name__

    @abstractmethod
    def __call__(self, solver_obj):
        """Evaluate the diagnostic value.

        :arg solver_obj: The solver object.
        """
        pass

    @abstractmethod
    def __str__(self, *args):
        """A string representation.

        :arg args: If provided, these will be the return value from :meth:`__call__`.
        """
        return "{} diagnostic".format(self.name)

    def log(self, *args):
        """Print status message to log"""
        print_info(self.__str__(*args))


# class ValidationCallback(object):
#     """
#     Base class of all objects used to check validity of simulation
#     at each export.
#     """
#     def __init__(self):
#         self._initialized = False
#
#     def initialize(self, comm=COMM_WORLD):
#         """Initializes the callback function.
#
#         :kwarg comm: The communicator for the callback."""
#         self._initialized = True
#         self.comm = comm
#
#     def update(self):
#         """Updates the validation metric for current state of the model"""
#         assert self._initialized, 'ValidationCallback object not initialized'
#
#     def report(self):
#         """Prints results on stdout"""
#         raise NotImplementedError('this function must be implemented in derived class')


class ScalarConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a scalar quantity"""
    def __init__(self, name, scalar_callback):
        """
        Creates scalar conservation check callback object

        name : str
            human readable name of the quantity
        scalar_callback : function
            function that takes the FlowSolver object as an argument and
            returns the scalar quantity of interest
        """
        super(ScalarConservationCallback, self).__init__()
        self.varname = name
        self.scalar_callback = scalar_callback
        self.initial_value = None
        # self.scalar_callback(solver_object)

    def __call__(self, solver_obj):
        value = self.scalar_callback(solver_obj)
        if self.initial_value is None:
            self.initial_value = value
        rel_diff = (value - self.initial_value)/self.initial_value
        return value, rel_diff

    def __str__(self, args):
        line = '{0:s} rel. error {1:11.4e}'.format(self.varname, args[1])
        return line

#    def initialize(self, solver_object):
#        comm = solver_object.comm
#        print_info('Initial {0:s} {1:f}'.format(self.name, self.initial_value),
#                   comm=comm)
#        super(ScalarConservationCallback, self).initialize(comm)
#
#    def update(self, solver_object):
#        super(ScalarConservationCallback, self).update()
#        self.value = self.scalar_callback(solver_object)
#        return self.value
#
#    def report(self):
#        if self.comm.rank == 0:
#            line = '{0:s} rel. error {1:11.4e}'
#            print_info(line.format(self.name, (self.initial_value - self.value)/self.initial_value), comm=self.comm)
#            sys.stdout.flush()


class VolumeConservation3DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    def __init__(self):
        def vol3d(solver_object):
            return comp_volume_3d(solver_object.mesh)
        super(VolumeConservation3DCallback, self).__init__('volume 3D', vol3d)


class VolumeConservation2DCallback(ScalarConservationCallback):
    """Checks conservation of 3D volume (volume of 3D mesh)"""
    def __init__(self):
        def vol2d(solver_object):
            return comp_volume_2d(solver_object.fields.elev_2d,
                                  solver_object.fields.bathymetry_2d)
        super(VolumeConservation2DCallback, self).__init__('volume 2D', vol2d)


class TracerMassConservationCallback(ScalarConservationCallback):
    """Checks conservation of total tracer mass"""
    def __init__(self, tracer_name):
        def mass(solver_object):
            return comp_tracer_mass_3d(solver_object.fields[tracer_name])
        name = '{:} mass'.format(tracer_name)
        super(TracerMassConservationCallback, self).__init__(name, mass)


class MinMaxConservationCallback(DiagnosticCallback):
    """Base class for callbacks that check conservation of a minimum/maximum"""
    def __init__(self, name, minmax_callback):
        """
        Creates scalar conservation check callback object

        name : str
            human readable name of the quantity
        minmax_callback : function
            function that takes the FlowSolver object as an argument and
            returns the minimum and maximum values as a tuple
        """
        super(MinMaxConservationCallback, self).__init__()
        self.varname = name
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
        line = '{0:s} overshoots {1:g} {2:g}'.format(self.varname, args[2], args[3])
        return line

#     def initialize(self, solver_object):
#         self.initial_value = self.minmax_callback(solver_object)
#         comm = solver_object.comm
#         print_info('Initial {0:s} value range {1:f} - {2:f}'.format(self.name, *self.initial_value), comm=comm)
#         super(MinMaxConservationCallback, self).initialize(comm)
#
#     def update(self, solver_object):
#         super(MinMaxConservationCallback, self).update()
#         self.value = self.minmax_callback(solver_object)
#         return self.value
#
#     def report(self):
#         if self.comm.rank == 0:
#             overshoot = max(self.value[1] - self.initial_value[1], 0.0)
#             undershoot = min(self.value[0] - self.initial_value[0], 0.0)
#             print_info('{0:s} overshoots {1:g} {2:g}'.format(self.name, undershoot, overshoot), comm=self.comm)
#             sys.stdout.flush()


class TracerOvershootCallBack(MinMaxConservationCallback):
    """Checks overshoots of the given tracer field."""
    def __init__(self, tracer_name):
        def minmax(solver_object):
            tracer_min = solver_object.fields[tracer_name].dat.data.min()
            tracer_max = solver_object.fields[tracer_name].dat.data.max()
            comm = solver_object.comm
            tracer_min = comm.allreduce(tracer_min, op=MPI.MIN)
            tracer_max = comm.allreduce(tracer_max, op=MPI.MAX)
            return tracer_min, tracer_max
        super(TracerOvershootCallBack, self).__init__(tracer_name, minmax)
