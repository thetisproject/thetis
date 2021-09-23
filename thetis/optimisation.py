"""
Some classes to help optimisation problems formulated with thetis_adjoint.

In particular this module contains some OptimisationCallbacks that can be used
as callbacks of a :class:`ReducedFunctional` called at various stages during the optimisation
process:
- eval_cb_pre(controls) and eval_cb_post(functional, controls)                    called before and after (re)evaluation of the forward model
- derivative_cb_pre(controls) and eval_cb_post(functional, derivative, controls)  called before and after the gradient computation using the adjoint of the model
- hessian_cb_pre(controls) and eval_cb_post(functional, derivative, controls)     called before and after the hessian computation
OptimisationCallbacks that (can) use controls, functional and derivative information, work out
what is provided by the number of arguments: current control values are always in the last argument;
if more than 2 arguments are provided, the first is the latest evaluated functional value.
"""
from firedrake import *
from .callback import DiagnosticCallback
from .exporter import ExportManager
import thetis.field_defs as field_defs
from abc import abstractmethod
import numpy


class UserExportManager(ExportManager):
    """
    ExportManager for user provided functions (not necessarily known to Thetis)

    In the standard :class:`.ExportManager` all provided functions need to have standard names
    present in :py:data:`.field_metadata`. Here, any functions can be provided. If function.name() is in
    :py:data:`.field_metadata`, the standard filename and shortname  are used.
    If the function.name() is unknown, both are based on function.name()
    directly (with an optional additional filename_prefix). Filenames and
    shortnames can be overruled by the shortnames and filenames arguments."""
    def __init__(self, solver_obj_or_outputdir, functions_to_export,
                 filenames=None, filename_prefix='',
                 shortnames=None, **kwargs):
        """
        :arg solver_obj_or_outputdir: a :class:`.FlowSolver2d` object, used to determine the output directory. Alternatively, the
              outputdir can be specified with a string as the first argument.
        :arg functions_to_export: a list of :class:`Function` s
        :arg filenames: a list of strings that specify the filename for each provided function. If not provided,
              filenames are based on function.name().
        :arg filename_prefix: a string prefixed to each filename
        :arg shortnames: a list of strings with the shortnames used for each provided function. If not provided,
              shortnames are based on function.name().
        :arg kwargs: any further keyword arguments are passed on to :class:`.ExportManager`"""
        try:
            outputdir = solver_obj_or_outputdir.options.output_directory
        except AttributeError:
            outputdir = solver_obj_or_outputdir

        if shortnames is None:
            field_name_list = [function.name() for function in functions_to_export]
        else:
            field_name_list = shortnames

        field_dict = {}
        field_metadata = {}
        for field_name, function in zip(field_name_list, functions_to_export):
            field_dict[field_name] = function
            if shortnames is None and field_name in field_defs.field_metadata:
                field_metadata[field_name] = {'shortname': field_defs.field_metadata[field_name]}
            else:
                field_metadata[field_name] = {'shortname': field_name}

        if filenames is None:
            for field_name in field_name_list:
                if field_name in field_defs.field_metadata:
                    field_metadata[field_name]['filename'] = field_defs.field_metadata['filename']
                else:
                    field_metadata[field_name]['filename'] = filename_prefix + field_name
        else:
            for field_name, filename in zipt(field_name_list, filenames):
                field_metadata[field_name]['filename'] = filename

        super().__init__(outputdir, field_name_list, field_dict, field_metadata, **kwargs)


class DeferredExportManager(object):
    """
    A wrapper around a UserExportManager that is only created on the first export() call.

    In addition the functions provided in the export call are copied into a fixed set of functions,
    where the functions provided in subsequent calls may be different (they need to be in the same
    function space). This is used in the :class:`.ControlsExportOptimisationCallback`
    and :class:`.DerivativesExportOptimisationCallback`."""
    def __init__(self, solver_obj_or_outputdir, **kwargs):
        """
        :arg solver_obj_or_outputdir: a :class:`.FlowSolver2d` object, used to determine the output directory. Alternatively, the
              outputdir can be specified with a string as the first argument.
        :arg kwargs: any further keyword arguments are passed on to :class:`.UserExportManager`"""
        self.solver_obj_or_outputdir = solver_obj_or_outputdir
        self.kwargs = kwargs
        self.export_manager = None

    def export(self, functions, suggested_names=None):
        """
        Create the :class:`.UserExportManager` (first call only), and call its export() method.

        :arg functions: a list of :class:`Function` s that the :class:`.UserExportManager` will be based on. Their values
              are first copied. The list may contain different functions in subsequent calls,
              but their function space should remain the same.
        """
        try:
            len(functions)
        except (TypeError, NotImplementedError):
            functions = [functions]

        if self.export_manager is None:
            if suggested_names is None:
                self.functions = [Function(function.function_space(), name=function.name()) for function in functions]
            else:
                self.functions = [Function(function.function_space(), name=name) for function, name in zip(functions, suggested_names)]
            self.export_manager = UserExportManager(self.solver_obj_or_outputdir, self.functions, **self.kwargs)
        for function, function_arg in zip(self.functions, functions):
            assert function.function_space() is function_arg.function_space()
            function.assign(function_arg)
        self.export_manager.export()


class UserExportOptimisationCallback(UserExportManager):
    """A :class:`.UserExportManager` that can be used as a :class:`ReducedFunctional` callback

    Any callback arguments (functional value, derivatives, controls) are ignored"""
    def __init__(self, solver_obj_or_outputdir, functions_to_export, **kwargs):
        """
        :arg solver_obj_or_outputdir: a :class:`.FlowSolver2d` object, used to determine the output directory. Alternatively, the
              outputdir can be specified with a string as the first argument.
        :arg functions_to_export: a list of :class:`Function` s
        :arg kwargs: any further keyword arguments are passed on to :class:`.UserExportManager`"""
        kwargs.setdefault('filename_prefix', 'optimisation_')  # use prefix to avoid overwriting forward model output
        super().__init__(solver_obj_or_outputdir, functions_to_export, **kwargs)
        # we need to maintain the original functions in the dict as it
        # is their block_variables (representing the current "end"-state)
        # that determine what will be written
        self.orig_functions = self.functions.copy()

    def __call__(self, *args):
        """
        Ensure the :class:`.UserExportManager` uses the checkpointed values and call its export().

        :args: these are ignored"""
        for name in self.fields_to_export:
            self.functions[name] = self.orig_functions[name].block_variable.saved_output
        self.export()


class ControlsExportOptimisationCallback(DeferredExportManager):
    """A callback that exports the current control values (assumed to all be :class:`Function` s)

    The control values are assumed to be the last argument in the callback (as for all :class:`ReducedFunctional` callbacks)."""
    def __init__(self, solver_obj_or_outputdir, **kwargs):
        """
        :arg solver_obj_or_outputdir: a :class:`.FlowSolver2d` object, used to determine the output directory. Alternatively, the
              outputdir can be specified with a string as the first argument.
        :arg kwargs: any further keyword arguments are passed on to :class:`.UserExportManager`"""
        kwargs.setdefault('filename_prefix', 'control_')
        super().__init__(solver_obj_or_outputdir, **kwargs)

    def __call__(self, *args):
        self.export(args[-1])


class DerivativesExportOptimisationCallback(DeferredExportManager):
    """A callback that exports the derivatives calculated by the adjoint.

    The derivatives are assumed to be the second argument in the callback. This can therefore
    be used as a derivative_cb_post callback in a :class:`ReducedFunctional`"""
    def __init__(self, solver_obj_or_outputdir, **kwargs):
        """
        :arg solver_obj_or_outputdir: a :class:`.FlowSolver2d` object, used to determine the output directory. Alternatively, the
              outputdir can be specified with a string as the first argument.
        :arg kwargs: any further keyword arguments are passed on to :class:`.UserExportManager`"""
        kwargs.setdefault('filename_prefix', 'derivative_')
        super().__init__(solver_obj_or_outputdir, **kwargs)

    def __call__(self, *args):
        if len(args) != 3:
            raise TypeError("DerivativesExportOptimsationCallback called with wrong number of arguments: should be used for derivative_cb_post callback only.")
        try:
            # get name from controls args[-1]
            names = [function.name() for function in args[-1]]
        except (TypeError, NotImplementedError):
            # args[-1] is not a list but a single control
            names = [args[-1].name()]
        self.export(args[1], suggested_names=names)


class OptimisationCallbackList(list):
    """
    A list of callbacks that can be used as a single callback itself.

    Calls all callbacks in order."""
    def __call__(self, *args):
        for callback in self:
            callback(*args)


class DiagnosticOptimisationCallback(DiagnosticCallback):
    """
    An OptimsationCallback similar to :class:`.DiagnosticCallback` that can be used as callback in a :class:`ReducedFunctional`.

    Note that in this case the computing of the values needs to be defined in the compute_values method,
    not in the __call__ method (as this one is directly called from the :class:`ReducedFunctional`). In addition,
    like any :class:`.DiagnosticCallback`, the name and variable_names properties and a message_str method need to be defined.
    """

    def __init__(self, solver_obj, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg kwargs: keyword arguments passed to :class:`.DiagnosticCallback`
        """
        kwargs.setdefault('include_time', False)
        super().__init__(solver_obj, **kwargs)

    @abstractmethod
    def compute_values(self, *args):
        """
        Compute diagnostic values.

        This method is to be implemented in concrete subclasses of a :class:`.DiagnosticOptimisationCallback`.
        The number of arguments varies depending on which of the 6 [eval|derivative|hessian]_cb_[pre|post] callbacks
        this is used as. The last argument always contains the current controls. In the "pre" callbacks this is
        the only argument. In all "post" callbacks the 0th argument is the current functional value. eval_cb_post
        is given two arguments: functional and controls. derivative_cb_post and hessian_cb_post are given three
        arguments with args[1] being the derivative/hessian just calculated."""
        pass

    def evaluate(self, *args, index=None):
        """Evaluates callback and pushes values to log and hdf file (if enabled)"""
        values = self.compute_values(*args)
        if len(args) > 0:
            functional = args[0]
        else:
            functional = numpy.nan

        if self.append_to_log:
            self.push_to_log(functional, values)
        if self.append_to_hdf5:
            self.push_to_hdf5(functional, values)

    def __call__(self, *args):
        self.evaluate(*args)


class FunctionalOptimisationCallback(DiagnosticOptimisationCallback):
    """
    A simple OptimisationCallback that records the functional value in the log and/or hdf5 file."""
    variable_names = ['functional']
    name = 'functional'

    def compute_values(self, *args):
        if len(args) == 0:
            raise TypeError('FunctionalOptimisationCallback can be used as _post callback only.')
        return [args[0]]

    def message_str(self, functional):
        return 'Functional value: {}'.format(functional)
