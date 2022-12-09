import firedrake as fd
from firedrake_adjoint import *
import ufl
from .configuration import FrozenHasTraits
from .solver2d import FlowSolver2d
from .utility import create_directory, print_function_value_range, get_functionspace, unfrozen
from .log import print_output
from .diagnostics import HessianRecoverer2D
from .exporter import HDF5Exporter
import abc
import numpy
import h5py
from scipy.interpolate import interp1d
import time as time_mod
import os


class InversionManager(FrozenHasTraits):
    """
    Class for handling inversion problems and stashing
    the progress of the associated optimization routines.
    """

    @unfrozen
    def __init__(self, sta_manager, output_dir='outputs', no_exports=False, real=False,
                 penalty_parameters=[], cost_function_scaling=None,
                 test_consistency=True, test_gradient=True):
        """
        :arg sta_manager: the :class:`StationManager` instance
        :kwarg output_dir: model output directory
        :kwarg no_exports: if True, nothing will be written to disk
        :kwarg real: is the inversion in the Real space?
        :kwarg penalty_parameters: a list of penalty parameters to pass
            to the :class:`ControlRegularizationManager`
        :kwarg cost_function_scaling: global scaling for the cost function.
            As rule of thumb, it's good to scale the functional to J < 1.
        :kwarg test_consistency: toggle testing the correctness with
            which the :class:`ReducedFunctional` can recompute values
        :kwarg test_gradient: toggle testing the correctness with
            which the :class:`ReducedFunctional` can recompute gradients
        """
        assert isinstance(sta_manager, StationObservationManager)
        self.sta_manager = sta_manager
        self.reg_manager = None
        self.output_dir = output_dir
        self.no_exports = no_exports or real
        self.real = real
        self.penalty_parameters = penalty_parameters
        self.cost_function_scaling = cost_function_scaling or fd.Constant(1.0)
        self.sta_manager.cost_function_scaling.assign(cost_function_scaling)
        self.test_consistency = test_consistency
        self.test_gradient = test_gradient
        self.outfiles_m = []
        self.outfiles_dJdm = []
        self.control_exporters = []
        self.initialized = False

        self.J = 0  # cost function value (float)
        self.J_reg = 0  # regularization term value (float)
        self.J_misfit = 0  # misfit term value (float)
        self.dJdm_list = None  # cost function gradient (Function)
        self.m_list = None  # control (Function)
        self.Jhat = None
        self.m_progress = []
        self.J_progress = []
        self.J_reg_progress = []
        self.J_misfit_progress = []
        self.dJdm_progress = []
        self.i = 0
        self.tic = None
        self.nb_grad_evals = 0
        self.control_coeff_list = []
        self.control_list = []

    def initialize(self):
        if not self.no_exports:
            if self.real:
                raise ValueError("Exports are not supported in Real mode.")
            create_directory(self.output_dir)
            create_directory(self.output_dir + '/hdf5')
            for i in range(len(self.control_coeff_list)):
                self.outfiles_m.append(
                    fd.File(f'{self.output_dir}/control_progress_{i:02d}.pvd'))
                self.outfiles_dJdm.append(
                    fd.File(f'{self.output_dir}/gradient_progress_{i:02d}.pvd'))
        self.initialized = True

    def add_control(self, f):
        """
        Add a control field.

        Can be called multiple times in case of multiparameter optimization.

        :arg f: Function or Constant to be used as a control variable.
        """
        self.control_coeff_list.append(f)
        self.control_list.append(Control(f))
        if isinstance(f, fd.Function) and not self.no_exports:
            j = len(self.control_coeff_list) - 1
            prefix = f'control_{j:02d}'
            self.control_exporters.append(
                HDF5Exporter(f.function_space(), self.output_dir + '/hdf5', prefix)
            )

    def reset_counters(self):
        self.nb_grad_evals = 0

    def set_control_state(self, j, djdm_list, m_list):
        """
        Stores optimization state.

        To call whenever variables are updated.

        :arg j: error functional value
        :arg djdm_list: list of gradient functions
        :arg m_list: list of control coefficents
        """
        self.J = j
        self.dJdm_list = djdm_list
        self.m_list = m_list

        tape = get_working_tape()
        reg_blocks = tape.get_blocks(tag="reg_eval")
        self.J_reg = sum([b.get_outputs()[0].saved_output for b in reg_blocks])
        misfit_blocks = tape.get_blocks(tag="misfit_eval")
        self.J_misfit = sum([b.get_outputs()[0].saved_output for b in misfit_blocks])

    def start_clock(self):
        self.tic = time_mod.perf_counter()

    def stop_clock(self):
        toc = time_mod.perf_counter()
        return toc

    def set_initial_state(self, *state):
        self.set_control_state(*state)
        self.update_progress()

    def update_progress(self):
        """
        Updates optimization progress and stores variables to disk.

        To call after successful line searches.
        """
        toc = self.stop_clock()
        if self.i == 0:
            for f in self.control_coeff_list:
                print_function_value_range(f, prefix='Initial')

        elapsed = '-' if self.tic is None else f'{toc - self.tic:.1f} s'
        self.tic = toc

        if not self.initialized:
            self.initialize()

        # cost function and gradient norm output
        djdm = [fd.norm(f) for f in self.dJdm_list]
        if self.real:
            controls = [m.dat.data[0] for m in self.m_list]
            self.m_progress.append(controls)
        self.J_progress.append(self.J)
        self.J_reg_progress.append(self.J_reg)
        self.J_misfit_progress.append(self.J_misfit)
        self.dJdm_progress.append(djdm)
        comm = self.control_coeff_list[0].comm
        if comm.rank == 0 and not self.no_exports:
            if self.real:
                numpy.save(f'{self.output_dir}/m_progress', self.m_progress)
            numpy.save(f'{self.output_dir}/J_progress', self.J_progress)
            numpy.save(f'{self.output_dir}/J_reg_progress', self.J_reg_progress)
            numpy.save(f'{self.output_dir}/J_misfit_progress', self.J_misfit_progress)
            numpy.save(f'{self.output_dir}/dJdm_progress', self.dJdm_progress)
        if len(djdm) > 10:
            djdm = f"[{numpy.min(djdm):.4e} .. {numpy.max(djdm):.4e}]"
        else:
            djdm = "[" + ", ".join([f"{dj:.4e}" for dj in djdm]) + "]"
        print_output(f'line search {self.i:2d}: '
                     f'J={self.J:.3e}, dJdm={djdm}, '
                     f'grad_ev={self.nb_grad_evals}, duration {elapsed}')

        if not self.no_exports:
            # control output
            for j in range(len(self.control_coeff_list)):
                m = self.m_list[j]
                # vtk format
                o = self.outfiles_m[j]
                m.rename(self.control_coeff_list[j].name())
                o.write(m)
                # hdf5 format
                e = self.control_exporters[j]
                e.export(m)
            # gradient output
            for f, o in zip(self.dJdm_list, self.outfiles_dJdm):
                # store gradient in vtk format
                f.rename('Gradient')
                o.write(f)

        self.i += 1
        self.reset_counters()

    @property
    def rf_kwargs(self):
        """
        Default keyword arguments to pass to the
        :class:`ReducedFunctional` class.
        """
        def gradient_eval_cb(j, djdm, m):
            self.set_control_state(j, djdm, m)
            self.nb_grad_evals += 1

        params = {
            'derivative_cb_post': gradient_eval_cb,
        }
        return params

    def get_cost_function(self, solver_obj, weight_by_variance=False):
        r"""
        Get a sum of square errors cost function for the problem:

      ..math::
            J(u) = \sum_{i=1}^{n_{ts}} \sum_{j=1}^{n_{sta}} (u_j^{(i)} - u_{j,o}^{(i)})^2,

        where :math:`u_{j,o}^{(i)}` and :math:`u_j^{(i)}` denote the
        observed and computed values at timestep :math:`i`, and
        :math:`n_{ts}` and :math:`n_{sta}` are the numbers of timesteps
        and stations, respectively.

        Regularization terms are included if a
        :class:`RegularizationManager` instance is provided.

        :arg solver_obj: the :class:`FlowSolver2d` instance
        :kwarg weight_by_variance: should the observation data be
            weighted by the variance at each station?
        """
        assert isinstance(solver_obj, FlowSolver2d)
        if len(self.penalty_parameters) > 0:
            self.reg_manager = ControlRegularizationManager(
                self.control_coeff_list,
                self.penalty_parameters,
                self.cost_function_scaling,
                RSpaceRegularizationCalculator if self.real else HessianRegularizationCalculator)
        self.J_reg = 0
        self.J_misfit = 0
        if self.reg_manager is not None:
            self.J_reg = self.reg_manager.eval_cost_function()
        self.J = self.J_reg

        if weight_by_variance:
            var = fd.Function(self.sta_manager.fs_points_0d)
            for i, j in enumerate(self.sta_manager.local_station_index):
                var.dat.data[i] = numpy.var(self.sta_manager.observation_values[j])
            self.sta_manager.station_weight_0d.interpolate(1/var)

        def cost_fn(t):
            misfit = self.sta_manager.eval_cost_function(t)
            self.J_misfit += misfit
            self.J += misfit

        return cost_fn

    @property
    def reduced_functional(self):
        """
        Create a Pyadjoint :class:`ReducedFunctional` for the optimization.
        """
        if self.Jhat is None:
            self.Jhat = ReducedFunctional(self.J, self.control_list, **self.rf_kwargs)
        return self.Jhat

    def stop_annotating(self):
        """
        Stop recording operations for the adjoint solver.

        This method should be called after the :meth:`iterate`
        method of :class:`FlowSolver2d`.
        """
        assert self.reduced_functional is not None
        if self.test_consistency:
            self.consistency_test()
        if self.test_gradient:
            self.taylor_test()
        pause_annotation()

    def get_optimization_callback(self):
        """
        Get a callback for stashing optimization progress
        after successful line search.
        """

        def optimization_callback(m):
            self.update_progress()
            if not self.no_exports:
                self.sta_manager.dump_time_series()

        return optimization_callback

    def minimize(self, opt_method="BFGS", bounds=None, **opt_options):
        """
        Minimize the reduced functional using a given optimization routine.

        :kwarg opt_method: the optimization routine
        :kwarg bounds: a list of bounds to pass to the optimization routine
        :kwarg opt_options: other optimization parameters to pass
        """
        print_output(f'Running {opt_method} optimization')
        self.reset_counters()
        self.start_clock()
        J = float(self.reduced_functional(self.control_coeff_list))
        self.set_initial_state(J, self.reduced_functional.derivative(), self.control_coeff_list)
        if not self.no_exports:
            self.sta_manager.dump_time_series()
        return minimize(
            self.reduced_functional, method=opt_method, bounds=bounds,
            callback=self.get_optimization_callback(), options=opt_options)

    def consistency_test(self):
        """
        Test that :attr:`reduced_functional` can correctly recompute the
        objective value, assuming that none of the controls have changed
        since it was created.
        """
        print_output("Running consistency test")
        J = self.reduced_functional(self.control_coeff_list)
        if not numpy.isclose(J, self.J):
            raise ValueError(f"Consistency test failed (expected {self.J}, got {J})")
        print_output("Consistency test passed!")

    def taylor_test(self):
        """
        Run a Taylor test to check that the :attr:`reduced_functional` can
        correctly compute consistent gradients.

        Note that the Taylor test is applied on the current control values.
        """
        func_list = []
        for f in self.control_coeff_list:
            dc = f.copy(deepcopy=True)
            func_list.append(dc)
        minconv = taylor_test(self.reduced_functional, self.control_coeff_list, func_list)
        if minconv < 1.9:
            raise ValueError("Taylor test failed")  # NOTE: Pyadjoint already prints the testing
        print_output("Taylor test passed!")


class StationObservationManager:
    """
    Implements error functional based on observation time series.

    The functional is the squared sum of error between the model and
    observations.

    This object evaluates the model fields at the station locations,
    interpolates the observations time series to the model time, computes the
    error functional, and also stores the model's time series data to disk.
    """
    def __init__(self, mesh, output_directory='outputs'):
        """
        :arg mesh: the 2D mesh object.
        :kwarg output_directory: directory where model time series are stored.
        """
        self.mesh = mesh
        on_sphere = self.mesh.geometric_dimension() == 3
        if on_sphere:
            raise NotImplementedError('Sphere meshes are not supported yet.')
        self.cost_function_scaling = fd.Constant(1.0)
        self.output_directory = output_directory
        # keep observation time series in memory
        self.obs_func_list = []
        # keep model time series in memory during optimization progress
        self.station_value_progress = []
        # model time when cost function was evaluated
        self.simulation_time = []
        self.model_observation_field = None
        self.initialized = False

    def register_observation_data(self, station_names, variable, time,
                                  values, x, y, start_times=None, end_times=None):
        """
        Add station time series data to the object.

        The `x`, and `y` coordinates must be such that
        they allow extraction of model data at the same coordinates.

        :arg list station_names: list of station names
        :arg str variable: canonical variable name, e.g. 'elev'
        :arg list time: array of time stamps, one for each station
        :arg list values: array of observations, one for each station
        :arg list x: list of station x coordinates
        :arg list y: list of station y coordinates
        :kwarg list start_times: optional start times for the observation periods
        :kwarg list end_times: optional end times for the observation periods
        """
        self.station_names = station_names
        self.variable = variable
        self.observation_time = time
        self.observation_values = values
        self.observation_x = x
        self.observation_y = y
        num_stations = len(station_names)
        self._start_times = start_times or -numpy.ones(num_stations)*numpy.inf
        self._end_times = end_times or numpy.ones(num_stations)*numpy.inf

    def set_model_field(self, function):
        """
        Set the model field that will be evaluated.
        """
        self.model_observation_field = function

    def load_observation_data(self, observation_data_dir, station_names, variable,
                              start_times=None, end_times=None):
        """
        Load observation data from disk.

        Assumes that observation data were stored with
        `TimeSeriesCallback2D` during the forward run. For generic case, use
        `register_observation_data` instead.

        :arg str observation_data_dir: directory where observation data is stored
        :arg list station_names: list of station names
        :arg str variable: canonical variable name, e.g. 'elev'
        :kwarg list start_times: optional start times for the observation periods
        :kwarg list end_times: optional end times for the observation periods
        """
        file_list = [
            f'{observation_data_dir}/'
            f'diagnostic_timeseries_{s}_{variable}.hdf5' for s in
            station_names
        ]
        # arrays of time stamps and values for each station
        observation_time = []
        observation_values = []
        observation_coords = []
        for f in file_list:
            with h5py.File(f) as h5file:
                t = h5file['time'][:].flatten()
                v = h5file[variable][:].flatten()
                x = h5file.attrs['x']
                y = h5file.attrs['y']
                observation_coords.append((x, y))
                observation_time.append(t)
                observation_values.append(v)
        # list of station coordinates
        observation_x, observation_y = numpy.array(observation_coords).T
        self.register_observation_data(
            station_names, variable, observation_time,
            observation_values, observation_x, observation_y,
            start_times=start_times, end_times=end_times,
        )
        self.construct_evaluator()

    def update_stations_in_use(self, t):
        """
        Indicate which stations are in use at the current time.

        An entry of unity indicates use, whereas zero indicates disuse.
        """
        if not hasattr(self, 'obs_start_times'):
            self.construct_evaluator()
        in_use = fd.Function(self.fs_points_0d)
        in_use.dat.data[:] = numpy.array(
            numpy.bitwise_and(
                self.obs_start_times <= t, t <= self.obs_end_times
            ), dtype=float)
        self.indicator_0d.assign(in_use)

    def construct_evaluator(self):
        """
        Builds evaluators needed to compute the error functional.
        """
        # Create 0D mesh for station evaluation
        xy = numpy.array((self.observation_x, self.observation_y)).T
        mesh0d = fd.VertexOnlyMesh(self.mesh, xy)
        self.fs_points_0d = fd.FunctionSpace(mesh0d, 'DG', 0)
        self.obs_values_0d = fd.Function(self.fs_points_0d, name='observations')
        self.mod_values_0d = fd.Function(self.fs_points_0d, name='model values')
        self.indicator_0d = fd.Function(self.fs_points_0d, name='station use indicator')
        self.indicator_0d.assign(1.0)
        self.station_weight_0d = fd.Function(self.fs_points_0d, name='station-wise weighting')
        self.station_weight_0d.assign(1.0)
        interp_kw = {}
        if numpy.isfinite(self._start_times).any() or numpy.isfinite(self._end_times).any():
            interp_kw.update({'bounds_error': False, 'fill_value': 0.0})

        # Construct timeseries interpolator
        self.station_interpolators = []
        self.local_station_index = []
        for i in range(self.fs_points_0d.dof_dset.size):
            # loop over local DOFs and match coordinates to observations
            # NOTE this must be done manually as VertexOnlyMesh reorders points
            x_mesh, y_mesh = mesh0d.coordinates.dat.data[i, :]
            xy_diff = xy - numpy.array([x_mesh, y_mesh])
            xy_dist = numpy.sqrt(xy_diff[:, 0]**2 + xy_diff[:, 1]**2)
            j = numpy.argmin(xy_dist)
            self.local_station_index.append(j)

            x, y = xy[j, :]
            t = self.observation_time[j]
            v = self.observation_values[j]
            x_mesh, y_mesh = mesh0d.coordinates.dat.data[i, :]

            msg = 'bad station location ' \
                f'{j} {i} {x} {x_mesh} {y} {y_mesh} {x-x_mesh} {y-y_mesh}'
            assert numpy.allclose([x, y], [x_mesh, y_mesh]), msg
            # create temporal interpolator
            ip = interp1d(t, v, **interp_kw)
            self.station_interpolators.append(ip)

        # Process start and end times for observations
        self.obs_start_times = numpy.array([
            self._start_times[i] for i in self.local_station_index
        ])
        self.obs_end_times = numpy.array([
            self._end_times[i] for i in self.local_station_index
        ])

        # expressions for cost function
        self.misfit_expr = self.obs_values_0d - self.mod_values_0d
        self.initialized = True

    def eval_observation_at_time(self, t):
        """
        Evaluate observation time series at the given time.

        :arg t: model simulation time
        :returns: list of observation time series values at time `t`
        """
        self.update_stations_in_use(t)
        return [float(ip(t)) for ip in self.station_interpolators]

    def eval_cost_function(self, t):
        """
        Evaluate the cost function.

        Should be called at every export of the forward model.
        """
        assert self.initialized, 'Not initialized, call construct_evaluator first.'
        assert self.model_observation_field is not None, 'Model field not set.'
        self.simulation_time.append(t)
        # evaluate observations at simulation time and stash the result
        obs_func = fd.Function(self.fs_points_0d)
        obs_func.dat.data[:] = self.eval_observation_at_time(t)
        self.obs_func_list.append(obs_func)

        # compute square error
        self.obs_values_0d.assign(obs_func)
        self.mod_values_0d.interpolate(self.model_observation_field, ad_block_tag='observation')
        s = self.cost_function_scaling * self.indicator_0d * self.station_weight_0d
        self.J_misfit = fd.assemble(s * self.misfit_expr ** 2 * fd.dx, ad_block_tag='misfit_eval')
        return self.J_misfit

    def dump_time_series(self):
        """
        Stores model time series to disk.

        Obtains station time series from the last optimization iteration,
        and stores the data to disk.

        The output files are have the format
        `{odir}/diagnostic_timeseries_progress_{station_name}_{variable}.hdf5`

        The file contains the simulation time in the `time` array, and the
        station name and coordinates as attributes. The time series data is
        stored as a 2D (n_iterations, n_time_steps) array.
        """
        assert self.station_names is not None

        create_directory(self.output_directory)
        tape = get_working_tape()
        blocks = tape.get_blocks(tag='observation')
        ts_data = [b.get_outputs()[0].saved_output.dat.data for b in blocks]
        # shape (ntimesteps, nstations)
        ts_data = numpy.array(ts_data)
        # append
        self.station_value_progress.append(ts_data)
        var = self.variable
        for ilocal, iglobal in enumerate(self.local_station_index):
            name = self.station_names[iglobal]
            # collect time series data, shape (niters, ntimesteps)
            ts = numpy.array([s[:, ilocal] for s in self.station_value_progress])
            fn = f'diagnostic_timeseries_progress_{name}_{var}.hdf5'
            fn = os.path.join(self.output_directory, fn)
            with h5py.File(fn, 'w') as hdf5file:
                hdf5file.create_dataset(var, data=ts)
                hdf5file.create_dataset('time', data=numpy.array(self.simulation_time))
                attrs = {
                    'x': self.observation_x[iglobal],
                    'y': self.observation_y[iglobal],
                    'location_name': name,
                }
                hdf5file.attrs.update(attrs)


class RegularizationCalculator(abc.ABC):
    """
    Base class for computing regularization terms.

    A derived class should set :attr:`regularization_expr` in
    :meth:`__init__`. Whenever the cost function is evaluated,
    the ratio of this expression and the total mesh area will
    be added.
    """
    @abc.abstractmethod
    def __init__(self, function, scaling=1.0):
        """
        :arg function: Control :class:`Function`
        """
        self.scaling = scaling
        self.regularization_expr = 0
        self.mesh = function.function_space().mesh()
        # calculate mesh area (to scale the cost function)
        self.mesh_area = fd.assemble(fd.Constant(1.0, domain=self.mesh) * fd.dx)
        self.name = function.name()

    def eval_cost_function(self):
        expr = self.scaling * self.regularization_expr / self.mesh_area * fd.dx
        return fd.assemble(expr, ad_block_tag="reg_eval")


class HessianRegularizationCalculator(RegularizationCalculator):
    r"""
    Computes the following regularization term for a cost function
    involving a control :class:`Function` :math:`f`:

    .. math::
        J = \gamma \| (\Delta x)^2 H(f) \|^2,

    where :math:`H` is the Hessian of field :math:`f`.
    """
    def __init__(self, function, gamma, scaling=1.0):
        """
        :arg function: Control :class:`Function`
        :arg gamma: Hessian penalty coefficient
        """
        super().__init__(function, scaling=scaling)
        # solvers to evaluate the gradient of the control
        P1v_2d = get_functionspace(self.mesh, "CG", 1, vector=True)
        P1t_2d = get_functionspace(self.mesh, "CG", 1, tensor=True)
        gradient_2d = fd.Function(P1v_2d, name=f"{self.name} gradient")
        hessian_2d = fd.Function(P1t_2d, name=f"{self.name} hessian")
        self.hessian_calculator = HessianRecoverer2D(
            function, hessian_2d, gradient_2d)

        h = fd.CellSize(self.mesh)
        # regularization expression |hessian|^2
        # NOTE this is normalized by the mesh element size
        # d^2 u/dx^2 * dx^2 ~ du^2
        self.regularization_expr = gamma * fd.inner(hessian_2d, hessian_2d) * h**4

    def eval_cost_function(self):
        self.hessian_calculator.solve()
        return super().eval_cost_function()


class RSpaceRegularizationCalculator(RegularizationCalculator):
    r"""
    Computes the following regularization term for a cost function
    involving a control :class:`Function` :math:`f` from an R-space:

    .. math::
        J = \gamma (f - f_0)^2,

    where :math:`f_0` is a prior, taken to be the initial value of
    :math:`f`.
    """
    def __init__(self, function, gamma, eps=1.0e-03, scaling=1.0):
        """
        :arg function: Control :class:`Function`
        :arg gamma: penalty coefficient
        :kwarg eps: tolerance for normalising by near-zero priors
        """
        super().__init__(function, scaling=scaling)
        R = function.function_space()
        if R.ufl_element().family() != "Real":
            raise ValueError("function must live in R-space")
        prior = fd.Function(R, name=f"{self.name} prior")
        prior.assign(function, annotate=False)  # Set the prior to the initial value
        self.regularization_expr = gamma * (function - prior) ** 2 / ufl.max_value(abs(prior), eps)
        # NOTE: If the prior is small then dividing by prior**2 puts too much emphasis
        #       on the regularization. Therefore, we divide by abs(prior) instead.


class ControlRegularizationManager:
    """
    Handles regularization of multiple control fields
    """
    def __init__(self, function_list, gamma_list, penalty_term_scaling=None,
                 calculator=HessianRegularizationCalculator):
        """
        :arg function_list: list of control functions
        :arg gamma_list: list of penalty parameters
        :kwarg penalty_term_scaling: Penalty term scaling factor
        :kwarg calculator: class used for obtaining regularization
        """
        self.reg_calculators = []
        assert len(function_list) == len(gamma_list), \
            'Number of control functions and parameters must match'
        self.reg_calculators = [
            calculator(f, g, scaling=penalty_term_scaling)
            for f, g in zip(function_list, gamma_list)
        ]

    def eval_cost_function(self):
        return sum([r.eval_cost_function() for r in self.reg_calculators])
