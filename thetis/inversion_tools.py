import firedrake as fd
from firedrake_adjoint import *
from .solver2d import FlowSolver2d
from .utility import create_directory, print_function_value_range
from .utility import get_functionspace
from .log import print_output
from .diagnostics import HessianRecoverer2D
import numpy
import h5py
from scipy.interpolate import interp1d
import time as time_mod
import os


class OptimisationProgress(object):
    """
    Class for stashing progress of the optimisation routine.
    """
    J = 0  # cost function value (float)
    dJdm_list = None  # cost function gradient (Function)
    m_list = None  # control (Function)
    m_progress = []
    J_progress = []
    dJdm_progress = []
    i = 0
    tic = None
    nb_grad_evals = 0
    control_coeff_list = []
    control_list = []

    def __init__(self, output_dir='outputs', no_exports=False, real=False):
        """
        :kwarg output_dir: model output directory
        :kwarg no_exports: toggle exports to vtu
        :kwarg real: is the inversion in the Real space?
        """
        self.output_dir = output_dir
        self.no_exports = no_exports or real
        self.real = real
        self.outfiles_m = []
        self.outfiles_dJdm = []
        self.initialized = False

    def initialize(self):
        if not self.no_exports:
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

        Can be called multiple times in case of multiparameter optimisation.

        :arg f: Function or Constant to be used as a control variable.
        """
        self.control_coeff_list.append(f)
        self.control_list.append(Control(f))

    def reset_counters(self):
        self.nb_grad_evals = 0

    def set_control_state(self, j, djdm_list, m_list):
        """
        Stores optimisation state.

        To call whenever variables are updated.

        :arg j: error functional value
        :arg djdm_list: list of gradient functions
        :arg m_list: list of control coefficents
        """
        self.J = j
        self.dJdm_list = djdm_list
        self.m_list = m_list

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
        Updates optimisation progress and stores variables to disk.

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
        self.dJdm_progress.append(djdm)
        comm = self.control_coeff_list[0].comm
        if comm.rank == 0:
            if self.real:
                numpy.save(f'{self.output_dir}/m_progress', self.m_progress)
            numpy.save(f'{self.output_dir}/J_progress', self.J_progress)
            numpy.save(f'{self.output_dir}/dJdm_progress', self.dJdm_progress)
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
                h5_filename = f'{self.output_dir}/hdf5/control_{j:02d}_{self.i:04d}'
                with fd.DumbCheckpoint(h5_filename, mode=fd.FILE_CREATE) as chk:
                    chk.store(m)
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


class StationObservationManager:
    """
    Implements error functional based on observation time series.

    The functional is the squared sum of error between the model and
    observations.

    This object evaluates the model fields at the station locations,
    interpolates the observations time series to the model time, computes the
    error functional, and also stores the model's time series data to disk.
    """
    def __init__(self, mesh, J_scalar=None, output_directory='outputs'):
        """
        :arg mesh: the 2D mesh object.
        :kwarg J_scalar: Optional factor to scale the error functional. As a
            rule of thumb, it's good to scale the functional to J < 1.
        :kwarg output_directory: directory where model time series are stored.
        """
        self.mesh = mesh
        on_sphere = self.mesh.geometric_dimension() == 3
        if on_sphere:
            raise NotImplementedError('Sphere meshes are not supported yet.')
        self.J_scalar = J_scalar if J_scalar else fd.Constant(1.0)
        self.output_directory = output_directory
        create_directory(self.output_directory)
        # keep observation time series in memory
        self.obs_func_list = []
        # keep model time series in memory during optimisation progress
        self.station_value_progress = []
        # model time when cost function was evaluated
        self.simulation_time = []
        self.model_observation_field = None
        self.initialzed = False

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
        self.initialzed = True

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
        assert self.initialzed, 'Not initialized, call construct_evaluator first.'
        assert self.model_observation_field is not None, 'Model field not set.'
        self.simulation_time.append(t)
        # evaluate observations at simulation time and stash the result
        obs_func = fd.Function(self.fs_points_0d)
        obs_func.dat.data[:] = self.eval_observation_at_time(t)
        self.obs_func_list.append(obs_func)

        # compute square error
        self.obs_values_0d.assign(obs_func)
        self.mod_values_0d.interpolate(self.model_observation_field, ad_block_tag='observation')
        J_misfit = fd.assemble(self.J_scalar*self.indicator_0d*self.station_weight_0d*self.misfit_expr**2*fd.dx)
        return J_misfit

    def dump_time_series(self):
        """
        Stores model time series to disk.

        Obtains station time series from the last optimisation iteration,
        and stores the data to disk.

        The output files are have the format
        `{odir}/diagnostic_timeseries_progress_{station_name}_{variable}.hdf5`

        The file contains the simulation time in the `time` array, and the
        station name and coordinates as attributes. The time series data is
        stored as a 2D (n_iterations, n_time_steps) array.
        """
        assert self.station_names is not None

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


class ControlRegularizationCalculator:
    r"""
    Computes regularization cost function for a control `Function`.

    .. math::
        J = \gamma | H(f) |^2

    where :math:`H` is the Hessian of field math:`f`:.
    """
    def __init__(self, function, gamma_hessian):
        """
        :arg function: Control `Function`
        :arg gamma_Hessian: Hessian penalty coefficient
        """
        self.function = function
        self.gamma_hessian = gamma_hessian
        # solvers to evaluate the gradient of the control
        mesh = function.function_space().mesh()
        P1v_2d = get_functionspace(mesh, 'CG', 1, vector=True)
        P1t_2d = get_functionspace(mesh, 'CG', 1, tensor=True)
        name = function.name()
        gradient_2d = fd.Function(P1v_2d, name=f'{name} gradient')
        hessian_2d = fd.Function(P1t_2d, name=f'{name} hessian')
        self.hessian_calculator = HessianRecoverer2D(
            function, hessian_2d, gradient_2d)

        h = fd.CellSize(mesh)
        # regularization expression |hessian|^2
        # NOTE this is normalized by the mesh element size
        # d^2 u/dx^2 * dx^2 ~ du^2
        self.regularization_hess_expr = gamma_hessian * fd.inner(hessian_2d, hessian_2d)*h**4
        # calculate mesh area (to scale the cost function)
        self.mesh_area = fd.assemble(fd.Constant(1.0, domain=mesh)*fd.dx)

    def eval_cost_function(self):
        self.hessian_calculator.solve()
        J_regularization = fd.assemble(
            self.regularization_hess_expr / self.mesh_area * fd.dx
        )
        return J_regularization


class ControlRegularizationManager:
    """
    Handles regularization of multiple control fields
    """
    def __init__(self, function_list, gamma_list, J_scalar=None):
        """
        :arg function_list: list of control functions
        :arg gamma_list: list of penalty parameters
        :kwarg J_scalar: Penalty term scaling factor
        """
        self.J_scalar = J_scalar
        self.reg_calculators = []
        assert len(function_list) == len(gamma_list), \
            'Number of control functions and parameters must match'
        for f, g in zip(function_list, gamma_list):
            r = ControlRegularizationCalculator(f, g)
            self.reg_calculators.append(r)

    def eval_cost_function(self):
        v = 0
        for r in self.reg_calculators:
            u = r.eval_cost_function()
            if self.J_scalar is not None:
                u *= float(self.J_scalar)
            v += u
        return v


def get_cost_function(solver_obj, op, stationmanager,
                      reg_manager=None, weight_by_variance=False):
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

    Note that the current value of the cost function is
    stashed on the :class:`OptimisationProgress` object.

    :arg solver_obj: the :class:`FlowSolver2d` instance
    :arg op: the :class:`OptimisationProgress` instance
    :arg stationmanager: the :class:`StationManager` instance
    :kwarg reg_manager: the :class:`RegularizationManager` instance
    :kwarg weight_by_variance: should the observation data be
        weighted by the variance at each station?
    """
    assert isinstance(solver_obj, FlowSolver2d)
    assert isinstance(op, OptimisationProgress)
    assert isinstance(stationmanager, StationObservationManager)
    if reg_manager is None:
        op.J = 0
    else:
        assert isinstance(reg_manager, ControlRegularizationManager)
        op.J = reg_manager.eval_cost_function()

    if weight_by_variance:
        var = fd.Function(stationmanager.fs_points_0d)
        for i, j in enumerate(stationmanager.local_station_index):
            var.dat.data[i] = numpy.var(stationmanager.observation_values[j])
        stationmanager.station_weight_0d.assign(1/var)

    def cost_fn():
        t = solver_obj.simulation_time
        J_misfit = stationmanager.eval_cost_function(t)
        op.J += J_misfit

    return cost_fn
