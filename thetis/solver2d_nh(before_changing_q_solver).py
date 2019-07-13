"""
Module for 2D depth averaged solver
"""
from __future__ import absolute_import
from .utility import *
from . import shallowwater_eq
from . import shallowwater_nh
from . import timeintegrator
from . import rungekutta
from . import implicitexplicit
from . import coupled_timeintegrator_2d
from . import tracer_eq_2d
import weakref
import time as time_mod
from mpi4py import MPI
from . import exporter
from .field_defs import field_metadata
from .options import ModelOptions2d
from . import callback
from .log import *
from collections import OrderedDict
import thetis.limiter as limiter


class FlowSolver2d(FrozenClass):
    """
    Main object for 2D depth averaged solver

    **Example**

    Create mesh

    .. code-block:: python

        from thetis import *
        mesh2d = RectangleMesh(20, 20, 10e3, 10e3)

    Create bathymetry function, set a constant value

    .. code-block:: python

        fs_p1 = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(fs_p1, name='Bathymetry').assign(10.0)

    Create solver object and set some options

    .. code-block:: python

        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        options = solver_obj.options
        options.element_family = 'dg-dg'
        options.polynomial_degree = 1
        options.timestepper_type = 'CrankNicolson'
        options.simulation_export_time = 50.0
        options.simulation_end_time = 3600.
        options.timestep = 25.0

    Assign initial condition for water elevation

    .. code-block:: python

        solver_obj.create_function_spaces()
        init_elev = Function(solver_obj.function_spaces.H_2d)
        coords = SpatialCoordinate(mesh2d)
        init_elev.project(exp(-((coords[0] - 4e3)**2 + (coords[1] - 4.5e3)**2)/2.2e3**2))
        solver_obj.assign_initial_conditions(elev=init_elev)

    Run simulation

    .. code-block:: python

        solver_obj.iterate()

    See the manual for more complex examples.
    """
    def __init__(self, mesh2d, bathymetry_2d, options=None):
        """
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg bathymetry_2d: Bathymetry of the domain. Bathymetry stands for
            the mean water depth (positive downwards).
        :type bathymetry_2d: :class:`Function`
        :kwarg options: Model options (optional). Model options can also be
            changed directly via the :attr:`.options` class property.
        :type options: :class:`.ModelOptions2d` instance
        """
        self._initialized = False
        self.mesh2d = mesh2d
        self.comm = mesh2d.comm

        # add boundary length info
        bnd_len = compute_boundary_length(self.mesh2d)
        self.mesh2d.boundary_len = bnd_len
        self.normal_2d = FacetNormal(self.mesh2d)
        self.boundary_markers = self.mesh2d.exterior_facets.unique_markers

        self.dt = None
        """Time step"""

        self.options = ModelOptions2d()
        """
        Dictionary of all options. A :class:`.ModelOptions2d` object.
        """
        if options is not None:
            self.options.update(options)

        # simulation time step bookkeeping
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + self.options.simulation_export_time

        self.callbacks = callback.CallbackManager()
        """
        :class:`.CallbackManager` object that stores all callbacks
        """

        self.fields = FieldDict()
        """
        :class:`.FieldDict` that holds all functions needed by the solver
        object
        """

        self.function_spaces = AttrDict()
        """
        :class:`.AttrDict` that holds all function spaces needed by the
        solver object
        """

        self.fields.bathymetry_2d = bathymetry_2d

        self.export_initial_state = True
        """Do export initial state. False if continuing a simulation"""

        self.bnd_functions = {'shallow_water': {}, 'momentum': {}, 'tracer': {}}

        self._isfrozen = True

    def compute_time_step(self, u_scale=Constant(0.0)):
        r"""
        Computes maximum explicit time step from CFL condition.

        .. math :: \Delta t = \frac{\Delta x}{U}

        Assumes velocity scale :math:`U = \sqrt{g H} + U_{scale}` where
        :math:`U_{scale}` is estimated advective velocity.

        :kwarg u_scale: User provided maximum advective velocity scale
        :type u_scale: float or :class:`Constant`
        """
        csize = self.fields.h_elem_size_2d
        bath = self.fields.bathymetry_2d
        fs = bath.function_space()
        bath_pos = Function(fs, name='bathymetry')
        bath_pos.assign(bath)
        min_depth = 0.05
        bath_pos.dat.data[bath_pos.dat.data < min_depth] = min_depth
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solution = Function(fs)
        g = physical_constants['g_grav']
        u = (sqrt(g * bath_pos) + u_scale)
        a = inner(test, trial) * dx
        l = inner(test, csize / u) * dx
        solve(a == l, solution)
        return solution

    def set_time_step(self, alpha=0.05):
        """
        Sets the model the model time step

        If the time integrator supports automatic time step, and
        :attr:`ModelOptions2d.timestepper_options.use_automatic_timestep` is
        `True`, we compute the maximum time step allowed by the CFL condition.
        Otherwise uses :attr:`ModelOptions2d.timestep`.

        :kwarg float alpha: CFL number scaling factor
        """
        automatic_timestep = (hasattr(self.options.timestepper_options, 'use_automatic_timestep') and
                              self.options.timestepper_options.use_automatic_timestep)
        # TODO revisit math alpha is OBSOLETE
        if automatic_timestep:
            mesh2d_dt = self.compute_time_step(u_scale=self.options.horizontal_velocity_scale)
            dt = self.options.cfl_2d*alpha*float(mesh2d_dt.dat.data.min())
            dt = self.comm.allreduce(dt, op=MPI.MIN)
            self.dt = dt
        else:
            assert self.options.timestep is not None
            assert self.options.timestep > 0.0
            self.dt = self.options.timestep
        if self.comm.rank == 0:
            print_output('dt = {:}'.format(self.dt))
            sys.stdout.flush()

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces: elev in H, uv in U, mixed is W
        self.function_spaces.P0_2d = FunctionSpace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = FunctionSpace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = FunctionSpace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.function_spaces.P1DG_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1, name='P1DGv_2d')
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = FunctionSpace(self.mesh2d, 'RT', self.options.polynomial_degree+1, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        elif self.options.element_family == 'dg-cg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'CG', self.options.polynomial_degree+1, name='H_2d')
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
            self.function_spaces.H_2d = FunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self.function_spaces.Q_2d = FunctionSpace(self.mesh2d, 'DG', 1, name='Q_2d')

        self._isfrozen = True

    def create_functions(self):
        """
        Creates extra functions, including fields
        """
        self.bathymetry_dg_old = Function(self.function_spaces.H_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
        self.elev_2d_old = Function(self.function_spaces.H_2d)
        self.elev_2d_mid = Function(self.function_spaces.H_2d)
        self.uv_2d_dg = Function(self.function_spaces.P1DGv_2d)
        self.uv_2d_old = Function(self.function_spaces.U_2d)
        self.uv_2d_mid = Function(self.function_spaces.U_2d)
        self.fields.uv_nh = Function(self.function_spaces.U_2d)
        self.fields.w_nh = Function(self.function_spaces.H_2d)
        self.fields.q_2d = Function(self.function_spaces.P2_2d)

        # functions for landslide modelling
        if self.options.landslide is True:
            self.bathymetry_ls = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
            self.fields.solution_ls = Function(self.function_spaces.V_2d)
            self.fields.uv_ls = self.fields.solution_ls.sub(0)
            self.fields.elev_ls = self.fields.solution_ls.sub(0).sub(1)
            self.fields.slide_source = Function(self.function_spaces.H_2d)

        # functions for multi-layer approach
        for k in range(self.options.n_layers):
            setattr(self, 'uv_av_' + str(k+1), Function(self.function_spaces.U_2d))
            #self.__dict__['uv_av_' + str(k+1)] = Function(self.function_spaces.U_2d)
            setattr(self, 'w_' + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'w_' + str(k) + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'q_' + str(k+1), Function(self.function_spaces.P2_2d))
            if k == 0:
                list_fs = [self.function_spaces.P2_2d]
                setattr(self, 'w_' + str(k), Function(self.function_spaces.H_2d))
                setattr(self, 'q_' + str(k), Function(self.function_spaces.P2_2d))
                setattr(self, 'q_' + str(self.options.n_layers+1), Function(self.function_spaces.P2_2d))
            else:
                list_fs.append(self.function_spaces.P2_2d)
        self.function_spaces.q_mixed_n_layers = MixedFunctionSpace(list_fs)
        self.q_mixed_n_layers = Function(self.function_spaces.q_mixed_n_layers)










    def create_equations(self):
        """
        Creates shallow water equations
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)
        self.create_functions()

        # ----- Equations
        self.eq_sw = shallowwater_eq.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg, #self.fields.bathymetry_2d,
            self.options
        )
        #self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_sw_mom = shallowwater_nh.ShallowWaterMomentumEquation(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)
        self.eq_free_surface = shallowwater_nh.FreeSurfaceEquation(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
            self.bathymetry_dg,
            self.options)
        if self.options.solve_tracer:
            self.fields.tracer_2d = Function(self.function_spaces.Q_2d, name='tracer_2d')
            self.eq_tracer = tracer_eq_2d.TracerEquation2D(self.function_spaces.Q_2d, bathymetry=self.fields.bathymetry_2d,
                                                           use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
            if self.options.use_limiter_for_tracers and self.options.polynomial_degree > 0:
                self.tracer_limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.Q_2d)
            else:
                self.tracer_limiter = None

        self._isfrozen = True  # disallow creating new attributes

    def create_timestepper(self):
        """
        Creates time stepper instance
        """
        if not hasattr(self, 'eq_sw'):
            self.create_equations()

        self._isfrozen = False

        if self.options.log_output and not self.options.no_exports:
            logfile = os.path.join(create_directory(self.options.output_directory), 'log')
            filehandler = logging.logging.FileHandler(logfile, mode='w')
            filehandler.setFormatter(logging.logging.Formatter('%(message)s'))
            output_logger.addHandler(filehandler)

        # ----- Time integrators
        self.field_dic = {
            'linear_drag_coefficient': self.options.linear_drag_coefficient,
            'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
            'manning_drag_coefficient': self.options.manning_drag_coefficient,
            'viscosity_h': self.options.horizontal_viscosity,
            'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
            'coriolis': self.options.coriolis_frequency,
            'wind_stress': self.options.wind_stress,
            'atmospheric_pressure': self.options.atmospheric_pressure,
            'momentum_source': self.options.momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
            'uv': self.fields.solution_2d.sub(0),
            'eta': self.fields.solution_2d.sub(1),}
        fields = self.field_dic
        self.set_time_step()
        if self.options.timestepper_type == 'SSPRK33':
            self.timestepper = rungekutta.SSPRK33(self.eq_sw, self.fields.solution_2d,
                                                  fields, self.dt,
                                                  bnd_conditions=self.bnd_functions['shallow_water'],
                                                  solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'ForwardEuler':
            self.timestepper = timeintegrator.ForwardEuler(self.eq_sw, self.fields.solution_2d,
                                                           fields, self.dt,
                                                           bnd_conditions=self.bnd_functions['shallow_water'],
                                                           solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'BackwardEuler':
            self.timestepper = rungekutta.BackwardEuler(self.eq_sw, self.fields.solution_2d,
                                                        fields, self.dt,
                                                        bnd_conditions=self.bnd_functions['shallow_water'],
                                                        solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'CrankNicolson':
            if self.options.solve_tracer:
                self.timestepper = coupled_timeintegrator_2d.CoupledCrankNicolson2D(weakref.proxy(self))
            else:
                self.timestepper = timeintegrator.CrankNicolson(self.eq_sw, self.fields.solution_2d,
                                                                fields, self.dt,
                                                                bnd_conditions=self.bnd_functions['shallow_water'],
                                                                solver_parameters=self.options.timestepper_options.solver_parameters,
                                                                semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                                theta=self.options.timestepper_options.implicitness_theta)
        elif self.options.timestepper_type == 'DIRK22':
            self.timestepper = rungekutta.DIRK22(self.eq_sw, self.fields.solution_2d,
                                                 fields, self.dt,
                                                 bnd_conditions=self.bnd_functions['shallow_water'],
                                                 solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'DIRK33':
            self.timestepper = rungekutta.DIRK33(self.eq_sw, self.fields.solution_2d,
                                                 fields, self.dt,
                                                 bnd_conditions=self.bnd_functions['shallow_water'],
                                                 solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'SteadyState':
            self.timestepper = timeintegrator.SteadyState(self.eq_sw, self.fields.solution_2d,
                                                          fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=self.options.timestepper_options.solver_parameters)
        elif self.options.timestepper_type == 'PressureProjectionPicard':

            u_test = TestFunction(self.function_spaces.U_2d)
            self.eq_mom = shallowwater_eq.ShallowWaterMomentumEquation(
                u_test, self.function_spaces.U_2d, self.function_spaces.H_2d,
                self.fields.bathymetry_2d,
                options=self.options
            )
            self.eq_mom.bnd_functions = self.bnd_functions['shallow_water']
            self.timestepper = timeintegrator.PressureProjectionPicard(self.eq_sw, self.eq_mom, self.fields.solution_2d,
                                                                       fields, self.dt,
                                                                       bnd_conditions=self.bnd_functions['shallow_water'],
                                                                       solver_parameters=self.options.timestepper_options.solver_parameters_pressure,
                                                                       solver_parameters_mom=self.options.timestepper_options.solver_parameters_momentum,
                                                                       semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                                       theta=self.options.timestepper_options.implicitness_theta,
                                                                       iterations=self.options.timestepper_options.picard_iterations)

        elif self.options.timestepper_type == 'SSPIMEX':
            # TODO meaningful solver params
            sp_impl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            sp_expl = {
                'ksp_type': 'gmres',
                'pc_type': 'fieldsplit',
                'pc_fieldsplit_type': 'multiplicative',
            }
            self.timestepper = implicitexplicit.IMEXLPUM2(self.eq_sw, self.fields.solution_2d, fields, self.dt,
                                                          bnd_conditions=self.bnd_functions['shallow_water'],
                                                          solver_parameters=sp_expl,
                                                          solver_parameters_dirk=sp_impl)
        else:
            raise Exception('Unknown time integrator type: '+str(self.options.timestepper_type))
        print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))
        self._isfrozen = True  # disallow creating new attributes

    def create_exporters(self):
        """
        Creates file exporters
        """
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        self._isfrozen = False
        # correct treatment of the split 2d functions
        uv_2d, elev_2d = self.fields.solution_2d.split()
        self.fields.uv_2d = uv_2d
        self.fields.elev_2d = elev_2d
        self.exporters = OrderedDict()
        if not self.options.no_exports:
            e = exporter.ExportManager(self.options.output_directory,
                                       self.options.fields_to_export,
                                       self.fields,
                                       field_metadata,
                                       export_type='vtk',
                                       verbose=self.options.verbose > 0)
            self.exporters['vtk'] = e
            hdf5_dir = os.path.join(self.options.output_directory, 'hdf5')
            e = exporter.ExportManager(hdf5_dir,
                                       self.options.fields_to_export_hdf5,
                                       self.fields,
                                       field_metadata,
                                       export_type='hdf5',
                                       verbose=self.options.verbose > 0)
            self.exporters['hdf5'] = e

        self._isfrozen = True  # disallow creating new attributes

    def initialize(self):
        """
        Creates function spaces, equations, time stepper and exporters
        """
        if not hasattr(self, 'U_2d'):
            self.create_function_spaces()
        if not hasattr(self, 'eq_sw'):
            self.create_equations()
        if not hasattr(self, 'timestepper'):
            self.create_timestepper()
        if not hasattr(self, 'exporters'):
            self.create_exporters()
        self._initialized = True

    def assign_initial_conditions(self, elev=None, uv=None, tracer=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()
        uv_2d, elev_2d = self.fields.solution_2d.split()
        if elev is not None:
            elev_2d.project(elev)
        if uv is not None:
            uv_2d.project(uv)
        if tracer is not None and self.options.solve_tracer:
            self.fields.tracer_2d.project(tracer)

        self.timestepper.initialize(self.fields.solution_2d)

    def add_callback(self, callback, eval_interval='export'):
        """
        Adds callback to solver object

        :arg callback: :class:`.DiagnosticCallback` instance
        :kwarg string eval_interval: Determines when callback will be evaluated,
            either 'export' or 'timestep' for evaluating after each export or
            time step.
        """
        self.callbacks.add(callback, eval_interval)

    def export(self):
        """
        Export all fields to disk

        Also evaluates all callbacks set to 'export' interval.
        """
        self.callbacks.evaluate(mode='export')
        for e in self.exporters.values():
            e.export()

    def load_state(self, i_export, outputdir=None, t=None, iteration=None):
        """
        Loads simulation state from hdf5 outputs.

        This replaces :meth:`.assign_initial_conditions` in model initilization.

        This assumes that model setup is kept the same (e.g. time step) and
        all pronostic state variables are exported in hdf5 format. The required
        state variables are: elev_2d, uv_2d

        Currently hdf5 field import only works for the same number of MPI
        processes.

        :arg int i_export: export index to load
        :kwarg string outputdir: (optional) directory where files are read from.
            By default ``options.output_directory``.
        :kwarg float t: simulation time. Overrides the time stamp stored in the
            hdf5 files.
        :kwarg int iteration: Overrides the iteration count in the hdf5 files.
        """
        if not self._initialized:
            self.initialize()
        if outputdir is None:
            outputdir = self.options.output_directory
        # create new ExportManager with desired outputdir
        state_fields = ['uv_2d', 'elev_2d']
        hdf5_dir = os.path.join(outputdir, 'hdf5')
        e = exporter.ExportManager(hdf5_dir,
                                   state_fields,
                                   self.fields,
                                   field_metadata,
                                   export_type='hdf5',
                                   verbose=self.options.verbose > 0)
        e.exporters['uv_2d'].load(i_export, self.fields.uv_2d)
        e.exporters['elev_2d'].load(i_export, self.fields.elev_2d)
        self.assign_initial_conditions()

        # time stepper bookkeeping for export time step
        self.i_export = i_export
        self.next_export_t = self.i_export*self.options.simulation_export_time
        if iteration is None:
            iteration = int(np.ceil(self.next_export_t/self.dt))
        if t is None:
            t = iteration*self.dt
        self.iteration = iteration
        self.simulation_time = t

        # for next export
        self.export_initial_state = outputdir != self.options.output_directory
        if self.export_initial_state:
            offset = 0
        else:
            offset = 1
        self.next_export_t += self.options.simulation_export_time
        for e in self.exporters.values():
            e.set_next_export_ix(self.i_export + offset)

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.solution_2d.split()[1])
        norm_u = norm(self.fields.solution_2d.split()[0])

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, e=norm_h,
                                 u=norm_u, cpu=cputime))
        sys.stdout.flush()

    def iterate(self, update_forcings=None,
                export_func=None):
        """
        Runs the simulation

        Iterates over the time loop until time ``options.simulation_end_time`` is reached.
        Exports fields to disk on ``options.simulation_export_time`` intervals.

        :kwarg update_forcings: User-defined function that takes simulation
            time as an argument and updates time-dependent boundary conditions
            (if any).
        :kwarg export_func: User-defined function (with no arguments) that will
            be called on every export.
        """
        # TODO I think export function is obsolete as callbacks are in place
        if not self._initialized:
            self.initialize()

        self.options.use_limiter_for_tracers &= self.options.polynomial_degree > 0

        t_epsilon = 1.0e-5
        cputimestamp = time_mod.clock()
        next_export_t = self.simulation_time + self.options.simulation_export_time

        dump_hdf5 = self.options.export_diagnostics and not self.options.no_exports
        if self.options.check_volume_conservation_2d:
            c = callback.VolumeConservation2DCallback(self,
                                                      export_to_hdf5=dump_hdf5,
                                                      append_to_log=True)
            self.add_callback(c)

        if self.options.check_tracer_conservation:
            c = callback.TracerMassConservation2DCallback('tracer_2d',
                                                          self,
                                                          export_to_hdf5=dump_hdf5,
                                                          append_to_log=True)
            self.add_callback(c, eval_interval='export')

        if self.options.check_tracer_overshoot:
            c = callback.TracerOvershootCallBack('tracer_2d',
                                                 self,
                                                 export_to_hdf5=dump_hdf5,
                                                 append_to_log=True)
            self.add_callback(c, eval_interval='export')

        # initial export
        self.print_state(0.0)
        if self.export_initial_state:
            self.export()
            if export_func is not None:
                export_func()
            if 'vtk' in self.exporters:
                self.exporters['vtk'].export_bathymetry(self.fields.bathymetry_2d)

        initial_simulation_time = self.simulation_time
        internal_iteration = 0

        # split solution to facilitate the following
        uv_2d, elev_2d = self.fields.solution_2d.split()
        # trial and test functions used to update
        uv_tri = TrialFunction(self.function_spaces.U_2d)
        uv_test = TestFunction(self.function_spaces.U_2d)
        w_tri = TrialFunction(self.function_spaces.H_2d)
        w_test = TestFunction(self.function_spaces.H_2d)
        uta_test, eta_test = TestFunctions(self.fields.solution_2d.function_space())

        while self.simulation_time <= self.options.simulation_end_time + t_epsilon:

            #self.timestepper.advance(self.simulation_time, update_forcings)
            self.uv_2d_old.assign(self.fields.uv_2d)
            self.elev_2d_old.assign(self.fields.elev_2d)
            self.elev_2d_mid.assign(self.fields.elev_2d)
            self.bathymetry_dg_old.assign(self.bathymetry_dg)

            if self.simulation_time <= t_epsilon:
                # timestepper for free surface equation
                timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_old,
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              #solver_parameters=solver_parameters,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)

            hydrostatic_solver_2d = False
            # --- Hydrostatic solver ---
            if hydrostatic_solver_2d:
                self.timestepper.advance(self.simulation_time, update_forcings)
            else: #arbitrary_multi_layer_NH_solver: # i.e. multi-layer NH model
                ### layer thickness accounting for total depth
                alpha = self.options.alpha_nh
                if len(self.options.alpha_nh) < self.options.n_layers:
                    n = self.options.n_layers - len(self.options.alpha_nh)
                    sum = 0.
                    if len(self.options.alpha_nh) >= 1:
                        for k in range(len(self.options.alpha_nh)):
                            sum = sum + self.options.alpha_nh[0]
                    for k in range(n):
                        alpha.append((1. - sum)/n)
                if self.options.n_layers == 1:
                    alpha[0] = 1.
                ###
                #h_old = shallowwater_eq.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg_old, self.options).get_total_depth(self.elev_2d_old)
                h_old = self.bathymetry_dg_old + self.elev_2d_old
                if self.options.use_wetting_and_drying and not self.options.thin_film:
                    h_old += 2.*self.options.wd_mindep**2 / (2.*self.options.wd_mindep + abs(h_old)) + 0.5*(abs(h_old) - h_old)
                # layer thickness and z-coordinate, note ghost layers added
                h_layer_old = [h_old*alpha[0]]
                for k in range(self.options.n_layers):
                    h_layer_old.append(h_old*alpha[k])
                    if k == self.options.n_layers - 1:
                        h_layer_old.append(h_old*alpha[k])
                z_old_dic = {'z_0': -self.bathymetry_dg_old}
                for k in range(self.options.n_layers):
                    z_old_dic['z_'+str(k+1)] = z_old_dic['z_'+str(k)] + h_layer_old[k+1]
                    z_old_dic['z_'+str(k)+str(k+1)] = 0.5*(z_old_dic['z_'+str(k)] + z_old_dic['z_'+str(k+1)])
                # solve 2D depth-integrated equations initially
                self.timestepper.advance(self.simulation_time, update_forcings)
                h_mid = self.bathymetry_dg + self.fields.elev_2d
                if self.options.use_wetting_and_drying and not self.options.thin_film:
                    h_mid += 2.*self.options.wd_mindep**2 / (2.*self.options.wd_mindep + abs(h_mid)) + 0.5*(abs(h_mid) - h_mid)
                # update layer thickness and z-coordinate
                h_layer = [h_mid*alpha[0]]
                for k in range(self.options.n_layers):
                    h_layer.append(h_mid*alpha[k])
                    if k == self.options.n_layers - 1:
                        h_layer.append(h_mid*alpha[k])
                z_dic = {'z_0': -self.bathymetry_dg}
                for k in range(self.options.n_layers):
                    z_dic['z_'+str(k+1)] = z_dic['z_'+str(k)] + h_layer[k+1]
                    z_dic['z_'+str(k)+str(k+1)] = 0.5*(z_dic['z_'+str(k)] + z_dic['z_'+str(k+1)])

                # velocities at the interface
                u_dic = {}
                w_dic = {}
                omega_dic = {}
                for k in range(self.options.n_layers + 1):
                    # old uv and w velocities at the interface
                    if k == 0:
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k+1)) - (h_layer[k+2]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+1)) + 
                                                                                    h_layer[k+1]/(h_layer[k+1] + h_layer[k+2])*getattr(self, 'uv_av_'+str(k+2)))
                        w_dic['z_'+str(k)] = -inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))
                        if self.options.landslide is True:
                            w_dic['z_'+str(k)] += -self.fields.slide_source
                    elif k > 0 and k < self.options.n_layers:
                        u_dic['z_'+str(k)] = h_layer[k+1]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k)) + \
                                             h_layer[k]/(h_layer[k] + h_layer[k+1])*getattr(self, 'uv_av_'+str(k+1))
                        w_dic['z_'+str(k)] = 2.*getattr(self, 'w_'+str(k-1)+str(k)) - w_dic['z_'+str(k-1)]
                    else: # i.e. k == self.options.n_layers
                        u_dic['z_'+str(k)] = 2.*getattr(self, 'uv_av_'+str(k)) - u_dic['z_'+str(k-1)]
                        w_dic['z_'+str(k)] = 2.*getattr(self, 'w_'+str(k-1)+str(k)) - w_dic['z_'+str(k-1)]
                    # mass flux due to mesh movement TODO check to modify and implement again
                    omega_dic['z_'+str(k)] = w_dic['z_'+str(k)] - (z_dic['z_'+str(k)] - z_old_dic['z_'+str(k)])/self.dt - inner(u_dic['z_'+str(k)], grad(z_dic['z_'+str(k)]))

                if self.simulation_time <= t_epsilon:
                    if self.options.n_layers >= 2:
                        timestepper_dic = {}
                        for k in range(self.options.n_layers - 1):
                            timestepper_dic['layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_sw_mom, getattr(self, 'uv_av_'+str(k+1)),
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)
                            consider_mesh_relative_velocity = False
                            if consider_mesh_relative_velocity:
                                timestepper_dic['layer_'+str(k+1)].F += self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(u_dic['z_'+str(k+1)] - getattr(self, 'uv_av_'+str(k+1))) -
                                                                                          omega_dic['z_'+str(k)]*(u_dic['z_'+str(k)] - getattr(self, 'uv_av_'+str(k+1))), uv_test)*dx
                                timestepper_dic['layer_'+str(k+1)].update_solver()

                if self.options.n_layers >= 2:
                    sum_uv_av = 0. 
                    # except the layer adjacent to the free surface
                    for k in range(self.options.n_layers - 1):
                        timestepper_dic['layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                        #sum_uv_av += getattr(self, 'uv_av_'+str(k+1)) # cannot sum by this way
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    getattr(self, 'uv_av_'+str(self.options.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.options.n_layers-1])

                # build the solver for the mixed Poisson equations
                if self.simulation_time <= t_epsilon:
                    q_test = TestFunctions(self.function_spaces.q_mixed_n_layers)
                    q_tuple = split(self.q_mixed_n_layers)
                    if self.options.n_layers == 1:
                        q_test = [TestFunction(self.q_0.function_space())]
                        q_tuple = [self.q_0]
                    # re-arrange the list of q
                    q = []
                    for k in range(self.options.n_layers):
                        q.append(q_tuple[k])
                        if k == self.options.n_layers - 1:
                            # free-surface NH pressure
                            q.append(0.)
                    f = 0.
                    for k in range(self.options.n_layers):
                        # weak form of div(h_{k+1}*uv_av_{k+1})
                        div_hu_term = div(h_layer[k+1]*getattr(self, 'uv_av_'+str(k+1)))*q_test[k]*dx + \
                                      0.5*self.dt*h_layer[k+1]*dot(grad(q[k]+q[k+1]), grad(q_test[k]))*dx + \
                                      self.dt*(q[k]-q[k+1])*dot(grad(z_dic['z_'+str(k)+str(k+1)]), grad(q_test[k]))*dx
                        if k >= 1:
                            for i in range(k):
                                div_hu_term += 2.*(div(h_layer[i+1]*getattr(self, 'uv_av_'+str(i+1)))*q_test[k]*dx + \
                                               0.5*self.dt*h_layer[i+1]*dot(grad(q[i]+q[i+1]), grad(q_test[k]))*dx + \
                                               self.dt*(q[i]-q[i+1])*dot(grad(z_dic['z_'+str(i)+str(i+1)]), grad(q_test[k]))*dx)
                        # weak form of w_{k}{k+1}
                        vert_vel_term = 2.*(getattr(self, 'w_'+str(k)+str(k+1)) + self.dt*(q[k] - q[k+1])/h_layer[k+1])*q_test[k]*dx
                        consider_vert_adv = False#True
                        if consider_vert_adv: # TODO if make sure that considering vertical advection is benefitial, delete this logical variable
                            #vert_vel_term += -2.*self.dt*dot(getattr(self, 'uv_av_'+str(k+1)), grad(getattr(self, 'w_'+str(k)+str(k+1))))*q_test[k]*dx
                            vert_vel_term += 2.*self.dt*(div(getattr(self, 'uv_av_'+str(k+1))*q_test[k])*getattr(self, 'w_'+str(k)+str(k+1))*dx -
                                                         avg(getattr(self, 'w_'+str(k)+str(k+1)))*jump(q_test[k], inner(getattr(self, 'uv_av_'+str(k+1)), self.normal_2d))*dS)
                            if consider_mesh_relative_velocity:
                                vert_vel_term += -2.*self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(w_dic['z_'+str(k+1)] - getattr(self, 'w_'+str(k)+str(k+1))) -
                                                                                omega_dic['z_'+str(k)]*(w_dic['z_'+str(k)] - getattr(self, 'w_'+str(k)+str(k+1))), q_test[k])*dx
                        # weak form of RHS terms
                        if k == 0: # i.e. the layer adjacent to the bottom
                            if self.options.n_layers == 1:
                                grad_1_layer1 = grad(z_dic['z_'+str(k)] + z_dic['z_'+str(k+1)])
                                interface_term = dot(grad_1_layer1, getattr(self, 'uv_av_'+str(k+1)))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]))*dx - \
                                                 self.dt*(1./h_layer[k+1]*dot(grad_1_layer1, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                            else:
                                grad_1_layer1 = grad(2.*z_dic['z_'+str(k)] + h_layer[k+1]*h_layer[k+2]/(h_layer[k+1] + h_layer[k+2]))
                                grad_2_layer1 = grad(h_layer[k+1]*h_layer[k+1]/(h_layer[k+1] + h_layer[k+2]))
                                interface_term = (dot(grad_1_layer1, getattr(self, 'uv_av_'+str(k+1))) + dot(grad_2_layer1, getattr(self, 'uv_av_'+str(k+2))))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]) - div(grad_2_layer1*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                                 self.dt*(1./h_layer[k+1]*dot(grad_1_layer1, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./h_layer[k+2]*dot(grad_2_layer1, grad(z_dic['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        elif k == self.options.n_layers - 1: # i.e. the layer adjacent to the free surface
                            grad_1_layern = grad(-h_layer[k+1]*h_layer[k+1]/(h_layer[k] + h_layer[k+1]))
                            grad_2_layern = grad(2.*z_dic['z_'+str(k+1)] - h_layer[k]*h_layer[k+1]/(h_layer[k] + h_layer[k+1]))
                            interface_term = (dot(grad_1_layern, getattr(self, 'uv_av_'+str(k))) + dot(grad_2_layern, getattr(self, 'uv_av_'+str(k+1))))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layern*q_test[k])*(q[k-1]+q[k]) - div(grad_2_layern*q_test[k])*(q[k]+q[k+1]))*dx - \
                                             self.dt*(1./h_layer[k]*dot(grad_1_layern, grad(z_dic['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./h_layer[k+1]*dot(grad_2_layern, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                        else:
                            grad_1_layerk = h_layer[k+1]/(h_layer[k] + h_layer[k+1])*grad(z_dic['z_'+str(k)])
                            grad_2_layerk = h_layer[k]/(h_layer[k] + h_layer[k+1])*grad(z_dic['z_'+str(k)]) + \
                                            h_layer[k+2]/(h_layer[k+1] + h_layer[k+2])*grad(z_dic['z_'+str(k+1)])
                            grad_3_layerk = h_layer[k+1]/(h_layer[k+1] + h_layer[k+2])*grad(z_dic['z_'+str(k+1)])
                            interface_term = (dot(grad_1_layerk, getattr(self, 'uv_av_'+str(k))) + 
                                              dot(grad_2_layerk, getattr(self, 'uv_av_'+str(k+1))) + 
                                              dot(grad_3_layerk, getattr(self, 'uv_av_'+str(k+2))))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layerk*q_test[k])*(q[k-1]+q[k]) - 
                                                          div(grad_2_layerk*q_test[k])*(q[k]+q[k+1]) - 
                                                          div(grad_3_layerk*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                             self.dt*(1./h_layer[k]*dot(grad_1_layerk, grad(z_dic['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./h_layer[k+1]*dot(grad_2_layerk, grad(z_dic['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                      1./h_layer[k+2]*dot(grad_3_layerk, grad(z_dic['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        # weak form of slide source term
                        if self.options.landslide:
                            slide_source_term = -2.*self.fields.slide_source*q_test[k]*dx
                            f += slide_source_term
                        f += div_hu_term + vert_vel_term - interface_term

                        for bnd_marker in self.boundary_markers:
                            func = self.bnd_functions['shallow_water'].get(bnd_marker)
                            ds_bnd = ds(int(bnd_marker))
                            if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                                # bnd terms of div(h_{k+1}*uv_av_{k+1})
                                f += -self.dt*(q[k]-q[k+1])*dot(grad(z_dic['z_'+str(k)+str(k+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                if k >= 1:
                                    for i in range(k):
                                        f += -2*self.dt*(q[i]-q[i+1])*dot(grad(z_dic['z_'+str(i)+str(i+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                # bnd terms of RHS terms about interface connection
                                if k == 0:
                                    if self.options.n_layers == 1:
                                        f += 0.5*self.dt*dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1])*q_test[k]*ds_bnd
                                    else:
                                        f += 0.5*self.dt*(dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1]) + 
                                                          dot(grad_2_layer1, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd
                                elif k == self.options.n_layers - 1:
                                    f += 0.5*self.dt*(dot(grad_1_layern, self.normal_2d)*(q[k-1]+q[k]) + 
                                                      dot(grad_2_layern, self.normal_2d)*(q[k]+q[k+1]))*q_test[k]*ds_bnd
                                else:
                                    f += 0.5*self.dt*(dot(grad_1_layerk, self.normal_2d)*(q[k-1]+q[k]) +
                                                      dot(grad_2_layerk, self.normal_2d)*(q[k]+q[k+1]) +
                                                      dot(grad_3_layerk, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd

                    prob = NonlinearVariationalProblem(f, self.q_mixed_n_layers)
                    if self.options.n_layers == 1:
                        prob = NonlinearVariationalProblem(f, self.q_0)
                    solver = NonlinearVariationalSolver(prob,
                                                        solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'snes_monitor': False,
                                                               'pc_type': 'lu'})
                if self.options.n_layers == 1:
                    self.uv_av_1.assign(uv_2d)
                solver.solve()
                for k in range(self.options.n_layers):
                    if self.options.n_layers > 1:
                        getattr(self, 'q_'+str(k)).assign(self.q_mixed_n_layers.split()[k])
                    if k == self.options.n_layers - 1:
                        getattr(self, 'q_'+str(k+1)).assign(0.)
                self.fields.q_2d.assign(self.q_0)

                # update depth-averaged uv_2d
                a = inner(uv_tri, uv_test)*dx
                l = 0.
                for k in range(self.options.n_layers):
                    l += inner(-self.dt/h_mid*grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]), uv_test)*dx
                    if k == self.options.n_layers - 1:
                        l += inner(uv_2d - self.dt/h_mid*(getattr(self, 'q_'+str(0))*grad(z_dic['z_'+str(0)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)])), uv_test)*dx
                solve(a == l, uv_2d)
                self.uv_2d_mid.assign(uv_2d)
                # update layer-averaged self.uv_av_{k+1}
                if self.options.n_layers >= 2:
                    sum_uv_av = 0.
                    for k in range(self.options.n_layers - 1):
                        a = inner(uv_tri, uv_test)*dx
                        l = inner(getattr(self, 'uv_av_'+str(k+1)) - self.dt/h_layer[k+1]*(grad((getattr(self, 'q_'+str(k)) + getattr(self, 'q_'+str(k+1)))/2.*h_layer[k+1]) + 
                                                          getattr(self, 'q_'+str(k))*grad(z_dic['z_'+str(k)]) - getattr(self, 'q_'+str(k+1))*grad(z_dic['z_'+str(k+1)])), uv_test)*dx
                        solve(a == l, getattr(self, 'uv_av_'+str(k+1)))
                        sum_uv_av = sum_uv_av + alpha[k]*getattr(self, 'uv_av_'+str(k+1))
                    # update layer-averaged velocity of the free-surface layer
                    getattr(self, 'uv_av_'+str(self.options.n_layers)).project((uv_2d - sum_uv_av)/alpha[self.options.n_layers-1])
                # update layer-integrated vertical velocity w_{k}{k+1}
                a = w_tri*w_test*dx
                for k in range(self.options.n_layers):
                    l = (getattr(self, 'w_'+str(k)+str(k+1)) + self.dt*(getattr(self, 'q_'+str(k)) - getattr(self, 'q_'+str(k+1)))/h_layer[k+1])*w_test*dx
                    if consider_vert_adv:
                        #l += -self.dt*dot(getattr(self, 'uv_av_'+str(k+1)), grad(getattr(self, 'w_'+str(k)+str(k+1))))*w_test*dx
                        l += self.dt*(div(getattr(self, 'uv_av_'+str(k+1))*w_test)*getattr(self, 'w_'+str(k)+str(k+1))*dx -
                             avg(getattr(self, 'w_'+str(k)+str(k+1)))*jump(w_test, inner(getattr(self, 'uv_av_'+str(k+1)), self.normal_2d))*dS)
                        if consider_mesh_relative_velocity:
                             l += -self.dt/h_layer[k+1]*inner(omega_dic['z_'+str(k+1)]*(w_dic['z_'+str(k+1)] - getattr(self, 'w_'+str(k)+str(k+1))) -
                                                              omega_dic['z_'+str(k)]*(w_dic['z_'+str(k)] - getattr(self, 'w_'+str(k)+str(k+1))), w_test)*dx
                    solve(a == l, getattr(self, 'w_'+str(k)+str(k+1)))

                # update water level elev_2d
                update_water_level = True
                if update_water_level:
                    timestepper_free_surface.advance(self.simulation_time, update_forcings)
                    self.fields.elev_2d.assign(self.elev_2d_old)





            # Move to next time step
            self.iteration += 1
            internal_iteration += 1
            self.simulation_time = initial_simulation_time + internal_iteration*self.dt

            self.callbacks.evaluate(mode='timestep')

            # Write the solution to file
            if self.simulation_time >= next_export_t - t_epsilon:
                self.i_export += 1
                next_export_t += self.options.simulation_export_time

                cputime = time_mod.clock() - cputimestamp
                cputimestamp = time_mod.clock()
                self.print_state(cputime)

                self.export()
                if export_func is not None:
                    export_func()
