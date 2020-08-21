"""
Module for 2D depth averaged solver
"""
from __future__ import absolute_import
from .utility_nh import *
from . import shallowwater_nh
from .. import timeintegrator
from .. import rungekutta
from .. import implicitexplicit
from .. import coupled_timeintegrator_2d
from .. import tracer_eq_2d
import weakref
import time as time_mod
from mpi4py import MPI
from .. import exporter
from ..field_defs import field_metadata
from ..options import ModelOptions2d
from .. import callback
from ..log import *
from collections import OrderedDict
from . import limiter_nh as limiter


class FlowSolver(FrozenClass):
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

        solver_obj = solver_ml.FlowSolver(mesh2d, bathymetry_2d)
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
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        self.function_spaces.P2_2d = get_functionspace(self.mesh2d, 'CG', 2, name='P2_2d')
        self.function_spaces.P1v_2d = VectorFunctionSpace(self.mesh2d, 'CG', 1, name='P1v_2d')
        self.function_spaces.P1DG_2d = get_functionspace(self.mesh2d, 'DG', 1, name='P1DG_2d')
        self.function_spaces.P1DGv_2d = VectorFunctionSpace(self.mesh2d, 'DG', 1, name='P1DGv_2d')
        # 2D velocity space
        if self.options.element_family == 'rt-dg':
            self.function_spaces.U_2d = get_functionspace(self.mesh2d, 'RT', self.options.polynomial_degree+1, name='U_2d')
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        elif self.options.element_family == 'dg-cg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'CG', self.options.polynomial_degree+1, name='H_2d')
        elif self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.U_2d, self.function_spaces.H_2d])

        self.function_spaces.Q_2d = get_functionspace(self.mesh2d, 'DG', 1, name='Q_2d')

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

        # functions for landslide modelling
        if self.options.landslide:
            self.bathymetry_ls = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)
            self.fields.solution_ls = Function(self.function_spaces.V_2d)
            self.fields.uv_ls = self.fields.solution_ls.sub(0)
            self.fields.elev_ls = self.fields.solution_ls.sub(1)
            self.fields.slide_source_2d = Function(self.function_spaces.H_2d)
            # new adds for rigid and granular landslides
            self.fields.h_ls = Function(self.function_spaces.H_2d)
            self.h_ls_old = Function(self.function_spaces.H_2d)

        # functions for multi-layer approach
        fs_q = get_functionspace(self.mesh2d, 'CG', self.options.polynomial_degree)
        self.fields.q_2d = Function(fs_q)
        q_fs_list = []
        for k in range(self.options.n_layers):
            q_fs_list.append(fs_q)
            setattr(self, 'uv_av_' + str(k+1), Function(self.function_spaces.U_2d))
            #self.__dict__['uv_av_' + str(k+1)] = Function(self.function_spaces.U_2d)
            #setattr(self, 'w_' + str(k+1), Function(self.function_spaces.H_2d))
            setattr(self, 'w_av_' + str(k+1), Function(self.function_spaces.H_2d))
            #setattr(self, 'q_' + str(k+1), Function(fs_q))
            #if k == 0:
            #    setattr(self, 'w_' + str(k), Function(self.function_spaces.H_2d))
            #    setattr(self, 'q_' + str(k), Function(fs_q))
            #    setattr(self, 'q_' + str(self.options.n_layers+1), Function(fs_q))
        self.function_spaces.q_mixed = MixedFunctionSpace(q_fs_list)
        self.q_mixed = Function(self.function_spaces.q_mixed)

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
        self.eq_sw = shallowwater_nh.ShallowWaterEquations(
            self.fields.solution_2d.function_space(),
            self.bathymetry_dg, #self.fields.bathymetry_2d,
            self.options
        )
        #self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']
        self.eq_uv_mom = shallowwater_nh.ShallowWaterMomentumEquation(
            TestFunction(self.function_spaces.U_2d),
            self.function_spaces.U_2d,
            self.function_spaces.H_2d,
            self.bathymetry_dg,
            self.options)
        self.eq_w_mom = shallowwater_nh.ShallowWaterMomentumEquation_vert(
            TestFunction(self.function_spaces.H_2d),
            self.function_spaces.H_2d,
            self.function_spaces.U_2d,
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

        # initialise limiter
        if self.options.polynomial_degree == 1:
            self.limiter_h = limiter.VertexBasedP1DGLimiter(self.function_spaces.H_2d)
            self.limiter_u = limiter.VertexBasedP1DGLimiter(self.function_spaces.U_2d)
        else:
            self.limiter_h = None
            self.limiter_u = None

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
            'uv_la': self.fields.uv_nh,
            'eta': self.fields.solution_2d.sub(1),
            'sponge_damping_2d': self.set_sponge_damping(self.options.sponge_layer_length, self.options.sponge_layer_start, alpha=10.),}
        if self.options.landslide:
            self.field_dic.update({'slide_source': self.fields.slide_source_2d,})
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
            self.eq_mom = shallowwater_nh.ShallowWaterMomentumEquation(
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

    def assign_initial_conditions(self, elev=None, uv=None, tracer=None, elev_slide=None, uv_slide=None, h_ls=None):
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

        # landslide
        if self.options.landslide:
            uv_ls, elev_ls = self.fields.solution_ls.split()
            if elev_slide is not None:
                elev_ls.project(elev_slide)
            if uv_slide is not None:
                uv_ls.project(uv_slide)
            if h_ls is not None:
                self.fields.h_ls.project(h_ls)

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

    def set_sponge_damping(self, length, x_start, y_start=None, alpha=10.):
        """
        Set damping terms to reduce the reflection on solid boundaries.
        """
        if length == [0., 0.]:
            return None
        damping_coeff = Function(self.function_spaces.P1_2d)
        mesh = damping_coeff.ufl_domain()
        xvector = mesh.coordinates.dat.data[:, 0]
        damp_vector = damping_coeff.dat.data[:]

        if mesh.coordinates.sub(0).dat.data.max() <= x_start[0] + length[0]:
            length[0] = xvector.max() - x_start[0]
            #if length[0] < 0:
                #print('Start point of the first sponge layer is out of computational domain!')
                #raise ValueError('Start point of the first sponge layer is out of computational domain!')
        if mesh.coordinates.sub(0).dat.data.max() <= x_start[1] + length[1]:
            length[1] = xvector.max() - x_start[1]
            #if length[1] < 0:
                #print('Start point of the second sponge layer is out of computational domain!')
                #raise ValueError('Start point of the second sponge layer is out of computational domain!')

        assert xvector.shape[0] == damp_vector.shape[0]
        for i, xy in enumerate(xvector):
            pi = 4*np.arctan(1.)
            x = (xy - x_start[0])/length[0]
            if y_start is not None:
                x = (xy[1] - y_start)/length[0]
            if x > 0 and x < 0.5:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
            elif x > 0.5 and x < 1.:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
            else:
                damp_vector[i] = 0.
        if length[1] == 0.:
            return damping_coeff
        for i, xy in enumerate(xvector):
            pi = 4*np.arctan(1.)
            x = (xy - x_start[1])/length[1]
            if y_start is not None:
                x = (xy[1] - y_start)/length[1]
            if x > 0 and x < 0.5:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(2.*x - 0.5))/(1. - (4.*x - 1.)**2)) + 1.)
            elif x > 0.5 and x < 1.:
                damp_vector[i] = alpha*0.25*(np.tanh(np.sin(pi*(1.5 - 2*x))/(1. - (3. - 4.*x)**2)) + 1.)
            else:
                damp_vector[i] = 0.
        return damping_coeff

    def get_alpha(self, H0):
        """
        An alternative to try alpha, finding minimum alpha to let all depths below the threshold wd_mindep.

        :arg H0: Minimum water depth
        """
        wd_mindep = self.options.wetting_and_drying_threshold     
        if H0 > 1.0E-5:
            return 0.
        elif not self.options.constant_mindep:
            return np.sqrt(0.25 * wd_mindep**2 - 0.5 * wd_mindep * H0) + 0.5 * wd_mindep # new formulated function, WPan
            #return np.sqrt(self.options.wd_mindep**2 - self.options.wd_mindep*H0) + self.options.wd_mindep # artificial porosity method
            #return np.sqrt(4*self.options.wd_mindep*(self.options.wd_mindep-H0)) # original bathymetry changed method
        else:
            return wd_mindep

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
        uta_2d, eta_2d = split(self.fields.solution_2d)
        # trial and test functions used to update
        uv_tri = TrialFunction(self.function_spaces.U_2d)
        uv_test = TestFunction(self.function_spaces.U_2d)
        w_tri = TrialFunction(self.function_spaces.H_2d)
        w_test = TestFunction(self.function_spaces.H_2d)
        uta_test, eta_test = TestFunctions(self.fields.solution_2d.function_space())

        # timestepper for free surface equation
        timestepper_free_surface = timeintegrator.CrankNicolson(self.eq_free_surface, self.elev_2d_old,
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['shallow_water'],
                                                              #solver_parameters=solver_parameters,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)

        # layer thickness accounting for total depth
        n_layers = self.options.n_layers
        alpha = self.options.alpha_nh
        if len(alpha) < n_layers:
            n = n_layers - len(alpha)
            sum = 0.
            if len(alpha) >= 1:
                for k in range(len(alpha)):
                    sum = sum + alpha[k]
            if sum >= 1.:
                print_output('Using uniform vertical layers due to improper option set ...')
                alpha = [1./n_layers for i in range(n_layers)]
            else:
                for k in range(n):
                    alpha.append((1. - sum)/n)
        else:
            sum = 0.
            for i in range(n_layers):
                sum += alpha[i]
            if not sum == 1:
                print_output('Using uniform vertical layers due to improper option set ...')
                alpha = [1./n_layers for i in range(n_layers)]
        # location parameter of layer interface
        beta = [0.]
        for k in range(n_layers):
            val = beta[k] + alpha[k]
            beta.append(val)

        # list the layer-averaged velocities
        u_list = []
        w_list = []
        q_list = []
        for k in range(n_layers):
            u_list.append(getattr(self, 'uv_av_'+str(k+1)))
            w_list.append(getattr(self, 'w_av_'+str(k+1)))
            q_list.append(self.q_mixed.split()[k])
            if k == n_layers - 1:
                q_list.append(0.)

        # store fields for convenience
        h_tot = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(self.fields.elev_2d)
        fields = self.field_dic
        elev_dt = Function(self.function_spaces.H_2d)
        bath_dt = 0#Function(self.function_spaces.H_2d)
        for k in range(n_layers + 1):
            # velocities at layer interface
            if k == 0:
                if n_layers == 1:
                    fields['u_z_'+str(k)] = u_list[k]
                else:
                    fields['u_z_'+str(k)] = 2.*u_list[k] - (alpha[k+1]/(alpha[k]+alpha[k+1])*u_list[k] + 
                                                            alpha[k]/(alpha[k]+alpha[k+1])*u_list[k+1])
                fields['w_z_'+str(k)] = -inner(fields['u_z_'+str(k)], grad(self.bathymetry_dg))
                if self.options.landslide:
                    fields['w_z_'+str(k)] += -self.fields.slide_source_2d # (self.bathymetry_dg - self.bathymetry_dg_old)/self.dt
            elif k > 0 and k < n_layers:
                fields['u_z_'+str(k)] = alpha[k]/(alpha[k-1]+alpha[k])*u_list[k-1] + alpha[k-1]/(alpha[k-1]+alpha[k])*u_list[k]
                fields['w_z_'+str(k)] = 2.*w_list[k-1] - fields['w_z_'+str(k-1)]
            else:
                fields['u_z_'+str(k)] = 2.*u_list[k-1] - fields['u_z_'+str(k-1)]
                fields['w_z_'+str(k)] = 2.*w_list[k-1] - fields['w_z_'+str(k-1)]
            # z-coordinate
            fields['z_'+str(k)] = beta[k]*elev_2d + (beta[k] - 1)*self.bathymetry_dg
            if k > 0:
                fields['z_'+str(k-1)+str(k)] = (beta[k-1] + beta[k])*elev_2d + (beta[k-1] + beta[k] - 2)*self.bathymetry_dg
                fields['omega_'+str(k-1)] = w_list[k-1] - 0.5*(dot(fields['u_z_'+str(k-1)], grad(fields['z_'+str(k-1)])) + beta[k-1]*elev_dt + (beta[k-1] -1)*bath_dt +
                                                               dot(fields['u_z_'+str(k)], grad(fields['z_'+str(k)])) + beta[k]*elev_dt + (beta[k] - 1)*bath_dt)

        if n_layers >= 2:
            timestepper_dic = {}
            for k in range(n_layers - 1):
                timestepper_dic['uv_layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_uv_mom, u_list[k],
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)
                consider_vertical_mass_flux = False
                if consider_vertical_mass_flux:
                    uv_mass_term = fields['omega_'+str(k)]*(fields['u_z_'+str(k+1)] - fields['u_z_'+str(k)])/(alpha[k]*h_tot)
                    timestepper_dic['uv_layer_'+str(k+1)].F += self.dt*inner(uv_mass_term, uv_test)*dx
                    timestepper_dic['uv_layer_'+str(k+1)].update_solver()

            if True:
                if True:
                    solve_vertical_velocity = False
                    if n_layers >= 2 and solve_vertical_velocity:
                        for k in range(n_layers):
                            timestepper_dic['w_layer_'+str(k+1)] = timeintegrator.CrankNicolson(self.eq_w_mom, w_list[k],
                                                              self.field_dic, self.dt,
                                                              bnd_conditions=self.bnd_functions['momentum'],
                                                              solver_parameters=self.options.timestepper_options.solver_parameters_momentum,
                                                              semi_implicit=self.options.timestepper_options.use_semi_implicit_linearization,
                                                              theta=self.options.timestepper_options.implicitness_theta)
                            if consider_vertical_mass_flux:
                                w_mass_term = fields['omega_'+str(k)]*(fields['w_z_'+str(k+1)] - fields['w_z_'+str(k)])/(alpha[k]*h_tot)
                                timestepper_dic['w_layer_'+str(k+1)].F += self.dt*w_mass_term*w_test*dx
                                timestepper_dic['w_layer_'+str(k+1)].update_solver()

        # solver for the mixed Poisson equation
        if True:
            if True:
                if True:
                    q_test = TestFunctions(self.function_spaces.q_mixed)
                    q_tuple = split(self.q_mixed)
                    if n_layers == 1:
                        q_test = [TestFunction(self.fields.q_2d.function_space())]
                        q_tuple = [self.fields.q_2d]
                    # re-arrange the list of q
                    q = []
                    for k in range(n_layers):
                        q.append(q_tuple[k])
                        if k == n_layers - 1:
                            # free-surface NH pressure
                            q.append(0.)
                    f = 0.
                    for k in range(n_layers):
                        # weak form of div(h_{k+1}*uv_av_{k+1})
                        div_hu_term = div((alpha[k]*h_tot)*u_list[k])*q_test[k]*dx + \
                                      0.5*self.dt*(alpha[k]*h_tot)*dot(grad(q[k]+q[k+1]), grad(q_test[k]))*dx + \
                                      0.5*self.dt*(q[k]-q[k+1])*dot(grad(fields['z_'+str(k)+str(k+1)]), grad(q_test[k]))*dx
                        if k >= 1:
                            for i in range(k):
                                div_hu_term += 2.*(div((alpha[i]*h_tot)*u_list[i])*q_test[k]*dx + \
                                               0.5*self.dt*(alpha[i]*h_tot)*dot(grad(q[i]+q[i+1]), grad(q_test[k]))*dx + \
                                               0.5*self.dt*(q[i]-q[i+1])*dot(grad(fields['z_'+str(i)+str(i+1)]), grad(q_test[k]))*dx)
                        # weak form of w_{k}{k+1}
                        vert_vel_term = 2.*(w_list[k] + self.dt*(q[k] - q[k+1])/(alpha[k]*h_tot))*q_test[k]*dx
                        consider_vert_adv = False#True
                        if consider_vert_adv: # TODO if make sure that considering vertical advection is benefitial, delete this logical variable
                            #vert_vel_term += -2.*self.dt*dot(u_list[k], grad(w_list[k]))*q_test[k]*dx
                            vert_vel_term += 2.*self.dt*(div(u_list[k]*q_test[k])*w_list[k]*dx -
                                                         avg(w_list[k])*jump(q_test[k], inner(u_list[k], self.normal_2d))*dS)
                            if consider_mesh_relative_velocity:
                                vert_vel_term += -2.*self.dt/(alpha[k]*h_tot)*inner(omega_dic['z_'+str(k+1)]*(fields['w_z_'+str(k+1)] - w_list[k]) -
                                                                                omega_dic['z_'+str(k)]*(fields['w_z_'+str(k)] - w_list[k]), q_test[k])*dx
                        # weak form of RHS terms
                        if k == 0: # i.e. the layer adjacent to the bottom
                            if n_layers == 1:
                                grad_1_layer1 = grad(fields['z_'+str(k)+str(k+1)])
                                interface_term = dot(grad_1_layer1, u_list[k])*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]))*dx - \
                                                 0.5*self.dt*(1./(alpha[k]*h_tot)*dot(grad_1_layer1, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                            else:
                                grad_1_layer1 = grad(2.*fields['z_'+str(k)] + alpha[k]*alpha[k+1]/(alpha[k] + alpha[k+1])*h_tot)
                                grad_2_layer1 = grad(alpha[k]*alpha[k]/(alpha[k] + alpha[k+1])*h_tot)
                                interface_term = (dot(grad_1_layer1, u_list[k]) + dot(grad_2_layer1, u_list[k+1]))*q_test[k]*dx - \
                                                 0.5*self.dt*(-div(grad_1_layer1*q_test[k])*(q[k]+q[k+1]) - div(grad_2_layer1*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                                 0.5*self.dt*(1./(alpha[k]*h_tot)*dot(grad_1_layer1, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./(alpha[k+1]*h_tot)*dot(grad_2_layer1, grad(fields['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        elif k == n_layers - 1: # i.e. the layer adjacent to the free surface
                            grad_1_layern = grad(-alpha[k]*alpha[k]/(alpha[k-1] + alpha[k])*h_tot)
                            grad_2_layern = grad(2.*fields['z_'+str(k+1)] - alpha[k-1]*alpha[k]/(alpha[k-1] + alpha[k])*h_tot)
                            interface_term = (dot(grad_1_layern, u_list[k-1]) + dot(grad_2_layern, u_list[k]))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layern*q_test[k])*(q[k-1]+q[k]) - div(grad_2_layern*q_test[k])*(q[k]+q[k+1]))*dx - \
                                             0.5*self.dt*(1./(alpha[k-1]*h_tot)*dot(grad_1_layern, grad(fields['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                      1./(alpha[k]*h_tot)*dot(grad_2_layern, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]))*q_test[k]*dx
                        else:
                            grad_1_layerk = alpha[k]/(alpha[k-1] + alpha[k])*grad(fields['z_'+str(k)])
                            grad_2_layerk = alpha[k-1]/(alpha[k-1] + alpha[k])*grad(fields['z_'+str(k)]) + \
                                            alpha[k+1]/(alpha[k] + alpha[k+1])*grad(fields['z_'+str(k+1)])
                            grad_3_layerk = alpha[k]/(alpha[k] + alpha[k+1])*grad(fields['z_'+str(k+1)])
                            interface_term = (dot(grad_1_layerk, u_list[k-1]) + 
                                              dot(grad_2_layerk, u_list[k]) + 
                                              dot(grad_3_layerk, u_list[k+1]))*q_test[k]*dx - \
                                             0.5*self.dt*(-div(grad_1_layerk*q_test[k])*(q[k-1]+q[k]) - 
                                                          div(grad_2_layerk*q_test[k])*(q[k]+q[k+1]) - 
                                                          div(grad_3_layerk*q_test[k])*(q[k+1]+q[k+2]))*dx - \
                                             0.5*self.dt*(1./(alpha[k-1]*h_tot)*dot(grad_1_layerk, grad(fields['z_'+str(k-1)+str(k)]))*(q[k-1]-q[k]) + 
                                                          1./(alpha[k]*h_tot)*dot(grad_2_layerk, grad(fields['z_'+str(k)+str(k+1)]))*(q[k]-q[k+1]) + 
                                                          1./(alpha[k+1]*h_tot)*dot(grad_3_layerk, grad(fields['z_'+str(k+1)+str(k+2)]))*(q[k+1]-q[k+2]))*q_test[k]*dx
                        # weak form of slide source term
                        if self.options.landslide:
                            slide_source_term = -2.*self.fields.slide_source_2d*q_test[k]*dx
                            f += slide_source_term
                        f += div_hu_term + vert_vel_term - interface_term

                        if k == 0:
                            bcs = []
                        for bnd_marker in self.boundary_markers:
                            func = self.bnd_functions['shallow_water'].get(bnd_marker)
                            ds_bnd = ds(int(bnd_marker))
                            if func is not None and n_layers == 1: # inflow bnd #TODO set more general conditional statement
                                bc = DirichletBC(self.fields.q_2d.function_space(), 0., int(bnd_marker))
                                bcs.append(bc)
                            if func is not None and n_layers > 1: # inflow bnd #TODO set more general conditional statement
                                bc = DirichletBC(self.q_mixed.function_space().sub(k), Constant(0.), int(bnd_marker))
                                bcs.append(bc)
                            if self.bnd_functions['shallow_water'] == {}:#func is None or 'q' not in func:
                                # bnd terms of div(h_{k+1}*uv_av_{k+1})
                                f += -0.5*self.dt*(q[k]-q[k+1])*dot(grad(fields['z_'+str(k)+str(k+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                if k >= 1:
                                    for i in range(k):
                                        f += -self.dt*(q[i]-q[i+1])*dot(grad(fields['z_'+str(i)+str(i+1)]), self.normal_2d)*q_test[k]*ds_bnd
                                # bnd terms of RHS terms about interface connection
                                if k == 0:
                                    if n_layers == 1:
                                        f += 0.5*self.dt*dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1])*q_test[k]*ds_bnd
                                    else:
                                        f += 0.5*self.dt*(dot(grad_1_layer1, self.normal_2d)*(q[k]+q[k+1]) + 
                                                          dot(grad_2_layer1, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd
                                elif k == n_layers - 1:
                                    f += 0.5*self.dt*(dot(grad_1_layern, self.normal_2d)*(q[k-1]+q[k]) + 
                                                      dot(grad_2_layern, self.normal_2d)*(q[k]+q[k+1]))*q_test[k]*ds_bnd
                                else:
                                    f += 0.5*self.dt*(dot(grad_1_layerk, self.normal_2d)*(q[k-1]+q[k]) +
                                                      dot(grad_2_layerk, self.normal_2d)*(q[k]+q[k+1]) +
                                                      dot(grad_3_layerk, self.normal_2d)*(q[k+1]+q[k+2]))*q_test[k]*ds_bnd

                    prob_q = NonlinearVariationalProblem(f, self.q_mixed, bcs=bcs)
                    if n_layers == 1:
                        prob_q = NonlinearVariationalProblem(f, self.fields.q_2d, bcs=bcs)
                    solver_q = NonlinearVariationalSolver(prob_q,
                                                          solver_parameters={'snes_type': 'ksponly', # ksponly, newtonls
                                                               'ksp_type': 'preonly', # gmres, preonly
                                                               'mat_type': 'aij',
                                                               'pc_type': 'lu'})

        # solvers for updating layer-averaged velocities
        solver_u = []
        solver_w = []
        a_u = inner(uv_tri, uv_test)*dx
        a_w = inner(w_tri, w_test)*dx
        for k in range(n_layers):
            l_u = dot(u_list[k] - 0.5*self.dt*(grad(q_list[k] + q_list[k+1]) - 
                                               (q_list[k+1] - q_list[k])/(alpha[k]*h_tot)*grad(fields['z_'+str(k)+str(k+1)])), uv_test)*dx
            l_w = (w_list[k] - self.dt*(q_list[k+1] - q_list[k])/(alpha[k]*h_tot))*w_test*dx
            prob_u = LinearVariationalProblem(a_u, l_u, u_list[k])
            prob_w = LinearVariationalProblem(a_w, l_w, w_list[k])
            solver_u.append(LinearVariationalSolver(prob_u))
            solver_w.append(LinearVariationalSolver(prob_w))

        while self.simulation_time <= self.options.simulation_end_time - t_epsilon:
            #self.timestepper.advance(self.simulation_time, update_forcings)
            self.uv_2d_old.assign(self.fields.uv_2d)
            self.elev_2d_old.assign(self.fields.elev_2d)
            self.elev_2d_mid.assign(self.fields.elev_2d)
            self.bathymetry_dg_old.assign(self.bathymetry_dg)

            if self.options.landslide:
                self.h_ls_old.assign(self.fields.h_ls)
                # update landslide motion source
                if update_forcings is not None:
                    update_forcings(self.simulation_time + self.dt) # update h_ls
                    self.fields.slide_source_2d.assign((self.fields.h_ls - self.h_ls_old)/self.dt)
                    self.bathymetry_dg.project(self.fields.bathymetry_2d - self.fields.h_ls)

            hydrostatic_solver_2d = False
            # --- Hydrostatic solver ---
            if hydrostatic_solver_2d:
                self.timestepper.advance(self.simulation_time, update_forcings)
            else: #arbitrary_multi_layer_NH_solver: # i.e. multi-layer NH model
                # parameter for wetting and drying
               # H_min = (self.bathymetry_dg.dat.data + self.fields.elev_2d.dat.data).min()
               # self.options.depth_wd_interface.assign(self.get_alpha(H_min))

                # solve 2D depth-integrated equations initially
                if self.options.landslide: # save time
                    self.timestepper.advance(self.simulation_time) # TODO modify to avoid necessary update_forcings not added
                else:
                    self.timestepper.advance(self.simulation_time, update_forcings)

                if self.options.use_limiter_for_elevation:
                    if self.limiter_h is not None:
                        self.limiter_h.apply(self.fields.elev_2d)
                    if self.limiter_u is not None:
                        self.limiter_u.apply(self.fields.uv_2d)

                if n_layers >= 2:
                    # solve layer-averaged horizontal velocity in the hydrostatic step
                    sum_uv_av = 0. 
                    # except the layer adjacent to the free surface
                    for k in range(n_layers - 1):
                        if self.options.landslide: # save time
                            timestepper_dic['uv_layer_'+str(k+1)].advance(self.simulation_time) #TODO modify to avoid necessary update_forcings not added
                        else:
                            timestepper_dic['uv_layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                        if self.options.use_limiter_for_multi_layer:
                            if self.limiter_u is not None:
                                self.limiter_u.apply(u_list[k])
                        #sum_uv_av += u_list[k] # cannot sum by this way
                        sum_uv_av = sum_uv_av + alpha[k]*u_list[k]
                    u_list[n_layers-1].project((uv_2d - sum_uv_av)/alpha[n_layers-1])
                    # solve layer-averaged vertical velocity in the hydrostatic step
                    #for k in range(n_layers):
                    #    self.fields.uv_nh.assign(u_list[k])
                    #    timestepper_dic['w_layer_'+str(k+1)].advance(self.simulation_time, update_forcings)
                else:
                    self.uv_av_1.assign(uv_2d)

                # solve non-hydrostatic pressure q
                solver_q.solve()
                if n_layers == 1:
                    q_list[0].assign(self.fields.q_2d)

                # update layer-averaged velocities
                sum_uv_av = 0.
                for k in range(n_layers):
                    solver_u[k].solve()
                    solver_w[k].solve()
                    sum_uv_av = sum_uv_av + alpha[k]*u_list[k]
                self.fields.uv_2d.project(sum_uv_av)

                if self.options.use_limiter_for_elevation:
                    if self.limiter_u is not None:
                        self.limiter_u.apply(self.fields.uv_2d)

                # update water level elev_2d
                solving_free_surface_eq = True
                if n_layers == 1:
                    solving_free_surface_eq = False
                if self.simulation_time <= t_epsilon and self.options.update_free_surface and not solving_free_surface_eq:
                    # update layer thickness and z-coordinate
                    h_tot_spl = shallowwater_nh.ShallowWaterTerm(self.fields.solution_2d.function_space(), self.bathymetry_dg, self.options).get_total_depth(eta_2d)
                    z_dic_spl = {}
                    for k in range(n_layers + 1):
                        z_dic_spl['z_'+str(k)] = beta[k]*eta_2d + (beta[k] - 1)*self.bathymetry_dg
                    for k in range(n_layers):
                        self.timestepper.F += self.dt/h_tot_spl*inner(grad((q_list[k] + q_list[k+1])/2.*(alpha[k]*h_tot_spl)), uta_test)*dx
                        if k == n_layers - 1:
                            self.timestepper.F += self.dt/h_tot_spl*inner(q_list[0]*grad(z_dic_spl['z_'+str(0)]) - 
                                                                          q_list[k+1]*grad(z_dic_spl['z_'+str(k+1)]), uta_test)*dx
                    prob_n_layers_int = NonlinearVariationalProblem(self.timestepper.F, self.fields.solution_2d)
                    solver_n_layers_int = NonlinearVariationalSolver(prob_n_layers_int,
                                                                     solver_parameters=self.options.timestepper_options.solver_parameters)
                if self.options.update_free_surface:
                    if not solving_free_surface_eq:
                        solver_n_layers_int.solve()
                        #uv_2d.assign(self.uv_2d_mid)
                    else:
                        if self.options.landslide: # save time
                            timestepper_free_surface.advance(self.simulation_time) # TODO modify to avoid necessary update_forcings not added
                        else:
                            timestepper_free_surface.advance(self.simulation_time, update_forcings)
                        self.fields.elev_2d.assign(self.elev_2d_old)

                    if self.options.use_limiter_for_elevation:
                        if self.limiter_h is not None:
                            self.limiter_h.apply(self.fields.elev_2d)

                if self.options.set_vertical_2d:
                    self.uv_2d_dg.project(uv_2d)
                    self.uv_2d_dg.sub(1).assign(0.)
                    uv_2d.project(self.uv_2d_dg)
                    for k in range(n_layers - 1):
                        self.uv_2d_dg.project(u_list[k])
                        self.uv_2d_dg.sub(1).assign(0.)
                        u_list[k].project(self.uv_2d_dg)

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

