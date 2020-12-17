"""
Module for 2D depth averaged solver
"""
from .utility import *
from . import granular_eq
from . import rungekutta
import weakref
import thetis.limiter as limiter
from .solver2d import FlowSolver2d


class FlowSolverCF(FlowSolver2d):
    """
    Main object for 2D depth averaged solver in conservative form

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
        super(FlowSolverCF, self).__init__(mesh2d, bathymetry_2d, options)

    def create_function_spaces(self):
        """
        Creates function spaces

        Function spaces are accessible via :attr:`.function_spaces`
        object.
        """
        self._isfrozen = False
        # ----- function spaces
        self.function_spaces.P0_2d = get_functionspace(self.mesh2d, 'DG', 0, name='P0_2d')
        self.function_spaces.P1_2d = get_functionspace(self.mesh2d, 'CG', 1, name='P1_2d')
        # function spaces with polynomial degree
        if self.options.element_family == 'dg-dg':
            self.function_spaces.U_2d = VectorFunctionSpace(self.mesh2d, 'DG', self.options.polynomial_degree, name='U_2d')
            self.function_spaces.H_2d = get_functionspace(self.mesh2d, 'DG', self.options.polynomial_degree, name='H_2d')
        else:
            raise Exception('Unsupported finite element family {:}'.format(self.options.element_family))
        # mixed function space
        self.function_spaces.V_2d = MixedFunctionSpace([self.function_spaces.H_2d, self.function_spaces.H_2d, self.function_spaces.H_2d])

        self.function_spaces.Q_2d = get_functionspace(self.mesh2d, 'DG', 1, name='Q_2d')

        self._isfrozen = True

    def create_equations(self):
        """
        Creates shallow water equations in conservative form
        """
        if not hasattr(self, 'H_2d'):
            self.create_function_spaces()
        self._isfrozen = False
        # ----- fields
        self.fields.solution_2d = Function(self.function_spaces.V_2d, name='solution_2d')
        self.fields.h_2d, self.fields.hu_2d, self.fields.hv_2d = self.fields.solution_2d.split()
        self.fields.elev_2d = Function(self.function_spaces.H_2d)
        self.fields.uv_2d = Function(self.function_spaces.U_2d)
        self.fields.h_elem_size_2d = Function(self.function_spaces.P1_2d)
        get_horizontal_elem_size_2d(self.fields.h_elem_size_2d)
        self.bathymetry_dg = Function(self.function_spaces.H_2d).project(self.fields.bathymetry_2d)

        # ----- Equations
        self.eq_sw = granular_eq.GranularEquations(
            self.fields.solution_2d.function_space(),
            self.fields.bathymetry_2d,
            self.options)
        self.eq_sw.bnd_functions = self.bnd_functions['shallow_water']

        self.options_nh = self.options.nh_model_options
        if self.options_nh.use_explicit_wetting_and_drying:
            self.wd_modification = treat_wetting_and_drying(self.function_spaces.H_2d)

        if self.options_nh.use_limiter_for_elevation or self.options_nh.use_limiter_for_momentum:
            self.limiter = limiter.VertexBasedP1DGLimiter(self.function_spaces.H_2d)
        else:
            self.limiter = None

        self._isfrozen = True

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

        self.set_time_step()

        # ----- Time integrators
        fields = {
            # 'uv_div': self.uv_div_ls,
            # 'strain_rate': self.strain_rate_ls,
            # 'fluid_pressure_gradient': self.grad_p_ls,
            # TODO add more
        }
        args = (self.eq_sw, self.fields.solution_2d, fields, self.dt, )
        kwargs = {'bnd_conditions': self.bnd_functions['shallow_water']}
        kwargs['solver_parameters'] = self.options.timestepper_options.solver_parameters
        self.timestepper = MyTimeIntegrator2D(weakref.proxy(self), *args, **kwargs)
        print_output('Using time integrator: {:}'.format(self.timestepper.__class__.__name__))

        self._isfrozen = True

    def assign_initial_conditions(self, elev=None, uv=None):
        """
        Assigns initial conditions

        :kwarg elev: Initial condition for water elevation
        :type elev: scalar :class:`Function`, :class:`Constant`, or an expression
        :kwarg uv: Initial condition for depth averaged velocity
        :type uv: vector valued :class:`Function`, :class:`Constant`, or an expression
        """
        if not self._initialized:
            self.initialize()
        if elev is not None:
            self.fields.elev_2d.project(elev)
        h_ini = self.fields.elev_2d + self.fields.bathymetry_2d
        self.fields.h_2d.interpolate(conditional(h_ini <= 0, 0, h_ini))
        if uv is not None:
            self.fields.hu_2d.project(self.fields.h_2d*uv[0])
            self.fields.hv_2d.project(self.fields.h_2d*uv[1])
        self.timestepper.initialize(self.fields.solution_2d)

    def print_state(self, cputime):
        """
        Print a summary of the model state on stdout

        :arg float cputime: Measured CPU time
        """
        norm_h = norm(self.fields.h_2d)
        norm_hu = norm(as_vector((self.fields.hu_2d, self.fields.hv_2d)))

        line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                'h norm: {h:10.4f} hu norm: {hu:10.4f} {cpu:5.2f}')
        print_output(line.format(iexp=self.i_export, i=self.iteration,
                                 t=self.simulation_time, h=norm_h,
                                 hu=norm_hu, cpu=cputime))
        sys.stdout.flush()


class MyTimeIntegrator2D(rungekutta.SSPRK33):
    def __init__(self, solver, *args, **kwargs):
        super(MyTimeIntegrator2D, self).__init__(*args, **kwargs)
        self.fields = solver.fields
        self.options_nh = solver.options_nh
        self.limiter = solver.limiter
        self.bathymetry_dg = solver.bathymetry_dg
        self.wd_modification = solver.wd_modification

    def advance(self, t, update_forcings=None):
        for i in range(self.n_stages):
            self.solve_stage(i, t, update_forcings)
            if self.options_nh.use_limiter_for_elevation:
                self.fields.elev_2d.assign(self.fields.h_2d - self.bathymetry_dg)
                self.limiter.apply(self.fields.elev_2d)
                self.fields.h_2d.assign(self.fields.elev_2d + self.bathymetry_dg)
            if self.options_nh.use_limiter_for_momentum:
                self.limiter.apply(self.fields.hu_2d)
                self.limiter.apply(self.fields.hv_2d)
            if self.options_nh.use_explicit_wetting_and_drying:
                self.wd_modification.apply(self.fields.solution_2d, self.options_nh.wetting_and_drying_threshold)
