"""
Generic time integration schemes to advance equations in time.
"""
from .utility import *
from abc import ABC, abstractmethod
import numpy
from pyop2.profiling import timed_region, timed_stage

CFL_UNCONDITIONALLY_STABLE = numpy.inf
# CFL coefficient for unconditionally stable methods


class TimeIntegratorBase(ABC):
    """
    Abstract class that defines the API for all time integrators

    Both :class:`TimeIntegrator` and :class:`CoupledTimeIntegrator` inherit
    from this class.
    """
    @abstractmethod
    def advance(self, t, update_forcings=None):
        """
        Advances equations for one time step

        :arg t: simulation time
        :type t: float
        :arg update_forcings: user-defined function that takes the simulation
            time and updates any time-dependent boundary conditions
        """
        pass

    @abstractmethod
    def initialize(self, init_solution):
        """
        Initialize the time integrator

        :arg init_solution: initial solution
        """
        pass


class TimeIntegrator(TimeIntegratorBase):
    """
    Base class for all time integrator objects that march a single equation
    """
    def __init__(self, equation, solution, fields, dt, options):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values
        """
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        self.solution = solution
        self.fields = fields
        self.dt = dt  # FIXME this might not be correctly updated ...
        self.dt_const = Constant(dt)

        # unique identifier for solver
        self.name = '-'.join([self.__class__.__name__,
                              self.equation.__class__.__name__])
        self.ad_block_tag = options.ad_block_tag or self.name
        self.solver_parameters = options.solver_parameters

    def set_dt(self, dt):
        """Update time step"""
        self.dt = dt
        self.dt_const.assign(dt)

    def advance_picard(self, t, update_forcings=None, update_lagged=True, update_fields=True):
        """
        Advances equations for one time step within a Picard iteration

        :arg t: simulation time
        :type t: float
        :arg update_forcings: user-defined function that takes the simulation
            time and updates any time-dependent boundary conditions
        :kwarg update_lagged: should the old solution be updated?
        :kwarg update_fields: should the fields be updated?
        """
        raise NotImplementedError(f"Picard iterations are not supported for {self} time integrators.")


class ForwardEuler(TimeIntegrator):
    """Standard forward Euler time integration scheme."""
    cfl_coeff = 1.0

    @PETSc.Log.EventDecorator("thetis.ForwardEuler.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        """
        super(ForwardEuler, self).__init__(equation, solution, fields, dt, options)
        self.solution_old = Function(self.equation.function_space)

        # create functions to hold the values of previous time step
        self.fields_old = {}
        for k in sorted(self.fields):
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space())
                elif isinstance(self.fields[k], Constant):
                    self.fields_old[k] = Constant(self.fields[k])

        u_old = self.solution_old
        u_tri = self.equation.trial
        self.A = self.equation.mass_term(u_tri)
        self.L = (self.equation.mass_term(u_old)
                  + self.dt_const*self.equation.residual('all', u_old, u_old, self.fields_old, self.fields_old, bnd_conditions)
                  )

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.ForwardEuler.update_solver")
    def update_solver(self):
        prob = LinearVariationalProblem(self.A, self.L, self.solution)
        self.solver = LinearVariationalSolver(prob, options_prefix=self.name,
                                              solver_parameters=self.solver_parameters,
                                              ad_block_tag=self.ad_block_tag)

    @PETSc.Log.EventDecorator("thetis.ForwardEuler.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    @PETSc.Log.EventDecorator("thetis.ForwardEuler.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solution_old.assign(self.solution)
        self.solver.solve()
        # shift time
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])


class CrankNicolson(TimeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE

    @PETSc.Log.EventDecorator("thetis.CrankNicolson.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        """
        super(CrankNicolson, self).__init__(equation, solution, fields, dt, options)
        theta = options.implicitness_theta
        semi_implicit = options.use_semi_implicit_linearization
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')
        self.solution_old = Function(self.equation.function_space, name='solution_old')
        # create functions to hold the values of previous time step
        # TODO is this necessary? is self.fields sufficient?
        self.fields_old = {}
        for k in sorted(self.fields):
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space(), name=self.fields[k].name()+'_old')
                elif isinstance(self.fields[k], Constant):
                    self.fields_old[k] = Constant(self.fields[k])

        u = self.solution
        u_old = self.solution_old
        if semi_implicit:
            # linearize around last timestep using the fact that all terms are
            # written in the form A(u_nl) u
            u_nl = u_old
        else:
            # solve the full nonlinear residual form
            u_nl = u
        bnd = bnd_conditions
        f = self.fields
        f_old = self.fields_old

        # Crank-Nicolson
        theta_const = Constant(theta)
        self.F = (self.equation.mass_term(u) - self.equation.mass_term(u_old)
                  - self.dt_const*(theta_const*self.equation.residual('all', u, u_nl, f, f, bnd)
                                   + (1-theta_const)*self.equation.residual('all', u_old, u_old, f_old, f_old, bnd))
                  )

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.CrankNicolson.update_solver")
    def update_solver(self):
        """Create solver objects"""
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name,
                                                 ad_block_tag=self.ad_block_tag)

    @PETSc.Log.EventDecorator("thetis.CrankNicolson.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    @PETSc.Log.EventDecorator("thetis.CrankNicolson.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solution_old.assign(self.solution)
        self.solver.solve()
        # shift time
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    @PETSc.Log.EventDecorator("thetis.CrankNicolson.advance_picard")
    def advance_picard(self, t, update_forcings=None, update_lagged=True, update_fields=True):
        """Advances equations for one time step in a Picard iteration."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        if update_lagged:
            self.solution_old.assign(self.solution)
        self.solver.solve()
        if update_fields:
            # shift time
            for k in sorted(self.fields_old):
                self.fields_old[k].assign(self.fields[k])


class SteadyState(TimeIntegrator):
    """
    Time integrator that solves the steady state equations, leaving out the
    mass terms
    """
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE

    @PETSc.Log.EventDecorator("thetis.SteadyState.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions):
        """
        :arg equation: the equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        """
        super(SteadyState, self).__init__(equation, solution, fields, dt, options)
        self.solver_parameters.setdefault('snes_type', 'newtonls')
        self.F = self.equation.residual('all', solution, solution, fields, fields, bnd_conditions)
        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.SteadyState.update_solver")
    def update_solver(self):
        """Create solver objects"""
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name,
                                                 ad_block_tag=self.ad_block_tag)

    @PETSc.Log.EventDecorator("thetis.SteadyState.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        # nothing to do here as the initial condition is passed in via solution
        return

    @PETSc.Log.EventDecorator("thetis.SteadyState.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solver.solve()


class PressureProjectionPicard(TimeIntegrator):
    """
    Pressure projection scheme with Picard iteration for shallow water
    equations

    """
    cfl_coeff = 1.0  # FIXME what is the right value?

    # TODO add more documentation
    @PETSc.Log.EventDecorator("thetis.PressureProjectionPicard.__init__")
    def __init__(self, equation, equation_mom, solution, fields, dt, options, bnd_conditions):
        """
        :arg equation: free surface equation
        :type equation: :class:`Equation` object
        :arg equation_mom: momentum equation
        :type equation_mom: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        """
        super(PressureProjectionPicard, self).__init__(equation, solution, fields, dt, options)
        theta = options.implicitness_theta
        semi_implicit = options.use_semi_implicit_linearization
        solver_parameters = options.solver_parameters_pressure
        solver_parameters_mom = options.solver_parameters_momentum
        iterations = options.picard_iterations

        self.equation_mom = equation_mom
        self.solver_parameters_mom = {}
        if solver_parameters_mom is not None:
            self.solver_parameters_mom.update(solver_parameters_mom)
        if semi_implicit:
            # solve a preliminary linearized momentum equation before
            # solving the linearized wave equation terms in a coupled system
            self.solver_parameters.setdefault('snes_type', 'ksponly')
            self.solver_parameters_mom.setdefault('snes_type', 'ksponly')
        else:
            # not sure this combination makes much sense: keep both systems nonlinear
            self.solver_parameters.setdefault('snes_type', 'newtonls')
            self.solver_parameters_mom.setdefault('snes_type', 'newtonls')
        self.iterations = iterations

        self.solution_old = Function(self.equation.function_space)
        if iterations > 1:
            self.solution_lagged = Function(self.equation.function_space)
        else:
            self.solution_lagged = self.solution_old
        uv_lagged, eta_lagged = self.solution_lagged.split()
        uv_old, eta_old = self.solution_old.split()

        if (solver_parameters['ksp_type'] == 'preonly'
                and 'fieldsplit_H_2d' in solver_parameters
                and solver_parameters['fieldsplit_H_2d']['ksp_type'] == 'preonly'
                and solver_parameters['fieldsplit_H_2d']['pc_python_type'] == 'thetis.AssembledSchurPC'
                and element_continuity(eta_old.function_space().ufl_element()).horizontal != 'cg'):
            # the default settings use AssembledSchurPC which assumes that the velocity block is only a dg mass matrix
            # Under these assumptions this preconditioner assembles the exact Schur complement. If this is not true, we either need
            # iterations with the unassembled Schur complement (fieldsplit_H_2d_ksp_type), or alternatively iterations outside
            # the fieldsplit to deal with the fact that we haven't solved the Schur complement exactly.
            # Currently only the dg-cg element pair, gives a pure DG  mass matrix velocity block: for dg-dg the pressure gradient adds a
            # Riemann term in the velocity block, for rt-dg the velocity block cannot be explicitly inverted either.
            raise Exception("The timestepper PressureProjectionPicard is only recommended in combination with the "
                            "dg-cg element_family. If you want to use it in combination with dg-dg or rt-dg you need to adjust the solver_parameters_pressure option.")

        # create functions to hold the values of previous time step
        self.fields_old = {}
        for k in sorted(self.fields):
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space())
                elif isinstance(self.fields[k], Constant):
                    self.fields_old[k] = Constant(self.fields[k])
        # for the mom. eqn. the 'eta' field is just one of the 'other' fields
        fields_mom = self.fields.copy()
        fields_mom_old = self.fields_old.copy()
        fields_mom['eta'] = eta_lagged
        fields_mom_old['eta'] = eta_old

        # the velocity solved for in the preliminary mom. solve:
        self.uv_star = Function(self.equation_mom.function_space)
        if semi_implicit:
            uv_star_nl = uv_lagged
            solution_nl = self.solution_lagged
        else:
            uv_star_nl = self.uv_star
            solution_nl = self.solution

        # form for mom. eqn.:
        theta_const = Constant(theta)
        self.F_mom = (
            self.equation_mom.mass_term(self.uv_star)-self.equation_mom.mass_term(uv_old)
            - self.dt_const*(
                theta_const*self.equation_mom.residual('all', self.uv_star, uv_star_nl, fields_mom, fields_mom, bnd_conditions)
                + (1-theta_const)*self.equation_mom.residual('all', uv_old, uv_old, fields_mom_old, fields_mom_old, bnd_conditions)
            )
        )

        # form for wave eqn. system:
        # M (u^n+1 - u^*) + G (eta^n+theta - eta_lagged) = 0
        # M (eta^n+1 - eta^n) + C (u^n+theta) = 0
        # the 'implicit' terms are the gradient (G) and divergence term (C) in the mom. and continuity eqn. resp.
        # where u^* is the velocity solved for in the mom. eqn., and G eta_lagged the gradient term in that eqn.
        uv_test, eta_test = split(self.equation.test)
        mass_term_star = inner(uv_test, self.uv_star)*dx + inner(eta_test, eta_old)*dx
        self.F = (
            self.equation.mass_term(self.solution) - mass_term_star
            - self.dt_const*(
                theta_const*self.equation.residual('implicit', self.solution, solution_nl, self.fields, self.fields, bnd_conditions)
                + (1-theta_const)*self.equation.residual('implicit', self.solution_old, self.solution_old, self.fields_old, self.fields_old, bnd_conditions)
            )
        )
        # subtract G eta_lagged: G is the implicit term in the mom. eqn.
        for key in self.equation_mom.terms:
            if self.equation_mom.labels[key] == 'implicit':
                term = self.equation.terms[key]
                self.F += -self.dt_const*(
                    - theta_const*term.residual(self.uv_star, eta_lagged, uv_star_nl, eta_lagged, self.fields, self.fields, bnd_conditions)
                    - (1-theta_const)*term.residual(uv_old, eta_old, uv_old, eta_old, self.fields_old, self.fields_old, bnd_conditions)
                )

        self.update_solver()

    @PETSc.Log.EventDecorator("thetis.PressureProjectionPicard.update_solver")
    def update_solver(self):
        """Create solver objects"""
        prob = NonlinearVariationalProblem(self.F_mom, self.uv_star)
        self.solver_mom = NonlinearVariationalSolver(prob,
                                                     solver_parameters=self.solver_parameters_mom,
                                                     options_prefix=self.name+'_mom',
                                                     ad_block_tag=self.ad_block_tag + '_mom')
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 appctx={'a': derivative(self.F, self.solution)},
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name,
                                                 ad_block_tag=self.ad_block_tag)

    @PETSc.Log.EventDecorator("thetis.PressureProjectionPicard.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        self.solution_lagged.assign(solution)
        # assign values to old functions
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])

    @PETSc.Log.EventDecorator("thetis.PressureProjectionPicard.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solution_old.assign(self.solution)

        for it in range(self.iterations):
            if self.iterations > 1:
                self.solution_lagged.assign(self.solution)
            with timed_stage("Momentum solve"):
                self.solver_mom.solve()
            with timed_stage("Pressure solve"):
                self.solver.solve()

        # shift time
        for k in sorted(self.fields_old):
            self.fields_old[k].assign(self.fields[k])


class LeapFrogAM3(TimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 ALE time integrator

    Defined in (2.27)-(2.30) in [1]; (2.21)-(2.22) in [2]

    [1] Shchepetkin and McWilliams (2005). The regional oceanic modeling system
    (ROMS): a split-explicit, free-surface, topography-following-coordinate
    oceanic model. Ocean Modelling, 9(4):347-404.
    http://dx.doi.org/10.1016/j.ocemod.2013.04.010

    [2] Shchepetkin and McWilliams (2009). Computational Kernel Algorithms for
    Fine-Scale, Multiprocess, Longtime Oceanic Simulations, 14:121-183.
    http://dx.doi.org/10.1016/S1570-8659(08)01202-0
    """
    cfl_coeff = 1.5874

    @PETSc.Log.EventDecorator("thetis.LeapFrogAM3.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(LeapFrogAM3, self).__init__(equation, solution, fields, dt, options)

        self.gamma = 1./12.
        self.gamma_const = Constant(self.gamma)

        fs = self.equation.function_space
        self.solution_old = Function(fs, name='old solution')
        self.msolution_old = Function(fs, name='dual solution')
        self.rhs_func = Function(fs, name='rhs linear form')

        # fully explicit evaluation
        self.a = self.equation.mass_term(self.equation.trial)
        self.l = self.dt_const*self.equation.residual(terms_to_add,
                                                      self.solution,
                                                      self.solution,
                                                      self.fields,
                                                      self.fields,
                                                      bnd_conditions)
        self.mass_new = inner(self.solution, self.equation.test)*dx
        self.mass_old = inner(self.solution_old, self.equation.test)*dx
        a = 0.5 - 2*self.gamma
        b = 0.5 + 2*self.gamma
        c = 1.0 - 2*self.gamma
        self.l_prediction = a*self.mass_old + b*self.mass_new + c*self.l

        self._nontrivial = self.l != 0

    @PETSc.Log.EventDecorator("thetis.LeapFrogAM3.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.mass_matrix = assemble(self.a)
        self.solution.assign(solution)
        self.solution_old.assign(solution)
        assemble(self.mass_new, self.msolution_old)
        self.lin_solver = LinearSolver(self.mass_matrix)
        # TODO: Linear solver is not annotated and does not accept ad_block_tag

    def _solve_system(self):
        """
        Solves system mass_matrix*solution = rhs_func

        If the function space is fully discontinuous, the mass matrix is
        inverted in place for efficiency.
        """
        self.lin_solver.solve(self.solution, self.rhs_func)

    @PETSc.Log.EventDecorator("thetis.LeapFrogAM3.predice")
    def predict(self):
        r"""
        Prediction step from :math:`t_{n-1/2}` to :math:`t_{n+1/2}`

        Let :math:`M_n` denote the mass matrix at time :math:`t_{n}`.
        The prediction step is

        .. math::
            T_{n-1/2} &= (1/2 - 2\gamma) T_{n-1} + (1/2 + 2 \gamma) T_{n} \\
            M_n T_{n+1/2} &= M_n T_{n-1/2} + \Delta t (1 - 2\gamma) M_n L_{n}

        This is computed in a fixed mesh: all terms are evaluated in
        :math:`\Omega_n`.
        """
        if self._nontrivial:
            with timed_region('lf_pre_asmb_sol'):
                assemble(self.mass_new, self.msolution_old)  # store current solution
            with timed_region('lf_pre_asmb_rhs'):
                assemble(self.l_prediction, self.rhs_func)
            with timed_region('lf_pre_asgn_sol'):
                self.solution_old.assign(self.solution)  # time shift
            with timed_region('lf_pre_solve'):
                self._solve_system()

    def eval_rhs(self):
        if self._nontrivial:
            with timed_region('lf_cor_asmb_rhs'):
                assemble(self.l, self.rhs_func)

    @PETSc.Log.EventDecorator("thetis.LeapFrogAM3.correct")
    def correct(self):
        r"""
        Correction step from :math:`t_{n}` to :math:`t_{n+1}`

        Let :math:`M_n` denote the mass matrix at time :math:`t_{n}`.
        The correction step is

        .. math::
            M_{n+1} T_{n+1} = M_{n} T_{n} + \Delta t L_{n+1/2}

        This is Euler ALE step: LHS is evaluated in :math:`\Omega_{n+1}`,
        RHS in :math:`\Omega_n`.
        """
        if self._nontrivial:
            # NOTE must call eval_rhs in the old mesh first
            with timed_region('lf_cor_incr_rhs'):
                self.rhs_func += self.msolution_old
            with timed_region('lf_cor_asmb_mat'):
                assemble(self.a, self.mass_matrix)
            with timed_region('lf_cor_solve'):
                self._solve_system()

    @PETSc.Log.EventDecorator("thetis.LeapFrogAM3.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.dt)
            self.predict()
            self.eval_rhs()
            self.correct()


class SSPRK22ALE(TimeIntegrator):
    r"""
    SSPRK(2,2) ALE time integrator for 3D fields

    The scheme is

    .. math::
        u^{(1)} &= u^{n} + \Delta t F(u^{n}) \\
        u^{n+1} &= u^{n} + \frac{\Delta t}{2}(F(u^{n}) +  F(u^{(1)}))

    Both stages are implemented as ALE updates from geometry :math:`\Omega_n`
    to :math:`\Omega_{(1)}`, and :math:`\Omega_{n+1}`.
    """
    cfl_coeff = 1.0

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.__init__")
    def __init__(self, equation, solution, fields, dt, options, bnd_conditions, terms_to_add='all'):
        """
        :arg equation: equation to solve
        :type equation: :class:`Equation` object
        :arg solution: :class:`Function` where solution will be stored
        :arg fields: Dictionary of fields that are passed to the equation
        :type fields: dict of :class:`Function` or :class:`Constant` objects
        :arg float dt: time step in seconds
        :arg options: :class:`TimeStepperOptions` instance containing parameter values.
        :arg dict bnd_conditions: Dictionary of boundary conditions passed to the equation
        :kwarg terms_to_add: Defines which terms of the equation are to be
            added to this solver. Default 'all' implies ['implicit', 'explicit', 'source'].
        :type terms_to_add: 'all' or list of 'implicit', 'explicit', 'source'.
        """
        super(SSPRK22ALE, self).__init__(equation, solution, fields, dt, options)

        fs = self.equation.function_space
        self.mu = Function(fs, name='dual solution')
        self.mu_old = Function(fs, name='dual solution')
        self.tendency = Function(fs, name='tendency')

        # fully explicit evaluation
        self.a = self.equation.mass_term(self.equation.trial)
        self.l = self.dt_const*self.equation.residual(terms_to_add,
                                                      self.solution,
                                                      self.solution,
                                                      self.fields,
                                                      self.fields,
                                                      bnd_conditions)
        self.mu_form = inner(self.solution, self.equation.test)*dx
        self._nontrivial = self.l != 0
        self._initialized = False

        self.n_stages = 2
        self.c = [0, 1]

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.initialize")
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution.assign(solution)

        mass_matrix = assemble(self.a)
        self.lin_solver = LinearSolver(mass_matrix,
                                       solver_parameters=self.solver_parameters)
        # TODO: Linear solver is not annotated and does not accept ad_block_tag
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.stage_one_prep")
    def stage_one_prep(self):
        """
        Preprocess first stage: compute all forms on the old geometry
        """
        if self._nontrivial:
            # Compute $Mu$ and assign $q_{old} = Mu$
            with timed_region('pre1_asseble_mu'):
                assemble(self.mu_form, self.mu_old)
            # Evaluate $k = \Delta t F(u)$
            with timed_region('pre1_asseble_f'):
                assemble(self.l, self.tendency)
            # $q = q_{old} + k$
            with timed_region('pre1_incr_rhs'):
                self.mu.assign(self.mu_old + self.tendency)

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.stage_one_solve")
    def stage_one_solve(self):
        r"""
        First stage: solve :math:`u^{(1)}` given previous solution :math:`u^n`.

        This is a forward Euler ALE step between domains :math:`\Omega^n` and :math:`\Omega^{(1)}`:

        .. math::

            \int_{\Omega^{(1)}} u^{(1)} \psi dx = \int_{\Omega^n} u^n \psi dx + \Delta t \int_{\Omega^n} F(u^n) \psi dx

        """
        if self._nontrivial:
            # Solve $u = M^{-1}q$
            with timed_region('sol1_assemble_A'):
                assemble(self.a, self.lin_solver.A)
            with timed_region('sol1_solve'):
                self.lin_solver.solve(self.solution, self.mu)

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.stage_two_prep")
    def stage_two_prep(self):
        """
        Preprocess 2nd stage: compute all forms on the old geometry
        """
        if self._nontrivial:
            # Evaluate $k = \Delta t F(u)$
            with timed_region('pre2_asseble_f'):
                assemble(self.l, self.tendency)
            # $q = \frac{1}{2}q + \frac{1}{2}q_{old} + \frac{1}{2}k$
            with timed_region('pre2_incr_rhs'):
                self.mu.assign(0.5*self.mu + 0.5*self.mu_old + 0.5*self.tendency)

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.stage_two_solve")
    def stage_two_solve(self):
        r"""
        2nd stage: solve :math:`u^{n+1}` given previous solutions :math:`u^n, u^{(1)}`.

        This is an ALE step:

        .. math::

            \int_{\Omega^{n+1}} u^{n+1} \psi dx &= \int_{\Omega^n} u^n \psi dx \\
                &+ \frac{\Delta t}{2} \int_{\Omega^n} F(u^n) \psi dx \\
                &+ \frac{\Delta t}{2} \int_{\Omega^{(1)}} F(u^{(1)}) \psi dx

        """
        if self._nontrivial:
            # Solve $u = M^{-1}q$
            with timed_region('sol2_assemble_A'):
                assemble(self.a, self.lin_solver.A)
            with timed_region('sol2_solve'):
                self.lin_solver.solve(self.solution, self.mu)

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.solve_stage")
    def solve_stage(self, i_stage):
        """Solves i-th stage"""
        if i_stage == 0:
            self.stage_one_solve()
        else:
            self.stage_two_solve()

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.prepare_stage")
    def prepare_stage(self, i_stage, t, update_forcings=None):
        """
        Preprocess stage i_stage.

        This must be called prior to updating mesh geometry.
        """
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        if i_stage == 0:
            self.stage_one_prep()
        else:
            self.stage_two_prep()

    @PETSc.Log.EventDecorator("thetis.SSPRK22ALE.advance")
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if not self._initialized:
            self.initialize(self.solution)
        for i_stage in range(self.n_stages):
            self.prepare_stage(i_stage, t, update_forcings)
            self.solve_stage(i_stage)
