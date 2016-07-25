"""
Generic time integration schemes to advance equations in time.

Tuomas Karna 2015-03-27
"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty


class TimeIntegratorBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        pass

    @abstractproperty
    def initialize(self, init_solution):
        """Initializes the time integrator"""
        pass


class TimeIntegrator(TimeIntegratorBase):
    """Base class for all time integrator objects."""
    def __init__(self, equation, solution, fields, dt, solver_parameters={}):
        """Assigns initial conditions to all required fields."""
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        self.solution = solution
        self.fields = fields
        self.dt = dt
        self.dt_const = Constant(dt)

        # unique identifier for solver
        self.name = '-'.join([self.__class__.__name__,
                              self.equation.__class__.__name__])
        self.solver_parameters = {}
        self.solver_parameters.update(solver_parameters)


class ForwardEuler(TimeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEuler, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.solution_old = Function(self.equation.function_space)

        # create functions to hold the values of previous time step
        self.fields_old = {}
        for k in self.fields:
            if self.fields[k] is not None:
                if isinstance(self.fields[k], FiredrakeFunction):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space())
                elif isinstance(self.fields[k], FiredrakeConstant):
                    self.fields_old[k] = Constant(self.fields[k])

        u_old = self.solution_old
        u_tri = self.equation.trial
        self.A = self.equation.mass_term(u_tri)
        self.L = (self.equation.mass_term(u_old) +
                  self.dt_const*self.equation.residual('all', u_old, u_old, self.fields_old, self.fields_old, bnd_conditions)
                  )

        self.update_solver()

    def update_solver(self):
        prob = LinearVariationalProblem(self.A, self.L, self.solution)
        self.solver = LinearVariationalSolver(prob, options_prefix=self.name,
                                              solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solution_old.assign(self.solution)
        self.solver.solve()
        # shift time
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])


class CrankNicolson(TimeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, theta=0.5, semi_implicit=False):
        """Creates forms for the time integrator"""
        super(CrankNicolson, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')
        self.solution_old = Function(self.equation.function_space, name='solution_old')
        # create functions to hold the values of previous time step
        # TODO is this necessary? is self.fields sufficient?
        self.fields_old = {}
        for k in self.fields:
            if self.fields[k] is not None:
                if isinstance(self.fields[k], FiredrakeFunction):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space(), name=self.fields[k].name()+'_old')
                elif isinstance(self.fields[k], FiredrakeConstant):
                    self.fields_old[k] = Constant(self.fields[k])

        u = self.solution
        u_old = self.solution_old
        if semi_implicit:
            # linearize around last timestep using the fact that all terms are written in the form A(u_nl) u
            # (currently only true for the SWE)
            u_nl = u_old
        else:
            # solve the full nonlinear residual form
            u_nl = u
        bnd = bnd_conditions
        f = self.fields
        f_old = self.fields_old

        # Crank-Nicolson
        theta_const = Constant(theta)
        # FIXME this is consistent with previous implementation but time levels are incorrect
        self.F = (self.equation.mass_term(u) - self.equation.mass_term(u_old) -
                  self.dt_const*(theta_const*self.equation.residual('all', u, u_nl, f, f, bnd) +
                                 (1-theta_const)*self.equation.residual('all', u_old, u_old, f_old, f_old, bnd)
                                 )
                  )

        self.update_solver()

    def update_solver(self):
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solution_old.assign(self.solution)
        self.solver.solve()
        # shift time
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])


class SteadyState(TimeIntegrator):
    """Time integrator that solves the steady state equations, leaving out the mass terms"""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(SteadyState, self).__init__(equation, solution, fields, dt, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')
        self.F = self.equation.residual('all', solution, solution, fields, fields, bnd_conditions)
        self.update_solver()

    def update_solver(self):
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        # nothing to do here as the initial condition is passed in via solution
        return

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.solver.solve()


class PressureProjectionPicard(TimeIntegrator):
    """Pressure projection scheme with Picard iteration."""
    def __init__(self, equation, equation_mom, solution, fields, dt, bnd_conditions=None, solver_parameters={}, solver_parameters_mom={},
                 theta=0.5, semi_implicit=False, iterations=2):
        """Creates forms for the time integrator"""
        super(PressureProjectionPicard, self).__init__(equation, solver_parameters)
        self.equation_mom = equation_mom
        self.solver_parameters_mom = solver_parameters_mom
        if semi_implicit:
            # solve a preliminary linearized momentum equation before
            # solving the linearized wave equation terms in a coupled system
            self.solver_parameters.setdefault('snes_type', 'ksponly')
            self.solver_parameters_mom.setdefault('snes_type', 'ksponly')
        else:
            # not sure this combination makes much sense: keep both systems nonlinear
            self.solver_parameters.setdefault('snes_type', 'newtonls')
            self.solver_parameters_mom.setdefault('snes_type', 'newtonls')
        # number of picard iterations
        self.iterations = iterations

        self.dt_const = Constant(dt)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space)
        if iterations > 1:
            self.solution_lagged = Function(self.equation.function_space)
        else:
            self.solution_lagged = self.solution_old
        uv_lagged, eta_lagged = self.solution_lagged.split()
        uv_old, eta_old = self.solution_old.split()

        self.fields = fields
        # create functions to hold the values of previous time step
        self.fields_old = {}
        for k in self.fields:
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
            self.equation_mom.mass_term(self.uv_star)-self.equation_mom.mass_term(uv_old) -
            self.dt_const*(
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
            self.equation.mass_term(self.solution) - mass_term_star -
            self.dt_const*(
                theta_const*self.equation.residual('implicit', self.solution, solution_nl, self.fields, self.fields, bnd_conditions)
                + (1-theta_const)*self.equation.residual('implicit', self.solution_old, self.solution_old, self.fields_old, self.fields_old, bnd_conditions)
            )
        )
        # subtract G eta_lagged: G is the implicit term in the mom. eqn.
        for key in self.equation_mom.terms:
            if self.equation_mom.labels[key] == 'implicit':
                self.F += -self.dt_const*(
                    - theta_const*self.equation.terms[key].residual(self.uv_star, eta_lagged, uv_star_nl, eta_lagged, self.fields, self.fields, bnd_conditions)
                    - (1-theta_const)*self.equation.terms[key].residual(uv_old, eta_old, uv_old, eta_old, self.fields_old, self.fields_old, bnd_conditions)
                )

        self.update_solver()

    def update_solver(self):
        prob = NonlinearVariationalProblem(self.F_mom, self.uv_star)
        self.solver_mom = NonlinearVariationalSolver(prob,
                                                     solver_parameters=self.solver_parameters_mom,
                                                     options_prefix=self.name+'_mom')
        # Ensure LU assembles monolithic matrices
        if self.solver_parameters.get('pc_type') == 'lu':
            self.solver_parameters['mat_type'] = 'aij'
        prob = NonlinearVariationalProblem(self.F, self.solution)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        self.solution_lagged.assign(solution)
        # assign values to old functions
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])

    def advance(self, t, dt, solution, updateForcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)

        for it in range(self.iterations):
            if self.iterations > 1:
                self.solution_lagged.assign(solution)
            self.solver_mom.solve()
            self.solver.solve()

        # shift time
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])


class LeapFrogAM3(TimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 ALE time integrator

    Defined in Shchepetkin (2005) and Karna (2013)
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}, terms_to_add='all'):
        super(LeapFrogAM3, self).__init__(equation, solution, fields, dt, solver_parameters)

        self.gamma = 1./12.
        self.gamma_const = Constant(self.gamma)

        fs = self.equation.function_space
        self.msol = Function(fs, name='dual solution')
        self.msol_old = Function(fs, name='old dual solution')
        self.l_form = Function(fs, name='rhs linear form')

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual(terms_to_add, self.solution, self.solution, self.fields, self.fields, bnd_conditions)
        self.mass_term = inner(self.solution, self.equation.test)*dx

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        self.mass_matrix = assemble(self.a_rk)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution.assign(solution)
        assemble(self.mass_term, self.msol)
        self.msol_old.assign(self.msol)
        assemble(self.l_rk, self.l_form)

    def eval_rhs(self):
        assemble(self.l_rk, self.l_form)

    def predict(self):
        """
        Prediction step from t_{n-1/2} to t_{n+1/2}

        MT_{n-1/2} = (0.5 - 2*gamma)*MT_{n-1} + (0.5 + 2*gamma)*MT_{n}
        MT_{n+1/2} = MT_{n-1/2} + dt*(1 - 2*gamma)*L_{n}

        This is computed in old geometry Omega_{n}.
        """
        a = 0.5 - 2*self.gamma
        b = 0.5 + 2*self.gamma
        c = 1.0 - 2*self.gamma

        self.l_form *= self.dt_const*c
        self.l_form += a*self.msol_old
        self.l_form += b*self.msol
        assemble(self.a_rk, self.mass_matrix)
        solve(self.mass_matrix, self.solution, self.l_form)

    def correct(self):
        """
        Correction step from t_{n} to t_{n+1}

        MT_{n+1} = MT_{n} + dt*L_{n+1/2}

        This is computed in new geometry Omega_{n+1}.
        """
        self.l_form += self.msol
        assemble(self.a_rk, self.mass_matrix)
        solve(self.mass_matrix, self.solution, self.l_form)
        # time shift
        self.msol_old.assign(self.msol)
        self.msol.assign(self.l_form)

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t + self.dt)
        self.predict()
        self.correct()
