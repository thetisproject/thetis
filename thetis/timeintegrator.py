"""
Generic time integration schemes to advance equations in time.

Tuomas Karna 2015-03-27
"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty

CFL_UNCONDITIONALLY_STABLE = 1.0e6


class AbstractRKScheme(object):
    """
    Abstract class for defining Runge-Kutta schemes.

    Derived classes must defined the Butcher tableau (a,b,c) and CFL number
    (cfl_coeff).

    Currently only explicit or diagonally implicit schemes are supported.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def a(self):
        pass

    @abstractproperty
    def b(self):
        pass

    @abstractproperty
    def c(self):
        pass

    @abstractproperty
    def cfl_coeff(self):
        pass

    def __init__(self):
        super(AbstractRKScheme, self).__init__()
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.c = np.array(self.c)

        assert not np.triu(self.a, 1).any(), 'Butcher tableau must be lower diagonal'
        assert np.allclose(np.sum(self.a, axis=1), self.c), 'Inconsistent Butcher tableau: Row sum of a is not c'

        self.n_stages = len(self.b)
        self.butcher = np.vstack((self.a, self.b))

        self.is_implicit = np.diag(self.a).any()
        self.is_dirk = np.diag(self.a).all()

        if self.is_dirk or not self.is_implicit:
            self.alpha, self.beta = butcher_to_shuosher_form(self.a, self.b)



class TimeIntegrator(object):
    """Base class for all time integrator objects."""
    def __init__(self, equation, solver_parameters={}):
        """Assigns initial conditions to all required fields."""
        super(TimeIntegrator, self).__init__()

        self.equation = equation
        # unique identifier for solver
        self.name = '-'.join([self.__class__.__name__,
                              self.equation.__class__.__name__])
        self.solver_parameters = {}
        self.solver_parameters.update(solver_parameters)

    def initialize(self, equation, dt, solution):
        """Assigns initial conditions to all required fields."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def advance(self):
        """Advances equations for one time step."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))


class ForwardEuler(TimeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEuler, self).__init__(equation, solver_parameters)
        self.dt_const = Constant(dt)
        self.solution = solution
        self.solution_old = Function(self.equation.function_space)

        # dict of all input functions needed for the equation
        self.fields = fields
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

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])


class CrankNicolson(TimeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, theta=0.5, semi_implicit=False):
        """Creates forms for the time integrator"""
        super(CrankNicolson, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.dt_const = Constant(dt)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='solution_old')
        self.fields = fields
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

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields[k])


class SteadyState(TimeIntegrator):
    """Time integrator that solves the steady state equations, leaving out the mass terms"""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(SteadyState, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.solution = solution
        self.fields = fields

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

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        if update_forcings is not None:
            update_forcings(t+dt)
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


class SSPIMEX(TimeIntegrator):
    """
    SSP-IMEX time integration scheme based on [1], method (17).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}):
        super(SSPIMEX, self).__init__(equation, solver_parameters)

        # implicit scheme
        self.dirk = DIRKLSPUM2(equation, solution, fields, dt, bnd_conditions,
                               solver_parameters=solver_parameters_dirk,
                               terms_to_add=('implicit'))
        # explicit scheme
        self.erk = ERKLSPUM2(equation, solution, fields, dt, bnd_conditions,
                             solver_parameters=solver_parameters,
                             terms_to_add=('explicit', 'source'))
        self.n_stages = len(self.erk.b)

    def update_solver(self):
        self.dirk.update_solver()
        self.erk.update_solver()

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, dt, solution, update_forcings)
        self.get_final_solution(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        self.erk.solve_stage(i_stage, t, dt, solution, update_forcings)
        self.dirk.solve_stage(i_stage, t, dt, solution, update_forcings)

    def get_final_solution(self, solution):
        self.erk.get_final_solution(solution)
        self.dirk.get_final_solution(solution)


class DIRKGeneric(TimeIntegrator):
    """
    Generic implementation of Diagonally Implicit Runge Kutta schemes.

    Method is defined by its Butcher tableau

    c[0] | a[0, 0]
    c[1] | a[1, 0] a[1, 1]
    c[2] | a[2, 0] a[2, 1] a[2, 2]
    ------------------------------
         | b[0]    b[1]    b[2]

    All derived classes must define the tableau via properties
    a  : array_like (n_stages, n_stages)
        coefficients for the Butcher tableau, must be lower diagonal
    b,c : array_like (n_stages,)
        coefficients for the Butcher tableau

    This method also works for explicit RK schemes if one with the zeros on the first row of a.
    """
    def __init__(self, equation, solution, fields, dt,
                 bnd_conditions=None, solver_parameters={}, terms_to_add='all'):
        """
        Create new DIRK solver.

        Parameters
        ----------
        equation : equation object
            the equation to solve
        dt : float
            time step (constant)
        solver_parameters : dict
            PETSc options for solver
        terms_to_add : 'all' or list of 'implicit', 'explicit', 'source'
            Defines which terms of the equation are to be added to this solver.
            Default 'all' implies terms_to_add = ['implicit', 'explicit', 'source']
        """
        super(DIRKGeneric, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        fs = self.equation.function_space
        self.dt = dt
        self.dt_const = Constant(dt)
        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')

        test = self.equation.test
        mixed_space = len(fs) > 1

        # Allocate tendency fields
        self.k = []
        for i in xrange(self.n_stages):
            fname = '{:}_k{:}'.format(self.name, i)
            self.k.append(Function(fs, name=fname))

        # construct variational problems
        self.F = []
        if not mixed_space:
            for i in xrange(self.n_stages):
                for j in xrange(i+1):
                    if j == 0:
                        u = self.solution_old + self.a[i][j]*self.dt_const*self.k[j]
                    else:
                        u += self.a[i][j]*self.dt_const*self.k[j]
                self.F.append(-inner(self.k[i], test)*dx +
                              self.equation.residual(terms_to_add, u, self.solution_old, fields, fields, bnd_conditions))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in xrange(self.n_stages):
                for j in xrange(i+1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(split(self.solution_old), split(self.k[j])):
                            u.append(s + self.a[i][j]*self.dt_const*k)
                    else:
                        for l, k in enumerate(split(self.k[j])):
                            u[l] += self.a[i][j]*self.dt_const*k
                self.F.append(-inner(self.k[i], test)*dx +
                              self.equation.residual(terms_to_add, u, self.solution_old, fields, fields, bnd_conditions))
        self.update_solver()

    def update_solver(self):
        # construct solvers
        self.solver = []
        for i in xrange(self.n_stages):
            p = NonlinearVariationalProblem(self.F[i], self.k[i])
            sname = '{:}_stage{:}_'.format(self.name, i)
            self.solver.append(
                NonlinearVariationalSolver(p,
                                           solver_parameters=self.solver_parameters,
                                           options_prefix=sname + '_k{}'.format(i)))

    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)

    def update_solution(self, i_stage, additive=False):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        If additive=False, will overwrite self.solution function, otherwise
        will add to it.
        """
        if not additive:
            self.solution.assign(self.solution_old)
        for j in range(i_stage + 1):
            self.solution += self.a[i_stage][j]*self.dt_const*self.k[j]

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency k at stage i_stage
        """
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()

    def get_final_solution(self, additive=False):
        """
        Evaluates the final solution
        """
        if not additive:
            self.solution.assign(self.solution_old)
        for j in range(self.n_stages):
            self.solution += self.dt_const*self.b[j]*self.k[j]
        self.solution_old.assign(self.solution)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        self.solve_tendency(i_stage, t, update_forcings)
        self.update_solution(i_stage)

    def advance(self, t, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)
        self.get_final_solution()

    # def solve_stage(self, i_stage, t, dt, output=None, update_forcings=None):
    #     """Advances equations for one stage."""
    #     if update_forcings is not None:
    #         update_forcings(t + self.c[i_stage]*self.dt)
    #     self.solver[i_stage].solve()
    #     if output is not None:
    #         if i_stage < self.n_stages - 1:
    #             self.update_solution(i_stage)
    #             # self.get_stage_solution(i_stage, output)
    #         else:
    #             # assign the final solution
    #             # self.get_final_solution(output)
    #             self.get_final_solution()
    #
    # def get_stage_solution(self, i_stage, output, additive=False):
    #     """Stores intermediate solution for stage i_stage to the output field"""
    #     if output != self.solution_old:
    #         # possible only if output is not the internal state container
    #         if not additive:
    #             output.assign(self.solution_old)
    #         for j in xrange(i_stage + 1):
    #             output += self.a[i_stage][j]*self.dt_const*self.k[j]
    #
    # def get_final_solution(self, output=None, additive=False):
    #     """Computes the final solution from the tendencies"""
    #     # update solution
    #     for i in xrange(self.n_stages):
    #         self.solution_old += self.dt_const*self.b[i]*self.k[i]
    #     if output is not None and output != self.solution_old:
    #         if not additive:
    #             # copy to output
    #             output.assign(self.solution_old)
    #         else:
    #             for i in xrange(self.n_stages):
    #                 output += self.dt_const*self.b[i]*self.k[i]


class BackwardEulerAbstract(AbstractRKScheme):
    """
    Backward Euler method

    This method has the Butcher tableau

    1   | 1
    ---------
        | 1
    """
    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class BackwardEuler(DIRKGeneric, BackwardEulerAbstract):
    pass


class ImplicitMidpointAbstract(AbstractRKScheme):
    """
    Implicit midpoint method, second order.

    This method has the Butcher tableau

    0.5 | 0.5
    ---------
        | 1
    """
    a = [[0.5]]
    b = [1.0]
    c = [0.5]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class ImplicitMidpoint(DIRKGeneric, ImplicitMidpointAbstract):
    pass


class DIRK22Abstract(AbstractRKScheme):
    """
    DIRK22, 2-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    gamma   | gamma     0
    1       | 1-gamma  gamma
    -------------------------
            | 0.5       0.5
    with
    gamma = (2 + sqrt(2))/2

    From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)

    [1] Ascher et al. (1997). Implicit-explicit Runge-Kutta methods for
        time-dependent partial differential equations. Applied Numerical
        Mathematics, 25:151-167.
    """
    gamma = (2.0 + np.sqrt(2.0))/2.0
    a = [[gamma, 0],
         [1-gamma, gamma]]
    b = [1-gamma, gamma]
    c = [gamma, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK22(DIRKGeneric, DIRK22Abstract):
    pass


class DIRK23Abstract(AbstractRKScheme):
    """
    DIRK23, 2-stage, 3rd order
    Diagonally Implicit Runge Kutta method

    This method has the Butcher tableau

    gamma   | gamma     0
    1-gamma | 1-2*gamma gamma
    -------------------------
            | 0.5       0.5
    with
    gamma = (3 + sqrt(3))/6

    From DIRK(2,3,3) IMEX scheme in Ascher et al. (1997)
    """
    gamma = (3 + np.sqrt(3))/6
    a = [[gamma, 0],
         [1-2*gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1-gamma]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK23(DIRKGeneric, DIRK23Abstract):
    pass


class DIRK33Abstract(AbstractRKScheme):
    """
    DIRK33, 3-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)
    """
    gamma = 0.4358665215
    b1 = -3.0/2.0*gamma**2 + 4*gamma - 1.0/4.0
    b2 = 3.0/2.0*gamma**2 - 5*gamma + 5.0/4.0
    a = [[gamma, 0, 0],
         [(1-gamma)/2, gamma, 0],
         [b1, b2, gamma]]
    b = [b1, b2, gamma]
    c = [gamma, (1+gamma)/2, 1]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK33(DIRKGeneric, DIRK33Abstract):
    pass


class DIRK43Abstract(AbstractRKScheme):
    """
    DIRK43, 4-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)
    """
    a = [[0.5, 0, 0, 0],
         [1.0/6.0, 0.5, 0, 0],
         [-0.5, 0.5, 0.5, 0],
         [3.0/2.0, -3.0/2.0, 0.5, 0.5]]
    b = [3.0/2.0, -3.0/2.0, 0.5, 0.5]
    c = [0.5, 2.0/3.0, 0.5, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRK43(DIRKGeneric, DIRK43Abstract):
    pass


class DIRKLSPUM2Abstract(AbstractRKScheme):
    """
    DIRKLSPUM2, 3-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    a = [[2.0/11.0, 0, 0],
         [205.0/462.0, 2.0/11.0, 0],
         [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.82


class DIRKLSPUM2(DIRKGeneric, DIRKLSPUM2Abstract):
    pass


class DIRKLPUM2Abstract(AbstractRKScheme):
    """
    DIRKLPUM2, 3-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    a = [[2.0/11.0, 0, 0],
         [41.0/154.0, 2.0/11.0, 0],
         [289.0/847.0, 42.0/121.0, 2.0/11.0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [2.0/11.0, 69.0/154.0, 67.0/77.0]
    cfl_coeff = 4.34  # NOTE for linear problems, nonlin => 3.09


class DIRKLPUM2(DIRKGeneric, DIRKLPUM2Abstract):
    pass


def butcher_to_shuosher_form(a, b):
    """
    Converts Butcher tableau to Shu-Osher form.

    The Shu-Osher form of a s-stage scheme is defined by two s+1 by s+1 arrays
    alpha and beta:

    u^{0} = u^n
    u^(i) = sum_{j=0}^s alpha_{i,j} u^(j) + sum_{j=0}^s beta_{i,j} F(u^(j))
    u^{n+1} = u^(s)

    The Shu-Osher form is not unique. Here we construct the form where beta
    values are the diagonal entries (for DIRK schemes) or sub-diagonal entries
    (for explicit schemes) of the concatenated (a,b) Butcher tableau.

    See Ketchelson et al. (2009) for more information
    http://dx.doi.org/10.1016/j.apnum.2008.03.034
    """
    import numpy.linalg as linalg

    butcher = np.vstack((a, b))

    implicit = np.diag(a).any()

    if implicit:
        # a is not singular
        # take diag entries of a to beta
        be_0 = np.diag(np.diag(a, k=0), k=0)
        be_1 = np.zeros_like(b)
        be_1[-1] = b[-1]
        be = np.vstack((be_0, be_1))

        n = a.shape[0]
        iden = np.eye(n)
        al_0 = iden - np.dot(be_0, linalg.inv(a))
        al_1 = np.dot((b - be_1), np.dot(linalg.inv(be_0), (iden - al_0)))
        al = np.vstack((al_0, al_1))

        # construct full shu-osher form
        alpha = np.zeros((n+1, n+1))
        alpha[:, 1:] = al
        # consistency
        alpha[:, 0] = 1.0 - np.sum(alpha, axis=1)
        beta = np.zeros((n+1, n+1))
        beta[:, 1:] = be
    else:
        # a is singular: solve for lower part of butcher tableau
        aa = butcher[1:, :]
        # take diag entries of aa to beta
        be_0 = np.diag(np.diag(aa, k=0), k=0)
        n = aa.shape[0]
        iden = np.eye(n)
        al_0 = iden - np.dot(be_0, linalg.inv(aa))

        # construct full shu-osher form
        alpha = np.zeros((n+1, n+1))
        alpha[1:, 1:] = al_0
        # consistency
        alpha[:, 0] = 1.0 - np.sum(alpha, axis=1)
        beta = np.zeros((n+1, n+1))
        beta[1:, :-1] = be_0

    # round off small entries
    alpha[np.abs(alpha) < 1e-13] = 0.0
    beta[np.abs(beta) < 1e-13] = 0.0

    # check sanity
    assert np.allclose(np.sum(alpha, axis=1), 1.0)
    if implicit:
        err = beta[:, 1:] - (butcher - np.dot(alpha[:, 1:], a))
    else:
        err = beta[:, :-1] - (butcher - np.dot(alpha[:, :-1], a))
    assert np.allclose(err, 0.0)

    return alpha, beta


class ERKGeneric(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Butcher form.
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, terms_to_add='all'):
        """Creates forms for the time integrator"""
        super(ERKGeneric, self).__init__(equation, solver_parameters)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')
        self.fields = fields

        self.tendency = []
        for i in range(self.n_stages):
            k = Function(self.equation.function_space, name='tendency{:}'.format(i))
            self.tendency.append(k)

        self.dt = dt
        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.L_RK = self.dt_const*self.equation.residual(terms_to_add, self.solution, self.solution, self.fields, self.fields, bnd_conditions)

        self._nontrivial = self.L_RK != 0
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        if self._nontrivial:
            self.solver = []
            for i in range(self.n_stages):
                prob = LinearVariationalProblem(self.a_rk, self.L_RK, self.tendency[i])
                solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                                 solver_parameters=self.solver_parameters)
                self.solver.append(solver)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def update_solution(self, i_stage, additive=False):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        If additive=False, will overwrite self.solution function, otherwise
        will add to it.
        """
        if not additive:
            self.solution.assign(self.solution_old)
        if self._nontrivial:
            for j in range(i_stage):
                self.solution += float(self.a[i_stage][j])*self.tendency[j]

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency k at stage i_stage
        """
        if self._nontrivial:
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*self.dt)
            self.solver[i_stage].solve()

    def get_final_solution(self, additive=False):
        """
        Evaluates the final solution
        """
        if not additive:
            self.solution.assign(self.solution_old)
        if self._nontrivial:
            for j in range(self.n_stages):
                self.solution += float(self.b[j])*self.tendency[j]
        self.solution_old.assign(self.solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(self.n_stages):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class ERKStageGeneric(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator.

    Implements the Shu-Osher form.
    """

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, terms_to_add='all'):
        """Creates forms for the time integrator"""
        super(ERKStageGeneric, self).__init__(equation, solver_parameters)

        self.solution = solution
        self.fields = fields

        self.tendency = Function(self.equation.function_space, name='tendency')
        self.stage_sol = []
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name='sol'.format(i))
            self.stage_sol.append(s)

        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.L_RK = self.dt_const*self.equation.residual(terms_to_add,
                                                         self.solution, self.solution,
                                                         self.fields, self.fields,
                                                         bnd_conditions)
        self._nontrivial = self.L_RK != 0
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        if self._nontrivial:
            prob = LinearVariationalProblem(self.a_rk, self.L_RK, self.tendency)
            self.solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k',
                                                  solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        pass

    def update_solution(self, i_stage, additive=False):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        If additive=False, will overwrite self.solution function, otherwise
        will add to it.
        """
        if not additive:
            self.solution.assign(self.solution_old)
        for j in range(i_stage):
            self.solution += self.a[i_stage][j]*self.k[j]

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        if self._nontrivial:
            self.dt_const.assign(dt)
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*dt)

            if i_stage == 0:
                self.stage_sol[0].assign(self.solution)

            # solve tendency
            self.solver.solve()

            # solve the next internal solution
            # BUG starting with sol_sum = 0 results in different solution!
            sol_sum = float(self.beta[i_stage + 1][i_stage])*self.tendency
            for j in range(i_stage + 1):
                sol_sum += float(self.alpha[i_stage + 1][j])*self.stage_sol[j]
            self.solution.assign(sol_sum)
            if i_stage < self.n_stages - 1:
                self.stage_sol[i_stage + 1].assign(self.solution)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(self.n_stages):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class ERKGenericALE(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator for conservative ALE schemes.

    Implements the Shu-Osher form.
    """

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ERKGenericALE, self).__init__(equation, solver_parameters)

        self.solution = solution
        self.fields = fields

        self.l_form = Function(self.equation.function_space, name='linear form')
        self.stage_msol = []
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name='dual solution {:}'.format(i))
            self.stage_msol.append(s)

        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual('all', self.solution, self.solution, self.fields, self.fields, bnd_conditions)
        self.mass_term = self.equation.mass_term(self.solution)

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        self.A = assemble(self.a_rk)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        # assemble(self.mass_term, self.stage_msol[0])

    def pre_solve(self, i_stage, t, dt, update_forcings=None):
        """Assemble L in the old geometry"""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*dt)

        assemble(float(self.beta[i_stage + 1][i_stage])*self.l_rk, self.l_form)
        assemble(self.mass_term, self.stage_msol[i_stage])

    def finalize_solve(self, i_stage, t, dt, update_forcings=None):
        """Solve problem M*sol = M_old*sol_old + ... + L in the new geometry"""

        # construct full form: L = c*dt*F + M_n*sol_n + ...
        # everything is assembled, just sum up functions
        for j in range(i_stage + 1):
            self.l_form += float(self.alpha[i_stage + 1][j])*self.stage_msol[j]

        # solve
        assemble(self.a_rk, self.A)
        solve(self.A, self.solution, self.l_form)

        # assemble for next iteration
        # target = np.mod(i_stage + 1, self.n_stages)  # final sol to stage 0
        # assemble(self.mass_term, self.stage_msol[target])

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        error('You should not call solve_state with ALE formulation but pre_solve and finalize_solve')
        self.pre_solve(i_stage, t, dt, update_forcings)
        self.finalize_solve(i_stage, t, dt, update_forcings)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(self.n_stages):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class ERKGenericALE2(TimeIntegrator):
    """
    Generic explicit Runge-Kutta time integrator for conservative ALE schemes.

    Implements the Butcher tableau.
    """

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ERKGenericALE2, self).__init__(equation, solver_parameters)

        self.solution = solution
        self.fields = fields

        self.l_form = Function(self.equation.function_space, name='linear form')
        self.msol_old = Function(self.equation.function_space, name='old dual solution')
        self.stage_mk = []  # mass_matrix*tendency
        for i in range(self.n_stages):
            s = Function(self.equation.function_space, name='dual tendency {:}'.format(i))
            self.stage_mk.append(s)

        self.dt = dt
        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.l_rk = self.dt_const*self.equation.residual('all', self.solution, self.solution, self.fields, self.fields, bnd_conditions)
        self.mass_term = self.equation.mass_term(self.solution)

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        self.A = assemble(self.a_rk)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        assemble(self.mass_term, self.msol_old)

    def update_solution(self, i_stage):
        """
        Updates solution to i_stage sub-stage.

        Tendencies must have been evaluated first.
        """
        # construct full form: L = c*dt*F + M_n*sol_n + ...
        # everything is assembled, just sum up functions
        self.l_form.assign(self.msol_old)
        for j in range(i_stage):
            self.l_form += float(self.a[i_stage][j])*self.stage_mk[j]

        assemble(self.a_rk, self.A)
        solve(self.A, self.solution, self.l_form)

    def solve_tendency(self, i_stage, t, update_forcings=None):
        """
        Evaluates the tendency k at stage i_stage
        """
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        assemble(self.l_rk, self.stage_mk[i_stage])

    def get_final_solution(self):
        """
        Evaluates the final solution
        """
        self.l_form.assign(self.msol_old)
        for j in range(self.n_stages):
            self.l_form += float(self.b[j])*self.stage_mk[j]
        assemble(self.a_rk, self.A)
        solve(self.A, self.solution, self.l_form)
        assemble(self.mass_term, self.msol_old)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        self.update_solution(i_stage)
        self.solve_tendency(i_stage, t, update_forcings)
        if i_stage == self.n_stages - 1:
            self.get_final_solution()

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(self.n_stages):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)

    # def pre_solve(self, i_stage, t, dt, update_forcings=None):
    #     """Assemble L in the old geometry"""
    #     self.dt_const.assign(dt)
    #     if update_forcings is not None:
    #         update_forcings(t + self.c[i_stage]*dt)
    #
    #     assemble(self.l_rk, self.stage_mk[i_stage])
    #     # assemble(self.mass_term, self.stage_mk[i_stage])
    #
    # def finalize_solve(self, i_stage, t, dt, update_forcings=None):
    #     """Solve problem M*sol = M_old*sol_old + ... + L in the new geometry"""
    #
    #     # construct full form: L = c*dt*F + M_n*sol_n + ...
    #     # everything is assembled, just sum up functions
    #     self.l_form.assign(self.msol_old)
    #     for j in range(i_stage):
    #         self.l_form += float(self.a[i_stage+1][j])*self.stage_mk[j]
    #
    #     # solve
    #     assemble(self.a_rk, self.A)
    #     solve(self.A, self.solution, self.l_form)
    #
    #     # assemble for next iteration
    #     # target = np.mod(i_stage + 1, self.n_stages)  # final sol to stage 0
    #     # assemble(self.mass_term, self.stage_msol[target])
    #
    # def get_final_solution(self, ):
    #     pass
    #
    # def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
    #     """
    #     Solves a single stage of step from t to t+dt.
    #     All functions that the equation depends on must be at right state
    #     corresponding to each sub-step.
    #     """
    #     error('You should not call solve_state with ALE formulation but pre_solve and finalize_solve')
    #     self.pre_solve(i_stage, t, dt, update_forcings)
    #     self.finalize_solve(i_stage, t, dt, update_forcings)
    #
    # def advance(self, t, dt, solution, update_forcings=None):
    #     """Advances one full time step from t to t+dt.
    #     This assumes that all the functions that the equation depends on are
    #     constants across this interval. If dependent functions need to be
    #     updated call solve_stage instead.
    #     """
    #     for k in range(self.n_stages):
    #         self.solve_stage(k, t, dt, solution,
    #                          update_forcings)


class ERKSemiImplicitGeneric(TimeIntegrator):
    """
    Generic implementation of explicit RK schemes.
    """

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, semi_implicit=False, theta=0.5):
        """Creates forms for the time integrator"""
        super(ERKSemiImplicitGeneric, self).__init__(equation, solver_parameters)

        assert self.n_stages == 3, 'This method supports only for 3 stages'

        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.theta = Constant(theta)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.fields = fields

        self.stage_sol = []
        for i in range(self.n_stages - 1):
            s = Function(self.equation.function_space, name='solution stage {:}'.format(i))
            self.stage_sol.append(s)
        self.stage_sol.append(self.solution)

        self.dt_const = Constant(dt)

        sol_nl = [None]*self.n_stages
        if semi_implicit:
            # linearize around previous sub-timestep using the fact that all terms are written in the form A(u_nl) u
            sol_nl[0] = self.solution_old
            sol_nl[1] = self.stage_sol[0]
            sol_nl[2] = self.stage_sol[1]
        else:
            # solve the full nonlinear residual form
            sol_nl[0] = self.stage_sol[0]
            sol_nl[1] = self.stage_sol[1]
            sol_nl[2] = self.solution

        args = (self.fields, self.fields, bnd_conditions)
        self.L = [None]*self.n_stages
        self.F = [None]*self.n_stages
        self.L[0] = (self.theta*self.equation.residual('implicit', self.stage_sol[0], sol_nl[0], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.solution_old, self.solution_old, *args) +
                     self.equation.residual('explicit', self.solution_old, self.solution_old, *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[0] = (self.equation.mass_term(self.stage_sol[0]) -
                     self.alpha[1][0]*self.equation.mass_term(self.solution_old) -
                     self.beta[1][0]*self.dt_const*self.L[0])
        self.L[1] = (self.theta*self.equation.residual('implicit', self.stage_sol[1], sol_nl[1], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.stage_sol[0], self.stage_sol[0], *args) +
                     self.equation.residual('explicit', self.stage_sol[0], self.stage_sol[0], *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[1] = (self.equation.mass_term(self.stage_sol[1]) -
                     self.alpha[2][0]*self.equation.mass_term(self.solution_old) -
                     self.alpha[2][1]*self.equation.mass_term(self.stage_sol[0]) -
                     self.beta[2][1]*self.dt_const*self.L[1])
        self.L[2] = (self.theta*self.equation.residual('implicit', self.stage_sol[2], sol_nl[2], *args) +
                     (1-self.theta)*self.equation.residual('implicit', self.stage_sol[1], self.stage_sol[1], *args) +
                     self.equation.residual('explicit', self.stage_sol[1], self.stage_sol[1], *args) +
                     self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F[2] = (self.equation.mass_term(self.stage_sol[2]) -
                     self.alpha[3][0]*self.equation.mass_term(self.solution_old) -
                     self.alpha[3][1]*self.equation.mass_term(self.stage_sol[0]) -
                     self.alpha[3][2]*self.equation.mass_term(self.stage_sol[1]) -
                     self.beta[3][2]*self.dt_const*self.L[2])
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        self.solver = []
        for i in range(self.n_stages):
            prob = NonlinearVariationalProblem(self.F[i], self.stage_sol[i])
            solv = NonlinearVariationalSolver(prob, options_prefix=self.name + '_k{:}'.format(i),
                                              solver_parameters=self.solver_parameters)
            self.solver.append(solv)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at rigth state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*dt)
        self.solver[i_stage].solve()
        if i_stage == self.n_stages - 1:
            self.solution_old.assign(solution)
        else:
            solution.assign(self.stage_sol[i_stage])

    def advance(self, t, dt, solution, update_forcings):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(3):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class SSPRK33Abstract(AbstractRKScheme):
    """
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).

    This scheme has Butcher tableau
    0   |
    1   | 1
    1/2 | 1/4 1/4
    ---------------
        | 1/6 1/6 2/3

    CFL coefficient is 1.0
    """
    a = [[0, 0, 0],
         [1.0, 0, 0],
         [0.25, 0.25, 0]]
    b = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    c = [0, 1.0, 0.5]
    cfl_coeff = 1.0


class SSPRK33Stage(ERKStageGeneric, SSPRK33Abstract):
    pass


class SSPRK33StageSemiImplicit(ERKSemiImplicitGeneric, SSPRK33Abstract):
    pass


class ERKLSPUM2Abstract(AbstractRKScheme):
    """
    ERKLSPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    a = [[0, 0, 0],
         [5.0/6.0, 0, 0],
         [11.0/24.0, 11.0/24.0, 0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [0, 5.0/6.0, 11.0/12.0]
    cfl_coeff = 1.2


class ERKLSPUM2StageSemiImplicit(ERKSemiImplicitGeneric, ERKLSPUM2Abstract):
    pass


class ERKLSPUM2Stage(ERKStageGeneric, ERKLSPUM2Abstract):
    pass


class ERKLSPUM2(ERKGeneric, ERKLSPUM2Abstract):
    pass


class ERKLSPUM2ALE(ERKGenericALE2, ERKLSPUM2Abstract):
    pass


class ERKLPUM2Abstract(AbstractRKScheme):
    """
    ERKLPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (20) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    a = [[0, 0, 0],
         [1.0/2.0, 0, 0],
         [1.0/2.0, 1.0/2.0, 0]]
    b = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    c = [0, 1.0/2.0, 1.0]
    cfl_coeff = 2.0


class ERKLPUM2StageSemiImplicit(ERKSemiImplicitGeneric, ERKLPUM2Abstract):
    pass


class ERKLPUM2Stage(ERKStageGeneric, ERKLPUM2Abstract):
    pass


class ERKLPUM2(ERKGeneric, ERKLPUM2Abstract):
    pass


class ERKLPUM2ALE(ERKGenericALE2, ERKLPUM2Abstract):
    pass


class ForwardEulerAbstract(AbstractRKScheme):
    """
    Forward Euler
    Explicit Runge Kutta method

    CFL coefficient is 1.0
    """
    a = [[0]]
    b = [1.0]
    c = [0]
    cfl_coeff = 1.0


class ForwardEulerSemiImplicit(TimeIntegrator, ForwardEulerAbstract):
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, semi_implicit=True, theta=0.5):
        super(ForwardEulerSemiImplicit, self).__init__(equation, solver_parameters)

        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.theta = Constant(theta)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.fields = fields

        self.dt_const = Constant(dt)

        if semi_implicit:
            # linearize around previous sub-timestep using the fact that all terms are written in the form A(u_nl) u
            sol_nl0 = self.solution_old
        else:
            # solve the full nonlinear residual form
            sol_nl0 = self.solution

        args = (self.fields, self.fields, bnd_conditions)
        self.L_0 = (self.theta*self.equation.residual('implicit', self.solution, sol_nl0, *args) +
                    (1-self.theta)*self.equation.residual('implicit', self.solution_old, self.solution_old, *args) +
                    self.equation.residual('explicit', self.solution_old, self.solution_old, *args) +
                    self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.F_0 = (self.equation.mass_term(self.solution) -
                    self.equation.mass_term(self.solution_old) -
                    self.dt_const*self.L_0)
        self.mass_form = self.equation.mass_term(self.equation.trial)
        self.f = self.dt_const*(
            self.theta*self.equation.residual('implicit', self.solution, sol_nl0, *args) +
            (1-self.theta)*self.equation.residual('implicit', self.solution_old, self.solution_old, *args) +
            self.equation.residual('explicit', self.solution_old, self.solution_old, *args) +
            self.equation.residual('source', self.solution_old, self.solution_old, *args))
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_f0 = NonlinearVariationalProblem(self.F_0, self.solution)
        self.solver_f0 = NonlinearVariationalSolver(prob_f0, options_prefix=self.name + '_k0',
                                                    solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at rigth state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if i_stage == 0:
            # stage 0
            if update_forcings is not None:
                update_forcings(t + self.c[i_stage]*dt)
            self.solver_f0.solve()
            # l_form = assemble(self.f)
            # a_form = assemble(self.mass_term)
            # solve(a_form, solution, l_form)
            # solution += self.solution_old
            self.solution_old.assign(solution)

    def advance(self, t, dt, solution, update_forcings):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(1):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class ForwardEulerStage(TimeIntegrator, ForwardEulerAbstract):
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEulerStage, self).__init__(equation, solver_parameters)

        self.solution = solution
        self.fields = fields

        fs = self.equation.function_space
        self.tendency = Function(fs, name='tendency')
        self.solution_form_old = Function(fs, name='solution form old')
        self.l_form = Function(fs, name='linear form')

        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.L_RK = self.dt_const*self.equation.residual('all', self.solution, self.solution, self.fields, self.fields, bnd_conditions)
        self.mass_term = inner(self.solution, self.equation.test)*dx

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        self.mass_matrix = assemble(self.a_rk)
        self.linsolver = LinearSolver(self.mass_matrix)
        # prob = LinearVariationalProblem(self.a_rk, self.L_RK, self.tendency)
        # self.solver = LinearVariationalSolver(prob, options_prefix=self.name + '_k',
        #                                       solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        assemble(self.mass_term + self.L_RK, self.l_form)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*dt)

        # self.presolve()
        # self.postsolve()

        # solve equations for the current geometry
        assemble(self.a_rk, self.mass_matrix)
        self.linsolver = LinearSolver(self.mass_matrix)
        self.linsolver.solve(self.solution, self.l_form)

        # pre-assemble form for the next time step
        assemble(self.mass_term + self.L_RK, self.l_form)

        # # solve tendency
        # # self.solver.solve()
        #
        # assemble(self.L_RK, self.l_form)
        # a_form = assemble(self.a_rk)
        # solve(a_form, self.tendency, self.l_form)
        #
        # # update old solution
        # # solve(a_form, self.solution, self.solution_form_old)
        # # self.solution += self.tendency
        #
        # # # solve the next internal solution
        # # sol_sum = float(self.beta[i_stage + 1][i_stage])*self.tendency
        # # sol_sum += self.solution
        # # self.solution.assign(sol_sum)
        # self._update_old_solution_form(solution)
        #
        # assemble(inner(self.solution + self.tendency, self.equation.test)*dx, self.l_form)

    def pre_solve(self, i_stage, t, dt, update_forcings=None):
        # # solve tendency
        # self.solver.solve()
        # # solve the next internal solution
        # self.solution += self.tendency

        # # solve tendency
        assemble(self.mass_term + self.L_RK, self.l_form)
        # a_form = assemble(self.a_rk)
        # solve(a_form, self.solution, self.l_form)

        # assemble new solution
        # assemble(inner(self.solution, self.equation.test)*dx, self.l_form)

    def finalize_solve(self, i_stage, t, dt, update_forcings=None):
        # update solution in current mesh
        # a_form = assemble(self.a_rk)
        # solve(a_form, self.solution, self.l_form)
        assemble(self.a_rk, self.mass_matrix)
        self.linsolver = LinearSolver(self.mass_matrix)
        self.linsolver.solve(self.solution, self.l_form)
        #solve(self.mass_matrix, self.solution, self.l_form)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(self.n_stages):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class IMEXGeneric(TimeIntegrator):
    """
    Generic implementation of IMEX schemes
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def dirk_class(self):
        pass

    @abstractproperty
    def erk_class(self):
        pass

    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}):
        super(IMEXGeneric, self).__init__(equation, solver_parameters)

        # implicit scheme
        self.dirk = self.dirk_class(equation, solution, fields, dt, bnd_conditions,
                                    solver_parameters=solver_parameters_dirk,
                                    terms_to_add=('implicit'))
        # explicit scheme
        self.erk = self.erk_class(equation, solution, fields, dt, bnd_conditions,
                                  solver_parameters=solver_parameters,
                                  terms_to_add=('explicit', 'source'))
        assert self.erk.n_stages == self.dirk.n_stages
        self.n_stages = self.erk.n_stages

    def update_solver(self):
        self.dirk.update_solver()
        self.erk.update_solver()

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, update_forcings)

    def solve_stage(self, i_stage, t, update_forcings=None):
        """
        Solves one sub-stage.
        """
        # set solution to u_n + dt*sum(a*k_erk)
        self.erk.update_solution(i_stage, additive=False)
        # solve implicit tendency (this is implicit solve)
        self.dirk.solve_tendency(i_stage, t, update_forcings)
        # set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
        self.dirk.update_solution(i_stage, additive=True)
        # evaluate explicit tendency
        self.erk.solve_tendency(i_stage, t, update_forcings)

    def get_final_solution(self):
        """
        Evaluates the final solution.
        """
        # set solution to u_n + sum(b*k_erk)
        self.erk.get_final_solution(additive=False)
        # set solution to u_n + sum(b*k_erk) + sum(b*k_dirk)
        self.dirk.get_final_solution(additive=True)
        # update old solution
        # TODO share old solution func between dirk and erk
        self.erk.solution_old.assign(self.dirk.solution)


class IMEXLPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (20) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    erk_class = ERKLPUM2
    dirk_class = DIRKLPUM2


class IMEXLSPUM2(IMEXGeneric):
    """
    SSP-IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.

    CFL coefficient is 2.0
    """
    erk_class = ERKLSPUM2
    dirk_class = DIRKLSPUM2


class ERKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.0]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class ERKMidpoint(ERKGeneric, ERKMidpointAbstract):
    pass


class ERKMidpointALE(ERKGenericALE2, ERKMidpointAbstract):
    pass


class ESDIRKMidpointAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.0, 0.5]]
    b = [0.0, 1.0]
    c = [0.0, 0.5]
    cfl_coeff = 1.0


class ESDIRKMidpoint(DIRKGeneric, ESDIRKMidpointAbstract):
    pass


class IMEXMidpoint(IMEXGeneric):
    """
    Implicit-explicit midpoint scheme (1, 2, 2)

    From Ascher (1997)
    """
    erk_class = ERKMidpoint
    dirk_class = ESDIRKMidpoint


class ERKEulerAbstract(AbstractRKScheme):
    a = [[0.0]]
    b = [1.0]
    c = [0.0]
    cfl_coeff = 1.0


class ERKEuler(ERKGeneric, ERKEulerAbstract):
    pass


class ERKEulerALE(ERKGenericALE2, ERKEulerAbstract):
    pass


class DIRKEulerAbstract(AbstractRKScheme):
    a = [[1.0]]
    b = [1.0]
    c = [1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class DIRKEuler(DIRKGeneric, DIRKEulerAbstract):
    pass


class IMEXEuler(IMEXGeneric):
    """
    Forward-Backward Euler
    """
    erk_class = ERKEuler
    dirk_class = DIRKEuler


class LeapFrogAM3(TimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 ALE time integrator

    Defined in Shchepetkin (2005) and Karna (2013)
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, solver_parameters_dirk={}, terms_to_add='all'):
        super(LeapFrogAM3, self).__init__(equation, solver_parameters)

        self.dt = dt
        self.dt_const = Constant(dt)
        self.solution = solution
        self.fields = fields

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


class CrankNicolsonAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [0.5, 0.5]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = CFL_UNCONDITIONALLY_STABLE


class CrankNicolsonRK(DIRKGeneric, CrankNicolsonAbstract):
    pass


class ERKTrapezoidAbstract(AbstractRKScheme):
    a = [[0.0, 0.0],
         [1.0, 0.0]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    cfl_coeff = 1.0


class ERKTrapezoidRK(ERKGeneric, ERKTrapezoidAbstract):
    pass
