"""
Generic time integration schemes to advance equations in time.

Tuomas Karna 2015-03-27
"""
from __future__ import absolute_import
from .utility import *
from abc import ABCMeta, abstractproperty


class TimeIntegrator(object):
    """Base class for all time integrator objects."""
    def __init__(self, equation, solver_parameters={}):
        """Assigns initial conditions to all required fields."""
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


class SSPRK33(TimeIntegrator):
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
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={},
                 funcs_nplushalf={}):
        """Creates forms for the time integrator"""
        super(SSPRK33, self).__init__(equation, solver_parameters)
        self.explicit = True
        self.CFL_coeff = 1.0

        fs = self.equation.function_space
        self.solution_old = Function(fs)
        self.solution_n = Function(fs)  # for single stages

        self.K0 = Function(fs)
        self.K1 = Function(fs)
        self.K2 = Function(fs)

        # dict of all input functions needed for the equation
        self.fields_new = fields
        # create functions to hold the values of previous time step
        self.fields_old = {}
        for k in self.fields_new:
            if self.fields_new[k] is not None:
                if isinstance(self.fields_new[k], Function):
                    self.fields_old[k] = Function(
                        self.fields_new[k].function_space())
                elif isinstance(self.fields_new[k], Constant):
                    self.fields_old[k] = Constant(self.fields_new[k])
        self.funcs_nplushalf = funcs_nplushalf
        # values used in equations
        self.fields = {}
        for k in self.fields_old:
            if isinstance(self.fields_new[k], Function):
                self.fields[k] = Function(self.fields_new[k].function_space())
            elif isinstance(self.fields_new[k], Constant):
                self.fields[k] = Constant(self.fields_new[k])

        self.dt_const = Constant(dt)

        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.L_RK0 = self.dt_const*self.equation.residual('all', self.solution_old, self.solution_old, self.fields_old, self.fields_old, bnd_conditions)
        self.L_RK1 = self.dt_const*self.equation.residual('all', self.solution_old, self.solution_old, self.fields_new, self.fields_new, bnd_conditions)
        self.L_RK2 = self.dt_const*self.equation.residual('all', self.solution_old, self.solution_old, self.fields, self.fields, bnd_conditions)
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_k0 = LinearVariationalProblem(self.a_rk, self.L_RK0, self.K0)
        self.solver_k0 = LinearVariationalSolver(prob_k0, options_prefix=self.name + '_k0',
                                                 solver_parameters=self.solver_parameters)
        prob_k1 = LinearVariationalProblem(self.a_rk, self.L_RK1, self.K1)
        self.solver_k1 = LinearVariationalSolver(prob_k1, options_prefix=self.name + '_k1',
                                                 solver_parameters=self.solver_parameters)
        prob_k2 = LinearVariationalProblem(self.a_rk, self.L_RK2, self.K2)
        self.solver_k2 = LinearVariationalSolver(prob_k2, options_prefix=self.name + '_k2',
                                                 solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields_new[k])

    def advance(self, t, dt, solution, update_forcings):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        # stage 0
        if update_forcings is not None:
            update_forcings(t)
        self.solver_k0.solve()
        # stage 1
        self.solution_old.assign(solution + self.K0)
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solver_k1.solve()
        # stage 2
        self.solution_old.assign(solution + 0.25*self.K0 + 0.25*self.K1)
        for k in self.fields:  # set args to t+dt/2
            if k in self.funcs_nplushalf:
                self.fields[k].assign(self.funcs_nplushalf[k])
            else:
                self.fields[k].assign(0.5*self.fields_new[k] + 0.5*self.fields_old[k])
        if update_forcings is not None:
            update_forcings(t+dt/2)
        self.solver_k2.solve()
        # final solution
        solution.assign(solution + (1.0/6.0)*self.K0 + (1.0/6.0)*self.K1 +
                        (2.0/3.0)*self.K2)

        # store old values
        for k in self.fields_old:
            self.fields_old[k].assign(self.fields_new[k])
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        if i_stage == 0:
            # stage 0
            self.solution_n.assign(solution)
            self.solution_old.assign(solution)
            if update_forcings is not None:
                update_forcings(t)
            self.solver_k0.solve()
            solution.assign(self.solution_n + self.K0)
        elif i_stage == 1:
            # stage 1
            self.solution_old.assign(solution)
            if update_forcings is not None:
                update_forcings(t+dt)
            self.solver_k1.solve()
            solution.assign(self.solution_n + 0.25*self.K0 + 0.25*self.K1)
        elif i_stage == 2:
            # stage 2
            self.solution_old.assign(solution)
            for k in self.fields:  # set args to t+dt/2
                if k in self.funcs_nplushalf:
                    self.fields[k].assign(self.funcs_nplushalf[k])
                else:
                    self.fields[k].assign(0.5*self.fields_new[k] + 0.5*self.fields_old[k])
            if update_forcings is not None:
                update_forcings(t+dt/2)
            self.solver_k2.solve()
            # final solution
            solution.assign(self.solution_n + (1.0/6.0)*self.K0 +
                            (1.0/6.0)*self.K1 + (2.0/3.0)*self.K2)


class SSPRK33Stage(TimeIntegrator):
    """
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).
    This class only advances one step at a time.

    This scheme has Butcher tableau
    0   |
    1   | 1
    1/2 | 1/4 1/4
    ---------------
        | 1/6 1/6 2/3

    CFL coefficient is 1.0
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(SSPRK33Stage, self).__init__(equation, solver_parameters)
        self.explicit = True
        self.CFL_coeff = 1.0
        self.n_stages = 3

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')
        self.solution_n = Function(self.equation.function_space, name='stage solution')
        self.fields = fields

        self.K0 = Function(self.equation.function_space, name='tendency0')
        self.K1 = Function(self.equation.function_space, name='tendency1')
        self.K2 = Function(self.equation.function_space, name='tendency2')

        self.dt_const = Constant(dt)

        # fully explicit evaluation
        self.a_rk = self.equation.mass_term(self.equation.trial)
        self.L_RK = self.dt_const*self.equation.residual('all', self.solution_old, self.solution_old, self.fields, self.fields, bnd_conditions)

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_k0 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K0)
        self.solver_k0 = LinearVariationalSolver(prob_k0, options_prefix=self.name + '_k0',
                                                 solver_parameters=self.solver_parameters)
        prob_k1 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K1)
        self.solver_k1 = LinearVariationalSolver(prob_k1, options_prefix=self.name + '_k1',
                                                 solver_parameters=self.solver_parameters)
        prob_k2 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K2)
        self.solver_k2 = LinearVariationalSolver(prob_k2, options_prefix=self.name + '_k2',
                                                 solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at right state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if i_stage == 0:
            # stage 0
            self.solution_n.assign(solution)
            self.solution_old.assign(solution)
            if update_forcings is not None:
                update_forcings(t)
            self.solver_k0.solve()
            solution.assign(self.solution_n + self.K0)
        elif i_stage == 1:
            # stage 1
            self.solution_old.assign(solution)
            if update_forcings is not None:
                update_forcings(t+dt)
            self.solver_k1.solve()
            solution.assign(self.solution_n + 0.25*self.K0 + 0.25*self.K1)
        elif i_stage == 2:
            # stage 2
            self.solution_old.assign(solution)
            if update_forcings is not None:
                update_forcings(t+dt/2)
            self.solver_k2.solve()
            # final solution
            solution.assign(self.solution_n + (1.0/6.0)*self.K0 +
                            (1.0/6.0)*self.K1 + (2.0/3.0)*self.K2)

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(3):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


class SSPRK33StageSemiImplicit(TimeIntegrator):
    """
    3rd order Strong Stability Preserving Runge-Kutta scheme, SSP(3,3).
    This class only advances one step at a time.

    This scheme has Butcher tableau
    0   |
    1   | 1
    1/2 | 1/4 1/4
    ---------------
        | 1/6 1/6 2/3

    CFL coefficient is 1.0
    """
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None,
                 solver_parameters={}, semi_implicit=False, theta=0.5):
        """Creates forms for the time integrator"""
        super(SSPRK33StageSemiImplicit, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        if semi_implicit:
            self.solver_parameters.setdefault('snes_type', 'ksponly')
        else:
            self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.explicit = True
        self.CFL_coeff = 1.0
        self.n_stages = 3
        self.theta = Constant(theta)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.fields = fields

        self.sol0 = Function(self.equation.function_space)
        self.sol1 = Function(self.equation.function_space)

        self.dt_const = Constant(dt)

        if semi_implicit:
            # linearize around previous sub-timestep using the fact that all terms are written in the form A(u_nl) u
            sol_nl0 = self.solution_old
            sol_nl1 = self.sol0
            sol_nl2 = self.sol1
        else:
            # solve the full nonlinear residual form
            sol_nl0 = self.sol0
            sol_nl1 = self.sol1
            sol_nl2 = self.solution

        # FIXME old solution should be set correctly, this is consistent with old formulation
        args = (self.fields, self.fields, bnd_conditions)
        self.F_0 = (self.equation.mass_term(self.sol0) - self.equation.mass_term(self.solution_old) -
                    self.dt_const*(
                        self.theta*self.equation.residual('implicit', self.sol0, sol_nl0, *args) +
                        (1-self.theta)*self.equation.residual('implicit', self.solution_old, self.solution_old, *args) +
                        self.equation.residual('explicit', self.solution_old, self.solution_old, *args) +
                        self.equation.residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.F_1 = (self.equation.mass_term(self.sol1) -
                    3.0/4.0*self.equation.mass_term(self.solution_old) - 1.0/4.0*self.equation.mass_term(self.sol0) -
                    1.0/4.0*self.dt_const*(
                        self.theta*self.equation.residual('implicit', self.sol1, sol_nl1, *args) +
                        (1-self.theta)*self.equation.residual('implicit', self.sol0, self.sol0, *args) +
                        self.equation.residual('explicit', self.sol0, self.sol0, *args) +
                        self.equation.residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.F_2 = (self.equation.mass_term(self.solution) -
                    1.0/3.0*self.equation.mass_term(self.solution_old) - 2.0/3.0*self.equation.mass_term(self.sol1) -
                    2.0/3.0*self.dt_const*(
                        self.theta*self.equation.residual('implicit', self.solution, sol_nl2, *args) +
                        (1-self.theta)*self.equation.residual('implicit', self.sol1, self.sol1, *args) +
                        self.equation.residual('explicit', self.sol1, self.sol1, *args) +
                        self.equation.residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_f0 = NonlinearVariationalProblem(self.F_0, self.sol0)
        self.solver_f0 = NonlinearVariationalSolver(prob_f0, options_prefix=self.name + '_k0',
                                                    solver_parameters=self.solver_parameters)
        prob_f1 = NonlinearVariationalProblem(self.F_1, self.sol1)
        self.solver_f1 = NonlinearVariationalSolver(prob_f1, options_prefix=self.name + '_k1',
                                                    solver_parameters=self.solver_parameters)
        prob_f2 = NonlinearVariationalProblem(self.F_2, self.solution)
        self.solver_f2 = NonlinearVariationalSolver(prob_f2, options_prefix=self.name + '_k2',
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
                update_forcings(t)
            self.solver_f0.solve()
            solution.assign(self.sol0)
        elif i_stage == 1:
            # stage 1
            if update_forcings is not None:
                update_forcings(t+dt)
            self.solver_f1.solve()
            solution.assign(self.sol1)
        elif i_stage == 2:
            # stage 2
            if update_forcings is not None:
                update_forcings(t+dt/2)
            self.solver_f2.solve()
            self.solution_old.assign(solution)

    def advance(self, t, dt, solution, update_forcings):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solve_stage instead.
        """
        for k in range(3):
            self.solve_stage(k, t, dt, solution,
                             update_forcings)


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
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space())
                elif isinstance(self.fields[k], Constant):
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
        self.solution_old = Function(self.equation.function_space)
        self.fields = fields
        # create functions to hold the values of previous time step
        # TODO is this necessary? is self.fields sufficient?
        self.fields_old = {}
        for k in self.fields:
            if self.fields[k] is not None:
                if isinstance(self.fields[k], Function):
                    self.fields_old[k] = Function(
                        self.fields[k].function_space())
                elif isinstance(self.fields[k], Constant):
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
        nest = not ('pc_type' in self.solver_parameters and self.solver_parameters['pc_type'] == 'lu')
        prob = NonlinearVariationalProblem(self.F, self.solution, nest=nest)
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

        self.n_stages = len(self.b)

        fs = self.equation.function_space
        self.dt = dt
        self.dt_const = Constant(dt)
        self.solution_old = solution

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

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.n_stages):
            self.solve_stage(i, t, dt, solution, update_forcings)

    def solve_stage(self, i_stage, t, dt, output=None, update_forcings=None):
        """Advances equations for one stage."""
        if update_forcings is not None:
            update_forcings(t + self.c[i_stage]*self.dt)
        self.solver[i_stage].solve()
        if output is not None:
            if i_stage < self.n_stages - 1:
                self.get_stage_solution(i_stage, output)
            else:
                # assign the final solution
                self.get_final_solution(output)

    def get_stage_solution(self, i_stage, output):
        """Stores intermediate solution for stage i_stage to the output field"""
        if output != self.solution_old:
            # possible only if output is not the internal state container
            output.assign(self.solution_old)
            for j in xrange(i_stage+1):
                output += self.a[i_stage][j]*self.dt_const*self.k[j]

    def get_final_solution(self, output=None):
        """Computes the final solution from the tendencies"""
        # update solution
        for i in xrange(self.n_stages):
            self.solution_old += self.dt_const*self.b[i]*self.k[i]
        if output is not None and output != self.solution_old:
            # copy to output
            output.assign(self.solution_old)


class BackwardEuler(DIRKGeneric):
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


class ImplicitMidpoint(DIRKGeneric):
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


class DIRK22(DIRKGeneric):
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
    gamma = Constant((2 + np.sqrt(2))/2)
    a = [[gamma, 0],
         [1-gamma, gamma]]
    b = [0.5, 0.5]
    c = [gamma, 1]


class DIRK23(DIRKGeneric):
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


class DIRK33(DIRKGeneric):
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


class DIRK43(DIRKGeneric):
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


class DIRKLSPUM2(DIRKGeneric):
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


class ERKLSPUM2(DIRKGeneric):
    """
    ERKLSPUM2, 3-stage, 2nd order
    Explicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    a = [[0, 0, 0],
         [5.0/6.0, 0, 0],
         [11.0/24.0, 11.0/24.0, 0]]
    b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
    c = [0, 5.0/6.0, 11.0/12.0]
