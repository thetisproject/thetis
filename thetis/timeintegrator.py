"""
Generic time integration schemes to advance equations in time.

Tuomas Karna 2015-03-27
"""
from utility import *


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
    def __init__(self, equation, dt, solver_parameters={},
                 funcs_nplushalf={}):
        """Creates forms for the time integrator"""
        super(SSPRK33, self).__init__(equation, solver_parameters)
        self.explicit = True
        self.CFL_coeff = 1.0

        self.solution_old = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)  # for single stages

        self.K0 = Function(self.equation.space)
        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                if isinstance(self.funcs[k], Function):
                    self.funcs_old[k] = Function(
                        self.funcs[k].function_space())
                elif isinstance(self.funcs[k], Constant):
                    self.funcs_old[k] = Constant(self.funcs[k])
        self.funcs_nplushalf = funcs_nplushalf
        # values used in equations
        self.args = {}
        for k in self.funcs_old:
            if isinstance(self.funcs[k], Function):
                self.args[k] = Function(self.funcs[k].function_space())
            elif isinstance(self.funcs[k], Constant):
                self.args[k] = Constant(self.funcs[k])

        self.dt_const = Constant(dt)

        mass_term = self.equation.mass_term
        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source

        u_old = self.solution_old
        u_tri = self.equation.tri
        self.a_rk = mass_term(u_tri)
        self.L_RK = self.dt_const*(rhs(u_old, **self.args) +
                                   rhsi(u_old, **self.args) +
                                   source(**self.args))
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_k0 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K0)
        self.solver_k0 = LinearVariationalSolver(prob_k0,
                                                 solver_parameters=self.solver_parameters)
        prob_k1 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K1)
        self.solver_k1 = LinearVariationalSolver(prob_k1,
                                                 solver_parameters=self.solver_parameters)
        prob_k2 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K2)
        self.solver_k2 = LinearVariationalSolver(prob_k2,
                                                 solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, update_forcings):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        # stage 0
        for k in self.args:  # set args to t
            self.args[k].assign(self.funcs_old[k])
        if update_forcings is not None:
            update_forcings(t)
        self.solver_k0.solve()
        # stage 1
        self.solution_old.assign(solution + self.K0)
        for k in self.args:  # set args to t+dt
            self.args[k].assign(self.funcs[k])
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solver_k1.solve()
        # stage 2
        self.solution_old.assign(solution + 0.25*self.K0 + 0.25*self.K1)
        for k in self.args:  # set args to t+dt/2
            if k in self.funcs_nplushalf:
                self.args[k].assign(self.funcs_nplushalf[k])
            else:
                self.args[k].assign(0.5*self.funcs[k] + 0.5*self.funcs_old[k])
        if update_forcings is not None:
            update_forcings(t+dt/2)
        self.solver_k2.solve()
        # final solution
        solution.assign(solution + (1.0/6.0)*self.K0 + (1.0/6.0)*self.K1 +
                        (2.0/3.0)*self.K2)

        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)

    def solve_stage(self, i_stage, t, dt, solution, update_forcings=None):
        if i_stage == 0:
            # stage 0
            self.solution_n.assign(solution)
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t
                self.args[k].assign(self.funcs[k])
            if update_forcings is not None:
                update_forcings(t)
            self.solver_k0.solve()
            solution.assign(self.solution_n + self.K0)
        elif i_stage == 1:
            # stage 1
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t+dt
                self.args[k].assign(self.funcs[k])
            if update_forcings is not None:
                update_forcings(t+dt)
            self.solver_k1.solve()
            solution.assign(self.solution_n + 0.25*self.K0 + 0.25*self.K1)
        elif i_stage == 2:
            # stage 2
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t+dt/2
                self.args[k].assign(self.funcs[k])
            if update_forcings is not None:
                update_forcings(t+dt/2)
            self.solver_k2.solve()
            # final solution
            solution.assign(self.solution_n + (1.0/6.0)*self.K0 +
                            (1.0/6.0)*self.K1 + (2.0/3.0)*self.K2)


class SSPRK33StageNew(TimeIntegrator):
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
        super(SSPRK33StageNew, self).__init__(equation, solver_parameters)
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
        self.L_RK = self.dt_const*self.equation.get_residual('all', self.solution_old, self.solution_old, self.fields, self.fields, bnd_conditions)

        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_k0 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K0)
        self.solver_k0 = LinearVariationalSolver(prob_k0,
                                                 solver_parameters=self.solver_parameters)
        prob_k1 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K1)
        self.solver_k1 = LinearVariationalSolver(prob_k1,
                                                 solver_parameters=self.solver_parameters)
        prob_k2 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K2)
        self.solver_k2 = LinearVariationalSolver(prob_k2,
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
    def __init__(self, equation, dt, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(SSPRK33Stage, self).__init__(equation, solver_parameters)
        self.explicit = True
        self.CFL_coeff = 1.0
        self.n_stages = 3

        self.solution_old = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)  # for single stages

        self.K0 = Function(self.equation.space)
        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.args = self.equation.kwargs

        self.dt_const = Constant(dt)

        mass_term = self.equation.mass_term
        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source

        u_old = self.solution_old
        u_tri = self.equation.tri
        self.a_rk = mass_term(u_tri)
        self.L_RK = self.dt_const*(rhs(u_old, **self.args) +
                                   rhsi(u_old, **self.args) +
                                   source(**self.args))
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_k0 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K0)
        self.solver_k0 = LinearVariationalSolver(prob_k0,
                                                 solver_parameters=self.solver_parameters)
        prob_k1 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K1)
        self.solver_k1 = LinearVariationalSolver(prob_k1,
                                                 solver_parameters=self.solver_parameters)
        prob_k2 = LinearVariationalProblem(self.a_rk, self.L_RK, self.K2)
        self.solver_k2 = LinearVariationalSolver(prob_k2,
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
    def __init__(self, equation, dt, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(SSPRK33StageSemiImplicit, self).__init__(equation, solver_parameters)
        self.explicit = True
        self.CFL_coeff = 1.0
        self.n_stages = 3
        self.theta = Constant(0.5)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.solution_old = Function(self.equation.space)
        self.solution_rhs = Function(self.equation.space)

        self.sol0 = Function(self.equation.space)
        self.sol1 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.args = self.equation.kwargs

        self.dt_const = Constant(dt)
        mass_term = self.equation.mass_term
        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source

        u_old = self.solution_old
        u_0 = self.sol0
        u_1 = self.sol1
        sol = self.equation.solution

        self.F_0 = (mass_term(u_0) - mass_term(u_old) -
                    self.dt_const*(
                        self.theta*rhsi(u_0, **self.args) +
                        (1-self.theta)*rhsi(u_old, **self.args) +
                        rhs(u_old, **self.args) +
                        source(**self.args))
                    )
        self.F_1 = (mass_term(u_1) -
                    3.0/4.0*mass_term(u_old) - 1.0/4.0*mass_term(u_0) -
                    1.0/4.0*self.dt_const*(
                        self.theta*rhsi(u_1, **self.args) +
                        (1-self.theta)*rhsi(u_0, **self.args) +
                        rhs(u_0, **self.args) +
                        source(**self.args))
                    )
        self.F_2 = (mass_term(sol) -
                    1.0/3.0*mass_term(u_old) - 2.0/3.0*mass_term(u_1) -
                    2.0/3.0*self.dt_const*(
                        self.theta*rhsi(sol, **self.args) +
                        (1-self.theta)*rhsi(u_1, **self.args) +
                        rhs(u_1, **self.args) +
                        source(**self.args))
                    )
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_f0 = NonlinearVariationalProblem(self.F_0, self.sol0)
        self.solver_f0 = NonlinearVariationalSolver(prob_f0,
                                                    solver_parameters=self.solver_parameters)
        prob_f1 = NonlinearVariationalProblem(self.F_1, self.sol1)
        self.solver_f1 = NonlinearVariationalSolver(prob_f1,
                                                    solver_parameters=self.solver_parameters)
        prob_f2 = NonlinearVariationalProblem(self.F_2, self.equation.solution)
        self.solver_f2 = NonlinearVariationalSolver(prob_f2,
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
            # BUG there's a bug in assembly cache, need to set to false
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


class SSPRK33StageSemiImplicitNew(TimeIntegrator):
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
        super(SSPRK33StageSemiImplicitNew, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.explicit = True
        self.CFL_coeff = 1.0
        self.n_stages = 3
        self.theta = Constant(0.5)

        self.solution = solution
        self.solution_old = Function(self.equation.function_space, name='old solution')

        self.fields = fields

        self.sol0 = Function(self.equation.function_space)
        self.sol1 = Function(self.equation.function_space)

        self.dt_const = Constant(dt)

        # FIXME old solution should be set correctly, this is consistent with old formulation
        args = (self.fields, self.fields, bnd_conditions)
        self.F_0 = (self.equation.mass_term(self.sol0) - self.equation.mass_term(self.solution_old) -
                    self.dt_const*(
                        self.theta*self.equation.get_residual('implicit', self.sol0, self.sol0, *args) +
                        (1-self.theta)*self.equation.get_residual('implicit', self.solution_old, self.solution_old, *args) +
                        self.equation.get_residual('explicit', self.solution_old, self.solution_old, *args) +
                        self.equation.get_residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.F_1 = (self.equation.mass_term(self.sol1) -
                    3.0/4.0*self.equation.mass_term(self.solution_old) - 1.0/4.0*self.equation.mass_term(self.sol0) -
                    1.0/4.0*self.dt_const*(
                        self.theta*self.equation.get_residual('implicit', self.sol1, self.sol1, *args) +
                        (1-self.theta)*self.equation.get_residual('implicit', self.sol0, self.sol0, *args) +
                        self.equation.get_residual('explicit', self.sol0, self.sol0, *args) +
                        self.equation.get_residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.F_2 = (self.equation.mass_term(self.solution) -
                    1.0/3.0*self.equation.mass_term(self.solution_old) - 2.0/3.0*self.equation.mass_term(self.sol1) -
                    2.0/3.0*self.dt_const*(
                        self.theta*self.equation.get_residual('implicit', self.solution, self.solution, *args) +
                        (1-self.theta)*self.equation.get_residual('implicit', self.sol1, self.sol1, *args) +
                        self.equation.get_residual('explicit', self.sol1, self.sol1, *args) +
                        self.equation.get_residual('source', self.solution_old, self.solution_old, *args))
                    )
        self.update_solver()

    def update_solver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        prob_f0 = NonlinearVariationalProblem(self.F_0, self.sol0)
        self.solver_f0 = NonlinearVariationalSolver(prob_f0,
                                                    solver_parameters=self.solver_parameters)
        prob_f1 = NonlinearVariationalProblem(self.F_1, self.sol1)
        self.solver_f1 = NonlinearVariationalSolver(prob_f1,
                                                    solver_parameters=self.solver_parameters)
        prob_f2 = NonlinearVariationalProblem(self.F_2, self.solution)
        self.solver_f2 = NonlinearVariationalSolver(prob_f2,
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
    def __init__(self, equation, dt, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEuler, self).__init__(equation, solver_parameters)
        mass_term = self.equation.mass_term
        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source

        self.dt_const = Constant(dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                if isinstance(self.funcs[k], Function):
                    self.funcs_old[k] = Function(
                        self.funcs[k].function_space())
                elif isinstance(self.funcs[k], Constant):
                    self.funcs_old[k] = Constant(self.funcs[k])

        u_old = self.solution_old
        u_tri = self.equation.tri
        self.A = mass_term(u_tri)
        self.L = (mass_term(u_old) +
                  self.dt_const*(rhs(u_old, **self.funcs_old) +
                                 rhsi(u_old, **self.funcs_old) +
                                 source(**self.funcs_old)
                                 )
                  )
        self.update_solver()

    def update_solver(self):
        prob = LinearVariationalProblem(self.A, self.L, self.equation.solution)
        self.solver = LinearVariationalSolver(prob,
                                              solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class ForwardEulerNew(TimeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEulerNew, self).__init__(equation, solver_parameters)
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
                  self.dt_const*self.equation.get_residual('all', u_old, u_old, self.fields_old, self.fields_old, bnd_conditions)
                  )

        self.update_solver()

    def update_solver(self):
        prob = LinearVariationalProblem(self.A, self.L, self.solution)
        self.solver = LinearVariationalSolver(prob,
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
    def __init__(self, equation, dt, solver_parameters={}, gamma=0.5):
        """Creates forms for the time integrator"""
        super(CrankNicolson, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        mass_term = self.equation.mass_term
        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source

        self.dt_const = Constant(dt)

        self.solution_old = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                if isinstance(self.funcs[k], Function):
                    self.funcs_old[k] = Function(
                        self.funcs[k].function_space())
                elif isinstance(self.funcs[k], Constant):
                    self.funcs_old[k] = Constant(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri
        # Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (mass_term(u) - mass_term(u_old) -
                  self.dt_const*(gamma_const*rhs(u, **self.funcs) +
                                 gamma_const*rhsi(u, **self.funcs) +
                                 gamma_const*source(**self.funcs) +
                                 (1-gamma_const)*rhs(u_old, **self.funcs_old) +
                                 (1-gamma_const)*rhsi(u_old, **self.funcs_old) +
                                 (1-gamma_const)*source(**self.funcs_old)
                                 )
                  )

        self.A = (mass_term(u_tri) -
                  self.dt_const*(
                      gamma_const*rhs(u_tri, **self.funcs) +
                      gamma_const*rhsi(u_tri, **self.funcs))
                  )
        self.L = (mass_term(u_old) +
                  self.dt_const*(
                      gamma_const*source(**self.funcs) +
                      (1-gamma_const)*rhs(u_old, **self.funcs_old) +
                      (1-gamma_const)*rhsi(u_old, **self.funcs_old) +
                      (1-gamma_const)*source(**self.funcs_old))
                  )
        self.update_solver()

    def update_solver(self):
        nest = not ('pc_type' in self.solver_parameters and self.solver_parameters['pc_type'] == 'lu')
        prob = NonlinearVariationalProblem(self.F, self.equation.solution, nest=nest)
        self.solver = NonlinearVariationalSolver(prob,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix=self.name)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, update_forcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance_linear(self, t, dt, solution, update_forcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'ksponly',
        }
        if update_forcings is not None:
            update_forcings(t+dt)
        self.solution_old.assign(solution)
        solve(self.A == self.L, solution, solver_parameters=solver_parameters)
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class CrankNicolsonNew(TimeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, solution, fields, dt, bnd_conditions=None, solver_parameters={}, gamma=0.5):
        """Creates forms for the time integrator"""
        super(CrankNicolsonNew, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
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
        bnd = bnd_conditions
        f = self.fields
        f_old = self.fields_old

        # Crank-Nicolson
        gamma_const = Constant(gamma)
        # FIXME this is consistent with previous implementation but time levels are incorrect
        self.F = (self.equation.mass_term(u) - self.equation.mass_term(u_old) -
                  self.dt_const*(gamma_const*self.equation.get_residual('all', u, u, f, f, bnd) +
                                 (1-gamma_const)*self.equation.get_residual('all', u_old, u_old, f_old, f_old, bnd)
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

    The Butcher tableaus are

    ... to be written

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, dt, solver_parameters={},
                 solver_parameters_dirk={}, solution=None):
        super(SSPIMEX, self).__init__(equation, solver_parameters)

        # implicit scheme
        self.dirk = DIRKLSPUM2(equation, dt,
                               solver_parameters=solver_parameters_dirk,
                               terms_to_add=['implicit'],
                               solution=solution)
        # explicit scheme
        erk_a = [[0, 0, 0],
                 [5.0/6.0, 0, 0],
                 [11.0/24.0, 11.0/24.0, 0]]
        erk_b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
        erk_c = [0, 5.0/6.0, 11.0/12.0]
        self.erk = DIRKGeneric(equation, dt, erk_a, erk_b, erk_c,
                               solver_parameters=solver_parameters_dirk,
                               terms_to_add=['explicit', 'source'],
                               solution=solution)
        self.n_stages = len(erk_b)

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

    Method is defined by its Butcher tableau given as arguments

    c[0] | a[0, 0]
    c[1] | a[1, 0] a[1, 1]
    c[2] | a[2, 0] a[2, 1] a[2, 2]
    ------------------------------
         | b[0]    b[1]    b[2]

    This method also works for explicit RK schemes if one with the zeros on the first row of a.
    """
    def __init__(self, equation, dt, a, b, c,
                 solver_parameters={},
                 terms_to_add='all',
                 solution=None):
        """
        Create new DIRK solver.

        Parameters
        ----------
        equation : equation object
            the equation to solve
        dt : float
            time step (constant)
        a  : array_like (n_stages, n_stages)
            coefficients for the Butcher tableau, must be lower diagonal
        b,c : array_like (n_stages,)
            coefficients for the Butcher tableau
        solver_parameters : dict
            PETSc options for solver
        terms_to_add : 'all' or list of 'implicit', 'explicit', 'source'
            Defines which terms of the equation are to be added to this solver.
            Default 'all' implies terms_to_add = ['implicit', 'explicit', 'source']
        """
        super(DIRKGeneric, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.n_stages = len(b)
        self.a = a
        self.b = b
        self.c = c
        self.terms_to_add = terms_to_add

        rhs = self.equation.rhs
        rhsi = self.equation.rhs_implicit
        source = self.equation.source
        self.dt = dt
        self.dt_const = Constant(dt)
        if solution is not None:
            self.solution_old = solution
        else:
            self.solution_old = self.equation.solution
        self.funcs = self.equation.kwargs
        self.funcs['solution_old'] = self.solution_old
        test = TestFunction(self.equation.space)

        fs = self.equation.solution.function_space()
        from firedrake.functionspaceimpl import MixedFunctionSpace, WithGeometry
        mixed_space = (isinstance(fs, MixedFunctionSpace) or
                       (isinstance(fs, WithGeometry) and
                        isinstance(fs.topological, MixedFunctionSpace)))

        def all_terms(u, **args):
            """Gather all terms that need to be added to the form"""
            f = 0
            if self.terms_to_add == 'all':
                return rhsi(u, **args) + rhs(u, **args) + source(**args)
            if 'implicit' in self.terms_to_add:
                f += rhsi(u, **args)
            if 'explicit' in self.terms_to_add:
                f += rhs(u, **args)
            if 'source' in self.terms_to_add:
                f += source(**args)
            # assert f != 0, \
            #     'adding t  erms {:}: empty form'.format(self.terms_to_add)
            return f

        # Allocate tendency fields
        self.k = []
        for i in xrange(self.n_stages):
            fname = '{:}_k{:}'.format(self.name, i)
            self.k.append(Function(self.equation.space, name=fname))
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
                              all_terms(u, **self.funcs))
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
                              all_terms(u, **self.funcs))
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
                                           options_prefix=sname))

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


class DIRKGenericNew(TimeIntegrator):
    """
    Generic implementation of Diagonally Implicit Runge Kutta schemes.

    Method is defined by its Butcher tableau given as arguments

    c[0] | a[0, 0]
    c[1] | a[1, 0] a[1, 1]
    c[2] | a[2, 0] a[2, 1] a[2, 2]
    ------------------------------
         | b[0]    b[1]    b[2]

    This method also works for explicit RK schemes if one with the zeros on the first row of a.
    """
    def __init__(self, equation, solution, fields, dt, a, b, c,
                 bnd_conditions=None, solver_parameters={}):
        """
        Create new DIRK solver.

        Parameters
        ----------
        equation : equation object
            the equation to solve
        dt : float
            time step (constant)
        a  : array_like (n_stages, n_stages)
            coefficients for the Butcher tableau, must be lower diagonal
        b,c : array_like (n_stages,)
            coefficients for the Butcher tableau
        solver_parameters : dict
            PETSc options for solver
        terms_to_add : 'all' or list of 'implicit', 'explicit', 'source'
            Defines which terms of the equation are to be added to this solver.
            Default 'all' implies terms_to_add = ['implicit', 'explicit', 'source']
        """
        super(DIRKGenericNew, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.n_stages = len(b)
        self.a = a
        self.b = b
        self.c = c

        fs = self.equation.function_space
        self.dt = dt
        self.dt_const = Constant(dt)
        self.solution_old = solution

        test = self.equation.test

        from firedrake.functionspaceimpl import MixedFunctionSpace, WithGeometry
        mixed_space = (isinstance(fs, MixedFunctionSpace) or
                       (isinstance(fs, WithGeometry) and
                        isinstance(fs.topological, MixedFunctionSpace)))

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
                              self.equation.get_residual('all', u, self.solution_old, fields, fields, bnd_conditions))
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
                              self.equation.get_residual('all', u, self.solution_old, fields, fields, bnd_conditions))
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
                                           options_prefix=sname))

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
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        a = [[1.0]]
        b = [1.0]
        c = [1.0]
        super(BackwardEuler, self).__init__(equation, dt, a, b, c,
                                            solver_parameters, terms_to_add)


class BackwardEulerNew(DIRKGenericNew):
    """
    Backward Euler method

    This method has the Butcher tableau

    1   | 1
    ---------
        | 1
    """
    def __init__(self, equation, solution, fields, dt,
                 bnd_conditions=None, solver_parameters={}):
        a = [[1.0]]
        b = [1.0]
        c = [1.0]
        super(BackwardEulerNew, self).__init__(equation, solution, fields, dt,
                                               a, b, c,
                                               bnd_conditions=bnd_conditions,
                                               solver_parameters=solver_parameters)


class ImplicitMidpoint(DIRKGeneric):
    """
    Implicit midpoint method, second order.

    This method has the Butcher tableau

    0.5 | 0.5
    ---------
        | 1
    """
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        a = [[0.5]]
        b = [1.0]
        c = [0.5]
        super(ImplicitMidpoint, self).__init__(equation, dt, a, b, c,
                                               solver_parameters, terms_to_add)


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
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        gamma = Constant((2 + np.sqrt(2))/2)
        a = [[gamma, 0], [1-gamma, gamma]]
        b = [0.5, 0.5]
        c = [gamma, 1]
        super(DIRK22, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, terms_to_add)


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
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        gamma = (3 + np.sqrt(3))/6
        a = [[gamma, 0], [1-2*gamma, gamma]]
        b = [0.5, 0.5]
        c = [gamma, 1-gamma]
        super(DIRK23, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, terms_to_add)


class DIRK33(DIRKGeneric):
    """
    DIRK33, 3-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)
    """
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        gamma = 0.4358665215
        b1 = -3.0/2.0*gamma**2 + 4*gamma - 1.0/4.0
        b2 = 3.0/2.0*gamma**2 - 5*gamma + 5.0/4.0
        a = [[gamma, 0, 0],
             [(1-gamma)/2, gamma, 0],
             [b1, b2, gamma]]
        b = [b1, b2, gamma]
        c = [gamma, (1+gamma)/2, 1]
        super(DIRK33, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, terms_to_add)


class DIRK43(DIRKGeneric):
    """
    DIRK43, 4-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)
    """
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all'):
        a = [[0.5, 0, 0, 0],
             [1.0/6.0, 0.5, 0, 0],
             [-0.5, 0.5, 0.5, 0],
             [3.0/2.0, -3.0/2.0, 0.5, 0.5]]
        b = [3.0/2.0, -3.0/2.0, 0.5, 0.5]
        c = [0.5, 2.0/3.0, 0.5, 1.0]
        super(DIRK43, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, terms_to_add)


class DIRKLSPUM2(DIRKGeneric):
    """
    DIRKLSPUM2, 3-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, dt, solver_parameters={}, terms_to_add='all',
                 solution=None):
        a = [[2.0/11.0, 0, 0],
             [205.0/462.0, 2.0/11.0, 0],
             [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
        b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
        c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
        super(DIRKLSPUM2, self).__init__(equation, dt, a, b, c,
                                         solver_parameters, terms_to_add,
                                         solution)


class DIRKLSPUM2New(DIRKGenericNew):
    """
    DIRKLSPUM2, 3-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, solution, fields, dt,
                 bnd_conditions=None, solver_parameters={}):
        a = [[2.0/11.0, 0, 0],
             [205.0/462.0, 2.0/11.0, 0],
             [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
        b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
        c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
        super(DIRKLSPUM2New, self).__init__(equation, solution, fields, dt,
                                            a, b, c,
                                            bnd_conditions=bnd_conditions,
                                            solver_parameters=solver_parameters)


def cos_time_av_filter(m):
    """
    Raised cos time average filters as in older versions of ROMS.
    a_i : weights for t_{n+1}
          sum(a_i) = 1.0, sum(i*a_i/M) = 1.0
    b_i : weights for t_{n+1/2}
          sum(b_i) = 1.0, sum(i*b_i/M) = 0.5

    Filters have lenght 2*M.
    """
    l = np.arange(1, 2*m+1, dtype=float)/m
    # a raised cos centered at M
    a = np.zeros_like(l)
    ix = (l >= 0.5) * (l <= 1.5)
    a[ix] = 1 + np.cos(2*np.pi*(l[ix]-1))
    a /= sum(a)

    # b as in Shchepetkin and MacWilliams 2005
    b = np.cumsum(a[::-1])[::-1]/m
    # correct b to match 2nd criterion exactly
    error = sum(l*b)-0.5
    p = np.linspace(-1, 1, len(b))
    p /= sum(l*p)
    b -= p*error

    m_star = np.nonzero((np.abs(a) > 1e-10) + (np.abs(b) > 1e-10))[0].max()
    if commrank == 0:
        print 'M', m, m_star
        print 'a', sum(a), sum(l*a)
        print 'b', sum(b), sum(l*b)

    return m_star, [float(f) for f in a], [float(f) for f in b]


class MacroTimeStepIntegrator(TimeIntegrator):
    """Takes an explicit time integrator and iterates it over M time steps.
    Computes time averages to represent solution at M*dt resolution."""
    # NOTE the time averages can be very diffusive
    # NOTE diffusivity depends on M and the choise of time av filter
    # NOTE boxcar filter is very diffusive!
    def __init__(self, timestepper_cls, m, restart_from_av=False):
        super(MacroTimeStepIntegrator, self).__init__(self.subiterator.equation)
        self.subiterator = timestepper_cls
        self.m = m
        self.restart_from_av = restart_from_av
        # functions to hold time averaged solutions
        space = self.subiterator.solution_old.function_space()
        self.solution_n = Function(space)
        self.solution_nplushalf = Function(space)
        self.solution_start = Function(space)
        self.M_star, self.w_full, self.w_half = cos_time_av_filter(m)

    def initialize(self, solution):
        self.subiterator.initialize(solution)
        self.solution_n.assign(solution)
        self.solution_nplushalf.assign(solution)
        self.solution_start.assign(solution)

    def advance(self, t, dt, solution, update_forcings, verbose=False):
        """Advances equations for one macro time step DT=M*dt"""
        m = self.m
        solution_old = self.subiterator.solution_old
        # initialize
        solution_old.assign(self.solution_start)
        solution.assign(self.solution_start)
        # reset time filtered solutions
        # filtered to T_{n+1/2}
        self.solution_nplushalf.assign(0.0)
        # filtered to T_{n+1}
        self.solution_n.assign(0.0)

        # advance fields from T_{n} to T{n+1}
        if verbose and commrank == 0:
            sys.stdout.write('Solving 2D ')
        for i in range(self.M_star):
            self.subiterator.advance(t + i*dt, dt, solution, update_forcings)
            self.solution_nplushalf += self.w_half[i]*solution
            self.solution_n += self.w_full[i]*solution
            if verbose and commrank == 0:
                sys.stdout.write('.')
                if i == m-1:
                    sys.stdout.write('|')
                sys.stdout.flush()
            if not self.restart_from_av and i == m-1:
                # store state at T_{n+1}
                self.solution_start.assign(solution)
        if verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()
        # use filtered solution as output
        solution.assign(self.solution_n)
        if self.restart_from_av:
            self.solution_start.assign(self.solution_n)
