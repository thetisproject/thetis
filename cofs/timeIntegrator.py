"""
Generic time integration schemes to advance equations in time.

Tuomas Karna 2015-03-27
"""
from utility import *


class timeIntegrator(object):
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


class SSPRK33(timeIntegrator):
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

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source

        u_old = self.solution_old
        u_tri = self.equation.tri
        self.a_RK = massTerm(u_tri)
        self.L_RK = self.dt_const*(RHS(u_old, **self.args) +
                                   RHSi(u_old, **self.args) +
                                   Source(**self.args))
        self.updateSolver()

    def updateSolver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        probK0 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K0)
        self.solverK0 = LinearVariationalSolver(probK0,
                                                solver_parameters=self.solver_parameters)
        probK1 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1,
                                                solver_parameters=self.solver_parameters)
        probK2 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K2)
        self.solverK2 = LinearVariationalSolver(probK2,
                                                solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        # stage 0
        for k in self.args:  # set args to t
            self.args[k].assign(self.funcs_old[k])
        if updateForcings is not None:
            updateForcings(t)
        self.solverK0.solve()
        # stage 1
        self.solution_old.assign(solution + self.K0)
        for k in self.args:  # set args to t+dt
            self.args[k].assign(self.funcs[k])
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solverK1.solve()
        # stage 2
        self.solution_old.assign(solution + 0.25*self.K0 + 0.25*self.K1)
        for k in self.args:  # set args to t+dt/2
            if k in self.funcs_nplushalf:
                self.args[k].assign(self.funcs_nplushalf[k])
            else:
                self.args[k].assign(0.5*self.funcs[k] + 0.5*self.funcs_old[k])
        if updateForcings is not None:
            updateForcings(t+dt/2)
        self.solverK2.solve()
        # final solution
        solution.assign(solution + (1.0/6.0)*self.K0 + (1.0/6.0)*self.K1 +
                        (2.0/3.0)*self.K2)

        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)

    def solveStage(self, iStage, t, dt, solution, updateForcings=None):
        if iStage == 0:
            # stage 0
            self.solution_n.assign(solution)
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t
                self.args[k].assign(self.funcs[k])
            if updateForcings is not None:
                updateForcings(t)
            self.solverK0.solve()
            solution.assign(self.solution_n + self.K0)
        elif iStage == 1:
            # stage 1
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t+dt
                self.args[k].assign(self.funcs[k])
            if updateForcings is not None:
                updateForcings(t+dt)
            self.solverK1.solve()
            solution.assign(self.solution_n + 0.25*self.K0 + 0.25*self.K1)
        elif iStage == 2:
            # stage 2
            self.solution_old.assign(solution)
            for k in self.args:  # set args to t+dt/2
                self.args[k].assign(self.funcs[k])
            if updateForcings is not None:
                updateForcings(t+dt/2)
            self.solverK2.solve()
            # final solution
            solution.assign(self.solution_n + (1.0/6.0)*self.K0 +
                            (1.0/6.0)*self.K1 + (2.0/3.0)*self.K2)


class SSPRK33Stage(timeIntegrator):
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
        self.nStages = 3

        self.solution_old = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)  # for single stages

        self.K0 = Function(self.equation.space)
        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.args = self.equation.kwargs

        self.dt_const = Constant(dt)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source

        u_old = self.solution_old
        u_tri = self.equation.tri
        self.a_RK = massTerm(u_tri)
        self.L_RK = self.dt_const*(RHS(u_old, **self.args) +
                                   RHSi(u_old, **self.args) +
                                   Source(**self.args))
        self.updateSolver()

    def updateSolver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        probK0 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K0)
        self.solverK0 = LinearVariationalSolver(probK0,
                                                solver_parameters=self.solver_parameters)
        probK1 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1,
                                                solver_parameters=self.solver_parameters)
        probK2 = LinearVariationalProblem(self.a_RK, self.L_RK, self.K2)
        self.solverK2 = LinearVariationalSolver(probK2,
                                                solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def solveStage(self, iStage, t, dt, solution, updateForcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at rigth state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if iStage == 0:
            # stage 0
            self.solution_n.assign(solution)
            self.solution_old.assign(solution)
            if updateForcings is not None:
                updateForcings(t)
            self.solverK0.solve()
            solution.assign(self.solution_n + self.K0)
        elif iStage == 1:
            # stage 1
            self.solution_old.assign(solution)
            if updateForcings is not None:
                updateForcings(t+dt)
            self.solverK1.solve()
            solution.assign(self.solution_n + 0.25*self.K0 + 0.25*self.K1)
        elif iStage == 2:
            # stage 2
            self.solution_old.assign(solution)
            if updateForcings is not None:
                updateForcings(t+dt/2)
            self.solverK2.solve()
            # final solution
            solution.assign(self.solution_n + (1.0/6.0)*self.K0 +
                            (1.0/6.0)*self.K1 + (2.0/3.0)*self.K2)

    def advance(self, t, dt, solution, updateForcings):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solveStage instead.
        """
        for k in range(3):
            self.solveStage(k, t, dt, solution,
                            updateForcings)


class SSPRK33StageSemiImplicit(timeIntegrator):
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
        self.nStages = 3
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
        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source

        u_old = self.solution_old
        u_0 = self.sol0
        u_1 = self.sol1
        sol = self.equation.solution

        self.F_0 = (massTerm(u_0) - massTerm(u_old) -
                    self.dt_const*(
                        self.theta*RHSi(u_0, **self.args) +
                        (1-self.theta)*RHSi(u_old, **self.args) +
                        RHS(u_old, **self.args) +
                        Source(**self.args))
                    )
        self.F_1 = (massTerm(u_1) -
                    3.0/4.0*massTerm(u_old) - 1.0/4.0*massTerm(u_0) -
                    1.0/4.0*self.dt_const*(
                        self.theta*RHSi(u_1, **self.args) +
                        (1-self.theta)*RHSi(u_0, **self.args) +
                        RHS(u_0, **self.args) +
                        Source(**self.args))
                    )
        self.F_2 = (massTerm(sol) -
                    1.0/3.0*massTerm(u_old) - 2.0/3.0*massTerm(u_1) -
                    2.0/3.0*self.dt_const*(
                        self.theta*RHSi(sol, **self.args) +
                        (1-self.theta)*RHSi(u_1, **self.args) +
                        RHS(u_1, **self.args) +
                        Source(**self.args))
                    )
        self.updateSolver()

    def updateSolver(self):
        """Builds linear problems for each stage. These problems need to be
        re-created after each mesh update."""
        probF0 = NonlinearVariationalProblem(self.F_0, self.sol0)
        self.solverF0 = NonlinearVariationalSolver(probF0,
                                                   solver_parameters=self.solver_parameters)
        probF1 = NonlinearVariationalProblem(self.F_1, self.sol1)
        self.solverF1 = NonlinearVariationalSolver(probF1,
                                                   solver_parameters=self.solver_parameters)
        probF2 = NonlinearVariationalProblem(self.F_2, self.equation.solution)
        self.solverF2 = NonlinearVariationalSolver(probF2,
                                                   solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def solveStage(self, iStage, t, dt, solution, updateForcings=None):
        """
        Solves a single stage of step from t to t+dt.
        All functions that the equation depends on must be at rigth state
        corresponding to each sub-step.
        """
        self.dt_const.assign(dt)
        if iStage == 0:
            # stage 0
            if updateForcings is not None:
                updateForcings(t)
            # BUG there's a bug in assembly cache, need to set to false
            self.solverF0.solve()
            solution.assign(self.sol0)
        elif iStage == 1:
            # stage 1
            if updateForcings is not None:
                updateForcings(t+dt)
            self.solverF1.solve()
            solution.assign(self.sol1)
        elif iStage == 2:
            # stage 2
            if updateForcings is not None:
                updateForcings(t+dt/2)
            self.solverF2.solve()
            self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances one full time step from t to t+dt.
        This assumes that all the functions that the equation depends on are
        constants across this interval. If dependent functions need to be
        updated call solveStage instead.
        """
        for k in range(3):
            self.solveStage(k, t, dt, solution,
                            updateForcings)


class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt, solver_parameters={}):
        """Creates forms for the time integrator"""
        super(ForwardEuler, self).__init__(equation, solver_parameters)
        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source

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
        self.A = massTerm(u_tri)
        self.L = (massTerm(u_old) +
                  self.dt_const*(RHS(u_old, **self.funcs_old) +
                                 RHSi(u_old, **self.funcs_old) +
                                 Source(**self.funcs_old)
                                 )
                  )
        self.updateSolver()

    def updateSolver(self):
        prob = LinearVariationalProblem(self.A, self.L, self.equation.solution)
        self.solver = LinearVariationalSolver(prob,
                                              solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assign values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, solver_parameters={}, gamma=0.5):
        """Creates forms for the time integrator"""
        super(CrankNicolson, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source

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
        self.F = (massTerm(u) - massTerm(u_old) -
                  self.dt_const*(gamma_const*RHS(u, **self.funcs) +
                                 gamma_const*RHSi(u, **self.funcs) +
                                 gamma_const*Source(**self.funcs) +
                                 (1-gamma_const)*RHS(u_old, **self.funcs_old) +
                                 (1-gamma_const)*RHSi(u_old, **self.funcs_old) +
                                 (1-gamma_const)*Source(**self.funcs_old)
                                 )
                  )

        self.A = (massTerm(u_tri) -
                  self.dt_const*(
                      gamma_const*RHS(u_tri, **self.funcs) +
                      gamma_const*RHSi(u_tri, **self.funcs))
                  )
        self.L = (massTerm(u_old) +
                  self.dt_const*(
                      gamma_const*Source(**self.funcs) +
                      (1-gamma_const)*RHS(u_old, **self.funcs_old) +
                      (1-gamma_const)*RHSi(u_old, **self.funcs_old) +
                      (1-gamma_const)*Source(**self.funcs_old))
                  )
        self.updateSolver()

    def updateSolver(self):
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

    def advance(self, t, dt, solution, updateForcings=None):
        """Advances equations for one time step."""
        self.dt_const.assign(dt)
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        self.solver.solve()
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advanceLinear(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'ksponly',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        solve(self.A == self.L, solution, solver_parameters=solver_parameters)
        # shift time
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])


class SSPIMEX(timeIntegrator):
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
        self.dirk = DIRK_LSPUM2(equation, dt,
                                solver_parameters=solver_parameters_dirk,
                                termsToAdd=['implicit'],
                                solution=solution)
        # explicit scheme
        erk_a = [[0, 0, 0],
                 [5.0/6.0, 0, 0],
                 [11.0/24.0, 11.0/24.0, 0]]
        erk_b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
        erk_c = [0, 5.0/6.0, 11.0/12.0]
        self.erk = DIRK_generic(equation, dt, erk_a, erk_b, erk_c,
                                solver_parameters=solver_parameters_dirk,
                                termsToAdd=['explicit', 'source'],
                                solution=solution)
        self.nStages = len(erk_b)

    def updateSolver(self):
        self.dirk.updateSolver()
        self.erk.updateSolver()

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.dirk.initialize(solution)
        self.erk.initialize(solution)

    def advance(self, t, dt, solution, updateForcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.nStages):
            self.solveStage(i, t, dt, solution, updateForcings)
        self.getFinalSolution(solution)

    def solveStage(self, iStage, t, dt, solution, updateForcings=None):
        self.erk.solveStage(iStage, t, dt, solution, updateForcings)
        self.dirk.solveStage(iStage, t, dt, solution, updateForcings)

    def getFinalSolution(self, solution):
        self.erk.getFinalSolution(solution)
        self.dirk.getFinalSolution(solution)


class DIRK_generic(timeIntegrator):
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
                 termsToAdd='all',
                 solution=None):
        """
        Create new DIRK solver.

        Parameters
        ----------
        equation : equation object
            the equation to solve
        dt : float
            time step (constant)
        a  : array_like (nStages, nStages)
            coefficients for the Butcher tableau, must be lower diagonal
        b,c : array_like (nStages,)
            coefficients for the Butcher tableau
        solver_parameters : dict
            PETSc options for solver
        termsToAdd : 'all' or list of 'implicit', 'explicit', 'source'
            Defines which terms of the equation are to be added to this solver.
            Default 'all' implies termsToAdd = ['implicit', 'explicit', 'source']
        """
        super(DIRK_generic, self).__init__(equation, solver_parameters)
        self.solver_parameters.setdefault('snes_monitor', False)
        self.solver_parameters.setdefault('snes_type', 'newtonls')

        self.nStages = len(b)
        self.a = a
        self.b = b
        self.c = c
        self.termsToAdd = termsToAdd

        RHS = self.equation.RHS
        RHSi = self.equation.RHS_implicit
        Source = self.equation.Source
        self.dt = dt
        self.dt_const = Constant(dt)
        if solution is not None:
            self.solution_old = solution
        else:
            self.solution_old = self.equation.solution
        self.funcs = self.equation.kwargs
        dx = self.equation.dx
        test = TestFunction(self.equation.space)

        mixedSpace = isinstance(self.equation.solution.function_space(),
                                MixedFunctionSpace)

        def allTerms(u, **args):
            """Gather all terms that need to be added to the form"""
            f = 0
            if self.termsToAdd == 'all':
                return RHSi(u, **args) + RHS(u, **args) + Source(**args)
            if 'implicit' in self.termsToAdd:
                f += RHSi(u, **args)
            if 'explicit' in self.termsToAdd:
                f += RHS(u, **args)
            if 'source' in self.termsToAdd:
                f += Source(**args)
            # assert f != 0, \
            #     'adding t  erms {:}: empty form'.format(self.termsToAdd)
            return f

        # Allocate tendency fields
        self.k = []
        for i in xrange(self.nStages):
            fname = '{:}_k{:}'.format(self.name, i)
            self.k.append(Function(self.equation.space, name=fname))
        # construct variational problems
        self.F = []
        if not mixedSpace:
            for i in xrange(self.nStages):
                for j in xrange(i+1):
                    if j == 0:
                        u = self.solution_old + self.a[i][j]*self.dt_const*self.k[j]
                    else:
                        u += self.a[i][j]*self.dt_const*self.k[j]
                self.F.append(-inner(self.k[i], test)*dx +
                              allTerms(u, **self.funcs))
        else:
            # solution must be split before computing sum
            # pass components to equation in a list
            for i in xrange(self.nStages):
                for j in xrange(i+1):
                    if j == 0:
                        u = []  # list of components in the mixed space
                        for s, k in zip(split(self.solution_old), split(self.k[j])):
                            u.append(s + self.a[i][j]*self.dt_const*k)
                    else:
                        for l, k in enumerate(split(self.k[j])):
                            u[l] += self.a[i][j]*self.dt_const*k
                self.F.append(-inner(self.k[i], test)*dx +
                              allTerms(u, **self.funcs))
        self.updateSolver()

    def updateSolver(self):
        # construct solvers
        self.solver = []
        for i in xrange(self.nStages):
            p = NonlinearVariationalProblem(self.F[i], self.k[i])
            sname = '{:}_stage{:}_'.format(self.name, i)
            self.solver.append(
                NonlinearVariationalSolver(p,
                                           solver_parameters=self.solver_parameters,
                                           options_prefix=sname))

    def initialize(self, init_cond):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(init_cond)

    def advance(self, t, dt, solution, updateForcings=None):
        """Advances equations for one time step."""
        for i in xrange(self.nStages):
            self.solveStage(i, t, dt, solution, updateForcings)

    def solveStage(self, iStage, t, dt, output=None, updateForcings=None):
        """Advances equations for one stage."""
        if updateForcings is not None:
            updateForcings(t + self.c[iStage]*self.dt)
        self.solver[iStage].solve()
        if output is not None:
            if iStage < self.nStages - 1:
                self.getStageSolution(iStage, output)
            else:
                # assign the final solution
                self.getFinalSolution(output)

    def getStageSolution(self, iStage, output):
        """Stores intermediate solution for stage iStage to the output field"""
        if output != self.solution_old:
            # possible only if output is not the internal state container
            output.assign(self.solution_old)
            for j in xrange(iStage+1):
                output += self.a[iStage][j]*self.dt_const*self.k[j]

    def getFinalSolution(self, output=None):
        """Computes the final solution from the tendencies"""
        # update solution
        for i in xrange(self.nStages):
            self.solution_old += self.dt_const*self.b[i]*self.k[i]
        if output is not None and output != self.solution_old:
            # copy to output
            output.assign(self.solution_old)


class BackwardEuler(DIRK_generic):
    """
    Backward Euler method

    This method has the Butcher tableau

    1   | 1
    ---------
        | 1
    """
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all'):
        a = [[1.0]]
        b = [1.0]
        c = [1.0]
        super(BackwardEuler, self).__init__(equation, dt, a, b, c,
                                            solver_parameters, termsToAdd)


class DIRK22(DIRK_generic):
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
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all'):
        gamma = Constant((2 + np.sqrt(2))/2)
        a = [[gamma, 0], [1-gamma, gamma]]
        b = [0.5, 0.5]
        c = [gamma, 1]
        super(DIRK22, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, termsToAdd)


class DIRK23(DIRK_generic):
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
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all'):
        gamma = (3 + np.sqrt(3))/6
        a = [[gamma, 0], [1-2*gamma, gamma]]
        b = [0.5, 0.5]
        c = [gamma, 1-gamma]
        super(DIRK23, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, termsToAdd)


class DIRK33(DIRK_generic):
    """
    DIRK33, 3-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(3,4,3) IMEX scheme in Ascher et al. (1997)
    """
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all'):
        gamma = 0.4358665215
        b1 = -3.0/2.0*gamma**2 + 4*gamma - 1.0/4.0
        b2 = 3.0/2.0*gamma**2 - 5*gamma + 5.0/4.0
        a = [[gamma, 0, 0],
             [(1-gamma)/2, gamma, 0],
             [b1, b2, gamma]]
        b = [b1, b2, gamma]
        c = [gamma, (1+gamma)/2, 1]
        super(DIRK33, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, termsToAdd)


class DIRK43(DIRK_generic):
    """
    DIRK43, 4-stage, 3rd order, L-stable
    Diagonally Implicit Runge Kutta method

    From DIRK(4,4,3) IMEX scheme in Ascher et al. (1997)
    """
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all'):
        a = [[0.5, 0, 0, 0],
             [1.0/6.0, 0.5, 0, 0],
             [-0.5, 0.5, 0.5, 0],
             [3.0/2.0, -3.0/2.0, 0.5, 0.5]]
        b = [3.0/2.0, -3.0/2.0, 0.5, 0.5]
        c = [0.5, 2.0/3.0, 0.5, 1.0]
        super(DIRK43, self).__init__(equation, dt, a, b, c,
                                     solver_parameters, termsToAdd)


class DIRK_LSPUM2(DIRK_generic):
    """
    DIRK_LSPUM2, 3-stage, 2nd order, L-stable
    Diagonally Implicit Runge Kutta method

    From IMEX RK scheme (17) in Higureras et al. (2014).

    [1] Higueras et al (2014). Optimized strong stability preserving IMEX
        Runge-Kutta methods. Journal of Computational and Applied
        Mathematics 272(2014) 116-140.
    """
    def __init__(self, equation, dt, solver_parameters={}, termsToAdd='all',
                 solution=None):
        a = [[2.0/11.0, 0, 0],
             [205.0/462.0, 2.0/11.0, 0],
             [2033.0/4620.0, 21.0/110.0, 2.0/11.0]]
        b = [24.0/55.0, 1.0/5.0, 4.0/11.0]
        c = [2.0/11.0, 289.0/462.0, 751.0/924.0]
        super(DIRK_LSPUM2, self).__init__(equation, dt, a, b, c,
                                          solver_parameters, termsToAdd,
                                          solution)


def cosTimeAvFilter(M):
    """
    Raised cos time average filters as in older versions of ROMS.
    a_i : weights for t_{n+1}
          sum(a_i) = 1.0, sum(i*a_i/M) = 1.0
    b_i : weights for t_{n+1/2}
          sum(b_i) = 1.0, sum(i*b_i/M) = 0.5

    Filters have lenght 2*M.
    """
    l = np.arange(1, 2*M+1, dtype=float)/M
    # a raised cos centered at M
    a = np.zeros_like(l)
    ix = (l >= 0.5) * (l <= 1.5)
    a[ix] = 1 + np.cos(2*np.pi*(l[ix]-1))
    a /= sum(a)

    # b as in Shchepetkin and MacWilliams 2005
    b = np.cumsum(a[::-1])[::-1]/M
    # correct b to match 2nd criterion exactly
    error = sum(l*b)-0.5
    p = np.linspace(-1, 1, len(b))
    p /= sum(l*p)
    b -= p*error

    M_star = np.nonzero((np.abs(a) > 1e-10) + (np.abs(b) > 1e-10))[0].max()
    if commrank == 0:
        print 'M', M, M_star
        print 'a', sum(a), sum(l*a)
        print 'b', sum(b), sum(l*b)

    return M_star, [float(f) for f in a], [float(f) for f in b]


class macroTimeStepIntegrator(timeIntegrator):
    """Takes an explicit time integrator and iterates it over M time steps.
    Computes time averages to represent solution at M*dt resolution."""
    # NOTE the time averages can be very diffusive
    # NOTE diffusivity depends on M and the choise of time av filter
    # NOTE boxcar filter is very diffusive!
    def __init__(self, timeStepperCls, M, restartFromAv=False):
        super(macroTimeStepIntegrator, self).__init__(self.subiterator.equation)
        self.subiterator = timeStepperCls
        self.M = M
        self.restartFromAv = restartFromAv
        # functions to hold time averaged solutions
        space = self.subiterator.solution_old.function_space()
        self.solution_n = Function(space)
        self.solution_nplushalf = Function(space)
        self.solution_start = Function(space)
        self.M_star, self.w_full, self.w_half = cosTimeAvFilter(M)

    def initialize(self, solution):
        self.subiterator.initialize(solution)
        self.solution_n.assign(solution)
        self.solution_nplushalf.assign(solution)
        self.solution_start.assign(solution)

    def advance(self, t, dt, solution, updateForcings, verbose=False):
        """Advances equations for one macro time step DT=M*dt"""
        M = self.M
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
            self.subiterator.advance(t + i*dt, dt, solution, updateForcings)
            self.solution_nplushalf += self.w_half[i]*solution
            self.solution_n += self.w_full[i]*solution
            if verbose and commrank == 0:
                sys.stdout.write('.')
                if i == M-1:
                    sys.stdout.write('|')
                sys.stdout.flush()
            if not self.restartFromAv and i == M-1:
                # store state at T_{n+1}
                self.solution_start.assign(solution)
        if verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()
        # use filtered solution as output
        solution.assign(self.solution_n)
        if self.restartFromAv:
            self.solution_start.assign(self.solution_n)
