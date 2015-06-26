"""
Time integration schemes for 2D, 3D, and coupled 2D-3D equations.

Tuomas Karna 2015-03-27
"""
from utility import *


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
    p = np.linspace(-1,1,len(b))
    p /= sum(l*p)
    b -= p*error

    ## weird cos filter
    ## TODO check with waveEq if this is any better than uncorrected b above
    #b = np.zeros_like(l)
    #alpha = 0.40

    ##maxiter = 100
    ##tol = 1e-8
    ##ix = (l <= 1.0 + alpha)
    ##for i in range(maxiter):
        ##b[:] = 0
        ##ix = (l <= 1.0 + alpha)
        ##b[ix] = np.cos(np.pi*(l[ix]-alpha)) + 1
        ##b /= sum(b)
        ##err = sum(l*b) - 0.5
        ##print 'alpha', alpha, err
        ##if abs(err) < tol :
            ##break
        ##alpha -= err

    #gtol = 1e-10
    #from scipy.optimize import fmin_bfgs as minimize
    #def costfun(alpha, b, l):
        #b[:] = 0
        #ix = (l <= 1.0 + alpha)
        #b[ix] = np.cos(np.pi*(l[ix]-alpha)) + 1
        #b /= sum(b)
        #return (sum(l*b) - 0.5)**2
    #res = minimize(costfun, 0.4, args=(b, l), gtol=gtol)

    M_star = np.nonzero((np.abs(a) > 1e-10) + (np.abs(b) > 1e-10))[0].max()
    if commrank==0:
      print 'M', M, M_star
      print 'a', sum(a), sum(l*a)
      print 'b', sum(b), sum(l*b)

    return M_star, [float(f) for f in a], [float(f) for f in b]


class timeIntegrator(object):
    """Base class for all time integrator objects."""
    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))

    def advance(self):
        """Advances equations for one time step."""
        raise NotImplementedError(('This method must be implemented '
                                   'in the derived class'))


class macroTimeStepIntegrator(timeIntegrator):
    """Takes an explicit time integrator and iterates it over M time steps.
    Computes time averages to represent solution at M*dt resolution."""
    # NOTE the time averages can be very diffusive
    # NOTE diffusivity depends on M and the choise of time av filter
    # NOTE boxcar filter is very diffusive!
    def __init__(self, timeStepperCls, M, restartFromAv=False):
        self.subiterator = timeStepperCls
        self.equation = self.subiterator.equation
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
    def __init__(self, equation, dt, solver_parameters=None,
                 funcs_nplushalf={}):
        """Creates forms for the time integrator"""
        self.equation = equation
        self.explicit = True
        self.CFL_coeff = 1.0
        self.solver_parameters = solver_parameters

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
    def __init__(self, equation, dt, solver_parameters=None):
        """Creates forms for the time integrator"""
        self.equation = equation
        self.explicit = True
        self.CFL_coeff = 1.0
        self.nstages = 3
        self.solver_parameters = solver_parameters

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
    def __init__(self, equation, dt, solver_parameters=None):
        """Creates forms for the time integrator"""
        self.equation = equation
        self.explicit = True
        self.CFL_coeff = 1.0
        self.nstages = 3
        self.theta = Constant(0.5)
        self.solver_parameters = solver_parameters
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

        #F_CrNi = (massTerm(sol) - massTerm(u_old) -
               #self.dt_const*(
                   #self.theta*RHSi(sol, **self.args) +
                   #(1-self.theta)*RHSi(u_old, **self.args) +
                   #RHS(sol, **self.args) +
                   #Source(**self.args))
               #)
        #probCrNi = NonlinearVariationalProblem(F_CrNi, sol)
        #self.solverCrNi = NonlinearVariationalSolver(probCrNi,
            #solver_parameters=self.solver_parameters)

        self.F_0 = (massTerm(u_0) - massTerm(u_old) -
                    self.dt_const*(
                        self.theta*RHSi(u_0, **self.args) +
                        (1-self.theta)*RHSi(u_old, **self.args) +
                        RHS(u_old, **self.args) +
                        Source(**self.args)
                        )
                    )
        self.F_1 = (massTerm(u_1) -
                    3.0/4.0*massTerm(u_old) - 1.0/4.0*massTerm(u_0) -
                    1.0/4.0*self.dt_const*(
                        self.theta*RHSi(u_1, **self.args) +
                        (1-self.theta)*RHSi(u_0, **self.args) +
                        RHS(u_0, **self.args) +
                        Source(**self.args)
                        )
                    )
        self.F_2 = (massTerm(sol) -
                    1.0/3.0*massTerm(u_old) - 2.0/3.0*massTerm(u_1) -
                    2.0/3.0*self.dt_const*(
                        self.theta*RHSi(sol, **self.args) +
                        (1-self.theta)*RHSi(u_1, **self.args) +
                        RHS(u_1, **self.args) +
                        Source(**self.args)
                        )
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
    def __init__(self, equation, dt, solver_parameters=None):
        """Creates forms for the time integrator"""
        self.equation = equation
        self.solver_parameters = solver_parameters
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
    def __init__(self, equation, dt, solver_parameters={}, gamma=0.6):
        """Creates forms for the time integrator"""
        self.equation = equation
        self.solver_parameters=solver_parameters
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
                      gamma_const*RHSi(u_tri, **self.funcs)
                      )
                  )
        self.L = (massTerm(u_old) +
                  self.dt_const*(
                      gamma_const*Source(**self.funcs) +
                      (1-gamma_const)*RHS(u_old, **self.funcs_old) +
                      (1-gamma_const)*RHSi(u_old, **self.funcs_old) +
                      (1-gamma_const)*Source(**self.funcs_old)
                      )
                  )
        self.updateSolver()

    def updateSolver(self):
        prob = NonlinearVariationalProblem(self.F, self.equation.solution)
        self.solver = LinearVariationalSolver(prob,
            solver_parameters=self.solver_parameters)

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
        #solve(self.F == 0, solution, solver_parameters=solver_parameters)
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


class coupledSSPRK(timeIntegrator):
    """
    Split-explicit time integration that uses SSPRK for both 2d and 3d modes.
    """
    def __init__(self, solver):
        self.solver = solver
        subIterator = SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        self.timeStepper2d = macroTimeStepIntegrator(
            subIterator,
            solver.M_modesplit,
            restartFromAv=True)

        self.timeStepper_mom3d = SSPRK33(
            solver.eq_momentum, solver.dt,
            funcs_nplushalf={'eta': solver.eta3d_nplushalf})
        if self.solver.solveSalt:
            self.timeStepper_salt3d = SSPRK33(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d = CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        # SSPRK33 time integration loop
        with timed_region('mode2d'):
            self.timeStepper2d.advance(t, s.dt_2d, s.solution2d,
                                       updateForcings)
        with timed_region('aux_eta3d'):
            eta_n = s.solution2d.split()[1]
            copy2dFieldTo3d(eta_n, s.eta3d)  # at t_{n+1}
            eta_nph = self.timeStepper2d.solution_nplushalf.split()[1]
            copy2dFieldTo3d(eta_nph, s.eta3d_nplushalf)  # at t_{n+1/2}
        with timed_region('aux_mesh_ale'):
            if s.useALEMovingMesh:
                updateCoordinates(
                    s.mesh, s.eta3d, s.bathymetry3d,
                    s.z_coord3d, s.z_coord_ref3d)
        with timed_region('aux_friction'):
            if s.useBottomFriction:
                computeBottomFriction(
                    s.uv3d, s.uv_bottom2d,
                    s.uv_bottom3d, s.z_coord3d,
                    s.z_bottom2d, s.z_bottom3d,
                    s.bathymetry2d, s.bottom_drag2d,
                    s.bottom_drag3d)
            if s.useParabolicViscosity:
                computeParabolicViscosity(
                    s.uv_bottom3d, s.bottom_drag3d,
                    s.bathymetry3d,
                    s.viscosity_v3d)
        with timed_region('aux_barolinicity'):
            if s.baroclinic:
                computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                      s.baroHead2d, s.baroHeadInt3d,
                                      s.bathymetry3d)

        with timed_region('momentumEq'):
            self.timeStepper_mom3d.advance(t, s.dt, s.uv3d,
                                           updateForcings3d)
        with timed_region('vert_diffusion'):
            if s.solveVertDiffusion:
                self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d, None)
        with timed_region('continuityEq'):
            computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                s.eq_momentum.boundary_markers,
                                s.eq_momentum.bnd_functions)
        with timed_region('aux_mesh_ale'):
            if s.useALEMovingMesh:
                computeMeshVelocity(
                    s.eta3d, s.uv3d, s.w3d,
                    s.w_mesh3d, s.w_mesh_surf3d,
                    s.dw_mesh_dz_3d, s.bathymetry3d,
                    s.z_coord_ref3d)
        with timed_region('aux_friction'):
            if s.useBottomFriction:
                computeBottomFriction(
                    s.uv3d, s.uv_bottom2d,
                    s.uv_bottom3d, s.z_coord3d,
                    s.z_bottom2d, s.z_bottom3d,
                    s.bathymetry2d, s.bottom_drag2d,
                    s.bottom_drag3d)
        with timed_region('supg'):
            if s.useSUPG:
                updateSUPGGamma(
                    s.uv3d, s.w3d, s.u_mag_func,
                    s.u_mag_func_h, s.u_mag_func_v,
                    s.hElemSize3d, s.vElemSize3d,
                    s.SUPG_alpha,
                    s.supg_gamma_h, s.supg_gamma_v)

        with timed_region('saltEq'):
            if s.solveSalt:
                self.timeStepper_salt3d.advance(t, s.dt, s.salt3d,
                                                updateForcings3d)
        with timed_region('gjv'):
            if s.useGJV:
                computeHorizGJVParameter(
                    s.gjv_alpha, s.salt3d, s.nonlinStab_h, s.hElemSize3d,
                    s.u_mag_func_h, maxval=800.0*s.uAdvection.dat.data[0])
                computeVertGJVParameter(
                    s.gjv_alpha, s.salt3d, s.nonlinStab_v, s.vElemSize3d,
                    s.u_mag_func_v, maxval=800.0*s.uAdvection.dat.data[0])
        with timed_region('aux_mom_coupling'):
            bndValue = Constant((0.0, 0.0, 0.0))
            computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                    s.uv3d.function_space(),
                                    bottomToTop=True, bndValue=bndValue,
                                    average=True,
                                    bathymetry=s.bathymetry3d)
            copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                            useBottomValue=False)
            copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav)
            # 2d-3d coupling: restart 2d mode from depth ave 3d velocity
            uv2d_start = self.timeStepper2d.solution_start.split()[0]
            uv2d_start.assign(s.uv2d_dav)


class coupledSSPRKSync(timeIntegrator):
    """
    Split-explicit solves equations with simultaneous SSPRK33 stages.
    """
    def __init__(self, solver):
        self._initialized = False
        self.solver = solver
        self.timeStepper2d = SSPRK33(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()
        self.sol2d_n = Function(fs, name='sol2dtmp')

        self.timeStepper_mom3d = SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.solveSalt:
            self.timeStepper_salt3d = SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d = CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

        # ----- stage 1 -----
        # from n to n+1 with RHS at (u_n, t_n)
        # u_init = u_n
        # ----- stage 2 -----
        # from n+1/4 to n+1/2 with RHS at (u_(1), t_{n+1})
        # u_init = 3/4*u_n + 1/4*u_(1)
        # ----- stage 3 -----
        # from n+1/3 to n+1 with RHS at (u_(2), t_{n+1/2})
        # u_init = 1/3*u_n + 2/3*u_(2)
        # -------------------

        # length of each step (fraction of dt)
        self.dt_frac = [1.0, 1.0/4.0, 2.0/3.0]
        # start of each step (fraction of dt)
        self.start_frac = [0.0, 1.0/4.0, 1.0/3.0]
        # weight to multiply u_n in weighted average to obtain start value
        self.stage_w = [1.0 - self.start_frac[0]]
        for i in range(1, len(self.dt_frac)):
            prev_end_time = self.start_frac[i-1] + self.dt_frac[i-1]
            self.stage_w.append(prev_end_time*(1.0 - self.start_frac[i]))
        printInfo('dt_frac ' + str(self.dt_frac))
        printInfo('start_frac ' + str(self.start_frac))
        printInfo('stage_w ' + str(self.stage_w))

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            M = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/M
            printInfo('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, M, f))
            self.M.append(M)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        s = self.solver
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d.split()[1]
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, s.eta3d, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    computeElemHeight(s.z_coord3d, s.vElemSize3d)
                    copy3dFieldTo2d(s.vElemSize3d, s.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            s.uv3d.function_space(),
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False, elemHeight=s.vElemSize2d)
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav, elemHeight=s.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(s.uv2d_dav)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    uv2d = sol2d.split()[0]
                    s.uv3d -= s.uv3d_dav
                    copy2dFieldTo3d(uv2d, s.uv3d_dav, elemHeight=s.vElemSize3d)
                    s.uv3d += s.uv3d_dav
            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    computeBottomFriction(
                        s.uv3d, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('supg'):
                if s.useSUPG:
                    updateSUPGGamma(
                        s.uv3d, s.w3d, s.u_mag_func,
                        s.u_mag_func_h, s.u_mag_func_v,
                        s.hElemSize3d, s.vElemSize3d,
                        s.SUPG_alpha,
                        s.supg_gamma_h, s.supg_gamma_v)
            with timed_region('gjv'):
                if s.useGJV:
                    computeHorizGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_h, s.hElemSize3d,
                        s.u_mag_func_h, maxval=800.0*s.uAdvection.dat.data[0])
                    computeVertGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_v, s.vElemSize3d,
                        s.u_mag_func_v, maxval=800.0*s.uAdvection.dat.data[0])
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange)

        self.sol2d_n.assign(sol2d)  # keep copy of eta_n
        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, s.salt3d,
                                                       updateForcings3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, s.uv3d)
            with timed_region('mode2d'):
                t_rhs = t + self.start_frac[k]*s.dt
                dt_2d = self.dt_2d[k]
                # initialize
                w = self.stage_w[k]
                sol2d.assign(w*self.sol2d_n + (1.0-w)*sol2d)

                # advance fields from T_{n} to T{n+1}
                for i in range(self.M[k]):
                    self.timeStepper2d.advance(t_rhs + i*dt_2d, dt_2d, sol2d,
                                            updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=lastStep,
                               doStabParams=lastStep)


class coupledSSPRKSemiImplicit(timeIntegrator):
    """
    Solves coupled equations with simultaneous SSPRK33 stages, where 2d gravity
    waves are solved semi-implicitly.
    """
    def __init__(self, solver):
        self._initialized = False
        self.solver = solver
        self.timeStepper2d = SSPRK33StageSemiImplicit(
            solver.eq_sw, solver.dt,
            solver.eq_sw.solver_parameters)
        fs = self.timeStepper2d.solution_old.function_space()

        self.timeStepper_mom3d = SSPRK33Stage(
            solver.eq_momentum, solver.dt)
        if self.solver.solveSalt:
            self.timeStepper_salt3d = SSPRK33Stage(
                solver.eq_salt,
                solver.dt)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d = CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt, gamma=0.6)

        # ----- stage 1 -----
        # from n to n+1 with RHS at (u_n, t_n)
        # u_init = u_n
        # ----- stage 2 -----
        # from n+1/4 to n+1/2 with RHS at (u_(1), t_{n+1})
        # u_init = 3/4*u_n + 1/4*u_(1)
        # ----- stage 3 -----
        # from n+1/3 to n+1 with RHS at (u_(2), t_{n+1/2})
        # u_init = 1/3*u_n + 2/3*u_(2)
        # -------------------

        # length of each step (fraction of dt)
        self.dt_frac = [1.0, 1.0/4.0, 2.0/3.0]
        # start of each step (fraction of dt)
        self.start_frac = [0.0, 1.0/4.0, 1.0/3.0]
        # weight to multiply u_n in weighted average to obtain start value
        self.stage_w = [1.0 - self.start_frac[0]]
        for i in range(1, len(self.dt_frac)):
            prev_end_time = self.start_frac[i-1] + self.dt_frac[i-1]
            self.stage_w.append(prev_end_time*(1.0 - self.start_frac[i]))
        printInfo('dt_frac ' + str(self.dt_frac))
        printInfo('start_frac ' + str(self.start_frac))
        printInfo('stage_w ' + str(self.stage_w))

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.solver.solution2d)
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

        # construct 2d time steps for sub-stages
        self.M = []
        self.dt_2d = []
        for i, f in enumerate(self.dt_frac):
            M = int(np.ceil(f*self.solver.dt/self.solver.dt_2d))
            dt = f*self.solver.dt/M
            printInfo('stage {0:d} {1:.6f} {2:d} {3:.4f}'.format(i, dt, M, f))
            self.M.append(M)
            self.dt_2d.append(dt)
        self._initialized = True

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        if not self._initialized:
            self.initialize()
        s = self.solver
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doALEUpdate=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d.split()[1]
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
                #s.eta3dCG.project(s.eta3d)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh and doALEUpdate:
                    updateCoordinates(
                        s.mesh, s.eta3dCG, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    computeElemHeight(s.z_coord3d, s.vElemSize3d)
                    copy3dFieldTo2d(s.vElemSize3d, s.vElemSize2d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt, s.uv3d)
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            s.uv3d_dav.function_space(),
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False,
                                    elemHeight=s.vElemSize2d)
                    copy2dFieldTo3d(s.uv2d_dav, s.uv3d_dav,
                                    elemHeight=s.vElemSize3d)
                    # 2d-3d coupling: restart 2d mode from depth ave uv3d
                    # NOTE unstable!
                    #uv2d_start = sol2d.split()[0]
                    #uv2d_start.assign(s.uv2d_dav)
                    # 2d-3d coupling v2: force DAv(uv3d) to uv2d
                    s.uv3d -= s.uv3d_dav
                    copy2dFieldTo3d(sol2d.split()[0], s.uv3d_dav,
                                    elemHeight=s.vElemSize3d)
                    s.uv3d += s.uv3d_dav
            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    computeBottomFriction(
                        s.uv3d, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d,
                        s.vElemSize2d, s.vElemSize3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('supg'):
                if s.useSUPG:
                    updateSUPGGamma(
                        s.uv3d, s.w3d, s.u_mag_func,
                        s.u_mag_func_h, s.u_mag_func_v,
                        s.hElemSize3d, s.vElemSize3d,
                        s.SUPG_alpha,
                        s.supg_gamma_h, s.supg_gamma_v)
            with timed_region('gjv'):
                if s.useGJV:
                    computeHorizGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_h, s.hElemSize3d,
                        s.u_mag_func_h, maxval=800.0*s.uAdvection.dat.data[0])
                    computeVertGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_v, s.vElemSize3d,
                        s.u_mag_func_v, maxval=800.0*s.uAdvection.dat.data[0])
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange)

        for k in range(len(self.dt_frac)):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt, s.salt3d,
                                                       updateForcings3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt, s.uv3d)
            with timed_region('mode2d'):
                self.timeStepper2d.solveStage(k, t, s.dt, sol2d,
                                              updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=lastStep,
                               doALEUpdate=lastStep,
                               doStabParams=lastStep)


class coupledSSPRKSingleMode(timeIntegrator):
    """
    Split-explicit SSPRK33 solver without mode-splitting.
    """
    def __init__(self, solver):
        self.solver = solver
        self.timeStepper2d = SSPRK33Stage(
            solver.eq_sw, solver.dt_2d,
            solver.eq_sw.solver_parameters)

        self.timeStepper_mom3d = SSPRK33Stage(
            solver.eq_momentum,
            solver.dt_2d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d = SSPRK33Stage(
                solver.eq_salt,
                solver.dt_2d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d = CrankNicolson(
                solver.eq_vertmomentum,
                solver.dt_2d, gamma=0.6)

        # ----- stage 1 -----
        # from n to n+1 with RHS at (u_n, t_n)
        # u_init = u_n
        # ----- stage 2 -----
        # from n+1/4 to n+1/2 with RHS at (u_(1), t_{n+1})
        # u_init = 3/4*u_n + 1/4*u_(1)
        # ----- stage 3 -----
        # from n+1/3 to n+1 with RHS at (u_(2), t_{n+1/2})
        # u_init = 1/3*u_n + 2/3*u_(2)
        # -------------------

    def initialize(self):
        """Assign initial conditions to all necessary fields"""
        self.timeStepper2d.initialize(self.solver.solution2d.split()[1])
        self.timeStepper_mom3d.initialize(self.solver.uv3d)
        if self.solver.solveSalt:
            self.timeStepper_salt3d.initialize(self.solver.salt3d)
        if self.solver.solveVertDiffusion:
            self.timeStepper_vmom3d.initialize(self.solver.uv3d)

    def advance(self, t, dt, updateForcings=None, updateForcings3d=None):
        """Advances the equations for one time step"""
        s = self.solver
        sol2d_old = self.timeStepper2d.solution_old
        sol2d = self.timeStepper2d.equation.solution

        def updateDependencies(do2DCoupling=False,
                               doVertDiffusion=False,
                               doStabParams=False):
            """Updates all dependencies of the primary variables"""
            with timed_region('aux_eta3d'):
                eta = sol2d
                copy2dFieldTo3d(eta, s.eta3d)  # at t_{n+1}
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    updateCoordinates(
                        s.mesh, s.eta3d, s.bathymetry3d,
                        s.z_coord3d, s.z_coord_ref3d)
                    # need to destroy all cached solvers!
                    linProblemCache.clear()
                    self.timeStepper_mom3d.updateSolver()
                    if s.solveSalt:
                        self.timeStepper_salt3d.updateSolver()
                    if s.solveVertDiffusion:
                        self.timeStepper_vmom3d.updateSolver()
            with timed_region('vert_diffusion'):
                if doVertDiffusion and s.solveVertDiffusion:
                        self.timeStepper_vmom3d.advance(t, s.dt_2d, s.uv3d)
            with timed_region('continuityEq'):
                computeVertVelocity(s.w3d, s.uv3d, s.bathymetry3d,
                                    s.eq_momentum.boundary_markers,
                                    s.eq_momentum.bnd_functions)
            with timed_region('aux_mesh_ale'):
                if s.useALEMovingMesh:
                    computeMeshVelocity(
                        s.eta3d, s.uv3d, s.w3d,
                        s.w_mesh3d, s.w_mesh_surf3d,
                        s.w_mesh_surf2d,
                        s.dw_mesh_dz_3d, s.bathymetry3d,
                        s.z_coord_ref3d)
            with timed_region('aux_friction'):
                if s.useBottomFriction:
                    computeBottomFriction(
                        s.uv3d, s.uv_bottom2d,
                        s.uv_bottom3d, s.z_coord3d,
                        s.z_bottom2d, s.z_bottom3d,
                        s.bathymetry2d, s.bottom_drag2d,
                        s.bottom_drag3d)
                if s.useParabolicViscosity:
                    computeParabolicViscosity(
                        s.uv_bottom3d, s.bottom_drag3d,
                        s.bathymetry3d,
                        s.viscosity_v3d)
            with timed_region('aux_barolinicity'):
                if s.baroclinic:
                    computeBaroclinicHead(s.salt3d, s.baroHead3d,
                                          s.baroHead2d, s.baroHeadInt3d,
                                          s.bathymetry3d)
            with timed_region('supg'):
                if s.useSUPG:
                    updateSUPGGamma(
                        s.uv3d, s.w3d, s.u_mag_func,
                        s.u_mag_func_h, s.u_mag_func_v,
                        s.hElemSize3d, s.vElemSize3d,
                        s.SUPG_alpha,
                        s.supg_gamma_h, s.supg_gamma_v)
            with timed_region('gjv'):
                if s.useGJV:
                    computeHorizGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_h, s.hElemSize3d,
                        s.u_mag_func_h, maxval=800.0*s.uAdvection.dat.data[0])
                    computeVertGJVParameter(
                        s.gjv_alpha, s.salt3d, s.nonlinStab_v, s.vElemSize3d,
                        s.u_mag_func_v, maxval=800.0*s.uAdvection.dat.data[0])
            with timed_region('aux_mom_coupling'):
                if do2DCoupling:
                    bndValue = Constant((0.0, 0.0, 0.0))
                    computeVerticalIntegral(s.uv3d, s.uv3d_dav,
                                            s.uv3d_dav.function_space(),
                                            bottomToTop=True, bndValue=bndValue,
                                            average=True,
                                            bathymetry=s.bathymetry3d)
                    copy3dFieldTo2d(s.uv3d_dav, s.uv2d_dav,
                                    useBottomValue=False, elemHeight=s.vElemSize2d)
                    s.solution2d.split()[0].assign(s.uv2d_dav)
            with timed_region('aux_stabilization'):
                if doStabParams:
                    # update velocity magnitude
                    computeVelMagnitude(s.uv3d_mag, u=s.uv3d)
                    # update P1 velocity field
                    s.uvP1_projector.project()
                    if s.smagorinskyFactor is not None:
                        smagorinskyViscosity(s.uv3d_P1, s.smag_viscosity,
                                             s.smagorinskyFactor, s.hElemSize3d)
                    if s.saltJumpDiffFactor is not None:
                        computeHorizJumpDiffusivity(s.saltJumpDiffFactor, s.salt3d,
                                                    s.saltJumpDiff, s.hElemSize3d,
                                                    s.uv3d_mag, s.saltRange)

        for k in range(self.timeStepper2d.nstages):
            with timed_region('saltEq'):
                if s.solveSalt:
                    self.timeStepper_salt3d.solveStage(k, t, s.dt_2d, s.salt3d,
                                                       updateForcings3d)
            with timed_region('momentumEq'):
                self.timeStepper_mom3d.solveStage(k, t, s.dt_2d, s.uv3d)
            with timed_region('mode2d'):
                uv, eta = s.solution2d.split()
                self.timeStepper2d.solveStage(k, t, s.dt_2d, eta, updateForcings)
            lastStep = (k == 2)
            # move fields to next stage
            updateDependencies(doVertDiffusion=lastStep,
                               do2DCoupling=True,
                               doStabParams=lastStep)
