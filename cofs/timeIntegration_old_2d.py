# NOTE this file is OBSOLETE
# NOTE all these methods should be generalized and implemented in timeIntegration.py

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


class macroTimeStepIntegrator(object):
    """Takes an explicit time integrator and iterates it over M time steps.
    Computes time averages to represent solution at M*dt resolution."""
    # NOTE the time averages can be very diffusive
    # NOTE diffusivity depends on M and the choise of time av filter
    # NOTE boxcar filter is very diffusive!
    def __init__(self, timeStepperCls, M, restartFromAv=False):
        self.timeStepper = timeStepperCls
        self.M = M
        self.restartFromAv = restartFromAv
        # functions to hold time averaged solutions
        space = self.timeStepper.solution_old.function_space()
        self.solution_n = Function(space)
        self.solution_nplushalf = Function(space)
        self.solution_start = Function(space)
        self.M_star, self.w_full, self.w_half = cosTimeAvFilter(M)

    def initialize(self, solution):
        self.timeStepper.initialize(solution)
        self.solution_n.assign(solution)
        self.solution_nplushalf.assign(solution)
        self.solution_start.assign(solution)

    def advance(self, t, dt, solution, updateForcings, verbose=False):
        """Advances equations for one macro time step DT=M*dt"""
        M = self.M
        solution_old = self.timeStepper.solution_old
        if self.restartFromAv:
            # initialize from time averages
            solution_old.assign(self.solution_n)
            solution.assign(self.solution_n)
        else:
            # start from saved istantaneous state
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
            self.timeStepper.advance(t + i*dt, dt, solution, updateForcings)
            self.solution_nplushalf += self.w_half[i]*solution
            self.solution_n += self.w_full[i]*solution
            if verbose and commrank == 0:
                sys.stdout.write('.')
                if i == M-1:
                    sys.stdout.write('|')
                sys.stdout.flush()
            if i == M-1:
                # store state at T_{n+1}
                self.solution_start.assign(solution)
        if verbose and commrank == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()
        # use filtered solution as output
        solution.assign(self.solution_n)


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
    def __init__(self, equation, dt, solver_parameters=None):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)
        self.explicit = True
        self.CFL_coeff = 1.0
        self.solver_parameters = solver_parameters

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        self.solution_old = Function(self.equation.space)

        self.K0 = Function(self.equation.space)
        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())
        # values used in equations
        self.args = {}
        for k in self.funcs_old:
            self.args[k] = Function(self.funcs[k].function_space())

        u_old = self.solution_old
        u_tri = self.equation.tri

        dt_const = Constant(dt)
        a_RK = massTerm(u_tri)
        L_RK = dt_const*(RHS(u_old, **self.args) + Source(**self.args))

        probK0 = LinearVariationalProblem(a_RK, L_RK, self.K0)
        self.solverK0 = LinearVariationalSolver(probK0, solver_parameters=self.solver_parameters)
        probK1 = LinearVariationalProblem(a_RK, L_RK, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1, solver_parameters=self.solver_parameters)
        probK2 = LinearVariationalProblem(a_RK, L_RK, self.K2)
        self.solverK2 = LinearVariationalSolver(probK2, solver_parameters=self.solver_parameters)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""

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


class AdamsBashforth3(timeIntegrator):
    """Standard 3rd order Adams-Bashforth scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)
        self.explicit = True
        self.CFL_coeff = 1.0

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        Source = self.equation.Source
        tri = self.equation.tri

        self.dt = dt

        self.solution_old = Function(self.equation.space)
        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())

        self.K1 = Function(self.equation.space)
        K1_u, K1_h = split(self.K1)
        self.K2 = Function(self.equation.space)
        K2_u, K2_h = split(self.K2)
        self.K3 = Function(self.equation.space)

        eta = self.equation.eta
        U = self.equation.U
        # mass matrix for a linear equation
        a = massTermBasic(tri)
        L = (RHS(self.solution_old, **self.funcs_old) +
             Source(**self.funcs_old))
        probK1 = LinearVariationalProblem(a, L, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
            #'snes_rtol': 1e-4,
            #'snes_stol': 1e-4,
            #'snes_atol': 1e-16,
            'ksp_type': 'fgmres',
            #'ksp_rtol': 1e-4,
            #'ksp_atol': 1e-16,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',  # 'jacobi', #'gamg',
            'fieldsplit_1_pc_type': 'jacobi',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solverK1.solve()
        K1_mix = 23.0/12.0
        K2_mix = -4.0/3.0
        K3_mix = 5.0/12.0
        solution.assign(self.solution_old +
                        dt*K1_mix*self.K1 +
                        dt*K2_mix*self.K2 +
                        dt*K3_mix*self.K3)
        # shift tendencies for next time step
        self.K3.assign(self.K2)
        self.K2.assign(self.K1)
        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)
        self.explicit = True
        self.CFL_coeff = 0.5

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)
        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())

        solution = self.equation.solution
        self.F = (invdt*massTerm(solution) - invdt*massTerm(self.solution_old) -
                  RHS(self.solution_old, **self.funcs_old) -
                  Source(**self.funcs_old))
        self.a = invdt*massTerm(self.equation.tri)
        self.L = (invdt*massTerm(self.solution_old) +
                  RHS(self.solution_old, **self.funcs_old) +
                  Source(**self.funcs_old))

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advanceNonLin(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
            #'snes_rtol': 1e-4,
            #'snes_stol': 1e-4,
            #'snes_atol': 1e-16,
            'ksp_type': 'fgmres',
            #'ksp_rtol': 1e-4,
            #'ksp_atol': 1e-16,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',  # 'jacobi', #'gamg',
            'fieldsplit_1_pc_type': 'jacobi',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
        # store old values
        self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'ksponly',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        solve(self.a == self.L, solution, solver_parameters=solver_parameters)
        # store old values
        self.solution_old.assign(solution)


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, gamma=0.6):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)
        self.explicit = False
        self.CFL_coeff = 0.0

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)
        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())

        solution = self.equation.solution
        #Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (invdt*massTerm(solution) - invdt*massTerm(self.solution_old) -
                  gamma_const*RHS(solution, **self.funcs) -
                  gamma_const*Source(**self.funcs) -
                  (1-gamma_const)*RHS(self.solution_old, **self.funcs_old) -
                  (1-gamma_const)*Source(**self.funcs_old))

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
            #'snes_rtol': 1e-4,
            #'snes_stol': 1e-4,
            #'snes_atol': 1e-16,
            'ksp_type': 'fgmres',
            #'ksp_rtol': 1e-4,
            #'ksp_atol': 1e-16,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',  # 'jacobi', #'gamg',
            'fieldsplit_1_pc_type': 'jacobi',
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
        # store old values
        self.solution_nplushalf.assign(0.5*solution + 0.5*self.solution_old)
        self.solution_old.assign(solution)


class DIRK3(timeIntegrator):
    """Implements 3rd order Diagonally Implicit Runge Kutta time integration
    method, DIRK(2, 3, 3).

    This method has the Butcher tableau (Asher et al. 1997)

    gamma   | gamma     0
    1-gamma | 1-2*gamma gamma
    -------------------------
            | 0.5       0.5
    """

    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)
        self.explicit = False
        self.CFL_coeff = 0.0

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        Source = self.equation.Source
        test = self.equation.test
        dx = self.equation.dx

        invdt = Constant(1.0/dt)
        self.solution1 = Function(self.equation.space)
        self.solution2 = Function(self.equation.space)

        self.solution_old = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)
        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            if self.funcs[k] is not None:
                self.funcs_old[k] = Function(self.funcs[k].function_space())
        # the values to feed to the RHS
        self.args = {}
        for k in self.funcs_old:
            self.args[k] = Function(self.funcs[k].function_space())

        self.K1 = Function(self.equation.space)
        K1_u, K1_h = split(self.K1)
        self.K2 = Function(self.equation.space)
        K2_u, K2_h = split(self.K2)

        # 3rd order DIRK time integrator
        self.alpha = (3.0 + sqrt(3.0)) / 6.0
        self.alpha_const = Constant(self.alpha)
        # first 2 steps are implicit => dump all in F, use solution instead of
        # trial functions
        self.K1_RHS = (RHS(self.solution1, **self.funcs_old) +
                       Source(**self.funcs_old))
        self.F_step1 = (invdt*massTerm(self.solution1) -
                        invdt*massTerm(self.solution_old) -
                        self.alpha_const*self.K1_RHS)
        self.K2_RHS = (RHS(self.solution2, **self.funcs_old) +
                       Source(**self.funcs_old))
        self.F_step2 = (invdt*massTerm(self.solution2) -
                        invdt*massTerm(self.solution_old) -
                        (1 - 2*self.alpha_const)*inner(test, self.K1)*dx -
                        self.alpha_const * self.K2_RHS)
        self.K1_mix = 0.5
        self.K2_mix = 0.5
        # last step is linear => separate bilinear form a, with trial funcs,
        # and linear form L
        self.a = massTermBasic(self.equation.tri)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
            #'snes_rtol': 1e-4,
            #'snes_stol': 1e-4,
            #'snes_atol': 1e-16,
            'ksp_type': 'fgmres',
            #'ksp_rtol': 1e-4,
            #'ksp_atol': 1e-16,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'jacobi',  # 'jacobi', #'gamg',
            'fieldsplit_1_pc_type': 'jacobi',
        }
        bcs = []
        # 3rd order DIRK
        # updateForcings(t+dt)
        if updateForcings is not None:
            updateForcings(t + self.alpha * dt)
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.alpha*self.funcs[k] +
                                     (1.0-self.alpha)*self.funcs_old[k])
        self.solution1.assign(solution)
        ## if commrank==0 : print 'Solving F1'
        solve(self.F_step1 == 0, self.solution1, bcs=bcs,
              solver_parameters=solver_parameters)
        ## if commrank==0 : print 'Solving K1'
        solve(self.a == self.K1_RHS, self.K1,
              solver_parameters={'ksp_type': 'cg'})
        if updateForcings is not None:
            updateForcings(t + (1 - self.alpha) * dt)
        for k in self.funcs_old:
            self.funcs_old[k].assign((1.0-self.alpha)*self.funcs[k] +
                                     self.alpha*self.funcs_old[k])
        self.solution2.assign(self.solution1)
        # if commrank==0 : print 'Solving F2'
        solve(self.F_step2 == 0, self.solution2, bcs=bcs,
              solver_parameters=solver_parameters)
        # if commrank==0 : print 'Solving K2'
        solve(self.a == self.K2_RHS, self.K2,
              solver_parameters={'ksp_type': 'cg'})
        # if commrank==0 : print 'Solving F'
        solution.assign(self.solution_old + dt*self.K1_mix*self.K1 +
                        dt*self.K2_mix*self.K2)
        # store old values
        self.solution_nplushalf.assign(0.5*solution + 0.5*self.solution_old)
        self.solution_old.assign(solution)
