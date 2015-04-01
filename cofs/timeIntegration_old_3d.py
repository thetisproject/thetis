# NOTE this file is OBSOLETE
# NOTE all these methods should be generalized and implemented in timeIntegration.py

class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
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

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        #self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  #RHS(u_old, **self.funcs_old))

        a = (invdt*massTerm(u_tri))
        L = (invdt*massTerm(u_old) +
             RHS(u_old, **self.funcs_old) +
             Source(**self.funcs_old))
        prob = LinearVariationalProblem(a, L, self.equation.solution)
        self.solver = LinearVariationalSolver(prob)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        # assing values to old functions
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        #solve(self.F == 0, solution, solver_parameters=solver_parameters)
        self.solver.solve()
        #solve(self.A == self.L, solution)
        # store old values
        for k in self.funcs_old:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


class LeapFrogAM3(timeIntegrator):
    """Leap frog - Adams Moulton 3 predictor-corrector scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        invdt = Constant(1.0/dt)

        self.solution_nminushalf = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)
        self.solution_nminusone = Function(self.equation.space)
        self.solution_n = Function(self.equation.space)

        # dict of all input functions at t_{n}
        self.funcs = self.equation.kwargs

        u = self.equation.solution
        u_tri = self.equation.tri

        self.gamma = 1.0/12.0
        self.A = (invdt*massTerm(u_tri))
        self.L_predict = (invdt*massTerm(self.solution_nminushalf) +
                          RHS(self.solution_n, **self.funcs) +
                          Source(**self.funcs))
        self.L_correct = (invdt*massTerm(self.solution_n) +
                          RHS(self.solution_nplushalf, **self.funcs) +
                          Source(**self.funcs))

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_nminusone.assign(solution)
        self.solution_n.assign(solution)

    def predict(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n-1/2} to t_{n+1/2}.
        RHS is evaluated at t_{n}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t)
        # all self.funcs_old need to be at t_{n}
        self.solution_nminushalf.assign((0.5-2*self.gamma)*self.solution_nminusone +
                                        (0.5+2*self.gamma)*self.solution_n)
        solve(self.A == self.L_predict, solution)
        self.solution_nplushalf.assign(solution)

    def correct(self, t, dt, solution, updateForcings):
        """Advances equations from t_{n} to t_{n+1}
        RHS is evaluated at t_{n+1/2}."""
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_monitor': True,
        }
        if updateForcings is not None:
            updateForcings(t+dt/2)
        # all self.funcs_nplushalf need to be at t_{n+1/2}

        solve(self.A == self.L_correct, solution)
        # shift time
        self.solution_nminusone.assign(self.solution_n)
        self.solution_n.assign(solution)


class AdamsBashforth3(timeIntegrator):
    """Standard 3rd order Adams-Bashforth scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        RHS = self.equation.RHS
        Source = self.equation.Source

        self.solution_old = Function(self.equation.space)

        self.K1 = Function(self.equation.space)
        self.K2 = Function(self.equation.space)
        self.K3 = Function(self.equation.space)

        # dict of all input functions needed for the equation
        self.funcs = self.equation.kwargs
        # create functions to hold the values of previous time step
        self.funcs_old = {}
        for k in self.funcs:
            self.funcs_old[k] = Function(self.funcs[k].function_space())
        # assing values to old functions
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri

        # mass matrix for a linear equation
        self.A = massTerm(u_tri)
        self.RHS = RHS(u_old, **self.funcs_old) + Source(**self.funcs_old)

    def initialize(self, solution):
        """Assigns initial conditions to all required fields."""
        self.solution_old.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
        """Advances equations for one time step."""
        if updateForcings is not None:
            updateForcings(t+dt/2)
        #for k in self.funcs:
            #self.funcs_old[k].assign(0.5*(self.funcs[k]+self.funcs_old[k]))

        solve(self.A == self.RHS, self.K1)
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
        for k in self.funcs:
            self.funcs_old[k].assign(self.funcs[k])
        self.solution_old.assign(solution)


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
    def __init__(self, equation, dt, funcs_nplushalf={}):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

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
        self.funcs_nplushalf = funcs_nplushalf
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
        self.solverK0 = LinearVariationalSolver(probK0)
        probK1 = LinearVariationalProblem(a_RK, L_RK, self.K1)
        self.solverK1 = LinearVariationalSolver(probK1)
        probK2 = LinearVariationalProblem(a_RK, L_RK, self.K2)
        self.solverK2 = LinearVariationalSolver(probK2)

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


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, gamma=0.6):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
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

        u = self.equation.solution
        u_old = self.solution_old
        u_tri = self.equation.tri
        #Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (invdt*massTerm(u) - invdt*massTerm(u_old) -
                  gamma_const*RHS(u, **self.funcs) -
                  gamma_const*Source(**self.funcs) -
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) -
                  (1-gamma_const)*Source(**self.funcs_old))

        self.A = (invdt*massTerm(u_tri) -
                  gamma_const*RHS(u_tri, **self.funcs))
        self.L = (invdt*massTerm(u_old) + gamma_const*Source(**self.funcs) +
                  (1-gamma_const)*RHS(u_old, **self.funcs_old) +
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
        }
        if updateForcings is not None:
            updateForcings(t+dt)
        self.solution_old.assign(solution)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
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
