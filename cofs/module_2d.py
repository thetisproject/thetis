"""
Depth averaged shallow water equations

Tuomas Karna 2015-02-23
"""
from utility import *
from physical_constants import *
commrank = op2.MPI.comm.rank

g_grav = physical_constants['g_grav']
viscosity_h = physical_constants['viscosity_h']
wd_alpha = physical_constants['wd_alpha']
mu_manning = physical_constants['mu_manning']


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
    a = np.zeros_like(l)
    ix = (l >= 0.5) * (l < 1.5)
    a[ix] = 1 + np.cos(2*np.pi*(l[ix]-1))
    a /= sum(a)
    # b as in Shchepetkin and MacWilliams 2005
    b = np.cumsum(a[::-1])[::-1]/M
    # correct b to match 2nd criterion exactly
    error = sum(l*b)-0.5
    p = np.linspace(-1,1,len(b))
    p /= sum(l*p)
    b -= p*error
    print 'a', sum(a), sum(l*a)
    print 'b', sum(b), sum(l*b)
    return [float(f) for f in a], [float(f) for f in b]


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
        self.w_full, self.w_half = cosTimeAvFilter(M)

    def initialize(self, solution):
        self.timeStepper.initialize(solution)
        self.solution_n.assign(solution)
        self.solution_nplushalf.assign(solution)
        self.solution_start.assign(solution)

    def advance(self, t, dt, solution, updateForcings):
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
        verbose = False
        if verbose and commrank == 0:
            sys.stdout.write('Solving 2D ')
        for i in range(M):
            self.timeStepper.advance(t + i*dt, dt, solution, updateForcings)
            self.solution_nplushalf += self.w_half[i]*solution
            self.solution_n += self.w_full[i]*solution
            if verbose and commrank == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        if verbose and commrank == 0:
            sys.stdout.write('|')
            sys.stdout.flush()
        # store state at T_{n+1}
        self.solution_start.assign(solution)
        # advance fields from T_{n+1} to T{n+2}
        for i in range(M):
            self.timeStepper.advance(t + (M+i)*dt, dt, solution, updateForcings)
            self.solution_n += self.w_full[M+i]*solution
            if verbose and commrank == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
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


class freeSurfaceEquations(equation):
    """2D depth averaged shallow water equations"""
    def __init__(self, mesh, space, solution, bathymetry,
                 uv_bottom=None, bottom_drag=None,
                 nonlin=True, use_wd=True):
        self.mesh = mesh
        self.space = space
        self.U_space, self.eta_space = self.space.split()
        self.solution = solution
        self.U, self.eta = split(self.solution)
        self.bathymetry = bathymetry
        self.use_wd = use_wd
        self.nonlin = nonlin
        # this dict holds all time dep. args to the equation
        self.kwargs = {'uv_old': self.solution.split()[0],
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       }

        # create mixed function space
        self.tri = TrialFunction(self.space)
        self.test = TestFunction(self.space)
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.U_is_DG = 'DG' in self.U_space.ufl_element().shortstr()
        self.eta_is_DG = 'DG' in self.eta_space.ufl_element().shortstr()

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS = dS(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = set(self.mesh.exterior_facets.unique_markers)

        # compute lenth of all boundaries
        self.boundary_len = {}
        for i in self.boundary_markers:
            ds_restricted = Measure('ds', subdomain_id=int(i))
            one_func = Function(self.eta_space).interpolate(Expression(1.0))
            self.boundary_len[i] = assemble(one_func * ds_restricted)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(0.0)):
        csize = CellSize(self.mesh)
        H = self.bathymetry.function_space()
        h_pos = Function(H, name='bathymetry')
        h_pos.assign(self.bathymetry)
        vect = h_pos.vector()
        vect.set_local(np.maximum(vect.array(), 0.05))
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / (sqrt(g_grav * h_pos) + Umag) * self.dx
        solve(a == L, res)
        return res

    def getTimeStepAdvection(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = self.bathymetry.function_space()
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / Umag * self.dx
        solve(a == L, res)
        return res

    def wd_bath_displacement(self, eta):
        h_add = 0.05  # additional depth for H>=0
        H_0 = 12.0  # neg depth where additional depth goes to zero
        return 0.5 * (sqrt((eta + self.bathymetry)**2 + wd_alpha**2) -
                      (eta + self.bathymetry)) + h_add

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        uv, eta = split(solution)

        # Mass term of momentum equation
        M_momentum = inner(uv, self.U_test)
        F += M_momentum

        # Mass term of free surface equation
        M_continuity = inner(eta, self.eta_test)
        if self.use_wd:
            M_continuity += inner(self.wd_bath_displacement(eta),
                                  self.eta_test)
        F += M_continuity

        return F * self.dx

    def massTermBasic(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        uv, eta = split(solution)

        # Mass term of momentum equation
        M_momentum = inner(uv, self.U_test)
        F += M_momentum

        # Mass term of free surface equation
        M_continuity = inner(eta, self.eta_test)
        F += M_continuity

        return F * self.dx

    def supgMassTerm(self, solution, eta, uv):
        """Additional term for SUPG stabilization"""
        F = 0
        uv_diff, eta_diff = split(solution)

        residual = eta_diff
        # F += (1.0/dt_const)*stabilization_SU*inner(residual,dot(uv,
        # nabla_grad(v)) ) #+ (eta+h_mean)*tr(nabla_grad(w)))
        residual = uv_diff
        # TODO test new SUPG terms! better? can reduce nu?
        # F += (1.0/dt_const)*stabilization_SU*inner(residual,dot(uv,
        # diag(nabla_grad(w))) ) #+ (eta+h_mean)*nabla_grad(v) )
        return F * self.dx

    def RHS(self, solution, uv_old=None, uv_bottom=None, bottom_drag=None):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        uv, eta = split(solution)

        # Advection of momentum
        if self.nonlin:
            # d/dxi( u_i w_j ) u_j
            Adv_mom = -(Dx(uv[0]*self.U_test[0], 0)*uv[0] +
                        Dx(uv[0]*self.U_test[1], 0)*uv[1] +
                        Dx(uv[1]*self.U_test[0], 1)*uv[0] +
                        Dx(uv[1]*self.U_test[1], 1)*uv[1])
            #Adv_mom = -inner(nabla_div(outer(uv, self.U_test)), uv)
            if self.U_is_DG:
                #H = self.bathymetry + eta
                ##un = dot(uv, self.normal)
                ##c_roe = avg(sqrt(g_grav*H))
                ##s2 = sign(avg(un) + c_roe) # 1
                ##s3 = sign(avg(un) - c_roe) # -1
                ##Huv = H*uv
                ##un_roe = avg(sqrt(H)*un)/avg(sqrt(H))
                ##Huv_rie = avg(Huv) + (s2+s3)/2*jump(Huv) + (s2-s3)/2*un_roe/c_roe*(jump(Huv))
                ##uv_rie = Huv_rie/avg(H)

                ##Huv_rie = avg(Huv) + un_roe/c_roe*(jump(Huv))
                ##uv_rie = Huv_rie/avg(H)

                uv_av = avg(uv)
                s = 0.5*(sign(uv_av[0]*self.normal('-')[0] +
                              uv_av[1]*self.normal('-')[1]) + 1.0)
                uv_up = uv('-')*s + uv('+')*(1-s)
                G += (uv_av[0]*uv_av[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_av[0]*uv_av[1]*jump(self.U_test[1], self.normal[0]) +
                      uv_av[1]*uv_av[0]*jump(self.U_test[0], self.normal[1]) +
                      uv_av[1]*uv_av[1]*jump(self.U_test[1], self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                gamma = abs(self.normal[0]('-')*uv_av[0] +
                            self.normal[1]('-')*uv_av[1])
                G += gamma*dot(jump(self.U_test), jump(uv))*self.dS

            F += Adv_mom * self.dx

        # External pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.U_test) * self.dx

        if self.nonlin and self.use_wd:
            total_H = self.bathymetry + eta + self.wd_bath_displacement(eta)
        elif self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry
        # Divergence of depth-integrated velocity
        F += -total_H * inner(uv, nabla_grad(self.eta_test)) * self.dx
        if self.eta_is_DG:
            Hu_star = avg(total_H*uv) +\
                sqrt(g_grav*avg(total_H))*jump(total_H, self.normal)
            G += inner(jump(self.eta_test, self.normal), Hu_star)*self.dS

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

            elif 'slip_alpha' in funcs:
                # partial slip bnd condition
                # viscosity: nu*du/dn*w
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut = dot(uv, t)
                slipFactor = funcs['slip_alpha']
                G += viscosity_h * \
                    inner(slipFactor*viscosity_h*ut*t, self.U_test)*ds_bnd

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                uv_ext = uv
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut_in = dot(uv, t)
                # ut_ext = -dot(uv_ext,t) # assume zero
                un_in = dot(uv, self.normal)
                un_ext = dot(uv_ext, self.normal)

                if self.nonlin:
                    H = self.bathymetry + (eta + h_ext) / 2
                else:
                    H = self.bathymetry
                c_roe = sqrt(g_grav * H)
                un_riemann = dot(uv, self.normal) + c_roe / H * (eta - h_ext)/2
                H_riemann = H
                ut_riemann = tanh(4 * un_riemann / 0.02) * (ut_in)
                uv_riemann = un_riemann * self.normal + ut_riemann * t

                G += H_riemann * un_riemann * self.eta_test * ds_bnd
                if self.nonlin:
                    G += un_riemann * un_riemann * dot(self.normal, self.U_test) * ds_bnd
                # added correct flux for eta
                G += g_grav * ((h_ext - eta) / 2) * \
                    inner(self.normal, self.U_test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal velocity (negative into domain)
                un_ext = funcs['un']
                un_in = dot(uv, self.normal)
                un_riemann = (un_in + un_ext)/2
                G += total_H * un_riemann * self.eta_test * ds_bnd
                if self.nonlin:
                    G += un_riemann*un_riemann*inner(self.normal, self.U_test)*ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                un_av = (un_in + un_ext)/2
                G += total_H * un_av * self.eta_test * ds_bnd
                if self.nonlin:
                    #s = 0.5*(sign(un_av) + 1.0)
                    #un_up = un_in*s + un_ext*(1-s)
                    #G += un_av*un_av*inner(self.normal, self.U_test)*ds_bnd
                    uv_av = 0.5*(uv + un_ext*self.normal)
                    G += (uv_av[0]*uv_av[0]*(self.U_test[0]*self.normal[0]) +
                          uv_av[0]*uv_av[1]*(self.U_test[1]*self.normal[0]) +
                          uv_av[1]*uv_av[0]*(self.U_test[0]*self.normal[1]) +
                          uv_av[1]*uv_av[1]*(self.U_test[1]*self.normal[1]))*ds_bnd

                    # Lax-Friedrichs stabilization
                    gamma = abs(self.normal[0]*uv_av[0] +
                                self.normal[1]*uv_av[1])
                    G += gamma*dot(self.U_test, (uv - un_ext*self.normal)/2)*ds_bnd

            elif 'radiation':
                # prescribe radiation condition that allows waves to pass tru
                un_ext = sqrt(g_grav / total_H) * eta
                G += total_H * un_ext * self.eta_test * ds_bnd
                G += un_ext * un_ext * inner(self.normal, self.U_test) * ds_bnd

        # Quadratic drag
        BottomFri = g_grav * mu_manning ** 2 * \
            total_H ** (-4. / 3.) * sqrt(dot(uv_old, uv_old)) * inner(self.U_test, uv)*self.dx
        F += BottomFri

        # bottom friction from a 3D model
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_H
            BotFriction = (stress[0]*self.U_test[0] +
                           stress[1]*self.U_test[1]) * self.dx
            F += BotFriction

        # viscosity
        # A double dot product of the stress tensor and grad(w).
        Diff_mom = -viscosity_h * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                   Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                   Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                   Dx(uv[1], 1) * Dx(self.U_test[1], 1))
        Diff_mom += viscosity_h/total_H*inner(dot(grad(total_H), grad(uv)),
                                              self.U_test)
        F -= Diff_mom * self.dx

        return -F - G

    def Source(self, uv_old=None, uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0*self.dx  # holds all dx volume integral terms

        return -F
