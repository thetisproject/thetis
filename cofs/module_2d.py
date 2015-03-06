"""
Depth averaged shallow water equations

Tuomas Karna 2015-02-23
"""
from utility import *
from physical_constants import *

g_grav = physical_constants['g_grav']
viscosity_h = physical_constants['viscosity_h']
wd_alpha = physical_constants['wd_alpha']
mu_manning = physical_constants['mu_manning']


class AdamsBashforth3(timeIntegrator):
    """Standard 3rd order Adams-Bashforth scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        dx = self.equation.dx
        U_test = self.equation.U_test
        U_tri = self.equation.U_tri
        eta_test = self.equation.eta_test
        eta_tri = self.equation.eta_tri

        self.dt = dt

        self.solution_old = Function(self.equation.space)
        self.U_old, self.eta_old = split(self.solution_old)

        # time filtered solutions for coupling with 3D equations
        self.solution_n = Function(self.equation.space)
        self.solution_nplushalf = Function(self.equation.space)

        self.K1 = Function(self.equation.space)
        K1_u, K1_h = split(self.K1)
        self.K2 = Function(self.equation.space)
        K2_u, K2_h = split(self.K2)
        self.K3 = Function(self.equation.space)

        eta = self.equation.eta
        U = self.equation.U
        # mass matrix for a linear equation
        self.a = massTermBasic(eta_tri, U_tri)
        self.RHS = RHS(self.eta_old, self.U_old, self.U_old)

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
        dt_const_inv = Constant(1.0/dt)
        updateForcings(t+dt)
        solve(self.a == self.RHS, self.K1)
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

    def advanceMacroStep(self, t, dt, M, solution, updateForcings):
        """Advances equations for one macro time step DT=M*dt"""
        U, eta = solution.split()
        U_old, eta_old = self.solution_old.split()
        # filtered to T_{n+1/2}
        U_nph, eta_nph = self.solution_nplushalf.split()
        # filtered to T_{n+1}
        U_n, eta_n = self.solution_n.split()
        # initialize from time averages
        U_old.assign(U_n)
        eta_old.assign(eta_n)
        # reset time filtered solutions
        eta_nph.dat.data[:] = eta_old.dat.data[:]
        U_nph.dat.data[:] = U_old.dat.data[:]
        eta_n.dat.data[:] = eta_old.dat.data[:]
        U_n.dat.data[:] = U_old.dat.data[:]

        # advance fields from T_{n} to T{n+1}
        sys.stdout.write('Solving 2D ')
        for i in range(M):
            self.advance(t + i*dt, dt, solution, updateForcings)
            U_old.assign(U)
            eta_old.assign(eta)
            eta_nph.dat.data[:] += eta.dat.data[:]
            U_nph.dat.data[:] += U.dat.data[:]
            eta_n.dat.data[:] += eta.dat.data[:]
            U_n.dat.data[:] += U.dat.data[:]
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('|')
        sys.stdout.flush()
        eta_nph.dat.data[:] /= (M+1)
        U_nph.dat.data[:] /= (M+1)
        # advance fields from T_{n+1} to T{n+2}
        for i in range(M):
            self.advance(t + (M+1)*i*dt, dt, solution, updateForcings)
            U_old.assign(U)
            eta_old.assign(eta)
            eta_n.dat.data[:] += eta.dat.data[:]
            U_n.dat.data[:] += U.dat.data[:]
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        eta_n.dat.data[:] /= (2*M+1)
        U_n.dat.data[:] /= (2*M+1)
        # use filtered solution as output
        U.assign(U_n)
        eta.assign(eta_n)


class ForwardEuler(timeIntegrator):
    """Standard forward Euler time integration scheme."""
    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        dx = self.equation.dx
        U_test = self.equation.U_test
        U_tri = self.equation.U_tri
        eta_test = self.equation.eta_test
        eta_tri = self.equation.eta_tri

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)
        self.U_old, self.eta_old = split(self.solution_old)
        eta = self.equation.eta
        U = self.equation.U
        self.F = (invdt*massTerm(eta, U) - invdt*massTerm(self.eta_old, self.U_old) +
                  supgMassTerm(eta-self.eta_old, U-self.U_old, eta, U) -
                  RHS(self.eta_old, self.U_old, self.U_old))

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
        updateForcings(t+dt)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
        # store old values
        self.solution_old.assign(solution)


class CrankNicolson(timeIntegrator):
    """Standard Crank-Nicolson time integration scheme."""
    def __init__(self, equation, dt, gamma=0.6):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        dx = self.equation.dx
        U_test = self.equation.U_test
        U_tri = self.equation.U_tri
        eta_test = self.equation.eta_test
        eta_tri = self.equation.eta_tri

        invdt = Constant(1.0/dt)

        self.solution_old = Function(self.equation.space)
        self.U_old, self.eta_old = split(self.solution_old)
        eta = self.equation.eta
        U = self.equation.U
        #Crank-Nicolson
        gamma_const = Constant(gamma)
        self.F = (invdt*massTerm(eta, U) - invdt*massTerm(self.eta_old, self.U_old) +
                  supgMassTerm(eta-self.eta_old, U-self.U_old, eta, U) -
                  gamma_const*RHS(eta, U, self.U_old) -
                  (1-gamma_const)*RHS(self.eta_old, self.U_old, self.U_old))

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
        updateForcings(t+dt)
        solve(self.F == 0, solution, solver_parameters=solver_parameters)
        # store old values
        self.solution_old.assign(solution)


class DIRK3(timeIntegrator):
    """Implements 3rd order DIRK time integration method.
    DIRK = Diagonally Implicit Runge Kutta
    """

    def __init__(self, equation, dt):
        """Creates forms for the time integrator"""
        timeIntegrator.__init__(self, equation)

        massTerm = self.equation.massTerm
        massTermBasic = self.equation.massTermBasic
        supgMassTerm = self.equation.supgMassTerm
        RHS = self.equation.RHS
        dx = self.equation.dx
        U_test = self.equation.U_test
        U_tri = self.equation.U_tri
        eta_test = self.equation.eta_test
        eta_tri = self.equation.eta_tri

        invdt = Constant(1.0/dt)
        self.solution1 = Function(self.equation.space)
        self.U1, self.eta1 = split(self.solution1)
        self.solution2 = Function(self.equation.space)
        self.U2, self.eta2 = split(self.solution2)

        self.solution_old = Function(self.equation.space)
        self.U_old, self.eta_old = split(self.solution_old)
        self.solution_nplushalf = Function(self.equation.space)

        self.K1 = Function(self.equation.space)
        K1_u, K1_h = split(self.K1)
        self.K2 = Function(self.equation.space)
        K2_u, K2_h = split(self.K2)

        # 3rd order DIRK time integrator
        self.alpha = (3.0 + sqrt(3.0)) / 6.0
        self.alpha_const = Constant(self.alpha)
        # first 2 steps are implicit => dump all in F, use solution instead of
        # trial functions
        self.K1_RHS = RHS(self.eta1, self.U1, self.U_old)
        self.F_step1 = (invdt*massTerm(self.eta1, self.U1) -
                        invdt*massTerm(self.eta_old, self.U_old) +
                        supgMassTerm(self.eta1 - self.eta_old,
                                     self.U1 - self.U_old,
                                     self.eta1, self.U1) -
                        self.alpha_const*self.K1_RHS)
        self.K2_RHS = RHS(self.eta2, self.U2, self.U_old)
        self.F_step2 = (invdt*massTerm(self.eta2, self.U2) -
                        invdt*massTerm(self.eta_old, self.U_old) +
                        supgMassTerm(self.eta2 - self.eta_old,
                                     self.U2 - self.U_old,
                                     self.eta2, self.U2) -
                        (1 - 2*self.alpha_const)*(inner(U_test, K1_u) +
                                      inner(eta_test, K1_h))*dx -
                        self.alpha_const * self.K2_RHS)
        # last step is linear => separate bilinear form a, with trial funcs,
        # and linear form L
        self.a = invdt*massTermBasic(eta_tri, U_tri)

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
        K1_mix = 0.5
        K2_mix = 0.5
        dt_const_inv = Constant(1.0/dt)
        # 3rd order DIRK
        # updateForcings(t+dt)
        updateForcings(t + self.alpha * dt)
        ## if commrank==0 : print 'Solving F1'
        solve(self.F_step1 == 0, self.solution1, bcs=bcs,
              solver_parameters=solver_parameters)
        ## if commrank==0 : print 'Solving K1'
        solve(self.a == dt_const_inv * self.K1_RHS, self.K1,
              solver_parameters={'ksp_type': 'cg'})
        updateForcings(t + (1 - self.alpha) * dt)
        # if commrank==0 : print 'Solving F2'
        solve(self.F_step2 == 0, self.solution2, bcs=bcs,
              solver_parameters=solver_parameters)
        # if commrank==0 : print 'Solving K2'
        solve(self.a == dt_const_inv * self.K2_RHS, self.K2,
              solver_parameters={'ksp_type': 'cg'})
        # if commrank==0 : print 'Solving F'
        solution.assign(self.solution_old + dt*K1_mix*self.K1 +
                        dt*K2_mix*self.K2)
        # store old values
        self.solution_nplushalf.assign(0.5*solution + 0.5*self.solution_old)
        self.solution_old.assign(solution)


class freeSurfaceEquations(equation):
    """2D depth averaged shallow water equations"""
    def __init__(self, mesh, U_space, eta_space, bathymetry,
                 uv_bottom, bottom_drag,
                 nonlin=True, use_wd=True):
        self.mesh = mesh
        self.eta_space = eta_space
        self.U_space = U_space
        self.bathymetry = bathymetry
        self.use_wd = use_wd
        self.nonlin = nonlin
        self.uv_bottom = uv_bottom
        self.bottom_drag = bottom_drag

        # create mixed function space
        self.space = MixedFunctionSpace([U_space, eta_space])
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.U_is_DG = 'DG' in U_space.ufl_element().shortstr()
        self.eta_is_DG = 'DG' in eta_space.ufl_element().shortstr()

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

        # create solution fields
        self.solution = Function(self.space)
        self.U, self.eta = split(self.solution)

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

    def massTerm(self, eta, uv):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0

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

    def massTermBasic(self, eta, uv):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0

        # Mass term of momentum equation
        M_momentum = inner(uv, self.U_test)
        F += M_momentum

        # Mass term of free surface equation
        M_continuity = inner(eta, self.eta_test)
        F += M_continuity

        return F * self.dx

    def supgMassTerm(self, eta_diff, uv_diff, eta, uv):
        """Additional term for SUPG stabilization"""
        F = 0
        residual = eta_diff
        # F += (1.0/dt_const)*stabilization_SU*inner(residual,dot(uv,
        # nabla_grad(v)) ) #+ (eta+h_mean)*tr(nabla_grad(w)))
        residual = uv_diff
        # TODO test new SUPG terms! better? can reduce nu?
        # F += (1.0/dt_const)*stabilization_SU*inner(residual,dot(uv,
        # diag(nabla_grad(w))) ) #+ (eta+h_mean)*nabla_grad(v) )
        return F * self.dx

    def RHS(self, eta, uv, uv_old):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection of momentum
        if self.nonlin:
            # d/dxi( u_i w_j ) u_j
            Adv_mom = -(Dx(uv[0]*self.U_test[0], 0)*uv[0] +
                        Dx(uv[0]*self.U_test[1], 0)*uv[1] +
                        Dx(uv[1]*self.U_test[0], 1)*uv[0] +
                        Dx(uv[1]*self.U_test[1], 1)*uv[1])
            #Adv_mom = -inner(nabla_div(outer(uv, self.U_test)), uv)
            if self.U_is_DG:
                G += (jump(uv[0]*self.U_test[0], self.normal[0]*uv[0]) +
                      jump(uv[0]*self.U_test[1], self.normal[0]*uv[1]) +
                      jump(uv[1]*self.U_test[0], self.normal[1]*uv[0]) +
                      jump(uv[1]*self.U_test[1], self.normal[1]*uv[1]))*self.dS
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
                    G += un_riemann * dot(uv_riemann, self.U_test) * ds_bnd
                # added correct flux for eta
                G += g_grav * ((h_ext - eta) / 2) * \
                    inner(self.normal, self.U_test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal velocity (negative into domain)
                un_ext = funcs['un']
                G += total_H * un_ext * self.eta_test * ds_bnd
                G += un_ext * un_ext * inner(self.normal, self.U_test) * ds_bnd
            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    un_riemann = un_ext
                else:
                    # lin eqns doesn't compile without this
                    un_riemann = (un_in + un_ext)/2
                G += total_H * un_riemann * self.eta_test * ds_bnd
                if self.nonlin:
                    G += un_ext*un_ext*inner(self.normal, self.U_test)*ds_bnd
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
        stress = self.bottom_drag*sqrt(self.uv_bottom[0]**2 +
                                       self.uv_bottom[1]**2)*self.uv_bottom
        BotFriction = total_H**-1.*(stress[0]*self.U_test[0] + stress[1]*self.U_test[1])*self.dx
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
