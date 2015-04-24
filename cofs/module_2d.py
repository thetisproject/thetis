"""
Depth averaged shallow water equations

Tuomas Karna 2015-02-23
"""
from utility import *

commrank = op2.MPI.comm.rank

g_grav = physical_constants['g_grav']
wd_alpha = physical_constants['wd_alpha']


class shallowWaterEquations(equation):
    """2D depth averaged shallow water equations in non-conservative form"""
    def __init__(self, mesh, space, solution, bathymetry,
                 uv_bottom=None, bottom_drag=None, viscosity_h=None,
                 mu_manning=None, baro_head=None,
                 uvLaxFriedrichs=None,
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
                       'viscosity_h': viscosity_h,
                       'mu_manning': mu_manning,
                       'baro_head': baro_head,
                       'uvLaxFriedrichs': uvLaxFriedrichs,
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

        # compute length of all boundaries
        self.boundary_len = {}
        for i in self.boundary_markers:
            ds_restricted = Measure('ds', subdomain_id=int(i))
            one_func = Function(self.eta_space).interpolate(Expression(1.0))
            self.boundary_len[i] = assemble(one_func * ds_restricted)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

        # Gauss-Seidel
        self.solver_parameters = {
            'ksp_type': 'fgmres',
            'ksp_rtol': 1e-10,  # 1e-12
            'ksp_atol': 1e-10,  # 1e-16
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            # 'fieldsplit_0_ksp_type': 'preonly',
            # 'fieldsplit_0_pc_type': 'jacobi',
            # 'fieldsplit_1_ksp_type': 'preonly',
            # 'fieldsplit_1_pc_type': 'jacobi',
            }

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

    def RHS(self, solution, uv_old=None, uv_bottom=None, bottom_drag=None,
            viscosity_h=None, mu_manning=None, uvLaxFriedrichs=None, **kwargs):
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
                un_av = dot(uv_av, self.normal('-'))
                s = 0.5*(sign(un_av) + 1.0)
                uv_up = uv('-')*s + uv('+')*(1-s)
                # TODO write this with dot() to speed up!
                G += (uv_av[0]*uv_up[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_av[0]*uv_up[1]*jump(self.U_test[1], self.normal[0]) +
                      uv_av[1]*uv_up[0]*jump(self.U_test[0], self.normal[1]) +
                      uv_av[1]*uv_up[1]*jump(self.U_test[1], self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if uvLaxFriedrichs is not None:
                    gamma = abs(un_av)*uvLaxFriedrichs
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
                if viscosity_h is not None:
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
                    s = 0.5*(sign(un_av) + 1.0)
                    uv_up = uv*s + un_ext*self.normal*(1-s)
                    uv_av = 0.5*(uv + un_ext*self.normal)
                    G += (uv_av[0]*uv_up[0]*(self.U_test[0]*self.normal[0]) +
                          uv_av[0]*uv_up[1]*(self.U_test[1]*self.normal[0]) +
                          uv_av[1]*uv_up[0]*(self.U_test[0]*self.normal[1]) +
                          uv_av[1]*uv_up[1]*(self.U_test[1]*self.normal[1]))*ds_bnd

                    # Lax-Friedrichs stabilization
                    un_av = dot(self.normal, uv_av)
                    gamma = abs(un_av)
                    G += gamma*dot(self.U_test, (uv - un_ext*self.normal)/2)*ds_bnd

            elif 'radiation':
                # prescribe radiation condition that allows waves to pass tru
                un_ext = sqrt(g_grav / total_H) * eta
                G += total_H * un_ext * self.eta_test * ds_bnd
                G += un_ext * un_ext * inner(self.normal, self.U_test) * ds_bnd

        # Quadratic drag
        if mu_manning is not None:
            BottomFri = g_grav * mu_manning ** 2 * \
                total_H ** (-4. / 3.) * sqrt(dot(uv_old, uv_old)) * inner(self.U_test, uv)*self.dx
            F += BottomFri

        # bottom friction from a 3D model
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_H
            BotFriction = dot(stress, self.U_test)* self.dx
            F += BotFriction

        # viscosity
        # A double dot product of the stress tensor and grad(w).
        if viscosity_h is not None:
            F_visc = viscosity_h * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                    Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                    Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                    Dx(uv[1], 1) * Dx(self.U_test[1], 1))
            F_visc += -viscosity_h/total_H*inner(
                dot(grad(total_H), grad(uv)),
                self.U_test)
            F += F_visc * self.dx

        return -F - G

    def Source(self, uv_old=None, uv_bottom=None, bottom_drag=None,
               baro_head=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0 * self.dx  # holds all dx volume integral terms

        # Internal pressure gradient
        if baro_head is not None:
            F += g_grav * inner(nabla_grad(baro_head), self.U_test) * self.dx

        return -F


class freeSurfaceEquation(equation):
    """Non-conservative free surface equation written for depth averaged
    velocity"""
    def __init__(self, mesh, space, solution, uv, bathymetry,
                 nonlin=True, use_wd=True):
        self.mesh = mesh
        self.space = space
        self.solution = solution
        self.bathymetry = bathymetry
        self.use_wd = use_wd
        self.nonlin = nonlin
        # this dict holds all time dep. args to the equation
        self.kwargs = {'uv': uv,
                       }

        # create mixed function space
        self.tri = TrialFunction(self.space)
        self.test = TestFunction(self.space)

        self.eta_is_DG = 'DG' in self.space.ufl_element().shortstr()

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

        # compute length of all boundaries
        self.boundary_len = {}
        for i in self.boundary_markers:
            ds_restricted = Measure('ds', subdomain_id=int(i))
            one_func = Function(self.space).interpolate(Expression(1.0))
            self.boundary_len[i] = assemble(one_func * ds_restricted)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

        # default solver parameters
        self.solver_parameters = {
            'ksp_type': 'fgmres',
            'ksp_rtol': 1e-10,  # 1e-12
            'ksp_atol': 1e-10,  # 1e-16
            }

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

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        # Mass term of free surface equation
        M_continuity = inner(solution, self.test)
        F += M_continuity

        return F * self.dx

    def massTermBasic(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        # Mass term of free surface equation
        M_continuity = inner(solution, self.test)
        F += M_continuity

        return F * self.dx

    def supgMassTerm(self, solution, eta, uv):
        """Additional term for SUPG stabilization"""
        F = 0
        return F * self.dx

    def RHS(self, solution, uv, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        eta = solution

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry
        # Divergence of depth-integrated velocity
        F += -total_H * inner(uv, nabla_grad(self.test)) * self.dx
        if self.eta_is_DG:
            Hu_star = avg(total_H*uv) +\
                sqrt(g_grav*avg(total_H))*jump(total_H, self.normal)
            G += inner(jump(self.test, self.normal), Hu_star)*self.dS

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']

            elif 'radiation':
                # prescribe radiation condition that allows waves to pass tru
                un_ext = sqrt(g_grav / total_H) * eta
                G += total_H * un_ext * self.test * ds_bnd

        return -F - G

    def Source(self, uv_old=None, uv_bottom=None, bottom_drag=None,
               baro_head=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0 * self.dx  # holds all dx volume integral terms

        return -F
