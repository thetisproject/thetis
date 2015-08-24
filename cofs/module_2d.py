"""
Depth averaged shallow water equations

Tuomas Karna 2015-02-23
"""
from utility import *

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class shallowWaterEquations(equation):
    """2D depth averaged shallow water equations in non-conservative form"""
    def __init__(self, mesh, space, solution, bathymetry,
                 uv_bottom=None, bottom_drag=None, viscosity_h=None,
                 mu_manning=None, lin_drag=None, baro_head=None,
                 coriolis=None,
                 wind_stress=None,
                 uvLaxFriedrichs=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.U_space, self.eta_space = self.space.split()
        self.solution = solution
        self.U, self.eta = split(self.solution)
        self.bathymetry = bathymetry
        self.nonlin = nonlin
        # this dict holds all time dep. args to the equation
        self.kwargs = {'uv_old': split(self.solution)[0],
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_h': viscosity_h,
                       'mu_manning': mu_manning,
                       'lin_drag': lin_drag,
                       'baro_head': baro_head,
                       'coriolis': coriolis,
                       'wind_stress': wind_stress,
                       'uvLaxFriedrichs': uvLaxFriedrichs,
                       }

        # create mixed function space
        self.tri = TrialFunction(self.space)
        self.test = TestFunction(self.space)
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.U_is_DG = self.U_space.ufl_element().family() != 'Lagrange'
        self.eta_is_DG = self.eta_space.ufl_element().family() != 'Lagrange'
        self.U_is_HDiv = self.U_space.ufl_element().family() == 'Raviart-Thomas'

        self.huByParts = self.U_is_DG or self.U_is_HDiv
        self.gradEtaByParts = self.eta_is_DG
        self.horizAdvectionByParts = True

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = self.mesh._dx
        self.dS = self.mesh._dS
        self.ds = self.mesh._ds

        # boundary definitions
        self.boundary_markers = set(self.mesh.exterior_facets.unique_markers)

        # compute length of all boundaries
        # FIXME not parallel safe!
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
            #'ksp_initial_guess_nonzero': True,
            'ksp_type': 'gmres',
            #'ksp_rtol': 1e-10,  # 1e-12
            #'ksp_atol': 1e-10,  # 1e-16
            'pc_type': 'fieldsplit',
            #'pc_fieldsplit_type': 'additive',
            'pc_fieldsplit_type': 'multiplicative',
            #'pc_fieldsplit_type': 'schur',
            #'pc_fieldsplit_schur_factorization_type': 'diag',
            #'pc_fieldsplit_schur_fact_type': 'FULL',
            #'fieldsplit_velocity_ksp_type': 'preonly',
            #'fieldsplit_pressure_ksp_type': 'preonly',
            #'fieldsplit_velocity_pc_type': 'jacobi',
            #'fieldsplit_pressure_pc_type': 'jacobi',
            }

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
        return inner(solution, self.test)*self.dx

    def pressureGrad(self, head, uv=None, total_H=None, internalPG=False, **kwargs):
        if self.gradEtaByParts:
            f = -g_grav*head*nabla_div(self.U_test)*self.dx
            if uv is not None:
                un = dot(uv, self.normal)
                head_star = avg(head) + 0.5*sqrt(avg(total_H)/g_grav)*jump(un)
            else:
                head_star = avg(head)
            f += g_grav*head_star*jump(self.U_test, self.normal)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None or 'symm' in funcs or internalPG:
                    f += g_grav*head*dot(self.U_test, self.normal)*ds_bnd
        else:
            f = g_grav*inner(grad(head), self.U_test) * self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is not None and 'elev' in funcs:
                    f -= g_grav*head*dot(self.U_test, self.normal)*ds_bnd
        return f

    def HUDivTerm(self, uv, total_H, **kwargs):
        if self.huByParts:
            f = -inner(grad(self.eta_test), total_H*uv)*self.dx
            if self.eta_is_DG:
                Hu_star = avg(total_H*uv) +\
                    0.5*sqrt(g_grav*avg(total_H))*jump(total_H, self.normal)
                f += inner(jump(self.eta_test, self.normal), Hu_star)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is not None and ('symm' in funcs or 'elev' in funcs):
                    f += total_H*inner(self.normal, uv)*self.eta_test*ds_bnd
        else:
            f = div(total_H*uv)*self.eta_test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None or 'un' in funcs:
                    f += -total_H*dot(uv, self.normal)*self.eta_test*ds_bnd
            # f += -avg(total_H)*avg(dot(uv, normal))*jump(self.eta_test)*dS
        return f

    def horizontalAdvection(self, uv, uvLaxFriedrichs):
        if self.horizAdvectionByParts:
            #f = -inner(nabla_div(outer(uv, self.U_test)), uv)
            f = -(Dx(uv[0]*self.U_test[0], 0)*uv[0] +
                  Dx(uv[0]*self.U_test[1], 0)*uv[1] +
                  Dx(uv[1]*self.U_test[0], 1)*uv[0] +
                  Dx(uv[1]*self.U_test[1], 1)*uv[1])*self.dx
            if self.U_is_DG:
                uv_av = avg(uv)
                un_av = dot(uv_av, self.normal('-'))
                s = 0.5*(sign(un_av) + 1.0)
                uv_up = uv('-')*s + uv('+')*(1-s)
                f += (uv_up[0]*jump(self.U_test[0], uv[0]*self.normal[0]) +
                      uv_up[1]*jump(self.U_test[1], uv[0]*self.normal[0]) +
                      uv_up[0]*jump(self.U_test[0], uv[1]*self.normal[1]) +
                      uv_up[1]*jump(self.U_test[1], uv[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if uvLaxFriedrichs is not None:
                    gamma = 0.5*abs(un_av)*uvLaxFriedrichs
                    f += gamma*dot(jump(self.U_test), jump(uv))*self.dS
                    for bnd_marker in self.boundary_markers:
                        funcs = self.bnd_functions.get(bnd_marker)
                        ds_bnd = self.ds(int(bnd_marker))
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            un = dot(uv, self.normal)
                            uv_ext = uv - 2*un*self.normal
                            gamma = 0.5*abs(un)*uvLaxFriedrichs
                            f += gamma*dot(self.U_test, uv-uv_ext)*ds_bnd
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None or not 'un' in funcs:
                    f += (uv[0]*self.U_test[0]*uv[0]*self.normal[0] +
                          uv[1]*self.U_test[1]*uv[0]*self.normal[0] +
                          uv[0]*self.U_test[0]*uv[1]*self.normal[1] +
                          uv[1]*self.U_test[1]*uv[1]*self.normal[1])*ds_bnd
        return f

    def RHS_implicit(self, solution, wind_stress=None,
                     **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        uv, eta = split(solution)

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # External pressure gradient
        F += self.pressureGrad(eta, uv, total_H)

        # Divergence of depth-integrated velocity
        F += self.HUDivTerm(uv, total_H)

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = self.ds(int(bnd_marker))
            if funcs is None:
                continue
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

                # replaced by simple dot(uv, n)*eta_test*ds in HUDivTerm()
                #if self.huByParts:
                #    c_roe = sqrt(g_grav * H)
                #    un_riemann = dot(uv, self.normal) + c_roe / H * (eta - h_ext)/2
                #    H_riemann = H
                #    ut_riemann = tanh(4 * un_riemann / 0.02) * (ut_in)
                #    uv_riemann = un_riemann * self.normal + ut_riemann * t
                #    G += H_riemann * un_riemann * self.eta_test * ds_bnd

                # added correct flux for eta
                G += g_grav * h_ext * \
                    inner(self.normal, self.U_test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal velocity (negative into domain)
                un_ext = funcs['un']
                un_in = dot(uv, self.normal)
                un_riemann = (un_in + un_ext)/2
                G += total_H * un_riemann * self.eta_test * ds_bnd
                if self.gradEtaByParts:
                    G += g_grav * eta * \
                        inner(self.normal, self.U_test) * ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                un_av = (un_in + un_ext)/2
                G += total_H * un_av * self.eta_test * ds_bnd
                G += g_grav * eta * \
                    inner(self.normal, self.U_test) * ds_bnd

            elif 'radiation':
                # prescribe radiation condition that allows waves to pass tru
                un_ext = sqrt(g_grav / total_H) * eta
                G += total_H * un_ext * self.eta_test * ds_bnd
                G += g_grav * eta * \
                    inner(self.normal, self.U_test) * ds_bnd

        return -F - G

    def RHS(self, solution, uv_old=None, uv_bottom=None, bottom_drag=None,
            viscosity_h=None, mu_manning=None, lin_drag=None,
            coriolis=None, wind_stress=None,
            uvLaxFriedrichs=None,
            **kwargs):
        """Returns all terms that are treated explicitly."""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        uv, eta = split(solution)

        # Advection of momentum
        if self.nonlin:
            F += self.horizontalAdvection(uv, uvLaxFriedrichs)

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = self.ds(int(bnd_marker))
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

                if self.nonlin:
                    G += un_riemann * un_riemann * dot(self.normal, self.U_test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal velocity (negative into domain)
                un_ext = funcs['un']
                un_in = dot(uv, self.normal)
                un_riemann = (un_in + un_ext)/2
                if self.nonlin:
                    G += un_riemann*un_riemann*inner(self.normal, self.U_test)*ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                un_av = (un_in + un_ext)/2
                if self.nonlin:
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
                if self.nonlin:
                    G += un_ext * un_ext * inner(self.normal, self.U_test) * ds_bnd

        # Coriolis
        if coriolis is not None:
            F += coriolis*(-uv[1]*self.U_test[0]+uv[0]*self.U_test[1])*self.dx

        # Wind stress
        if wind_stress is not None:
            F += -dot(wind_stress, self.U_test)/total_H/rho_0*self.dx

        # Quadratic drag
        if mu_manning is not None:
            BottomFri = g_grav * mu_manning ** 2 * \
                total_H ** (-4. / 3.) * sqrt(dot(uv_old, uv_old)) * inner(self.U_test, uv)*self.dx
            F += BottomFri

        # Linear drag
        if lin_drag is not None:
            BottomFri = lin_drag*inner(self.U_test, uv)*self.dx
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
        """Returns the source terms that do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        # Internal pressure gradient
        if baro_head is not None:
            F += self.pressureGrad(baro_head, None, None, internalPG=True)

        return -F

class freeSurfaceEquation(equation):
    """Non-conservative free surface equation written for depth averaged
    velocity. This equation can be coupled to 3D mode directly."""
    def __init__(self, mesh, space, solution, uv, bathymetry,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.solution = solution
        self.bathymetry = bathymetry
        self.nonlin = nonlin
        # this dict holds all time dep. args to the equation
        self.kwargs = {'uv': uv,
                       }

        # create mixed function space
        self.tri = TrialFunction(self.space)
        self.test = TestFunction(self.space)

        self.U_is_DG = uv.function_space().ufl_element().family() != 'Lagrange'
        self.eta_is_DG = self.space.ufl_element().family() != 'Lagrange'
        self.U_is_HDiv = uv.function_space().ufl_element().family() == 'Raviart-Thomas'

        self.huByParts = True # self.U_is_DG and not self.U_is_HDiv
        self.gradEtaByParts = self.eta_is_DG

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = self.mesh._dx
        self.dS = self.mesh._dS

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
            'ksp_initial_guess_nonzero': True,
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

    def HUDivTerm(self, uv, total_H, **kwargs):
        if self.huByParts:
            f = -inner(grad(self.test), total_H*uv)*self.dx
            if self.eta_is_DG:
                #f += avg(total_H)*jump(uv*self.test,
                                        #self.normal)*self.dS # NOTE fails
                Hu_star = avg(total_H*uv) +\
                    0.5*sqrt(g_grav*avg(total_H))*jump(total_H, self.normal) # NOTE works
                #Hu_star = avg(total_H*uv) # NOTE fails
                f += inner(jump(self.test, self.normal), Hu_star)*self.dS
                # TODO come up with better stabilization here!
                # NOTE scaling sqrt(gH) doesn't help
        else:
            f = div(total_H*uv)*self.test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None:
                    f += -total_H*dot(uv, self.normal)*self.test*ds_bnd
            # f += -avg(total_H)*avg(dot(uv, normal))*jump(self.test)*dS
        return f

    def RHS_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        return -F - G

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
        F += self.HUDivTerm(uv, total_H)

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = self.ds(int(bnd_marker))
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                raise NotImplementedError('elev boundary condition not implemented')
                h_ext = funcs['elev']

            elif 'flux' in funcs:
                # prescribe normal flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                un_av = (un_in + un_ext)/2
                G += total_H * un_av * self.test * ds_bnd

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
