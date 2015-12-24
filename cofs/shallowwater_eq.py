"""
Depth averaged shallow water equations

TODO: add documentation

Boundary conditions are set with ShallowWaterEquations.bnd_functions dict.
For example to assign elevation and volume flux for boundary 1:
sw.bnd_functions[1] = {'elev':myfunc1, 'flux':myfunc2}
where myfunc1 and myfunc2 are Functions in the appropriate function space.

Supported boundary conditions are:

 - 'elev': elevation only (usually unstable)
 - 'uv': 2d velocity vector (in model coordinates)
 - 'un': normal velocity (scalar, positive out of domain)
 - 'flux': normal volume flux (scalar, positive out of domain)
 - 'elev' and 'uv': water elevation and uv vector
 - 'elev' and 'un': water elevation and normal velocity (scalar)
 - 'elev' and 'flux': water elevation and normal flux (scalar)

Tuomas Karna 2015-02-23
"""
from utility import *

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterEquations(Equation):
    """2D depth averaged shallow water equations in non-conservative form"""
    def __init__(self, solution, bathymetry,
                 uv_bottom=None, bottom_drag=None, viscosity_h=None,
                 mu_manning=None, lin_drag=None, baroc_head=None,
                 coriolis=None, wind_stress=None, uv_lax_friedrichs=None,
                 uv_source=None, elev_source=None,
                 nonlin=True):
        self.space = solution.function_space()
        self.mesh = self.space.mesh()
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
                       'baroc_head': baroc_head,
                       'coriolis': coriolis,
                       'wind_stress': wind_stress,
                       'uv_lax_friedrichs': uv_lax_friedrichs,
                       'uv_source': uv_source,
                       'elev_source': elev_source,
                       }

        # create mixed function space
        self.tri = TrialFunction(self.space)
        self.test = TestFunction(self.space)
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.U_is_DG = self.U_space.ufl_element().family() != 'Lagrange'
        self.eta_is_dg = self.eta_space.ufl_element().family() != 'Lagrange'
        self.U_is_HDiv = self.U_space.ufl_element().family() == 'Raviart-Thomas'

        self.hu_by_parts = self.U_is_DG or self.U_is_HDiv
        self.grad_eta_by_parts = self.eta_is_dg
        self.horiz_advection_by_parts = True

        # mesh dependent variables
        self.normal = FacetNormal(self.mesh)
        self.cellsize = CellSize(self.mesh)
        self.xyz = SpatialCoordinate(self.mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = self.mesh._dx
        self.dS = self.mesh._dS
        self.ds = self.mesh._ds

        # boundary definitions
        self.boundary_markers = set(self.mesh.exterior_facets.unique_markers)

        # compute length of all boundaries
        self.boundary_len = {}
        for i in self.boundary_markers:
            ds_restricted = self.ds(int(i))
            one_func = Function(self.eta_space).assign(1.0)
            self.boundary_len[i] = assemble(one_func * ds_restricted)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

        # Gauss-Seidel
        self.solver_parameters = {
            # 'ksp_initial_guess_nonzero': True,
            'ksp_type': 'gmres',
            # 'ksp_rtol': 1e-10,  # 1e-12
            # 'ksp_atol': 1e-10,  # 1e-16
            'pc_type': 'fieldsplit',
            # 'pc_fieldsplit_type': 'additive',
            'pc_fieldsplit_type': 'multiplicative',
            # 'pc_fieldsplit_type': 'schur',
            # 'pc_fieldsplit_schur_factorization_type': 'diag',
            # 'pc_fieldsplit_schur_fact_type': 'FULL',
            # 'fieldsplit_velocity_ksp_type': 'preonly',
            # 'fieldsplit_pressure_ksp_type': 'preonly',
            # 'fieldsplit_velocity_pc_type': 'jacobi',
            # 'fieldsplit_pressure_pc_type': 'jacobi',
        }

    def get_time_step(self, Umag=Constant(0.0)):
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

    def get_time_step_advection(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = self.bathymetry.function_space()
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / Umag * self.dx
        if Umag.dat.data == 0.0:
            raise Exception('Unable to compute time step: zero velocity scale')
        solve(a == L, res)
        return res

    def mass_term(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test)*self.dx

    def get_bnd_functions(self, eta_in, uv_in, funcs, bath, bnd_len):
        """
        Returns external values of elev and uv for all supported
        boundary conditions.

        volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
        if 'elev' in funcs and 'uv' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['uv']
        elif 'elev' in funcs and 'un' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['un']*self.normal
        elif 'elev' in funcs and 'flux' in funcs:
            eta_ext = funcs['elev']
            H_ext = eta_ext + bath
            area = H_ext*bnd_len  # NOTE using external data only
            uv_ext = funcs['flux']/area*self.normal
        elif 'elev' in funcs:
            eta_ext = funcs['elev']
            uv_ext = uv_in  # assume symmetry
        elif 'uv' in funcs:
            eta_ext = eta_in  # assume symmetry
            uv_ext = funcs['uv']
        elif 'un' in funcs:
            eta_ext = eta_in  # assume symmetry
            uv_ext = funcs['un']*self.normal
        elif 'flux' in funcs:
            eta_ext = eta_in  # assume symmetry
            H_ext = eta_ext + bath
            area = H_ext*bnd_len  # NOTE using internal elevation
            uv_ext = funcs['flux']/area*self.normal
        else:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
        return eta_ext, uv_ext

    def pressure_grad(self, head, uv=None, total_h=None, internal_pg=False, **kwargs):
        if self.grad_eta_by_parts:
            f = -g_grav*head*nabla_div(self.U_test)*self.dx
            if uv is not None:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            f += g_grav*head_star*jump(self.U_test, self.normal)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if internal_pg:
                    # use internal value
                    head_rie = head
                    f += g_grav*head_rie*dot(self.U_test, self.normal)*ds_bnd
                else:
                    if funcs is not None:
                        h = self.bathymetry
                        l = self.boundary_len[bnd_marker]
                        eta_ext, uv_ext = self.get_bnd_functions(head, uv, funcs, h, l)
                        # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                        un_jump = inner(uv - uv_ext, self.normal)
                        eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                        f += g_grav*eta_rie*dot(self.U_test, self.normal)*ds_bnd
                    if funcs is None or 'symm' in funcs or internal_pg:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(uv, self.normal)
                        h = self.bathymetry
                        head_rie = head + sqrt(h/g_grav)*un_jump
                        f += g_grav*head_rie*dot(self.U_test, self.normal)*ds_bnd
        else:
            raise Exception('Should never call this')
            f = g_grav*inner(grad(head), self.U_test) * self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is not None and 'elev' in funcs:
                    f -= g_grav*head*dot(self.U_test, self.normal)*ds_bnd
        return f

    def HUDivTerm(self, uv, eta, total_h, **kwargs):
        if self.hu_by_parts:
            f = -inner(grad(self.eta_test), total_h*uv)*self.dx
            if self.eta_is_dg:
                H = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/H)*jump(eta, self.normal)
                Hu_star = H*uv_rie
                # Hu_star = avg(total_h*uv) +\
                #     0.5*sqrt(g_grav*avg(total_h))*jump(total_h, self.normal)
                f += inner(jump(self.eta_test, self.normal), Hu_star)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is not None:
                    h = self.bathymetry
                    l = self.boundary_len[bnd_marker]
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, funcs, h, l)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    H_av = self.bathymetry + 0.5*(eta + eta_ext)
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_jump = eta - eta_ext
                    eta_rie = 0.5*(eta + eta_ext) + sqrt(H_av/g_grav)*un_jump
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/H_av)*eta_jump
                    H_rie = self.bathymetry + eta_rie
                    f += H_rie*un_rie*self.eta_test*ds_bnd
                # if funcs is not None and ('symm' in funcs or 'elev' in funcs):
                #     f += total_h*inner(self.normal, uv)*self.eta_test*ds_bnd
        else:
            f = div(total_h*uv)*self.eta_test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None or 'un' in funcs:
                    f += -total_h*dot(uv, self.normal)*self.eta_test*ds_bnd
            # f += -avg(total_h)*avg(dot(uv, normal))*jump(self.eta_test)*dS
        return f

    def horizontal_advection(self, uv, eta, uv_lax_friedrichs):
        if self.horiz_advection_by_parts:
            # f = -inner(nabla_div(outer(uv, self.U_test)), uv)
            f = -(Dx(uv[0]*self.U_test[0], 0)*uv[0] +
                  Dx(uv[0]*self.U_test[1], 0)*uv[1] +
                  Dx(uv[1]*self.U_test[0], 1)*uv[0] +
                  Dx(uv[1]*self.U_test[1], 1)*uv[1])*self.dx
            if self.U_is_DG:
                uv_av = avg(uv)
                un_av = dot(uv_av, self.normal('-'))
                # NOTE solver can stagnate
                # s = 0.5*(sign(un_av) + 1.0)
                # NOTE smooth sign change between [-0.02, 0.02], slow
                # s = 0.5*tanh(100.0*un_av) + 0.5
                # uv_up = uv('-')*s + uv('+')*(1-s)
                # NOTE mean flux
                uv_up = uv_av
                f += (uv_up[0]*jump(self.U_test[0], uv[0]*self.normal[0]) +
                      uv_up[1]*jump(self.U_test[1], uv[0]*self.normal[0]) +
                      uv_up[0]*jump(self.U_test[0], uv[1]*self.normal[1]) +
                      uv_up[1]*jump(self.U_test[1], uv[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if uv_lax_friedrichs is not None:
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.U_test), jump(uv))*self.dS
                    for bnd_marker in self.boundary_markers:
                        funcs = self.bnd_functions.get(bnd_marker)
                        ds_bnd = self.ds(int(bnd_marker))
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            un = dot(uv, self.normal)
                            uv_ext = uv - 2*un*self.normal
                            gamma = 0.5*abs(un)*uv_lax_friedrichs
                            f += gamma*dot(self.U_test, uv-uv_ext)*ds_bnd
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is not None:
                    h = self.bathymetry
                    l = self.boundary_len[bnd_marker]
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, funcs, h, l)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    uv_av = 0.5*(uv_ext + uv)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h)*eta_jump
                    f += (uv_av[0]*self.U_test[0]*un_rie +
                          uv_av[1]*self.U_test[1]*un_rie)*ds_bnd
                # if funcs is None or not 'un' in funcs:
                #     f += (uv[0]*self.U_test[0]*uv[0]*self.normal[0] +
                #           uv[1]*self.U_test[1]*uv[0]*self.normal[0] +
                #           uv[0]*self.U_test[0]*uv[1]*self.normal[1] +
                #           uv[1]*self.U_test[1]*uv[1]*self.normal[1])*ds_bnd
        return f

    def rhs_implicit(self, solution, wind_stress=None,
                     **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)

        if self.nonlin:
            total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry

        # External pressure gradient
        F += self.pressure_grad(eta, uv, total_h)

        # Divergence of depth-integrated velocity
        F += self.HUDivTerm(uv, eta, total_h)

        return -F - G

    def rhs(self, solution, uv_old=None, uv_bottom=None, bottom_drag=None,
            viscosity_h=None, mu_manning=None, lin_drag=None,
            coriolis=None, wind_stress=None,
            uv_lax_friedrichs=None,
            **kwargs):
        """Returns all terms that are treated explicitly."""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)

        # Advection of momentum
        if self.nonlin:
            F += self.horizontal_advection(uv, eta, uv_lax_friedrichs)

        if self.nonlin:
            total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry

        # Coriolis
        if coriolis is not None:
            F += coriolis*(-uv[1]*self.U_test[0]+uv[0]*self.U_test[1])*self.dx

        # Wind stress
        if wind_stress is not None:
            F += -dot(wind_stress, self.U_test)/total_h/rho_0*self.dx

        # Quadratic drag
        if mu_manning is not None:
            BottomFri = g_grav * mu_manning ** 2 * \
                total_h ** (-4. / 3.) * sqrt(dot(uv_old, uv_old)) * inner(self.U_test, uv)*self.dx
            F += BottomFri

        # Linear drag
        if lin_drag is not None:
            BottomFri = lin_drag*inner(self.U_test, uv)*self.dx
            F += BottomFri

        # bottom friction from a 3D model
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            BotFriction = dot(stress, self.U_test)*self.dx
            F += BotFriction

        # viscosity
        # A double dot product of the stress tensor and grad(w).
        if viscosity_h is not None:
            F_visc = viscosity_h * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                    Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                    Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                    Dx(uv[1], 1) * Dx(self.U_test[1], 1))
            F_visc += -viscosity_h/total_h*inner(
                dot(grad(total_h), grad(uv)),
                self.U_test)
            F += F_visc * self.dx

        return -F - G

    def source(self, uv_source=None, elev_source=None,
               uv_old=None, uv_bottom=None, bottom_drag=None,
               baroc_head=None, **kwargs):
        """Returns the source terms that do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        # Internal pressure gradient
        if baroc_head is not None:
            F += self.pressure_grad(baroc_head, None, None, internal_pg=True)

        if uv_source is not None:
            F += -inner(uv_source, self.U_test)*self.dx
        if elev_source is not None:
            F += -inner(elev_source, self.eta_test)*self.dx

        return -F


class FreeSurfaceEquation(Equation):
    """Non-conservative free surface equation written for depth averaged
    velocity. This equation can be coupled to 3D mode directly."""
    def __init__(self, solution, uv, bathymetry,
                 nonlin=True):
        self.space = solution.function_space()
        self.mesh = self.space.mesh()
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
        self.eta_is_dg = self.space.ufl_element().family() != 'Lagrange'
        self.U_is_HDiv = uv.function_space().ufl_element().family() == 'Raviart-Thomas'

        self.hu_by_parts = True  # self.U_is_DG and not self.U_is_HDiv
        self.grad_eta_by_parts = self.eta_is_dg

        # mesh dependent variables
        self.normal = FacetNormal(self.mesh)
        self.cellsize = CellSize(self.mesh)
        self.xyz = SpatialCoordinate(self.mesh)
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

    def get_time_step(self, Umag=Constant(0.0)):
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

    def get_time_step_advection(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = self.bathymetry.function_space()
        uu = TestFunction(H)
        grid_dt = TrialFunction(H)
        res = Function(H)
        a = uu * grid_dt * self.dx
        L = uu * csize / Umag * self.dx
        solve(a == L, res)
        return res

    def mass_term(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        F = 0
        # Mass term of free surface equation
        M_continuity = inner(solution, self.test)
        F += M_continuity

        return F * self.dx

    def HUDivTerm(self, uv, total_h, **kwargs):
        if self.hu_by_parts:
            f = -inner(grad(self.test), total_h*uv)*self.dx
            if self.eta_is_dg:
                # f += avg(total_h)*jump(uv*self.test,
                #                        self.normal)*self.dS # NOTE fails
                Hu_star = avg(total_h*uv) +\
                    0.5*sqrt(g_grav*avg(total_h))*jump(total_h, self.normal)  # NOTE works
                # Hu_star = avg(total_h*uv) # NOTE fails
                f += inner(jump(self.test, self.normal), Hu_star)*self.dS
                # TODO come up with better stabilization here!
                # NOTE scaling sqrt(g_h) doesn't help
        else:
            f = div(total_h*uv)*self.test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                if funcs is None:
                    f += -total_h*dot(uv, self.normal)*self.test*ds_bnd
            # f += -avg(total_h)*avg(dot(uv, normal))*jump(self.test)*dS
        return f

    def rhs_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        return -F - G

    def rhs(self, solution, uv, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        eta = solution

        if self.nonlin:
            total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry

        # Divergence of depth-integrated velocity
        F += self.HUDivTerm(uv, total_h)

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

            elif 'flux' in funcs:
                # prescribe normal flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(uv, self.normal)
                un_ext = funcs['flux'] / total_h / sect_len
                un_av = (un_in + un_ext)/2
                G += total_h * un_av * self.test * ds_bnd

            elif 'radiation':
                # prescribe radiation condition that allows waves to pass tru
                un_ext = sqrt(g_grav / total_h) * eta
                G += total_h * un_ext * self.test * ds_bnd

        return -F - G

    def source(self, uv_old=None, uv_bottom=None, bottom_drag=None,
               baroc_head=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        return -F
