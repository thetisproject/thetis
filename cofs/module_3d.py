"""
3D momentum and tracer equations for hydrostatic Boussinesq flow.

Tuomas Karna 2015-02-23
"""
from utility import *
from physical_constants import *

g_grav = physical_constants['g_grav']
wd_alpha = physical_constants['wd_alpha']
mu_manning = physical_constants['mu_manning']
z0_friction = physical_constants['z0_friction']
von_karman = physical_constants['von_karman']


class momentumEquation(equation):
    """3D momentum equation for hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, bnd_markers, bnd_len,
                 solution, eta, bathymetry, w=None,
                 w_mesh=None, dw_mesh_dz=None,
                 uv_bottom=None, bottom_drag=None, viscosity_v=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.nonlin = nonlin
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'eta': eta,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_v': viscosity_v,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        ## compute lenth of all boundaries
        #self.boundary_len = {}
        #for i in self.boundary_markers:
            ##ds_restricted = Measure('ds', subdomain_id=int(i))
            ##one_func = Function(self.space_scalar).interpolate(Expression(1.0))
            ##self.boundary_len[i] = assemble(one_func * ds_restricted)
            ##print 'bnd', i, self.boundary_len[i]

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        csize = CellSize(self.mesh)
        H = FunctionSpace(self.mesh, 'CG', 1, vfamily='CG', vdegree=1)
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
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def RHS(self, solution, eta, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None,
            w_mesh=None, dw_mesh_dz=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if self.nonlin:
            # in 3d: nabla_grad dot (u u)
            # weak form: nabla_grad(psi) : u u
            # NOTE not validated (term small in channel flow)
            Adv_h = -(Dx(self.test[0], 0)*solution[0]*solution[0] +
                      Dx(self.test[0], 1)*solution[0]*solution[1] +
                      Dx(self.test[1], 0)*solution[1]*solution[0] +
                      Dx(self.test[1], 1)*solution[1]*solution[1])
            F += Adv_h * self.dx
            if self.horizontal_DG:
                uv_av = avg(solution)
                un_av = (uv_av[0]*self.normal('-')[0] +
                         uv_av[1]*self.normal('-')[1])
                s = 0.5*(sign(un_av) + 1.0)
                uv_up = solution('-')*s + solution('+')*(1-s)
                G += (uv_up[0]*jump(self.test[0], self.normal[0]*solution[0]) +
                      uv_up[0]*jump(self.test[0], self.normal[1]*solution[1]) +
                      uv_up[1]*jump(self.test[1], self.normal[0]*solution[0]) +
                      uv_up[1]*jump(self.test[1], self.normal[1]*solution[1]))*(self.dS_v)
                # Lax-Friedrichs stabilization
                gamma = abs(un_av)
                G += gamma*dot(jump(self.test), jump(solution))*self.dS_v
            if self.vertical_DG:
                # NOTE bottom bnd doesn't work for DG vertical mesh
                uv_av = avg(solution)
                G += (uv_av[0]*uv_av[0]*jump(self.test[0], self.normal[0]) +
                      uv_av[0]*uv_av[1]*jump(self.test[0], self.normal[1]) +
                      uv_av[1]*uv_av[0]*jump(self.test[1], self.normal[0]) +
                      uv_av[1]*uv_av[1]*jump(self.test[1], self.normal[1]))*(self.dS_h)
            G += (solution[0]*solution[0]*self.test[0]*self.normal[0] +
                  solution[0]*solution[1]*self.test[0]*self.normal[1] +
                  solution[1]*solution[0]*self.test[1]*self.normal[0] +
                  solution[1]*solution[1]*self.test[1]*self.normal[1])*(ds_t + ds_b)
            # Vertical advection term
            vertvelo = w
            if w_mesh is not None:
                vertvelo = w-w_mesh
            #F += -solution*vertvelo*Dx(self.test, 2)*dx
            # Vertical advection
            Adv_v = -(Dx(self.test[0], 2)*solution[0]*vertvelo +
                      Dx(self.test[1], 2)*solution[1]*vertvelo)
            F += Adv_v * self.dx
            G += (solution[0]*vertvelo*self.test[0]*self.normal[2] +
                  solution[1]*vertvelo*self.test[1]*self.normal[2])*(ds_t + ds_b)
            #if self.horizontal_DG:
                #w_rie = avg(w)
                #uv_rie = avg(solution)
                #G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      #uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

            # Non-conservative ALE source term
            if dw_mesh_dz is not None:
                F += dw_mesh_dz*(solution[0]*self.test[0] +
                                 solution[1]*self.test[1])*dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue
                #if self.nonlin:
                    #G += (self.test[0]*solution[0]*solution[0]*self.normal[0] +
                          #self.test[0]*solution[0]*solution[1]*self.normal[1] +
                          #self.test[1]*solution[1]*solution[0]*self.normal[0] +
                          #self.test[1]*solution[1]*solution[1]*self.normal[1])*(ds_bnd)

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                uv_ext = solution
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut_in = dot(solution, t)
                # ut_ext = -dot(uv_ext,t) # assume zero
                un_in = dot(solution, self.normal)
                un_ext = dot(uv_ext, self.normal)

                if self.nonlin:
                    H = self.bathymetry + (eta + h_ext) / 2
                else:
                    H = self.bathymetry
                c_roe = sqrt(g_grav * H)
                un_riemann = dot(solution, self.normal) + c_roe / H * (eta - h_ext)/2
                H_riemann = H
                ut_riemann = tanh(4 * un_riemann / 0.02) * (ut_in)
                uv_riemann = un_riemann * self.normal + ut_riemann * t

                if self.nonlin:
                    # NOTE just symmetric 3D flux with 2D eta correction
                    G += un_riemann * un_riemann * dot(self.normal, self.test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal volume flux
                un_in = dot(solution, self.normal)
                un_ext = funcs['un']
                if self.nonlin:
                    #un_av = 0.5*(un_ext+un_in)
                    #G += un_av*un_av*inner(self.normal, self.test)*ds_bnd
                    ## Lax-Friedrichs stabilization
                    #gamma = abs(self.normal[0]*(solution[0]) +
                                #self.normal[1]*(solution[1]))
                    #G += gamma*dot((self.test), (solution-self.normal*un_ext)/2)*ds_bnd
                    uv_in = solution
                    uv_ext = self.normal*un_ext
                    uv_av = 0.5*(uv_in + uv_ext)
                    un_av = uv_av[0]*self.normal[0] + uv_av[1]*self.normal[1]
                    s = 0.5*(sign(un_av) + 1.0)
                    uv_up = uv_in*s + uv_ext*(1-s)
                    G += (uv_up[0]*self.test[0]*self.normal[0]*uv_in[0] +
                          uv_up[0]*self.test[0]*self.normal[1]*uv_in[1] +
                          uv_up[1]*self.test[1]*self.normal[0]*uv_in[0] +
                          uv_up[1]*self.test[1]*self.normal[1]*uv_in[1])*ds_bnd
                    # Lax-Friedrichs stabilization
                    gamma = abs(un_av)
                    G += gamma*dot(self.test, (uv_in - uv_ext)/2)*ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_in = dot(solution, self.normal)
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    #un_av = 0.5*(un_ext+un_in)
                    #G += un_av*un_av*inner(self.normal, self.test)*ds_bnd
                    ## Lax-Friedrichs stabilization
                    #gamma = abs(self.normal[0]*(solution[0]) +
                                #self.normal[1]*(solution[1]))
                    #G += gamma*dot((self.test), (solution-self.normal*un_ext)/2)*ds_bnd

                    uv_in = solution
                    uv_ext = self.normal*un_ext
                    uv_av = 0.5*(uv_in + uv_ext)
                    un_av = uv_av[0]*self.normal[0] + uv_av[1]*self.normal[1]
                    s = 0.5*(sign(un_av) + 1.0)
                    uv_up = uv_in*s + uv_ext*(1-s)
                    G += (uv_up[0]*self.test[0]*self.normal[0]*uv_in[0] +
                          uv_up[0]*self.test[0]*self.normal[1]*uv_in[1] +
                          uv_up[1]*self.test[1]*self.normal[0]*uv_in[0] +
                          uv_up[1]*self.test[1]*self.normal[1]*uv_in[1])*ds_bnd
                    # Lax-Friedrichs stabilization
                    gamma = abs(un_av)
                    G += gamma*dot(self.test, (uv_in - uv_ext)/2)*ds_bnd

        ## viscosity
        ## A double dot product of the stress tensor and grad(w).
        #K_momentum = -viscosity * (Dx(uv[0], 0) * Dx(self.U_test[0], 0) +
                                   #Dx(uv[0], 1) * Dx(self.U_test[0], 1) +
                                   #Dx(uv[1], 0) * Dx(self.U_test[1], 0) +
                                   #Dx(uv[1], 1) * Dx(self.U_test[1], 1))
        #K_momentum += viscosity/total_H*inner(dot(grad(total_H), grad(uv)),
                                              #self.U_test)
        #F -= K_momentum * self.dx

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, eta, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        # external pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.test) * self.dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                G += g_grav * ((h_ext - eta) / 2) * \
                    inner(self.normal, self.test) * ds_bnd

            elif 'flux' in funcs:
                # prescribe normal volume flux
                sect_len = Constant(self.boundary_len[bnd_marker])
                un_ext = funcs['flux'] / total_H / sect_len
                if self.nonlin:
                    G += un_ext*un_ext*inner(self.normal, self.test)*ds_bnd

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class verticalMomentumEquation(equation):
    """Vertical advection and diffusion terms of 3D momentum equation for
    hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, solution, w=None,
                 viscosity_v=None, uv_bottom=None, bottom_drag=None):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'viscosity_v': viscosity_v,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       }

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def getTimeStep(self, Umag=Constant(1.0)):
        raise NotImplementedError('getTimeStep not implemented')

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        Contains all terms that depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # Advection term
        if w is not None:
            # Vertical advection
            Adv_v = -(Dx(self.test[0], 2)*solution[0]*w +
                      Dx(self.test[1], 2)*solution[1]*w)
            F += Adv_v * self.dx
            if self.horizontal_DG:
                raise NotImplementedError('Adv term not implemented for DG')
                #w_rie = avg(w)
                #uv_rie = avg(solution)
                #G += (uv_rie[0]*w_rie*jump(self.test[0], self.normal[2]) +
                      #uv_rie[1]*w_rie*jump(self.test[1], self.normal[2]))*self.dS_h

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')
                # G += -viscosity_v * dot(psi, du/dz) * normal[2]
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        return -F - G

    def Source(self, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F


class tracerEquation(equation):
    """3D tracer advection-diffusion equation"""
    def __init__(self, mesh, space, solution, eta, uv, w,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None,
                 test_supg_h=None, test_supg_v=None, test_supg_mass=None,
                 nonlinStab_h=None, nonlinStab_v=None,
                 bnd_markers=None, bnd_len=None, nonlin=True):
        self.mesh = mesh
        self.space = space
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.kwargs = {'eta': eta,
                       'uv': uv,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'diffusivity_h': diffusivity_h,
                       'nonlinStab_h': nonlinStab_h,
                       'nonlinStab_v': nonlinStab_v}
        # SUPG terms (add to forms)
        self.test_supg_h = test_supg_h
        self.test_supg_v = test_supg_v
        self.test_supg_mass = test_supg_mass

        # trial and test functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        self.horizontal_DG = self.space.ufl_element()._A.family() != 'Lagrange'
        self.vertical_DG = self.space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        test = self.test
        if self.test_supg_mass is not None:
            test = self.test + self.test_supg_mass
        return inner(solution, test) * self.dx

    def RHS(self, solution, eta, uv, w, w_mesh=None, dw_mesh_dz=None,
            diffusivity_h=None, nonlinStab_h=None, nonlinStab_v=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        #test_h = self.test
        #if self.test_supg_h is not None:
            #test_h = self.test + self.test_supg_h

        # NOTE advection terms must be exactly as in 3d continuity equation
        # Horizontal advection term
        F += -solution*(uv[0]*Dx(self.test, 0) +
                        uv[1]*Dx(self.test, 1))*self.dx
        #F += self.test*(uv[0]*Dx(solution, 0) +
                        #uv[1]*Dx(solution, 1))*self.dx
        # Vertical advection term
        vertvelo = w
        if w_mesh is not None:
            vertvelo = w-w_mesh
        F += -solution*vertvelo*Dx(self.test, 2)*self.dx

        # Non-conservative ALE source term
        if dw_mesh_dz is not None:
            F += solution*dw_mesh_dz*self.test*self.dx

        # Bottom/top impermeability boundary conditions
        G += +solution*(uv[0]*self.normal[0] +
                        uv[1]*self.normal[1])*self.test*(ds_t + ds_b)
        # TODO what is the correct free surf bnd condition?
        G += +solution*vertvelo*self.normal[2]*self.test*(ds_t + ds_b)

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary NOTE uv.n should be very close to 0
                #continue
                G += solution*(self.normal[0]*uv[0] +
                               self.normal[1]*uv[1])*self.test*ds_bnd

            elif 'value' in funcs:
                # prescribe external tracer value
                nudge = 1.00
                c_in = solution
                c_ext = nudge*funcs['value'] + (1-nudge)*c_in
                un = self.normal[0]*uv[0] + self.normal[1]*uv[1]
                alpha = 0.5*(tanh(4 * un / 0.02) + 1)
                c_up = alpha*c_in + (1-alpha)*c_ext  # for inv.part adv term
                #c_up = (1-alpha)*(c_ext-c_in)/4  # for direct adv term
                G += c_up*un*self.test*ds_bnd

            #if diffusivity_h is not None:
                #dflux = diffusivity_h*(Dx(solution, 0)*self.normal[0] +
                                       #Dx(solution, 1)*self.normal[1])/2
                #G += -dflux*self.test*ds_bnd


        # diffusion
        if diffusivity_h is not None:
            F += diffusivity_h*(Dx(solution, 0)*Dx(self.test, 0) +
                                Dx(solution, 1)*Dx(self.test, 1))*self.dx

        # SUPG stabilization
        if self.test_supg_h is not None:
            F += self.test_supg_h*(uv[0]*Dx(solution, 0) +
                                   uv[1]*Dx(solution, 1))*self.dx
            if diffusivity_h is not None:
                F += -diffusivity_h*self.test_supg_h*(Dx(Dx(solution, 0), 0) +
                                                      Dx(Dx(solution, 1), 1))*self.dx
        if self.test_supg_v is not None:
            F += self.test_supg_v*vertvelo*Dx(solution, 2)*self.dx

        # non-linear damping
        if nonlinStab_h is not None:
            F += nonlinStab_h*(Dx(solution, 0)*Dx(self.test, 0) +
                               Dx(solution, 1)*Dx(self.test, 1))*self.dx

        if nonlinStab_v is not None:
            F += nonlinStab_v*(Dx(solution, 2)*Dx(self.test, 2))*self.dx

        return -F - G

    def Source(self, eta, uv, w, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        return -F


class freeSurfaceEquations3d(equation):
    """3D shallow water equations"""
    def __init__(self, mesh, space, bnd_markers, bnd_len,
                 solution, bathymetry, w=None,
                 uv_bottom=None, bottom_drag=None, viscosity_v=None,
                 nonlin=True):
        self.mesh = mesh
        self.space = space
        self.eta_space, self.U_space = self.space.split()
        self.nonlin = nonlin

        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'viscosity_v': viscosity_v,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # create mixed function space
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)
        self.U_test, self.eta_test = TestFunctions(self.space)
        self.U_tri, self.eta_tri = TrialFunctions(self.space)

        self.uv_horizontal_DG = self.U_space.ufl_element()._A.family() != 'Lagrange'
        self.uv_vertical_DG = self.U_space.ufl_element()._B.family() != 'Lagrange'
        self.eta_horizontal_DG = self.eta_space.ufl_element()._A.family() != 'Lagrange'
        self.eta_vertical_DG = self.eta_space.ufl_element()._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.cellsize = CellSize(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y = unit_vectors(2)

        # integral measures
        self.dx = Measure('dx', domain=self.mesh, subdomain_id='everywhere')
        self.dS = dS(domain=self.mesh)
        self.dS_v = dS_v(domain=self.mesh)
        self.dS_h = dS_h(domain=self.mesh)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def ds_v(self, bnd_marker):
        """Returns boundary measure for the appropriate mesh"""
        return ds_v(int(bnd_marker), domain=self.mesh)

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
        F += M_continuity

        return F * self.dx

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        uv, eta = split(solution)

        # Advection term
        if self.nonlin:
            # in 3d: nabla_grad dot (u u)
            # weak form: nabla_grad(psi) : u u
            # NOTE not validated (term small in channel flow)
            Adv_h = -(Dx(self.U_test[0], 0)*uv[0]*uv[0] +
                      Dx(self.U_test[0], 1)*uv[0]*uv[1] +
                      Dx(self.U_test[1], 0)*uv[1]*uv[0] +
                      Dx(self.U_test[1], 1)*uv[1]*uv[1])
            F += Adv_h * self.dx
            if self.uv_horizontal_DG:
                uv_rie = avg(uv)
                s = 0.5*(sign(uv_rie[0]*self.normal('-')[0] +
                              uv_rie[1]*self.normal('-')[1]) + 1.0)
                uv_up = uv('-')*s + uv('+')*(1-s)
                G += (uv_up[0]*uv_rie[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_up[0]*uv_rie[1]*jump(self.U_test[0], self.normal[1]) +
                      uv_up[1]*uv_rie[0]*jump(self.U_test[1], self.normal[0]) +
                      uv_up[1]*uv_rie[1]*jump(self.U_test[1], self.normal[1]))*(self.dS_v)
                # Lax-Friedrichs stabilization
                #gamma = Constant(1.0)
                #G += gamma*dot(jump(self.U_test), jump(uv))*self.dS_v
            if self.uv_vertical_DG:
                # NOTE bottom bnd doesn't work for DG vertical mesh
                G += (uv_rie[0]*uv_rie[0]*jump(self.U_test[0], self.normal[0]) +
                      uv_rie[0]*uv_rie[1]*jump(self.U_test[0], self.normal[1]) +
                      uv_rie[1]*uv_rie[0]*jump(self.U_test[1], self.normal[0]) +
                      uv_rie[1]*uv_rie[1]*jump(self.U_test[1], self.normal[1]))*(self.dS_h)
            G += (uv[0]*uv[0]*self.U_test[0]*self.normal[0] +
                  uv[0]*uv[1]*self.U_test[0]*self.normal[1] +
                  uv[1]*uv[0]*self.U_test[1]*self.normal[0] +
                  uv[1]*uv[1]*self.U_test[1]*self.normal[1])*(ds_t + ds_b)
            ## Vertical advection
            #Adv_v = -(Dx(self.U_test[0], 2)*uv[0]*w +
                      #Dx(self.U_test[1], 2)*uv[1]*w)
            #F += Adv_v * self.dx
            ##if self.horizontal_DG:
                ##w_rie = avg(w)
                ##uv_rie = avg(uv)
                ##G += (uv_rie[0]*w_rie*jump(self.U_test[0], self.normal[2]) +
                      ##uv_rie[1]*w_rie*jump(self.U_test[1], self.normal[2]))*self.dS_h

        # External pressure gradient
        F += g_grav * inner(nabla_grad(eta), self.U_test) * self.dx

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # Divergence of depth-integrated velocity
        F += -total_H * inner(uv, nabla_grad(self.eta_test)) * self.dx
        if self.eta_horizontal_DG:
            Hu_star = avg(total_H*uv) +\
                sqrt(g_grav*avg(total_H))*jump(total_H, self.normal)
            G += inner(jump(self.eta_test, self.normal), Hu_star)*self.dS_v

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), domain=self.mesh)
            if funcs is None:
                # assume land boundary
                continue

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.U_test[0], 2)*Dx(uv[0], 2) +
                              Dx(self.U_test[1], 2)*Dx(uv[1], 2))*dx
            if self.vertical_DG:
                raise NotImplementedError('Vertical diffusion has not been implemented for DG')

        return -F - G

    def Source(self, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.U_test[0] +
                               stress[1]*self.U_test[1])*ds_t
                F += BotFriction

        return -F
