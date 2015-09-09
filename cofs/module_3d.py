"""
3D momentum and tracer equations for hydrostatic Boussinesq flow.

Tuomas Karna 2015-02-23
"""
from utility import *

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class momentumEquation(equation):
    """3D momentum equation for hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, bnd_markers, bnd_len,
                 solution, eta, bathymetry, w=None,
                 w_mesh=None, dw_mesh_dz=None,
                 uv_bottom=None, bottom_drag=None, lin_drag=None,
                 viscosity_v=None, viscosity_h=None,
                 coriolis=None,
                 baro_head=None,
                 laxFriedrichsFactor=None, uvMag=None,
                 uvP1=None,
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
                       'lin_drag': lin_drag,
                       'viscosity_v': viscosity_v,
                       'viscosity_h': viscosity_h,
                       'baro_head': baro_head,
                       'coriolis': coriolis,
                       'laxFriedrichsFactor': laxFriedrichsFactor,
                       'uvMag': uvMag,
                       'uvP1': uvP1,
                       }
        # time independent arg
        self.bathymetry = bathymetry

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        ufl_elem = self.space.ufl_element()
        if isinstance(ufl_elem, EnrichedElement):
            # get the first elem of enriched space
            ufl_elem = ufl_elem._elements[0]
        if not hasattr(ufl_elem, '_A'):
            # For HDiv elements
            ufl_elem = ufl_elem._element
        self.horizontal_DG = ufl_elem._A.family() != 'Lagrange'
        self.vertical_DG = ufl_elem._B.family() != 'Lagrange'
        self.HDiv = ufl_elem._A.family() == 'Raviart-Thomas'

        eta_elem = eta.function_space().ufl_element()
        self.eta_is_DG = eta_elem._A.family() != 'Lagrange'

        self.gradEtaByParts = self.eta_is_DG
        self.horizAdvectionByParts = True

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = self.mesh._dx
        self.dS_v = self.mesh._dS_v
        self.dS_h = self.mesh._dS_h
        self.ds_v = self.mesh._ds_v
        self.ds_surf = self.mesh._ds_b
        self.ds_bottom = self.mesh._ds_t

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx

    def pressureGrad(self, eta, baro_head, uv, total_H, byParts=True,
                     **kwargs):
        if baro_head is not None:
            head = eta + baro_head
        else:
            head = eta
        if byParts:
            divTest = (Dx(self.test[0], 0) +
                       Dx(self.test[1], 1))
            f = -g_grav*head*divTest*self.dx
            #head_star = avg(head) + 0.5*sqrt(avg(total_H)/g_grav)*jump(uv, self.normal)
            head_star = avg(head)
            jumpNDotTest = (jump(self.test[0], self.normal[0]) +
                            jump(self.test[1], self.normal[1]))
            f += g_grav*head_star*jumpNDotTest*(self.dS_v + self.dS_h)
            nDotTest = (self.normal[0]*self.test[0] +
                        self.normal[1]*self.test[1])
            f += g_grav*head*nDotTest*(self.ds_bottom + self.ds_surf)
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds_v(int(bnd_marker))
                if baro_head is not None:
                    f += g_grav*baro_head*nDotTest*ds_bnd
                specialEtaFlux = funcs is not None and 'elev' in funcs
                if not specialEtaFlux:
                    f += g_grav*eta*nDotTest*ds_bnd
        else:
            gradHeadDotTest = (Dx(head, 0)*self.test[0] +
                               Dx(head, 1)*self.test[1])
            f = g_grav * gradHeadDotTest * self.dx
        return f

    def horizontalAdvection(self, solution, total_H, laxFriedrichsFactor,
                            uvMag=None, uvP1=None, **kwargs):
        if not self.nonlin:
            return 0
        if self.horizAdvectionByParts:
            f = -(Dx(self.test[0], 0)*solution[0]*solution[0] +
                  Dx(self.test[0], 1)*solution[0]*solution[1] +
                  Dx(self.test[1], 0)*solution[1]*solution[0] +
                  Dx(self.test[1], 1)*solution[1]*solution[1])*self.dx
            uv_av = avg(solution)
            un_av = (uv_av[0]*self.normal('-')[0] +
                     uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            uv_up = solution('-')*s + solution('+')*(1-s)
            if self.horizontal_DG:
                f += (uv_up[0]*uv_av[0]*jump(self.test[0], self.normal[0]) +
                      uv_up[0]*uv_av[1]*jump(self.test[0], self.normal[1]) +
                      uv_up[1]*uv_av[0]*jump(self.test[1], self.normal[0]) +
                      uv_up[1]*uv_av[1]*jump(self.test[1], self.normal[1]))*(self.dS_v + self.dS_h)
                # Lax-Friedrichs stabilization
                if laxFriedrichsFactor is not None and uvMag is not None:
                    if uvP1 is not None:
                        gamma = 0.5*abs((avg(uvP1)[0]*self.normal('-')[0] +
                                         avg(uvP1)[1]*self.normal('-')[1]))*laxFriedrichsFactor
                    elif uvMag is not None:
                        gamma = 0.5*avg(uvMag)*laxFriedrichsFactor
                    else:
                        raise Exception('either uvP1 or uvMag must be given')
                    f += gamma*(jump(self.test[0])*jump(solution[0]) +
                                jump(self.test[1])*jump(solution[1]))*self.dS_v
                for bnd_marker in self.boundary_markers:
                    funcs = self.bnd_functions.get(bnd_marker)
                    ds_bnd = self.ds_v(int(bnd_marker))
                    if funcs is None:
                        un = dot(solution, self.normal)
                        uv_ext = solution - 2*un*self.normal
                        if laxFriedrichsFactor is not None:
                            gamma = 0.5*abs(un)*laxFriedrichsFactor
                            f += gamma*(self.test[0]*(solution[0] - uv_ext[0]) +
                                        self.test[1]*(solution[1] - uv_ext[1]))*ds_bnd
                    elif 'flux' in funcs:
                        # prescribe normal volume flux
                        sect_len = Constant(self.boundary_len[bnd_marker])
                        un_ext = funcs['flux'] / total_H / sect_len
                        if self.nonlin:
                            uv_in = solution
                            uv_ext = self.normal*un_ext
                            uv_av = 0.5*(uv_in + uv_ext)
                            un_av = uv_av[0]*self.normal[0] + uv_av[1]*self.normal[1]
                            s = 0.5*(sign(un_av) + 1.0)
                            uv_up = uv_in*s + uv_ext*(1-s)
                            f += (uv_up[0]*self.test[0]*self.normal[0]*uv_av[0] +
                                  uv_up[0]*self.test[0]*self.normal[1]*uv_av[1] +
                                  uv_up[1]*self.test[1]*self.normal[0]*uv_av[0] +
                                  uv_up[1]*self.test[1]*self.normal[1]*uv_av[1])*ds_bnd
                            # Lax-Friedrichs stabilization
                            if laxFriedrichsFactor is not None:
                                gamma = 0.5*abs(un_av)*laxFriedrichsFactor
                                f += gamma*(self.test[0]*(uv_in[0] - uv_ext[0]) +
                                            self.test[1]*(uv_in[1] - uv_ext[1]))*ds_bnd

            # surf/bottom boundary conditions: closed at bed, symmetric at surf
            f += (solution[0]*solution[0]*self.test[0]*self.normal[0] +
                  solution[0]*solution[1]*self.test[0]*self.normal[1] +
                  solution[1]*solution[0]*self.test[1]*self.normal[0] +
                  solution[1]*solution[1]*self.test[1]*self.normal[1])*(self.ds_surf)
        else:
            f = inner(div(outer(solution, solution)), self.test)*self.dx
        return f

    def RHS_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        return -F - G

    def RHS(self, solution, eta, w=None, viscosity_v=None,
            viscosity_h=None, coriolis=None, baro_head=None,
            uv_bottom=None, bottom_drag=None, lin_drag=None,
            w_mesh=None, dw_mesh_dz=None, laxFriedrichsFactor=None,
            uvMag=None, uvP1=None, **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        # external pressure gradient
        F += self.pressureGrad(eta, baro_head, solution, total_H, byParts=self.gradEtaByParts)

        # Advection term
        if self.nonlin:
            F += self.horizontalAdvection(solution, total_H, laxFriedrichsFactor,
                                          uvMag=uvMag, uvP1=uvP1)

            # Vertical advection term
            if w is not None:
                vertvelo = w[2]
                if w_mesh is not None:
                    vertvelo = w[2]-w_mesh
                Adv_v = -(Dx(self.test[0], 2)*solution[0]*vertvelo +
                          Dx(self.test[1], 2)*solution[1]*vertvelo)
                F += Adv_v * self.dx
                if self.vertical_DG:
                    s = 0.5*(sign(avg(w[2])*self.normal[2]('-')) + 1.0)
                    uv_up = solution('-')*s + solution('+')*(1-s)
                    w_av = avg(w[2])
                    G += (uv_up[0]*w_av*jump(self.test[0], self.normal[2]) +
                          uv_up[1]*w_av*jump(self.test[1], self.normal[2]))*self.dS_h
                    if laxFriedrichsFactor is not None:
                        # Lax-Friedrichs
                        gamma = 0.5*abs(w_av*self.normal('-')[2])*laxFriedrichsFactor
                        G += gamma*(jump(self.test[0])*jump(solution[0]) +
                                    jump(self.test[1])*jump(solution[1]))*self.dS_h
                G += (solution[0]*vertvelo*self.test[0]*self.normal[2] +
                      solution[1]*vertvelo*self.test[1]*self.normal[2])*(self.ds_surf)
            # NOTE bottom impermeability condition is naturally satisfied by the defition of w

        # Non-conservative ALE source term
        if dw_mesh_dz is not None:
            F += dw_mesh_dz*(solution[0]*self.test[0] +
                             solution[1]*self.test[1])*dx

        # boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = self.bnd_functions.get(bnd_marker)
            ds_bnd = self.ds_v(int(bnd_marker))
            un_in = (solution[0]*self.normal[0] + solution[1]*self.normal[1])
            if funcs is None:
                # assume land boundary
                continue

            elif 'elev' in funcs:
                # prescribe elevation only
                h_ext = funcs['elev']
                uv_ext = solution
                t = self.normal[1] * self.e_x - self.normal[0] * self.e_y
                ut_in = dot(solution, t)
                # ut_ext = -dot(uv_ext,t) # assume zero
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

                G += g_grav*(eta + h_ext)/2*dot(self.normal, self.test)*ds_bnd
                if self.nonlin:
                    # NOTE just symmetric 3D flux with 2D eta correction
                    G += un_riemann * un_riemann * dot(self.normal, self.test) * ds_bnd

            elif 'un' in funcs:
                # prescribe normal volume flux
                un_ext = funcs['un']
                if self.nonlin:
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

            elif 'symm' in funcs:
                if self.nonlin:
                    uv_in = un_in*self.normal
                    G += (uv_in[0]*self.test[0]*self.normal[0]*uv_in[0] +
                          uv_in[0]*self.test[0]*self.normal[1]*uv_in[1] +
                          uv_in[1]*self.test[1]*self.normal[0]*uv_in[0] +
                          uv_in[1]*self.test[1]*self.normal[1]*uv_in[1])*ds_bnd

        # Coriolis
        if coriolis is not None:
            F += coriolis*(-solution[1]*self.test[0] +
                           solution[0]*self.test[1])*self.dx

        # horizontal viscosity
        if viscosity_h is not None:
            F_visc = viscosity_h * (Dx(solution[0], 0) * Dx(self.test[0], 0) +
                                    Dx(solution[1], 0) * Dx(self.test[1], 0) +
                                    Dx(solution[0], 1) * Dx(self.test[0], 1) +
                                    Dx(solution[1], 1) * Dx(self.test[1], 1))
            if self.horizontal_DG:
                # interface term
                muGradSol = viscosity_h*nabla_grad(solution)
                F += -(avg(muGradSol[0, 0])*jump(self.test[0], self.normal[0]) +
                       avg(muGradSol[0, 1])*jump(self.test[1], self.normal[0]) +
                       avg(muGradSol[1, 0])*jump(self.test[0], self.normal[1]) +
                       avg(muGradSol[1, 1])*jump(self.test[1], self.normal[1]))*(self.dS_v+self.dS_h)
                # TODO symmetric interior penalty term
            F += F_visc * self.dx

        # vertical viscosity
        if viscosity_v is not None:
            F += viscosity_v*(Dx(self.test[0], 2)*Dx(solution[0], 2) +
                              Dx(self.test[1], 2)*Dx(solution[1], 2)) * self.dx
            if self.vertical_DG:
                intViscFlux = (jump(self.test[0]*Dx(solution[0], 2), self.normal[2]) +
                               jump(self.test[1]*Dx(solution[1], 2), self.normal[2]))
                G += -avg(viscosity_v) * intViscFlux * self.dS_h
                # viscflux = viscosity_v*Dx(solution, 2)
                # G += -(avg(viscflux[0])*jump(self.test[0], normal[2]) +
                #        avg(viscflux[0])*jump(self.test[1], normal[2]))

        # Linear drag (consistent with drag in 2D mode)
        if lin_drag is not None:
            BottomFri = lin_drag*inner(self.test, solution)*self.dx
            F += BottomFri

        return -F - G

    def Source(self, eta, w=None, viscosity_v=None,
               uv_bottom=None, bottom_drag=None, baro_head=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms
        G = 0

        if self.nonlin:
            total_H = self.bathymetry + eta
        else:
            total_H = self.bathymetry

        if viscosity_v is not None:
            # bottom friction
            if bottom_drag is not None and uv_bottom is not None:
                stress = bottom_drag*sqrt(uv_bottom[0]**2 +
                                          uv_bottom[1]**2)*uv_bottom
                BotFriction = (stress[0]*self.test[0] +
                               stress[1]*self.test[1])*ds_t
                F += BotFriction

        return -F -G


class verticalMomentumEquation(equation):
    """Vertical advection and diffusion terms of 3D momentum equation for
    hydrostatic Boussinesq flow."""
    def __init__(self, mesh, space, space_scalar, solution, w=None,
                 viscosity_v=None, uv_bottom=None, bottom_drag=None,
                 wind_stress=None):
        self.mesh = mesh
        self.space = space
        self.space_scalar = space_scalar
        self.solution = solution
        # this dict holds all time dep. args to the equation
        self.kwargs = {'w': w,
                       'viscosity_v': viscosity_v,
                       'uv_bottom': uv_bottom,
                       'bottom_drag': bottom_drag,
                       'wind_stress': wind_stress,
                       }

        # test and trial functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        ufl_elem = self.space.ufl_element()
        if isinstance(ufl_elem, EnrichedElement):
            # get the first elem of enriched space
            ufl_elem = ufl_elem._elements[0]
        if not hasattr(ufl_elem, '_A'):
            # For HDiv elements
            ufl_elem = ufl_elem._element
        self.horizontal_DG = ufl_elem._A.family() != 'Lagrange'
        self.vertical_DG = ufl_elem._B.family() != 'Lagrange'

        # mesh dependent variables
        self.normal = FacetNormal(mesh)
        self.xyz = SpatialCoordinate(mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # integral measures
        self.dx = self.mesh._dx
        self.dS_v = self.mesh._dS_v
        self.dS_h = self.mesh._dS_h
        self.ds_v = self.mesh._ds_v
        self.ds_surf = self.mesh._ds_b
        self.ds_bottom = self.mesh._ds_t

        # set boundary conditions
        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def getTimeStep(self, Umag=Constant(1.0)):
        raise NotImplementedError('getTimeStep not implemented')

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        return inner(solution, self.test) * self.dx
        #return (solution[0]*self.test[0] + solution[1]*self.test[1]) * self.dx

    def RHS_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        return -F - G

    def RHS(self, solution, w=None, viscosity_v=None,
            uv_bottom=None, bottom_drag=None,
            **kwargs):
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
            if self.vertical_DG:
                # FIXME implement interface terms
                pass
                #raise NotImplementedError('Adv term not implemented for DG')

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
               uv_bottom=None, bottom_drag=None,
               wind_stress=None,
               **kwargs):
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
            # wind stress
            if wind_stress is not None:
                F -= (wind_stress[0]*self.test[0] +
                      wind_stress[1]*self.test[1])/rho_0*ds_b

        return -F
