"""
3D advection diffusion equation for tracers.

Tuomas Karna 2015-09-08
"""
from utility import *


class tracerEquation(equation):
    """3D tracer advection-diffusion equation"""
    def __init__(self, mesh, space, solution, eta, uv=None, w=None,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 uvMag=None, uvP1=None, laxFriedrichsFactor=None,
                 bnd_markers=None, bnd_len=None, nonlin=True,
                 vElemSize=None):
        self.mesh = mesh
        self.space = space
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.vElemSize = vElemSize
        self.kwargs = {'eta': eta,
                       'uv': uv,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'diffusivity_h': diffusivity_h,
                       'diffusivity_v': diffusivity_v,
                       'uvMag': uvMag,
                       'uvP1': uvP1,
                       'laxFriedrichsFactor': laxFriedrichsFactor,
                       }
        self.computeHorizAdvection = uv is not None
        self.computeVertAdvection = w is not None
        self.computeHorizDiffusion = diffusivity_h is not None
        self.computeVertDiffusion = diffusivity_v is not None

        # trial and test functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        ufl_elem = self.space.ufl_element()
        if not hasattr(ufl_elem, '_A'):
            # For HDiv elements
            ufl_elem = ufl_elem._element
        self.horizontal_DG = ufl_elem._A.family() != 'Lagrange'
        self.vertical_DG = ufl_elem._B.family() != 'Lagrange'

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

        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def massTerm(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        test = self.test
        return inner(solution, test) * self.dx

    def RHS_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms
        return -F - G

    def RHS(self, solution, eta=None, uv=None, w=None, w_mesh=None, dw_mesh_dz=None,
            diffusivity_h=None, diffusivity_v=None,
            laxFriedrichsFactor=None,
            uvMag=None, uvP1=None,
            **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        F = 0  # holds all dx volume integral terms
        G = 0  # holds all ds boundary interface terms

        # NOTE advection terms must be exactly as in 3d continuity equation
        # Horizontal advection term
        if self.computeHorizAdvection:
            if self.horizAdvectionByParts:
                F += -solution*(uv[0]*Dx(self.test, 0) +
                                uv[1]*Dx(self.test, 1))*self.dx
                if self.horizontal_DG:
                    # add interface term
                    uv_av = avg(uv)
                    un_av = (uv_av[0]*self.normal('-')[0] +
                             uv_av[1]*self.normal('-')[1])
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = solution('-')*s + solution('+')*(1-s)
                    G += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                               uv_av[1]*jump(self.test, self.normal[1]))*(self.dS_v + self.dS_h)
                    # Lax-Friedrichs stabilization
                    if laxFriedrichsFactor is not None:
                        if uvP1 is not None:
                            gamma = 0.5*abs((avg(uvP1)[0]*self.normal('-')[0] +
                                             avg(uvP1)[1]*self.normal('-')[1]))*laxFriedrichsFactor
                        elif uvMag is not None:
                            gamma = 0.5*avg(uvMag)*laxFriedrichsFactor
                        else:
                            raise Exception('either uvP1 or uvMag must be given')
                        G += gamma*dot(jump(self.test), jump(solution))*(self.dS_v + self.dS_h)
                    for bnd_marker in self.boundary_markers:
                        funcs = self.bnd_functions.get(bnd_marker)
                        ds_bnd = self.ds_v(int(bnd_marker))
                        if funcs is None:
                            continue
                        elif 'value' in funcs:
                            # prescribe external tracer value
                            c_in = solution
                            c_ext = funcs['value']
                            uv_av = uv
                            un_av = self.normal[0]*uv[0] + self.normal[1]*uv[1]
                            s = 0.5*(sign(un_av) + 1.0)
                            c_up = c_in*s + c_ext*(1-s)
                            # TODO should take external un from bnd conditions
                            G += c_up*(uv_av[0]*self.normal[0] +
                                       uv_av[1]*self.normal[1])*self.test*ds_bnd
            else:
                F += (Dx(uv[0]*solution, 0) + Dx(uv[1]*solution, 1))*self.test*self.dx
                G += -solution*(uv[0]*self.normal[0] +
                                uv[1]*self.normal[1])*self.test*(self.ds_bottom)
            # boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = self.ds_v(int(bnd_marker))
                if funcs is None:
                    if not self.horizAdvectionByParts:
                        G += -solution*(self.normal[0]*uv[0] +
                                        self.normal[1]*uv[1])*self.test*ds_bnd
                    continue

        # Vertical advection term
        if self.computeVertAdvection:
            vertvelo = w[2]
            if w_mesh is not None:
                vertvelo = w[2]-w_mesh
            F += -solution*vertvelo*Dx(self.test, 2)*self.dx
            if self.vertical_DG:
                w_av = avg(vertvelo)
                s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
                c_up = solution('-')*s + solution('+')*(1-s)
                G += c_up*w_av*jump(self.test, self.normal[2])*self.dS_h
                if laxFriedrichsFactor is not None:
                    # Lax-Friedrichs
                    gamma = 0.5*abs(w_av*self.normal('-')[2])*laxFriedrichsFactor
                    G += gamma*dot(jump(self.test), jump(solution))*self.dS_h

            # Non-conservative ALE source term
            if dw_mesh_dz is not None:
                F += solution*dw_mesh_dz*self.test*self.dx

            # NOTE Bottom impermeability condition is naturally satisfied by the definition of w
            if w_mesh is None:
                G += solution*vertvelo*self.normal[2]*self.test*self.ds_surf
            else:
                G += solution*vertvelo*self.normal[2]*self.test*self.ds_surf

        # diffusion
        if self.computeHorizDiffusion:
            F += diffusivity_h*(Dx(solution, 0)*Dx(self.test, 0) +
                                Dx(solution, 1)*Dx(self.test, 1))*self.dx
            if self.horizontal_DG:
                # interface term
                muGradSol = diffusivity_h*grad(solution)
                F += -(avg(muGradSol[0])*jump(self.test, self.normal[0]) +
                    avg(muGradSol[1])*jump(self.test, self.normal[1]))*(self.dS_v+self.dS_h)
                ## TODO symmetric penalty term
                ## sigma = (o+1)(o+d)/d*N_0/(2L) (Shahbazi, 2005)
                ## o: order of space, 
                #sigma = 1e-4
                #nMag = self.normal[0]('-')**2 + self.normal[1]('-')**2
                #F += -sigma*avg(diffusivity_h)*nMag*jump(solution)*jump(self.test)*(self.dS_v+self.dS_h)

        if self.computeVertDiffusion:
            F += diffusivity_v*(Dx(solution, 2)*Dx(self.test, 2))*self.dx
            if self.vertical_DG:
                # interface term
                muGradSol = diffusivity_v*Dx(solution, 2)
                F += -avg(muGradSol)*jump(self.test, self.normal[2])*(self.dS_h)
                # symmetric interior penalty stabilization
                if self.vElemSize is None:
                    raise Exception('vElemSize must be provided')
                L = avg(self.vElemSize)
                nbNeigh = 2
                o = 1  # polynomial order
                d = 3  # dimension
                sigma = Constant((o + 1)*(o + d)/d * nbNeigh / 2) / L
                gamma = sigma*avg(diffusivity_v)*abs(self.normal('-')[2])**2
                F += -gamma * jump(self.test)*jump(solution) * self.dS_h

        return -F - G

    def Source(self, eta=None, uv=None, w=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        F = 0  # holds all dx volume integral terms

        return -F
