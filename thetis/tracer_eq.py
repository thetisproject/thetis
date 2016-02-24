"""
3D advection diffusion equation for tracers.

Tuomas Karna 2015-09-08
"""
from utility import *


class TracerEquation(Equation):
    """3D tracer advection-diffusion equation"""
    def __init__(self, solution, eta, uv=None, w=None,
                 w_mesh=None, dw_mesh_dz=None,
                 diffusivity_h=None, diffusivity_v=None,
                 source=None,
                 uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
                 bnd_markers=None, bnd_len=None, nonlin=True,
                 v_elem_size=None):
        self.space = solution.function_space()
        self.mesh = self.space.mesh()
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.v_elem_size = v_elem_size
        self.kwargs = {'eta': eta,
                       'uv': uv,
                       'w': w,
                       'w_mesh': w_mesh,
                       'dw_mesh_dz': dw_mesh_dz,
                       'diffusivity_h': diffusivity_h,
                       'diffusivity_v': diffusivity_v,
                       'source': source,
                       'uv_mag': uv_mag,
                       'uv_p1': uv_p1,
                       'lax_friedrichs_factor': lax_friedrichs_factor,
                       }
        self.compute_horiz_advection = uv is not None
        self.compute_vert_advection = w is not None
        self.compute_horiz_diffusion = diffusivity_h is not None
        self.compute_vert_diffusion = diffusivity_v is not None

        # trial and test functions
        self.test = TestFunction(self.space)
        self.tri = TrialFunction(self.space)

        continuity = element_continuity(self.space.fiat_element)
        self.horizontal_dg = continuity.horizontal_dg
        self.vertical_dg = continuity.vertical_dg

        self.horiz_advection_by_parts = True

        # mesh dependent variables
        self.normal = FacetNormal(self.mesh)
        self.xyz = SpatialCoordinate(self.mesh)
        self.e_x, self.e_y, self.e_y = unit_vectors(3)

        # boundary definitions
        self.boundary_markers = bnd_markers
        self.boundary_len = bnd_len

        # maps bnd_marker to dict of external functions e.g. {'elev':eta_ext}
        self.bnd_functions = {}

    def mass_term(self, solution):
        """All time derivative terms on the LHS, without the actual time
        derivative.

        Implements A(u) for  d(A(u_{n+1}) - A(u_{n}))/dt
        """
        test = self.test
        return inner(solution, test) * dx

    def rhs_implicit(self, solution, wind_stress=None, **kwargs):
        """Returns all the terms that are treated semi-implicitly.
        """
        f = 0
        return -f

    def rhs(self, solution, eta=None, uv=None, w=None, w_mesh=None, dw_mesh_dz=None,
            diffusivity_h=None, diffusivity_v=None,
            lax_friedrichs_factor=None,
            uv_mag=None, uv_p1=None,
            **kwargs):
        """Returns the right hand side of the equations.
        RHS is all terms that depend on the solution (eta,uv)"""
        f = 0  # holds all dx volume integral terms
        g = 0  # holds all ds boundary interface terms

        # NOTE advection terms must be exactly as in 3d continuity equation
        # Horizontal advection term
        if self.compute_horiz_advection:
            if self.horiz_advection_by_parts:
                f += -solution*inner(uv, nabla_grad(self.test))*dx
                if self.horizontal_dg:
                    # add interface term
                    uv_av = avg(uv)
                    un_av = (uv_av[0]*self.normal('-')[0] +
                             uv_av[1]*self.normal('-')[1])
                    s = 0.5*(sign(un_av) + 1.0)
                    c_up = solution('-')*s + solution('+')*(1-s)
                    g += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                               uv_av[1]*jump(self.test, self.normal[1]) +
                               uv_av[2]*jump(self.test, self.normal[2]))*(dS_v + dS_h)
                    # Lax-Friedrichs stabilization
                    if lax_friedrichs_factor is not None:
                        if uv_p1 is not None:
                            gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0] +
                                             avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                        elif uv_mag is not None:
                            gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                        else:
                            raise Exception('either uv_p1 or uv_mag must be given')
                        g += gamma*dot(jump(self.test), jump(solution))*(dS_v + dS_h)
                    for bnd_marker in self.boundary_markers:
                        funcs = self.bnd_functions.get(bnd_marker)
                        ds_bnd = ds_v(int(bnd_marker))
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
                            g += c_up*(uv_av[0]*self.normal[0] +
                                       uv_av[1]*self.normal[1])*self.test*ds_bnd
            else:
                f += (Dx(uv[0]*solution, 0) + Dx(uv[1]*solution, 1))*self.test*dx
                g += -solution*(uv[0]*self.normal[0] +
                                uv[1]*self.normal[1])*self.test*(ds_bottom)
            # boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker))
                if funcs is None:
                    if not self.horiz_advection_by_parts:
                        g += -solution*(self.normal[0]*uv[0] +
                                        self.normal[1]*uv[1])*self.test*ds_bnd
                    continue

        # Vertical advection term
        if self.compute_vert_advection:
            vertvelo = w[2]
            if w_mesh is not None:
                vertvelo = w[2]-w_mesh
            f += -solution*vertvelo*Dx(self.test, 2)*dx
            if self.vertical_dg:
                w_av = avg(vertvelo)
                s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
                c_up = solution('-')*s + solution('+')*(1-s)
                g += c_up*w_av*jump(self.test, self.normal[2])*dS_h
                if lax_friedrichs_factor is not None:
                    # Lax-Friedrichs
                    gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                    g += gamma*dot(jump(self.test), jump(solution))*dS_h

            # Non-conservative ALE source term
            if dw_mesh_dz is not None:
                f += solution*dw_mesh_dz*self.test*dx

            # NOTE Bottom impermeability condition is naturally satisfied by the definition of w
            if w_mesh is None:
                g += solution*vertvelo*self.normal[2]*self.test*ds_surf
            else:
                g += solution*vertvelo*self.normal[2]*self.test*ds_surf

        # diffusion
        if self.compute_horiz_diffusion:
            f += diffusivity_h*(Dx(solution, 0)*Dx(self.test, 0) +
                                Dx(solution, 1)*Dx(self.test, 1))*dx
            if self.horizontal_dg:
                # interface term
                mu_grad_sol = diffusivity_h*grad(solution)
                f += -(avg(mu_grad_sol[0])*jump(self.test, self.normal[0]) +
                       avg(mu_grad_sol[1])*jump(self.test, self.normal[1]))*(dS_v+dS_h)
                # # TODO symmetric penalty term
                # # sigma = (o+1)(o+d)/d*N_0/(2L) (Shahbazi, 2005)
                # # o: order of space
                # sigma = 1e-4
                # n_mag = self.normal[0]('-')**2 + self.normal[1]('-')**2
                # F += -sigma*avg(diffusivity_h)*n_mag*jump(solution)*jump(self.test)*(dS_v+dS_h)
            for bnd_marker in self.boundary_markers:
                funcs = self.bnd_functions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker))
                if funcs is None or 'value' in funcs or 'symm' in funcs:
                    # use symmetric diffusion flux through boundary
                    f += -inner(mu_grad_sol, self.normal)*self.test*ds_bnd
        if self.compute_vert_diffusion:
            f += diffusivity_v*inner(Dx(solution, 2), Dx(self.test, 2)) * dx
            if self.vertical_dg:
                # interface term
                diff_flux = diffusivity_v*Dx(solution, 2)
                f += -(dot(avg(diff_flux), self.test('+'))*self.normal[2]('+') +
                       dot(avg(diff_flux), self.test('-'))*self.normal[2]('-')) * dS_h
                # symmetric interior penalty stabilization
                ip_fact = Constant(1.0)
                if self.v_elem_size is None:
                    raise Exception('v_elem_size must be provided')
                l = avg(self.v_elem_size)
                nb_neigh = 2.
                o = 1.
                d = 3.
                sigma = Constant((o + 1.0)*(o + d)/d * nb_neigh / 2.0) / l
                gamma = sigma*avg(diffusivity_v) * ip_fact
                jump_test = (self.test('+')*self.normal[2]('+') +
                             self.test('-')*self.normal[2]('-'))
                f += gamma * dot(jump(solution), jump_test) * dS_h

        return -f - g

    def source(self, eta=None, uv=None, w=None, source=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        f = 0  # holds all dx volume integral terms

        if source is not None:
            f += -inner(source, self.test)*dx

        return -f
