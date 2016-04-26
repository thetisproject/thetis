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
                 source=None, bathymetry=None,
                 uv_mag=None, uv_p1=None, lax_friedrichs_factor=None,
                 bnd_markers=None, bnd_len=None, nonlin=True,
                 h_elem_size=None, v_elem_size=None):
        self.space = solution.function_space()
        self.mesh = self.space.mesh()
        # this dict holds all args to the equation (at current time step)
        self.solution = solution
        self.v_elem_size = v_elem_size
        self.h_elem_size = h_elem_size
        self.bathymetry = bathymetry
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

    def get_bnd_functions(self, c_in, uv_in, eta_in, bnd_id):
        """
        Returns external values tracer and uv for all supported
        boundary conditions.

        volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
        funcs = self.bnd_functions.get(bnd_id)
        bath = self.bathymetry
        bnd_len = self.boundary_len[bnd_id]

        if 'elev' in funcs:
            eta_ext = funcs['elev']
        else:
            eta_ext = eta_in
        if 'value' in funcs:
            c_ext = funcs['value']
        else:
            c_ext = c_in
        if 'uv' in funcs:
            uv_ext = funcs['uv']
        elif 'flux' in funcs:
            h_ext = eta_ext + bath
            area = h_ext*bnd_len  # NOTE using external data only
            uv_ext = funcs['flux']/area*self.normal
        elif 'un' in funcs:
            uv_ext = funcs['un']*self.normal
        else:
            uv_ext = uv_in

        return c_ext, uv_ext, eta_ext

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
                               uv_av[2]*jump(self.test, self.normal[2]))*(dS_v)
                    g += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                               uv_av[1]*jump(self.test, self.normal[1]) +
                               uv_av[2]*jump(self.test, self.normal[2]))*(dS_h)
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
                        else:
                            c_in = solution
                            c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, eta, bnd_marker)
                            uv_av = 0.5*(uv + uv_ext)
                            un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                            s = 0.5*(sign(un_av) + 1.0)
                            c_up = c_in*s + c_ext*(1-s)
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
            # NOTE imex solver fails with this in tracerBox example
            if w_mesh is None:
                g += solution*vertvelo*self.normal[2]*self.test*ds_surf
            else:
                g += solution*vertvelo*self.normal[2]*self.test*ds_surf

        # diffusion
        if self.compute_horiz_diffusion:
            diff_tensor = as_matrix([[diffusivity_h, 0, 0],
                                     [0, diffusivity_h, 0],
                                     [0, 0, 0]])
            grad_test = grad(self.test)
            diff_flux = dot(diff_tensor, grad(solution))

            f += inner(grad_test, diff_flux)*dx

            if self.horizontal_dg:
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                # Interior Penalty method by
                # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
                # sigma = 3*k_max**2/k_min*p*(p+1)*cot(Theta)
                # k_max/k_min  - max/min diffusivity
                # p            - polynomial degree
                # Theta        - min angle of triangles
                # assuming k_max/k_min=2, Theta=pi/3
                # sigma = 6.93 = 3.5*p*(p+1)
                degree_h, degree_v = self.space.ufl_element().degree()
                # TODO compute elemsize as CellVolume/FacetArea
                # h = n.D.n where D = diag(h_h, h_h, h_v)
                elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                              self.normal[1]**2) +
                            self.v_elem_size*self.normal[2]**2)
                sigma = 5.0*degree_h*(degree_h + 1)/elemsize
                if degree_h == 0:
                    raise NotImplementedError('horizontal diffusion not implemented for p0')
                alpha = avg(sigma)
                ds_interior = (dS_h + dS_v)
                f += alpha*inner(jump(self.test, self.normal),
                                 dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
                f += -inner(avg(dot(diff_tensor, grad(self.test))),
                            jump(solution, self.normal))*ds_interior
                f += -inner(jump(self.test, self.normal),
                            avg(dot(diff_tensor, grad(solution))))*ds_interior

            # symmetric bottom boundary condition
            # NOTE introduces a flux through the bed - breaks mass conservation
            f += - inner(diff_flux, self.normal)*self.test*ds_bottom
            f += - inner(diff_flux, self.normal)*self.test*ds_surf

        if self.compute_vert_diffusion:
            grad_test = Dx(self.test, 2)
            diff_flux = dot(diffusivity_v, Dx(solution, 2))

            f += inner(grad_test, diff_flux)*dx

            if self.vertical_dg:
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                # Interior Penalty method by
                # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
                degree_h, degree_v = self.space.ufl_element().degree()
                # TODO compute elemsize as CellVolume/FacetArea
                # h = n.D.n where D = diag(h_h, h_h, h_v)
                elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                              self.normal[1]**2) +
                            self.v_elem_size*self.normal[2]**2)
                sigma = 5.0*degree_v*(degree_v + 1)/elemsize
                if degree_v == 0:
                    sigma = 1.0/elemsize
                alpha = avg(sigma)
                ds_interior = (dS_h)
                f += alpha*inner(jump(self.test, self.normal[2]),
                                 dot(avg(diffusivity_v), jump(solution, self.normal[2])))*ds_interior
                f += -inner(avg(dot(diffusivity_v, Dx(self.test, 2))),
                            jump(solution, self.normal[2]))*ds_interior
                f += -inner(jump(self.test, self.normal[2]),
                            avg(dot(diffusivity_v, Dx(solution, 2))))*ds_interior

        return -f - g

    def source(self, eta=None, uv=None, w=None, source=None, **kwargs):
        """Returns the right hand side of the source terms.
        These terms do not depend on the solution."""
        f = 0  # holds all dx volume integral terms

        if source is not None:
            f += -inner(source, self.test)*dx

        return -f
