"""
3D advection diffusion equation for tracers.

Tuomas Karna 2015-09-08
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation


class TracerTerm(Term):
    """
    Generic tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True):
        super(TracerTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.fiat_element)
        self.horizontal_dg = continuity.horizontal_dg
        self.vertical_dg = continuity.vertical_dg
        self.use_symmetric_surf_bnd = use_symmetric_surf_bnd

        # define measures with a reasonable quadrature degree
        p, q = self.function_space.ufl_element().degree()
        self.quad_degree = (2*p + 1, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

    def get_bnd_functions(self, c_in, uv_in, elev_in, bnd_id, bnd_conditions):
        """
        Returns external values tracer and uv for all supported
        boundary conditions.

        volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
        funcs = bnd_conditions.get(bnd_id)

        if 'elev' in funcs:
            elev_ext = funcs['elev']
        else:
            elev_ext = elev_in
        if 'value' in funcs:
            c_ext = funcs['value']
        else:
            c_ext = c_in
        if 'uv' in funcs:
            uv_ext = funcs['uv']
        elif 'flux' in funcs:
            assert self.bathymetry is not None
            h_ext = elev_ext + self.bathymetry
            area = h_ext*self.boundary_len  # NOTE using external data only
            uv_ext = funcs['flux']/area*self.normal
        elif 'un' in funcs:
            uv_ext = funcs['un']*self.normal
        else:
            uv_ext = uv_in

        return c_ext, uv_ext, elev_ext


class HorizontalAdvectionTerm(TracerTerm):
    """
    Horizontal scalar advection term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_3d') is None:
            return 0
        elev = fields_old['elev_3d']
        uv = fields_old['uv_3d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')

        f = 0
        f += -solution*inner(uv, nabla_grad(self.test))*self.dx
        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0] +
                     uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            f += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                       uv_av[1]*jump(self.test, self.normal[1]) +
                       uv_av[2]*jump(self.test, self.normal[2]))*(self.dS_v)
            f += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                       uv_av[1]*jump(self.test, self.normal[1]) +
                       uv_av[2]*jump(self.test, self.normal[2]))*(self.dS_h)
            # Lax-Friedrichs stabilization
            if lax_friedrichs_factor is not None:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0] +
                                     avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    raise Exception('either uv_p1 or uv_mag must be given')
                f += gamma*dot(jump(self.test), jump(solution))*(self.dS_v + self.dS_h)
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                    if funcs is None:
                        continue
                    else:
                        c_in = solution
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        f += c_up*(uv_av[0]*self.normal[0] +
                                   uv_av[1]*self.normal[1])*self.test*ds_bnd

        if self.use_symmetric_surf_bnd:
            f += solution*(uv[0]*self.normal[0] + uv[1]*self.normal[1])*self.test*ds_surf
        return -f


class VerticalAdvectionTerm(TracerTerm):
    """
    Vertical scalar advection term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        # TODO linearize, use solution/solution_old correctly
        w = fields_old.get('w')
        if w is None:
            return 0
        w_mesh = fields_old.get('w_mesh')
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')

        vertvelo = w[2]
        if w_mesh is not None:
            vertvelo = w[2] - w_mesh
        f = 0
        f += -solution*vertvelo*Dx(self.test, 2)*self.dx
        if self.vertical_dg:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            f += c_up*w_av*jump(self.test, self.normal[2])*self.dS_h
            if lax_friedrichs_factor is not None:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS_h

        # NOTE Bottom impermeability condition is naturally satisfied by the definition of w
        # NOTE imex solver fails with this in tracerBox example
        f += solution*vertvelo*self.normal[2]*self.test*self.ds_surf
        return -f


class ALESourceTerm(TracerTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        dw_mesh_dz = fields_old.get('dw_mesh_dz')
        f = 0
        # Non-conservative ALE source term
        if dw_mesh_dz is not None:
            f += dw_mesh_dz*solution*self.test*self.dx
        return -f


class HorizontalDiffusionTerm(TracerTerm):
    """
    Horizontal scalar diffusion term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, 0],
                                 [0, diffusivity_h, 0],
                                 [0, 0, 0]])
        grad_test = grad(self.test)
        diff_flux = dot(diff_tensor, grad(solution))

        f = 0
        f += inner(grad_test, diff_flux)*self.dx

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
            degree_h, degree_v = self.function_space.ufl_element().degree()
            # TODO compute elemsize as CellVolume/FacetArea
            # h = n.D.n where D = diag(h_h, h_h, h_v)
            elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                          self.normal[1]**2) +
                        self.v_elem_size*self.normal[2]**2)
            sigma = 2.5*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h + self.dS_v)
            f += alpha*inner(jump(self.test, self.normal),
                             dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
            f += -inner(avg(dot(diff_tensor, grad(self.test))),
                        jump(solution, self.normal))*ds_interior
            f += -inner(jump(self.test, self.normal),
                        avg(dot(diff_tensor, grad(solution))))*ds_interior

        # symmetric bottom boundary condition
        # NOTE introduces a flux through the bed - breaks mass conservation
        f += - inner(diff_flux, self.normal)*self.test*self.ds_bottom
        f += - inner(diff_flux, self.normal)*self.test*self.ds_surf

        return -f


class VerticalDiffusionTerm(TracerTerm):
    """
    Vertical scalar diffusion term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_v') is None:
            return 0

        diffusivity_v = fields_old['diffusivity_v']

        grad_test = Dx(self.test, 2)
        diff_flux = dot(diffusivity_v, Dx(solution, 2))

        f = 0
        f += inner(grad_test, diff_flux)*self.dx

        if self.vertical_dg:
            assert self.h_elem_size is not None, 'h_elem_size must be defined'
            assert self.v_elem_size is not None, 'v_elem_size must be defined'
            # Interior Penalty method by
            # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
            degree_h, degree_v = self.function_space.ufl_element().degree()
            # TODO compute elemsize as CellVolume/FacetArea
            # h = n.D.n where D = diag(h_h, h_h, h_v)
            elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                          self.normal[1]**2) +
                        self.v_elem_size*self.normal[2]**2)
            sigma = 5.0*degree_v*(degree_v + 1)/elemsize
            if degree_v == 0:
                sigma = 1.0/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h)
            f += alpha*inner(jump(self.test, self.normal[2]),
                             dot(avg(diffusivity_v), jump(solution, self.normal[2])))*ds_interior
            f += -inner(avg(dot(diffusivity_v, Dx(self.test, 2))),
                        jump(solution, self.normal[2]))*ds_interior
            f += -inner(jump(self.test, self.normal[2]),
                        avg(dot(diffusivity_v, Dx(solution, 2))))*ds_interior

        return -f


class SourceTerm(TracerTerm):
    """
    Generic tracer source term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += -inner(source, self.test)*self.dx
        return -f


class TracerEquation(Equation):
    """
    3D tracer advection-diffusion equation
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True):
        super(TracerEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_symmetric_surf_bnd)
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
        # self.add_term(ALESourceTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(VerticalDiffusionTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
