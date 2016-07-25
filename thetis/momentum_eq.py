"""
3D momentum and tracer equations for hydrostatic Boussinesq flow.

Tuomas Karna 2015-02-23
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class MomentumTerm(Term):
    """
    Generic term for momentum equation that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 nonlin=True, use_bottom_friction=False):
        super(MomentumTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.fiat_element)
        self.horizontal_dg = continuity.horizontal_dg
        self.vertical_dg = continuity.vertical_dg
        self.nonlin = nonlin
        self.use_bottom_friction = use_bottom_friction

        # define measures with a reasonable quadrature degree
        p, q = self.function_space.ufl_element().degree()
        self.quad_degree = (2*p + 1, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

        # TODO add generic get_bnd_functions?


class PressureGradientTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        baroc_head = fields_old.get('baroc_head')
        eta = fields_old.get('eta')

        if eta is None and baroc_head is None:
            return 0
        if eta is None:
            by_parts = element_continuity(baroc_head.function_space().fiat_element).dg
            head = baroc_head
        elif baroc_head is None:
            by_parts = element_continuity(eta.function_space().fiat_element).dg
            head = eta
        else:
            by_parts = element_continuity(eta.function_space().fiat_element).dg
            head = eta + baroc_head

        if by_parts:
            div_test = (Dx(self.test[0], 0) +
                        Dx(self.test[1], 1))
            f = -g_grav*head*div_test*self.dx
            # head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            head_star = avg(head)
            jump_n_dot_test = (jump(self.test[0], self.normal[0]) +
                               jump(self.test[1], self.normal[1]))
            f += g_grav*head_star*jump_n_dot_test*(self.dS_v + self.dS_h)
            n_dot_test = (self.normal[0]*self.test[0] +
                          self.normal[1]*self.test[1])
            f += g_grav*head*n_dot_test*(self.ds_bottom + self.ds_surf)
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if baroc_head is not None:
                    f += g_grav*baroc_head*n_dot_test*ds_bnd
                special_eta_flux = eta is not None and funcs is not None and 'elev' in funcs
                if not special_eta_flux:
                    f += g_grav*eta*n_dot_test*ds_bnd
                if funcs is not None:
                    if 'elev' in funcs:
                        # prescribe elevation only
                        h_ext = funcs['elev']
                        f += g_grav*(eta + h_ext)/2*dot(self.normal, self.test)*ds_bnd
        else:
            grad_head_dot_test = (Dx(head, 0)*self.test[0] +
                                  Dx(head, 1)*self.test[1])
            f = g_grav * grad_head_dot_test * self.dx
        return -f


class HorizontalAdvectionTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if not self.nonlin:
            return 0
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        f = -(Dx(self.test[0], 0)*solution[0]*solution_old[0] +
              Dx(self.test[0], 1)*solution[0]*solution_old[1] +
              Dx(self.test[1], 0)*solution[1]*solution_old[0] +
              Dx(self.test[1], 1)*solution[1]*solution_old[1])*self.dx
        uv_av = avg(solution_old)
        un_av = (uv_av[0]*self.normal('-')[0] +
                 uv_av[1]*self.normal('-')[1])
        s = 0.5*(sign(un_av) + 1.0)
        uv_up = solution('-')*s + solution('+')*(1-s)
        if self.horizontal_dg:
            f += (uv_up[0]*uv_av[0]*jump(self.test[0], self.normal[0]) +
                  uv_up[0]*uv_av[1]*jump(self.test[0], self.normal[1]) +
                  uv_up[1]*uv_av[0]*jump(self.test[1], self.normal[0]) +
                  uv_up[1]*uv_av[1]*jump(self.test[1], self.normal[1]))*(self.dS_v + self.dS_h)
            # Lax-Friedrichs stabilization
            if lax_friedrichs_factor is not None and uv_mag is not None:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0] +
                                     avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    raise Exception('either uv_p1 or uv_mag must be given')
                f += gamma*(jump(self.test[0])*jump(solution[0]) +
                            jump(self.test[1])*jump(solution[1]))*self.dS_v
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if funcs is None:
                    un = dot(solution, self.normal)
                    uv_ext = solution - 2*un*self.normal
                    if lax_friedrichs_factor is not None:
                        gamma = 0.5*abs(un)*lax_friedrichs_factor
                        f += gamma*(self.test[0]*(solution[0] - uv_ext[0]) +
                                    self.test[1]*(solution[1] - uv_ext[1]))*ds_bnd
                else:
                    uv_in = solution
                    use_lf = True
                    if 'symm' in funcs or 'elev' in funcs:
                        # use internal normal velocity
                        # NOTE should this be symmetic normal velocity?
                        uv_ext = uv_in
                        use_lf = False
                    elif 'un' in funcs:
                        # prescribe normal velocity
                        un_ext = funcs['un']
                        uv_ext = self.normal*un_ext
                    elif 'flux' in funcs:
                        # prescribe normal volume flux
                        sect_len = Constant(self.boundary_len[bnd_marker])
                        eta = fields_old['eta']
                        total_h = self.bathymetry + eta
                        un_ext = funcs['flux'] / total_h / sect_len
                        uv_ext = self.normal*un_ext
                    if self.nonlin:
                        uv_av = 0.5*(uv_in + uv_ext)
                        un_av = uv_av[0]*self.normal[0] + uv_av[1]*self.normal[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        uv_up = uv_in*s + uv_ext*(1-s)
                        f += (uv_up[0]*self.test[0]*self.normal[0]*uv_av[0] +
                              uv_up[0]*self.test[0]*self.normal[1]*uv_av[1] +
                              uv_up[1]*self.test[1]*self.normal[0]*uv_av[0] +
                              uv_up[1]*self.test[1]*self.normal[1]*uv_av[1])*ds_bnd
                        if use_lf:
                            # Lax-Friedrichs stabilization
                            if lax_friedrichs_factor is not None:
                                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                                f += gamma*(self.test[0]*(uv_in[0] - uv_ext[0]) +
                                            self.test[1]*(uv_in[1] - uv_ext[1]))*ds_bnd

        # surf/bottom boundary conditions: closed at bed, symmetric at surf
        f += (solution_old[0]*solution[0]*self.test[0]*self.normal[0] +
              solution_old[0]*solution[1]*self.test[0]*self.normal[1] +
              solution_old[1]*solution[0]*self.test[1]*self.normal[0] +
              solution_old[1]*solution[1]*self.test[1]*self.normal[1])*(self.ds_surf)
        return -f


class VerticalAdvectionTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        w = fields_old.get('w')
        w_mesh = fields_old.get('w_mesh')
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')
        if w is None or not self.nonlin:
            return 0
        f = 0
        vertvelo = w[2]
        if w_mesh is not None:
            vertvelo = w[2]-w_mesh
        adv_v = -(Dx(self.test[0], 2)*solution[0]*vertvelo +
                  Dx(self.test[1], 2)*solution[1]*vertvelo)
        f += adv_v * self.dx
        if self.vertical_dg:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
            uv_up = solution('-')*s + solution('+')*(1-s)
            f += (uv_up[0]*w_av*jump(self.test[0], self.normal[2]) +
                  uv_up[1]*w_av*jump(self.test[1], self.normal[2]))*self.dS_h
            if lax_friedrichs_factor is not None:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                f += gamma*(jump(self.test[0])*jump(solution[0]) +
                            jump(self.test[1])*jump(solution[1]))*self.dS_h
        f += (solution[0]*vertvelo*self.test[0]*self.normal[2] +
              solution[1]*vertvelo*self.test[1]*self.normal[2])*(self.ds_surf)
        # NOTE bottom impermeability condition is naturally satisfied by the defition of w
        return -f


class ALESourceTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        dw_mesh_dz = fields_old.get('dw_mesh_dz')
        f = 0
        # Non-conservative ALE source term
        if dw_mesh_dz is not None:
            f += dw_mesh_dz*(solution[0]*self.test[0] +
                             solution[1]*self.test[1])*self.dx
        return -f


class HorizontalViscosityTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_h = fields_old.get('viscosity_h')
        if viscosity_h is None:
            return 0
        f = 0

        def grad_h(a):
            return as_matrix([[Dx(a[0], 0), Dx(a[0], 1), 0],
                              [Dx(a[1], 0), Dx(a[1], 1), 0],
                              [0, 0, 0]])
        visc_tensor = as_matrix([[viscosity_h, 0, 0],
                                 [0, viscosity_h, 0],
                                 [0, 0, 0]])

        grad_uv = grad_h(solution)
        grad_test = grad_h(self.test)
        stress = dot(visc_tensor, grad_uv)
        f += inner(grad_test, stress)*self.dx

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
            sigma = 5.0*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h + self.dS_v)
            f += alpha*inner(tensor_jump(self.normal, self.test),
                             dot(avg(visc_tensor), tensor_jump(self.normal, solution)))*ds_interior
            f += -inner(avg(dot(visc_tensor, nabla_grad(self.test))),
                        tensor_jump(self.normal, solution))*ds_interior
            f += -inner(tensor_jump(self.normal, self.test),
                        avg(dot(visc_tensor, nabla_grad(solution))))*ds_interior

        # symmetric bottom boundary condition
        f += -inner(stress, outer(self.test, self.normal))*self.ds_surf
        f += -inner(stress, outer(self.test, self.normal))*self.ds_bottom

        # TODO boundary conditions
        # TODO impermeability condition at bottom
        # TODO implement as separate function
        return -f


class VerticalViscosityTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_v = fields_old.get('viscosity_v')
        if viscosity_v is None:
            return 0
        f = 0
        grad_test = Dx(self.test, 2)
        diff_flux = viscosity_v*Dx(solution, 2)
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
            f += alpha*inner(tensor_jump(self.normal[2], self.test),
                             avg(viscosity_v)*tensor_jump(self.normal[2], solution))*ds_interior
            f += -inner(avg(viscosity_v*Dx(self.test, 2)),
                        tensor_jump(self.normal[2], solution))*ds_interior
            f += -inner(tensor_jump(self.normal[2], self.test),
                        avg(viscosity_v*Dx(solution, 2)))*ds_interior
        return -f


class BottomFrictionTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if self.use_bottom_friction:
            z0_friction = physical_constants['z0_friction']
            z_bot = 0.5*self.v_elem_size
            von_karman = physical_constants['von_karman']
            drag = (von_karman / ln((z_bot + z0_friction)/z0_friction))**2
            # compute uv_bottom implicitly
            uv_bot = solution + Dx(solution, 2)*z_bot
            uv_bot_old = solution_old + Dx(solution_old, 2)*z_bot
            uv_bot_mag = sqrt(uv_bot_old[0]**2 + uv_bot_old[1]**2)
            stress = drag*uv_bot_mag*uv_bot
            bot_friction = (stress[0]*self.test[0] +
                            stress[1]*self.test[1])*self.ds_bottom
            f += bot_friction
        return -f


class LinearDragTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        linear_drag = fields_old.get('linear_drag')
        f = 0
        # Linear drag (consistent with drag in 2D mode)
        if linear_drag is not None:
            bottom_fri = linear_drag*inner(self.test, solution)*self.dx
            f += bottom_fri
        return -f


class CoriolisTerm(MomentumTerm):
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-solution[1]*self.test[0] +
                           solution[0]*self.test[1])*self.dx
        return -f


class SourceTerm(MomentumTerm):
    """
    Generic source term
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        viscosity_v = fields_old.get('viscosity_v')
        wind_stress = fields_old.get('wind_stress')
        if viscosity_v is not None:
            # wind stress
            if wind_stress is not None:
                f -= (wind_stress[0]*self.test[0] +
                      wind_stress[1]*self.test[1])/rho_0*self.ds_surf
        if source is not None:
            f += - inner(source, self.test)*self.dx
        return -f


class MomentumEquation(Equation):
    """
    3D momentum equation for hydrostatic Boussinesq flow.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 nonlin=True, use_bottom_friction=False):
        super(MomentumEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, nonlin, use_bottom_friction)
        self.add_term(PressureGradientTerm(*args), 'source')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
        # self.add_term(ALESourceTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(VerticalViscosityTerm(*args), 'explicit')
        self.add_term(BottomFrictionTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
