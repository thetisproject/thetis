r"""
3D momentum equation for hydrostatic Boussinesq flow.

The three dimensional momentum equation reads

.. math::
    \frac{\partial \textbf{u}}{\partial t}
        + \nabla_h \cdot (\textbf{u} \textbf{u})
        + \frac{\partial \left(w\textbf{u} \right)}{\partial z}
        + f\textbf{e}_z\wedge\textbf{u} + g\nabla_h \eta + g\nabla_h r
        = \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)
        + \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right)
    :label: mom_eq

where :math:`\textbf{u}` and :math:`w` denote horizontal and vertical velocity,
:math:`\nabla_h` is the horizontal gradient,
:math:`\wedge` denotes the cross product,
:math:`g` is the gravitational acceleration, :math:`f` is the Coriolis
frequency, :math:`\textbf{e}_z` is a vertical unit vector, and
:math:`\nu_h, \nu` stand for horizontal and vertical viscosity.
Water density is given by :math:`\rho = \rho'(T, S, p) + \rho_0`,
where :math:`\rho_0` is a constant reference density.
Above :math:`r` denotes the baroclinic head

.. math::
    r = \frac{1}{\rho_0} \int_{z}^\eta  \rho' d\zeta.
    :label: baroc_head

In the case of purely barotropic problems the :math:`r` and the internal pressure
gradient are omitted.

When using mode splitting we split the velocity field into a depth average and
a deviation, :math:`\textbf{u} = \bar{\textbf{u}} + \textbf{u}'`.
Following Higdon and de Szoeke (1997) we write an equation for the deviation
:math:`\textbf{u}'`:

.. math::
    \frac{\partial \textbf{u}'}{\partial t} =
        + \nabla_h \cdot (\textbf{u} \textbf{u})
        + \frac{\partial \left(w\textbf{u} \right)}{\partial z}
        + f\textbf{e}_z\wedge\textbf{u}' + g\nabla_h r
        = \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)
        + \frac{\partial }{\partial z}\left( \nu  \frac{\partial \textbf{u}}{\partial z}\right)
    :label: mom_eq_split

In :eq:`mom_eq_split` the external pressure gradient :math:`g\nabla_h \eta` vanishes and the
Coriolis term only contains the deviation :math:`\textbf{u}'`.
Advection and diffusion terms are not changed.

Higdon and de Szoeke (1997). Barotropic-Baroclinic Time Splitting for Ocean
Circulation Modeling. Journal of Computational Physics, 135(1):30-53.
http://dx.doi.org/10.1006/jcph.1997.5733
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'MomentumEquation',
    'MomentumTerm',
    'HorizontalAdvectionTerm',
    'VerticalAdvectionTerm',
    'HorizontalViscosityTerm',
    'VerticalViscosityTerm',
    'PressureGradientTerm',
    'CoriolisTerm',
    'BottomFrictionTerm',
    'LinearDragTerm',
    'SourceTerm',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class MomentumTerm(Term):
    """
    Generic term for momentum equation that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 nonlin=True, use_bottom_friction=False,
                 use_elevation_gradient=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        :kwarg bool use_elevation_gradient: If True includes external pressure
            gradient in the pressure gradient term
        """
        super(MomentumTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.fiat_element)
        self.horizontal_dg = continuity.horizontal_dg
        self.vertical_dg = continuity.vertical_dg
        self.nonlin = nonlin
        self.use_bottom_friction = use_bottom_friction
        self.use_elevation_gradient = use_elevation_gradient

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
    r"""
    Internal pressure gradient term, :math:`g\nabla_h r`

    where :math:`r` is the baroclinic head :eq:`baroc_head`. Let :math:`s`
    denote :math:`r/H`. We can then write

    .. math::
        F_{IPG} = g\nabla_h((s -\bar{s}) H)
            + g\nabla_h\left(\frac{1}{H}\right) H^2\bar{s}
            + g s_{bot}\nabla_h h

    where :math:`\bar{s},s_{bot}` are the depth average and bottom value of
    :math:`s`.

    If :math:`s` belongs to a discontinuous function space, the first term is
    integrated by parts. Its weak form reads

    .. math::
        \int_\Omega g\nabla_h((s -\bar{s}) H) \cdot \boldsymbol{\psi} dx
            = - \int_\Omega g (s -\bar{s}) H \nabla_h \cdot \boldsymbol{\psi} dx
            + \int_{\mathcal{I}_h \cup \mathcal{I}_v} g (s -\bar{s}) H \boldsymbol{\psi}  \cdot \textbf{n}_h dx
    """
    # TODO revise
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        if self.nonlin:
            total_h = self.bathymetry + fields_old.get('eta')
        else:
            total_h = self.bathymetry

        # TODO include additional 2D pressure gradient terms ...
        # TODO update naming convention: baroc_head is actually s not r
        baroc_head = fields_old.get('baroc_head')
        mean_baroc_head = fields_old.get('mean_baroc_head')
        if baroc_head is not None:
            assert mean_baroc_head is not None, 'mean_baroc_head must be provided'
            baroc_head = (baroc_head - mean_baroc_head)*total_h
        eta = fields_old.get('eta') if self.use_elevation_gradient else None

        if eta is None and baroc_head is None:
            return 0
        if eta is None:
            by_parts = element_continuity(fields_old.get('baroc_head').function_space().fiat_element).dg
            head = baroc_head
        elif baroc_head is None:
            by_parts = element_continuity(eta.function_space().fiat_element).dg
            head = eta
        else:
            by_parts = element_continuity(eta.function_space().fiat_element).dg
            head = eta + baroc_head

        use_lin_stab = False

        if by_parts:
            div_test = (Dx(self.test[0], 0) +
                        Dx(self.test[1], 1))
            f = -g_grav*head*div_test*self.dx
            if use_lin_stab:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(solution_old, self.normal)
            else:
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
                if eta is not None:
                    special_eta_flux = funcs is not None and 'elev' in funcs
                    if not special_eta_flux:
                        if use_lin_stab:
                            un_jump = (solution_old[0]*self.normal[0] +
                                       solution_old[1]*self.normal[1])
                            eta_star = eta + sqrt(total_h/g_grav)*un_jump
                        else:
                            eta_star = eta
                        f += g_grav*eta_star*n_dot_test*ds_bnd
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
    r"""
    Horizontal advection term, :math:`\nabla_h \cdot (\textbf{u} \textbf{u})`

    The weak form reads

    .. math::
        \int_\Omega \nabla_h \cdot (\textbf{u} \textbf{u}) \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \nabla_h \boldsymbol{\psi} : (\textbf{u} \textbf{u}) dx
        + \int_{\mathcal{I}_h \cup \mathcal{I}_v} \textbf{u}^{\text{up}} \cdot
        \text{jump}(\boldsymbol{\psi} \textbf{n}_h) \cdot \text{avg}(\textbf{u}) dS

    where the right hand side has been integrated by parts; :math:`:` stand for
    the Frobenius inner product, :math:`\textbf{n}_h` is the horizontal
    projection of the normal vector, :math:`\textbf{u}^{\text{up}}` is the
    upwind value, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if not self.nonlin:
            return 0
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
            uv_old = solution_old + uv_depth_av
        else:
            uv = solution
            uv_old = solution_old

        f = -(Dx(self.test[0], 0)*uv[0]*uv_old[0] +
              Dx(self.test[0], 1)*uv[0]*uv_old[1] +
              Dx(self.test[1], 0)*uv[1]*uv_old[0] +
              Dx(self.test[1], 1)*uv[1]*uv_old[1])*self.dx
        uv_av = avg(uv_old)
        un_av = (uv_av[0]*self.normal('-')[0] +
                 uv_av[1]*self.normal('-')[1])
        s = 0.5*(sign(un_av) + 1.0)
        uv_up = uv('-')*s + uv('+')*(1-s)
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
                f += gamma*(jump(self.test[0])*jump(uv[0]) +
                            jump(self.test[1])*jump(uv[1]))*self.dS_v
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if funcs is None:
                    un = dot(uv, self.normal)
                    uv_ext = uv - 2*un*self.normal
                    if lax_friedrichs_factor is not None:
                        gamma = 0.5*abs(un)*lax_friedrichs_factor
                        f += gamma*(self.test[0]*(uv[0] - uv_ext[0]) +
                                    self.test[1]*(uv[1] - uv_ext[1]))*ds_bnd
                else:
                    uv_in = uv
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
        f += (uv_old[0]*uv[0]*self.test[0]*self.normal[0] +
              uv_old[0]*uv[1]*self.test[0]*self.normal[1] +
              uv_old[1]*uv[0]*self.test[1]*self.normal[0] +
              uv_old[1]*uv[1]*self.test[1]*self.normal[1])*(self.ds_surf)
        return -f


class VerticalAdvectionTerm(MomentumTerm):
    r"""
    Vertical advection term, :math:`\partial \left(w\textbf{u} \right)/(\partial z)`

    The weak form reads

    .. math::
        \int_\Omega \frac{\partial \left(w\textbf{u} \right)}{\partial z} \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \left( w \textbf{u} \right) \cdot \frac{\partial \boldsymbol{\psi}}{\partial z} dx
        + \int_{\mathcal{I}_{h}} \textbf{u}^{\text{up}} \cdot \text{jump}(\boldsymbol{\psi} n_z) \text{avg}(w) dS
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        w = fields_old.get('w')
        w_mesh = fields_old.get('w_mesh')
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_factor')
        if w is None or not self.nonlin:
            return 0
        f = 0

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
        else:
            uv = solution

        vertvelo = w[2]
        if w_mesh is not None:
            vertvelo = w[2]-w_mesh
        adv_v = -(Dx(self.test[0], 2)*uv[0]*vertvelo +
                  Dx(self.test[1], 2)*uv[1]*vertvelo)
        f += adv_v * self.dx
        if self.vertical_dg:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
            uv_up = uv('-')*s + uv('+')*(1-s)
            f += (uv_up[0]*w_av*jump(self.test[0], self.normal[2]) +
                  uv_up[1]*w_av*jump(self.test[1], self.normal[2]))*self.dS_h
            if lax_friedrichs_factor is not None:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                f += gamma*(jump(self.test[0])*jump(uv[0]) +
                            jump(self.test[1])*jump(uv[1]))*self.dS_h
        f += (uv[0]*vertvelo*self.test[0]*self.normal[2] +
              uv[1]*vertvelo*self.test[1]*self.normal[2])*(self.ds_surf)
        # NOTE bottom impermeability condition is naturally satisfied by the defition of w
        return -f


class HorizontalViscosityTerm(MomentumTerm):
    r"""
    Horizontal viscosity term, :math:`- \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        - \int_\Omega \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right) \cdot \boldsymbol{\psi} dx
        =& \int_\Omega \nu_h (\nabla_h \boldsymbol{\psi}) : (\nabla_h \textbf{u})^T dx \\
        &- \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\boldsymbol{\psi} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \textbf{u}) dS
        - \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \boldsymbol{\psi}) dS \\
        &+ \int_{\mathcal{I}_h \cup \mathcal{I}_v} \sigma \text{avg}(\nu_h) \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{jump}(\boldsymbol{\psi} \textbf{n}_h) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    .. note ::
        Note the minus sign due to :class:`.equation.Term` sign convention
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_h = fields_old.get('viscosity_h')
        if viscosity_h is None:
            return 0
        f = 0

        uv_depth_av = fields_old.get('uv_depth_av')
        if uv_depth_av is not None:
            uv = solution + uv_depth_av
        else:
            uv = solution

        def grad_h(a):
            return as_matrix([[Dx(a[0], 0), Dx(a[0], 1), 0],
                              [Dx(a[1], 0), Dx(a[1], 1), 0],
                              [0, 0, 0]])
        visc_tensor = as_matrix([[viscosity_h, 0, 0],
                                 [0, viscosity_h, 0],
                                 [0, 0, 0]])

        grad_uv = grad_h(uv)
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
                             dot(avg(visc_tensor), tensor_jump(self.normal, uv)))*ds_interior
            f += -inner(avg(dot(visc_tensor, nabla_grad(self.test))),
                        tensor_jump(self.normal, uv))*ds_interior
            f += -inner(tensor_jump(self.normal, self.test),
                        avg(dot(visc_tensor, nabla_grad(uv))))*ds_interior

        # symmetric bottom boundary condition
        f += -inner(stress, outer(self.test, self.normal))*self.ds_surf
        f += -inner(stress, outer(self.test, self.normal))*self.ds_bottom

        # TODO boundary conditions
        # TODO impermeability condition at bottom
        # TODO implement as separate function
        return -f


class VerticalViscosityTerm(MomentumTerm):
    r"""
    Vertical viscosity term, :math:`- \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        - \int_\Omega \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right) \cdot \boldsymbol{\psi} dx
        =& \int_\Omega \nu \frac{\partial \boldsymbol{\psi}}{\partial z} \cdot \frac{\partial \textbf{u}}{\partial z} dx \\
        &- \int_{\mathcal{I}_h} \text{jump}(\boldsymbol{\psi} n_z) \cdot \text{avg}\left(\nu \frac{\partial \textbf{u}}{\partial z}\right) dS
        - \int_{\mathcal{I}_h} \text{jump}(\textbf{u} n_z) \cdot \text{avg}\left(\nu \frac{\partial \boldsymbol{\psi}}{\partial z}\right) dS \\
        &+ \int_{\mathcal{I}_h} \sigma \text{avg}(\nu) \text{jump}(\textbf{u} n_z) \cdot \text{jump}(\boldsymbol{\psi} n_z) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    .. note ::
        Note the minus sign due to :class:`.equation.Term` sign convention
    """
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
    r"""
    Quadratic bottom friction term, :math:`\tau_b = C_D \| \textbf{u}_b \| \textbf{u}_b`

    The weak formulation reads

    .. math::
        \int_{\Gamma_{bot}} \tau_b \cdot \boldsymbol{\psi} dx = \int_{\Gamma_{bot}} C_D \| \textbf{u}_b \| \textbf{u}_b \cdot \boldsymbol{\psi} dx

    where :math:`\textbf{u}_b` is reconstructed velocity in the middle of the
    bottom element:

    .. math::
        \textbf{u}_b = \textbf{u}\Big|_{\Gamma_{bot}} + \frac{\partial \textbf{u}}{\partial z}\Big|_{\Gamma_{bot}} h_b,

    :math:`h_b` being half of the element height.
    For implicit solvers we linearize the stress as
    :math:`\tau_b = C_D \| \textbf{u}_b^{n} \| \textbf{u}_b^{n+1}`

    The drag is computed from the law-of-the wall

    .. math::
        C_D = \left( \frac{\kappa}{\ln (h_b + z_0)/z_0} \right)^2

    where :math:`z_0` is the bottom roughness length, read from ``z0_friction``
    field.
    The user can override the :math:`C_D` value by providing ``quadratic_drag``
    field.

    """
    # TODO z_0 should be a field in the options dict, remove from physical_constants
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if self.use_bottom_friction:
            uv_depth_av = fields_old.get('uv_depth_av')
            if uv_depth_av is not None:
                uv = solution + uv_depth_av
                uv_old = solution_old + uv_depth_av
            else:
                uv = solution
                uv_old = solution_old

            drag = fields_old.get('quadratic_drag')
            if drag is None:
                z0_friction = physical_constants['z0_friction']
                z_bot = 0.5*self.v_elem_size
                von_karman = physical_constants['von_karman']
                drag = (von_karman / ln((z_bot + z0_friction)/z0_friction))**2
            # compute uv_bottom implicitly
            uv_bot = uv + Dx(uv, 2)*z_bot
            uv_bot_old = uv_old + Dx(uv_old, 2)*z_bot
            uv_bot_mag = sqrt(uv_bot_old[0]**2 + uv_bot_old[1]**2)
            stress = drag*uv_bot_mag*uv_bot
            bot_friction = (stress[0]*self.test[0] +
                            stress[1]*self.test[1])*self.ds_bottom
            f += bot_friction
        return -f


class LinearDragTerm(MomentumTerm):
    r"""
    Linear drag term, :math:`\tau_b = D \textbf{u}_b`

    where :math:`D` is the drag coefficient, read from ``linear_drag`` field.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        linear_drag = fields_old.get('linear_drag')
        f = 0
        # Linear drag (consistent with drag in 2D mode)
        if linear_drag is not None:
            bottom_fri = linear_drag*inner(self.test, solution)*self.dx
            f += bottom_fri
        return -f


class CoriolisTerm(MomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{u}}`
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-solution[1]*self.test[0] +
                           solution[0]*self.test[1])*self.dx
        return -f


class SourceTerm(MomentumTerm):
    r"""
    Generic momentum source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \cdot \boldsymbol{\psi} dx

    where :math:`\sigma` is a user defined vector valued :class:`Function`.

    This term also implements the wind stress, :math:`-\tau_w/(H \rho_0)`.
    :math:`\tau_w` is a user-defined wind stress :class:`Function`
    ``wind_stress``. The weak form is

    .. math::
        F_w = \int_{\Gamma_s} \frac{1}{\rho_0} \tau_w \cdot \boldsymbol{\psi} dx

    Wind stress is only included if vertical viscosity is provided.

    .. note ::
        Due to the sign convention of :class:`.equation.Term`, this term is assembled to the left hand side of the equation
    """
    # TODO implement wind stress as a separate term
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
    Hydrostatic 3D momentum equation :eq:`mom_eq_split` for mode split models
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 nonlin=True, use_bottom_friction=False,
                 use_elevation_gradient=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool nonlin: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        :kwarg bool use_elevation_gradient: If True includes external pressure
            gradient in the pressure gradient term
        """
        # TODO remove use_elevation_gradient because it's OBSOLETE
        # TODO rename for reflect the fact that this is eq for the split eqns
        super(MomentumEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, nonlin, use_bottom_friction,
                use_elevation_gradient)
        self.add_term(PressureGradientTerm(*args), 'source')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(VerticalViscosityTerm(*args), 'explicit')
        self.add_term(BottomFrictionTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
