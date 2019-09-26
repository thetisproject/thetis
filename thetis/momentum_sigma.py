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

The internal pressure gradient is computed as a separate diagnostic field:

.. math::
    \mathbf{F}_{pg} = g\nabla_h r.
    :label: int_pg_eq

In the case of purely barotropic problems the :math:`r` and
:math:`\mathbf{F}_{pg}` fields are omitted.

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
    'InternalPressureGradientCalculator',
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
                 use_nonlinear_equations=True, use_lax_friedrichs=True, use_bottom_friction=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_nonlinear_equations: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        """
        super(MomentumTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_continuity = continuity.horizontal
        self.vertical_continuity = continuity.vertical
        self.use_nonlinear_equations = use_nonlinear_equations
        self.use_lax_friedrichs = use_lax_friedrichs
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
    def get_bnd_functions(self, c_in, uv_in, elev_in, bnd_id, bnd_conditions):
        """
        Returns external values of w and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.

        :arg c_in: Internal value of vertical velocity
        :arg uv_in: Internal value of horizontal velocity
        :arg elev_in: Internal value of elevation
        :arg bnd_id: boundary id
        :type bnd_id: int
        :arg bnd_conditions: dict of boundary conditions:
            ``{bnd_id: {field: value, ...}, ...}``
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
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        int_pg = fields.get('int_pg')
        f = 0
        if int_pg is not None:
            total_h = fields_old.get('eta') + self.bathymetry #TODO move this to InternalPressureGradientCalculator or baroc_head_3d solver
            f += (total_h*int_pg[0]*self.test[0])*self.dx

        #f = (int_pg[0]*self.test[0] + int_pg[1]*self.test[1])*self.dx

        ext_pg = fields.get('ext_pg')
        if ext_pg is not None:
            f += (ext_pg[0]*self.test[0])*self.dx

        return -f


class HorizontalAdvectionTerm(MomentumTerm):
    r"""
    Horizontal advection term in the horizontal momentum equations, :math:`\nabla_h \cdot (\textbf{u} \textbf{u})`

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
        if not self.use_nonlinear_equations:
            return 0
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')
        huv_over_h = fields.get('huv_over_h')

        # modified for operator-splitting method used in Telemac3D
        uv = solution
        uv_old = solution_old
        #
       # f = -(Dx(self.test[0], 0)*uv[0]*uv_old[0])*self.dx
        f = -(Dx(uv_old[0]*self.test[0], 0)*uv[0])*self.dx # non-conservative form
        uv_av = avg(uv_old)
        un_av = (uv_av[0]*self.normal('-')[0])
        s = 0.5*(sign(un_av) + 1.0)
        uv_up = uv('-')*s + uv('+')*(1-s)
        if self.horizontal_continuity in ['dg', 'hdiv']:
           # f += (uv_up[0]*uv_av[0]*jump(self.test[0], self.normal[0]))*(self.dS_v + self.dS_h)
            f += (avg(uv)[0]*jump(self.test[0], uv_old[0]*self.normal[0]))*(self.dS_v + self.dS_h) # non-conservative form
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if uv_p1 is not None:
                    gamma = 0.5*abs(avg(uv_p1)[0]*self.normal('-')[0])*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    raise Exception('either uv_p1 or uv_mag must be given')
                f += gamma*(jump(self.test[0])*jump(uv[0]))*(self.dS_v + self.dS_h)
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if funcs is None:
                    un = dot(uv, self.normal)
                    uv_ext = uv - 2*un*self.normal
                    if self.use_lax_friedrichs:
                        gamma = 0.5*abs(un)*lax_friedrichs_factor
                        f += gamma*(self.test[0]*(uv[0] - uv_ext[0]))*ds_bnd
                else:
                    uv_in = uv
                    use_lf = True
                    if 'symm' in funcs:
                        # use internal normal velocity
                        # NOTE should this be symmetric normal velocity?
                        uv_ext = uv_in
                        use_lf = False
                    elif 'uv' in funcs:
                        # prescribe external velocity
                        uv_ext = funcs['uv']
                        un_ext = dot(uv_ext, self.normal)
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
                    else:
                        raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
                    if self.use_nonlinear_equations:
                        uv_av = 0.5*(uv_in + uv_ext)
                        un_av = uv_av[0]*self.normal[0]
                        s = 0.5*(sign(un_av) + 1.0)
                        uv_up = uv_in*s + uv_ext*(1-s)
                        f += (uv_up[0]*self.test[0]*self.normal[0]*uv_av[0])*ds_bnd
                        if use_lf:
                            # Lax-Friedrichs stabilization
                            if self.use_lax_friedrichs:
                                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                                f += gamma*(self.test[0]*(uv_in[0] - uv_ext[0]))*ds_bnd

        # surf/bottom boundary conditions: closed at bed, symmetric at surf
        f += (uv_old[0]*uv[0]*self.test[0]*self.normal[0])*(self.ds_surf)
        return -f


class VerticalAdvectionTerm(MomentumTerm):
    r"""
    Vertical advection term in the horizontal momentum equations, :math:`\partial \left(w\textbf{u} \right)/(\partial z)`

    The weak form reads

    .. math::
        \int_\Omega \frac{\partial \left(w\textbf{u} \right)}{\partial z} \cdot \boldsymbol{\psi} dx
        = - \int_\Omega \left( w \textbf{u} \right) \cdot \frac{\partial \boldsymbol{\psi}}{\partial z} dx
        + \int_{\mathcal{I}_{h}} \textbf{u}^{\text{up}} \cdot \text{jump}(\boldsymbol{\psi} n_z) \text{avg}(w) dS
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if not self.use_nonlinear_equations:
            return 0
        w = fields_old.get('w')
        w_mesh = fields_old.get('w_mesh')
        vertvelo = solution_old[1]
        vertvelo = w[1]
        if w_mesh is not None:
            vertvelo -= w_mesh
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')

        uv_3d = solution
        eta = fields_old.get('eta')
        bath = self.bathymetry
        sigma_dt = fields_old.get('sigma_dt')
        sigma_dx = fields_old.get('sigma_dx')
        omega = fields_old.get('omega')
        ###
        vertvelo = omega#sigma_dt + uv_3d[0]*sigma_dx + w[1]/(eta + bath)
        ###
        f = 0
       # adv_v = -(Dx(self.test[0], 1)*uv_3d[0]*vertvelo)
        adv_v = -(Dx(vertvelo*self.test[0], 1)*uv_3d[0]) # non-conservative form
        f += adv_v * self.dx
        if self.vertical_continuity in ['dg', 'hdiv']:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[1]('-')) + 1.0)
            uv_up = uv_3d('-')*s + uv_3d('+')*(1-s)
           # f += (uv_up[0]*w_av*jump(self.test[0], self.normal[1]))*self.dS_h
            f += (avg(uv_3d)[0]*jump(self.test[0], vertvelo*self.normal[1]))*self.dS_h # non-conservative form
            if self.use_lax_friedrichs:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[1])*lax_friedrichs_factor
                f += gamma*dot(jump(self.test[0]), jump(uv_3d[0]))*self.dS_h
        f += (uv_3d[0]*vertvelo*self.test[0]*self.normal[1])*(self.ds_surf)
        # NOTE bottom impermeability condition is naturally satisfied by the defition of w
        return -f


class HorizontalViscosityTerm(MomentumTerm):
    r"""
    Horizontal viscosity term in the horizontal momentum equations, :math:`- \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \nabla_h \cdot \left( \nu_h \nabla_h \textbf{u} \right) \cdot \boldsymbol{\psi} dx
        =& -\int_\Omega \nu_h (\nabla_h \boldsymbol{\psi}) : (\nabla_h \textbf{u})^T dx \\
        &+ \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\boldsymbol{\psi} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \textbf{u}) dS
        + \int_{\mathcal{I}_h \cup \mathcal{I}_v} \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{avg}( \nu_h \nabla_h \boldsymbol{\psi}) dS \\
        &- \int_{\mathcal{I}_h \cup \mathcal{I}_v} \sigma \text{avg}(\nu_h) \text{jump}(\textbf{u} \textbf{n}_h) \cdot \text{jump}(\boldsymbol{\psi} \textbf{n}_h) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
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
        uv = solution # for operator-splitting method used in Telemac3D

        grad_test = Dx(self.test[0], 0)
        stress = viscosity_h*Dx(uv[0], 0)
        f += inner(grad_test, stress)*self.dx

        if self.horizontal_continuity in ['dg', 'hdiv']:
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
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h + self.dS_v)
            f += alpha*inner(tensor_jump(self.normal[0], self.test[0]),
                             dot(avg(viscosity_h), tensor_jump(self.normal[0], uv[0])))*ds_interior
            f += -inner(avg(viscosity_h*Dx(self.test[0], 0)),
                        tensor_jump(self.normal[0], uv[0]))*ds_interior
            f += -inner(tensor_jump(self.normal[0], self.test[0]),
                        avg(viscosity_h*Dx(uv[0], 0)))*ds_interior

        # symmetric bottom boundary condition
        f += -(stress*self.test[0]*self.normal[0])*(self.ds_surf + self.ds_bottom)

        # TODO boundary conditions
        # TODO impermeability condition at bottom
        # TODO implement as separate function
        return -f


class VerticalViscosityTerm(MomentumTerm):
    r"""
    Vertical viscosity term in the horizontal momentum equations, :math:`- \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \frac{\partial }{\partial z}\left( \nu \frac{\partial \textbf{u}}{\partial z}\right) \cdot \boldsymbol{\psi} dx
        =& -\int_\Omega \nu \frac{\partial \boldsymbol{\psi}}{\partial z} \cdot \frac{\partial \textbf{u}}{\partial z} dx \\
        &+ \int_{\mathcal{I}_h} \text{jump}(\boldsymbol{\psi} n_z) \cdot \text{avg}\left(\nu \frac{\partial \textbf{u}}{\partial z}\right) dS
        + \int_{\mathcal{I}_h} \text{jump}(\textbf{u} n_z) \cdot \text{avg}\left(\nu \frac{\partial \boldsymbol{\psi}}{\partial z}\right) dS \\
        &- \int_{\mathcal{I}_h} \sigma \text{avg}(\nu) \text{jump}(\textbf{u} n_z) \cdot \text{jump}(\boldsymbol{\psi} n_z) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_v = fields_old.get('viscosity_v')
        if viscosity_v is None:
            return 0
        f = 0
        total_h = fields_old.get('eta') + self.bathymetry
        const = 1./total_h**2
        grad_test = Dx(const*self.test[0], 1)
        diff_flux = viscosity_v*Dx(solution[0], 1)
        f += inner(grad_test, diff_flux)*self.dx

        if self.vertical_continuity in ['dg', 'hdiv']:
            assert self.h_elem_size is not None, 'h_elem_size must be defined'
            assert self.v_elem_size is not None, 'v_elem_size must be defined'
            # Interior Penalty method by
            # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
            degree_h, degree_v = self.function_space.ufl_element().degree()
            # TODO compute elemsize as CellVolume/FacetArea
            # h = n.D.n where D = diag(h_h, h_h, h_v)
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_v*(degree_v + 1)/elemsize
            if degree_v == 0:
                sigma = 1.0/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h)
            f += alpha*inner(tensor_jump(self.normal[1], const*self.test[0]),
                             avg(viscosity_v)*tensor_jump(self.normal[1], solution[0]))*ds_interior
            f += -inner(avg(viscosity_v*Dx(const*self.test[0], 1)),
                        tensor_jump(self.normal[1], solution[0]))*ds_interior
            f += -inner(tensor_jump(self.normal[1], const*self.test[0]),
                        avg(viscosity_v*Dx(solution[0], 1)))*ds_interior
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
    The user can override the :math:`C_D` value by providing ``quadratic_drag_coefficient``
    field.
    """
    # TODO z_0 should be a field in the options dict, remove from physical_constants
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        if self.use_bottom_friction:
            uv = solution[0]
            uv_old = solution_old[0]

            z_bot = 0.5*self.v_elem_size
            drag = fields_old.get('quadratic_drag_coefficient')
            if drag is None:
                z0_friction = physical_constants['z0_friction']
                von_karman = physical_constants['von_karman']
                drag = (von_karman / ln((z_bot + z0_friction)/z0_friction))**2
            # compute uv_bottom implicitly
            uv_bot = uv + Dx(uv, 1)*z_bot
            uv_bot_old = uv_old + Dx(uv_old, 1)*z_bot
            uv_bot_mag = uv_bot_old
            stress = drag*uv_bot_mag*uv_bot
            bot_friction = (stress*self.test[0])*self.ds_bottom
            f += bot_friction
        return -f


class LinearDragTerm(MomentumTerm):
    r"""
    Linear drag term, :math:`\tau_b = D \textbf{u}_b`

    where :math:`D` is the drag coefficient, read from ``linear_drag_coefficient`` field.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        # Linear drag (consistent with drag in 2D mode)
        if linear_drag_coefficient is not None:
            uv = solution[0]

            bottom_fri = linear_drag_coefficient*inner(self.test[0], uv)*self.dx
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
                           solution[0]*self.test[1])*self.dx #TODO seems no Coriolis force in vertical 2d equation
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
    """
    # TODO implement wind stress as a separate term
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        viscosity_v = fields_old.get('viscosity_v')
        wind_stress = fields_old.get('wind_stress')
        if wind_stress is not None and viscosity_v is None:
            warning('Wind stress is prescribed but vertical viscosity is not:\n  Wind stress will be ignored.')
        if viscosity_v is not None:
            # wind stress
            if wind_stress is not None:
                f -= (wind_stress[0]*self.test[0])/rho_0*self.ds_surf
        if source is not None:
            f += - inner(source, self.test[0])*self.dx
        return -f


class ElevationGradientTerm(MomentumTerm):
    r"""
    Horizontal advection term in the vertical momentum equations, :math:`\textbf{u} \cdot \nabla w`

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if not self.use_nonlinear_equations:
            return 0
        f = 0

        eta = fields_old.get('eta')
        total_h = eta + self.bathymetry
        test = self.test[0]
        uw = solution
        normal = self.normal
        dS_v = self.dS_v

        f += -g_grav*eta*Dx(test, 0)*dx
        eta_star = avg(eta) #+ sqrt(avg(total_h)/g_grav)*jump(uw[0], normal[0])
        f += g_grav*eta_star*jump(test, normal[0])*dS_v
        f += g_grav*eta*(test*normal[0])*(self.ds_bottom + ds_surf)
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None and 'elev3d' in funcs:
                eta_ext = funcs['elev3d']
                eta_rie = 0.5*(eta + eta_ext)
                f += g_grav*eta_rie*dot(test, normal[0])*ds_bnd
            if funcs is None or 'symm' in funcs:
                # assume land boundary
                f += g_grav*eta*dot(test, normal[0])*ds_bnd
        return -f


class MomentumEquation(Equation):
    """
    Hydrostatic 3D momentum equation :eq:`mom_eq_split` for mode split models
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_nonlinear_equations=True, use_lax_friedrichs=True, use_bottom_friction=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_nonlinear_equations: If False defines the linear shallow water equations
        :kwarg bool use_bottom_friction: If True includes bottom friction term
        """
        # TODO rename for reflect the fact that this is eq for the split eqns
        super(MomentumEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_nonlinear_equations, use_lax_friedrichs, use_bottom_friction)
        self.add_term(PressureGradientTerm(*args), 'source')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(VerticalViscosityTerm(*args), 'explicit')
        self.add_term(BottomFrictionTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
       # self.add_term(ElevationGradientTerm(*args), 'explicit')


class InternalPressureGradientCalculator(MomentumTerm):
    r"""
    Computes the internal pressure gradient term, :math:`g\nabla_h r`

    where :math:`r` is the baroclinic head :eq:`baroc_head`.

    If :math:`r` belongs to a discontinuous function space, the term is
    integrated by parts:

    .. math::
        \int_\Omega g \nabla_h r \cdot \boldsymbol{\psi} dx
            = - \int_\Omega g r \nabla_h \cdot \boldsymbol{\psi} dx
            + \int_{\mathcal{I}_h \cup \mathcal{I}_v} g \text{avg}(r) \text{jump}(\boldsymbol{\psi} \cdot \textbf{n}_h) dx

    .. note ::
        Due to the :class:`Term` sign convention this term is assembled on the right-hand-side.
    """
    def __init__(self, fields, options, bnd_functions, solver_parameters=None):
        """
        :arg solver: `class`FlowSolver` object
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        self.fields = fields
        self.options = options
        function_space = self.fields.int_pg_3d.function_space()
        bathymetry = self.fields.bathymetry_3d
        super(InternalPressureGradientCalculator, self).__init__(
            function_space, bathymetry=bathymetry)

        solution = self.fields.int_pg_3d
        fields = {
            'baroc_head': self.fields.baroc_head_3d,
        }
        l = -self.residual(solution, solution, fields, fields,
                           bnd_conditions=bnd_functions)
        trial = TrialFunction(self.function_space)
        a = inner(trial, self.test) * self.dx
        prob = LinearVariationalProblem(a, l, solution)
        self.lin_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """
        Computes internal pressure gradient and stores it in int_pg_3d field
        """
        self.lin_solver.solve()

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        bhead = fields_old.get('baroc_head')

        if bhead is None:
            return 0

        by_parts = element_continuity(bhead.function_space().ufl_element()).horizontal == 'dg'

        if by_parts:
            div_test = (Dx(self.test[0], 0))
            f = -g_grav*bhead*div_test*self.dx
            head_star = avg(bhead)
            jump_n_dot_test = (jump(self.test[0], self.normal[0]))
            f += g_grav*head_star*jump_n_dot_test*(self.dS_v + self.dS_h)
            n_dot_test = (self.normal[0]*self.test[0])
            f += g_grav*bhead*n_dot_test*(self.ds_bottom + self.ds_surf)
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                if bhead is not None:
                    if funcs is not None and 'baroc_head' in funcs:
                        r_ext = funcs['baroc_head']
                        head_ext = r_ext
                        head_in = bhead
                        head_star = 0.5*(head_ext + head_in)
                    else:
                        head_star = bhead
                    f += g_grav*head_star*n_dot_test*ds_bnd

        else:
            grad_head_dot_test = (Dx(bhead, 0)*self.test[0])
            f = g_grav * grad_head_dot_test * self.dx

        return -f


####################################
##                                ##
###                              ###
#### Vertical Momentum Equation #### <<<<####################################################### Alternatively #######################################################
###                              ###
##                                ##
####################################


class VertMomentumTerm(Term):
    """
    Generic vertical momentum term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True, use_lax_friedrichs=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(VertMomentumTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_dg = continuity.horizontal == 'dg'
        self.vertical_dg = continuity.vertical == 'dg'
        self.use_symmetric_surf_bnd = use_symmetric_surf_bnd
        self.use_lax_friedrichs = use_lax_friedrichs

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
        Returns external values of w and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.

        :arg c_in: Internal value of vertical velocity
        :arg uv_in: Internal value of horizontal velocity
        :arg elev_in: Internal value of elevation
        :arg bnd_id: boundary id
        :type bnd_id: int
        :arg bnd_conditions: dict of boundary conditions:
            ``{bnd_id: {field: value, ...}, ...}``
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


class HorizontalAdvectionTerm_in_VertMom(VertMomentumTerm):
    r"""
    Horizontal advection term :math:`\nabla_h \cdot (\textbf{u} w)`

    The weak formulation reads

    .. math::
        \int_\Omega \nabla_h \cdot (\textbf{u} w) \phi dx
            = -\int_\Omega w\textbf{u} \cdot \nabla_h \phi dx
            + \int_{\mathcal{I}_h\cup\mathcal{I}_v}
                w^{\text{up}} \text{avg}(\textbf{u}) \cdot
                \text{jump}(\phi \textbf{n}_h) dS

    where the right hand side has been integrated by parts;
    :math:`\mathcal{I}_h,\mathcal{I}_v` denote the set of horizontal and
    vertical facets,
    :math:`\textbf{n}_h` is the horizontal projection of the unit normal vector,
    :math:`w^{\text{up}}` is the upwind value, and :math:`\text{jump}` and
    :math:`\text{avg}` denote the jump and average operators across the
    interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_3d') is None:
            return 0
        elev = fields_old['eta']
        uv = fields_old['uv_3d']

       # cut for operator-splitting method in NH modelling
       # uv_depth_av = fields_old['uv_depth_av']
       # if uv_depth_av is not None:
       #     uv = uv + uv_depth_av

        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')

        f = 0
       # f += -solution[1]*inner(uv[0], Dx(self.test[1], 0))*self.dx
        f += -solution[1]*Dx(uv[0]*self.test[1], 0)*self.dx # non-conservative form
        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
           # f += c_up[1]*(uv_av[0]*jump(self.test[1], self.normal[0]))*(self.dS_v + self.dS_h)
            f += avg(solution)[1]*jump(self.test[1], uv[0]*self.normal[0])*(self.dS_v + self.dS_h) # non-conservative form
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if uv_p1 is not None:
                    gamma = 0.5*abs(avg(uv_p1)[0]*self.normal('-')[0])*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    raise Exception('either uv_p1 or uv_mag must be given')
                f += gamma*dot(jump(self.test[1]), jump(solution[1]))*(self.dS_v + self.dS_h)
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
                        un_av = self.normal[0]*uv_av[0]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        f += c_up[1]*(uv_av[0]*self.normal[0])*self.test[1]*ds_bnd

        if self.use_symmetric_surf_bnd:
            f += solution[1]*(uv[0]*self.normal[0])*self.test[1]*ds_surf
        return -f


class VerticalAdvectionTerm_in_VertMom(VertMomentumTerm):
    r"""
    Vertical advection term :math:`\partial (w w)/(\partial z)`

    The weak form reads

    .. math::
        \int_\Omega \frac{\partial (w w)}{\partial z} \phi dx
        = - \int_\Omega w w \frac{\partial \phi}{\partial z} dx
        + \int_{\mathcal{I}_v} w^{\text{up}} \text{avg}(w) \text{jump}(\phi n_z) dS

    where the right hand side has been integrated by parts;
    :math:`\mathcal{I}_v` denotes the set of vertical facets,
    :math:`n_z` is the vertical projection of the unit normal vector,
    :math:`w^{\text{up}}` is the
    upwind value, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.

    In the case of ALE moving mesh we substitute :math:`w` with :math:`w - w_m`,
    :math:`w_m` being the mesh velocity.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        w = fields_old.get('w')
        if w is None:
            return 0
        w_mesh = fields_old.get('w_mesh')
        vertvelo = w[1]
        if w_mesh is not None:
            vertvelo -= w_mesh
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_velocity_scaling_factor')

        uv_3d = fields_old.get('uv_3d')
        eta = fields_old.get('eta')
        bath = self.bathymetry
        sigma_dt = fields_old.get('sigma_dt')
        sigma_dx = fields_old.get('sigma_dx')
        omega = fields_old.get('omega')
        ###
        vertvelo = omega#sigma_dt + uv_3d[0]*sigma_dx + w[1]/(eta + bath)
        ###
        f = 0
       # f += -solution[1]*vertvelo*Dx(self.test[1], 1)*self.dx
        f += -solution[1]*Dx(vertvelo*self.test[1], 1)*self.dx # non-conservative form
        if self.vertical_dg:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[1]('-')) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
           # f += c_up[1]*w_av*jump(self.test[1], self.normal[1])*self.dS_h
            f += avg(solution)[1]*jump(self.test[1], vertvelo*self.normal[1])*self.dS_h # non-conservative form
            if self.use_lax_friedrichs:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[1])*lax_friedrichs_factor
                f += gamma*dot(jump(self.test[1]), jump(solution[1]))*self.dS_h

        # NOTE Bottom impermeability condition is naturally satisfied by the definition of w
        # NOTE imex solver fails with this in tracerBox example, also in bb_bar case
       # f += solution[1]*vertvelo*self.normal[1]*self.test[1]*self.ds_surf
        return -f


class HorizontalViscosityTerm_in_VertMom(VertMomentumTerm):
    r"""
    Horizontal Viscosity term :math:`-\nabla_h \cdot (\mu_h \nabla_h w)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        =& -\int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h T) dS
        + \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(T \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(T \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('viscosity_h') is None:
            return 0

        viscosity_h = fields_old.get('viscosity_h')

        grad_test = Dx(self.test[1], 0)
        diff_flux = dot(viscosity_h, Dx(solution[1], 0))

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
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h + self.dS_v)
            f += alpha*inner(jump(self.test[1], self.normal[0]),
                             dot(avg(viscosity_h), jump(solution[1], self.normal[0])))*ds_interior
            f += -inner(avg(dot(viscosity_h, Dx(self.test[1], 0))),
                        jump(solution[1], self.normal[0]))*ds_interior
            f += -inner(jump(self.test[1], self.normal[0]),
                        avg(dot(viscosity_h, Dx(solution[1], 0))))*ds_interior

        # symmetric bottom boundary condition
        # NOTE introduces a flux through the bed - breaks mass conservation
        f += - inner(diff_flux, self.normal[0])*self.test[1]*(self.ds_bottom + self.ds_surf)

        return -f


class VerticalViscosityTerm_in_VertMom(VertMomentumTerm):
    r"""
    Vertical Viscosity term :math:`-\frac{\partial}{\partial z} \Big(\mu \frac{w}{\partial z}\Big)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \frac{\partial}{\partial z} \Big(\mu \frac{w}{\partial z}\Big) \phi dx
        =& -\int_\Omega \mu \frac{\partial w}{\partial z} \frac{\partial \phi}{\partial z} dz \\
        &+ \int_{\mathcal{I}_{h}} \text{jump}(\phi n_z) \text{avg}\Big(\mu \frac{\partial w}{\partial z}\Big) dS
        + \int_{\mathcal{I}_{h}} \text{jump}(w n_z) \text{avg}\Big(\mu \frac{\partial \phi}{\partial z}\Big) dS \\
        &- \int_{\mathcal{I}_{h}} \sigma \text{avg}(\mu) \text{jump}(w n_z) \cdot
            \text{jump}(\phi n_z) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        viscosity_v = fields_old.get('viscosity_v')
        if viscosity_v is None:
            return 0

        total_h = fields_old.get('eta') + self.bathymetry
        const = 1./total_h**2
        grad_test = Dx(const*self.test[1], 1)
        diff_flux = dot(viscosity_v, Dx(solution[1], 1))

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
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_v*(degree_v + 1)/elemsize
            if degree_v == 0:
                sigma = 1.0/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h)
            f += alpha*inner(jump(const*self.test[1], self.normal[1]),
                             dot(avg(viscosity_v), jump(solution[1], self.normal[1])))*ds_interior
            f += -inner(avg(dot(viscosity_v, Dx(const*self.test[1], 1))),
                        jump(solution[1], self.normal[1]))*ds_interior
            f += -inner(jump(const*self.test[1], self.normal[1]),
                        avg(dot(viscosity_v, Dx(solution[1], 1))))*ds_interior

        return -f


class SourceTerm_in_VertMom(VertMomentumTerm):
    """
    Generic source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += inner(source, self.test[1])*self.dx
        return f


class VertMomentumEquation(Equation):
    """
    Vertical momentum equation in conservative form
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True, use_lax_friedrichs=True):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(VertMomentumEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_symmetric_surf_bnd, use_lax_friedrichs)
        self.add_term(HorizontalAdvectionTerm_in_VertMom(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm_in_VertMom(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm_in_VertMom(*args), 'explicit')
        self.add_term(VerticalViscosityTerm_in_VertMom(*args), 'explicit')
        self.add_term(SourceTerm_in_VertMom(*args), 'source')





















