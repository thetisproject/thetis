r"""
Shallow water equations in conservative form


"""
#FIXME documentation
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'BaseCShallowWaterEquation',
    'ShallowWaterEquations',
    'CShallowWaterTerm',
    'CShallowWaterMomentumTerm',
    'CShallowWaterContinuityTerm',
    'HUDivTerm',
    'ExternalPressureGradientTerm',
    'HorizontalAdvectionTerm',
    'CoriolisTerm',
    'WindStressTerm',
    'AtmosphericPressureTerm',
    'QuadraticDragTerm',
    'LinearDragTerm',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class CShallowWaterTerm(Term):
    """
    Generic term in the shallow water equations that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, space,
                 depth,
                 options=None):
        super(CShallowWaterTerm, self).__init__(space)

        self.depth = depth
        self.options = options

        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        self.dS = dS(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())

    def get_bnd_functions(self, h_in, hu_in, bnd_id, bnd_conditions):
        """
        Returns external values of h and hu for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
        # FIXME
        bath = self.depth.bathymetry_2d
        bnd_len = self.boundary_len[bnd_id]
        funcs = bnd_conditions.get(bnd_id)
        if 'elev' in funcs and 'uv' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['uv']
        elif 'elev' in funcs and 'un' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['un']*self.normal
        elif 'elev' in funcs and 'flux' in funcs:
            eta_ext = funcs['elev']
            h_ext = eta_ext + bath
            area = h_ext*bnd_len  # NOTE using external data only
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
            h_ext = eta_ext + bath
            area = h_ext*bnd_len  # NOTE using internal elevation
            uv_ext = funcs['flux']/area*self.normal
        else:
            raise Exception('Unsupported bnd type: {:}'.format(funcs.keys()))
        h_ext = eta_ext + bath
        hu_ext = h_ext * uv_ext  # FIXME generalize this
        return h_ext, hu_ext


class CShallowWaterMomentumTerm(CShallowWaterTerm):
    """
    Generic term in the shallow water momentum equation that provides commonly
    used members and mapping for boundary functions.
    """
    def __init__(self, hu_test, hu_space, h_space,
                 depth,
                 options=None):
        super(CShallowWaterMomentumTerm, self).__init__(hu_space,
                                                        depth, options)

        self.options = options

        self.hu_test = hu_test
        self.hu_space = hu_space
        self.h_space = h_space

        hu_elem = self.hu_space.ufl_element()
        h_elem = self.h_space.ufl_element()
        self.hu_continuity = element_continuity(hu_elem).horizontal
        self.h_is_dg = element_continuity(h_elem).horizontal == 'dg'


class CShallowWaterContinuityTerm(CShallowWaterTerm):
    """
    Generic term in the depth-integrated continuity equation that provides
    commonly used members and mapping for boundary functions.
    """
    def __init__(self, h_test, h_space, hu_space,
                 depth,
                 options=None):
        super(CShallowWaterContinuityTerm, self).__init__(h_space,
                                                          depth, options)

        self.h_test = h_test
        self.h_space = h_space
        self.hu_space = hu_space

        hu_elem = self.hu_space.ufl_element()
        h_elem = self.h_space.ufl_element()
        self.hu_continuity = element_continuity(hu_elem).horizontal
        self.h_is_dg = element_continuity(h_elem).horizontal == 'dg'


def flux_hll_wave_speed(hu, h, bath, normal):
    """
    Compute left/right wave speeds.
    """
    uv_minus = hu('-')/h('-')
    uv_plus = hu('+')/h('+')

    un_minus = dot(uv_minus, normal('-'))
    un_plus = dot(uv_plus, normal('-'))

    c_minus = sqrt(g_grav * h('-'))
    c_plus = sqrt(g_grav * h('+'))

    u_star = 0.5 * (un_minus + un_plus) + c_minus - c_plus
    h_star = (0.5*(c_minus + c_plus) + 0.25*(un_minus - un_plus))**2 / g_grav
    c_star = sqrt(g_grav * h_star)

    s_star_minus = u_star - c_star
    s_star_plus = u_star + c_star

    # wave speeds
    # FIXME simplified wave speeds
    s_minus = un_minus - sqrt(g_grav * h('-'))
    s_plus = un_plus + sqrt(g_grav * h('+'))

    s_low = conditional(s_minus < s_star_minus, s_minus, s_star_minus)
    s_hi = conditional(s_plus > s_star_plus, s_plus, s_star_plus)

    return s_low, s_hi


def flux_hll_penalty_scalar(hu, h, bath, normal):
    """
    Return scaling factor for variable jump term.
    """
    s_minus, s_plus = flux_hll_wave_speed(hu, h, bath, normal)
    return s_minus * s_plus / (s_plus - s_minus)


def flux_hll_pg(hu, h, bath, normal):
    """
    External pressure gradient flux
    """
    f_minus = 0.5 * g_grav * h('-')**2
    f_plus = 0.5 * g_grav * h('+')**2
    s_minus, s_plus = flux_hll_wave_speed(hu, h, bath, normal)
    return (s_plus * f_minus - s_minus * f_plus)/(s_plus - s_minus)


class ExternalPressureGradientTerm(CShallowWaterMomentumTerm):
    r"""
    External pressure gradient term, ...
    """
    # FIXME documentation
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):

        grad_h_by_parts = self.h_is_dg

        flux = 0.5 * g_grav * h_old * h
        if grad_h_by_parts:
            f = - flux * nabla_div(self.hu_test) * self.dx
            edge_flux = flux_hll_pg(hu, h, self.depth.bathymetry_2d, self.normal)
            f += inner(jump(self.hu_test, self.normal), edge_flux) * self.dS
            edge_penalty = flux_hll_penalty_scalar(hu, h, self.depth.bathymetry_2d, self.normal)
            f += -edge_penalty * inner(jump(hu), jump(self.hu_test)) * self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    h_ext, hu_ext = self.get_bnd_functions(h, hu, bnd_marker, bnd_conditions)
                    # FIXME
                else:
                    f += flux * inner(self.hu_test, self.normal) * ds_bnd
            # bathymetry source term
            f += -inner(g_grav * h * grad(self.depth.bathymetry_2d), self.hu_test) * self.dx
        else:
            raise NotImplementedError('only flux form of pressure gradient is implemented')
        return -f


def flux_hll_divhu(hu, h, bath, normal):
    """
    div(hu) flux
    """
    f_minus = hu('-')
    f_plus = hu('+')
    s_minus, s_plus = flux_hll_wave_speed(hu, h, bath, normal)
    return (s_plus * f_minus - s_minus * f_plus)/(s_plus - s_minus)


class HUDivTerm(CShallowWaterContinuityTerm):
    r"""
    Divergence term, ...
    """
    # FIXME documentation
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        f = -inner(grad(self.h_test), hu) * self.dx
        if self.h_is_dg:
            edge_flux = flux_hll_divhu(hu, h, self.depth.bathymetry_2d, self.normal)
            f += inner(jump(self.h_test, self.normal), edge_flux) * self.dS
            edge_penalty = flux_hll_penalty_scalar(hu, h, self.depth.bathymetry_2d, self.normal)
            f += -edge_penalty * inner(jump(h), jump(self.h_test)) * self.dS
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None:
                h_ext, hu_ext = self.get_bnd_functions(h, hu, bnd_marker, bnd_conditions)
                h_ext_old, hu_ext_old = self.get_bnd_functions(h_old, hu_old, bnd_marker, bnd_conditions)
                # FIXME
        return -f


def flux_hll_adv(hu, h, bath, normal):
    """
    advection flux
    """

    hun_minus = dot(hu('-'), normal('-'))
    hun_plus = dot(hu('+'), normal('-'))

    un_minus = hun_minus / h('-')
    un_plus = hun_plus / h('+')

    ut_minus = (hu('-') - hun_minus * normal('-')) / h('-')
    ut_plus = (hu('+') - hun_plus * normal('-')) / h('+')

    s_minus, s_plus = flux_hll_wave_speed(hu, h, bath, normal)

    s_mean = 0.5 * (s_minus + s_plus)  # FIXME this is wrong
    ut_upwind = conditional(s_mean > 0, ut_minus, ut_plus)

    f_minus = hun_minus * un_minus * normal('-') + ut_minus * hun_minus
    f_plus = hun_plus * un_plus * normal('-') + ut_plus * hun_plus

    return (s_plus * f_minus - s_minus * f_plus)/(s_plus - s_minus)


class HorizontalAdvectionTerm(CShallowWaterMomentumTerm):
    r"""
    Advection of momentum term ...
    """
    # FIXME documentation
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        # flux = [(huu)_x + (huv)_y, (huv)_x + (hvv)_y]

        flux_x = hu_old[0] * hu / h_old
        flux_y = hu_old[1] * hu / h_old

        f = -(inner(flux_x, grad(self.hu_test[0]))
              + inner(flux_y, grad(self.hu_test[1]))) * self.dx
        edge_flux = flux_hll_adv(hu, h, self.depth.bathymetry_2d, self.normal)
        f += -inner(jump(self.hu_test), edge_flux) * self.dS

        return -f


class CoriolisTerm(CShallowWaterMomentumTerm):
    r"""
    Coriolis term, :math:`f\textbf{e}_z\wedge \bar{\textbf{Hu}}`
    """
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-hu[1]*self.hu_test[0] + hu[0]*self.hu_test[1])*self.dx
        return -f


class WindStressTerm(CShallowWaterMomentumTerm):
    r"""
    Wind stress term, :math:`-\tau_w/\rho_0`

    Here :math:`\tau_w` is a user-defined wind stress :class:`Function`.
    """
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        wind_stress = fields_old.get('wind_stress')
        f = 0
        if wind_stress is not None:
            f += dot(wind_stress, self.hu_test)/rho_0*self.dx
        return f


class AtmosphericPressureTerm(CShallowWaterMomentumTerm):
    r"""
    Atmospheric pressure term, :math:`\nabla (p_a / \rho_0) H`

    Here :math:`p_a` is a user-defined atmospheric pressure :class:`Function`.
    """
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        f = 0
        if atmospheric_pressure is not None:
            f += dot(grad(atmospheric_pressure), self.hu_test)*h_old/rho_0*self.dx
        return -f


class QuadraticDragTerm(CShallowWaterMomentumTerm):
    r"""
    Quadratic Manning bottom friction term
    :math:`C_D \| \bar{\textbf{Hu}} \| \bar{\textbf{Hu}}/H`

    where the drag term is computed with the Manning formula

    .. math::
        C_D = g \frac{\mu^2}{H^{1/3}}

    if the Manning coefficient :math:`\mu` is defined (see field :attr:`manning_drag_coefficient`).
    Otherwise :math:`C_D` is taken as a constant (see field :attr:`quadratic_drag_coefficient`).
    """
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / h_old**(1./3.)

        if C_D is not None:
            f += C_D * sqrt(dot(hu_old, hu_old)) * inner(self.hu_test, hu) / h_old**2 * self.dx
        return -f


class LinearDragTerm(CShallowWaterMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{Hu}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*inner(self.hu_test, hu)*self.dx
            f += bottom_fri
        return -f


class BaseCShallowWaterEquation(Equation):
    """
    Abstract base class for CShallowWaterEquation.

    Provides common functionality to compute time steps and add either momentum
    or continuity terms.
    """
    def __init__(self, function_space, depth, options):
        super(BaseCShallowWaterEquation, self).__init__(function_space)
        self.depth = depth
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        #self.add_term(CoriolisTerm(*args), 'explicit')
        #self.add_term(WindStressTerm(*args), 'source')
        #self.add_term(AtmosphericPressureTerm(*args), 'source')
        #self.add_term(QuadraticDragTerm(*args), 'explicit')
        #self.add_term(LinearDragTerm(*args), 'explicit')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivTerm(*args), 'implicit')

    def residual_hu_h(self, label, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions):
        f = 0
        for term in self.select_terms(label):
            f += term.residual(hu, h, hu_old, h_old, fields, fields_old, bnd_conditions)
        return f


class CShallowWaterEquations(BaseCShallowWaterEquation):
    """
    2D depth-averaged shallow water equations in conservative form.

    """
    # FIXME documentation
    def __init__(self, function_space, depth, options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg depth: :class: `DepthExpression` containing depth info
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(CShallowWaterEquations, self).__init__(function_space, depth, options)

        hu_test, h_test = TestFunctions(function_space)
        hu_space, h_space = function_space.split()

        self.add_momentum_terms(hu_test, hu_space, h_space,
                                depth, options)

        self.add_continuity_terms(h_test, h_space, hu_space, depth, options)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            hu, h = solution
        else:
            hu, h = split(solution)
        hu_old, h_old = split(solution_old)
        return self.residual_hu_h(label, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions)

