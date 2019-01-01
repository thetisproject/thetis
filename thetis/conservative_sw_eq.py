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
                 bathymetry=None,
                 options=None):
        super(CShallowWaterTerm, self).__init__(space)

        self.bathymetry = bathymetry
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
        bath = self.bathymetry
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
        h_ext = eta_ext + self.bathymetry
        hu_ext = h_ext * uv_ext  # FIXME generalize this
        return h_ext, hu_ext


class CShallowWaterMomentumTerm(CShallowWaterTerm):
    """
    Generic term in the shallow water momentum equation that provides commonly
    used members and mapping for boundary functions.
    """
    def __init__(self, hu_test, hu_space, h_space,
                 bathymetry=None,
                 options=None):
        super(CShallowWaterMomentumTerm, self).__init__(hu_space,
                                                        bathymetry, options)

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
                 bathymetry=None,
                 options=None):
        super(CShallowWaterContinuityTerm, self).__init__(h_space,
                                                          bathymetry, options)

        self.h_test = h_test
        self.h_space = h_space
        self.hu_space = hu_space

        hu_elem = self.hu_space.ufl_element()
        h_elem = self.h_space.ufl_element()
        self.hu_continuity = element_continuity(hu_elem).horizontal
        self.h_is_dg = element_continuity(h_elem).horizontal == 'dg'


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
            hu_av = avg(hu_old)
            hu_mag = sqrt(hu_av[0]*hu_av[0] + hu_av[1]*hu_av[1])
            u = sqrt(avg(h_old)*g_grav) + hu_mag/avg(h_old)
            # edge_flux = avg(flux)
            edge_flux = avg(flux) + u*jump(hu, self.normal)
            f += edge_flux * jump(self.hu_test, self.normal) * self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    h_ext, hu_ext = self.get_bnd_functions(h, hu, bnd_marker, bnd_conditions)
                    # FIXME
                else:
                    f += flux * inner(self.hu_test, self.normal) * ds_bnd
            # bathymetry source term
            f += -inner(g_grav * h * grad(self.bathymetry), self.hu_test) * self.dx
        else:
            raise NotImplementedError('only flux form of pressure gradient is implemented')
        return -f


class HUDivTerm(CShallowWaterContinuityTerm):
    r"""
    Divergence term, ...
    """
    # FIXME documentation
    def residual(self, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions=None):
        f = -inner(grad(self.h_test), hu) * self.dx
        if self.h_is_dg:
            hu_av = avg(hu_old)
            hu_mag = sqrt(hu_av[0]*hu_av[0] + hu_av[1]*hu_av[1])
            u = sqrt(avg(h_old)*g_grav) + hu_mag/avg(h_old)
            hu_star = avg(hu) + u*jump(h, self.normal)
            f += inner(jump(self.h_test, self.normal), hu_star)*self.dS
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None:
                h_ext, hu_ext = self.get_bnd_functions(h, hu, bnd_marker, bnd_conditions)
                h_ext_old, hu_ext_old = self.get_bnd_functions(h_old, hu_old, bnd_marker, bnd_conditions)
                # FIXME
        return -f


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
        edge_flux_x = avg(flux_x)
        edge_flux_y = avg(flux_y)
        f += inner(jump(self.hu_test[0], self.normal), edge_flux_x) * self.dS
        f += inner(jump(self.hu_test[1], self.normal), edge_flux_y) * self.dS

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
    def __init__(self, function_space,
                 bathymetry,
                 options):
        super(BaseCShallowWaterEquation, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(WindStressTerm(*args), 'source')
        self.add_term(AtmosphericPressureTerm(*args), 'source')
        self.add_term(QuadraticDragTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')

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
    def __init__(self, function_space,
                 bathymetry,
                 options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(CShallowWaterEquations, self).__init__(function_space, bathymetry, options)

        hu_test, h_test = TestFunctions(function_space)
        hu_space, h_space = function_space.split()

        self.add_momentum_terms(hu_test, hu_space, h_space,
                                bathymetry, options)

        self.add_continuity_terms(h_test, h_space, hu_space, bathymetry, options)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            hu, h = solution
        else:
            hu, h = split(solution)
        hu_old, h_old = split(solution_old)
        return self.residual_hu_h(label, hu, h, hu_old, h_old, fields, fields_old, bnd_conditions)

