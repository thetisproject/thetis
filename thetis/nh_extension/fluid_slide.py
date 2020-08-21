"""
calculation of the landslide motion to obtain the function D(x,y,t), i.e. the thickness of the slide
ref: I.V. Fine, et al., 1998 & 2005. 
"""
from __future__ import absolute_import
from .utility_nh import *
from thetis.equation import Term, Equation

__all__ = [
    'LandslideTerm',
    'LandslideMomentumTerm',
    'LandslideContinuityTerm',
    'FluidSlideHUDivTerm',
    'FluidSlideContinuitySourceTerm',
    'FluidSlideHorizontalAdvectionTerm',
    'FluidSlideHorizontalViscosityTerm',
    'FluidSlidePressureGradientTerm',
    'FluidSlideLinearDragTerm',
    'FluidSlideQuadraticDragTerm',
    'FluidSlideMomentumSourceTerm',
    'FluidSlideMomDivTerm',
    'BaseFluidSlideEquation',
    'FluidSlideEquations',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']
k = 1
lamda = 0.
phi_bed = 0.*pi/180.
phi_int = 0#41.*pi/180.
fluid_volume_fraction = 0.

class LandslideTerm(Term):
    """
    Generic term in the landslide motion equations that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, space,
                 bathymetry=None,
                 options=None):
        super(LandslideTerm, self).__init__(space)

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

    def get_bnd_functions(self, eta_in, uv_in, bnd_id, bnd_conditions):
        """
        Returns external values of elev and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.
        """
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
        return eta_ext, uv_ext      


class LandslideMomentumTerm(LandslideTerm):
    """
    Generic term in the landslide momentum equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, u_test, u_space, eta_space,
                 bathymetry=None,
                 options=None):
        super(LandslideMomentumTerm, self).__init__(u_space, bathymetry, options)

        self.options = options

        self.u_test = u_test
        self.u_space = u_space
        self.eta_space = eta_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class LandslideContinuityTerm(LandslideTerm):
    """
    Generic term in the depth-integrated landslide continuity equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry=None,
                 options=None):
        super(LandslideContinuityTerm, self).__init__(eta_space, bathymetry, options)

        self.eta_test = eta_test
        self.eta_space = eta_space
        self.u_space = u_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class FluidSlidePressureGradientTerm(LandslideMomentumTerm):
    r"""
    Slide Internal pressure gradient term, :math:`3/2 (\rho_2 - \rho_1)/\rho_2 g \nabla (eta_s)`
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):

        const = 3./2.*(self.options.rho_slide - self.options.rho_water)/self.options.rho_slide

        head = eta

        grad_eta_by_parts = self.eta_is_dg

        if grad_eta_by_parts:
            f = -const*g_grav*head*nabla_div(self.u_test)*self.dx
            if uv is not None:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            f += const*g_grav*head_star*jump(self.u_test, self.normal)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += const*g_grav*eta_rie*dot(self.u_test, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    head_rie = head + sqrt(total_h/g_grav)*un_jump
                    f += const*g_grav*head_rie*dot(self.u_test, self.normal)*ds_bnd
        else:
            f = const*g_grav*inner(grad(head), self.u_test) * self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += const*g_grav*(eta_rie-head)*dot(self.u_test, self.normal)*ds_bnd
        return -f


class FluidSlideHUDivTerm(LandslideContinuityTerm):
    r"""
    Divergence term, :math:`2/3 \nabla \cdot (D \bar{\textbf{u}})` 'H' is replaced by 'D', and '2/3' added
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):

        const = 2./3.

        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        if hu_by_parts:
            f = -const*inner(grad(self.eta_test), total_h*uv)*self.dx
            if self.eta_is_dg:
                h = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
                hu_star = h*uv_rie
                f += const*inner(jump(self.eta_test, self.normal), hu_star)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    total_h_ext = BaseFluidSlideEquation(self.function_space, self.bathymetry, self.options).update_depth_wd(eta_ext_old)
                    h_av = 0.5*(total_h + total_h_ext)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump # explore further? WPan. 

                    un_jump = inner(uv_old - uv_ext_old, self.normal)
                    eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                    h_rie = self.bathymetry + eta_rie
                    f += const*h_rie*un_rie*self.eta_test*ds_bnd
        else:
            f = const*div(total_h*uv)*self.eta_test*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is None or 'un' in funcs: # probably for 'None', on wall, dot(uv, self.normal) absolutely = 0, so 'None' can be cancelled, WPan; maybe 'uv' also should be added 
                    f += -const*total_h*dot(uv, self.normal)*self.eta_test*ds_bnd
        return -f


class FluidSlideHorizontalAdvectionTerm(LandslideMomentumTerm):
    r"""
    Advection of momentum term, :math:`4/5 \bar{\textbf{u}} \cdot \nabla\bar{\textbf{u}}`, '4/5' added
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):

        const = 4./5.

        if not self.options.use_nonlinear_equations:
            return 0

        horiz_advection_by_parts = True

        if horiz_advection_by_parts:
            f = -const*(uv[0]*div(self.u_test[0]*uv_old) +
                        uv[1]*div(self.u_test[1]*uv_old)) * self.dx
            if self.u_continuity in ['dg', 'hdiv']:
                un_av = dot(avg(uv_old), self.normal('-'))
                # NOTE solver can stagnate
                # s = 0.5*(sign(un_av) + 1.0)
                # NOTE smooth sign change between [-0.02, 0.02], slow
                # s = 0.5*tanh(100.0*un_av) + 0.5
                # uv_up = uv('-')*s + uv('+')*(1-s)
                # NOTE mean flux
                uv_up = avg(uv)
                f += const*(uv_up[0]*jump(self.u_test[0], inner(uv_old, self.normal)) +
                            uv_up[1]*jump(self.u_test[1], inner(uv_old, self.normal)))*self.dS
                # Lax-Friedrichs stabilization
                if self.options.use_lax_friedrichs_velocity:
                    uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += const*gamma*dot(jump(self.u_test), jump(uv))*self.dS
                    for bnd_marker in self.boundary_markers:
                        funcs = bnd_conditions.get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            n = self.normal
                            uv_ext = uv - 2*dot(uv, n)*n
                            gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                            f += const*gamma*dot(self.u_test, uv-uv_ext)*ds_bnd # for two vectors, dot() is equivalent to inner(), WPan
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    eta_jump = eta_old - eta_ext_old
                    un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                    uv_av = 0.5*(uv_ext + uv)
                    f += const*(uv_av[0]*self.u_test[0]*un_rie +
                                uv_av[1]*self.u_test[1]*un_rie)*ds_bnd
        return -f


class FluidSlideHorizontalViscosityTerm(LandslideMomentumTerm):
    r"""
    Viscosity of landslide momentum term, :math:'\frac{\nu \bar{\textbf{u}}}{3 D^2}'
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):

        nu = fields_old.get('slide_viscosity')
        if nu is None:
            return 0
        n = self.normal
        h = self.cellsize

        if self.options.use_grad_div_viscosity_term:
            stress = nu*2.*sym(grad(uv))
            stress_jump = avg(nu)*2.*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        f = inner(grad(self.u_test), stress)*self.dx

        if self.u_continuity in ['dg', 'hdiv']:
            # from Epshteyn et al. 2007 (http://dx.doi.org/10.1016/j.cam.2006.08.029)
            # the scheme is stable for alpha > 3*X*p*(p+1)*cot(theta), where X is the
            # maximum ratio of viscosity within a triangle, p the degree, and theta
            # with X=2, theta=6: cot(theta)~10, 3*X*cot(theta)~60
            p = self.u_space.ufl_element().degree()
            alpha = 5.*p*(p+1)
            if p == 0:
                alpha = 1.5
            f += (
                + alpha/avg(h)*inner(tensor_jump(self.u_test, n), stress_jump)*self.dS # in the math provided, sigma = alpha/avg(h), WPan
                - inner(avg(grad(self.u_test)), stress_jump)*self.dS
                - inner(tensor_jump(self.u_test, n), avg(stress))*self.dS
            )

            # Dirichlet bcs only for DG
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    if 'un' in funcs:
                        delta_uv = (dot(uv, n) - funcs['un'])*n
                    else:
                        eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                        if uv_ext is uv:
                            continue
                        delta_uv = uv - uv_ext

                    if self.options.include_grad_div_viscosity_term:
                        stress_jump = nu*2.*sym(outer(delta_uv, n))
                    else:
                        stress_jump = nu*outer(delta_uv, n)

                    f += (
                        alpha/h*inner(outer(self.u_test, n), stress_jump)*ds_bnd
                        - inner(grad(self.u_test), stress_jump)*ds_bnd
                        - inner(outer(self.u_test, n), stress)*ds_bnd
                    )

        if self.options.use_grad_depth_viscosity_term:
            f += -dot(self.u_test, dot(grad(total_h)/total_h, stress))*self.dx

        # use below for landslide modelling
        f = nu*inner(self.u_test, uv)/(3*total_h**2)*self.dx
        return -f


class FluidSlideQuadraticDragTerm(LandslideMomentumTerm):
    r"""
    Quadratic Manning bottom friction term
    :math:`C_D \| \bar{\textbf{u}} \| \bar{\textbf{u}}`

    where the drag term is computed with the Manning formula

    .. math::
        C_D = g \frac{\mu^2}{H^{1/3}}

    if the Manning coefficient :math:`\mu` is defined (see field :attr:`mu_manning`).
    Otherwise :math:`C_D` is taken as a constant (see field :attr:`quadratic_drag`).
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')
        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

        if C_D is not None:
            f += C_D * sqrt(dot(uv_old, uv_old)) * inner(self.u_test, uv) / total_h * self.dx # in the math provided, extra '/h' should be added, WPan
        return -f


class FluidSlideLinearDragTerm(LandslideMomentumTerm):
    r"""
    Linear friction term, :math:`C \bar{\textbf{u}}`

    Here :math:`C` is a user-defined drag coefficient.
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):
        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        sponge_damping = fields_old.get('sponge_damping')
        f = 0
        if linear_drag_coefficient is not None:
            bottom_fri = linear_drag_coefficient*inner(self.u_test, uv)*self.dx
            f += bottom_fri
        if sponge_damping is not None:
            sponge_drag = sponge_damping*inner(self.u_test, uv)*self.dx
            f += sponge_drag
        return -f


class FluidSlideMomentumSourceTerm(LandslideMomentumTerm):
    r"""
    Generic source term in the shallow water momentum equation

    The weak form reads

    .. math::
        F_s = \int_\Omega \boldsymbol{\tau} \cdot \boldsymbol{\psi} dx

    where :math:`\boldsymbol{\tau}` is a user defined vector valued :class:`Function`.

    .. note ::
        Due to the sign convention of :class:`.equation.Term`, this term is assembled to the left hand side of the equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):
        f = 0
        momentum_source = fields_old.get('momentum_source')

        if momentum_source is not None:
            f += inner(momentum_source, self.u_test)*self.dx
        return f


class FluidSlideContinuitySourceTerm(LandslideContinuityTerm):
    r"""
    Generic source term in the depth-averaged continuity equation

    The weak form reads

    .. math::
        F_s = \int_\Omega S \phi dx

    where :math:`S` is a user defined scalar :class:`Function`.

    .. note ::
        Due to the sign convention of :class:`.equation.Term`, this term is assembled to the left hand side of the equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):
        f = 0
        volume_source = fields_old.get('volume_source')

        if volume_source is not None:
            f += inner(volume_source, self.eta_test)*self.dx
        return f


class FluidSlideExternalPressureTerm(LandslideMomentumTerm):
    """
    External pressure term reflecting effects of the surface elevation computed from coupled NSWEs,

    .. math::
        \frac{3}{2} \frac{\rho_1}{\rho_2} g \nabla \eta
    """
    def residual(self, uv_water, eta_water, uv_water_old, eta_slide, fields, bnd_conditions=None):

        total_h = eta_water - eta_slide
        head = eta_water
        uv = uv_water
        uv_old = uv_water_old
        const = 3./2.*self.options.rho_water/self.options.rho_slide
        grad_eta_by_parts = True

        if grad_eta_by_parts:
            f = -const*g_grav*head*nabla_div(self.u_test)*self.dx
            if uv is not None:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            f += const*g_grav*head_star*jump(self.u_test, self.normal)*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += const*g_grav*eta_rie*dot(self.u_test, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    head_rie = head + sqrt(total_h/g_grav)*un_jump
                    f += const*g_grav*head_rie*dot(self.u_test, self.normal)*ds_bnd
        else:
            f = g_grav*inner(grad(head), self.u_test) * self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*(eta_rie-head)*dot(self.u_test, self.normal)*ds_bnd
        return -f


class FluidSlideMomDivTerm(LandslideMomentumTerm):
    r"""
    Divergence term, :math:`2/15 \frac{\bar{\textbf{u}}}{D} \nabla \cdot (D \bar{\textbf{u}})` 'H' is replaced by 'D'
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions=None):

        const = 2./15.

        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        if hu_by_parts:
            f = -const*inner(grad(inner(uv/total_h, self.u_test)), total_h*uv)*self.dx
            h = avg(total_h)
            uv_rie = avg(uv_old) + sqrt(g_grav/h)*jump(eta, self.normal)
            hu_star = h*uv_rie
            f += const*jump(dot(uv, uv), inner(self.u_test, self.normal))*self.dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    total_h_ext = BaseFluidSlideEquation(self.function_space, self.bathymetry, self.options).update_depth_wd(eta_ext_old)
                    h_av = 0.5*(total_h + total_h_ext)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                    f += const*dot(un_rie, un_rie)*inner(self.u_test, self.normal)*ds_bnd
        else:
            f = const*inner(uv/total_h, self.u_test)*div(total_h*uv)*self.dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is None or 'un' in funcs: # probably for 'None', on wall, dot(uv, self.normal) absolutely = 0, so 'None' can be cancelled, WPan; maybe 'uv' also should be added 
                    f += -const*inner(uv, self.u_test)*dot(uv, self.normal)*ds_bnd
        return -f


class BaseFluidSlideEquation(Equation):
    """
    Abstract base class for FluidSlidemotionEquations, FluidSlidemotionMomentumEquation.

    Provides common functionality to compute time steps and add either momentum
    or continuity terms.
    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        super(BaseFluidSlideEquation, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

    def add_momentum_terms(self, *args):
        self.add_term(FluidSlidePressureGradientTerm(*args), 'implicit')
        self.add_term(FluidSlideHorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(FluidSlideHorizontalViscosityTerm(*args), 'explicit')
        self.add_term(FluidSlideQuadraticDragTerm(*args), 'explicit')
        self.add_term(FluidSlideLinearDragTerm(*args), 'explicit')
        self.add_term(FluidSlideMomentumSourceTerm(*args), 'source')
        self.add_term(FluidSlideMomDivTerm(*args), 'implicit')

    def add_continuity_terms(self, *args):
        self.add_term(FluidSlideHUDivTerm(*args), 'implicit')
        self.add_term(FluidSlideContinuitySourceTerm(*args), 'source')

    def update_depth_wd(self, eta):
        """
        Returns new total water column depth because of wetting-drying judges
        """
        if self.options.use_nonlinear_equations:  
            total_h = self.bathymetry + eta + self.water_height_displacement(eta)
            if self.options.thin_film:    
                total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry
        return total_h

    def water_height_displacement(self, eta):
        """
        Returns wetting and drying water height discplacement as described in:
        Karna et al.,  2011.
        """ 
        H = self.bathymetry + eta
        return 2 * self.options.depth_wd_interface**2 / (2 * self.options.depth_wd_interface + abs(H)) + 0.5 * (abs(H) - H) # new formulated function, WPan
        #return conditional(H < self.options.depth_wd_interface, self.options.depth_wd_interface**2 / (2 * self.options.depth_wd_interface - H) - H, 0.) # artificial porosity method
        #return 0.5 * (sqrt(H**2 + self.options.depth_wd_interface**2) - H) # original bathymetry changed method

    def WaterDisplacementMassTerm(self, solution):
        """
        Elevation mass displacement term, :math:`\partial \eta / \partial t + \partial \tilde{h} / \partial t`

        The weak form reads

        .. math::
            \int_\Omega ( \partial \eta / \partial t + \partial \tilde{h} / \partial t ) \phi dx
             = \int_\Omega (\partial \tilde{H} / \partial t) \phi dx
        """
        f = 0
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        u_test, eta_test = TestFunctions(self.function_space)
        f += inner(self.water_height_displacement(solution[2]), eta_test)*self.dx # solution[2] == eta
        return f

    def residual_uv_eta(self, label, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions):
        f = 0
        for term in self.select_terms(label):
            f += term.residual(uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions)
        return f


class FluidSlideEquations(BaseFluidSlideEquation):
    """
    2D depth-averaged fluid slide motion equations in non-conservative form.

    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        """
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(FluidSlideEquations, self).__init__(function_space, bathymetry, options)

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry, options) 

        self.add_continuity_terms(eta_test, eta_space, u_space, bathymetry, options)

        self.externalpressure_residual_base = FluidSlideExternalPressureTerm(u_test, u_space, eta_space, bathymetry, options)

    def mass_term(self, solution):
        """Solution just consists of eta, so other terms, i.e. bathymetry depth, artificial porosity, need to be added here"""
        f = super(FluidSlideEquations, self).mass_term(solution)
        f += self.WaterDisplacementMassTerm(solution)
        return f

    def add_external_surface_term(self, uv_water, eta_water, uv_water_old, eta_old, fields, bnd_conditions):
        return self.externalpressure_residual_base.residual(uv_water, eta_water, uv_water_old, eta_old, fields, bnd_conditions)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        if isinstance(solution, list):
            uv, eta = solution
        else:
            uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        total_h = self.update_depth_wd(eta_old)
        #uv, eta = solution.split()
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, total_h, bnd_conditions)

