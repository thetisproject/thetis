from __future__ import absolute_import
from .utility import *
from .equation import ErrorEstimatorTerm, ErrorEstimator
from .tracer_eq_2d import TracerTerm
from .shallowwater_eq import ShallowWaterTerm

__all__ = [
    'TracerErrorEstimator',
    'ShallowWaterErrorEstimator',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterErrorEstimatorTerm(ErrorEstimatorTerm, ShallowWaterTerm):
    """
    Generic :class:`ErrorEstimatorTerm` in the shallow water model which provides the contribution
    to the total error estimator.
    """
    def __init__(self, function_space, bathymetry=None, options=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        :kwarg options: :class:`ModelOptions2d` parameter object
        """
        ShallowWaterTerm.__init__(self, function_space, bathymetry, options)
        ErrorEstimatorTerm.__init__(self, function_space.mesh())

        # self.eta_is_dg = element_continuity(function_space.sub(1).ufl_element()).horizontal == 'dg'
        # self.u_continuity = element_continuity(function_space.sub(0).ufl_element()).horizontal
        # TODO: TEMPORARY. Should instead split into momentum and continuity terms
        self.eta_is_dg = options.element_family in ('dg-dg', 'rt-dg')
        self.u_continuity = 'hdiv' if options.element_family == 'rt-dg' else 'dg'


class TracerErrorEstimatorTerm(ErrorEstimatorTerm, TracerTerm):
    """
    Generic :class:`ErrorEstimatorTerm` in the 2D tracer model which provides the contribution
    to the total error estimator.
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        """
        TracerTerm.__init__(self, function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        ErrorEstimatorTerm.__init__(self, function_space.mesh())


class ExternalPressureGradientErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the
    :class:`ExternalPressureGradientTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)

        return -self.p0test*g_grav*inner(z, grad(eta))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg

        flux_terms = 0
        if grad_eta_by_parts:

            # Terms arising from DG discretisation
            if uv is not None:
                head_star = avg(head) + sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            loc = -self.p0test*g_grav*dot(z, self.normal)
            flux_terms += head_star*(loc('+') + loc('-'))*self.dS

            # Term arising from integration by parts
            loc = self.p0test*g_grav*eta*dot(z, self.normal)
            flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.get_total_depth(eta_old)
        head = eta
        grad_eta_by_parts = self.eta_is_dg

        flux_terms = 0
        if grad_eta_by_parts:

            # Term arising from integration by parts
            loc = self.p0test*g_grav*eta*dot(z, self.normal)
            flux_terms += loc*ds

            # TODO: Terms arising from boundary conditions
        raise NotImplementedError


class HUDivErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`HUDivTerm` term of
    the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)

        return -self.p0test*zeta*div(total_h*uv)*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.get_total_depth(eta_old)
        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        flux_terms = 0
        if hu_by_parts:

            # Terms arising from DG discretisation
            if self.eta_is_dg:
                h = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
                hu_star = h*uv_rie
                loc = -self.p0test*zeta*self.normal
                f += inner(loc('+') + loc('-'), hu_star)*self.dS

            # Term arising from integration by parts
            loc = self.p0test*zeta*dot(total_h*uv, self.normal)
            flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.get_total_depth(eta_old)
        hu_by_parts = self.u_continuity in ['dg', 'hdiv']

        flux_terms = 0
        if hu_by_parts:

            # Term arising from integration by parts
            loc = self.p0test*zeta*dot(total_h*uv, self.normal)
            flux_terms += loc*self.ds

            # TODO: Terms arising from boundary conditions
        raise NotImplementedError  # TODO


class HorizontalAdvectionErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the
    :class:`HorizontalAdvectionTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if not self.options.use_nonlinear_equations:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        return -self.p0test*inner(z, dot(uv_old, nabla_grad(uv)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if not self.options.use_nonlinear_equations:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        if self.u_continuity in ['dg', 'hdiv']:

            # Terms arising from DG discretisation
            un_av = dot(avg(uv_old), self.normal('-'))
            uv_up = avg(uv)
            loc = -self.p0test*z[0]
            flux_terms = jump(uv[0], self.normal[0])*dot(uv_up[0], loc('+') + loc('-'))*self.dS
            flux_terms += jump(uv[1], self.normal[1])*dot(uv_up[0], loc('+') + loc('-'))*self.dS
            loc = -self.p0test*z[1]
            flux_terms += jump(uv[0], self.normal[0])*dot(uv_up[1], loc('+') + loc('-'))*self.dS
            flux_terms += jump(uv[1], self.normal[1])*dot(uv_up[1], loc('+') + loc('-'))*self.dS
            if self.options.use_lax_friedrichs_velocity:
                uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                local_jump = -self.p0test('+')*z('+') + self.p0test('-')*z('-')
                flux_terms += gamma*dot(local_jump, jump(uv))*dS

        # Term arising from integration by parts
        loc = self.p0test*inner(dot(outer(uv, z), uv), self.normal)
        flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if not self.options.use_nonlinear_equations:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        # Term arising from integration by parts
        flux_terms += self.p0test*inner(dot(outer(uv, z), vu), self.normal)*self.ds

        # TODO: Terms arising from boundary conditions
        raise NotImplementedError


class HorizontalViscosityErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the
    :class:`HorizontalViscosityTerm` term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
        else:
            stress = nu*grad(uv)

        f = self.p0test*inner(z, div(stress))*self.dx
        if self.options.use_grad_depth_viscosity_term:
            f += self.p0test*inner(z, dot(grad(total_h)/total_h))*self.dx

        return f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)

        h = self.cellsize
        n = self.normal

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
            stress_jump = 2.0*avg(nu)*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        flux_terms = 0
        if self.u_continuity in ['dg', 'hdiv']:

            # Terms arising from DG discretisation
            alpha = self.options.sipg_parameter
            assert alpha is not None
            loc = self.p0test*outer(z, n)
            flux_terms += -avg(alpha/h)*inner(loc('+') + loc('-'), stress_jump)*self.dS
            flux_terms += inner(loc('+') + loc('-'), avg(stress))*self.dS
            loc = self.p0test*grad(z)
            flux_terms += 0.5*inner(loc('+') + loc('-'), stress_jump)*self.dS

            # Term arising from integration by parts
            loc = -self.p0test*inner(dot(z, stress), n)
            flux_terms += (loc('+') + loc('-'))*self.dS

        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)

        h = self.cellsize
        n = self.normal

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
            stress_jump = 2.0*avg(nu)*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        flux_terms = 0
        if self.u_continuity in ['dg', 'hdiv']:

            # Term arising from integration by parts
            loc = -self.p0test*inner(dot(z, stress), n)
            flux_terms += loc*ds

            # TODO: Terms arising from boundary conditions
        raise NotImplementedError


class CoriolisErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`CoriolisTerm` term
    of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)
        coriolis = fields_old.get('coriolis')

        f = 0
        if coriolis is not None:
            f += self.p0test*coriolis*(-uv[1]*z[0] + uv[0]*z[1])*self.dx

        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class WindStressErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`WindStressTerm` term
    of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        wind_stress = fields_old.get('wind_stress')
        total_h = self.get_total_depth(eta_old)
        f = 0
        if wind_stress is not None:
            f += self.p0test*dot(wind_stress, z)/total_h/rho_0*self.dx
        return f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class AtmosphericPressureErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`AtmosphericPressureTerm` term
    of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        atmospheric_pressure = fields_old.get('atmospheric_pressure')
        f = 0
        if atmospheric_pressure is not None:
            f += self.p0test*dot(grad(atmospheric_pressure), z)/rho_0*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class QuadraticDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`QuadraticDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')

        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

        if C_D is not None:
            unorm = sqrt(dot(uv_old, uv_old) + self.options.norm_smoother**2)
            f += self.p0test*C_D*unorm*inner(z, uv)/total_h*self.dx

        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class LinearDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`LinearDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        z, zeta = split(arg)

        linear_drag_coefficient = fields_old.get('linear_drag_coefficient')
        f = 0
        if linear_drag_coefficient is not None:
            f += self.p0test*linear_drag_coefficient*inner(z, uv)*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class BottomDrag3DErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`BottomDrag3DTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)

        total_h = self.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        f = 0
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            f += self.p0test*dot(stress, z)*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class TurbineDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`TurbineDragTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(eta_old)

        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            density = farm_options.turbine_density
            C_T = farm_options.turbine_options.thrust_coefficient
            A_T = pi * (farm_options.turbine_options.diameter/2.)**2
            C_D = (C_T * A_T * density)/2.
            unorm = sqrt(dot(uv_old, uv_old))
            f += self.p0test*C_D*unorm*inner(z, uv)/total_h*self.dx(subdomain_id)

        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class MomentumSourceErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`MomentumSourceTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        z, zeta = split(arg)

        f = 0
        momentum_source = fields_old.get('momentum_source')

        if momentum_source is not None:
            f += self.p0test*inner(momentum_source, z)*self.dx
        return f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class ContinuitySourceErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the :class:`ContinuitySourceTerm`
    term of the shallow water model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        z, zeta = split(arg)

        f = 0
        volume_source = fields_old.get('volume_source')

        if volume_source is not None:
            f += self.p0test*inner(volume_source, zeta)*self.dx
        return f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class BathymetryDisplacementErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    """
    :class:`ShallowWaterErrorEstimatorTerm` object associated with the
    :class:`BathymetryDisplacementTerm` term of the shallow water model.
    """
    def element_residual(self, solution, arg):
        uv, eta = split(solution)
        z, zeta = split(arg)

        f = 0
        if self.options.use_wetting_and_drying:
            f += self.p0test*inner(self.wd_bathymetry_displacement(eta), zeta)*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class TracerHorizontalAdvectionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    """
    :class:`TracerErrorEstimatorTerm` object associated with the :class:`HorizontalAdvectionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']

        return -self.p0test*arg*inner(uv, grad(solution))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        raise NotImplementedError  # TODO

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # TODO


class TracerHorizontalDiffusionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    """
    :class:`TracerErrorEstimatorTerm` object associated with the :class:`HorizontalDiffusionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        return self.p0test*arg*div(dot(diff_tensor, grad(solution)))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        raise NotImplementedError  # TODO

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        raise NotImplementedError  # TODO


class TracerSourceErrorEstimatorTerm(TracerErrorEstimatorTerm):
    """
    :class:`TracerErrorEstimatorTerm` object associated with the :class:`SourceTerm` term of the
    2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += -self.p0test*inner(source, self.test)*self.dx
        return -f

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        return 0

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        return 0


class ShallowWaterErrorEstimator(ErrorEstimator):
    """
    :class:`ErrorEstimator` for the shallow water model.
    """
    def __init__(self, function_space, bathymetry, options):
        super(ShallowWaterErrorEstimator, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

        # Momentum terms
        args = (function_space.sub(0), bathymetry, options)
        self.add_term(ExternalPressureGradientErrorEstimatorTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityErrorEstimatorTerm(*args), 'explicit')
        self.add_term(CoriolisErrorEstimatorTerm(*args), 'explicit')
        self.add_term(WindStressErrorEstimatorTerm(*args), 'source')
        self.add_term(AtmosphericPressureErrorEstimatorTerm(*args), 'source')
        self.add_term(QuadraticDragErrorEstimatorTerm(*args), 'explicit')
        self.add_term(LinearDragErrorEstimatorTerm(*args), 'explicit')
        self.add_term(BottomDrag3DErrorEstimatorTerm(*args), 'source')
        self.add_term(TurbineDragErrorEstimatorTerm(*args), 'implicit')
        self.add_term(MomentumSourceErrorEstimatorTerm(*args), 'source')

        # Continuity terms
        args = (function_space.sub(1), bathymetry, options)
        self.add_term(HUDivErrorEstimatorTerm(*args), 'implicit')
        self.add_term(ContinuitySourceErrorEstimatorTerm(*args), 'source')


class TracerErrorEstimator(ErrorEstimator):
    """
    :class:`ErrorEstimator` for the 2D tracer model.
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerErrorEstimator, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(TracerHorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceErrorEstimatorTerm(*args), 'source')
