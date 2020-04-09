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


class ShallowWaterErrorEstimatorTerm(ErrorEstimatorTerm, ShallowWaterTerm):
    # TODO: doc
    def __init__(self, function_space, bathymetry=None, options=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        :kwarg options: :class:`ModelOptions2d` parameter object
        """
        ShallowWaterTerm.__init__(self, function_space, bathymetry, options)
        ErrorEstimatorTerm.__init__(self, function_space.mesh())


class TracerErrorEstimatorTerm(ErrorEstimatorTerm, TracerTerm):
    # TODO: doc
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
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        z, zeta = split(arg)

        return -self.p0test*g_grav*inner(z, grad(elev))*self.dx

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        raise NotImplementedError  # TODO


class HUDivErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        uv_old, elev_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(elev_old)

        return -self.p0test*zeta*div(total_h*uv)*self.dx

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalAdvectionErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        if not self.options.use_nonlinear_equations:
            return 0
        uv, elev = split(solution)
        z, zeta = split(arg)

        return -self.p0test*inner(z, dot(uv, nabla_grad(uv)))*self.dx  # TODO: Maybe should use uv_old for one

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class HorizontalViscosityErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0
        uv, elev = split(solution)
        uv_old, elev_old = split(solution_old)
        total_h = self.get_total_depth(elev_old)

        if self.options.use_grad_div_viscosity_term:
            stress = 2.0*nu*sym(grad(uv))
        else:
            stress = nu*grad(uv)

        f = self.p0test*inner(z, div(stress))*self.dx
        if self.options.use_grad_depth_viscosity_term:
            f += self.p0test*inner(z, dot(grad(total_h)/total_h))*self.dx

        return f

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class CoriolisErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        z, zeta = split(arg)
        coriolis = fields_old.get('coriolis')

        f = 0
        if coriolis is not None:
            f += self.p0test*coriolis*(-uv[1]*z[0] + uv[0]*z[1])*self.dx

        return -f

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        return 0


class QuadraticDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        uv_old, elev_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(elev_old)
        manning_drag_coefficient = fields_old.get('manning_drag_coefficient')
        C_D = fields_old.get('quadratic_drag_coefficient')

        f = 0
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / total_h**(1./3.)

        if C_D is not None:
            # unorm = sqrt(dot(uv_old, uv_old) + self.options.norm_smoother**2)
            unorm = sqrt(dot(uv, uv) + self.options.norm_smoother**2)
            f += self.p0test*C_D*unorm*inner(z, uv)/total_h*self.dx

        return -f

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        return 0



class TurbineDragErrorEstimatorTerm(ShallowWaterErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        uv, elev = split(solution)
        uv_old, elev_old = split(solution_old)
        z, zeta = split(arg)
        total_h = self.get_total_depth(elev_old)

        f = 0
        for subdomain_id, farm_options in self.options.tidal_turbine_farms.items():
            density = farm_options.turbine_density
            C_T = farm_options.turbine_options.thrust_coefficient
            A_T = pi * (farm_options.turbine_options.diameter/2.)**2
            C_D = (C_T * A_T * density)/2.
            # unorm = sqrt(dot(uv_old, uv_old))
            unorm = sqrt(dot(uv, uv))
            f += C_D * unorm * inner(z, uv) / total_h * dx(subdomain_id)

        return -f

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        return 0


class TracerHorizontalAdvectionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']

        return -self.p0test*arg*inner(uv, grad(solution))*self.dx

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerHorizontalDiffusionErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        return self.p0test*arg*div(dot(diff_tensor, grad(solution)))*self.dx

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        raise NotImplementedError  # TODO


class TracerSourceErrorEstimatorTerm(TracerErrorEstimatorTerm):
    # TODO: doc
    def residual(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source')
        if source is not None:
            f += -self.p0test*inner(source, self.test)*self.dx
        return -f

    def flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions=None):
        return 0


class ShallowWaterErrorEstimator(ErrorEstimator):
    # TODO: doc
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
        # self.add_term(WindStressErrorEstimatorTerm(*args), 'source')  # TODO
        # self.add_term(AtmosphericPressureErrorEstimatorTerm(*args), 'source')  # TODO
        self.add_term(QuadraticDragErrorEstimatorTerm(*args), 'explicit')
        # self.add_term(LinearDragErrorEstimatorTerm(*args), 'explicit')  # TODO
        # self.add_term(BottomDrag3DErrorEstimatorTerm(*args), 'source')  # TODO
        self.add_term(TurbineDragErrorEstimatorTerm(*args), 'implicit')
        # self.add_term(MomentumSourceErrorEstimatorTerm(*args), 'source')  # TODO

        # Continuity terms
        args = (function_space.sub(1), bathymetry, options)
        self.add_term(HUDivErrorEstimatorTerm(*args), 'implicit')
        # self.add_term(ContinuitySourceErrorEstimatorTerm(*args), 'source')  # TODO


class TracerErrorEstimator(ErrorEstimator):
    # TODO: doc
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerErrorEstimator, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(TracerHorizontalAdvectionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceErrorEstimatorTerm(*args), 'source')
