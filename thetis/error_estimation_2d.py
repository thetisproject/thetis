from __future__ import absolute_import
from .utility import *
from .equation import GOErrorEstimatorTerm, GOErrorEstimator
from .tracer_eq_2d import TracerTerm
from .shallowwater_eq import ShallowWaterTerm


__all__ = [
    'TracerGOErrorEstimator',
    'ShallowWaterGOErrorEstimator',
]

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterGOErrorEstimatorTerm(GOErrorEstimatorTerm, ShallowWaterTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the shallow
    water model.
    """
    def __init__(self, function_space, bathymetry=None, options=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        :kwarg options: :class:`ModelOptions2d` parameter object
        """
        ShallowWaterTerm.__init__(self, function_space, bathymetry, options)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())


class ShallowWaterGOErrorEstimatorMomentumTerm(ShallowWaterGOErrorEstimatorTerm):
    """
    Generic :class:`ShallowWaterGOErrorEstimatorTerm` term that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, u_space, eta_space, bathymetry=None, options=None):
        super(ShallowWaterGOErrorEstimatorMomentumTerm, self).__init__(u_space, bathymetry, options)

        self.options = options

        self.u_space = u_space
        self.eta_space = eta_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class ShallowWaterGOErrorEstimatorContinuityTerm(ShallowWaterGOErrorEstimatorTerm):
    """
    Generic :class:`ShallowWaterGOErrorEstimatorTerm` term that provides commonly used members and
    mapping for boundary functions.
    """
    def __init__(self, eta_space, u_space, bathymetry=None, options=None):
        super(ShallowWaterGOErrorEstimatorContinuityTerm, self).__init__(eta_space, bathymetry, options)

        self.eta_space = eta_space
        self.u_space = u_space

        self.u_continuity = element_continuity(self.u_space.ufl_element()).horizontal
        self.eta_is_dg = element_continuity(self.eta_space.ufl_element()).horizontal == 'dg'


class TracerGOErrorEstimatorTerm(GOErrorEstimatorTerm, TracerTerm):
    """
    Generic :class:`GOErrorEstimatorTerm` term in a goal-oriented error estimator for the 2D tracer
    model.
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        """
        TracerTerm.__init__(self, function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        GOErrorEstimatorTerm.__init__(self, function_space.mesh())


class ExternalPressureGradientGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
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

            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*eta_rie*dot(z, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    head_rie = head + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*head_rie*dot(z, self.normal)*ds_bnd
        else:
            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    flux_terms += -self.p0test*g_grav*(eta_rie-head)*dot(z, self.normal)*ds_bnd

        return flux_terms


class HUDivGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`HUDivTerm` term of
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
                flux_terms += inner(loc('+') + loc('-'), hu_star)*self.dS

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
            flux_terms += self.p0test*zeta*dot(total_h*uv, self.normal)*ds

            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    total_h_ext = self.get_total_depth(eta_ext_old)
                    h_av = 0.5*(total_h + total_h_ext)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                    un_jump = inner(uv_old - uv_ext_old, self.normal)
                    eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                    h_rie = self.bathymetry + eta_rie
                    flux_terms += -self.p0test*h_rie*un_rie*zeta*ds_bnd
        else:
            # Terms arising from boundary conditions
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                if funcs is None or 'un' in funcs:
                    flux_terms += self.p0test*total_h*dot(uv, self.normal)*zeta*ds_bnd

        return flux_terms


class HorizontalAdvectionGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
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
        flux_terms = self.p0test*inner(dot(outer(uv, z), uv), self.normal)*ds

        # Terms arising from boundary conditions
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            un_av = dot(avg(uv_old), self.normal('-'))
            if funcs is not None:
                eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                eta_jump = eta_old - eta_ext_old
                total_h = self.get_total_depth(eta_old)
                un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                uv_av = 0.5*(uv_ext + uv)
                flux_terms += -self.p0test*(uv_av[0]*z[0]*un_rie + uv_av[1]*z[1]*un_rie)*ds_bnd

            if self.options.use_lax_friedrichs_velocity:
                uv_lax_friedrichs = fields_old.get('lax_friedrichs_velocity_scaling_factor')
                gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                if funcs is None:
                    # impose impermeability with mirror velocity
                    n = self.normal
                    uv_ext = uv - 2*dot(uv, n)*n
                    gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                    flux_terms += -self.p0test*gamma*dot(z, uv-uv_ext)*ds_bnd

        return flux_terms


class HorizontalViscosityGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
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
            flux_terms += -self.p0test*inner(dot(z, stress), n)*ds

            # Terms arising from boundary conditions
            alpha = self.options.sipg_parameter
            assert alpha is not None
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

                    if self.options.use_grad_div_viscosity_term:
                        stress_jump = 2.0*nu*sym(outer(delta_uv, n))
                    else:
                        stress_jump = nu*outer(delta_uv, n)

                    flux_terms += -self.p0test*alpha/h*inner(outer(z, n), stress_jump)*ds_bnd
                    flux_terms += self.p0test*inner(grad(z), stress_jump)*ds_bnd
                    flux_terms += self.p0test*inner(outer(z, n), stress)*ds_bnd

        return flux_terms


class CoriolisGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`CoriolisTerm` term
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


class WindStressGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`WindStressTerm` term
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


class AtmosphericPressureGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`AtmosphericPressureTerm` term
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


class QuadraticDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`QuadraticDragTerm`
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


class LinearDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`LinearDragTerm`
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


class BottomDrag3DGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`BottomDrag3DTerm`
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


class TurbineDragGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`TurbineDragTerm`
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


class MomentumSourceGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorMomentumTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`MomentumSourceTerm`
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


class ContinuitySourceGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the :class:`ContinuitySourceTerm`
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


class BathymetryDisplacementGOErrorEstimatorTerm(ShallowWaterGOErrorEstimatorContinuityTerm):
    """
    :class:`ShallowWaterGOErrorEstimatorTerm` object associated with the
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


class TracerHorizontalAdvectionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalAdvectionTerm`
    term of the 2D tracer model.
    """
    def element_residual(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor*fields_old['uv_2d']

        return -self.p0test*arg*inner(uv, grad(solution))*self.dx

    def inter_element_flux(self, solution, solution_old, arg, arg_old, fields, fields_old):
        if fields_old.get('uv_2d') is None:
            return 0
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor*fields_old['uv_2d']
        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        flux_terms = 0
        if self.horizontal_dg:

            # Interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            loc = self.p0test*arg
            flux_terms += -c_up*(loc('+') + loc('-'))*jump(uv, self.normal)*self.dS

            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if uv_p1 is not None:
                    gamma = 0.5*abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                arg_jump = self.p0test*arg('+') - self.p0test*arg('-')
                flux_terms += -gamma*dot(arg_jump, jump(solution))*self.dS
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        self.corr_factor = fields_old.get('tracer_advective_velocity_factor')

        uv = self.corr_factor * fields_old['uv_2d']

        flux_terms = 0
        if self.horizontal_dg:
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    c_in = solution
                    if funcs is not None and 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        flux_terms += -c_up*(uv_av[0]*self.normal[0]
                                             + uv_av[1]*self.normal[1])*arg*ds_bnd
                    else:
                        flux_terms += -c_in*(uv[0]*self.normal[0]
                                             + uv[1]*self.normal[1])*arg*ds_bnd

        return flux_terms


class TracerHorizontalDiffusionGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`HorizontalDiffusionTerm`
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
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0
        if self.horizontal_dg:
            alpha = self.sipg_parameter
            assert alpha is not None
            sigma = avg(alpha/self.cellsize)
            ds_interior = self.dS
            arg_n = self.p0test*arg*self.normal('+') + self.p0test*arg*self.normal('-')
            flux_terms += -sigma*inner(arg_n,
                                       dot(avg(diff_tensor),
                                           jump(solution, self.normal)))*ds_interior
            flux_terms += inner(arg_n, avg(dot(diff_tensor, grad(solution))))*ds_interior
            arg_av = self.p0test*0.5*arg
            flux_terms += inner(dot(avg(diff_tensor), grad(arg_av)),
                                jump(solution, self.normal))*ds_interior
        return flux_terms

    def boundary_flux(self, solution, solution_old, arg, arg_old, fields, fields_old, bnd_conditions):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])

        flux_terms = 0

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution
                elev = fields_old['elev_2d']
                self.corr_factor = fields_old.get('tracer_advective_velocity_factor')
                uv = self.corr_factor * fields_old['uv_2d']
                if funcs is not None:
                    if 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        diff_flux_up = dot(diff_tensor, grad(c_up))
                        flux_terms += self.p0test*arg*dot(diff_flux_up, self.normal)*ds_bnd
                    elif 'diff_flux' in funcs:
                        f += self.p0test*arg*funcs['diff_flux']*ds_bnd
                    else:
                        # Open boundary case
                        f += self.p0test*arg*dot(diff_flux, self.normal)*ds_bnd


class TracerSourceGOErrorEstimatorTerm(TracerGOErrorEstimatorTerm):
    """
    :class:`TracerGOErrorEstimatorTerm` object associated with the :class:`SourceTerm` term of the
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


class ShallowWaterGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the shallow water model.
    """
    def __init__(self, function_space, bathymetry, options):
        super(ShallowWaterGOErrorEstimator, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.options = options

        u_space, eta_space = function_space.split()
        self.add_momentum_terms(u_space, eta_space, bathymetry, options)
        self.add_continuity_terms(u_space, eta_space, bathymetry, options)

    def add_momentum_terms(self, *args):
        self.add_term(ExternalPressureGradientGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(CoriolisGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(WindStressGOErrorEstimatorTerm(*args), 'source')
        self.add_term(AtmosphericPressureGOErrorEstimatorTerm(*args), 'source')
        self.add_term(QuadraticDragGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(LinearDragGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(BottomDrag3DGOErrorEstimatorTerm(*args), 'source')
        self.add_term(TurbineDragGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(MomentumSourceGOErrorEstimatorTerm(*args), 'source')

    def add_continuity_terms(self, *args):
        self.add_term(HUDivGOErrorEstimatorTerm(*args), 'implicit')
        self.add_term(ContinuitySourceGOErrorEstimatorTerm(*args), 'source')


class TracerGOErrorEstimator(GOErrorEstimator):
    """
    :class:`GOErrorEstimator` for the 2D tracer model.
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        super(TracerGOErrorEstimator, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(TracerHorizontalAdvectionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerHorizontalDiffusionGOErrorEstimatorTerm(*args), 'explicit')
        self.add_term(TracerSourceGOErrorEstimatorTerm(*args), 'source')
