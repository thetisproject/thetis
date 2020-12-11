r"""
Depth averaged granular flow equations in conservative form
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

g_grav = physical_constants['g_grav']


class BaseGranularEquation(Equation):
    """
    Abstract base class for `GranularEquations`.
    """
    def __init__(self, space, bathymetry, options):
        super(BaseGranularEquation, self).__init__(space)

        self.bathymetry = bathymetry
        self.options = options

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        self.dS = dS(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())


class GranularEquations(BaseGranularEquation):
    """
    2D depth-averaged granular flow equations in conservative form.

    The equations become the full 2D SWE equations if lamda = 1.
    """
    def __init__(self, space, bathymetry, options):
        super(GranularEquations, self).__init__(space, bathymetry, options)

        self.options_nh = options.nh_model_options
        self.test_h = self.test[0]
        self.test_uv = as_vector((self.test[1], self.test[2]))
        self.boundary_markers = sorted(space.mesh().exterior_facets.unique_markers)
        # components of gravity in a slope-oriented coordinate system
        self.grav_x = g_grav*self.options_nh.bed_slope[0]
        self.grav_y = g_grav*self.options_nh.bed_slope[1]
        self.grav_z = g_grav*self.options_nh.bed_slope[2]

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions=None):

        h, hu, hv = split(solution)
        h_old, hu_old, hv_old = split(solution_old)

        def mom(hu, h):
            return conditional(h <= 0, zero(hu.ufl_shape), hu)

        def vel(hu, h):
            return conditional(h <= 0, zero(hu.ufl_shape), hu / h)

        # momentum
        mom_uv = as_vector((mom(hu, h), mom(hv, h)))
        mom_uv_old = as_vector((mom(hu_old, h_old), mom(hv_old, h_old)))
        # velocity
        vel_uv = as_vector((vel(hu, h), vel(hv, h)))
        vel_uv_old = as_vector((vel(hu_old, h_old), vel(hv_old, h_old)))

        if self.options_nh.flow_is_granular:
            lamda = self.options_nh.lamda
            phi_i = self.options_nh.phi_i
            phi_b = self.options_nh.phi_b
            # uv_div = fields_old.get('uv_div')
            uv_div = div(vel_uv)
            # s_xy = fields_old.get('strain_rate')
            s_xy = 0.5*(Dx(vel_uv[0], 1) + Dx(vel_uv[1], 0))
            if phi_i <= phi_b:
                kap = (1 + sin(phi_i)**2) / (1 - sin(phi_i)**2)
            else:
                kap_div = 2*(1 - sqrt(1 - cos(phi_i)**2*(1 + tan(phi_b)**2)))/(cos(phi_i)**2) - 1
                kap_conv = 2*(1 + sqrt(1 - cos(phi_i)**2*(1 + tan(phi_b)**2)))/(cos(phi_i)**2) - 1
                kap_mid = (1 + sin(phi_i)**2) / (1 - sin(phi_i)**2)
                kap = conditional(uv_div > 1e-9, kap_div, conditional(uv_div < -1e-9, kap_conv, kap_mid))
            self.lam_kap = (1 - lamda)*kap + lamda
        else:
            kap = 1.0
            self.lam_kap = 1.0

        # --- construct forms ---
        include_hu_div = True  # TODO put in `options`
        include_ext_pressure_grad = True
        include_hori_advection = True

        # horizontal advection and external pressure gradient terms
        F1 = as_vector((hu_old, hu_old*vel_uv[0] + 0.5*self.lam_kap*self.grav_z*h_old**2, hv_old * vel_uv[0]))
        F2 = as_vector((hv_old, hu_old*vel_uv[1], hv_old*vel_uv[1] + 0.5*self.lam_kap*self.grav_z*h_old**2))
        f = -(dot(Dx(self.test, 0), F1) + dot(Dx(self.test, 1), F2))*self.dx

        # set up modified vectors and evaluate fluxes
        w_plus = as_vector((h_old, mom_uv[0], mom_uv[1]))('+')
        w_minus = as_vector((h_old, mom_uv[0], mom_uv[1]))('-')
        flux_plus = self.interior_flux(self.normal('+'), self.function_space, w_plus, w_minus)
        flux_minus = self.interior_flux(self.normal('-'), self.function_space, w_minus, w_plus)
        f += (dot(flux_minus, self.test('-')) + dot(flux_plus, self.test('+')))*self.dS

        if self.options_nh.flow_is_granular:
            grad_p = fields_old.get('fluid_pressure_gradient')
            # TODO input upper surface fluid pressure in fields
            if grad_p is not None:
                grad_p_mod = conditional(h <= 0, zero(grad_p.ufl_shape), grad_p*self.options_nh.bed_slope[2])
            else:
                grad_p_mod = Constant((0.0, 0.0))
            uv_mag = sqrt(vel_uv[0]**2 + vel_uv[1]**2)
            src_x = (
                self.grav_x*h_old + h_old/self.options_nh.rho_g*grad_p_mod[0]
                - (1.0 - lamda)*self.grav_z*h_old*tan(phi_b)*vel_uv[0]/(uv_mag + 1e-16)
                - sign(s_xy)*(1.0 - lamda)*self.grav_z*h_old*kap*Dx(h_old, 1)*sin(phi_i)
            )
            src_y = (
                self.grav_y*h_old + h_old/self.options_nh.rho_g*grad_p_mod[1]
                - (1.0 - lamda)*self.grav_z*h_old*tan(phi_b)*vel_uv[1]/(uv_mag + 1e-16)
                - sign(s_xy)*(1.0 - lamda)*self.grav_z*h_old*kap*Dx(h_old, 0)*sin(phi_i)
            )
            f += -(src_x*self.test_uv[0] + src_y*self.test_uv[1])*self.dx
        else:
            # bathymetry gradient term
            bath_grad = as_vector((0, self.grav_z*h_old*Dx(self.bathymetry, 0), self.grav_z*h_old*Dx(self.bathymetry, 1)))
            f += -dot(bath_grad, self.test)*self.dx

        # add in boundary fluxes
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if funcs is not None:  # TODO enrich
                if 'elev' in funcs:
                    h_ini = funcs['elev'] + self.bathymetry
                    h_ext = conditional(h_ini <= 0, zero(h_ini.ufl_shape), h_ini)
                    h_ext_old = h_ext
                else:  # assume symmetry
                    h_ext = h
                    h_ext_old = h_old
                if 'uv' in funcs:
                    uv_ext = funcs['uv']  # or funcs['flux']/area*self.normal
                    uv_ext_old = uv_ext
                else:  # inflow or outflow
                    uv_ext = vel_uv
                    uv_ext_old = vel_uv_old
            # HUDivTerm
            if include_hu_div:
                if funcs is not None:
                    # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                    h_av = 0.5*(h_old + h_ext_old)
                    h_jump = h - h_ext
                    un_rie = 0.5*dot(vel_uv + uv_ext, self.normal) + sqrt(self.grav_z/h_av)*h_jump
                    un_jump = dot(vel_uv_old - uv_ext_old, self.normal)
                    h_rie = 0.5*(h_old + h_ext_old) + sqrt(h_av/self.grav_z)*un_jump
                    f += h_rie*un_rie*self.test_h*ds_bnd
            # ExternalPressureGradientTerm
            if include_ext_pressure_grad:
                if funcs is not None:
                    # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                    un_jump = dot(vel_uv - uv_ext, self.normal)
                    h_rie = 0.5*(h + h_ext) + sqrt(h_old/self.grav_z)*un_jump
                    f += 0.5*self.lam_kap*self.grav_z*h_rie*h_rie*dot(self.test_uv, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # NOTE seems inaccurate for granular flow with inclined slope TODO improve
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(vel_uv_old, self.normal)
                    h_rie = h_old + sqrt(h_old/self.grav_z)*un_jump
                    f += 0.5*self.lam_kap*self.grav_z*h_rie*h_rie*dot(self.test_uv, self.normal)*ds_bnd
            # HorizontalAdvectionTerm
            if include_hori_advection:
                if funcs is not None:
                    h_jump = h_old - h_ext_old
                    un_rie = 0.5*inner(vel_uv_old + uv_ext_old, self.normal) + sqrt(self.grav_z/h_old)*h_jump
                    uv_av = 0.5*(uv_ext + vel_uv)
                    f += h_old*(uv_av[0]*self.test_uv[0]*un_rie + uv_av[1]*self.test_uv[1]*un_rie)*ds_bnd
                if funcs is None:
                    # NOTE seems inaccurate for granular flow with inclined slope TODO improve, WPan 2020-03-29
                    # impose impermeability with mirror velocity
                    uv_ext = vel_uv - 2*dot(vel_uv, self.normal)*self.normal
                    if self.options.use_lax_friedrichs_velocity:
                        uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                        gamma = 0.5*abs(dot(mom_uv_old, self.normal))*uv_lax_friedrichs
                        f += gamma*dot(self.test_uv, vel_uv - uv_ext)*ds_bnd

        # quadratic Manning bottom friction term
        manning_drag_coefficient = self.options.manning_drag_coefficient
        nikuradse_bed_roughness = self.options.nikuradse_bed_roughness
        C_D = self.options.quadratic_drag_coefficient
        if manning_drag_coefficient is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * manning_drag_coefficient**2 / h_old**(1./3.)
        if nikuradse_bed_roughness is not None:
            if manning_drag_coefficient is not None:
                raise Exception('Cannot set both Nikuradse drag and Manning drag parameter')
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Nikuradse drag parameter')
            kappa = physical_constants['von_karman']
            C_D = conditional(h_old > nikuradse_bed_roughness, 2*(kappa**2)/(ln(11.036*total_h/nikuradse_bed_roughness)**2), Constant(0.0))
        if C_D is not None:
            f += C_D * sqrt(dot(vel_uv_old, vel_uv_old) + self.options.norm_smoother**2) * inner(self.test_uv, vel_uv) * self.dx

        # source term in vector form
        source = fields_old.get('source')
        if source is not None:
            f += -dot(source, self.test)*self.dx

        return -f

    def interior_flux(self, N, V, wr, wl):
        """
        Evaluate the interior fluxes between the positively and negatively restricted vectors wr, wl.

        """
        hr, mur, mvr = wr[0], wr[1], wr[2]
        hl, mul, mvl = wl[0], wl[1], wl[2]

        # negigible depth for the explicit wetting and drying method
        E = self.options_nh.wetting_and_drying_threshold
        gravity = self.grav_z
        g = conditional(And(hr < E, hl < E), zero(gravity('+').ufl_shape), gravity('+'))

        # use HLLC flux
        hl_zero = conditional(hl <= 0, 0, 1)
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr)).ufl_shape),
                         hl_zero * as_vector((mur / hr, mvr / hr)))
        hr_zero = conditional(hr <= 0, 0, 1)
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl)).ufl_shape),
                         hr_zero * as_vector((mul / hl, mvl / hl)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        lam_kap = avg(self.lam_kap)
        c_minus = Min(vr - sqrt(lam_kap * g * hr), vl - sqrt(lam_kap * g * hl))
        c_plus = Max(vr + sqrt(lam_kap * g * hr), vl + sqrt(lam_kap * g * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        y = (
            (0.5 * g * hr * hr - 0.5 * g * hl * hl + hl * vl * (c_plus - vl) - hr * vr * (c_minus - vr))
            / (hl * (c_plus - vl) - hr * (c_minus - vr))
        )
        c_s = conditional(abs(hr * (c_minus - vr) - hl * (c_plus - vl)) <= 1e-16, zero(y.ufl_shape), y)

        velocityl = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mvl) / hl)
        velocity_ul = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mul) / hl)
        velocity_vl = conditional(hl <= 0, zero(mvl.ufl_shape), (hr_zero * mvl * mvl) / hl)
        velocityr = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mvr) / hr)
        velocity_ur = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mur) / hr)
        velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mvr) / hr)

        F1r = as_vector((mur,
                         velocity_ur + 0.5 * lam_kap * g * hr**2,
                         velocityr))
        F2r = as_vector((mvr,
                         velocityr,
                         velocity_vr + 0.5 * lam_kap * g * hr**2))

        F1l = as_vector((mul,
                         velocity_ul + 0.5 * lam_kap * g * hl**2,
                         velocityl))
        F2l = as_vector((mvl,
                         velocityl,
                         velocity_vl + 0.5 * lam_kap * g * hl**2))

        F_plus = as_vector((F1r, F2r))
        F_minus = as_vector((F1l, F2l))

        W_plus = as_vector((hr, mur, mvr))
        W_minus = as_vector((hl, mul, mvl))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-16, zero(y.ufl_shape), y)
        # conditional to prevent dividing by zero
        y = (c_minus - vr) / (c_minus - c_s) * (
            W_plus - as_vector((0,
                                hr * (c_s - v_star) * N[0],
                                hr * (c_s - v_star) * N[1]))
        )
        w_plus = conditional(abs(c_minus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = (c_plus - vl) / (c_plus - c_s) * (
            W_minus - as_vector((0,
                                 hl * (c_s - v_star) * N[0],
                                 hl * (c_s - v_star) * N[1]))
        )
        w_minus = conditional(abs(c_plus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        Flux = (
            0.5 * dot(N, F_plus + F_minus)
            + (
                0.5 * (- (abs(c_minus) - abs(c_s)) * w_minus
                       + (abs(c_plus) - abs(c_s)) * w_plus
                       + abs(c_minus) * W_plus
                       - abs(c_plus) * W_minus)
            )
        )

        return Flux
