r"""
Depth averaged shallow water equations in conservative form
"""
from __future__ import absolute_import
from .utility_nh import *
from thetis.equation import Equation

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']

class BaseShallowWaterEquation(Equation):
    """
    Abstract base class for ShallowWaterEquations.

    """
    def __init__(self, function_space,
                 bathymetry,
                 options):
        super(BaseShallowWaterEquation, self).__init__(function_space)
        # define bunch of members needed to construct forms
        self.function_space = function_space
        self.bathymetry = bathymetry
        self.options = options
        self.mesh = self.function_space.mesh()
        self.test = TestFunction(self.function_space)
        self.trial = TrialFunction(self.function_space)
        self.normal = FacetNormal(self.mesh)
        self.boundary_markers = sorted(self.function_space.mesh().exterior_facets.unique_markers)
        self.boundary_len = self.function_space.mesh().boundary_len
        # negigible depth set for wetting and drying
        self.threshold = self.options.depth_wd_interface
        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)
        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())
        self.dS = dS(degree=self.quad_degree,
                     domain=self.function_space.ufl_domain())

    def interior_flux(self, N, V, wr, wl):
        """ 
        This evaluates the interior fluxes between the positively and negatively restricted vectors wr, wl.

        """
        hr, mur, mvr = wr[0], wr[1], wr[2]
        hl, mul, mvl = wl[0], wl[1], wl[2]

        E = self.threshold
        gravity = Function(V.sub(0)).assign(g_grav)
        g = conditional(And(hr < E, hl < E), zero(gravity('+').ufl_shape), gravity('+'))

        # Do HLLC flux
        hl_zero = conditional(hl <= 0, 0, 1)
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr)).ufl_shape),
                         hl_zero * as_vector((mur / hr, mvr / hr)))
        hr_zero = conditional(hr <= 0, 0, 1)
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl)).ufl_shape),
                         hr_zero * as_vector((mul / hl, mvl / hl)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        c_minus = Min(vr - sqrt(g * hr), vl - sqrt(g * hl))
        c_plus = Min(vr + sqrt(g * hr), vl + sqrt(g * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        c_s = conditional(abs(hr * (c_minus - vr) - hl * (c_plus - vl)) <= 1e-16, zero(y.ufl_shape), y)

        velocityl = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mvl) / hl)
        velocity_ul = conditional(hl <= 0, zero(mul.ufl_shape), (hr_zero * mul * mul) / hl)
        velocity_vl = conditional(hl <= 0, zero(mvl.ufl_shape), (hr_zero * mvl * mvl) / hl)
        velocityr = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mvr) / hr)
        velocity_ur = conditional(hr <= 0, zero(mur.ufl_shape), (hl_zero * mur * mur) / hr)
        velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (hl_zero * mvr * mvr) / hr)

        F1r = as_vector((mur,
                         velocity_ur + 0.5 * g * hr**2,
                         velocityr))
        F2r = as_vector((mvr,
                         velocityr,
                         velocity_vr + 0.5 * g * hr**2))

        F1l = as_vector((mul,
                         velocity_ul + 0.5 * g * hl**2,
                         velocityl))
        F2l = as_vector((mvl,
                         velocityl,
                         velocity_vl + 0.5 * g * hl**2))

        F_plus = as_vector((F1r, F2r))
        F_minus = as_vector((F1l, F2l))

        W_plus = as_vector((hr, mur, mvr))
        W_minus = as_vector((hl, mul, mvl))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr) #+ sqrt(g * hr) - sqrt(g * hl)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-16, zero(y.ufl_shape), y)
        # conditional to prevent dividing by zero
        y = ((c_minus - vr) / (c_minus - c_s)) * (W_plus -
                                                  as_vector((0,
                                                            hr * (c_s - v_star) * N[0],
                                                            hr * (c_s - v_star) * N[1])))
        w_plus = conditional(abs(c_minus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_plus - vl) / (c_plus - c_s)) * (W_minus -
                                                as_vector((0,
                                                          hl * (c_s - v_star) * N[0],
                                                          hl * (c_s - v_star) * N[1])))
        w_minus = conditional(abs(c_plus - c_s) <= 1e-16, zero(y.ufl_shape), y)

        Flux = ((0.5 * dot(N, F_plus + F_minus)) +
                (0.5 * (-((abs(c_minus) - abs(c_s)) * w_minus) +
                        ((abs(c_plus) - abs(c_s)) * w_plus) +
                        (abs(c_minus) * W_plus) -
                        (abs(c_plus) * W_minus))))

        return Flux

    def boundary_flux(self, V, w, bc_funcs):
        """ 
        This evaluates the boundary flux between the vector and a solid reflective wall (temporarily zero velocity and same depth) or other boundary conditions options.
        Here, mur and mul denote outside and inside of momentum cell, respectively.

        """
        N = self.normal

        h, mu, mv = split(w)

        if bc_funcs is None: # TODO improve stability with increased time step size
            mul = Constant(0)
            mur = mu
            mvl = Constant(0)
            mvr = mv
            hr = h
            hl = h
        else:
            if 'inflow' in bc_funcs:
                mul = value.sub(1) # TODO
                mur = mu
                mvl = value.sub(2)
                mvr = mv
                hr = h
                hl = h
            if 'outflow' in bc_funcs:
                mul = mu
                mur = mu
                mvr = mv
                mvl = mv
                hr = h
                hl = h

        # Do HLLC flux
        ul = conditional(hl <= 0, zero(as_vector((mul / hl, mvl / hl)).ufl_shape),
                         as_vector((mul / hl, mvl / hl)))
        ur = conditional(hr <= 0, zero(as_vector((mur / hr, mvr / hr)).ufl_shape),
                         as_vector((mur / hr, mvr / hr)))
        vr = dot(ur, N)
        vl = dot(ul, N)
        # wave speed depending on wavelength
        c_minus = Min(vr - sqrt(g_grav * hr), vl - sqrt(g_grav * hl))
        c_plus = Min(vr + sqrt(g_grav * hr), vl + sqrt(g_grav * hl))
        # not divided by zero height
        y = (hl * c_minus * (c_plus - vl) - hr * c_plus * (c_minus - vr)) / (hl * (c_plus - vl) - hr * (c_minus - vr))
        c_s = conditional(abs(hr * (c_minus - vr) - hl * (c_plus - vl)) <= 1e-8, zero(y.ufl_shape), y)

        velocityl = conditional(hl <= 0, zero(mul.ufl_shape), (mul * mvl) / hl)
        velocity_ul = conditional(hl <= 0, zero(mul.ufl_shape), (mul * mul) / hl)
        velocity_ur = conditional(hr <= 0, zero(mul.ufl_shape), (mur * mur) / hr)
        velocityr = conditional(hr <= 0, zero(mul.ufl_shape), (mur * mvr) / hr)
        velocity_vr = conditional(hr <= 0, zero(mvr.ufl_shape), (mvr * mvr) / hr)
        velocity_vl = conditional(hl <= 0, zero(mvl.ufl_shape), (mvl * mvl) / hl)

        F1r = as_vector((mur,
                         velocity_ur + 0.5 * g_grav * hr**2,
                         velocityr))
        F2r = as_vector((mvr,
                         velocityr,
                         velocity_vr + 0.5 * g_grav * hr**2))

        F1l = as_vector((mul,
                         velocity_ul + 0.5 * g_grav * hl**2,
                         velocityl))
        F2l = as_vector((mvl,
                         velocityl,
                         velocity_vl + 0.5 * g_grav * hl**2))

        F_plus = as_vector((F1r, F2r))
        F_minus = as_vector((F1l, F2l))

        W_plus = as_vector((hr, mur, mvr))
        W_minus = as_vector((hl, mul, mvl))

        y = ((sqrt(hr) * vr) + (sqrt(hl) * vl)) / (sqrt(hl) + sqrt(hr))
        y = 0.5 * (vl + vr) #+ sqrt(g * hr) - sqrt(g * hl)
        v_star = conditional(abs(sqrt(hl) + sqrt(hr)) <= 1e-8, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_minus - vr) / (c_minus - c_s)) * (W_plus -
                                                  as_vector((0,
                                                            hl * (c_s - v_star) * N[0],
                                                            hl * (c_s - v_star) * N[1])))
        w_plus = conditional(abs(c_minus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        # conditional to prevent dividing by zero
        y = ((c_plus - vl) / (c_plus - c_s)) * (W_minus -
                                                as_vector((0,
                                                          hr * (c_s - v_star) * N[0],
                                                          hr * (c_s - v_star) * N[1])))
        w_minus = conditional(abs(c_plus - c_s) <= 1e-8, zero(y.ufl_shape), y)

        Flux = ((0.5 * dot(N, F_plus + F_minus)) +
                (0.5 * (-((abs(c_minus) - abs(c_s)) * w_minus) +
                        ((abs(c_plus) - abs(c_s)) * w_plus) +
                        (abs(c_minus) * W_plus) -
                        (abs(c_plus) * W_minus))))

        return Flux

    def wd_depth_displacement(self, eta):
        """
        Returns depth change due to wetting and drying
        """
        if (not self.options.use_hllc_flux) and self.options.use_wetting_and_drying:
            h = self.bathymetry + eta
            return 2 * self.threshold**2 / (2 * self.threshold + abs(h)) + 0.5 * (abs(h) - h)
        else:
            return 0


class ShallowWaterEquations(BaseShallowWaterEquation):
    """
    2D depth-averaged shallow water equations in conservative form.

    This defines the full 2D SWE equations.
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
        super(ShallowWaterEquations, self).__init__(function_space, bathymetry, options)
        self.test_h = self.test[0]
        self.test_uv = as_vector((self.test[1], self.test[2]))

    def mass_term(self, solution):
        f = super(ShallowWaterEquations, self).mass_term(solution)
       # if self.options.use_wetting_and_drying:
        #    assert self.options.use_hllc_flux is True
        if (not self.options.use_hllc_flux) and self.options.use_wetting_and_drying:
            f += dot(self.wd_depth_displacement(solution[0]), self.test_h)*self.dx
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):

        f = 0

        include_hu_div = True
        include_ext_pressure_grad = True
        include_hori_advection = True

        if self.options.use_hllc_flux:

            eta, hu, hv = split(solution)
            eta_old, hu_old, hv_old = split(solution_old)
            h = self.bathymetry + eta
            h_old = self.bathymetry + eta_old

            def mom(hu, h):
                return conditional(h <= 0, zero(hu.ufl_shape), hu)
            def vel(hu, h):
                return conditional(h <= 0, zero(hu.ufl_shape), hu / h)

            # momentum
            mom_vec = as_vector((hu, hv))
            mom_vec_old = as_vector((hu_old, hv_old))
            mom_uv = as_vector((mom(hu, h), mom(hv, h)))
            mom_uv_old = as_vector((mom(hu_old, h_old), mom(hv_old, h_old)))

            # velocity
            vel_uv = as_vector((vel(hu, h), vel(hv, h)))
            vel_uv_old = as_vector((vel(hu_old, h_old), vel(hv_old, h_old)))

            # construct forms
            F1 = as_vector((hu_old, hu_old*vel_uv[0] + 0.5*g_grav*h_old**2, hv_old * vel_uv[0]))
            F2 = as_vector((hv_old, hu_old*vel_uv[1], hv_old*vel_uv[1] + 0.5*g_grav*h_old**2))
            f += -(dot(Dx(self.test, 0), F1) + dot(Dx(self.test, 1), F2))*self.dx

            # set up modified vectors and evaluate fluxes
            w_plus = as_vector((h_old, mom_uv[0], mom_uv[1]))('+')
            w_minus = as_vector((h_old, mom_uv[0], mom_uv[1]))('-')
            flux_plus = self.interior_flux(self.normal('+'), self.function_space, w_plus, w_minus)
            flux_minus = self.interior_flux(self.normal('-'), self.function_space, w_minus, w_plus)
            f += (dot(flux_minus, self.test('-')) + dot(flux_plus, self.test('+')))*self.dS

            # bathymetry gradient term
            bath_grad = as_vector((0, g_grav * h_old * Dx(self.bathymetry, 0), g_grav * h_old * Dx(self.bathymetry, 1)))
            f += -dot(bath_grad, self.test)*self.dx

        else:

            eta, uv = split(solution)
            eta_old, uv_old = split(solution_old)
            total_h = self.bathymetry + eta_old + self.wd_depth_displacement(eta_old)

            # HUDivTerm
            if include_hu_div:
                f += -inner(grad(self.test_h), total_h*uv)*self.dx
                h_avg = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h_avg)*jump(eta, self.normal)
                hu_star = h_avg*uv_rie
                f += inner(jump(self.test_h, self.normal), hu_star)*self.dS

            # ExternalPressureGradientTerm
            if include_ext_pressure_grad:
                f += -g_grav*eta*div(self.test_uv)*self.dx
                eta_star = avg(eta) + sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
                f += g_grav*eta_star*jump(self.test_uv, self.normal)*self.dS

            # HorizontalAdvectionTerm
            if include_hori_advection:
                f += -(uv[0]*Dx(uv_old[0]*self.test_uv[0], 0)
                       + uv[1]*Dx(uv_old[0]*self.test_uv[1], 0)
                       + uv[0]*Dx(uv_old[1]*self.test_uv[0], 1)
                       + uv[1]*Dx(uv_old[1]*self.test_uv[1], 1))*self.dx
                uv_up = avg(uv) # mean flux
                f += (uv_up[0]*jump(self.test_uv[0], uv_old[0]*self.normal[0])
                      + uv_up[1]*jump(self.test_uv[1], uv_old[0]*self.normal[0])
                      + uv_up[0]*jump(self.test_uv[0], uv_old[1]*self.normal[1])
                      + uv_up[1]*jump(self.test_uv[1], uv_old[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if self.options.use_lax_friedrichs_velocity:
                    uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                    un_av = dot(avg(uv_old), self.normal('-'))
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.test_uv), jump(uv))*self.dS

            # SpongeDampingTerm  
            sponge_damping = fields_old.get('sponge_damping_2d')
            if sponge_damping is not None:
                f += sponge_damping*inner(self.test_uv, uv)*self.dx

        if False: # for backup
            include_hu_div = True
            include_ext_pressure_grad = True
            include_hori_advection = True
            h_mod = h_old + self.wd_depth_displacement(h_old)
            # HUDivTerm
            if include_hu_div:
                f += -dot(grad(self.test_h), mom_uv)*self.dx # mom_vec?
                h_avg = avg(h_old)
                uv_rie = avg(vel_uv) + sqrt(g_grav/h_avg)*jump(h, self.normal)
                f += inner(jump(self.test_h, self.normal), h_avg*uv_rie)*self.dS
            # ExternalPressureGradientTerm
            if include_ext_pressure_grad:
                f += -0.5*g_grav*h*h*div(self.test_uv)*self.dx
                h_star = avg(h) + sqrt(avg(h_old)/g_grav)*jump(vel_uv, self.normal)
                f += 0.5*g_grav*h_star*h_star*jump(self.test_uv, self.normal)*self.dS
            # HorizontalAdvectionTerm
            if include_hori_advection:
                f += -(mom_uv_old[0]*vel_uv[0]*Dx(self.test_uv[0], 0)
                       + mom_uv_old[0]*vel_uv[1]*Dx(self.test_uv[1], 0)
                       + mom_uv_old[1]*vel_uv[0]*Dx(self.test_uv[0], 1)
                       + mom_uv_old[1]*vel_uv[1]*Dx(self.test_uv[1], 1))*self.dx
                uv_up = avg(vel_uv) # mean flux
                f += (uv_up[0]*jump(self.test_uv[0], mom_uv_old[0]*self.normal[0])
                      + uv_up[1]*jump(self.test_uv[1], mom_uv_old[0]*self.normal[0])
                      + uv_up[0]*jump(self.test_uv[0], mom_uv_old[1]*self.normal[1])
                      + uv_up[1]*jump(self.test_uv[1], mom_uv_old[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if self.options.use_lax_friedrichs_velocity:
                    uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                    mom_av = dot(avg(mom_uv_old), self.normal('-'))
                    gamma = 0.5*abs(mom_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.test_uv), jump(vel_uv))*self.dS

        # add in boundary fluxes
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if self.options.use_hllc_flux:
                # TODO improve, especially in terms of decay in surface gravity wave
               # flux_bnd = self.boundary_flux(self.function_space, solution_old, funcs)
               # f += dot(flux_bnd, self.test)*ds_bnd
                if funcs is not None: # TODO enrich
                    if 'elev' in funcs:
                        h_ext = self.bathymetry + funcs['elev']
                        h_ext_old = h_ext
                    else: # 'inflow' or 'outflow'
                        h_ext = h
                        h_ext_old = h_old
                    if 'uv' in funcs:
                        uv_ext = funcs['uv'] # or funcs['flux']/h_ext
                        uv_ext_old = funcs['uv']
                    else: # 'inflow' or 'outflow'
                        uv_ext = mom_uv/h_ext
                        uv_ext_old = mom_uv_old/h_ext
                # HUDivTerm
                if include_hu_div:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        h_av = 0.5*(h_old + h_ext_old)
                        h_jump = h - h_ext
                        # u_star = 0.5 * (vl + vr) + sqrt(g * hl) - sqrt(g * hr)
                        # h_star = (0.5 * (sqrt(g * hl) + sqrt(g * hr)) + 0.25 * (vl - vr))**2/g
                        un_rie = 0.5*dot(vel_uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*h_jump
                        un_jump = dot(vel_uv_old - uv_ext_old, self.normal)
                        h_rie = 0.5*(h_old + h_ext_old) + sqrt(h_av/g_grav)*un_jump
                        f += h_rie*un_rie*self.test_h*ds_bnd
                # ExternalPressureGradientTerm
                if include_ext_pressure_grad:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        un_jump = dot(vel_uv - uv_ext, self.normal)
                        h_rie = 0.5*(h + h_ext) + sqrt(h_old/g_grav)*un_jump
                        f += 0.5*g_grav*h_rie*h_rie*dot(self.test_uv, self.normal)*ds_bnd
                    if funcs is None or 'symm' in funcs:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(vel_uv_old, self.normal)
                        h_rie = h_old + sqrt(h_old/g_grav)*un_jump
                        f += 0.5*g_grav*h_rie*h_rie*dot(self.test_uv, self.normal)*ds_bnd
                # HorizontalAdvectionTerm
                if include_hori_advection:
                    if funcs is not None:
                        h_jump = h_old - h_ext_old
                        un_rie = 0.5*inner(vel_uv_old + uv_ext_old, self.normal) + sqrt(g_grav/h_old)*h_jump
                        uv_av = 0.5*(uv_ext + vel_uv)
                        f += h_old*(uv_av[0]*self.test_uv[0]*un_rie + uv_av[1]*self.test_uv[1]*un_rie)*ds_bnd
                    if funcs is None:
                        # impose impermeability with mirror velocity
                        uv_ext = vel_uv - 2*dot(vel_uv, self.normal)*self.normal
                        if self.options.use_lax_friedrichs_velocity:
                            uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                            gamma = 0.5*abs(dot(mom_uv_old, self.normal))*uv_lax_friedrichs
                            f += gamma*dot(self.test_uv, vel_uv - uv_ext)*ds_bnd
            else:
                if funcs is not None: # TODO enrich
                    if 'elev' in funcs:
                        eta_ext = funcs['elev']
                        eta_ext_old = funcs['elev']
                    else: # 'inflow' or 'outflow'
                        eta_ext = eta
                        eta_ext_old = eta_old
                    if 'uv' in funcs:
                        uv_ext = funcs['uv'] # or funcs['flux']/h_ext
                        uv_ext_old = funcs['uv']
                    else: # 'inflow' or 'outflow'
                        uv_ext = uv
                        uv_ext_old = uv_old
                    total_h_ext = self.bathymetry + eta_ext_old + self.wd_depth_displacement(eta_ext_old)

                # HUDivTerm
                if include_hu_div:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        h_av = 0.5*(total_h + total_h_ext)
                        eta_jump = eta - eta_ext
                        un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                        un_jump = inner(uv_old - uv_ext_old, self.normal)
                        h_rie = self.bathymetry + 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                        f += h_rie*un_rie*self.test_h*ds_bnd

                # ExternalPressureGradientTerm
                if include_ext_pressure_grad:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        un_jump = dot(uv - uv_ext, self.normal)
                        eta_rie = 0.5*(eta + eta_ext) + sqrt(total_h/g_grav)*un_jump
                        f += g_grav*eta_rie*dot(self.test_uv, self.normal)*ds_bnd
                    if funcs is None or 'symm' in funcs:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(uv, self.normal)
                        eta_rie = eta + sqrt(total_h/g_grav)*un_jump
                        f += g_grav*eta_rie*dot(self.test_uv, self.normal)*ds_bnd

                # HorizontalAdvectionTerm
                if include_hori_advection:
                    if funcs is not None:
                        eta_jump = eta_old - eta_ext_old
                        un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                        uv_av = 0.5*(uv_ext + uv)
                        f += (uv_av[0]*self.test_uv[0]*un_rie + uv_av[1]*self.test_uv[1]*un_rie)*ds_bnd
                    if funcs is None:
                        # impose impermeability with mirror velocity
                        uv_ext = uv - 2*dot(uv, self.normal)*self.normal
                        if self.options.use_lax_friedrichs_velocity:
                            uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                            gamma = 0.5*abs(dot(uv_old, self.normal))*uv_lax_friedrichs
                            f += gamma*dot(self.test_uv, uv - uv_ext)*ds_bnd

        # source term in vector form
        source_sw = fields_old.get('source_sw')
        if source_sw is not None:
            f += -dot(source_sw, self.test)*self.dx

        # landslide source
        slide_source = fields_old.get('slide_source')
        if slide_source is not None:
            f += -slide_source * self.test_h * self.dx

        return -f


class FreeSurfaceEquation(BaseShallowWaterEquation):
    """
    2D free surface equation :eq:`swe_freesurf` in conservative form.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry,
                 options):
        """
        :arg eta_test: test function of the elevation function space
        :arg eta_space: elevation function space
        :arg u_space: velocity function space
        :arg function_space: Mixed function space where the solution belongs
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(FreeSurfaceEquation, self).__init__(eta_space, bathymetry, options)
        self.test_h = eta_test

    def mass_term(self, solution):
        f = super(FreeSurfaceEquation, self).mass_term(solution)
        if (not self.options.use_hllc_flux) and self.options.use_wetting_and_drying:
            f += dot(self.wd_depth_displacement(solution), self.test_h)*self.dx
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        f = 0
        eta = solution
        eta_old = solution_old

        h_2d = self.bathymetry + eta
        h = conditional(h_2d <= 0, 0, h_2d)

        uv = fields.get('uv_2d')
        total_h = self.bathymetry + eta_old + self.wd_depth_displacement(eta_old)

        # momentum
        if fields.get('mom_2d') is not None:
            mom = fields.get('mom_2d')
            mom_uv = conditional(h <= 0, zero(mom.ufl_shape), mom)
            vel_uv = conditional(h <= 0, zero(mom.ufl_shape), mom / h)

        # construct forms
        if self.options.use_hllc_flux:
            f += -inner(grad(self.test_h), mom_uv)*self.dx
            # set up modified vectors and evaluate fluxes
            w_plus = as_vector((h, mom_uv[0], mom_uv[1]))('+')
            w_minus = as_vector((h, mom_uv[0], mom_uv[1]))('-')
            flux_plus = self.interior_flux(self.normal('+'), self.function_space, w_plus, w_minus)
            flux_minus = self.interior_flux(self.normal('-'), self.function_space, w_minus, w_plus)
            f += (dot(flux_minus[0], self.test_h('-')) + dot(flux_plus[0], self.test_h('+')))*self.dS
        else:
            # default: `dg`, i.e. hu by parts
           # f += -inner(grad(self.test_h), mom_uv)*self.dx
           # h_avg = avg(h)
           # mom_rie = avg(mom_uv) + sqrt(g_grav*h_avg)*jump(h, self.normal)
           # f += inner(jump(self.test_h, self.normal), mom_rie)*self.dS

            total_h = self.bathymetry + eta_old + self.wd_depth_displacement(eta_old)
            # uv in fields
            use_uv = True
            if use_uv:
                uv = fields.get('uv_2d')
                uv_mod = uv#conditional(h <= 0, zero(uv.ufl_shape), uv)
                f = -inner(grad(self.test_h), total_h*uv_mod)*self.dx
                h_avg = avg(total_h)
                uv_rie = avg(uv_mod) + sqrt(g_grav/h_avg)*jump(eta, self.normal)
                f += inner(jump(self.test_h, self.normal), h_avg*uv_rie)*self.dS

        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if False:#self.options.use_hllc_flux:
                # TODO improve, especially in terms of decay in surface gravity wave
               # flux_bnd = self.boundary_flux(self.function_space, solution_old, funcs)
               # f += dot(flux_bnd, self.test)*ds_bnd
                if funcs is not None: # TODO enrich
                    if 'elev' in funcs:
                        h_ext = self.bathymetry + funcs['elev']
                    else: # 'inflow' or 'outflow'
                        h_ext = h
                    if 'uv' in funcs:
                        uv_ext = funcs['uv'] # or funcs['flux']/h_ext
                    else: # 'inflow' or 'outflow'
                        uv_ext = mom_uv/h_ext
                # HUDivTerm
                if True:
                    if funcs is not None:
                        # Compute linear riemann solution with h_ext, vel_uv, uv_ext
                        h_av = 0.5*(h + h_ext)
                        h_jump = h - h_ext
                        un_rie = 0.5*dot(vel_uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*h_jump
                        un_jump = dot(vel_uv - uv_ext, self.normal)
                        h_rie = 0.5*(h + h_ext) + sqrt(h_av/g_grav)*un_jump
                        f += h_rie*un_rie*self.test_h*ds_bnd
            else:
                if funcs is not None: # TODO enrich
                    if 'elev' in funcs:
                        eta_ext = funcs['elev']
                        eta_ext_old = funcs['elev']
                    else: # 'inflow' or 'outflow'
                        eta_ext = eta
                        eta_ext_old = eta_old
                    if 'uv' in funcs:
                        uv_ext = funcs['uv'] # or funcs['flux']/h_ext
                        uv_ext_old = funcs['uv']
                    else: # 'inflow' or 'outflow'
                        uv_ext = uv
                        uv_ext_old = uv
                    total_h_ext = self.bathymetry + eta_ext_old + self.wd_depth_displacement(eta_ext_old)
                # HUDivTerm
                if True:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        h_av = 0.5*(total_h + total_h_ext)
                        eta_jump = eta - eta_ext
                        un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                        un_jump = inner(uv - uv_ext_old, self.normal)
                        h_rie = self.bathymetry + 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                        f += h_rie*un_rie*self.test_h*ds_bnd

        # landslide source
        slide_source = fields_old.get('slide_source')
        if slide_source is not None:
           # f += -conditional(h <= self.threshold, 0, slide_source) * self.test_h * self.dx
            f += -slide_source * self.test_h * self.dx

        return -f


class OperatorSplitEquations(BaseShallowWaterEquation):
    r"""
    2D depth-averaged shallow water equations for operator splitting schemes.

    Defines the equations :eq:`swe_freesurf_modesplit` -
    :eq:`swe_momentum_modesplit`.
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
        # TODO remove include_grad_* options as viscosity operator is omitted
        super(OperatorSplitEquations, self).__init__(function_space, bathymetry, options)

        self.test_h = self.test[0]
        self.test_uv = as_vector((self.test[1], self.test[2]))

    def mass_term(self, solution):
        f = super(OperatorSplitEquations, self).mass_term(solution)
       # if self.options.use_wetting_and_drying:
        #    assert self.options.use_hllc_flux is True
        if (not self.options.use_hllc_flux) and self.options.use_wetting_and_drying:
            f += dot(self.wd_depth_displacement(solution[0]), self.test_h)*self.dx
        return f

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):

        f = 0

        include_hu_div = True
        include_ext_pressure_grad = True
        include_hori_advection = not True

        if True:

            eta, uv = split(solution)
            eta_old, uv_old = split(solution_old)
            total_h = self.bathymetry + eta_old + self.wd_depth_displacement(eta_old)

            # HUDivTerm
            if include_hu_div:
                f += -inner(grad(self.test_h), total_h*uv)*self.dx
                h_avg = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h_avg)*jump(eta, self.normal)
                hu_star = h_avg*uv_rie
                f += inner(jump(self.test_h, self.normal), hu_star)*self.dS

            # ExternalPressureGradientTerm
            if include_ext_pressure_grad:
                f += -g_grav*eta*div(self.test_uv)*self.dx
                eta_star = avg(eta) + sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
                f += g_grav*eta_star*jump(self.test_uv, self.normal)*self.dS

            # HorizontalAdvectionTerm
            if include_hori_advection:
                f += -(uv[0]*Dx(uv_old[0]*self.test_uv[0], 0)
                       + uv[1]*Dx(uv_old[0]*self.test_uv[1], 0)
                       + uv[0]*Dx(uv_old[1]*self.test_uv[0], 1)
                       + uv[1]*Dx(uv_old[1]*self.test_uv[1], 1))*self.dx
                uv_up = avg(uv) # mean flux
                f += (uv_up[0]*jump(self.test_uv[0], uv_old[0]*self.normal[0])
                      + uv_up[1]*jump(self.test_uv[1], uv_old[0]*self.normal[0])
                      + uv_up[0]*jump(self.test_uv[0], uv_old[1]*self.normal[1])
                      + uv_up[1]*jump(self.test_uv[1], uv_old[1]*self.normal[1]))*self.dS
                # Lax-Friedrichs stabilization
                if self.options.use_lax_friedrichs_velocity:
                    uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                    un_av = dot(avg(uv_old), self.normal('-'))
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.test_uv), jump(uv))*self.dS

        # add in boundary fluxes
        for bnd_marker in self.boundary_markers:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
            if True:
                if funcs is not None: # TODO enrich
                    if 'elev' in funcs:
                        eta_ext = funcs['elev']
                        eta_ext_old = funcs['elev']
                    else: # 'inflow' or 'outflow'
                        eta_ext = eta
                        eta_ext_old = eta_old
                    if 'uv' in funcs:
                        uv_ext = funcs['uv'] # or funcs['flux']/h_ext
                        uv_ext_old = funcs['uv']
                    else: # 'inflow' or 'outflow'
                        uv_ext = uv
                        uv_ext_old = uv_old
                    total_h_ext = self.bathymetry + eta_ext_old + self.wd_depth_displacement(eta_ext_old)

                # HUDivTerm
                if include_hu_div:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        h_av = 0.5*(total_h + total_h_ext)
                        eta_jump = eta - eta_ext
                        un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                        un_jump = inner(uv_old - uv_ext_old, self.normal)
                        h_rie = self.bathymetry + 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                        f += h_rie*un_rie*self.test_h*ds_bnd

                # ExternalPressureGradientTerm
                if include_ext_pressure_grad:
                    if funcs is not None:
                        # Compute linear riemann solution with h_old, h_ext, vel_uv, uv_ext
                        un_jump = dot(uv - uv_ext, self.normal)
                        eta_rie = 0.5*(eta + eta_ext) + sqrt(total_h/g_grav)*un_jump
                        f += g_grav*eta_rie*dot(self.test_uv, self.normal)*ds_bnd
                    if funcs is None or 'symm' in funcs:
                        # assume land boundary
                        # impermeability implies external un=0
                        un_jump = inner(uv, self.normal)
                        eta_rie = eta + sqrt(total_h/g_grav)*un_jump
                        f += g_grav*eta_rie*dot(self.test_uv, self.normal)*ds_bnd

                # HorizontalAdvectionTerm
                if include_hori_advection:
                    if funcs is not None:
                        eta_jump = eta_old - eta_ext_old
                        un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/total_h)*eta_jump
                        uv_av = 0.5*(uv_ext + uv)
                        f += (uv_av[0]*self.test_uv[0]*un_rie + uv_av[1]*self.test_uv[1]*un_rie)*ds_bnd
                    if funcs is None:
                        # impose impermeability with mirror velocity
                        uv_ext = uv - 2*dot(uv, self.normal)*self.normal
                        if self.options.use_lax_friedrichs_velocity:
                            uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
                            gamma = 0.5*abs(dot(uv_old, self.normal))*uv_lax_friedrichs
                            f += gamma*dot(self.test_uv, uv - uv_ext)*ds_bnd

        # source term in vector form
        source_sw = fields_old.get('source_sw')
        if source_sw is not None:
            f += -dot(source_sw, self.test)*self.dx

        # landslide source
        slide_source = fields_old.get('slide_source')
        if slide_source is not None:
            f += -slide_source * self.test_h * self.dx

        return -f


class ShallowWaterMomentumEquation(BaseShallowWaterEquation):
    """
    2D depth averaged momentum equation :eq:`swe_momentum` in conservative form.
    """
    def __init__(self, u_test, u_space, eta_space,
                 bathymetry,
                 options):
        """
        :arg u_test: test function of the velocity function space
        :arg u_space: velocity function space
        :arg eta_space: elevation function space
        :arg bathymetry: bathymetry of the domain
        :type bathymetry: :class:`Function` or :class:`Constant`
        :arg options: :class:`.AttrDict` object containing all circulation model options
        """
        super(ShallowWaterMomentumEquation, self).__init__(u_space, bathymetry, options)
        self.add_momentum_terms(u_test, u_space, eta_space,
                                bathymetry, options)

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        uv = solution
        uv_old = solution_old
        eta = fields['eta']
        eta_old = fields_old['eta']
        return self.residual_uv_eta(label, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
