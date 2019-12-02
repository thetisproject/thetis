r"""
2D advection diffusion equation for tracers.

The advection-diffusion equation of tracer :math:`T` in non-conservative form reads

.. math::
    \frac{\partial T}{\partial t}
    + \nabla_h \cdot (\textbf{u} T)
    = \nabla_h \cdot (\mu_h \nabla_h T)
    :label: tracer_eq_2d

where :math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` are the horizontal
velocities, and
:math:`\mu_h` denotes horizontal diffusivity.
"""
from __future__ import absolute_import
from .utility import *
from .equation import Term, Equation

__all__ = [
    'TracerEquation2D',
    'TracerTerm',
    'HorizontalAdvectionTerm',
    'HorizontalDiffusionTerm',
    'SourceTerm',
]


class TracerTerm(Term):
    """
    Generic tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, use_lax_friedrichs=True, sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`
        """
        super(TracerTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.cellsize = CellSize(self.mesh)
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_dg = continuity.horizontal == 'dg'
        self.use_lax_friedrichs = use_lax_friedrichs
        self.sipg_parameter = sipg_parameter

        # define measures with a reasonable quadrature degree
        p = self.function_space.ufl_element().degree()
        self.quad_degree = 2*p + 1
        self.dx = dx(degree=self.quad_degree)
        self.dS = dS(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)

    def get_bnd_functions(self, c_in, uv_in, elev_in, bnd_id, bnd_conditions):
        """
        Returns external values of tracer and uv for all supported
        boundary conditions.

        Volume flux (flux) and normal velocity (un) are defined positive out of
        the domain.

        :arg c_in: Internal value of tracer
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

    def wd_bathymetry_displacement(self, eta):
        """
        Returns wetting and drying bathymetry displacement as described in:
        Karna et al.,  2011.
        """
        H = self.bathymetry + eta
        return 0.5 * (sqrt(H ** 2 + self.options.wetting_and_drying_alpha ** 2) - H)

    def get_total_depth(self, eta):
        """
        Returns total water column depth
        """
        if self.options.use_nonlinear_equations:
            total_h = self.bathymetry + eta
            if hasattr(self.options, 'use_wetting_and_drying') and self.options.use_wetting_and_drying:
                total_h += self.wd_bathymetry_displacement(eta)
        else:
            total_h = self.bathymetry
        return total_h


class HorizontalAdvectionTerm(TracerTerm):
    r"""
    Advection of tracer term, :math:`\bar{\textbf{u}} \cdot \nabla T`

    The weak form is

    .. math::
        \int_\Omega \bar{\textbf{u}} \cdot \boldsymbol{\psi} \cdot \nabla T  dx
        = - \int_\Omega \nabla_h \cdot (\bar{\textbf{u}} \boldsymbol{\psi}) \cdot T dx
        + \int_\Gamma \text{avg}(T) \cdot \text{jump}(\boldsymbol{\psi}
        (\bar{\textbf{u}}\cdot\textbf{n})) dS

    where the right hand side has been integrated by parts;
    :math:`\textbf{n}` is the unit normal of
    the element interfaces, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_2d') is None:
            return 0
        elev = fields_old['elev_2d']
        uv = fields_old.get('tracer_advective_velocity')
        conservative = self.options.use_tracer_conservative_form
        if uv is None:
            uv = fields_old['uv_2d']

        uv_p1 = fields_old.get('uv_p1')
        uv_mag = fields_old.get('uv_mag')
        # FIXME is this an option?
        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        f = 0
        if conservative:
            H = self.get_total_depth(fields["elev_2d"])
            f += -(Dx(self.test, 0) * H* uv[0] * solution
                   + Dx(self.test, 1) * H* uv[1] * solution) * self.dx
        else:
            f += -(Dx(uv[0] * self.test, 0) * solution
                   + Dx(uv[1] * self.test, 1) * solution) * self.dx

        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0]
                     + uv_av[1]*self.normal('-')[1])
            s = 0.5*(sign(un_av) + 1.0)

            if conservative:
                Huvc_up = (H*uv*solution)('-')*s + (H*uv*solution)('+')*(1-s)
                f += dot(Huvc_up, jump(self.test, self.normal)) * self.dS
            else:
                c_up = solution('-')*s + solution('+')*(1-s)
                f += c_up*(jump(self.test, uv[0] * self.normal[0])
                           + jump(self.test, uv[1] * self.normal[1])) * self.dS
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                if conservative:
                    raise NotImplemented("Combination of Lax-Friedrichs with conservative form not implemented.")
                if uv_p1 is not None:
                    gamma = 0.5 * abs((avg(uv_p1)[0]*self.normal('-')[0]
                                     + avg(uv_p1)[1]*self.normal('-')[1]))*lax_friedrichs_factor
                elif uv_mag is not None:
                    gamma = 0.5*avg(uv_mag)*lax_friedrichs_factor
                else:
                    gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                    c_in = solution
                    if funcs is not None and 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        if conservative:
                            H_ext = self.get_total_depth(eta_ext)
                            uv_ext = H_ext * uv_ext
                            uv_av = 0.5*(H * uv + uv_ext)
                        else:
                            uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        f += c_up*(uv_av[0]*self.normal[0]
                                   + uv_av[1]*self.normal[1])*self.test*ds_bnd
                    elif conservative:
                        f += c_in * H * (uv[0]*self.normal[0]
                                     + uv[1]*self.normal[1])*self.test*ds_bnd
                    else:
                        f += c_in * (uv[0]*self.normal[0]
                                     + uv[1]*self.normal[1])*self.test*ds_bnd

        return -f


class HorizontalDiffusionTerm(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        -\int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        =& \int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h T) dS
        - \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(T \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(T \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS \\
        &- \int_\Gamma \mu_h (\nabla_h \phi) \cdot \textbf{n}_h ds

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029

    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('diffusivity_h') is None:
            return 0
        diffusivity_h = fields_old['diffusivity_h']
        if self.options.use_tracer_conservative_form:
            H = self.get_total_depth(fields["elev_2d"])
            diffusivity_h = H * diffusivity_h
        diff_tensor = as_matrix([[diffusivity_h, 0, ],
                                 [0, diffusivity_h, ]])
        grad_test = grad(self.test)
        diff_flux = dot(diff_tensor, grad(solution))

        f = 0
        f += inner(grad_test, diff_flux)*self.dx

        if self.horizontal_dg:
            alpha = self.sipg_parameter
            assert alpha is not None
            sigma = avg(alpha / self.cellsize)
            ds_interior = self.dS
            f += sigma*inner(jump(self.test, self.normal),
                             dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
            f += -inner(avg(dot(diff_tensor, grad(self.test))),
                        jump(solution, self.normal))*ds_interior
            f += -inner(jump(self.test, self.normal),
                        avg(dot(diff_tensor, grad(solution))))*ds_interior

        if bnd_conditions is not None:
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker), degree=self.quad_degree)
                c_in = solution
                elev = fields_old['elev_2d']
                uv = fields_old.get('tracer_advective_velocity')
                if uv is None:
                    uv = fields_old['uv_2d']
                if funcs is not None:
                    if 'value' in funcs:
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                        s = 0.5*(sign(un_av) + 1.0)
                        c_up = c_in*s + c_ext*(1-s)
                        diff_flux_up = dot(diff_tensor, grad(c_up))
                        f += -self.test*dot(diff_flux_up, self.normal)*ds_bnd
                    elif 'diff_flux' in funcs:
                        f += -self.test*funcs['diff_flux']*ds_bnd
                    else:
                        # Open boundary case
                        f += -self.test*dot(diff_flux, self.normal)*ds_bnd

        return -f


class SourceTerm(TracerTerm):
    r"""
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
            if self.options.use_tracer_conservative_form:
                H = self.get_total_depth(fields["elev_2d"])
                f += -inner(source * H, self.test) * self.dx
            else:
                f += -inner(source, self.test)*self.dx
        return -f


class MassTerm(TracerTerm):
    r"""
    Mass term for 2d tracer equation.
    
    In the default form (non-conservative) this is just the standard mass term

    .. math::
        F_s = \int_\Omega T \phi

    In the conservative form we multiply by a depth H

    .. math::
        F_s = \int_\Omega T H \phi

    """
    def residual(self, solution, fields):
        """NOTE: the arguments to this method are not consistent
        with the other terms."""
        if self.options.use_tracer_conservative_form:
            H = self.get_total_depth(fields["elev_2d"])
            return solution * H * self.test * self.dx
        else:
            return solution * self.test * self.dx


class TracerEquation2D(Equation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space,
                 bathymetry=None,
                 use_lax_friedrichs=False,
                 sipg_parameter=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`

        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(TracerEquation2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs, sipg_parameter)
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')
        self._mass_term = MassTerm(*args)

    def mass_term(self, solution, fields):
        """
        Mass term. For the conservative form fields needs to contain elev_2d at the old or new time level."""
        return self._mass_term.residual(solution, fields)
