"""
Depth averaged shallow water equations

TODO: add documentation

Boundary conditions are set with ShallowWaterEquations.bnd_functions dict.
For example to assign elevation and volume flux for boundary 1:
sw.bnd_functions[1] = {'elev':myfunc1, 'flux':myfunc2}
where myfunc1 and myfunc2 are Functions in the appropriate function space.

Supported boundary conditions are:

 - 'elev': elevation only (usually unstable)
 - 'uv': 2d velocity vector (in model coordinates)
 - 'un': normal velocity (scalar, positive out of domain)
 - 'flux': normal volume flux (scalar, positive out of domain)
 - 'elev' and 'uv': water elevation and uv vector
 - 'elev' and 'un': water elevation and normal velocity (scalar)
 - 'elev' and 'flux': water elevation and normal flux (scalar)

Tuomas Karna 2015-02-23
"""
from utility import *
from equation import Term, Equation

g_grav = physical_constants['g_grav']
rho_0 = physical_constants['rho0']


class ShallowWaterMomentumTerm(Term):
    """
    Generic term in the shallow water momentum equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, u_test, u_space, eta_space,
                 bathymetry=None,
                 nonlin=True,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        super(ShallowWaterMomentumTerm, self).__init__(u_space)

        self.bathymetry = bathymetry
        self.nonlin = nonlin
        self.include_grad_div_viscosity_term = include_grad_div_viscosity_term
        self.include_grad_depth_viscosity_term = include_grad_depth_viscosity_term

        self.u_test = u_test
        self.u_space = u_space
        self.eta_space = eta_space

        self.u_is_dg = element_continuity(self.u_space.fiat_element).dg
        self.eta_is_dg = element_continuity(self.eta_space.fiat_element).dg
        self.u_is_hdiv = self.u_space.ufl_element().family() == 'Raviart-Thomas'

        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)

    def get_bnd_functions(self, eta_in, uv_in, bnd_id, bnd_conditions):
        """
        Returns external values of elev and uv for all supported
        boundary conditions.

        volume flux (flux) and normal velocity (un) are defined positive out of
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

    def get_total_depth(self, eta):
        """
        Returns total water column depth
        """
        if self.nonlin:
            total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry
        return total_h


class ShallowWaterContinuityTerm(Term):
    """
    Generic term in the depth-integrated continuity equation that provides commonly used
    members and mapping for boundary functions.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry=None,
                 nonlin=True):
        super(ShallowWaterContinuityTerm, self).__init__(eta_space)

        self.bathymetry = bathymetry
        self.nonlin = nonlin

        # for mixed function space
        self.eta_test = eta_test
        self.eta_space = eta_space
        self.u_space = u_space
        self.u_is_dg = element_continuity(self.u_space.fiat_element).dg
        self.eta_is_dg = element_continuity(self.eta_space.fiat_element).dg
        self.u_is_hdiv = self.u_space.ufl_element().family() == 'Raviart-Thomas'

        # mesh dependent variables
        self.cellsize = CellSize(self.mesh)

    def get_bnd_functions(self, eta_in, uv_in, bnd_id, bnd_conditions):
        """
        Returns external values of elev and uv for all supported
        boundary conditions.

        volume flux (flux) and normal velocity (un) are defined positive out of
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

    def get_total_depth(self, eta):
        """
        Returns total water column depth
        """
        if self.nonlin:
            total_h = self.bathymetry + eta
        else:
            total_h = self.bathymetry
        return total_h


class ExternalPressureGradientTerm(ShallowWaterMomentumTerm):
    """
    External pressure gradient term
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)

        head = eta

        grad_eta_by_parts = self.eta_is_dg

        if grad_eta_by_parts:
            f = -g_grav*head*nabla_div(self.u_test)*dx
            if uv is not None:
                head_star = avg(head) + 0.5*sqrt(avg(total_h)/g_grav)*jump(uv, self.normal)
            else:
                head_star = avg(head)
            f += g_grav*head_star*jump(self.u_test, self.normal)*dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*eta_rie*dot(self.u_test, self.normal)*ds_bnd
                if funcs is None or 'symm' in funcs:
                    # assume land boundary
                    # impermeability implies external un=0
                    un_jump = inner(uv, self.normal)
                    h = self.bathymetry
                    head_rie = head + sqrt(h/g_grav)*un_jump
                    f += g_grav*head_rie*dot(self.u_test, self.normal)*ds_bnd
        else:
            f = g_grav*inner(grad(head), self.u_test) * dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(head, uv, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    un_jump = inner(uv - uv_ext, self.normal)
                    eta_rie = 0.5*(head + eta_ext) + sqrt(total_h/g_grav)*un_jump
                    f += g_grav*(eta_rie-head)*dot(self.u_test, self.normal)*ds_bnd
        return -f


class HUDivTerm(ShallowWaterContinuityTerm):
    """
    Divergence of Hu
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)

        hu_by_parts = self.u_is_dg or self.u_is_hdiv

        if hu_by_parts:
            f = -inner(grad(self.eta_test), total_h*uv)*dx
            if self.eta_is_dg:
                h = avg(total_h)
                uv_rie = avg(uv) + sqrt(g_grav/h)*jump(eta, self.normal)
                hu_star = h*uv_rie
                f += inner(jump(self.eta_test, self.normal), hu_star)*dS
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    h_av = self.bathymetry + 0.5*(eta_old + eta_ext_old)
                    eta_jump = eta - eta_ext
                    un_rie = 0.5*inner(uv + uv_ext, self.normal) + sqrt(g_grav/h_av)*eta_jump
                    un_jump = inner(uv_old - uv_ext_old, self.normal)
                    eta_rie = 0.5*(eta_old + eta_ext_old) + sqrt(h_av/g_grav)*un_jump
                    h_rie = self.bathymetry + eta_rie
                    f += h_rie*un_rie*self.eta_test*ds_bnd
        else:
            f = div(total_h*uv)*self.eta_test*dx
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is None or 'un' in funcs:
                    f += -total_h*dot(uv, self.normal)*self.eta_test*ds_bnd
        return -f


class HorizontalAdvectionTerm(ShallowWaterMomentumTerm):
    """
    Horizontal advection of momentum
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        uv_lax_friedrichs = fields_old.get('uv_lax_friedrichs')

        if not self.nonlin:
            return 0

        horiz_advection_by_parts = True

        if horiz_advection_by_parts:
            # f = -inner(nabla_div(outer(uv, self.u_test)), uv)
            f = -(Dx(uv_old[0]*self.u_test[0], 0)*uv[0] +
                  Dx(uv_old[0]*self.u_test[1], 0)*uv[1] +
                  Dx(uv_old[1]*self.u_test[0], 1)*uv[0] +
                  Dx(uv_old[1]*self.u_test[1], 1)*uv[1])*dx
            if self.u_is_dg:
                un_av = dot(avg(uv_old), self.normal('-'))
                # NOTE solver can stagnate
                # s = 0.5*(sign(un_av) + 1.0)
                # NOTE smooth sign change between [-0.02, 0.02], slow
                # s = 0.5*tanh(100.0*un_av) + 0.5
                # uv_up = uv('-')*s + uv('+')*(1-s)
                # NOTE mean flux
                uv_up = avg(uv)
                f += (uv_up[0]*jump(self.u_test[0], uv_old[0]*self.normal[0]) +
                      uv_up[1]*jump(self.u_test[1], uv_old[0]*self.normal[0]) +
                      uv_up[0]*jump(self.u_test[0], uv_old[1]*self.normal[1]) +
                      uv_up[1]*jump(self.u_test[1], uv_old[1]*self.normal[1]))*dS
                # Lax-Friedrichs stabilization
                if uv_lax_friedrichs is not None:
                    gamma = 0.5*abs(un_av)*uv_lax_friedrichs
                    f += gamma*dot(jump(self.u_test), jump(uv))*dS
                    for bnd_marker in self.boundary_markers:
                        funcs = bnd_conditions.get(bnd_marker)
                        ds_bnd = ds(int(bnd_marker))
                        if funcs is None:
                            # impose impermeability with mirror velocity
                            n = self.normal
                            uv_ext = uv - 2*dot(uv, n)*n
                            gamma = 0.5*abs(dot(uv_old, n))*uv_lax_friedrichs
                            f += gamma*dot(self.u_test, uv-uv_ext)*ds_bnd
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is not None:
                    eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                    eta_ext_old, uv_ext_old = self.get_bnd_functions(eta_old, uv_old, bnd_marker, bnd_conditions)
                    # Compute linear riemann solution with eta, eta_ext, uv, uv_ext
                    eta_jump = eta_old - eta_ext_old
                    un_rie = 0.5*inner(uv_old + uv_ext_old, self.normal) + sqrt(g_grav/self.bathymetry)*eta_jump
                    uv_av = 0.5*(uv_ext + uv)
                    f += (uv_av[0]*self.u_test[0]*un_rie +
                          uv_av[1]*self.u_test[1]*un_rie)*ds_bnd
        return -f


class HorizontalViscosityTerm(ShallowWaterMomentumTerm):
    """
    Viscosity of momentum
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)

        nu = fields_old.get('viscosity_h')
        if nu is None:
            return 0

        n = self.normal
        h = self.cellsize

        if self.include_grad_div_viscosity_term:
            stress = nu*2.*sym(grad(uv))
            stress_jump = avg(nu)*2.*sym(tensor_jump(uv, n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, n)

        f = inner(grad(self.u_test), stress)*dx

        if self.u_is_dg:
            # from Epshteyn et al. 2007 (http://dx.doi.org/10.1016/j.cam.2006.08.029)
            # the scheme is stable for alpha > 3*X*p*(p+1)*cot(theta), where X is the
            # maximum ratio of viscosity within a triangle, p the degree, and theta
            # with X=2, theta=6: cot(theta)~10, 3*X*cot(theta)~60
            p = self.u_space.ufl_element().degree()
            alpha = 5.*p*(p+1)
            f += (
                + alpha/avg(h)*inner(tensor_jump(self.u_test, n), stress_jump)*dS
                - inner(avg(grad(self.u_test)), stress_jump)*dS
                - inner(tensor_jump(self.u_test, n), avg(stress))*dS
            )

            # Dirichlet bcs only for DG
            for bnd_marker in self.boundary_markers:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = ds(int(bnd_marker))
                if funcs is not None:
                    if 'un' in funcs:
                        delta_uv = (dot(uv, n) - funcs['un'])*n
                    else:
                        eta_ext, uv_ext = self.get_bnd_functions(eta, uv, bnd_marker, bnd_conditions)
                        if uv_ext is uv:
                            continue
                        delta_uv = uv - uv_ext

                    if self.include_grad_div_viscosity_term:
                        stress_jump = nu*2.*sym(outer(delta_uv, n))
                    else:
                        stress_jump = nu*outer(delta_uv, n)

                    f += (
                        alpha/h*inner(outer(self.u_test, n), stress_jump)*ds_bnd
                        - inner(grad(self.u_test), stress_jump)*ds_bnd
                        - inner(outer(self.u_test, n), stress)*ds_bnd
                    )

        if self.include_grad_depth_viscosity_term:
            f += -dot(self.u_test, dot(grad(total_h)/total_h, stress))*dx

        return -f


class CoriolisTerm(ShallowWaterMomentumTerm):
    """
    Coriolis term
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        coriolis = fields_old.get('coriolis')
        f = 0
        if coriolis is not None:
            f += coriolis*(-uv[1]*self.u_test[0] + uv[0]*self.u_test[1])*dx
        return -f


class WindStressTerm(ShallowWaterMomentumTerm):
    """
    Wind stress
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        wind_stress = fields_old.get('wind_stress')
        total_h = self.get_total_depth(eta_old)
        f = 0
        if wind_stress is not None:
            f += -dot(wind_stress, self.u_test)/total_h/rho_0*dx
        return -f


class QuadraticDragTerm(ShallowWaterMomentumTerm):
    """
    Quadratic Manning bottom friction term
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)
        mu_manning = fields_old.get('mu_manning')
        C_D = fields_old.get('quadratic_drag')
        f = 0
        if mu_manning is not None:
            if C_D is not None:
                raise Exception('Cannot set both dimensionless and Manning drag parameter')
            C_D = g_grav * mu_manning**2 / total_h**(1./3.)

        if C_D is not None:
            f += C_D * sqrt(dot(uv_old, uv_old)) * inner(self.u_test, uv) / total_h * dx
        return -f


class LinearDragTerm(ShallowWaterMomentumTerm):
    """
    Linear bottom friction term
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        linear_drag = fields_old.get('linear_drag')
        f = 0
        if linear_drag is not None:
            bottom_fri = linear_drag*inner(self.u_test, uv)*dx
            f += bottom_fri
        return -f


class BottomDrag3DTerm(ShallowWaterMomentumTerm):
    """
    Bottom drag term consistent with 3D model
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)
        bottom_drag = fields_old.get('bottom_drag')
        uv_bottom = fields_old.get('uv_bottom')
        f = 0
        if bottom_drag is not None and uv_bottom is not None:
            uvb_mag = sqrt(uv_bottom[0]**2 + uv_bottom[1]**2)
            stress = bottom_drag*uvb_mag*uv_bottom/total_h
            bot_friction = dot(stress, self.u_test)*dx
            f += bot_friction
        return -f


class InternalPressureGradientTerm(ShallowWaterMomentumTerm):
    """
    Internal pressure gradient term
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        baroc_head = fields_old.get('baroc_head')

        if baroc_head is None:
            return 0

        f = 0
        f = -g_grav*baroc_head*nabla_div(self.u_test)*dx
        head_star = avg(baroc_head)
        f += g_grav*head_star*jump(self.u_test, self.normal)*dS
        for bnd_marker in self.boundary_markers:
            ds_bnd = ds(int(bnd_marker))
            # use internal value
            head_rie = baroc_head
            f += g_grav*head_rie*dot(self.u_test, self.normal)*ds_bnd
        return -f


class MomentumSourceTerm(ShallowWaterMomentumTerm):
    """
    Generic source term in the shallow water momentum equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        uv_source = fields_old.get('uv_source')

        if uv_source is not None:
            f += -inner(uv_source, self.u_test)*dx
        return -f


class ContinuitySourceTerm(ShallowWaterContinuityTerm):
    """
    Generic source term in the depth-averaged continuity equation
    """
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        f = 0
        elev_source = fields_old.get('elev_source')

        if elev_source is not None:
            f += -inner(elev_source, self.eta_test)*dx
        return -f


class ShallowWaterEquations(Equation):
    """
    2D depth-averaged shallow water equations in non-conservative form.
    """
    def __init__(self, function_space,
                 bathymetry,
                 nonlin=True,
                 include_grad_div_viscosity_term=False,
                 include_grad_depth_viscosity_term=True):
        super(ShallowWaterEquations, self).__init__(function_space)
        self.bathymetry = bathymetry

        u_test, eta_test = TestFunctions(function_space)
        u_space, eta_space = function_space.split()

        # momentum equation terms:
        args = (u_test, u_space, eta_space,
                bathymetry,
                nonlin,
                include_grad_div_viscosity_term,
                include_grad_depth_viscosity_term)

        self.add_term(ExternalPressureGradientTerm(*args), 'implicit')
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalViscosityTerm(*args), 'explicit')
        self.add_term(CoriolisTerm(*args), 'explicit')
        self.add_term(WindStressTerm(*args), 'source')
        self.add_term(QuadraticDragTerm(*args), 'explicit')
        self.add_term(LinearDragTerm(*args), 'explicit')
        self.add_term(BottomDrag3DTerm(*args), 'source')
        self.add_term(InternalPressureGradientTerm(*args), 'source')
        self.add_term(MomentumSourceTerm(*args), 'source')

        # continuity equation terms:
        args = (eta_test, eta_space, u_space, bathymetry, nonlin)
        self.add_term(HUDivTerm(*args), 'implicit')
        self.add_term(ContinuitySourceTerm(*args), 'source')

    def get_time_step(self, u_mag=Constant(0.0)):
        """
        Computes maximum explicit time step from CFL condition.

        Assumes velocity scale U = sqrt(g*H) + u_mag
        where u_mag is estimated advective velocity
        """
        csize = CellSize(self.mesh)
        h = self.bathymetry.function_space()
        h_pos = Function(h, name='bathymetry')
        h_pos.assign(self.bathymetry)
        min_depth = 0.05
        h_pos.dat.data[h_pos.dat.data < min_depth] = min_depth
        uu = TestFunction(h)
        grid_dt = TrialFunction(h)
        res = Function(h)
        a = uu * grid_dt * dx
        l = uu * csize / (sqrt(g_grav * h_pos) + u_mag) * dx
        solve(a == l, res)
        return res

    def get_time_step_advection(self, u_mag=Constant(1.0)):
        """
        Computes maximum explicit time step from CFL condition.

        Assumes velocity scale U = u_mag
        where u_mag is estimated advective velocity
        """
        csize = CellSize(self.mesh)
        h = self.bathymetry.function_space()
        uu = TestFunction(h)
        grid_dt = TrialFunction(h)
        res = Function(h)
        a = uu * grid_dt * dx
        l = uu * csize / u_mag * dx
        solve(a == l, res)
        return res

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the residual by summing up all the terms with the desired label.

        Sign convention: all terms are assumed to be on the left hand side of the equation A + term = 0.

        :arg label: string defining the type of terms to sum up. Currently one of
            'source'|'explicit'|'implicit'|'nonlinear'. Can be a list of multiple labels, or 'all' in which
            case all defined terms are summed.
        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        uv, eta = split(solution)
        uv_old, eta_old = split(solution_old)

        f = 0
        for term in self.select_terms(label):
            f += term.residual(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
        return f


class FreeSurfaceEquation(Equation):
    """
    2D free surface equation.
    """
    def __init__(self, eta_test, eta_space, u_space,
                 bathymetry,
                 nonlin=True):
        super(FreeSurfaceEquation, self).__init__(eta_space)
        self.bathymetry = bathymetry

        # default solver parameters
        self.solver_parameters = {
            'ksp_type': 'gmres',
        }

        # continuity equation terms:
        args = (eta_test, eta_space, u_space, bathymetry, nonlin)
        self.add_term(HUDivTerm(*args), 'explicit')
        self.add_term(ContinuitySourceTerm(*args), 'source')

    def get_time_step(self, u_mag=Constant(0.0)):
        """
        Computes maximum explicit time step from CFL condition.

        Assumes velocity scale U = sqrt(g*H) + u_mag
        where u_mag is estimated advective velocity
        """
        csize = CellSize(self.mesh)
        h = self.bathymetry.function_space()
        h_pos = Function(h, name='bathymetry')
        h_pos.assign(self.bathymetry)
        min_depth = 0.05
        h_pos.dat.data[h_pos.dat.data < min_depth] = min_depth
        uu = TestFunction(h)
        grid_dt = TrialFunction(h)
        res = Function(h)
        a = uu * grid_dt * dx
        l = uu * csize / (sqrt(g_grav * h_pos) + u_mag) * dx
        solve(a == l, res)
        return res

    def get_time_step_advection(self, u_mag=Constant(1.0)):
        """
        Computes maximum explicit time step from CFL condition.

        Assumes velocity scale U = u_mag
        where u_mag is estimated advective velocity
        """
        csize = CellSize(self.mesh)
        h = self.bathymetry.function_space()
        uu = TestFunction(h)
        grid_dt = TrialFunction(h)
        res = Function(h)
        a = uu * grid_dt * dx
        l = uu * csize / u_mag * dx
        solve(a == l, res)
        return res

    def residual(self, label, solution, solution_old, fields, fields_old, bnd_conditions):
        """
        Returns an UFL form of the residual by summing up all the terms with the desired label.

        Sign convention: all terms are assumed to be on the left hand side of the equation A + term = 0.

        :arg label: string defining the type of terms to sum up. Currently one of
            'source'|'explicit'|'implicit'|'nonlinear'. Can be a list of multiple labels, or 'all' in which
            case all defined terms are summed.
        :arg solution: solution :class:`.Function` of the corresponding equation
        :arg solution_old: a time lagged solution :class:`.Function`
        :arg fields: a dictionary that provides all the remaining fields that the term depends on.
            The keys of the dictionary should standard field names in `field_metadata`
        :arg fields_old: Time lagged dictionary of fields
        :arg bnd_conditions: A dictionary describing boundary conditions.
            E.g. {3: {'elev_2d': Constant(1.0)}} replaces elev_2d function by a constant on boundary ID 3.
        """
        uv = fields['uv']
        uv_old = fields_old['uv']
        eta = solution
        eta_old = solution_old

        f = 0
        for term in self.select_terms(label):
            f += term.residual(uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions)
        return f
