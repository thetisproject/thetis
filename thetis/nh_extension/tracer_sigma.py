r"""
3D advection diffusion equation for tracers.

The advection-diffusion equation of tracer :math:`T` in conservative form reads

.. math::
   \frac{\partial T}{\partial t}
    + \nabla_h \cdot (\textbf{u} T)
    + \frac{\partial (w T)}{\partial z}
    = \nabla_h \cdot (\mu_h \nabla_h T)
    + \frac{\partial}{\partial z} \Big(\mu \frac{T}{\partial z}\Big)
   :label: tracer_eq

where :math:`\nabla_h` denotes horizontal gradient, :math:`\textbf{u}` and
:math:`w` are the horizontal and vertical velocities, respectively, and
:math:`\mu_h` and :math:`\mu` denote horizontal and vertical diffusivity.
"""
from __future__ import absolute_import
from .utility_nh import *
from thetis.equation import Term, Equation

__all__ = [
    'TracerEquation',
    'TracerTerm',
    'HorizontalAdvectionTerm',
    'VerticalAdvectionTerm',
    'HorizontalDiffusionTerm',
    'VerticalDiffusionTerm',
    'SourceTerm',
]


class TracerTerm(Term):
    """
    Generic tracer term that provides commonly used members and mapping for
    boundary functions.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True, use_lax_friedrichs=True,
                 sipg_parameter=Constant(10.0), sipg_parameter_vertical=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(TracerTerm, self).__init__(function_space)
        self.bathymetry = bathymetry
        self.h_elem_size = h_elem_size
        self.v_elem_size = v_elem_size
        continuity = element_continuity(self.function_space.ufl_element())
        self.horizontal_dg = continuity.horizontal == 'dg'
        self.vertical_dg = continuity.vertical == 'dg'
        self.use_symmetric_surf_bnd = use_symmetric_surf_bnd
        self.use_lax_friedrichs = use_lax_friedrichs
        self.sipg_parameter = sipg_parameter
        self.sipg_parameter_vertical = sipg_parameter_vertical

        # define measures with a reasonable quadrature degree
        p, q = self.function_space.ufl_element().degree()
        self.quad_degree = (2*p + 1, 2*q + 1)
        self.dx = dx(degree=self.quad_degree)
        self.dS_h = dS_h(degree=self.quad_degree)
        self.dS_v = dS_v(degree=self.quad_degree)
        self.ds = ds(degree=self.quad_degree)
        self.ds_surf = ds_surf(degree=self.quad_degree)
        self.ds_bottom = ds_bottom(degree=self.quad_degree)

        self.horizontal_domain_is_2d = self.mesh.geometric_dimension() == 3

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


class HorizontalAdvectionTerm(TracerTerm):
    r"""
    Horizontal advection term :math:`\nabla_h \cdot (\textbf{u} T)`

    The weak formulation reads

    .. math::
        \int_\Omega \nabla_h \cdot (\textbf{u} T) \phi dx
            = -\int_\Omega T\textbf{u} \cdot \nabla_h \phi dx
            + \int_{\mathcal{I}_h\cup\mathcal{I}_v}
                T^{\text{up}} \text{avg}(\textbf{u}) \cdot
                \text{jump}(\phi \textbf{n}_h) dS

    where the right hand side has been integrated by parts;
    :math:`\mathcal{I}_h,\mathcal{I}_v` denote the set of horizontal and
    vertical facets,
    :math:`\textbf{n}_h` is the horizontal projection of the unit normal vector,
    :math:`T^{\text{up}}` is the upwind value, and :math:`\text{jump}` and
    :math:`\text{avg}` denote the jump and average operators across the
    interface.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        if fields_old.get('uv_3d') is None:
            return 0
        elev = fields_old['elev_3d']
        uv = fields_old['uv_3d']

        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')

        if self.horizontal_domain_is_2d:
           # f = -solution*inner(uv, nabla_grad(self.test))*self.dx
            f = -solution*(Dx(uv[0]*self.test, 0) + Dx(uv[1]*self.test, 1))*self.dx # non-conservative form
            if self.horizontal_dg:
                # add interface term
                uv_av = avg(uv)
                un_av = (uv_av[0]*self.normal('-')[0] +
                         uv_av[1]*self.normal('-')[1])
                s = 0.5*(sign(un_av) + 1.0)
                c_up = solution('-')*s + solution('+')*(1-s)
                f += c_up*(uv_av[0]*jump(self.test, self.normal[0]) +
                           uv_av[1]*jump(self.test, self.normal[1]))*(self.dS_v + self.dS_h)
               # f += c_up*(jump(self.test, uv[0]*self.normal[0]) +
               #            jump(self.test, uv[1]*self.normal[1]))*(self.dS_v + self.dS_h) # non-conservative form
                # Lax-Friedrichs stabilization
                if self.use_lax_friedrichs:
                    gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                    f += gamma*dot(jump(self.test), jump(solution))*(self.dS_v + self.dS_h)
                if bnd_conditions is not None:
                    for bnd_marker in self.boundary_markers:
                        funcs = bnd_conditions.get(bnd_marker)
                        ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                        if funcs is None:
                            continue
                        else:
                            c_in = solution
                            c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                            # add interior tracer flux
                            f += c_in*(uv[0]*self.normal[0]
                                       + uv[1]*self.normal[1])*self.test*ds_bnd
                            # add boundary contribution if inflow
                            uv_av = 0.5*(uv + uv_ext)
                            un_av = self.normal[0]*uv_av[0] + self.normal[1]*uv_av[1]
                            s = 0.5*(sign(un_av) + 1.0)
                            f += (1-s)*(c_ext - c_in)*un_av*self.test*ds_bnd
            if self.use_symmetric_surf_bnd:
                f += solution*(uv[0]*self.normal[0] + uv[1]*self.normal[1])*self.test*ds_surf
            return -f

        # below for horizontal 1D domain
       # f = -solution*inner(uv[0], Dx(self.test, 0))*self.dx
        f = -solution*Dx(uv[0]*self.test, 0)*self.dx # non-conservative form
        if self.horizontal_dg:
            # add interface term
            uv_av = avg(uv)
            un_av = (uv_av[0]*self.normal('-')[0])
            s = 0.5*(sign(un_av) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            f += c_up*(uv_av[0]*jump(self.test, self.normal[0]))*(self.dS_v + self.dS_h)
           # f += c_up*(jump(self.test, uv[0]*self.normal[0]))*(self.dS_v + self.dS_h) # non-conservative form
            # Lax-Friedrichs stabilization
            if self.use_lax_friedrichs:
                gamma = 0.5*abs(un_av)*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*(self.dS_v + self.dS_h)
            if bnd_conditions is not None:
                for bnd_marker in self.boundary_markers:
                    funcs = bnd_conditions.get(bnd_marker)
                    ds_bnd = ds_v(int(bnd_marker), degree=self.quad_degree)
                    if funcs is None:
                        continue
                    else:
                        c_in = solution
                        c_ext, uv_ext, eta_ext = self.get_bnd_functions(c_in, uv, elev, bnd_marker, bnd_conditions)
                        # add interior tracer flux
                        f += c_in*(uv[0]*self.normal[0])*self.test*ds_bnd
                        # add boundary contribution if inflow
                        uv_av = 0.5*(uv + uv_ext)
                        un_av = self.normal[0]*uv_av[0]
                        s = 0.5*(sign(un_av) + 1.0)
                        f += (1-s)*(c_ext - c_in)*un_av*self.test*ds_bnd
        if self.use_symmetric_surf_bnd:
            f += solution*(uv[0]*self.normal[0])*self.test*ds_surf
        return -f


class VerticalAdvectionTerm(TracerTerm):
    r"""
    Vertical advection term :math:`\partial (w T)/(\partial z)`

    The weak form reads

    .. math::
        \int_\Omega \frac{\partial (w T)}{\partial z} \phi dx
        = - \int_\Omega T w \frac{\partial \phi}{\partial z} dx
        + \int_{\mathcal{I}_v} T^{\text{up}} \text{avg}(w) \text{jump}(\phi n_z) dS

    where the right hand side has been integrated by parts;
    :math:`\mathcal{I}_v` denotes the set of vertical facets,
    :math:`n_z` is the vertical projection of the unit normal vector,
    :math:`T^{\text{up}}` is the
    upwind value, and :math:`\text{jump}` and :math:`\text{avg}` denote the
    jump and average operators across the interface.

    In the case of ALE moving mesh we substitute :math:`w` with :math:`w - w_m`,
    :math:`w_m` being the mesh velocity.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        lax_friedrichs_factor = fields_old.get('lax_friedrichs_tracer_scaling_factor')
        omega = fields_old.get('omega')
        ###
        vertvelo = omega#sigma_dt + uv_3d[0]*sigma_dx + w[1]/(eta + bath)
        ###

        if self.horizontal_domain_is_2d:
           # f = -solution*vertvelo*Dx(self.test, 2)*self.dx
            f = -solution*Dx(vertvelo*self.test, 2)*self.dx # non-conservative form
            if self.vertical_dg:
                w_av = avg(vertvelo)
                s = 0.5*(sign(w_av*self.normal[2]('-')) + 1.0)
                c_up = solution('-')*s + solution('+')*(1-s)
                f += c_up*w_av*jump(self.test, self.normal[2])*self.dS_h
               # f += c_up*jump(self.test, vertvelo*self.normal[2])*self.dS_h # non-conservative form
                if self.use_lax_friedrichs:
                    # Lax-Friedrichs
                    gamma = 0.5*abs(w_av*self.normal('-')[2])*lax_friedrichs_factor
                    f += gamma*dot(jump(self.test), jump(solution))*self.dS_h
            f += solution*vertvelo*self.normal[2]*self.test*self.ds_surf
            return -f

        # below for horizontal 1D domain
       # f = -solution*vertvelo*Dx(self.test, 1)*self.dx
        f = -solution*Dx(vertvelo*self.test, 1)*self.dx # non-conservative form
        if self.vertical_dg:
            w_av = avg(vertvelo)
            s = 0.5*(sign(w_av*self.normal[1]('-')) + 1.0)
            c_up = solution('-')*s + solution('+')*(1-s)
            f += c_up*w_av*jump(self.test, self.normal[1])*self.dS_h
           # f += c_up*jump(self.test, vertvelo*self.normal[1])*self.dS_h # non-conservative form
            if self.use_lax_friedrichs:
                # Lax-Friedrichs
                gamma = 0.5*abs(w_av*self.normal('-')[1])*lax_friedrichs_factor
                f += gamma*dot(jump(self.test), jump(solution))*self.dS_h
        # NOTE Bottom impermeability condition is naturally satisfied by the definition of w
        # NOTE imex solver fails with this in tracerBox example
        f += solution*vertvelo*self.normal[1]*self.test*self.ds_surf
        return -f


class HorizontalDiffusionTerm(TracerTerm):
    r"""
    Horizontal diffusion term :math:`-\nabla_h \cdot (\mu_h \nabla_h T)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \nabla_h \cdot (\mu_h \nabla_h T) \phi dx
        =& -\int_\Omega \mu_h (\nabla_h \phi) \cdot (\nabla_h T) dx \\
        &+ \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(\phi \textbf{n}_h)
        \cdot \text{avg}(\mu_h \nabla_h T) dS
        + \int_{\mathcal{I}_h\cup\mathcal{I}_v} \text{jump}(T \textbf{n}_h)
        \cdot \text{avg}(\mu_h  \nabla \phi) dS \\
        &- \int_{\mathcal{I}_h\cup\mathcal{I}_v} \sigma \text{avg}(\mu_h) \text{jump}(T \textbf{n}_h) \cdot
            \text{jump}(\phi \textbf{n}_h) dS

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

        if self.horizontal_domain_is_2d:
            diff_tensor = as_matrix([[diffusivity_h, 0, 0],
                                     [0, diffusivity_h, 0],
                                     [0, 0, 0]])
            grad_test = grad(self.test)
            diff_flux = dot(diff_tensor, grad(solution))
            f = inner(grad_test, diff_flux)*self.dx
            if self.horizontal_dg:
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                # Interior Penalty method by
                # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
                # sigma = 3*k_max**2/k_min*p*(p+1)*cot(Theta)
                # k_max/k_min  - max/min diffusivity
                # p            - polynomial degree
                # Theta        - min angle of triangles
                # assuming k_max/k_min=2, Theta=pi/3
                # sigma = 6.93 = 3.5*p*(p+1)
                degree_h, degree_v = self.function_space.ufl_element().degree()
                # TODO compute elemsize as CellVolume/FacetArea
                # h = n.D.n where D = diag(h_h, h_h, h_v)
                elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                              self.normal[1]**2) +
                            self.v_elem_size*self.normal[2]**2)
                sigma = 5.0*degree_h*(degree_h + 1)/elemsize
                if degree_h == 0:
                    sigma = 1.5/elemsize
                alpha = avg(sigma)
                ds_interior = (self.dS_h + self.dS_v)
                f += alpha*inner(jump(self.test, self.normal),
                                 dot(avg(diff_tensor), jump(solution, self.normal)))*ds_interior
                f += -inner(avg(dot(diff_tensor, grad(self.test))),
                            jump(solution, self.normal))*ds_interior
                f += -inner(jump(self.test, self.normal),
                            avg(dot(diff_tensor, grad(solution))))*ds_interior
            # symmetric bottom boundary condition
            # NOTE introduces a flux through the bed - breaks mass conservation
            f += - inner(diff_flux, self.normal)*self.test*self.ds_bottom
            f += - inner(diff_flux, self.normal)*self.test*self.ds_surf
            return -f

        # below for horizontal 1D domain
        grad_test = Dx(self.test, 0)
        diff_flux = dot(diffusivity_h, Dx(solution, 0))
        f = inner(grad_test, diff_flux)*self.dx
        if self.horizontal_dg:
            assert self.h_elem_size is not None, 'h_elem_size must be defined'
            assert self.v_elem_size is not None, 'v_elem_size must be defined'
            # Interior Penalty method by
            # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
            # sigma = 3*k_max**2/k_min*p*(p+1)*cot(Theta)
            # k_max/k_min  - max/min diffusivity
            # p            - polynomial degree
            # Theta        - min angle of triangles
            # assuming k_max/k_min=2, Theta=pi/3
            # sigma = 6.93 = 3.5*p*(p+1)
            degree_h, degree_v = self.function_space.ufl_element().degree()
            # TODO compute elemsize as CellVolume/FacetArea
            # h = n.D.n where D = diag(h_h, h_h, h_v)
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_h*(degree_h + 1)/elemsize
            if degree_h == 0:
                sigma = 1.5/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h + self.dS_v)
            f += alpha*inner(jump(self.test, self.normal[0]),
                             dot(avg(diffusivity_h), jump(solution, self.normal[0])))*ds_interior
            f += -inner(avg(dot(diffusivity_h, Dx(self.test, 0))),
                        jump(solution, self.normal[0]))*ds_interior
            f += -inner(jump(self.test, self.normal[0]),
                        avg(dot(diffusivity_h, Dx(solution, 0))))*ds_interior
        # symmetric bottom boundary condition
        # NOTE introduces a flux through the bed - breaks mass conservation
        f += - inner(diff_flux, self.normal[0])*self.test*self.ds_bottom
        f += - inner(diff_flux, self.normal[0])*self.test*self.ds_surf
        return -f


class VerticalDiffusionTerm(TracerTerm):
    r"""
    Vertical diffusion term :math:`-\frac{\partial}{\partial z} \Big(\mu \frac{T}{\partial z}\Big)`

    Using the symmetric interior penalty method the weak form becomes

    .. math::
        \int_\Omega \frac{\partial}{\partial z} \Big(\mu \frac{T}{\partial z}\Big) \phi dx
        =& -\int_\Omega \mu \frac{\partial T}{\partial z} \frac{\partial \phi}{\partial z} dz \\
        &+ \int_{\mathcal{I}_{h}} \text{jump}(\phi n_z) \text{avg}\Big(\mu \frac{\partial T}{\partial z}\Big) dS
        + \int_{\mathcal{I}_{h}} \text{jump}(T n_z) \text{avg}\Big(\mu \frac{\partial \phi}{\partial z}\Big) dS \\
        &- \int_{\mathcal{I}_{h}} \sigma \text{avg}(\mu) \text{jump}(T n_z) \cdot
            \text{jump}(\phi n_z) dS

    where :math:`\sigma` is a penalty parameter,
    see Epshteyn and Riviere (2007).

    Epshteyn and Riviere (2007). Estimation of penalty parameters for symmetric
    interior penalty Galerkin methods. Journal of Computational and Applied
    Mathematics, 206(2):843-872. http://dx.doi.org/10.1016/j.cam.2006.08.029
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        diffusivity_v = fields_old['diffusivity_v']
        if diffusivity_v is None:
            return 0

        total_h = fields_old.get('elev_3d') + self.bathymetry
        const = 1./total_h**2

        if self.horizontal_domain_is_2d:
            grad_test = Dx(const*self.test, 2)
            diff_flux = dot(diffusivity_v, Dx(solution, 2))
            f = inner(grad_test, diff_flux)*self.dx
            if self.vertical_dg:
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                # Interior Penalty method by
                # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
                degree_h, degree_v = self.function_space.ufl_element().degree()
                # TODO compute elemsize as CellVolume/FacetArea
                # h = n.D.n where D = diag(h_h, h_h, h_v)
                elemsize = (self.h_elem_size*(self.normal[0]**2 +
                                              self.normal[1]**2) +
                            self.v_elem_size*self.normal[2]**2)
                sigma = 5.0*degree_v*(degree_v + 1)/elemsize
                if degree_v == 0:
                    sigma = 1.0/elemsize
                alpha = avg(sigma)
                ds_interior = (self.dS_h)
                f += alpha*inner(jump(const*self.test, self.normal[2]),
                                 dot(avg(diffusivity_v), jump(solution, self.normal[2])))*ds_interior
                f += -inner(avg(dot(diffusivity_v, Dx(const*self.test, 2))),
                            jump(solution, self.normal[2]))*ds_interior
                f += -inner(jump(const*self.test, self.normal[2]),
                            avg(dot(diffusivity_v, Dx(solution, 2))))*ds_interior
            return -f

        # below for horizontal 1D domain
        grad_test = Dx(const*self.test, 1)
        diff_flux = dot(diffusivity_v, Dx(solution, 1))
        f = inner(grad_test, diff_flux)*self.dx
        if self.vertical_dg:
            assert self.h_elem_size is not None, 'h_elem_size must be defined'
            assert self.v_elem_size is not None, 'v_elem_size must be defined'
            # Interior Penalty method by
            # Epshteyn (2007) doi:10.1016/j.cam.2006.08.029
            degree_h, degree_v = self.function_space.ufl_element().degree()
            # TODO compute elemsize as CellVolume/FacetArea
            # h = n.D.n where D = diag(h_h, h_h, h_v)
            elemsize = (self.h_elem_size*(self.normal[0]**2) +
                        self.v_elem_size*self.normal[1]**2)
            sigma = 5.0*degree_v*(degree_v + 1)/elemsize
            if degree_v == 0:
                sigma = 1.0/elemsize
            alpha = avg(sigma)
            ds_interior = (self.dS_h)
            f += alpha*inner(jump(const*self.test, self.normal[1]),
                             dot(avg(diffusivity_v), jump(solution, self.normal[1])))*ds_interior
            f += -inner(avg(dot(diffusivity_v, Dx(const*self.test, 1))),
                        jump(solution, self.normal[1]))*ds_interior
            f += -inner(jump(const*self.test, self.normal[1]),
                        avg(dot(diffusivity_v, Dx(solution, 1))))*ds_interior

        return -f


class SourceTerm(TracerTerm):
    """
    Generic source term

    The weak form reads

    .. math::
        F_s = \int_\Omega \sigma \phi dx

    where :math:`\sigma` is a user defined scalar :class:`Function`.
    """
    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        f = 0
        source = fields_old.get('source_tracer')
        if source is not None:
            f += inner(source, self.test)*self.dx
        return f


class SigmaDiffusionTerm(TracerTerm):

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        diffusivity_h = fields_old['diffusivity_h']
        diffusivity_v = fields_old['diffusivity_v']
        if diffusivity_h is None and diffusivity_v is None:
            return 0
        if diffusivity_h is None:
            diffusivity_h = Constant(0)
        if diffusivity_v is None:
            diffusivity_v = Constant(0)

        if self.horizontal_domain_is_2d:
            def vis(mom):
                viscosity_h = diffusivity_h
                viscosity_v = diffusivity_v
                F = 0
                F += (
                      viscosity_h*Dx(mom, 0)*Dx(self.test, 0)*self.dx
                      + viscosity_h*Dx(mom, 1)*Dx(self.test, 1)*self.dx
                      + viscosity_v*Dx(mom, 2)*Dx(sigma_dxyz*self.test, 2)*self.dx
                     )
                F += (
                      -jump(self.normal[0], self.test)*avg(viscosity_h*Dx(mom, 0))*ds_interior
                      - jump(self.normal[1], self.test)*avg(viscosity_h*Dx(mom, 1))*ds_interior
                      - avg(sigma_dxyz)*jump(self.normal[2], self.test)*avg(Dx(mom, 2))*ds_interior
                     )
                # SIPG terms
                F += (
                      alpha_h*avg(viscosity_h)*jump(mom, self.normal[0])*jump(self.test, self.normal[0])*ds_interior
                      + alpha_h*avg(viscosity_h)*jump(mom, self.normal[1])*jump(self.test, self.normal[1])*ds_interior
                      + avg(sigma_dxyz)*alpha_v*jump(mom, self.normal[2])*jump(self.test, self.normal[2])*ds_interior
                     )
                F += (
                      -jump(mom, self.normal[0])*avg(viscosity_h*Dx(self.test, 0))*ds_interior
                      - jump(mom, self.normal[1])*avg(viscosity_h*Dx(self.test, 1))*ds_interior
                      - avg(sigma_dxyz)*jump(mom, self.normal[2])*avg(Dx(self.test, 2))*ds_interior
                     )

                # terms from sigma transformation
                F += (
                      2*viscosity_h*Dx(mom, 2)*Dx(self.test*sigma_dx, 0)*self.dx
                      + 2*viscosity_h*Dx(mom, 2)*Dx(self.test*sigma_dy, 1)*self.dx
                     )
                F += (
                      -2*avg(sigma_dx*viscosity_h*Dx(mom, 2))*jump(self.test, self.normal[0])*ds_interior
                      - 2*avg(sigma_dy*viscosity_h*Dx(mom, 2))*jump(self.test, self.normal[1])*ds_interior
                     )
                F += -viscosity_h*(Dx(sigma_dx, 0) + Dx(sigma_dy, 1))*Dx(mom, 2)*self.test*self.dx #TODO note here no integration by parts
                # SIPG terms
                F += (
                      2*avg(sigma_dx)*alpha_h*avg(viscosity_h)*jump(mom, self.normal[2])*jump(self.test, self.normal[0])*ds_interior
                      + 2*avg(sigma_dy)*alpha_h*avg(viscosity_h)*jump(mom, self.normal[2])*jump(self.test, self.normal[1])*ds_interior
                     )
                F += (
                      -2*avg(sigma_dx)*jump(mom, self.normal[2])*avg(viscosity_h*Dx(self.test, 0))*ds_interior
                      - 2*avg(sigma_dy)*jump(mom, self.normal[2])*avg(viscosity_h*Dx(self.test, 1))*ds_interior
                     )

                # symmetric bottom boundary condition
                # NOTE introduces a flux through the bed - breaks mass conservation
                F += (
                      - viscosity_h*Dx(mom, 0)*self.normal[0]*self.test*(self.ds_bottom + self.ds_surf)
                      - viscosity_h*Dx(mom, 1)*self.normal[1]*self.test*(self.ds_bottom + self.ds_surf)
                     ) # TODO add more?

                return F

            if True:
                sigma_dx = fields.get('sigma_dx')
                sigma_dy = fields.get('sigma_dy')
               # sigma_dxyz = fields.get('sigma_dxyz')
                h_total = self.bathymetry + fields.get('elev_3d')
                sigma_dz = 1./h_total
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                elemsize = (self.h_elem_size*(self.normal[0]**2 + self.normal[1]**2)
                            + self.v_elem_size*self.normal[2]**2)
                assert self.sipg_parameter is not None and self.sipg_parameter_vertical is not None
                alpha_h = avg(self.sipg_parameter/elemsize)
                alpha_v = avg(self.sipg_parameter_vertical/elemsize)
                ds_interior = (self.dS_h + self.dS_v)
                sigma_dxyz = diffusivity_h*(sigma_dx**2 + sigma_dy**2) + diffusivity_v*sigma_dz**2
                f = vis(solution)

        else:
            def vis(mom):
                viscosity_h = diffusivity_h
                viscosity_v = diffusivity_v
                F = 0
                F += (
                      viscosity_h*Dx(mom, 0)*Dx(self.test, 0)*self.dx
                      + viscosity_v*Dx(mom, 1)*Dx(sigma_dxyz*self.test, 1)*self.dx
                     )
                F += (
                      -jump(self.normal[0], self.test)*avg(viscosity_h*Dx(mom, 0))*ds_interior
                      - avg(sigma_dxyz)*jump(self.normal[1], self.test)*avg(Dx(mom, 1))*ds_interior
                     )
                # SIPG terms
                F += (
                      alpha_h*avg(viscosity_h)*jump(mom, self.normal[0])*jump(self.test, self.normal[0])*ds_interior
                      + avg(sigma_dxyz)*alpha_v*jump(mom, self.normal[1])*jump(self.test, self.normal[1])*ds_interior
                     )
                F += (
                      -jump(mom, self.normal[0])*avg(viscosity_h*Dx(self.test, 0))*ds_interior
                      - avg(sigma_dxyz)*jump(mom, self.normal[1])*avg(Dx(self.test, 1))*ds_interior
                     )

                # terms from sigma transformation
                F += (
                      2*viscosity_h*Dx(mom, 1)*Dx(self.test*sigma_dx, 0)*self.dx
                     )
                F += (
                      -2*avg(sigma_dx*viscosity_h*Dx(mom, 1))*jump(self.test, self.normal[0])*ds_interior
                     )
                F += -viscosity_h*(Dx(sigma_dx, 0))*Dx(mom, 1)*self.test*self.dx #TODO note here no integration by parts
                # SIPG terms
                F += (
                      2*avg(sigma_dx)*alpha_h*avg(viscosity_h)*jump(mom, self.normal[1])*jump(self.test, self.normal[0])*ds_interior
                     )
                F += (
                      -2*avg(sigma_dx)*jump(mom, self.normal[1])*avg(viscosity_h*Dx(self.test, 0))*ds_interior
                     )

                # symmetric bottom boundary condition
                # NOTE introduces a flux through the bed - breaks mass conservation
                F += (
                      - viscosity_h*Dx(mom, 0)*self.normal[0]*self.test*(self.ds_bottom + self.ds_surf)
                     ) # TODO add more?

                return F

            if True:
                sigma_dx = fields.get('sigma_dx')
               # sigma_dxyz = fields.get('sigma_dxyz')
                h_total = self.bathymetry + fields.get('elev_3d')
                sigma_dz = 1./h_total
                assert self.h_elem_size is not None, 'h_elem_size must be defined'
                assert self.v_elem_size is not None, 'v_elem_size must be defined'
                elemsize = (self.h_elem_size*(self.normal[0]**2)
                            + self.v_elem_size*self.normal[1]**2)
                assert self.sipg_parameter is not None and self.sipg_parameter_vertical is not None
                alpha_h = avg(self.sipg_parameter/elemsize)
                alpha_v = avg(self.sipg_parameter_vertical/elemsize)
                ds_interior = (self.dS_h + self.dS_v)
                sigma_dxyz = diffusivity_h*(sigma_dx**2) + diffusivity_v*sigma_dz**2
                f = vis(solution)

        return -f


class TracerEquation(Equation):
    """
    3D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 use_symmetric_surf_bnd=True, use_lax_friedrichs=True,
                 sipg_parameter=Constant(10.0), sipg_parameter_vertical=Constant(10.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(TracerEquation, self).__init__(function_space)

        args = (function_space, bathymetry,
                v_elem_size, h_elem_size, use_symmetric_surf_bnd, use_lax_friedrichs)
        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(VerticalAdvectionTerm(*args), 'explicit')
       # self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
       # self.add_term(VerticalDiffusionTerm(*args), 'explicit')
        self.add_term(SourceTerm(*args), 'source')

        self.add_term(SigmaDiffusionTerm(*args), 'explicit') # account for sigma transformation

