from __future__ import absolute_import
from .utility import *

from .tracer_eq_2d import *
from .equation import Equation
from .turbulence import P1Average

class RateOfStrainSolver(object):
    """
    Computes vertical gradient in the weak sense.

    """
    # TODO add weak form of the problem
    def __init__(self, source, solution, solver_parameters=None):
        """
        :arg source: A :class:`Function` or expression to differentiate.
        :arg solution: A :class:`Function` where the solution will be stored.
            Must be in P0 space.
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')

        self.source = source
        self.solution = solution

        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()

        # weak gradient evaluator
        test = TestFunction(self.fs)
        tri = TrialFunction(self.fs)
        normal = FacetNormal(self.mesh)
        a = inner(test, tri)*dx
        uv = self.source
        stress = sym(grad(uv))
        stress_jump = sym(tensor_jump(uv, normal))
        l = inner(test, stress)*dx
        l += -inner(avg(test), stress_jump)*dS
#        l += -inner(avg(test),sym(outer(uv, normal)))*(ds_t + ds_b)
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """Computes the gradient"""
        self.weak_grad_solver.solve()

class ProductionSolver(object):
    r"""
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    .. math::
        M^2 = \left(\frac{\partial u}{\partial z}\right)^2
            + \left(\frac{\partial v}{\partial z}\right)^2
    """
    def __init__(self, uv, production, relaxation=1.0, minval=1e-12):
        """
        :arg uv: horizontal velocity field
        :type uv: :class:`Function`
        :arg m2: :math:`M^2` field
        :type m2: :class:`Function`
        :arg mu: field for x component of :math:`M^2`
        :type mu: :class:`Function`
        :arg mv: field for y component of :math:`M^2`
        :type mv: :class:`Function`
        :arg mu_tmp: temporary field
        :type mu_tmp: :class:`Function`
        :kwarg float relaxation: relaxation coefficient for mixing old and new values
            M2 = relaxation*M2_new + (1-relaxation)*M2_old
        :kwarg float minval: minimum value for :math:`M^2`
        """
        self.production = production
        self.minval = minval
        self.relaxation = relaxation

        self.var_solver = RateOfStrainSolver(uv, self.production)

    def solve(self, init_solve=False):
        """
        Computes buoyancy frequency

        :kwarg bool init_solve: Set to True if solving for the first time, skips
            relaxation
        """
        # TODO init_solve can be omitted with a boolean property
        with timed_stage('shear_freq_solv'):
            self.var_solver.solve()

class RANSTKESourceTerm(TracerTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        
        super(RANSTKESourceTerm, self).__init__(function_space, 
                                            bathymetry, h_elem_size)

        self.test = TestFunction(function_space)

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        nu_t = fields['eddy_viscosity']
        production = fields['production']
        
        f = 2.0*inner(production[0,0]**2+production[0,1]**2
                  +production[1,0]**2+production[1,1]**2, nu_t*self.test)*dx
        
        return f

class RANSTKEDestructionTerm(TracerTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        
        super(RANSTKEDestructionTerm, self).__init__(function_space, 
                                            bathymetry, h_elem_size)

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        gamma = fields['gamma']
        
        f = -inner(gamma*solution,self.test)*dx
        
        return f

class RANSPsiSourceTerm(TracerTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        
        super(RANSPsiSourceTerm, self).__init__(function_space, 
                                            bathymetry, h_elem_size)

        self.test = TestFunction(function_space)

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        nu_t = fields['eddy_viscosity']
        production = fields['production']
        gamma = fields['gamma']

        C_1 = 1.44
        
        f = 2.0*C_1*inner(production[0,0]**2+production[0,1]**2
                  +production[1,0]**2+production[1,1]**2, gamma*nu_t*self.test)*dx
        
        return f

class RANSPsiDestructionTerm(TracerTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def __init__(self, function_space,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        
        super(RANSPsiDestructionTerm, self).__init__(function_space, 
                                            bathymetry, h_elem_size)

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):

        gamma = fields['gamma']

        C_2 = 1.92
        
        f = -C_2*inner(gamma*solution,self.test)*dx
        
        return f

class TurbulenceModel(object):
    """Base class for all vertical turbulence models"""

    @abstractmethod
    def initialize(self):
        """Initialize all turbulence fields"""
        pass

    @abstractmethod
    def preprocess(self, init_solve=False):
        """
        Computes all diagnostic variables that depend on the mean flow model
        variables.

        To be called before updating the turbulence PDEs.
        """
        pass

    @abstractmethod
    def postprocess(self):
        """
        Updates all diagnostic variables that depend on the turbulence state
        variables.

        To be called after updating the turbulence PDEs.
        """
        pass

class RANSModel(TurbulenceModel):

    def __init__(self, solver, fields):

        self.solver = solver
        self.fields = fields
        self.options = solver.options
        self.timesteppers = AttrDict()
        

        P0_2d = self.solver.function_spaces.P0_2d
        P0_2dT = self.solver.function_spaces.P0_2dT
        P1_2d = self.solver.function_spaces.P1_2d

        self.fields.rans_mixing_length = Function(P0_2d, name='rans_mixing_length')
        self.fields.gamma = Function(P0_2d, name='rans_linea')
        self.fields.rans_eddy_viscosity = Function(P1_2d, name='rans_eddy_viscosity')
        self.fields.rans_tke = Function(P0_2d, name='rans_tke')
        self.fields.rans_psi = Function(P0_2d, name='rans_psi')

        self.sqrt_tke = Function(P0_2d, name='eddy_viscosity')
        self.production = Function(P0_2dT, name='production')

        self.solver.eq_rans_tke = RANSTKEEquation2D(P0_2d, self.production,
                                                 bathymetry=self.fields.bathymetry_2d,
                                                 use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
        self.solver.eq_rans_psi = RANSPsiEquation2D(P0_2d, self.production,
                                                 bathymetry=self.fields.bathymetry_2d,
                                                 use_lax_friedrichs=self.options.use_lax_friedrichs_tracer)
        
        self.uv, self.elev_2d = self.fields.solution_2d.split()
        self.eddy_viscosity = Function(self.solver.function_spaces.P0_2d, name='P0 eddy viscosity')
        self.psi = fields.rans_psi
        self.tke = fields.rans_tke

        self.production_solver = ProductionSolver(self.uv, self.production)
        self.p1_averager = P1Average(solver.function_spaces.P0_2d,
                                         solver.function_spaces.P1_2d,
                                         solver.function_spaces.P1DG_2d)

        self.C_mu = Constant(0.09)
        self.nu_0 = Constant(1.0e-6)

    def preprocess(self, init_solve=False):
        self.production_solver.var_solver.source.assign(self.uv)
        self.production_solver.solve()

        l_max = Constant(1.0)

        self.sqrt_tke.assign(conditional(self.tke>0, sqrt(self.tke), Constant(0.0)))

        self.fields.rans_mixing_length.assign(conditional(self.psi*l_max>self.C_mu*self.tke*self.sqrt_tke,
                                                self.C_mu*self.tke*self.sqrt_tke/self.psi,
                                                l_max))
        
        self.eddy_viscosity.assign(conditional(self.nu_0>self.fields.rans_mixing_length*self.sqrt_tke,
                                               self.nu_0,
                                               self.fields.rans_mixing_length*self.sqrt_tke))

        
        
        self.fields.gamma.assign(self.C_mu*self.tke/self.eddy_viscosity)

    def postprocess(self):

        self.p1_averager.apply(self.eddy_viscosity, self.fields.rans_eddy_viscosity)
        
    def _create_integrators(self, integrator, dt, bnd_conditions, solver_parameters):
        
        uv, elev = self.fields.solution_2d.split()
        diffusivity = (self.options.horizontal_diffusivity or Constant(0.0))
        diffusivity = diffusivity + self.solver.fields['rans_eddy_viscosity']
        fields = {'elev_2d': elev,
                  'uv_2d': uv,
                  'diffusivity_h': diffusivity,
                  'source': self.options.tracer_source_2d,
                  'production': self.production,
                  'eddy_viscosity': self.solver.fields.get('rans_eddy_viscosity'),
                  'gamma': self.solver.fields.get('gamma'),
                  'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                  }
        
        self.timesteppers.rans_tke = integrator(self.solver.eq_rans_tke, self.tke, fields, dt,
                                           bnd_conditions  = bnd_conditions['rans_tke'],
                                           solver_parameters=self.options.timestepper_options.solver_parameters_tracer)

        self.timesteppers.rans_psi = integrator(self.solver.eq_rans_tke, self.psi, fields, dt,
                                           bnd_conditions  = bnd_conditions['rans_psi'],
                                           solver_parameters=self.options.timestepper_options.solver_parameters_tracer)

    def initialize(self, rans_tke=Constant(0.0), rans_psi=Constant(0.0), **kwargs):
        self.tke.project(rans_tke)
        self.psi.project(rans_psi)
        self.timesteppers.rans_tke.initialize(self.tke)
        self.timesteppers.rans_psi.initialize(self.psi)

    def advance(self, t, update_forcings=None):
        self.preprocess()
        self.timesteppers.rans_tke.advance(t, update_forcings=update_forcings)
        self.timesteppers.rans_psi.advance(t, update_forcings=update_forcings)
        self.postprocess()

class RANSTKEEquation2D(Equation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space, production,
                 bathymetry=None,
                 use_lax_friedrichs=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`

        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(RANSTKEEquation2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs)
        
        self.source = RANSTKESourceTerm(*args)

        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(RANSTKEDestructionTerm(*args), 'implicit')
        self.add_term(self.source, 'source')

class RANSPsiEquation2D(Equation):
    """
    2D tracer advection-diffusion equation :eq:`tracer_eq` in conservative form
    """
    def __init__(self, function_space, production,
                 bathymetry=None,
                 use_lax_friedrichs=False):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 2D :class:`Function` or :class:`Constant`

        :kwarg bool use_symmetric_surf_bnd: If True, use symmetric surface boundary
            condition in the horizontal advection term
        """
        super(RANSPsiEquation2D, self).__init__(function_space)

        args = (function_space, bathymetry, use_lax_friedrichs)
        
        self.source = RANSPsiSourceTerm(*args)

        self.add_term(HorizontalAdvectionTerm(*args), 'explicit')
        self.add_term(HorizontalDiffusionTerm(*args), 'explicit')
        self.add_term(RANSPsiDestructionTerm(*args), 'implicit')
        self.add_term(self.source, 'source')        
        
