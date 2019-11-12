# VOF MODULE
from __future__ import absolute_import
from thetis.equation import Term, Equation

from ufl.classes import Zero

from . import slope_limiter
from .convection import convection, hric
from .linear_solvers import linear_solver_from_input


# Default values
CONVECTION_SCHEME = 'Upwind' #'Hric'
NUM_SUBCYCLES = 1


# Default values, can be changed in the input file
SOLVER_OPTIONS = {
    'use_ksp': True,
    'petsc_ksp_type': 'gmres',
    'petsc_pc_type': 'asm',
    'petsc_ksp_initial_guess_nonzero': True,
    'petsc_ksp_view': 'DISABLED',
    'inner_iter_rtol': [1e-10] * 3,
    'inner_iter_atol': [1e-15] * 3,
    'inner_iter_max_it': [1000] * 3,
}


class BaseVOF(object):
    """
    This is a base class for the blended algebraic VOF scheme,
    to avoid having duplicates of the methods calculating rho, nu and mu. 
    Any subclass using this class must define the method "get_colour_function(k)" 
    and can also redefine the boolean property that controls the way mu is calculated.
    """

    calculate_mu_directly_from_colour_function = True
    default_polynomial_degree_colour = 0
    _level_set_view = None

    def create_function_space(cls, simulation):
        mesh = simulation.data['mesh']
        cd = simulation.data['constrained_domain']
        Vc_name = simulation.input.get_value(
            'multiphase_solver/function_space_colour', 'Discontinuous Lagrange', 'string'
        )
        Pc = simulation.input.get_value(
            'multiphase_solver/polynomial_degree_colour',
            cls.default_polynomial_degree_colour,
            'int',
        )
        Vc = FunctionSpace(mesh, Vc_name, Pc, constrained_domain=cd)
        simulation.data['Vc'] = Vc
        simulation.ndofs += Vc.dim()

    def set_physical_properties(self, rho0=None, rho1=None, nu0=None, nu1=None, read_input=False):
        """
        Set rho and nu (density and kinematic viscosity) in both domain 0
        and 1. Either specify all of rho0, rho1, nu0 and nu1 or set
        read_input to True which will read from the physical_properties
        section of the simulation input object.
        """
        sim = self.simulation
        if read_input:
            rho0 = sim.input.get_value('physical_properties/rho0', required_type='float')
            rho1 = sim.input.get_value('physical_properties/rho1', required_type='float')
            nu0 = sim.input.get_value('physical_properties/nu0', required_type='float')
            nu1 = sim.input.get_value('physical_properties/nu1', required_type='float')
        self.df_rho0 = Constant(rho0)
        self.df_rho1 = Constant(rho1)
        self.df_nu0 = Constant(nu0)
        self.df_nu1 = Constant(nu1)
        self.df_smallest_rho = self.df_rho0 if rho0 <= rho1 else self.df_rho1

    def set_rho_min(self, rho_min):
        """
        This is used to bring rho_min closer to rho_max for the initial
        linear solver iterations (to speed up convergence)
        """
        self.df_smallest_rho.assign(Constant(rho_min))

    def get_colour_function(self, k):
        """
        Return the colour function on timestep t^{n+k}
        """
        raise NotImplementedError('The get_colour_function method must be implemented by subclass!')

    def get_density(self, k=None, c=None):
        """
        Calculate the blended density function as a weighted sum of
        rho0 and rho1. The colour function is unity when rho=rho0
        and zero when rho=rho1

        Return the function as defined on timestep t^{n+k}
        """
        if c is None:
            assert k is not None
            c = self.get_colour_function(k)
        else:
            assert k is None
        return self.df_rho0 * c + self.df_rho1 * (1 - c)

    def get_laminar_kinematic_viscosity(self, k=None, c=None):
        """
        Calculate the blended kinematic viscosity function as a weighted
        sum of nu0 and nu1. The colour function is unity when nu=nu0 and
        zero when nu=nu1

        Return the function as defined on timestep t^{n+k}
        """
        if c is None:
            assert k is not None
            c = self.get_colour_function(k)
        else:
            assert k is None
        return self.df_nu0 * c + self.df_nu1 * (1 - c)

    def get_laminar_dynamic_viscosity(self, k=None, c=None):
        """
        Calculate the blended dynamic viscosity function as a weighted
        sum of mu0 and mu1. The colour function is unity when mu=mu0 and
        zero when mu=mu1

        Return the function as defined on timestep t^{n+k}
        """
        if self.calculate_mu_directly_from_colour_function:
            if c is None:
                assert k is not None
                c = self.get_colour_function(k)
            else:
                assert k is None
            mu0 = self.df_nu0 * self.df_rho0
            mu1 = self.df_nu1 * self.df_rho1
            return mu0 * c + mu1 * (1 - c)

        else:
            nu = self.get_laminar_kinematic_viscosity(k, c)
            rho = self.get_density(k, c)
            return nu * rho

    def get_density_range(self):
        """
        Return the maximum and minimum densities, rho
        """
        rho0 = self.df_rho0.values()[0]
        rho1 = self.df_rho1.values()[0]
        return min(rho0, rho1), max(rho0, rho1)

    def get_laminar_kinematic_viscosity_range(self):
        """
        Return the maximum and minimum kinematic viscosities, nu
        """
        nu0 = self.df_nu0.values()[0]
        nu1 = self.df_nu1.values()[0]
        return min(nu0, nu1), max(nu0, nu1)

    def get_laminar_dynamic_viscosity_range(self):
        """
        The minimum and maximum laminar dynamic viscosities, mu.

        Mu is either calculated directly from the colour function, in this
        case mu is a linear function, or as a product of nu and rho, where
        it is a quadratic function and can have (in i.e the case of water
        and air) have maximum values() in the middle of the range c ∈ (0, 1)
        """
        rho0 = self.df_rho0.values()[0]
        rho1 = self.df_rho1.values()[0]
        nu0 = self.df_nu0.values()[0]
        nu1 = self.df_nu1.values()[0]

        if self.calculate_mu_directly_from_colour_function:
            mu0 = nu0 * rho0
            mu1 = nu1 * rho1
            return min(mu0, mu1), max(mu0, mu1)
        else:
            c = numpy.linspace(0, 1, 1000)
            nu = nu0 * c + nu1 * (1 - c)
            rho = rho0 * c + rho1 * (1 - c)
            mu = nu * rho
            return mu.min(), mu.max()

    def get_level_set_view(self):
        """
        Get a view of this VOF field as a level set function
        """
        if self._level_set_view is None:
            self._level_set_view = LevelSetView(self.simulation)
            c = self.get_colour_function(0)
            self._level_set_view.set_density_field(c)
        return self._level_set_view


class AlgebraicVofModel(Equation):
    description = 'A blended algebraic VOF scheme implementing HRIC/CICSAM type schemes'

    def __init__(self, function_space, options, 
                 continuous_fields=False, force_bounded=False, 
                 force_sharp=False, calculate_mu_directly_from_colour_function=False):
        """
        A blended algebraic VOF scheme works by using a specific
        convection scheme in the advection of the colour function
        that ensures a sharp interface.

        * The convection scheme should be the name of a convection
          scheme that is tailored for advection of the colour
          function, i.e "HRIC", "MHRIC", "RHRIC" etc,
        * The velocity field should be divergence free

        The colour function is unity when rho=rho0 and nu=nu0 and
        zero when rho=rho1 and nu=nu1.

        :arg function_space: Mixed function space where the solution belongs
        :arg options: :class:`.AttrDict` object containing all model options
        """
        super(AlgebraicVofModel, self).__init__(function_space)

        # Define function space and solution function
        self.degree = function_space.ufl_element().degree()
        self.options = options

        simulation.data['c'] = Function(V)
        simulation.data['cp'] = Function(V)
        simulation.data['cpp'] = Function(V)

        # The projected density and viscosity functions for the new time step can be made continuous
        self.continuous_fields = continuous_fields

        self.force_bounded = force_bounded
        self.force_sharp = force_sharp

        # Calculate mu from rho and nu (i.e mu is quadratic in c) or directly from c (linear in c)
        self.calculate_mu_directly_from_colour_function = calculate_mu_directly_from_colour_function

        # Get the physical properties
        self.set_physical_properties(read_input=True)

        # The convection blending function that counteracts numerical diffusion

        self.convection_scheme = hric.ConvectionSchemeHric2D(simulation, 'c')
        self.need_gradient = hric.ConvectionSchemeHric2D.need_alpha_gradient

        # Create the equations when the simulation starts
        simulation.hooks.add_pre_simulation_hook(
            self.on_simulation_start, 'BlendedAlgebraicVofModel setup equations'
        )

        # Update the rho and nu fields before each time step
        simulation.hooks.add_pre_timestep_hook(
            self.update, 'BlendedAlgebraicVofModel - update colour field'
        )
        simulation.hooks.register_custom_hook_point('MultiPhaseModelUpdated')

        # Linear solver
        # This causes the MPI unit tests to fail in "random" places for some reason
        # Quick fix: lazy loading of the solver
        LAZY_LOAD_SOLVER = True
        if LAZY_LOAD_SOLVER:
            self.solver = None
        else:
            self.solver = linear_solver_from_input(
                self.simulation, 'solver/c', default_parameters=SOLVER_OPTIONS
            )

        # Subcycle the VOF calculation multiple times per Navier-Stokes time step
        self.num_subcycles = 1

        if self.num_subcycles < 1:
            self.num_subcycles = 1

        # Time stepping based on the subcycled values
        if self.num_subcycles == 1:
            self.cp = simulation.data['cp']
            self.cpp = simulation.data['cpp']
        else:
            self.cp = dolfin.Function(V)
            self.cpp = dolfin.Function(V)

        # Slope limiter in case we are using DG1, not DG0
        self.slope_limiter = slope_limiter(simulation, 'c', simulation.data['c'])
        simulation.log.info('    Using slope limiter: %s' % self.slope_limiter.limiter_method)
        self.is_first_timestep = True

    def on_simulation_start(self):
        """
        This runs when the simulation starts. It does not run in __init__
        since the solver needs the density and viscosity we define, and
        we need the velocity that is defined by the solver
        """
        sim = self.simulation
        beta = self.convection_scheme.blending_function

        # The time step (real value to be supplied later)
        self.dt = Constant(sim.dt / self.num_subcycles)

        # Setup the equation to solve
        c = sim.data['c']
        cp = self.cp
        cpp = self.cpp
        dirichlet_bcs = sim.data['dirichlet_bcs'].get('c', [])

        # Use backward Euler (BDF1) for timestep 1
        self.time_coeffs = Constant([1, -1, 0])

        if dolfin.norm(cpp.vector()) > 0 and self.num_subcycles == 1:
            # Use BDF2 from the start
            self.time_coeffs.assign(Constant([3 / 2, -2, 1 / 2]))
            sim.log.info('Using second order timestepping from the start in BlendedAlgebraicVOF')

        # Make sure the convection scheme has something useful in the first iteration
        c.assign(sim.data['cp'])

        if self.num_subcycles > 1:
            cp.assign(sim.data['cp'])

        # Plot density and viscosity
        self.update_plot_fields()

        # Define equation for advection of the colour function
        #    ∂c/∂t +  ∇⋅(c u) = 0
        Vc = sim.data['Vc']
        project_dgt0 = sim.input.get_value('multiphase_solver/project_uconv_dgt0', True, 'bool')
        if self.degree == 0 and project_dgt0:
            self.vel_dgt0_projector = convection.VelocityDGT0Projector(sim, sim.data['u_conv'])
            self.u_conv = self.vel_dgt0_projector.velocity
        else:
            self.u_conv = sim.data['u_conv']
        forcing_zones = sim.data['forcing_zones'].get('c', [])
        self.eq_colour = AdvectionEquation(
            Vc,
            cp,
            cpp,
            self.u_conv,
            beta,
            time_coeffs=self.time_coeffs,
            dirichlet_bcs=dirichlet_bcs,
            forcing_zones=forcing_zones,
            dt=self.dt,
        )

        if self.need_gradient:
            # Reconstruct the gradient from the colour function DG0 field
            self.convection_scheme.initialize_gradient()

        # Notify listeners that the initial values are available
        sim.hooks.run_custom_hook('MultiPhaseModelUpdated')

    def get_colour_function(self, k):
        """
        Return the colour function on timestep t^{n+k}
        """
        if k == 0:
            if self.continuous_fields:
                c = self.continuous_c
            else:
                c = self.simulation.data['c']
        elif k == -1:
            if self.continuous_fields:
                c = self.continuous_c_old
            else:
                c = self.simulation.data['cp']
        elif k == -2:
            if self.continuous_fields:
                c = self.continuous_c_oldold
            else:
                c = self.simulation.data['cpp']

        if self.force_bounded:
            c = dolfin.max_value(dolfin.min_value(c, Constant(1.0)), Constant(0.0))

        if self.force_sharp:
            c = dolfin.conditional(dolfin.ge(c, 0.5), Constant(1.0), Constant(0.0))

        return c

    def update_plot_fields(self):
        """
        These fields are only needed to visualise the rho and nu fields
        in xdmf format for Paraview or similar
        """
        if not self.plot_fields:
            return
        V = self.rho_for_plot.function_space()
        dolfin.project(self.get_density(0), V, function=self.rho_for_plot)
        dolfin.project(self.get_laminar_kinematic_viscosity(0), V, function=self.nu_for_plot)

    def update(self, timestep_number, t, dt):
        """
        Update the VOF field by advecting it for a time dt
        using the given divergence free velocity field
        """
        timer = dolfin.Timer('Ocellaris update VOF')
        sim = self.simulation

        # Get the functions
        c = sim.data['c']
        cp = sim.data['cp']
        cpp = sim.data['cpp']

        # Stop early if the free surface is forced to stay still
        force_static = False
        if force_static:
            c.assign(cp)
            cpp.assign(cp)
            timer.stop()  # Stop timer before hook
            sim.hooks.run_custom_hook('MultiPhaseModelUpdated')
            self.is_first_timestep = False
            return

        if timestep_number != 1:
            # Update the previous values
            cpp.assign(cp)
            cp.assign(c)

            if self.degree == 0:
                self.vel_dgt0_projector.update()

        # Reconstruct the gradients
        if self.need_gradient:
            self.convection_scheme.gradient_reconstructor.reconstruct()

        # Update the convection blending factors
        is_static = isinstance(self.convection_scheme, convection.StaticScheme)
        if not is_static:
            self.convection_scheme.update(dt / self.num_subcycles, self.u_conv)

        # Update global bounds in slope limiter
        if self.is_first_timestep:
            lo, hi = self.slope_limiter.set_global_bounds(lo=0.0, hi=1.0)
            if self.slope_limiter.has_global_bounds:
                sim.log.info(
                    'Setting global bounds [%r, %r] in BlendedAlgebraicVofModel' % (lo, hi)
                )

        # Solve the advection equations for the colour field
        if timestep_number == 1 or is_static:
            c.assign(cp)
        else:
            if self.solver is None:
                sim.log.info('Creating colour function solver', flush=True)
                self.solver = linear_solver_from_input(
                    self.simulation, 'solver/c', default_parameters=SOLVER_OPTIONS
                )

            # Solve the advection equation
            A = self.eq_colour.assemble_lhs()
            for _ in range(self.num_subcycles):
                b = self.eq_colour.assemble_rhs()
                self.solver.inner_solve(A, c.vector(), b, 1, 0)
                self.slope_limiter.run()
                if self.num_subcycles > 1:
                    self.cpp.assign(self.cp)
                    self.cp.assign(c)

        # Optionally use a continuous predicted colour field
        if self.continuous_fields:
            Vcg = self.continuous_c.function_space()
            dolfin.project(c, Vcg, function=self.continuous_c)
            dolfin.project(cp, Vcg, function=self.continuous_c_old)
            dolfin.project(cpp, Vcg, function=self.continuous_c_oldold)

        # Report properties of the colour field
        sim.reporting.report_timestep_value('min(c)', c.vector().min())
        sim.reporting.report_timestep_value('max(c)', c.vector().max())

        # The next update should use the dt from this time step of the
        # main Navier-Stoke solver. The update just computed above uses
        # data from the previous Navier-Stokes solve with the previous dt
        self.dt.assign(dt / self.num_subcycles)

        if dt != sim.dt_prev:
            # Temporary switch to first order timestepping for the next
            # time step. This code is run before the Navier-Stokes solver
            # in each time step
            sim.log.info('VOF solver is first order this time step due to change in dt')
            self.time_coeffs.assign(Constant([1.0, -1.0, 0.0]))
        else:
            # Use second order backward time difference next time step
            self.time_coeffs.assign(Constant([3 / 2, -2.0, 1 / 2]))

        self.update_plot_fields()
        timer.stop()  # Stop timer before hook
        sim.hooks.run_custom_hook('MultiPhaseModelUpdated')
        self.is_first_timestep = False


class AdvectionEquation(Term):
    def __init__(
        self,
        function_space,
        cp,
        cpp,
        u_conv,
        beta,
        time_coeffs,
        dirichlet_bcs,
        forcing_zones=(),
        dt,
    ):
        """
        This class assembles the advection equation for a scalar function c
        """
        super(AdvectionEquation, self).__init__(function_space)

        self.Vc = function_space
        self.cp = cp
        self.cpp = cpp
        self.u_conv = u_conv
        self.beta = beta
        self.time_coeffs = time_coeffs
        self.dirichlet_bcs = dirichlet_bcs
        self.forcing_zones = forcing_zones
        self.dt = dt

        # Discontinuous or continuous elements
        Vc_family = function_space.ufl_element().family()
        self.colour_is_discontinuous = Vc_family == 'Discontinuous Lagrange'

        if isinstance(u_conv[0], (int, float, dolfin.Constant, Zero)):
            self.velocity_is_trace = False
        else:
            try:
                Vu_family = u_conv[0].function_space().ufl_element().family()
            except Exception:
                Vu_family = u_conv.function_space().ufl_element().family()
            self.velocity_is_trace = Vu_family == 'Discontinuous Lagrange Trace'

        if self.velocity_is_trace:
            assert self.colour_is_discontinuous
            assert function_space.ufl_element().degree() == 0

        # Create UFL forms
        self.define_advection_equation()

    def define_advection_equation(self):
        """
        Setup the advection equation for the colour function

        This implementation assembles the full LHS and RHS each time they are needed
        """

        n = self.normal

        # Trial and test functions
        c = self.trial
        d = self.test

        c1, c2, c3 = self.time_coeffs
        u_conv = self.u_conv

        if not self.colour_is_discontinuous:
            # Continous Galerkin implementation of the advection equation
            # FIXME: add stabilization
            eq = (c1 * self.trial + c2 * self.cp + c3 * self.cpp) / self.dt * self.test * dx + div(self.trial * u_conv) * self.test * dx

        elif self.velocity_is_trace:
            # Upstream and downstream normal velocities
            w_nU = (dot(u_conv, self.normal) + abs(dot(u_conv, self.normal))) / 2
            w_nD = (dot(u_conv, self.normal) - abs(dot(u_conv, self.normal))) / 2

            if self.beta is not None:
                # Define the blended flux
                # The blending factor beta is not DG, so beta('+') == beta('-')
                b = self.beta('+')
                flux = (1 - b) * jump(self.trial * w_nU) + b * jump(self.trial * w_nD)
            else:
                flux = jump(self.trial * w_nU)

            # Discontinuous Galerkin implementation of the advection equation
            eq = (c1 * self.trial + c2 * self.cp + c3 * self.cpp) / self.dt * self.test * dx + flux * jump(self.test) * dS

            # On each facet either w_nD or w_nU will be 0, the other is multiplied
            # with the appropriate flux, either the value c going out of the domain
            # or the Dirichlet value coming into the domain
            for dbc in self.dirichlet_bcs: #TODO WP: change
                eq += w_nD * dbc.func() * self.test * dbc.ds()
                eq += w_nU * self.trial * self.test * dbc.ds()

        elif self.beta is not None:
            # Upstream and downstream normal velocities
            w_nU = (dot(u_conv, self.normal) + abs(dot(u_conv, self.normal))) / 2
            w_nD = (dot(u_conv, self.normal) - abs(dot(u_conv, self.normal))) / 2

            if self.beta is not None:
                # Define the blended flux
                # The blending factor beta is not DG, so beta('+') == beta('-')
                b = self.beta('+')
                flux = (1 - b) * jump(self.trial * w_nU) + b * jump(self.trial * w_nD)
            else:
                flux = jump(self.trial * w_nU)

            # Discontinuous Galerkin implementation of the advection equation
            eq = (
                (c1 * self.trial + c2 * self.cp + c3 * self.cpp) / self.dt * self.test * dx
                - dot(self.trial * u_conv, grad(self.test)) * dx
                + flux * jump(self.test) * dS
            )

            # Enforce Dirichlet BCs weakly
            for dbc in self.dirichlet_bcs: #TODO WP: change
                eq += w_nD * dbc.func() * self.test * dbc.ds()
                eq += w_nU * self.trial * self.test * dbc.ds()

        else:
            # Downstream normal velocities
            w_nD = (dot(u_conv, self.normal) - abs(dot(u_conv, self.normal))) / 2

            # Discontinuous Galerkin implementation of the advection equation
            eq = (c1 * self.trial + c2 * self.cp + c3 * self.cpp) / self.dt * self.test * dx

            # Convection integrated by parts two times to bring back the original
            # div form (this means we must subtract and add all fluxes)
            eq += div(self.trial * u_conv) * self.test * dx

            # Replace downwind flux with upwind flux on downwind internal facets
            eq -= jump(w_nD * self.test) * jump(self.trial) * dS

            # Replace downwind flux with upwind BC flux on downwind external facets
            for dbc in self.dirichlet_bcs: #TODO WP: change
                # Subtract the "normal" downwind flux
                eq -= w_nD * self.trial * self.test * dbc.ds()
                # Add the boundary value upwind flux
                eq += w_nD * dbc.func() * self.test * dbc.ds()

        # Penalty forcing zones
        for fz in self.forcing_zones: #TODO WP: change
            eq += fz.penalty * fz.beta * (self.trial - fz.target) * self.test * dx

        a, L = system(eq)
        self.form_lhs = Form(a)
        self.form_rhs = Form(L)
        self.tensor_lhs = None
        self.tensor_rhs = None

    def assemble_lhs(self):
        if self.tensor_lhs is None:
            lhs = assemble(self.form_lhs)
            self.tensor_lhs = as_backend_type(lhs)
        else:
            assemble(self.form_lhs, tensor=self.tensor_lhs)
        return self.tensor_lhs

    def assemble_rhs(self):
        if self.tensor_rhs is None:
            rhs = assemble(self.form_rhs)
            self.tensor_rhs = as_backend_type(rhs)
        else:
            assemble(self.form_rhs, tensor=self.tensor_rhs)
        return self.tensor_rhs
