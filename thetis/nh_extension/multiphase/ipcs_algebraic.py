# Copyright (C) 2015-2019 Tormod Landet
# SPDX-License-Identifier: Apache-2.0

import dolfin
from ocellaris.utils import (
    verify_key,
    timeit,
    linear_solver_from_input,
    create_vector_functions,
    shift_fields,
    velocity_change,
    matmul,
    split_form_into_matrix,
    invert_block_diagonal_matrix,
)
from . import Solver, register_solver, BDM
from .coupled_equations import define_dg_equations
from ..solver_parts import (
    VelocityBDMProjection,
    setup_hydrostatic_pressure,
    SlopeLimiterVelocity,
    before_simulation,
    after_timestep,
    update_timestep,
)

# Solvers - default values, can be changed in the input file
SOLVER_U_OPTIONS = {
    'use_ksp': True,
    'petsc_ksp_type': 'gmres',
    'petsc_pc_type': 'asm',
    'petsc_ksp_initial_guess_nonzero': True,
    'petsc_ksp_view': 'DISABLED',
    'inner_iter_rtol': [1e-10] * 3,
    'inner_iter_atol': [1e-15] * 3,
    'inner_iter_max_it': [100] * 3,
}
SOLVER_P_OPTIONS = {
    'use_ksp': True,
    'petsc_ksp_type': 'gmres',
    'petsc_pc_type': 'hypre',
    'petsc_pc_hypre_type': 'boomeramg',
    'petsc_ksp_initial_guess_nonzero': True,
    'petsc_ksp_view': 'DISABLED',
    'inner_iter_rtol': [1e-10] * 3,
    'inner_iter_atol': [1e-15] * 3,
    'inner_iter_max_it': [100] * 3,
}
MAX_INNER_ITER = 10
ALLOWABLE_ERROR_INNER = 1e-10

# Equations - default values, can be changed in the input file
USE_STRESS_DIVERGENCE = False
USE_LAGRANGE_MULTIPLICATOR = False
USE_GRAD_P_FORM = False
USE_GRAD_Q_FORM = True
HYDROSTATIC_PRESSURE_CALCULATION_EVERY_TIMESTEP = False
INCOMPRESSIBILITY_FLUX_TYPE = 'central'
SPLIT_APPROX_MASS_WITH_RHO = 'mass'
SPLIT_APPROX_MASS_UNSCALED = 'unscaled mass'
SPLIT_APPROX_MASS_MIN_RHO = 'min rho mass'
SPLIT_APPROX_BLOCK_DIAG_MA = 'block diagonal'
SPLIT_APPROX_DEFAULT = SPLIT_APPROX_MASS_WITH_RHO


def assemble_into(form, tensor):
    tensor = dolfin.assemble(form, tensor=tensor)
    return dolfin.as_backend_type(tensor)


@register_solver('IPCS-A')
class SolverIPCSA(Solver):
    description = 'Incremental Pressure Correction Scheme (algebraic form)'

    def __init__(self, simulation):
        """
        A Navier-Stokes solver based on an algebraic Incremental Pressure correction scheme.

        Starting with the coupled Navier-Stokes saddle point system:

            | M+A  B |   | u |   | d |
            |        | . |   | = |   |                                     (1)
            | C    0 |   | p |   | e |

        where e is not allways zero since we use weak BCs for the normal velocity.

        (1) Momentum prediction: guess a pressure p* and then solve for u*

            (M + A) u* = d - B p*

        (2) Use the incompressibility constraint "C u = e" on "M(u - u*) = -B(p - p*)"
            (which can be added to (1) to get close to the correct momentum equation)
            the result is an equation for p (remember that M is block diagonal in DG)

            C M⁻¹ B p = C M⁻¹ B p* + Cu* - e

        (3) Repeat (1) and (2) until ||u-u*|| and/or ||p-p*|| are sufficiently close to
            zero, then update the velocity based on

            u = u* - M⁻¹ B (p - p*)
        """
        self.simulation = sim = simulation
        self.read_input()
        self.create_functions()
        self.hydrostatic_pressure = setup_hydrostatic_pressure(simulation, needs_initial_value=True)

        # First time step timestepping coefficients
        sim.data['time_coeffs'] = dolfin.Constant([1, -1, 0])

        # Solver control parameters
        sim.data['dt'] = dolfin.Constant(simulation.dt)

        # Define weak forms
        self.define_weak_forms()

        # Slope limiter for the momentum equation velocity components
        self.slope_limiter = SlopeLimiterVelocity(sim, sim.data['u'], 'u', vel_w=sim.data['u_conv'])
        self.using_limiter = self.slope_limiter.active

        # Projection for the velocity
        self.velocity_postprocessor = None
        if self.velocity_postprocessing == BDM:
            self.velocity_postprocessor = VelocityBDMProjection(
                sim, sim.data['u'], incompressibility_flux_type=self.incompressibility_flux_type
            )

        # Matrix and vector storage
        self.MplusA = self.B = self.C = self.M = self.Minv = self.D = self.E = None
        self.MinvB = self.CMinvB = None

        # Store number of iterations
        self.niters_u = None
        self.niters_p = None

    def read_input(self):
        """
        Read the simulation input
        """
        sim = self.simulation

        # Create linear solvers
        self.velocity_solver = linear_solver_from_input(
            self.simulation, 'solver/u', default_parameters=SOLVER_U_OPTIONS
        )
        self.pressure_solver = linear_solver_from_input(
            self.simulation, 'solver/p', default_parameters=SOLVER_P_OPTIONS
        )

        # Lagrange multiplicator or remove null space via PETSc
        self.remove_null_space = True
        self.pressure_null_space = None
        self.use_lagrange_multiplicator = sim.input.get_value(
            'solver/use_lagrange_multiplicator', USE_LAGRANGE_MULTIPLICATOR, 'bool'
        )
        if self.use_lagrange_multiplicator:
            self.remove_null_space = False

        # No need for special treatment if the pressure is coupled via outlet BCs
        if sim.data['outlet_bcs']:
            self.remove_null_space = False
            self.use_lagrange_multiplicator = False

        # Control the form of the governing equations
        self.use_stress_divergence_form = sim.input.get_value(
            'solver/use_stress_divergence_form', USE_STRESS_DIVERGENCE, 'bool'
        )
        self.use_grad_p_form = sim.input.get_value(
            'solver/use_grad_p_form', USE_GRAD_P_FORM, 'bool'
        )
        self.use_grad_q_form = sim.input.get_value(
            'solver/use_grad_q_form', USE_GRAD_Q_FORM, 'bool'
        )
        self.incompressibility_flux_type = sim.input.get_value(
            'solver/incompressibility_flux_type', INCOMPRESSIBILITY_FLUX_TYPE, 'string'
        )

        # Representation of velocity
        Vu_family = sim.data['Vu'].ufl_element().family()
        self.vel_is_discontinuous = Vu_family == 'Discontinuous Lagrange'

        # Velocity post_processing
        default_postprocessing = BDM if self.vel_is_discontinuous else 'none'
        self.velocity_postprocessing = sim.input.get_value(
            'solver/velocity_postprocessing', default_postprocessing, 'string'
        )
        verify_key(
            'velocity post processing', self.velocity_postprocessing, ('none', BDM), 'IPCS-A solver'
        )

        # What types of mass matrices to produce
        self.project_initial_velocity = sim.input.get_value(
            'solver/project_initial_velocity', False, 'bool'
        )
        self.splitting_approximation = sim.input.get_value(
            'solver/splitting_approximation', SPLIT_APPROX_DEFAULT, 'string'
        )
        verify_key(
            'solver/mass_approximation',
            self.splitting_approximation,
            (
                SPLIT_APPROX_MASS_WITH_RHO,
                SPLIT_APPROX_MASS_UNSCALED,
                SPLIT_APPROX_MASS_MIN_RHO,
                SPLIT_APPROX_BLOCK_DIAG_MA,
            ),
            'IPCS-A solver',
        )
        self.make_unscaled_M = (
            self.project_initial_velocity
            or self.splitting_approximation == SPLIT_APPROX_MASS_UNSCALED
        )

        # Quasi-steady simulation input
        self.steady_velocity_eps = sim.input.get_value(
            'solver/steady_velocity_stopping_criterion', None, 'float'
        )
        self.is_steady = self.steady_velocity_eps is not None

    def create_functions(self):
        """
        Create functions to hold solutions
        """
        sim = self.simulation

        # Function spaces
        Vu = sim.data['Vu']
        Vp = sim.data['Vp']

        # Create velocity functions on component and vector form
        create_vector_functions(sim, 'u', 'u%d', Vu)
        create_vector_functions(sim, 'up', 'up%d', Vu)
        create_vector_functions(sim, 'upp', 'upp%d', Vu)
        create_vector_functions(sim, 'u_conv', 'u_conv%d', Vu)
        create_vector_functions(sim, 'up_conv', 'up_conv%d', Vu)
        create_vector_functions(sim, 'upp_conv', 'upp_conv%d', Vu)
        create_vector_functions(sim, 'u_unlim', 'u_unlim%d', Vu)
        sim.data['ui_tmp'] = dolfin.Function(Vu)

        # Create coupled vector function
        ue = Vu.ufl_element()
        e_mixed = dolfin.MixedElement([ue] * sim.ndim)
        Vcoupled = dolfin.FunctionSpace(Vu.mesh(), e_mixed)
        sim.data['uvw_star'] = dolfin.Function(Vcoupled)
        sim.data['uvw_temp'] = dolfin.Function(Vcoupled)
        sim.ndofs += Vcoupled.dim() + Vp.dim()

        # Create assigner to extract split function from uvw and vice versa
        self.assigner_split = dolfin.FunctionAssigner([Vu] * sim.ndim, Vcoupled)
        self.assigner_merge = dolfin.FunctionAssigner(Vcoupled, [Vu] * sim.ndim)

        # Create pressure function
        sim.data['p'] = dolfin.Function(Vp)
        sim.data['p_hat'] = dolfin.Function(Vp)

    def define_weak_forms(self):
        sim = self.simulation
        self.Vuvw = sim.data['uvw_star'].function_space()
        Vp = sim.data['Vp']

        # The trial and test functions in a coupled space (to be split)
        func_spaces = [self.Vuvw, Vp]
        e_mixed = dolfin.MixedElement([fs.ufl_element() for fs in func_spaces])
        Vcoupled = dolfin.FunctionSpace(sim.data['mesh'], e_mixed)
        tests = dolfin.TestFunctions(Vcoupled)
        trials = dolfin.TrialFunctions(Vcoupled)

        # Split into components
        v = dolfin.as_vector(tests[0][:])
        u = dolfin.as_vector(trials[0][:])
        q = tests[-1]
        p = trials[-1]
        lm_trial = lm_test = None

        # Define the full coupled form and split it into subforms depending
        # on the test and trial functions
        eq = define_dg_equations(
            u,
            v,
            p,
            q,
            lm_trial,
            lm_test,
            self.simulation,
            include_hydrostatic_pressure=self.hydrostatic_pressure.every_timestep,
            incompressibility_flux_type=self.incompressibility_flux_type,
            use_grad_q_form=self.use_grad_q_form,
            use_grad_p_form=self.use_grad_p_form,
            use_stress_divergence_form=self.use_stress_divergence_form,
        )
        sim.log.info('    Splitting coupled form')
        mat, vec = split_form_into_matrix(eq, Vcoupled, Vcoupled, check_zeros=True)

        # Check matrix and vector shapes and that the matrix is a saddle point matrix
        assert mat.shape == (2, 2)
        assert vec.shape == (2,)
        assert mat[-1, -1] is None, 'Found p-q coupling, this is not a saddle point system!'

        # Compile and store the forms
        sim.log.info('    Compiling IPCS-A forms')
        self.eqA = dolfin.Form(mat[0, 0])
        self.eqB = dolfin.Form(mat[0, 1])
        self.eqC = dolfin.Form(mat[1, 0])
        self.eqD = dolfin.Form(vec[0])
        self.eqE = dolfin.Form(vec[1]) if vec[1] is not None else None

        # The mass matrix. Consistent with the implementation in define_dg_equations
        if self.splitting_approximation == SPLIT_APPROX_MASS_WITH_RHO:
            rho_for_M = sim.multi_phase_model.get_density(0)
        elif self.splitting_approximation == SPLIT_APPROX_MASS_MIN_RHO:
            min_rho, _max_rho = sim.multi_phase_model.get_density_range()
            rho_for_M = dolfin.Constant(min_rho)
        else:
            rho_for_M = None
        if rho_for_M is not None:
            c1 = sim.data['time_coeffs'][0]
            dt = sim.data['dt']
            eqM = rho_for_M * c1 / dt * dolfin.dot(u, v) * dolfin.dx
            matM, _vecM = split_form_into_matrix(eqM, Vcoupled, Vcoupled, check_zeros=True)
            self.eqM = dolfin.Form(matM[0, 0])

        # The mass matrix without density
        if self.make_unscaled_M:
            u2 = dolfin.as_vector(dolfin.TrialFunction(self.Vuvw)[:])
            v2 = dolfin.as_vector(dolfin.TestFunction(self.Vuvw)[:])
            a = dolfin.dot(u2, v2) * dolfin.dx
            Mus = assemble_into(a, None)
            self.M_unscaled_inv = invert_block_diagonal_matrix(self.Vuvw, Mus)

    @timeit
    def project_vector_field(self, vel_split, vel, name):
        """
        Project the initial conditions to remove any divergence
        """
        sim = self.simulation
        sim.log.info('Projecting %s to remove divergence' % name)
        p = sim.data['p'].copy()
        p.vector().zero()
        self.assigner_merge.assign(vel, list(vel_split))

        def mk_rhs():
            rhs = self.C * vel.vector()
            # If there is no flux (Dirichlet type) across the boundaries then e is None
            if self.eqE is not None:
                self.E = assemble_into(self.eqE, self.E)
                rhs.axpy(-1.0, self.E)
            rhs.apply('insert')
            return rhs

        # Assemble RHS
        sim.log.info('    Assembling projection matrices')
        self.B = assemble_into(self.eqB, self.B)
        self.C = assemble_into(self.eqC, self.C)
        MinvB = matmul(self.M_unscaled_inv, self.B)
        rhs = mk_rhs()

        # Check if projection is needed
        norm_before = rhs.norm('l2')
        sim.log.info('    Divergence norm before %.6e' % norm_before)
        if norm_before < 1e-15:
            sim.log.info('    Skipping this one, there is no divergence')
            return

        # Assemble LHS
        CMinvB = matmul(self.C, MinvB)
        lhs = CMinvB

        sim.log.info('    Solving elliptic problem')
        # niter = self.pressure_solver.inner_solve(lhs, p.vector(), rhs, 0, 0)
        niter = dolfin.solve(lhs, p.vector(), rhs)
        vel.vector().axpy(-1.0, MinvB * p.vector())
        vel.vector().apply('insert')

        self.assigner_split.assign(list(vel_split), vel)
        for d in range(sim.ndim):
            sim.data['u'][d].assign(vel_split[d])
        self.velocity_postprocessor.run()
        for d in range(sim.ndim):
            vel_split[d].assign(sim.data['u'][d])
        self.assigner_merge.assign(vel, list(vel_split))

        rhs = mk_rhs()
        norm_after = rhs.norm('l2')
        sim.log.info('    Done in %d iterations' % niter)
        sim.log.info('    Divergence norm after %.6e' % norm_after)

    @timeit
    def momentum_prediction(self):
        """
        Solve the momentum prediction equation
        """
        sim = self.simulation
        u_star = sim.data['uvw_star']
        u_temp = sim.data['uvw_temp']
        p_star = sim.data['p']

        # Assemble only once per time step
        if self.inner_iteration == 1:
            self.MplusA = assemble_into(self.eqA, self.MplusA)
            self.B = assemble_into(self.eqB, self.B)
            self.D = assemble_into(self.eqD, self.D)

        lhs = self.MplusA
        rhs = self.D - self.B * p_star.vector()

        # Solve the linearised convection-diffusion system
        u_temp.assign(u_star)
        self.niters_u = self.velocity_solver.inner_solve(
            lhs, u_star.vector(), rhs, in_iter=self.inner_iteration, co_iter=self.co_inner_iter
        )

        # Compute change from last iteration
        u_temp.vector().axpy(-1, u_star.vector())
        u_temp.vector().apply('insert')
        self._last_u_err = u_temp.vector().norm('l2')
        return self._last_u_err

    @timeit
    def compute_M_inverse(self):
        """
        Compute the inverse of the block diagonal mass matrix

        Uses either the mass matrix M or the block diagonal
        parts of A
        """
        # Get the block diagonal matrix
        if self.splitting_approximation == SPLIT_APPROX_MASS_WITH_RHO:
            # The standard rho/dt approximation
            self.M = assemble_into(self.eqM, self.M)
            approx_A = self.M
        if self.splitting_approximation == SPLIT_APPROX_MASS_MIN_RHO:
            # Use a min(rho)/dt approximation (assumed to not change with time)
            if self.Minv is not None:
                return self.Minv
            self.M = assemble_into(self.eqM, self.M)
            approx_A = self.M
        elif self.splitting_approximation == SPLIT_APPROX_MASS_UNSCALED:
            # Use the FEM "mass matrix", i.e. no rho or dt inside
            return self.M_unscaled_inv
        elif self.splitting_approximation == SPLIT_APPROX_BLOCK_DIAG_MA:
            # Use the block diagonal part of the velocity matrix
            approx_A = self.MplusA

        # Invert the block diagonal matrix
        self.Minv = invert_block_diagonal_matrix(self.Vuvw, approx_A, self.Minv)
        return self.Minv

    @timeit
    def pressure_correction(self):
        """
        Solve the Navier-Stokes equations on SIMPLE form
        (Semi-Implicit Method for Pressure-Linked Equations)
        """
        sim = self.simulation
        u_star = sim.data['uvw_star']
        p_star = sim.data['p']
        p_hat = self.simulation.data['p_hat']

        # Assemble only once per time step
        if self.inner_iteration == 1:
            self.C = assemble_into(self.eqC, self.C)
            self.Minv = dolfin.as_backend_type(self.compute_M_inverse())
            if self.eqE is not None:
                self.E = assemble_into(self.eqE, self.E)

            # Compute LHS
            self.MinvB = matmul(self.Minv, self.B, self.MinvB)
            self.CMinvB = matmul(self.C, self.MinvB, self.CMinvB)

        # The equation system
        lhs = self.CMinvB
        rhs = self.CMinvB * p_star.vector()
        rhs.axpy(1, self.C * u_star.vector())
        if self.eqE is not None:
            rhs.axpy(-1, self.E)
        rhs.apply('insert')

        # Inform PETSc about the pressure null space
        if self.remove_null_space:
            if self.pressure_null_space is None:
                # Create vector that spans the null space
                null_vec = dolfin.Vector(p_star.vector())
                null_vec[:] = 1
                null_vec *= 1 / null_vec.norm("l2")

                # Create null space basis object
                self.pressure_null_space = dolfin.VectorSpaceBasis([null_vec])

            # Make sure the null space is set on the matrix
            if self.inner_iteration == 1:
                lhs.set_nullspace(self.pressure_null_space)

            # Orthogonalize b with respect to the null space
            self.pressure_null_space.orthogonalize(rhs)

        # Temporarily store the old pressure
        p_hat.vector().zero()
        p_hat.vector().axpy(-1, p_star.vector())

        # Solve for the new pressure correction
        self.niters_p = self.pressure_solver.inner_solve(
            lhs, p_star.vector(), rhs, in_iter=self.inner_iteration, co_iter=self.co_inner_iter
        )

        # Removing the null space of the matrix system is not strictly the same as removing
        # the null space of the equation, so we correct for this here
        if self.remove_null_space:
            dx2 = dolfin.dx(domain=p_star.function_space().mesh())
            vol = dolfin.assemble(dolfin.Constant(1) * dx2)
            pavg = dolfin.assemble(p_star * dx2) / vol
            p_star.vector()[:] -= pavg

        # Calculate p_hat = p_new - p_old
        p_hat.vector().axpy(1, p_star.vector())

        return p_hat.vector().norm('l2')

    @timeit
    def velocity_update(self):
        """
        Update the velocity predictions with the updated pressure
        field from the pressure correction equation
        """
        p_hat = self.simulation.data['p_hat']
        uvw = self.simulation.data['uvw_star']
        uvw.vector().axpy(-1, self.MinvB * p_hat.vector())
        uvw.vector().apply('insert')

    @timeit
    def postprocess_velocity(self):
        """
        Apply a post-processing operator to the given velocity field
        """
        if self.velocity_postprocessor:
            self.velocity_postprocessor.run()

    @timeit
    def slope_limit_velocities(self):
        """
        Run the slope limiter
        """
        if not self.using_limiter:
            return 0

        # Store unlimited velocities and then run limiter
        shift_fields(self.simulation, ['u%d', 'u_unlim%d'])
        self.slope_limiter.run()

        # Measure the change in the field after limiting (l2 norm)
        change = velocity_change(
            u1=self.simulation.data['u'],
            u2=self.simulation.data['u_unlim'],
            ui_tmp=self.simulation.data['ui_tmp'],
        )

        return change

    def run(self):
        """
        Run the simulation
        """
        sim = self.simulation

        # Remove any divergence from the initial velocity field
        if self.project_initial_velocity:
            for name in ('up', 'upp'):
                self.project_vector_field(
                    sim.data[name], sim.data['uvw_star'], 'initial velocity %s' % name
                )
            shift_fields(sim, ['up%d', 'up_conv%d'])
            shift_fields(sim, ['upp%d', 'upp_conv%d'])

        # Setup timestepping and initial convecting velocity
        sim.hooks.simulation_started()
        before_simulation(sim)

        # Time loop
        t = sim.time
        it = sim.timestep

        with dolfin.Timer('Ocellaris run IPCS-A solver'):
            while True:
                # Get input values, these can possibly change over time
                dt = update_timestep(sim)
                tmax = sim.input.get_value('time/tmax', required_type='float')
                num_inner_iter = sim.input.get_value('solver/num_inner_iter', MAX_INNER_ITER, 'int')
                allowable_error_inner = sim.input.get_value(
                    'solver/allowable_error_inner', ALLOWABLE_ERROR_INNER, 'float'
                )

                # Check if the simulation is done
                if t + dt > tmax + 1e-6:
                    break

                # Advance one time step
                it += 1
                t += dt
                self.simulation.hooks.new_timestep(it, t, dt)

                # Calculate the hydrostatic pressure when the density is not constant
                self.hydrostatic_pressure.update()

                # Collect previous velocity components in coupled function
                self.assigner_merge.assign(sim.data['uvw_star'], list(sim.data['u']))

                # Run inner iterations
                self.inner_iteration = 1
                while self.inner_iteration <= num_inner_iter:
                    self.co_inner_iter = num_inner_iter - self.inner_iteration
                    err_u = self.momentum_prediction()
                    err_p = self.pressure_correction()
                    self.velocity_update()
                    sim.log.info(
                        '  IPCS-A iteration %3d - err u* %10.3e - err p %10.3e'
                        ' - Num Krylov iters - u %3d - p %3d'
                        % (self.inner_iteration, err_u, err_p, self.niters_u, self.niters_p)
                    )
                    self.inner_iteration += 1
                    sim.flush(False)  # Flushes output if sufficient time has passed

                    if err_u < allowable_error_inner:
                        break

                # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
                # Report total flux in the domain, should be zero
                n = dolfin.FacetNormal(self.simulation.data['mesh'])
                flux = sum(n[i] * sim.data['uvw_star'][i] for i in range(self.simulation.ndim))
                tflux = dolfin.assemble(flux * dolfin.ds)
                sim.reporting.report_timestep_value('TotFlux', tflux)

                # Extract the separate velocity component functions
                self.assigner_split.assign(list(sim.data['u']), sim.data['uvw_star'])

                # Postprocess and limit velocity outside the inner iteration
                self.postprocess_velocity()
                shift_fields(sim, ['u%d', 'u_conv%d'])
                if self.using_limiter:
                    self.slope_limit_velocities()

                # Move u -> up, up -> upp and prepare for the next time step
                vel_diff = after_timestep(sim, self.is_steady)

                # Stop steady state simulation if convergence has been reached
                if self.is_steady:
                    vel_diff = dolfin.MPI.max(dolfin.MPI.comm_world, float(vel_diff))
                    sim.reporting.report_timestep_value('max(ui_new-ui_prev)', vel_diff)
                    if vel_diff < self.steady_velocity_eps:
                        sim.log.info('Stopping simulation, steady state achieved')
                        sim.input.set_value('time/tmax', t)

                # Postprocess this time step
                sim.hooks.end_timestep()

        # We are done
        sim.hooks.simulation_ended(success=True)
