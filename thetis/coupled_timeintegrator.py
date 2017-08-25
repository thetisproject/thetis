"""
Time integrators for solving coupled 2D-3D system of equations.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from . import rungekutta
from . import implicitexplicit
from abc import ABCMeta, abstractproperty


class CoupledTimeIntegratorBase(timeintegrator.TimeIntegratorBase):
    """
    Base class for coupled 2D-3D time integrators

    Provides common functionality for updating diagnostic fields etc.
    """
    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.fields = solver.fields
        self.timesteppers = AttrDict()

    def _update_3d_elevation(self):
        """Projects elevation to 3D"""
        with timed_stage('aux_elev_3d'):
            self.solver.copy_elev_to_3d.solve()  # at t_{n+1}

    def _update_vertical_velocity(self):
        """Solve vertical velocity"""
        with timed_stage('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates 3D mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
                self.solver.mesh_updater.update_mesh_coordinates()

    def _update_bottom_friction(self):
        """Computes bottom friction related fields"""
        if self.options.use_bottom_friction:
            with timed_stage('aux_friction'):
                self.solver.uv_p1_projector.project()
                compute_bottom_friction(
                    self.solver,
                    self.fields.uv_p1_3d, self.fields.uv_bottom_2d,
                    self.fields.z_bottom_2d, self.fields.bathymetry_2d,
                    self.fields.bottom_drag_2d)
        if self.options.use_parabolic_viscosity:
            self.solver.parabolic_viscosity_solver.solve()

    def _update_2d_coupling(self):
        """Does 2D-3D coupling for the velocity field"""
        with timed_stage('aux_uv_coupling'):
            self._remove_depth_average_from_uv_3d()
            self._update_2d_coupling_term()
            self._copy_uv_2d_to_3d()

    def _remove_depth_average_from_uv_3d(self):
        """Computes depth averaged velocity and removes it from the 3D velocity field"""
        with timed_stage('aux_uv_coupling'):
            # compute depth averaged 3D velocity
            self.solver.uv_averager.solve()  # uv -> uv_dav_3d
            self.solver.extract_surf_dav_uv.solve()  # uv_dav_3d -> uv_dav_2d
            self.solver.copy_uv_dav_to_uv_dav_3d.solve()  # uv_dav_2d -> uv_dav_3d
            # remove depth average from 3D velocity
            self.fields.uv_3d -= self.fields.uv_dav_3d

    def _copy_uv_2d_to_3d(self):
        """Copies uv_2d to uv_dav_3d"""
        with timed_stage('aux_uv_coupling'):
            self.solver.copy_uv_to_uv_dav_3d.solve()

    def _update_2d_coupling_term(self):
        """Update split_residual_2d field for 2D-3D coupling"""
        with timed_stage('aux_uv_coupling'):
            # scale dav uv 2D to be used as a forcing in 2D mom eq.
            self.fields.split_residual_2d.assign(self.fields.uv_dav_2d)
            self.fields.split_residual_2d /= self.timesteppers.mom_expl.dt_const

    def _update_baroclinicity(self):
        """Computes baroclinic head"""
        if self.options.use_baroclinic_formulation:
            compute_baroclinic_head(self.solver)

    def _update_turbulence(self, t):
        """
        Updates turbulence related fields

        :arg t: simulation time
        """
        if self.options.use_turbulence:
            with timed_stage('turbulence'):
                self.solver.turbulence_model.preprocess()
                # NOTE psi must be solved first as it depends on tke
                if 'psi_impl' in self.timesteppers:
                    self.timesteppers.psi_impl.advance(t)
                if 'tke_impl' in self.timesteppers:
                    self.timesteppers.tke_impl.advance(t)
                self.solver.turbulence_model.postprocess()

    def _update_stabilization_params(self):
        """
        Computes Smagorinsky viscosity etc fields
        """
        with timed_stage('aux_stability'):
            self.solver.uv_mag_solver.solve()
            # update P1 velocity field
            self.solver.uv_p1_projector.project()
            if self.options.use_smagorinsky_viscosity:
                self.solver.smagorinsky_diff_solver.solve()

    def _update_all_dependencies(self, t,
                                 do_2d_coupling=False,
                                 do_vert_diffusion=False,
                                 do_ale_update=False,
                                 do_stab_params=False,
                                 do_turbulence=False):
        """Default routine for updating all dependent fields after a time step"""
        self._update_3d_elevation()
        if do_ale_update:
            self._update_moving_mesh()
        if do_2d_coupling:
            self._update_2d_coupling()
        self._update_vertical_velocity()
        self._update_bottom_friction()
        self._update_baroclinicity()
        if do_turbulence:
            self._update_turbulence(t)
        if do_vert_diffusion and self.options.use_implicit_vertical_diffusion:
            with timed_stage('impl_mom_vvisc'):
                self.timesteppers.mom_impl.advance(t)
            if self.options.solve_salinity:
                with timed_stage('impl_salt_vdiff'):
                    self.timesteppers.salt_impl.advance(t)
            if self.options.solve_temperature:
                with timed_stage('impl_temp_vdiff'):
                    self.timesteppers.temp_impl.advance(t)
        if do_stab_params:
            self._update_stabilization_params()


class CoupledTimeIntegrator(CoupledTimeIntegratorBase):
    """
    Base class of mode-split time integrators that use 2D, 3D and implicit 3D
    time integrators.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def integrator_2d(self):
        """time integrator for 2D equations"""
        pass

    @abstractproperty
    def integrator_3d(self):
        """time integrator for explicit 3D equations (momentum, tracers)"""
        pass

    @abstractproperty
    def integrator_vert_3d(self):
        """time integrator for implicit 3D equations (vertical diffusion)"""
        pass

    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        super(CoupledTimeIntegrator, self).__init__(solver)
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        print_output('  2D time integrator: {:}'.format(self.integrator_2d.__name__))
        print_output('  3D time integrator: {:}'.format(self.integrator_3d.__name__))
        print_output('  3D implicit time integrator: {:}'.format(self.integrator_vert_3d.__name__))
        self._initialized = False

        self._create_integrators()
        self.n_stages = self.timesteppers.swe2d.n_stages

    def _get_vert_diffusivity_functions(self):
        """
        Assign vertical viscosity/diffusivity to implicit/explicit parts
        """
        if self.options.use_implicit_vertical_diffusion:
            impl_v_visc = self.solver.tot_v_visc.get_sum()
            expl_v_visc = None
            impl_v_diff = self.solver.tot_v_diff.get_sum()
            expl_v_diff = None
        else:
            impl_v_visc = None
            expl_v_visc = self.solver.tot_v_visc.get_sum()
            impl_v_diff = None
            expl_v_diff = self.solver.tot_v_diff.get_sum()

        return impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff

    def _create_swe_integrator(self):
        """
        Create time integrator for 2D system
        """
        solver = self.solver
        momentum_source_2d = solver.fields.split_residual_2d
        if self.options.momentum_source_2d is not None:
            momentum_source_2d = solver.fields.split_residual_2d + self.options.momentum_source_2d
        fields = {
            'uv_bottom': solver.fields.get('uv_bottom_2d'),
            'bottom_drag': solver.fields.get('bottom_drag_2d'),
            'viscosity_h': self.options.horizontal_viscosity,  # FIXME should be total h visc
            'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
            'coriolis': self.options.coriolis_frequency,
            'wind_stress': self.options.wind_stress,
            'momentum_source': momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
            'linear_drag_coefficient': self.options.linear_drag_coefficient}

        if issubclass(self.integrator_2d, (rungekutta.ERKSemiImplicitGeneric)):
            self.timesteppers.swe2d = self.integrator_2d(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe,
                semi_implicit=True,
                theta=self.options.timestepper_options.implicitness_theta_2d)
        else:
            self.timesteppers.swe2d = self.integrator_2d(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe)

    def _create_mom_integrator(self):
        """
        Create time integrator for 3D momentum equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        fields = {'eta': self.fields.elev_3d,  # FIXME rename elev
                  'int_pg': self.fields.get('int_pg_3d'),
                  'uv_depth_av': self.fields.get('uv_dav_3d'),
                  'w': self.fields.w_3d,
                  'w_mesh': self.fields.get('w_mesh_3d'),
                  'viscosity_v': expl_v_visc,
                  'viscosity_h': self.solver.tot_h_visc.get_sum(),
                  'source': self.options.momentum_source_3d,
                  # uv_mag': self.fields.uv_mag_3d,
                  'uv_p1': self.fields.get('uv_p1_3d'),
                  'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                  'coriolis': self.fields.get('coriolis_3d'),
                  'linear_drag_coefficient': self.options.linear_drag_coefficient,
                  'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
                  }
        self.timesteppers.mom_expl = self.integrator_3d(
            solver.eq_momentum, solver.fields.uv_3d, fields, solver.dt,
            bnd_conditions=solver.bnd_functions['momentum'],
            solver_parameters=self.options.timestepper_options.solver_parameters_momentum_explicit)
        if self.solver.options.use_implicit_vertical_diffusion:
            fields = {'viscosity_v': impl_v_visc,
                      'wind_stress': self.fields.get('wind_stress_3d'),
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      }
            self.timesteppers.mom_impl = self.integrator_vert_3d(
                solver.eq_vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['momentum'],
                solver_parameters=self.options.timestepper_options.solver_parameters_momentum_implicit)

    def _create_salt_integrator(self):
        """
        Create time integrator for salinity equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        if self.solver.options.solve_salinity:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': expl_v_diff,
                      'source': self.options.salinity_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.salt_expl = self.integrator_3d(
                solver.eq_salt, solver.fields.salt_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['salt'],
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)
            if self.solver.options.use_implicit_vertical_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': impl_v_diff,
                          }
                self.timesteppers.salt_impl = self.integrator_vert_3d(
                    solver.eq_salt_vdff, solver.fields.salt_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['salt'],
                    solver_parameters=self.options.timestepper_options.solver_parameters_tracer_implicit)

    def _create_temp_integrator(self):
        """
        Create time integrator for temperature equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        if self.solver.options.solve_temperature:
            fields = {'elev_3d': self.fields.elev_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': expl_v_diff,
                      'source': self.options.temperature_source_3d,
                      # uv_mag': self.fields.uv_mag_3d,
                      'uv_p1': self.fields.get('uv_p1_3d'),
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.temp_expl = self.integrator_3d(
                solver.eq_temp, solver.fields.temp_3d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['temp'],
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)
            if self.solver.options.use_implicit_vertical_diffusion:
                fields = {'elev_3d': self.fields.elev_3d,
                          'diffusivity_v': impl_v_diff,
                          }
                self.timesteppers.temp_impl = self.integrator_vert_3d(
                    solver.eq_temp_vdff, solver.fields.temp_3d, fields, solver.dt,
                    bnd_conditions=solver.bnd_functions['temp'],
                    solver_parameters=self.options.timestepper_options.solver_parameters_tracer_implicit)

    def _create_turb_integrator(self):
        """
        Create time integrators for turbulence equations
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        eq_tke_diff = getattr(self.solver, 'eq_tke_diff', None)
        eq_psi_diff = getattr(self.solver, 'eq_psi_diff', None)
        eq_tke_adv = getattr(self.solver, 'eq_tke_adv', None)
        eq_psi_adv = getattr(self.solver, 'eq_psi_adv', None)
        if eq_tke_diff is not None and eq_psi_diff is not None:
            fields = {'diffusivity_v': impl_v_diff,
                      'viscosity_v': impl_v_visc,
                      'k': solver.fields.tke_3d,
                      'epsilon': solver.turbulence_model.epsilon,
                      'shear_freq2': solver.turbulence_model.m2,
                      'buoy_freq2_neg': solver.turbulence_model.n2_neg,
                      'buoy_freq2_pos': solver.turbulence_model.n2_pos
                      }
            self.timesteppers.tke_impl = self.integrator_vert_3d(
                eq_tke_diff, solver.fields.tke_3d, fields, solver.dt,
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer_implicit)
            self.timesteppers.psi_impl = self.integrator_vert_3d(
                eq_psi_diff, solver.fields.psi_3d, fields, solver.dt,
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer_implicit)
            if eq_tke_adv is not None and eq_psi_adv is not None:
                fields = {'elev_3d': self.fields.elev_3d,
                          'uv_3d': self.fields.uv_3d,
                          'uv_depth_av': self.fields.get('uv_dav_3d'),
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          # uv_mag': self.fields.uv_mag_3d,
                          'uv_p1': self.fields.get('uv_p1_3d'),
                          'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                          }
                self.timesteppers.tke_expl = self.integrator_3d(
                    eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)
                self.timesteppers.psi_expl = self.integrator_3d(
                    eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    solver_parameters=self.options.timestepper_options.solver_parameters_tracer_explicit)

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        self._create_swe_integrator()
        self._create_mom_integrator()
        self._create_salt_integrator()
        self._create_temp_integrator()
        self._create_turb_integrator()

        self.cfl_coeff_3d = self.timesteppers.mom_expl.cfl_coeff
        self.cfl_coeff_2d = self.timesteppers.swe2d.cfl_coeff

    def set_dt(self, dt, dt_2d):
        """
        Set time step for the coupled time integrator

        :arg float dt: Time step. This is the master (macro) time step used to
            march the 3D equations.
        :arg float dt_2d: Time step for 2D equations. For consistency
            :attr:`dt_2d` must be an integer fraction of :attr:`dt`.
            If 2D solver is implicit set :attr:`dt_2d` equal to :attr:`dt`.
        """
        assert np.isclose(dt/dt_2d, np.round(dt/dt_2d)), \
            'dt_2d is not integer fraction of dt'

        if dt != dt_2d:
            raise NotImplementedError('Case dt_2d < dt is not implemented yet')

        for stepper in self.timesteppers:
            self.timesteppers[stepper].set_dt(dt)

    def initialize(self):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        self.timesteppers.mom_expl.initialize(self.fields.uv_3d)
        if self.options.use_implicit_vertical_diffusion:
            self.timesteppers.mom_impl.initialize(self.fields.uv_3d)
        if self.options.solve_salinity:
            self.timesteppers.salt_expl.initialize(self.fields.salt_3d)
            if self.options.use_implicit_vertical_diffusion:
                self.timesteppers.salt_impl.initialize(self.fields.salt_3d)
        if self.options.solve_temperature:
            self.timesteppers.temp_expl.initialize(self.fields.temp_3d)
            if self.options.use_implicit_vertical_diffusion:
                self.timesteppers.temp_impl.initialize(self.fields.temp_3d)
        if 'psi_impl' in self.timesteppers:
            self.timesteppers.psi_impl.initialize(self.fields.psi_3d)
        if 'tke_impl' in self.timesteppers:
            self.timesteppers.tke_impl.initialize(self.fields.tke_3d)
        if 'psi_expl' in self.timesteppers:
            self.timesteppers.psi_expl.initialize(self.fields.psi_3d)
        if 'tke_expl' in self.timesteppers:
            self.timesteppers.tke_expl.initialize(self.fields.tke_3d)
        self._initialized = True


class CoupledSSPRKSemiImplicit(CoupledTimeIntegrator):
    """
    Solves coupled equations with SSPRK33 time integrator using the same time
    step for the 2D and 3D modes.

    In the 2D mode the surface gravity waves are solved semi-implicitly. This
    allows longer time steps but diffuses free surface waves.

    This time integrator uses a static 3D mesh. It is not compliant with the
    ALE moving mesh.
    """
    integrator_2d = rungekutta.SSPRK33SemiImplicit
    integrator_3d = rungekutta.SSPRK33
    integrator_vert_3d = rungekutta.BackwardEuler

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_stage(k, t, update_forcings3d)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_stage(k, t)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_stage(k, t)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_stage(k, t)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.solve_stage(k, t, update_forcings)
            last_step = (k == 2)
            # move fields to next stage
            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=last_step,
                                          do_ale_update=last_step,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class CoupledERKALE(CoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation

    A fully explicit mode-split time integrator where both the 2D and 3D modes
    use the same time step. The time step is typically chosen to match the 2D
    surface gravity wave speed. Only vertical diffusion is treated implicitly.
    """
    integrator_2d = rungekutta.ERKLPUM2
    integrator_3d = rungekutta.ERKLPUM2ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledERKALE, self).__init__(solver)

        self.elev_cg_old_2d = []
        for i in range(self.n_stages + 1):
            f = Function(self.solver.fields.elev_cg_2d)
            self.elev_cg_old_2d.append(f)

        import numpy.linalg as linalg
        ti = self.timesteppers.swe2d
        assert not ti.is_implicit
        a = ti.butcher[1:, :]
        self.a_inv = linalg.inv(a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :arg i_stage: state of the Runge-Kutta iteration
        """
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """
        Compute mesh velocity from 2D solver runge-kutta scheme

        Mesh velocity is solved from the Runge-Kutta coefficients of the
        implicit 2D solver.

        :arg i_stage: state of the Runge-Kutta iteration
        """
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velocity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref/h + 1.0)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            # FIXME mesh velocity is too high ~2x with EKRLPUM2
            last_step = (k == self.n_stages - 1)
            self._compute_mesh_velocity_pre(k)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.update_solution(k)
                self.timesteppers.swe2d.solve_tendency(k, t, update_forcings)
                if last_step:
                    self.timesteppers.swe2d.get_final_solution()
            self.compute_mesh_velocity(k)

            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_tendency(k, t, update_forcings3d)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_tendency(k, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_tendency(k, t, update_forcings3d)

            self._update_moving_mesh()

            if last_step:
                with timed_stage('salt_eq'):
                    if self.options.solve_salinity:
                        self.timesteppers.salt_expl.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temperature:
                        self.timesteppers.temp_expl.get_final_solution()
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if 'psi_expl' in self.timesteppers:
                        self.timesteppers.psi_expl.get_final_solution()
                    if 'tke_expl' in self.timesteppers:
                        self.timesteppers.tke_expl.get_final_solution()
                with timed_stage('momentum_eq'):
                    self.timesteppers.mom_expl.get_final_solution()
            else:
                with timed_stage('salt_eq'):
                    if self.options.solve_salinity:
                        self.timesteppers.salt_expl.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.salt_3d)
                with timed_stage('temp_eq'):
                    if self.options.solve_temperature:
                        self.timesteppers.temp_expl.update_solution(k)
                        if self.options.use_limiter_for_tracers:
                            self.solver.tracer_limiter.apply(self.fields.temp_3d)
                with timed_stage('turb_advection'):
                    if 'psi_expl' in self.timesteppers:
                        self.timesteppers.psi_expl.update_solution(k)
                    if 'tke_expl' in self.timesteppers:
                        self.timesteppers.tke_expl.update_solution(k)
                with timed_stage('momentum_eq'):
                    self.timesteppers.mom_expl.update_solution(k)

            self._update_all_dependencies(t, do_vert_diffusion=last_step,
                                          do_2d_coupling=True,
                                          do_ale_update=False,
                                          do_stab_params=last_step,
                                          do_turbulence=last_step)


class CoupledIMEXALE(CoupledTimeIntegrator):
    """
    Implicit-Explicit SSP RK solver for conservative ALE formulation

    Advances the 2D-3D system with IMEX scheme: the free surface gravity waves
    are solved with the implicit scheme while all other terms are solved with the
    explicit scheme. Vertical diffusion is however solved with a separate
    implicit scheme (backward Euler) for efficiency.
    """
    integrator_2d = implicitexplicit.IMEXLPUM2
    integrator_3d = rungekutta.ERKLPUM2ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledIMEXALE, self).__init__(solver)

        self.elev_cg_old_2d = []
        for i in range(self.n_stages + 1):
            f = Function(self.solver.fields.elev_cg_2d)
            self.elev_cg_old_2d.append(f)

        import numpy.linalg as linalg
        ti = self.timesteppers.swe2d.dirk
        assert ti.is_implicit
        self.a_inv = linalg.inv(ti.a)

    def _compute_mesh_velocity_pre(self, i_stage):
        """
        Begin mesh velocity computation by storing current elevation field

        :arg i_stage: state of the Runge-Kutta iteration
        """
        if i_stage == 0:
            fields = self.solver.fields
            self.solver.elev_2d_to_cg_projector.project()
            self.elev_cg_old_2d[i_stage].assign(fields.elev_cg_2d)

    def compute_mesh_velocity(self, i_stage):
        """
        Compute mesh velocity from 2D solver runge-kutta scheme

        Mesh velocity is solved from the Runge-Kutta coefficients of the
        implicit 2D solver.

        :arg i_stage: state of the Runge-Kutta iteration
        """
        fields = self.solver.fields

        self.solver.elev_2d_to_cg_projector.project()
        self.elev_cg_old_2d[i_stage + 1].assign(fields.elev_cg_2d)

        w_mesh = fields.w_mesh_surf_2d
        w_mesh.assign(0.0)
        # stage consistent mesh velocity is obtained from inv bucher tableau
        for j in range(i_stage + 1):
            x_j = self.elev_cg_old_2d[j + 1]
            x_0 = self.elev_cg_old_2d[0]
            w_mesh += self. a_inv[i_stage, j]*(x_j - x_0)/self.solver.dt

        # use that to compute w_mesh in whole domain
        self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
        # solve w_mesh at nodes
        w_mesh_surf = fields.w_mesh_surf_3d.dat.data[:]
        z_ref = fields.z_coord_ref_3d.dat.data[:]
        h = fields.bathymetry_3d.dat.data[:]
        fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref/h + 1.0)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for k in range(self.n_stages):
            # IMEX update
            # - EX: set solution to u_n + dt*sum(a*k_erk)
            # - IM: solve implicit tendency (this is implicit solve)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            # - EX: evaluate explicit tendency

            # - EX: set solution to u_n + dt*sum(a*k_erk)
            if k > 0:
                self.timesteppers.swe2d.erk.update_solution(k)
                self.timesteppers.mom_expl.update_solution(k)
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.update_solution(k)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                # TODO need to update all dependencies here
                self._update_3d_elevation()
                self._update_vertical_velocity()

            self._compute_mesh_velocity_pre(k)
            # - IM: solve implicit tendency (this is implicit solve)
            self.timesteppers.swe2d.dirk.solve_tendency(k, t, update_forcings3d)
            # - IM: set solution to u_n + dt*sum(a*k_erk) + *sum(a*k_dirk)
            self.timesteppers.swe2d.dirk.update_solution(k)
            self.compute_mesh_velocity(k)
            # TODO update all dependencies of implicit solutions here
            # NOTE if 3D implicit solves, must be done in new mesh!
            self._update_3d_elevation()

            # - EX: evaluate explicit tendency
            self.timesteppers.mom_expl.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.solve_tendency(k, t, update_forcings3d)
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.solve_tendency(k, t, update_forcings3d)

            last_step = (k == self.n_stages - 1)
            if last_step:
                self.timesteppers.swe2d.get_final_solution()
                self._update_3d_elevation()
            self._update_moving_mesh()
            if last_step:
                self.timesteppers.mom_expl.get_final_solution()
                self._update_vertical_velocity()
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.get_final_solution()
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.get_final_solution()
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.get_final_solution()

            self._update_2d_coupling()
            self._update_vertical_velocity()


class CoupledLeapFrogAM3(CoupledTimeIntegrator):
    """
    Leap-Frog Adams-Moulton 3 time integrator for coupled 2D-3D problem

    This is an ALE time integrator.
    Implementation follows the SLIM time integrator by Karna et al (2013)

    Karna, et al. (2013). A baroclinic discontinuous Galerkin finite element
    model for coastal flows. Ocean Modelling, 61(0):1-20.
    http://dx.doi.org/10.1016/j.ocemod.2012.09.009
    """
    integrator_2d = rungekutta.DIRK22
    integrator_3d = timeintegrator.LeapFrogAM3
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledLeapFrogAM3, self).__init__(solver)
        self.elev_old_3d = Function(self.fields.elev_3d)
        self.uv_old_2d = Function(self.fields.uv_2d)
        self.uv_new_2d = Function(self.fields.uv_2d)

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        # -------------------------------------------------
        # Prediction step
        # - from t_{n-1/2} to t_{n+1/2}
        # - RHS evaluated at t_{n}
        # - Forward Euler step in fixed domain Omega_n
        # -------------------------------------------------

        if self.options.use_ale_moving_mesh:
            self.fields.w_mesh_3d.assign(0.0)

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.predict()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.predict()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.predict()

        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.predict()

        # dependencies for 2D update
        self._update_2d_coupling()
        self._update_baroclinicity()
        self._update_bottom_friction()

        # update 2D
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
        self.uv_old_2d.assign(self.fields.uv_2d)
        with timed_stage('mode2d'):
            self.timesteppers.swe2d.advance(t, update_forcings)
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_finalize()
        self.uv_new_2d.assign(self.fields.uv_2d)

        # set 3D elevation to half step
        gamma = self.timesteppers.mom_expl.gamma
        self.elev_old_3d.assign(self.fields.elev_3d)
        self.solver.copy_elev_to_3d.solve()
        self.fields.elev_3d *= (0.5 + 2*gamma)
        self.fields.elev_3d += (0.5 - 2*gamma)*self.elev_old_3d

        # correct uv_3d to uv_2d at t_{n+1/2}
        self.fields.uv_2d *= (0.5 + 2*gamma)
        self.fields.uv_2d += (0.5 - 2*gamma)*self.uv_old_2d
        self._update_2d_coupling()
        self.fields.uv_2d.assign(self.uv_new_2d)  # restore
        self._update_vertical_velocity()
        self._update_baroclinicity()

        # -------------------------------------------------
        # Correction step
        # - from t_{n} to t_{n+1}
        # - RHS evaluated at t_{n+1/2}
        # - Forward Euler ALE step from Omega_n to Omega_{n+1}
        # -------------------------------------------------

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.eval_rhs()
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.eval_rhs()
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.eval_rhs()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.eval_rhs()
        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.eval_rhs()

        self._update_3d_elevation()
        self._update_moving_mesh()

        with timed_stage('salt_eq'):
            if self.options.solve_salinity:
                self.timesteppers.salt_expl.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.salt_3d)
        with timed_stage('temp_eq'):
            if self.options.solve_temperature:
                self.timesteppers.temp_expl.correct()
                if self.options.use_limiter_for_tracers:
                    self.solver.tracer_limiter.apply(self.fields.temp_3d)
        with timed_stage('turb_advection'):
            if 'psi_expl' in self.timesteppers:
                self.timesteppers.psi_expl.correct()
            if 'tke_expl' in self.timesteppers:
                self.timesteppers.tke_expl.correct()
        with timed_stage('momentum_eq'):
            self.timesteppers.mom_expl.correct()

        if self.options.use_implicit_vertical_diffusion:
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            self._update_turbulence(t)
            if self.options.solve_salinity:
                with timed_stage('impl_salt_vdiff'):
                    self.timesteppers.salt_impl.advance(t)
            if self.options.solve_temperature:
                with timed_stage('impl_temp_vdiff'):
                    self.timesteppers.temp_impl.advance(t)
            with timed_stage('impl_mom_vvisc'):
                self.timesteppers.mom_impl.advance(t)
            self._update_baroclinicity()
            self._update_vertical_velocity()
            self._update_stabilization_params()
        else:
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            self._update_vertical_velocity()
            self._update_stabilization_params()


class CoupledTwoStageRK(CoupledTimeIntegrator):
    """
    Coupled time integrator based on SSPRK(2,2) scheme

    This ALE time integration method uses SSPRK(2,2) scheme to advance the 3D
    equations and a compatible implicit Trapezoid method to advance the 2D
    equations. Backward Euler scheme is used for vertical diffusion.
    """
    integrator_2d = rungekutta.ESDIRKTrapezoid
    integrator_3d = timeintegrator.SSPRK22ALE
    integrator_vert_3d = rungekutta.BackwardEuler

    def __init__(self, solver):
        super(CoupledTwoStageRK, self).__init__(solver)
        # allocate CG elevation fields for storing mesh geometry
        self.elev_fields = []
        for i in range(self.n_stages):
            e = Function(self.fields.elev_cg_2d)
            self.elev_fields.append(e)

    def store_elevation(self, istage):
        """
        Store current elevation field for computing mesh velocity

        Must be called before updating the 2D mode.

        :arg istage: stage of the Runge-Kutta iteration
        :type istage: int
        """
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
            self.elev_fields[istage].assign(self.fields.elev_cg_2d)

    def compute_mesh_velocity(self, istage):
        """
        Computes mesh velocity for stage i

        Must be called after updating the 2D mode.

        :arg istage: stage of the Runge-Kutta iteration
        :type istage: int
        """
        if self.options.use_ale_moving_mesh:
            self.solver.mesh_updater.compute_mesh_velocity_begin()
            current_elev = self.fields.elev_cg_2d
            if istage == 0:
                # compute w_mesh at surface as (elev^{(1)} - elev^{n})/dt
                w_s = (current_elev - self.elev_fields[0])/self.solver.dt
            else:
                # compute w_mesh at surface as (2*elev^{n+1} - elev^{(1)} - elev^{n})/dt
                w_s = (2*current_elev - self.elev_fields[1] - self.elev_fields[0])/self.solver.dt
            self.fields.w_mesh_surf_2d.assign(w_s)
            # use that to compute w_mesh in whole domain
            self.solver.mesh_updater.cp_w_mesh_surf_2d_to_3d.solve()
            # solve w_mesh at nodes
            w_mesh_surf = self.fields.w_mesh_surf_3d.dat.data[:]
            z_ref = self.fields.z_coord_ref_3d.dat.data[:]
            h = self.fields.bathymetry_3d.dat.data[:]
            self.fields.w_mesh_3d.dat.data[:] = w_mesh_surf * (z_ref + h)/h

    def advance(self, t, update_forcings=None, update_forcings3d=None):
        """
        Advances the equations for one time step

        :arg float t: simulation time
        :kwarg update_forcings: Optional user-defined function that takes
            simulation time and updates time-dependent boundary conditions of
            the 2D equations.
        :kwarg update_forcings3d: Optional user defined function that updates
            boundary conditions of the 3D equations
        """
        if not self._initialized:
            self.initialize()

        for i_stage in range(self.n_stages):

            # solve 2D mode
            self.store_elevation(i_stage)
            with timed_stage('mode2d'):
                self.timesteppers.swe2d.solve_stage(i_stage, t, update_forcings)
            self.compute_mesh_velocity(i_stage)

            # solve 3D mode: preprocess in old mesh
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.prepare_stage(i_stage, t, update_forcings3d)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.prepare_stage(i_stage, t, update_forcings3d)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.prepare_stage(i_stage, t, update_forcings3d)

            # update mesh
            self._update_3d_elevation()
            self._update_moving_mesh()

            # solve 3D mode
            with timed_stage('salt_eq'):
                if self.options.solve_salinity:
                    self.timesteppers.salt_expl.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.salt_3d)
            with timed_stage('temp_eq'):
                if self.options.solve_temperature:
                    self.timesteppers.temp_expl.solve_stage(i_stage)
                    if self.options.use_limiter_for_tracers:
                        self.solver.tracer_limiter.apply(self.fields.temp_3d)
            with timed_stage('turb_advection'):
                if 'psi_expl' in self.timesteppers:
                    self.timesteppers.psi_expl.solve_stage(i_stage)
                if 'tke_expl' in self.timesteppers:
                    self.timesteppers.tke_expl.solve_stage(i_stage)
            with timed_stage('momentum_eq'):
                self.timesteppers.mom_expl.solve_stage(i_stage)

            # update coupling terms
            last = i_stage == self.n_stages - 1

            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_bottom_friction()
            if i_stage == last and self.options.use_implicit_vertical_diffusion:
                self._update_turbulence(t)
                if self.options.solve_salinity:
                    with timed_stage('impl_salt_vdiff'):
                        self.timesteppers.salt_impl.advance(t)
                if self.options.solve_temperature:
                    with timed_stage('impl_temp_vdiff'):
                        self.timesteppers.temp_impl.advance(t)
                with timed_stage('impl_mom_vvisc'):
                    self.timesteppers.mom_impl.advance(t)
                self._update_baroclinicity()
            self._update_vertical_velocity()
            if i_stage == last:
                self._update_stabilization_params()
