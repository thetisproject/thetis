"""
Time integrators for solving coupled 2D-3D system of equations.
"""
from .utility import *
from . import timeintegrator
from .log import *
from . import rungekutta
from abc import ABC, abstractproperty
import numpy


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
            self.solver.fields.elev_domain_2d.assign(self.solver.fields.elev_2d)

    def _update_vertical_velocity(self):
        """Solve vertical velocity"""
        with timed_stage('continuity_eq'):
            self.solver.w_solver.solve()

    def _update_moving_mesh(self):
        """Updates 3D mesh to match elevation field"""
        if self.options.use_ale_moving_mesh:
            with timed_stage('aux_mesh_ale'):
                self.solver.mesh_updater.update_mesh_coordinates()

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


class CoupledTimeIntegrator(CoupledTimeIntegratorBase, ABC):
    """
    Base class of mode-split time integrators that use 2D, 3D and implicit 3D
    time integrators.
    """
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
            'coriolis': self.options.coriolis_frequency,
            'momentum_source': momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
            'atmospheric_pressure': self.options.atmospheric_pressure,
        }

        self.timesteppers.swe2d = self.integrator_2d(
            solver.equations.sw, self.fields.solution_2d,
            fields, solver.dt, self.options.timestepper_options.swe_options,
            solver.bnd_functions['shallow_water'])

    def _create_mom_integrator(self):
        """
        Create time integrator for 3D momentum equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        fields = {'eta': self.fields.elev_domain_2d.view_3d,  # FIXME rename elev
                  'int_pg': self.fields.get('int_pg_3d'),
                  'uv_depth_av': self.fields.get('uv_dav_3d'),
                  'w': self.fields.w_3d,
                  'w_mesh': self.fields.get('w_mesh_3d'),
                  'viscosity_v': expl_v_visc,
                  'viscosity_h': self.solver.tot_h_visc.get_sum(),
                  'source': self.options.momentum_source_3d,
                  'lax_friedrichs_velocity_scaling_factor': self.options.lax_friedrichs_velocity_scaling_factor,
                  'coriolis': self.fields.get('coriolis_3d'),
                  }
        friction_fields = {
            'linear_drag_coefficient': self.options.linear_drag_coefficient,
            'quadratic_drag_coefficient': self.options.quadratic_drag_coefficient,
            'wind_stress': self.fields.get('wind_stress_3d'),
            'bottom_roughness': self.options.bottom_roughness,
        }
        if not self.solver.options.use_implicit_vertical_diffusion:
            fields.update(friction_fields)
        self.timesteppers.mom_expl = self.integrator_3d(
            solver.equations.momentum, solver.fields.uv_3d, fields, solver.dt,
            self.options.timestepper_options.explicit_momentum_options, solver.bnd_functions['momentum'])
        if self.solver.options.use_implicit_vertical_diffusion:
            fields = {'viscosity_v': impl_v_visc}
            fields.update(friction_fields)
            self.timesteppers.mom_impl = self.integrator_vert_3d(
                solver.equations.vertmomentum, solver.fields.uv_3d, fields, solver.dt,
                self.options.timestepper_options.implicit_momentum_options, solver.bnd_functions['momentum'])

    def _create_salt_integrator(self):
        """
        Create time integrator for salinity equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        if self.solver.options.solve_salinity:
            fields = {'elev_3d': self.fields.elev_domain_2d.view_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': expl_v_diff,
                      'source': self.options.salinity_source_3d,
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.salt_expl = self.integrator_3d(
                solver.equations.salt, solver.fields.salt_3d, fields, solver.dt,
                self.options.timestepper_options.explicit_tracer_options, solver.bnd_functions['salt'])
            if self.solver.options.use_implicit_vertical_diffusion:
                fields = {'elev_3d': self.fields.elev_domain_2d.view_3d,
                          'diffusivity_v': impl_v_diff,
                          }
                self.timesteppers.salt_impl = self.integrator_vert_3d(
                    solver.equations.salt_vdff, solver.fields.salt_3d, fields, solver.dt,
                    self.options.timestepper_options.implicit_tracer_options, solver.bnd_functions['salt'])

    def _create_temp_integrator(self):
        """
        Create time integrator for temperature equation
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        if self.solver.options.solve_temperature:
            fields = {'elev_3d': self.fields.elev_domain_2d.view_3d,
                      'uv_3d': self.fields.uv_3d,
                      'uv_depth_av': self.fields.get('uv_dav_3d'),
                      'w': self.fields.w_3d,
                      'w_mesh': self.fields.get('w_mesh_3d'),
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'diffusivity_v': expl_v_diff,
                      'source': self.options.temperature_source_3d,
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.temp_expl = self.integrator_3d(
                solver.equations.temp, solver.fields.temp_3d, fields, solver.dt,
                self.options.timestepper_options.explicit_tracer_options, solver.bnd_functions['temp'])
            if self.solver.options.use_implicit_vertical_diffusion:
                fields = {'elev_3d': self.fields.elev_domain_2d.view_3d,
                          'diffusivity_v': impl_v_diff,
                          }
                self.timesteppers.temp_impl = self.integrator_vert_3d(
                    solver.equations.temp_vdff, solver.fields.temp_3d, fields, solver.dt,
                    self.options.timestepper_options.implicit_tracer_options, solver.bnd_functions['temp'])

    def _create_turb_integrator(self):
        """
        Create time integrators for turbulence equations
        """
        solver = self.solver
        impl_v_visc, expl_v_visc, impl_v_diff, expl_v_diff = self._get_vert_diffusivity_functions()

        eq_tke_diff = getattr(self.solver.equations, 'tke_diff', None)
        eq_psi_diff = getattr(self.solver.equations, 'psi_diff', None)
        eq_tke_adv = getattr(self.solver.equations, 'tke_adv', None)
        eq_psi_adv = getattr(self.solver.equations, 'psi_adv', None)
        if eq_tke_diff is not None and eq_psi_diff is not None:
            fields = {'diffusivity_v': impl_v_diff,
                      'viscosity_v': impl_v_visc,
                      'k': solver.fields.tke_3d,
                      'epsilon': solver.turbulence_model.epsilon,
                      'shear_freq2': solver.turbulence_model.m2,
                      'buoy_freq2_neg': solver.turbulence_model.n2_neg,
                      'buoy_freq2_pos': solver.turbulence_model.n2_pos,
                      'bottom_roughness': self.options.bottom_roughness,
                      }
            self.timesteppers.tke_impl = self.integrator_vert_3d(
                eq_tke_diff, solver.fields.tke_3d, fields, solver.dt,
                self.options.timestepper_options.implicit_tracer_options, {})
            self.timesteppers.psi_impl = self.integrator_vert_3d(
                eq_psi_diff, solver.fields.psi_3d, fields, solver.dt,
                self.options.timestepper_options.implicit_tracer_options, {})
            if eq_tke_adv is not None and eq_psi_adv is not None:
                fields = {'elev_3d': self.fields.elev_domain_2d.view_3d,
                          'uv_3d': self.fields.uv_3d,
                          'uv_depth_av': self.fields.get('uv_dav_3d'),
                          'w': self.fields.w_3d,
                          'w_mesh': self.fields.get('w_mesh_3d'),
                          'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                          }
                self.timesteppers.tke_expl = self.integrator_3d(
                    eq_tke_adv, solver.fields.tke_3d, fields, solver.dt,
                    self.options.timestepper_options.explicit_tracer_options, {})
                self.timesteppers.psi_expl = self.integrator_3d(
                    eq_psi_adv, solver.fields.psi_3d, fields, solver.dt,
                    self.options.timestepper_options.explicit_tracer_options, {})

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
        assert numpy.isclose(dt/dt_2d, numpy.round(dt/dt_2d)), \
            'dt_2d is not integer fraction of dt'

        if dt != dt_2d:
            raise NotImplementedError('Case dt_2d < dt is not implemented yet')

        for stepper in sorted(self.timesteppers):
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

    @PETSc.Log.EventDecorator("thetis.CoupledLeapFrogAM3.__init__")
    def __init__(self, solver):
        super(CoupledLeapFrogAM3, self).__init__(solver)
        self.elev_domain_old_2d = Function(self.fields.elev_domain_2d)
        self.uv_old_2d = Function(self.fields.uv_2d)
        self.uv_new_2d = Function(self.fields.uv_2d)

    @PETSc.Log.EventDecorator("thetis.CoupledLeapFrogAM3.advance")
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
            if self.options.use_limiter_for_velocity:
                self.solver.uv_limiter.apply(self.fields.uv_3d)

        # dependencies for 2D update
        self._update_2d_coupling()
        self._update_baroclinicity()

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
        self.elev_domain_old_2d.assign(self.fields.elev_domain_2d)
        self.solver.fields.elev_domain_2d.assign(self.solver.fields.elev_2d)
        self.fields.elev_domain_2d *= (0.5 + 2*gamma)
        self.fields.elev_domain_2d += (0.5 - 2*gamma)*self.elev_domain_old_2d

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
            if self.options.use_limiter_for_velocity:
                self.solver.uv_limiter.apply(self.fields.uv_3d)

        if self.options.use_implicit_vertical_diffusion:
            self._update_2d_coupling()
            self._update_baroclinicity()
            self._update_turbulence(t)
            if self.options.solve_salinity:
                with timed_stage('impl_salt_vdiff'):
                    self.timesteppers.salt_impl.advance(t)
            if self.options.solve_temperature:
                with timed_stage('impl_temp_vdiff'):
                    self.timesteppers.temp_impl.advance(t)
            with timed_stage('impl_mom_vvisc'):
                self.fields.uv_3d += self.fields.uv_dav_3d
                self.timesteppers.mom_impl.advance(t)
                self.fields.uv_3d -= self.fields.uv_dav_3d
            self._update_baroclinicity()
            self._update_vertical_velocity()
            self._update_stabilization_params()
        else:
            self._update_2d_coupling()
            self._update_baroclinicity()
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

    @PETSc.Log.EventDecorator("thetis.CoupledTwoStageRK.__init__")
    def __init__(self, solver):
        super(CoupledTwoStageRK, self).__init__(solver)
        # allocate CG elevation fields for storing mesh geometry
        self.elev_fields = []
        for i in range(self.n_stages):
            e = Function(self.fields.elev_cg_2d)
            self.elev_fields.append(e)

    @PETSc.Log.EventDecorator("thetis.CoupledTwoStageRK.store_elevation")
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

    @PETSc.Log.EventDecorator("thetis.CoupledTwoStageRK.compute_mesh_velocity")
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
            self.solver.mesh_updater.compute_mesh_velocity_finalize(
                w_mesh_surf_expr=w_s)

    @PETSc.Log.EventDecorator("thetis.CoupledTwoStageRK.advance")
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
                if self.options.use_limiter_for_velocity:
                    self.solver.uv_limiter.apply(self.fields.uv_3d)

            last_stage = i_stage == self.n_stages - 1

            if last_stage:
                # compute final prognostic variables
                self._update_2d_coupling()
                if self.options.use_implicit_vertical_diffusion:
                    if self.options.solve_salinity:
                        with timed_stage('impl_salt_vdiff'):
                            self.timesteppers.salt_impl.advance(t)
                    if self.options.solve_temperature:
                        with timed_stage('impl_temp_vdiff'):
                            self.timesteppers.temp_impl.advance(t)
                    with timed_stage('impl_mom_vvisc'):
                        # compute full velocity
                        self.fields.uv_3d += self.fields.uv_dav_3d
                        self.timesteppers.mom_impl.advance(t)
                        self.fields.uv_3d -= self.fields.uv_dav_3d
                # compute final diagnostic fields
                self._update_baroclinicity()
                self._update_vertical_velocity()
                # update parametrizations
                self._update_turbulence(t)
                self._update_stabilization_params()
            else:
                # update variables that explict solvers depend on
                self._update_2d_coupling()
                self._update_baroclinicity()
                self._update_vertical_velocity()
