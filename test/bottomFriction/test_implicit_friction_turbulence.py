"""
Tests implicit bottom friction formulation
==========================================

Tuomas Karna 2015-09-16
"""
from thetis import *
import time as time_mod
import pytest


@pytest.mark.skipif(True, reason='test takes too long to execute')
def test_implicit_friction_turbulence(do_assert=True, do_export=False):
    physical_constants['z0_friction'] = 1.5e-3

    outputdir = 'outputs'
    # set mesh resolution
    scale = 1000.0
    reso = 2.5*scale
    layers = 50
    depth = 15.0

    # generate unit mesh and transform its coords
    x_max = 5.0*scale
    x_min = -5.0*scale
    lx = (x_max - x_min)
    n_x = int(lx/reso)
    mesh2d = RectangleMesh(n_x, n_x, lx, lx, reorder=True)

    print_info('Exporting to ' + outputdir)
    dt = 25.0  # 25.0
    t_end = 12 * 3600.0
    t_export = 100.0
    depth = 15.0
    u_mag = 1.0

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(p1_2d, name='Bathymetry')
    bathymetry2d.assign(depth)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.nonlin = False
    options.solve_salt = False
    options.solve_vert_diffusion = True
    options.use_bottom_friction = True
    # options.use_parabolic_viscosity = True
    options.use_turbulence = True
    options.use_ale_moving_mesh = False
    options.use_limiter_for_tracers = True
    options.uv_lax_friedrichs = Constant(1.0)
    options.tracer_lax_friedrichs = Constant(0.0)
    # options.v_viscosity = Constant(0.001)
    # options.h_viscosity = Constant(1.0)
    options.t_export = t_export
    options.dt = dt
    options.t_end = t_end
    options.no_exports = not do_export
    options.outputdir = outputdir
    options.u_advection = u_mag
    options.check_salt_deviation = True
    options.timer_labels = ['mode2d', 'momentum_eq', 'vert_diffusion', 'turbulence']
    # options.fields_to_export = []
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'uv_dav_2d', 'uv_bottom_2d',
                                'parab_visc_3d', 'eddy_visc_3d', 'shear_freq_3d',
                                'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
    # options.fields_to_export_numpy = ['uv3d', 'eddy_visc_3d', 'shear_freq_3d',
    #                                'tke3d', 'psi3d', 'eps3d', 'len3d']
    solver_obj.create_equations()

    elev_slope = -1.0e-5
    pressure_gradient_source = Constant((-9.81*elev_slope, 0, 0))

    s = solver_obj
    vert_mom_eq = momentum_equation.VerticalMomentumEquation(
        s.fields.uv_3d, w=None,
        viscosity_v=s.tot_v_visc.get_sum(),
        uv_bottom=s.fields.uv_bottom_3d,
        bottom_drag=s.fields.bottom_drag_3d,
        wind_stress=s.fields.get('wind_stress_3d'),
        v_elem_size=s.fields.v_elem_size_3d,
        source=pressure_gradient_source)

    sp = {}
    sp['ksp_type'] = 'gmres'
    # sp['pc_type'] = 'lu'
    # sp['snes_monitor'] = True
    # sp['snes_converged_reason'] = True
    # sp['snes_rtol'] = 1e-4  # to avoid stagnation
    sp['snes_rtol'] = 1e-18  # to avoid stagnation
    sp['ksp_rtol'] = 1e-22  # to avoid stagnation
    timestepper = timeintegrator.DIRKLSPUM2(vert_mom_eq, dt, solver_parameters=sp)
    # timestepper = timeintegrator.BackwardEuler(vert_mom_eq, dt, solver_parameters=sp)

    # TODO fix momemtum eq for parabolic visc
    # TODO mimic gotm implementation

    t = 0
    n_steps = int(np.round(t_end/dt))
    for it in range(n_steps):
        t = it*dt
        t0 = time_mod.clock()
        # momentum_eq
        timestepper.advance(t, dt, s.fields.uv_3d)
        s.uv_p1_projector.project()
        # update bottom friction
        compute_bottom_friction(
            s,
            s.fields.uv_p1_3d, s.fields.uv_bottom_2d,
            s.fields.uv_bottom_3d, s.fields.z_coord_3d,
            s.fields.z_bottom_2d,
            s.fields.bathymetry_2d, s.fields.bottom_drag_2d,
            s.fields.bottom_drag_3d,
            s.fields.v_elem_size_2d, s.fields.v_elem_size_3d)
        # update viscosity
        s.gls_model.preprocess()
        # NOTE psi must be solved first as it depends on tke
        s.timestepper.timestepper_psi_3d.advance(t, s.dt, s.fields.psi_3d)
        s.timestepper.timestepper_tke_3d.advance(t, s.dt, s.fields.tke_3d)
        s.gls_model.postprocess()
        t1 = time_mod.clock()
        # NOTE vtk exporter has a memory leak if output space is DG
        s.export()
        print '{:4d}  T={:9.1f} s  cpu={:7.2f} s uv:{:}'.format(it, t, t1-t0, norm(s.fields.uv_3d))

    if do_assert:
        target_u_min = 0.5
        target_u_max = 0.9
        target_u_tol = 5.0e-2
        target_zero = 1e-8
        solution_p1_dg = Function(s.P1DGv, name='velocity p1dg')
        solution_p1_dg.project(s.uv3d)
        uvw = solution_p1_dg.dat.data
        w_max = np.max(np.abs(uvw[:, 2]))
        v_max = np.max(np.abs(uvw[:, 1]))
        print 'w', w_max
        print 'v', v_max
        assert w_max < target_zero, 'z velocity component too large'
        assert v_max < target_zero, 'y velocity component too large'
        u_min = uvw[:, 0].min()
        u_max = uvw[:, 0].max()
        print 'u', u_min, u_max
        assert np.abs(u_min - target_u_min) < target_u_tol, 'minimum u velocity is wrong {:} != {:}'.format(u_min, target_u_min)
        assert np.abs(u_max - target_u_max) < target_u_tol, 'maximum u velocity is wrong {:} != {:}'.format(u_max, target_u_max)

if __name__ == '__main__':
    test_implicit_friction_turbulence(do_assert=True, do_export=True)
