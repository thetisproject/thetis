# Tracer box in 3D
# ================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# This version uses a constant tracer to check local/global conservation of tracers.
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
#
# Tuomas Karna 2015-03-11
from thetis import *
import pytest


def run_tracer_consistency(mimetic=False, meshtype='regular', do_export=False):
    t_cycle = 2000.0  # standing wave period
    depth = 50.0
    lx = np.sqrt(9.81*depth)*t_cycle  # wave length
    ly = 3000.0
    nx = 18
    ny = 2
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    salt_value = 4.5
    n_layers = 6
    elev_amp = 1.0
    # estimate of max advective velocity used to estimate time step
    u_mag = Constant(0.5)

    sloped = False
    warped = False
    if meshtype == 'sloped':
        sloped = True
    elif meshtype == 'warped':
        warped = True

    run_description = 'mimetic={:} meshtype={:}'.format(mimetic, meshtype)
    print_info('Running test: ' + run_description)

    suffix = ''
    if sloped:
        suffix = '_sloped'
    if warped:
        suffix = '_warped'
    outputdir = 'outputs' + suffix

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    if sloped:
        bathymetry_2d.interpolate(Expression('h + 20.0*x[0]/lx',
                                             h=depth, lx=lx))

    # set time step, export interval and run duration
    n_steps = 20
    dt = round(float(t_cycle/n_steps))
    t_export = dt
    t_end = t_cycle + 1e-3

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)

    if warped:
        # warp interior mesh, top/bottom surfaces are horizontal
        coords = solver_obj.mesh.coordinates
        z = coords.dat.data[:, 2].copy()
        x = coords.dat.data[:, 0]
        p = 2.5*x/lx + 0.5
        sigma = -depth * (0.5*np.tanh(p*(-2.0*z/depth - 1.0))/np.tanh(p) + 0.5)
        coords.dat.data[:, 2] = sigma

    options = solver_obj.options
    options.nonlin = False
    options.mimetic = False
    options.solve_salt = True
    options.solve_vert_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = False
    options.use_limiter_for_tracers = False
    options.tracer_lax_friedrichs = None
    options.uv_lax_friedrichs = None
    if options.use_mode_split:
        options.dt = dt/5.0
    else:
        options.dt = dt/40.0
    options.t_export = t_export
    options.no_exports = not do_export
    options.t_end = t_end
    options.u_advection = u_mag
    options.check_vol_conservation_2d = True
    options.check_vol_conservation_3d = True
    options.check_salt_conservation = True
    options.check_salt_overshoot = True
    options.outputdir = outputdir
    options.timer_labels = []
    options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d',
                                'uv_dav_2d', 'uv_bottom_2d']

    solver_obj.create_function_spaces()
    elev_init = Function(solver_obj.function_spaces.H_2d)
    elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/lx)', eta_amp=elev_amp,
                                 lx=lx))
    if options.solve_salt:
        salt_init3d = Function(solver_obj.function_spaces.H, name='initial salinity')
        salt_init3d.assign(salt_value)
    else:
        salt_init3d = None

    solver_obj.assign_initial_conditions(elev=elev_init, salt=salt_init3d)
    solver_obj.iterate()

    max_abs_overshoot = np.max(np.abs(np.array(solver_obj.callbacks['salt_3d_overshoot'].value) - salt_value))
    overshoot_tol = 1e-12
    msg = '{:} : Salt overshoots are too large: {:}'.format(run_description, max_abs_overshoot)
    assert max_abs_overshoot < overshoot_tol, msg


@pytest.mark.parametrize('mimetic', [True, False])
@pytest.mark.parametrize('meshtype', ['regular', 'sloped', 'warped'])
def test_consistency_dg_regular(mimetic, meshtype):
    run_tracer_consistency(mimetic=mimetic, meshtype=meshtype, do_export=False)


if __name__ == '__main__':
    run_tracer_consistency(mimetic=False, meshtype='warped', do_export=True)
