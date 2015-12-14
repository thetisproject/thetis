"""
Testing 3D horizontal diffusion of tracers against analytical solution.

Tuomas Karna 2015-12-14
"""
from thetis import *
import numpy
from scipy import stats
import pytest


def run(refinement, order=1, warped_mesh=False, do_export=True):
    print '--- running refinement {:}'.format(refinement)
    # domain dimensions - channel in x-direction
    lx = 15.0e3
    ly = 7.0e3/refinement
    area = lx*ly
    depth = 20.0
    h_diffusivity = 1.0e3

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement + 1
    ny = 1  # constant -- channel
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # set time steps
    # stable explicit time step for diffusion
    dx = lx/nx
    alpha = 1.0/150.0  # TODO theoretical alpha...
    dt = alpha * dx**2/h_diffusivity
    # simulation run time
    t_end = 3600.0/2
    # initial time
    t_init = 100.0  # NOTE start from t > 0 for smoother init cond
    # eliminate reminder
    ndt = np.ceil((t_end-t_init)/dt)
    dt = (t_end-t_init)/ndt
    dt_2d = dt/2
    t_export = (t_end-t_init)/20.0

    # outputs
    outputdir = 'outputs'

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)
    if warped_mesh:
        # linear bathymetry and elevation
        # NOTE should be linear so all meshes can fully resolve geometry
        bathymetry_2d.interpolate(Expression('h + 20.0*x[0]/lx', h=depth, lx=lx))

    solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    solverobj.options.order = order
    solverobj.options.mimetic = False
    solverobj.options.nonlin = False
    solverobj.options.use_ale_moving_mesh = True
    solverobj.options.u_advection = Constant(1.0)
    solverobj.options.no_exports = not do_export
    solverobj.options.outputdir = outputdir
    solverobj.options.t_end = t_end
    solverobj.options.t_export = t_export
    solverobj.options.dt = dt
    solverobj.options.dt_2d = dt_2d
    solverobj.options.solve_salt = True
    solverobj.options.solve_vert_diffusion = False
    solverobj.options.use_limiter_for_tracers = False
    solverobj.options.fields_to_export = ['salt_3d']
    solverobj.options.h_diffusivity = Constant(h_diffusivity)

    solverobj.create_equations()

    t = t_init  # simulation time

    ana_sol_expr = '0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((x[0] - x0)/sqrt(4*D*t))'
    t_const = Constant(t)
    ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_diffusivity, t=t_const)

    salt_ana = Function(solverobj.function_spaces.H, name='salt analytical')
    salt_ana_p1 = Function(solverobj.function_spaces.P1, name='salt analytical')

    p1dg_ho = FunctionSpace(solverobj.mesh, 'DG', order + 2)
    salt_ana_ho = Function(p1dg_ho, name='salt analytical')

    elev_init = Function(solverobj.function_spaces.H_2d, name='elev init')
    if warped_mesh:
        elev_init.interpolate(Expression('20.0*x[0]/lx', h=depth, lx=lx))
    solverobj.assign_initial_conditions(elev=elev_init, salt=ana_salt_expr)

    # export analytical solution
    if do_export:
        out_salt_ana = File(os.path.join(solverobj.options.outputdir, 'salt_ana.pvd'))

    def export_func():
        if do_export:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_diffusivity, t=t_const)
            salt_ana.project(ana_salt_expr)
            out_salt_ana << salt_ana_p1.project(salt_ana)

    # export initial conditions
    export_func()

    # custom time loop that solves tracer equation only
    ti = solverobj.timestepper.timestepper_salt_3d
    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.t_export
    while t < t_end - 1e-8:
        with timed_region('tracersolver'):
            ti.advance(t, dt, solverobj.fields.salt_3d)
        t += dt
        i += 1
        if t >= next_export_t - 1e-8:
            cpu_t = timing('tracersolver', reset=True)
            print('{:3d} i={:5d} t={:8.2f} s salt={:8.2f} cpu={:4.1f} s'.format(iexport, i, t, norm(solverobj.fields.salt_3d), cpu_t))
            export_func()
            next_export_t += solverobj.options.t_export
            iexport += 1

    # project analytical solultion on high order mesh
    t_const.assign(t)
    ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_diffusivity, t=t_const)
    salt_ana_ho.project(ana_salt_expr)
    # compute L2 norm
    l2_err = errornorm(salt_ana_ho, solverobj.fields.salt_3d)/numpy.sqrt(area)
    print 'L2 error {:.12f}'.format(l2_err)

    return l2_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    order = options.get('order', 1)
    options.setdefault('do_export', False)
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'h-diffusion'

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot):
        slope_rtol = 0.05
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if saveplot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*npoints/3], yy[2*npoints/3], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_str)
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str])
            imgfile += '.png'
            imgdir = create_directory('plots')
            imgfile = os.path.join(imgdir, imgfile)
            print 'saving figure', imgfile
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert slope > expected_slope*(1 - slope_rtol), err_msg
            print '{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope)
        else:
            print '{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope)
        return slope

    check_convergence(x_log, y_log, order+1, 'salt', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------

# NOTE h. viscosity does not work for p0 yet


@pytest.fixture(params=[True, False], ids=['warped', 'regular'])
def warped(request):
    return request.param


def test_horizontal_diffusion(warped):
    run_convergence([1, 2, 3], order=1, warped_mesh=warped)

# ---------------------------
# run individual setup for debugging
# ---------------------------

if __name__ == '__main__':
    # run(2, order=1)
    run_convergence([1, 2, 4], order=1, warped_mesh=True, do_export=True, saveplot=True)
