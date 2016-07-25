"""
Testing 3D horizontal viscosity of momemtum against analytical solution.

Tuomas Karna 2015-12-11
"""
from thetis import *
import numpy
from scipy import stats
import pytest


def run(refinement, order=1, element_family='dg-dg', warped_mesh=False, do_export=True):
    print_output('--- running refinement {:}'.format(refinement))
    # domain dimensions - channel in x-direction
    lx = 15.0e3
    ly = 6.0e3/refinement
    area = lx*ly
    depth = 40.0
    h_viscosity = 1.0e3

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement + 1
    ny = 1  # constant -- channel
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # set time steps
    # stable explicit time step for diffusion
    dx = lx/nx
    alpha = 1.0/200.0  # TODO theoretical alpha...
    dt = alpha * dx**2/h_viscosity
    # simulation run time
    t_end = 3000.0
    # initial time
    t_init = 1000.0  # NOTE start from t > 0 for smoother init cond
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
    solverobj.options.element_family = element_family
    solverobj.options.nonlin = False
    solverobj.options.use_ale_moving_mesh = False
    solverobj.options.u_advection = Constant(1.0)
    solverobj.options.no_exports = not do_export
    solverobj.options.outputdir = outputdir
    solverobj.options.t_end = t_end
    solverobj.options.t_export = t_export
    solverobj.options.dt = dt
    solverobj.options.dt_2d = dt_2d
    solverobj.options.solve_salt = False
    solverobj.options.solve_temp = False
    solverobj.options.fields_to_export = ['uv_3d']
    solverobj.options.h_viscosity = Constant(h_viscosity)

    solverobj.create_equations()

    t = t_init  # simulation time

    ana_sol_expr = '0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((x[0] - x0)/sqrt(4*D*t))'
    t_const = Constant(t)
    ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_viscosity, t=t_const)
    ana_uv_expr_2d = Expression((ana_sol_expr, 0.0), u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_viscosity, t=t_const)

    uv_ana = Function(solverobj.function_spaces.U, name='uv analytical')
    uv_ana_p1 = Function(solverobj.function_spaces.P1v, name='uv analytical')

    p1dg_v_ho = VectorFunctionSpace(solverobj.mesh, 'DG', order + 2)
    uv_ana_ho = Function(p1dg_v_ho, name='uv analytical')
    uv_ana.project(ana_uv_expr)

    elev_init = Function(solverobj.function_spaces.H_2d, name='elev init')
    solverobj.assign_initial_conditions(elev=elev_init, uv_2d=ana_uv_expr_2d)

    # export analytical solution
    if do_export:
        out_uv_ana = File(os.path.join(solverobj.options.outputdir, 'uv_ana.pvd'))

    def export_func():
        if do_export:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_viscosity, t=t_const)
            uv_ana.project(ana_uv_expr)
            out_uv_ana.write(uv_ana_p1.project(uv_ana))

    # export initial conditions
    export_func()

    # custom time loop that solves momemtum eq only
    ti = solverobj.timestepper.timestepper_mom_3d
    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.t_export
    while t < t_end - 1e-8:
        ti.advance(t, dt, solverobj.fields.uv_3d)
        t += dt
        i += 1
        if t >= next_export_t - 1e-8:
            print_output('{:3d} i={:5d} t={:8.2f} s uv={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.uv_3d)))
            export_func()
            next_export_t += solverobj.options.t_export
            iexport += 1

    # project analytical solultion on high order mesh
    t_const.assign(t)
    ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, x0=lx/2.0, D=h_viscosity, t=t_const)
    uv_ana_ho.project(ana_uv_expr)
    # compute L2 norm
    l2_err = errornorm(uv_ana_ho, solverobj.fields.uv_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    order = options.get('order', 1)
    options.setdefault('do_export', False)
    space_str = options.get('element_family')
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'h-viscosity'

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot):
        slope_rtol = 0.20
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
            ax.set_title(' '.join([setup_name, field_str, 'order={:}'.format(order), space_str]))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str, space_str])
            imgfile += '.png'
            imgdir = create_directory('plots')
            imgfile = os.path.join(imgdir, imgfile)
            print_output('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert slope > expected_slope*(1 - slope_rtol), err_msg
            print_output('{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    check_convergence(x_log, y_log, order+1, 'uv', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------

# NOTE mimetic elements do not converge optimally, rate is 1.48
# NOTE h. viscosity does not work for p0 yet


@pytest.fixture(params=[True, False], ids=['warped', 'regular'])
def warped(request):
    return request.param


def test_horizontal_viscosity(warped):
    run_convergence([1, 2, 3], order=1, warped_mesh=warped)

# ---------------------------
# run individual setup for debugging
# ---------------------------

if __name__ == '__main__':
    run_convergence([1, 2, 3], order=1, warped_mesh=True, element_family='dg-dg', do_export=True, saveplot=True)
