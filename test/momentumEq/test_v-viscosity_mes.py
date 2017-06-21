"""
Testing 3D vertical viscosity of momemtum against analytical solution.

Tuomas Karna 2015-12-11
"""
from thetis import *
import numpy
from scipy import stats
import pytest


def run(refinement, **model_options):
    print_output('--- running refinement {:}'.format(refinement))
    implicit = model_options.pop('implicit', False)
    # domain dimensions - channel in x-direction
    lx = 7.0e3
    ly = 5.0e3
    area = lx*ly
    depth = 40.0
    vertical_viscosity = 5e-3

    # mesh
    n_layers = 6*refinement
    nx = 3  # constant
    ny = 2  # constant
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # set time steps
    # stable explicit time step for diffusion
    dz = depth/n_layers
    alpha = 1.0/200.0  # TODO theoretical alpha...
    dt = alpha * dz**2/vertical_viscosity
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

    solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solverobj.options
    options.use_nonlinear_equations = False
    options.use_ale_moving_mesh = False
    options.horizontal_velocity_scale = Constant(1.0)
    options.no_exports = True
    options.output_directory = outputdir
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.use_automatic_timestep = False
    options.timestep = dt
    options.timestep_2d = dt_2d
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = implicit
    options.fields_to_export = ['uv_3d']
    options.vertical_viscosity = Constant(vertical_viscosity)
    options.update(model_options)

    solverobj.create_equations()

    t = t_init  # simulation time

    ana_sol_expr = '0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((x[2] - z0)/sqrt(4*D*t))'
    t_const = Constant(t)
    ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_viscosity, t=t_const)

    uv_ana = Function(solverobj.function_spaces.U, name='uv analytical')
    uv_ana_p1 = Function(solverobj.function_spaces.P1v, name='uv analytical')

    p1dg_v_ho = VectorFunctionSpace(solverobj.mesh, 'DG', options.polynomial_degree + 2,
                                    vfamily='DG', vdegree=options.polynomial_degree + 2)
    uv_ana_ho = Function(p1dg_v_ho, name='uv analytical')
    uv_ana.project(ana_uv_expr)

    solverobj.fields.uv_3d.project(ana_uv_expr)
    # export analytical solution
    if not options.no_exports:
        out_uv_ana = File(os.path.join(options.output_directory, 'uv_ana.pvd'))

    def export_func():
        if not options.no_exports:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_viscosity, t=t_const)
            uv_ana.project(ana_uv_expr)
            out_uv_ana.write(uv_ana_p1.project(uv_ana))

    # compute L2 norm
    uv_ana_ho.project(ana_uv_expr)
    l2_err = errornorm(uv_ana_ho, solverobj.fields.uv_3d)/numpy.sqrt(area)
    print_output('Initial norm uv     {:.12f}'.format(norm(solverobj.fields.uv_3d)))
    print_output('Initial norm uv ana {:.12f}'.format(norm(uv_ana_ho)))
    print_output('Initial L2 error {:.12f}'.format(l2_err))

    # custom time loop that solves momemtum eq only
    if implicit:
        ti = solverobj.timestepper.timestepper_mom_vdff_3d
    else:
        ti = solverobj.timestepper.timestepper_mom_3d
    ti.initialize(solverobj.fields.uv_3d)

    i = 0
    iexport = 0
    next_export_t = t + solverobj.options.simulation_export_time
    # export initial conditions
    print_output('{:3d} i={:5d} t={:8.2f} s uv={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.uv_3d)))
    export_func()

    while t < t_end - 1e-8:
        ti.advance(t)

        t += solverobj.dt
        i += 1
        if t >= next_export_t - 1e-8:
            iexport += 1
            print_output('{:3d} i={:5d} t={:8.2f} s uv={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.uv_3d)))
            export_func()
            next_export_t += solverobj.options.simulation_export_time

    # project analytical solultion on high order mesh
    t_const.assign(t)
    ana_uv_expr = Expression((ana_sol_expr, 0.0, 0.0), u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_viscosity, t=t_const)
    uv_ana_ho.project(ana_uv_expr)
    # compute L2 norm
    l2_err = errornorm(uv_ana_ho, solverobj.fields.uv_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    order = options.get('order', 1)
    space_str = options.get('element_family')
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'v-viscosity'

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot):
        slope_rtol = 0.07
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


@pytest.fixture(params=[pytest.mark.not_travis(reason='travis timeout')('rt-dg'), 'dg-dg'])
def element_family(request):
    return request.param


@pytest.fixture(params=[0, 1], ids=['order-0', 'order-1'])
def order(request):
    return request.param


@pytest.fixture(params=[pytest.mark.skip(reason='mysterious travis bug')(True), False], ids=['implicit', 'explicit'])
def implicit(request):
    return request.param


@pytest.mark.parametrize(('stepper', 'use_ale'),
                         [pytest.mark.not_travis(reason='travis timeout')(('ssprk33', False)),
                          ('leapfrog', True),
                          ('ssprk22', True)])
def test_vertical_viscosity(order, implicit, element_family, stepper, use_ale):
    run_convergence([1, 2, 3], order=order, implicit=implicit,
                    element_family=element_family,
                    timestepper_type=stepper, use_ale_moving_mesh=use_ale)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_convergence([1, 2, 3], order=1,
                    implicit=True,
                    element_family='dg-dg',
                    timestepper_type='ssprk33',
                    use_ale_moving_mesh=False,
                    no_exports=True, saveplot=False)
