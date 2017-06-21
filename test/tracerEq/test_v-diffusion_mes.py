"""
Testing 3D vertical diffusion of tracers against analytical solution.

Tuomas Karna 2015-12-14
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
    vertical_diffusivity = 5e-3

    # mesh
    n_layers = 6*refinement
    nx = 3  # constant
    ny = 2  # constant
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # set time steps
    # stable explicit time step for diffusion
    dz = depth/n_layers
    alpha = 1.0/200.0  # TODO theoretical alpha...
    dt = alpha * dz**2/vertical_diffusivity
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
    options.use_limiter_for_tracers = False
    options.horizontal_velocity_scale = Constant(1.0)
    options.no_exports = True
    options.output_directory = outputdir
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.timestep = dt
    options.timestep_2d = dt_2d
    options.solve_salinity = True
    options.use_implicit_vertical_diffusion = implicit
    options.fields_to_export = ['salt_3d']
    options.vertical_diffusivity = Constant(vertical_diffusivity)
    options.update(model_options)

    solverobj.create_equations()

    t = t_init  # simulation time

    ana_sol_expr = '0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((x[2] - z0)/sqrt(4*D*t))'
    t_const = Constant(t)
    ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_diffusivity, t=t_const)

    salt_ana = Function(solverobj.function_spaces.H, name='salt analytical')
    salt_ana_p1 = Function(solverobj.function_spaces.P1, name='salt analytical')

    p1dg_ho = FunctionSpace(solverobj.mesh, 'DG', options.polynomial_degree + 2,
                            vfamily='DG', vdegree=options.polynomial_degree + 2)
    salt_ana_ho = Function(p1dg_ho, name='salt analytical')

    solverobj.assign_initial_conditions(salt=ana_salt_expr)

    # export analytical solution
    if not options.no_exports:
        out_salt_ana = File(os.path.join(options.output_directory, 'salt_ana.pvd'))

    def export_func():
        if not options.no_exports:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_diffusivity, t=t_const)
            salt_ana.project(ana_salt_expr)
            out_salt_ana.write(salt_ana_p1.project(salt_ana))

    # export initial conditions
    export_func()

    # custom time loop that solves tracer eq only
    if implicit:
        ti = solverobj.timestepper.timestepper_salt_vdff_3d
    else:
        ti = solverobj.timestepper.timestepper_salt_3d
    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.simulation_export_time
    while t < t_end - 1e-8:
        ti.advance(t)
        t += solverobj.dt
        i += 1
        if t >= next_export_t - 1e-8:
            print_output('{:3d} i={:5d} t={:8.2f} s salt={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.salt_3d)))
            export_func()
            next_export_t += solverobj.options.simulation_export_time
            iexport += 1

    # project analytical solultion on high order mesh
    t_const.assign(t)
    ana_salt_expr = Expression(ana_sol_expr, u_max=1.0, u_min=-1.0, z0=-depth/2.0, D=vertical_diffusivity, t=t_const)
    salt_ana_ho.project(ana_salt_expr)
    # compute L2 norm
    l2_err = errornorm(salt_ana_ho, solverobj.fields.salt_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    order = options.get('order', 1)
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'v-diffusion'

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
            ax.set_title(' '.join([setup_name, field_str, 'order={:}'.format(order)]))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str])
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

    check_convergence(x_log, y_log, order+1, 'salt', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.fixture(params=[0, 1])
def order(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['implicit', 'explicit'])
def implicit(request):
    return request.param


@pytest.mark.parametrize(('stepper', 'use_ale'),
                         [('ssprk33', False),
                          ('leapfrog', True),
                          ('ssprk22', True)])
def test_vertical_diffusion(order, implicit, stepper, use_ale):
    run_convergence([1, 2, 4], order=order, implicit=implicit,
                    timestepper_type=stepper,
                    use_ale_moving_mesh=use_ale)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_convergence([1, 2, 3], order=1,
                    implicit=False,
                    element_family='dg-dg',
                    timestepper_type='ssprk22',
                    use_ale_moving_mesh=True,
                    no_exports=False, saveplot=True)
