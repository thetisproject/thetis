"""
Testing 2D horizontal advection of tracers
"""
from thetis import *
from scipy import stats
import pytest


def run(refinement, **model_options):
    print_output('--- running refinement {:}'.format(refinement))

    # domain dimensions - channel in x-direction
    lx = 15.0e3
    ly = 6.0e3/refinement
    area = lx*ly
    depth = 40.0
    u = 1.0

    # mesh
    nx = 6*refinement + 1
    ny = 1  # constant -- channel
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # simulation run time
    t_end = 3000.0
    # initial time
    t_init = 0.0
    t_export = (t_end - t_init)/8.0

    # outputs
    outputdir = 'outputs'

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    solverobj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solverobj.options
    options.use_nonlinear_equations = False
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_lax_friedrichs_tracer = False
    options.horizontal_velocity_scale = Constant(abs(u))
    options.no_exports = True
    options.output_directory = outputdir
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    solverobj.create_function_spaces()
    corr_factor = Function(solverobj.function_spaces.H_2d, name='uv tracer factor').interpolate(Constant(1.0))
    options.tracer_advective_velocity_factor = corr_factor
    options.add_tracer_2d('tracer_2d', 'Depth averaged tracer', 'Tracer2d')
    options.use_limiter_for_tracers = True
    options.fields_to_export = ['tracer_2d']
    options.update(model_options)
    uv_expr = as_vector((u, 0))
    bnd_salt_2d = {'value': Constant(0.0), 'uv': uv_expr}
    bnd_uv_2d = {'uv': uv_expr}
    solverobj.bnd_functions['tracer'] = {
        1: bnd_salt_2d,
        2: bnd_salt_2d,
    }
    solverobj.bnd_functions['momentum'] = {
        1: bnd_uv_2d,
        2: bnd_uv_2d,
    }

    solverobj.create_equations()

    t = t_init  # simulation time

    x0 = 0.3*lx
    sigma = 1600.
    xy = SpatialCoordinate(solverobj.mesh2d)
    t_const = Constant(t)
    ana_tracer_expr = exp(-(xy[0] - x0 - u*t_const)**2/sigma**2)
    tracer_ana = Function(solverobj.function_spaces.H_2d, name='tracer analytical')
    uv_init = Function(solverobj.function_spaces.U_2d, name='initial uv')
    uv_init.project(uv_expr)
    solverobj.assign_initial_conditions(uv=uv_init, tracer=ana_tracer_expr)

    # export analytical solution
    if not options.no_exports:
        out_tracer_ana = File(os.path.join(options.output_directory, 'tracer_ana.pvd'))

    def export_func():
        if not options.no_exports:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            out_tracer_ana.write(tracer_ana.project(ana_tracer_expr))

    # export initial conditions
    export_func()

    # custom time loop that solves tracer equation only
    ti = solverobj.timestepper.timesteppers.tracer_2d

    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.simulation_export_time
    while t < t_end - 1e-8:
        ti.advance(t)
        t += solverobj.dt
        i += 1
        if t >= next_export_t - 1e-8:
            print_output('{:3d} i={:5d} t={:8.2f} s salt={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.tracer_2d)))
            export_func()
            next_export_t += solverobj.options.simulation_export_time
            iexport += 1

    # project analytical solution on high order mesh
    t_const.assign(t)

    # compute L2 norm
    l2_err = errornorm(ana_tracer_expr, solverobj.fields.tracer_2d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(ref_list, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""

    polynomial_degree = options.get('polynomial_degree', 1)
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'h-advection'

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
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(' '.join([setup_name, field_str, 'degree={:}'.format(polynomial_degree)]))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            degree_str = 'o{:}'.format(polynomial_degree)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, degree_str])
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

    check_convergence(x_log, y_log, polynomial_degree+1, 'tracer', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.fixture(params=[1])
def polynomial_degree(request):
    return request.param


@pytest.fixture(params=['CrankNicolson', 'ForwardEuler', 'SSPRK33', 'DIRK22', 'DIRK33'])
def stepper(request):
    return request.param


def test_horizontal_advection(polynomial_degree, stepper):
    run_convergence([1, 2, 3], polynomial_degree=polynomial_degree,
                    swe_timestepper_type=stepper, tracer_timestepper_type=stepper)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_convergence([1, 2, 3, 4], polynomial_degree=1,
                    tracer_timestepper_type='CrankNicolson',
                    no_exports=False, saveplot=True)
