"""
Testing 3D tracer advection-diffusion equation with method of manufactured solution (MMS).

Tuomas Karna 2015-11-28
"""
from thetis import *
import numpy
from scipy import stats


def setup1(lx, ly, h0, kappa0, element_family='rt-dg'):
    """
    Constant bathymetry and u velocty, zero diffusivity, non-trivial tracer
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('1.0',
         '0.0',
         '0.0', ))
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0', ))
    out['tracer_expr'] = Expression(
        'sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/lx)',
        lx=lx)
    out['kappa_expr'] = Expression('0.0')
    out['res_expr'] = Expression(
        '0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/lx)/lx',
        lx=lx)
    out['options'] = {'element_family': element_family}
    return out


def setup1dg(lx, ly, h0, kappa0):
    """
    Constant bathymetry and u velocty, zero diffusivity, non-trivial tracer
    """
    return setup1(lx, ly, h0, kappa0, element_family='dg-dg')


def setup2(lx, ly, h0, kappa0, element_family='rt-dg'):
    """
    constant bathymetry, zero velocity, constant kappa, x-varying T
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('0.0',
         '0.0',
         '0.0', ))
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0', ))
    out['tracer_expr'] = Expression(
        'sin(3*pi*x[0]/lx)',
        lx=lx)
    out['kappa_expr'] = Expression(
        'kappa0',
        kappa0=kappa0)
    out['res_expr'] = Expression(
        '9*(pi*pi)*kappa0*sin(3*pi*x[0]/lx)/(lx*lx)',
        kappa0=kappa0, lx=lx)
    out['options'] = {'element_family': element_family}
    return out


def setup2dg(lx, ly, h0, kappa0):
    """
    constant bathymetry, zero velocity, constant kappa, x-varying T
    """
    return setup2(lx, ly, h0, kappa0, element_family='dg-dg')


def setup3(lx, ly, h0, kappa0, element_family='rt-dg'):
    """
    constant bathymetry, zero kappa, non-trivial velocity and T
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('sin(pi*(x[1]/ly + 2*x[0]/lx))*sin(pi*x[2]/h0)',
         'sin(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*sin(pi*x[2]/h0)',
         '0.0', ),
        h0=h0, lx=lx, ly=ly)
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*cos(pi*x[2]/h0)/ly + 0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))/ly + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))*cos(pi*x[2]/h0)/lx + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))/lx', ),
        h0=h0, lx=lx, ly=ly)
    out['tracer_expr'] = Expression(
        '(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))',
        h0=h0, lx=lx, ly=ly)
    out['kappa_expr'] = Expression('0.0')
    out['res_expr'] = Expression(
        '-0.4*pi*(0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*cos(pi*x[2]/h0)/ly + 0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))/ly + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))*cos(pi*x[2]/h0)/lx + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))/lx)*sin(0.5*pi*x[2]/h0)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))/h0 - 0.75*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*sin(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))*sin(pi*x[2]/h0)/ly - 1.5*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))*sin(pi*(x[1]/ly + 2*x[0]/lx))*sin(pi*x[2]/h0)/lx',
        h0=h0, lx=lx, ly=ly)
    out['options'] = {'element_family': element_family}
    return out


def setup3dg(lx, ly, h0, kappa0):
    """
    constant bathymetry, zero kappa, non-trivial velocity and T
    """
    return setup3(lx, ly, h0, kappa0, element_family='dg-dg')


def setup4(lx, ly, h0, kappa0, element_family='rt-dg'):
    """
    constant bathymetry, constant kappa, non-trivial velocity and T
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('sin(pi*(x[1]/ly + 2*x[0]/lx))*sin(pi*x[2]/h0)',
         'sin(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*sin(pi*x[2]/h0)',
         '0.0', ),
        h0=h0, lx=lx, ly=ly)
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*cos(pi*x[2]/h0)/ly + 0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))/ly + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))*cos(pi*x[2]/h0)/lx + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))/lx', ),
        h0=h0, lx=lx, ly=ly)
    out['tracer_expr'] = Expression(
        '(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))',
        h0=h0, lx=lx, ly=ly)
    out['kappa_expr'] = Expression(
        'kappa0',
        kappa0=kappa0)
    out['res_expr'] = Expression(
        '-0.4*pi*(0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*cos(pi*x[2]/h0)/ly + 0.3*h0*cos(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))/ly + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))*cos(pi*x[2]/h0)/lx + 2*h0*cos(pi*(x[1]/ly + 2*x[0]/lx))/lx)*sin(0.5*pi*x[2]/h0)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))/h0 - 0.75*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.3*x[1]/ly + 0.3*x[0]/lx))*sin(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))*sin(pi*x[2]/h0)/ly - 0.5625*(pi*pi)*kappa0*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))/(ly*ly) - 1.5*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))*sin(pi*(x[1]/ly + 2*x[0]/lx))*sin(pi*x[2]/h0)/lx + 2.25*(pi*pi)*kappa0*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/ly + 1.5*x[0]/lx))/(lx*lx)',
        h0=h0, kappa0=kappa0, lx=lx, ly=ly)
    out['options'] = {'element_family': element_family}
    return out


def setup4dg(lx, ly, h0, kappa0):
    """
    constant bathymetry, constant kappa, non-trivial velocity and T
    """
    return setup4(lx, ly, h0, kappa0, element_family='dg-dg')


def run(setup, refinement, order, do_export=True):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))
    # domain dimensions
    lx = 15e3
    ly = 10e3
    area = lx*ly
    depth = 40.0
    kappa0 = 5.0e2
    t_end = 100.0

    sdict = setup(lx, ly, depth, kappa0)

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # outputs
    outputdir = 'outputs'
    if do_export:
        out_t = File(os.path.join(outputdir, 'T.pvd'))

    # bathymetry
    p1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(sdict['bath_expr'])

    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    solver_obj.options.order = order
    solver_obj.options.u_advection = Constant(1.0)
    solver_obj.options.no_exports = not do_export
    solver_obj.options.outputdir = outputdir
    solver_obj.options.t_end = t_end
    solver_obj.options.fields_to_export = ['salt_3d', 'uv_3d', 'w_3d']
    solver_obj.options.update(sdict['options'])
    options.h_diffusivity = Constant(kappa0)
    options.nu_viscosity = Constant(kappa0)

    solver_obj.create_function_spaces()

    # functions for source terms
    source_salt = Function(solver_obj.function_spaces.H, name='salinity source')
    source_salt.project(sdict['res_expr'])
    solver_obj.options.salt_source_3d = source_salt

    # diffusivuty
    kappa = Function(solver_obj.function_spaces.P1, name='diffusivity')
    kappa.project(sdict['kappa_expr'])

    # analytical solution in high-order space for computing L2 norms
    h_ho = FunctionSpace(solver_obj.mesh, 'DG', order+3)
    trac_ana_ho = Function(h_ho, name='Analytical T')
    trac_ana_ho.project(sdict['tracer_expr'])
    # analytical solution
    trac_ana = Function(solver_obj.function_spaces.H, name='Analytical T')
    trac_ana.project(sdict['tracer_expr'])

    bnd_salt = {'value': trac_ana}
    solver_obj.bnd_functions['salt'] = {1: bnd_salt, 2: bnd_salt,
                                        3: bnd_salt, 4: bnd_salt}
    # NOTE use symmetic uv condition to get correct w
    bnd_mom = {'symm': None}
    solver_obj.bnd_functions['momentum'] = {1: bnd_mom, 2: bnd_mom,
                                            3: bnd_mom, 4: bnd_mom}

    solver_obj.create_equations()
    dt = solver_obj.dt
    # elevation field
    solver_obj.fields.elev_2d.project(sdict['elev_expr'])
    # update mesh and fields
    solver_obj.mesh_updater.update_mesh_coordinates()

    # salinity field
    solver_obj.fields.salt_3d.project(sdict['tracer_expr'])
    # velocity field
    solver_obj.fields.uv_3d.project(sdict['uv_expr'])
    solver_obj.w_solver.solve()

    if do_export:
        out_t.write(trac_ana)
        solver_obj.export()

    # solve salinity advection-diffusion equation with residual source term
    ti = solver_obj.timestepper
    ti.timestepper_salt_3d.initialize(ti.fields.salt_3d)
    t = 0
    while t < t_end - 1e-5:
        for k in range(ti.n_stages):
            last_step = k == ti.n_stages - 1
            ti.timestepper_salt_3d.solve_stage(k, t)
            if ti.options.use_limiter_for_tracers and last_step:
                ti.solver.tracer_limiter.apply(ti.fields.salt_3d)
        t += dt

    if do_export:
        out_t.write(trac_ana)
        solver_obj.export()

    l2_err = errornorm(trac_ana_ho, solver_obj.fields.salt_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(setup, ref_list, order, do_export=False, save_plot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, do_export=do_export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))

    def check_convergence(x_log, y_log, expected_slope, field_str, save_plot):
        slope_rtol = 0.2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        setup_name = setup.__name__
        if save_plot:
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
            img_dir = create_directory('plots')
            imgfile = os.path.join(img_dir, imgfile)
            print_output('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print_output('{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    check_convergence(x_log, y_log, order+1, 'tracer', save_plot)

# NOTE here mimetic option has no effect -- both use p1dg for tracers
# NOTE with diffusivity convergence rate is only 1.7, 2.0 without
# NOTE external-tracer-value BC and symmetric diffusion flux work fine
# TODO add more BCs

# ---------------------------
# standard tests for pytest
# ---------------------------


def test_setup3_dg():
    run_convergence(setup3dg, [1, 2, 3], 1, save_plot=False)


def test_setup4_dg():
    run_convergence(setup4dg, [1, 2, 3], 1, save_plot=False)

# ---------------------------
# run individual setup for debugging
# ---------------------------

# run(setup3, 3, 1)

# ---------------------------
# run individual scaling test
# ---------------------------


if __name__ == '__main__':
    run_convergence(setup3dg, [1, 2, 3], 1, save_plot=True)
