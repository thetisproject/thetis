"""
Test vertical velocity against analytical solutions.
"""
from thetis import *
from scipy import stats


class Setup1:
    """
    linear bath and elev, constant u,v
    """
    def bath(self, x, y, lx, ly):
        return 10.0 + 3*x/lx

    def elev(self, x, y, lx, ly):
        return x*y/lx**2

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(1.0),
                Constant(0.3),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return Constant(-3.0/lx)


class Setup2:
    """
    Constant bath and elev, linear u
    """
    def bath(self, x, y, lx, ly):
        return Constant(20.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                x/lx,
                Constant(0.0),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return -z/lx - 20.0/lx


class Setup3:
    """
    Non-trivial bath and elev, uv depends on (x,y)
    """
    def bath(self, x, y, lx, ly):
        return 6.0*cos(pi*sqrt(x**2 + y**2 + 1.0)/lx) + 21.0

    def elev(self, x, y, lx, ly):
        return 5.0*sin(0.4*pi*sqrt(1.5*x**2 + y**2 + 1.0)/lx)

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                sin(0.2*pi*(3.0*x + 1.0*y)/lx),
                0.2*sin(0.2*pi*(3.0*x + 1.0*y)/lx),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return 6.0*pi*x*sin(0.2*pi*(3.0*x + 1.0*y)/lx)*sin(pi*sqrt(x**2 + y**2 + 1.0)/lx)/(lx*sqrt(x**2 + y**2 + 1.0)) + 1.2*pi*y*sin(0.2*pi*(3.0*x + 1.0*y)/lx)*sin(pi*sqrt(x**2 + y**2 + 1.0)/lx)/(lx*sqrt(x**2 + y**2 + 1.0)) - 0.64*pi*z*cos(0.2*pi*(3.0*x + 1.0*y)/lx)/lx + 0.64*pi*(-6.0*cos(pi*sqrt(x**2 + y**2 + 1.0)/lx) - 21.0)*cos(0.2*pi*(3.0*x + 1.0*y)/lx)/lx


def run(setup, refinement, order, do_export=True):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))
    setup_name = setup.__name__
    # domain dimensions
    lx = 15e3
    ly = 10e3
    area = lx*ly

    setup_obj = setup()

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    x_2d, y_2d = SpatialCoordinate(mesh2d)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(setup_obj.bath(x_2d, y_2d, lx, ly))

    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solver_obj.options
    options.polynomial_degree = order
    options.element_family = 'dg-dg'
    options.solve_salinity = False
    options.solve_temperature = False
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.horizontal_velocity_scale = Constant(1.0)
    options.no_exports = not do_export
    options.timestep = 30.0
    options.timestep_2d = 10.0

    assert options.element_family == 'dg-dg', 'this test is not suitable for mimetic elements'
    # NOTE use symmetic uv condition to get correct w
    bnd_mom = {'symm': None}
    solver_obj.bnd_functions['momentum'] = {1: bnd_mom, 2: bnd_mom,
                                            3: bnd_mom, 4: bnd_mom}

    solver_obj.create_timestepper()
    x, y, z = SpatialCoordinate(solver_obj.mesh)
    # elevation field
    solver_obj.fields.elev_2d.project(setup_obj.elev(x_2d, y_2d, lx, ly))
    # update mesh and fields
    solver_obj.mesh_updater.update_mesh_coordinates()

    # velocity field
    uv_analytical = setup_obj.uv(x, y, z, lx, ly)
    solver_obj.fields.uv_3d.project(uv_analytical)
    # analytical solution
    w_analytical = setup_obj.w(x, y, z, lx, ly)

    if do_export:
        out_w = File(os.path.join(options.output_directory, 'w.pvd'))
        out_w_ana = File(os.path.join(options.output_directory, 'w_ana.pvd'))

    solver_obj.w_solver.solve()
    uvw = solver_obj.fields.uv_3d + solver_obj.fields.w_3d
    if do_export:
        w_ana = Function(solver_obj.function_spaces.H, name='w ana')
        w_ana.interpolate(w_analytical)
        out_w_ana.write(w_ana)
        out_w_ana.write(w_ana)
        out_w.write(solver_obj.fields.w_3d)
        out_w.write(solver_obj.fields.w_3d)

    # compute flux through bottom boundary
    normal = FacetNormal(solver_obj.mesh)
    bottom_flux = assemble(inner(uvw, normal)*ds_bottom)
    print_output('flux through bot {:}'.format(bottom_flux))

    err_msg = '{:}: Bottom impermeability violated: bottom flux {:.4g}'.format(setup_name, bottom_flux)
    assert abs(bottom_flux) < 1e-6, err_msg

    l2_err_w = errornorm(as_vector((0, 0, w_analytical)), solver_obj.fields.w_3d)/numpy.sqrt(area)
    print_output('L2 error w  {:.12f}'.format(l2_err_w))
    l2_err_uv = errornorm(uv_analytical, solver_obj.fields.uv_3d)/numpy.sqrt(area)
    print_output('L2 error uv {:.12f}'.format(l2_err_uv))

    return l2_err_w, l2_err_uv


def run_convergence(setup, ref_list, order, do_export=False, save_plot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, do_export=do_export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_w = y_log[:, 0]
    y_log_uv = y_log[:, 1]

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
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
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
            print_output('{:}: {:} convergence rate {:.4f} PASSED'.format(setup_name, field_str, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    check_convergence(x_log, y_log_w, order, 'w', save_plot)
    check_convergence(x_log, y_log_uv, order + 1, 'uv', save_plot)


def test_setup1():
    l2_err_w, l2_err_uv = run(Setup1, 3, 1, do_export=False)
    assert l2_err_w < 1e-9
    assert l2_err_uv < 1e-12


def test_setup2():
    l2_err_w, l2_err_uv = run(Setup2, 3, 1, do_export=False)
    assert l2_err_w < 1e-9
    assert l2_err_uv < 1e-12


def test_setup3():
    run_convergence(Setup3, [1, 2, 3], 1, save_plot=False)


if __name__ == '__main__':
    run(Setup3, 3, 1, do_export=True)
    run_convergence(Setup3, [1, 2, 3], 1, save_plot=True)
