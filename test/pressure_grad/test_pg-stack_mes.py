"""
Unit tests for computing density, baroclinic head and the internal pressure
gradient from a temperature field.

Runs MES convergence tests against a non-trivial analytical solution in a
deformed geometry.

NOTE currently only linear equation of state is tested
TODO test full nonlinear equation of state
"""
from thetis import *
from thetis.momentum_eq import InternalPressureGradientCalculator
from scipy import stats
import pytest


def compute_l2_error(refinement=1, quadratic_pressure=False, quadratic_density=False,
                     project_density=False, full_eos=False, no_exports=True):
    """
    Computes pressure gradient in a setting where bathymetry, mesh surface
    elevation, and pressure are analytical, non-trivial functions.
    """
    print_output(' ---- running refinement {:}'.format(refinement))

    # create mesh
    rho_0 = 1000.0
    physical_constants['rho0'] = rho_0
    g_grav = physical_constants['g_grav']

    delta_x = 120e3/refinement
    lx = 360e3
    ly = 360e3
    nx = int(lx/delta_x)
    ny = int(ly/delta_x)

    mesh2d = RectangleMesh(nx, ny, lx, ly)
    layers = 3*refinement

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')

    elev_warp_fact = 0.3
    bath_warp_fact = 0.6
    xy = SpatialCoordinate(mesh2d)
    depth = 3600.
    bath_expr = 0.5*(depth + depth)*(1 - bath_warp_fact*tanh(4*(xy[1]-ly/2)/ly)*sin(1.5*xy[0]/ly+0.2))
    bathymetry_2d.project(bath_expr)

    mesh = extrude_mesh_sigma(mesh2d, layers, bathymetry_2d)
    bnd_len = compute_boundary_length(mesh2d)
    mesh2d.boundary_len = bnd_len
    mesh.boundary_len = bnd_len

    # make function spaces and fields
    p1 = get_functionspace(mesh, 'CG', 1)
    p1dg = get_functionspace(mesh, 'DG', 1)

    if quadratic_density:
        fs_density = get_functionspace(mesh, 'DG', 2, vfamily='CG', vdegree=2)
    else:
        fs_density = p1dg

    if quadratic_pressure:
        # NOTE for 3rd order convergence both the scalar and grad must be p2
        fs_bhead = get_functionspace(mesh, 'DG', 2, vfamily='CG', vdegree=2)
        fs_pg = get_functionspace(mesh, 'DG', 2, 'CG', 2, vector=True, dim=2)
    else:
        # the default function spaces in Thetis
        fs_bhead = get_functionspace(mesh, 'DG', 1, vfamily='CG', vdegree=2)
        fs_pg = get_functionspace(mesh, 'DG', 1, 'CG', 2, vector=True, dim=2)

    temp_3d = Function(p1dg, name='temperature')
    density_3d = Function(fs_density, name='density')
    baroc_head_3d = Function(fs_bhead, name='baroclinic head')
    int_pg_3d = Function(fs_pg, name='pressure gradient')
    elev_3d = Function(p1, name='elevation')
    bathymetry_3d = Function(p1, name='elevation')
    ExpandFunctionTo3d(bathymetry_2d, bathymetry_3d).solve()

    # deform mesh by elevation
    xyz = SpatialCoordinate(mesh)
    elev_expr = elev_warp_fact*depth*cos(3*(xyz[0]/lx-0.3))*sin(2*xyz[1]/ly+0.3)
    elev_3d.project(elev_expr)
    z_ref = mesh.coordinates.dat.data[:, 2]
    bath = bathymetry_3d.dat.data[:]
    eta = elev_3d.dat.data[:]
    new_z = eta*(z_ref + bath)/bath + z_ref
    mesh.coordinates.dat.data[:, 2] = new_z

    # project initial temperature, range ~ [0, 10] deg C
    temp_expr = 5*cos((2*xyz[0] + xyz[1])/lx)*cos((xyz[2]/depth)) + 15
    temp_3d.project(temp_expr)

    # compute density
    alpha = 0.2  # thermal expansion coeff
    beta = 0.0  # haline contraction coeff
    temp_ref = 15.0
    salt_const = 10.0
    salt = Constant(salt_const)

    if not full_eos:
        eos_params = {
            'rho_ref': rho_0,
            's_ref': salt_const,
            'th_ref': temp_ref,
            'alpha': alpha,
            'beta': beta,
        }
        equation_of_state = LinearEquationOfState(**eos_params)
    else:
        equation_of_state = JackettEquationOfState()

    if project_density:
        density_solver = DensitySolverWeak(salt, temp_3d, density_3d,
                                           equation_of_state)
    else:
        density_solver = DensitySolver(salt, temp_3d, density_3d,
                                       equation_of_state)
    density_solver.solve()

    # solve baroclinic head
    VerticalIntegrator(density_3d,
                       baroc_head_3d,
                       bottom_to_top=False,
                       average=False).solve()
    baroc_head_3d *= -physical_constants['rho0_inv']

    # solve pressure gradient
    fields = FieldDict()
    fields.baroc_head_3d = baroc_head_3d
    fields.int_pg_3d = int_pg_3d
    bnd_functions = {}
    int_pg_solver = InternalPressureGradientCalculator(
        fields,
        bathymetry_3d,
        bnd_functions,
        solver_parameters=None)
    int_pg_solver.solve()

    # analytical solution
    if not full_eos:
        density_expr = - alpha*(temp_expr - temp_ref)
    else:
        # just use the nonlin expression
        density_expr = equation_of_state.eval(salt, temp_expr, p=0.0, rho0=rho_0)
    assert not full_eos
    # need to integrate f(z) = -alpha*temp_expr(z)
    # f(z) = -alpha*5*cos((2*xyz[0] + xyz[1])/lx)*cos((z/depth))
    # F(z) = -alpha*5*cos((2*xyz[0] + xyz[1])/lx)*depth*sin((z/depth))
    # int(f, eta, z) = F(eta) - F(z)
    a = -physical_constants['rho0_inv']*alpha*5
    b = cos((2*xyz[0] + xyz[1])/lx)
    c_xy = depth*sin((elev_expr/depth))
    c_z = -depth*sin((xyz[2]/depth))
    baroc_head_expr = a*b*(c_xy + c_z)

    # compute Dx Dy of the above
    b_dx = -sin((2*xyz[0] + xyz[1])/lx)*2/lx
    b_dy = -sin((2*xyz[0] + xyz[1])/lx)/lx
    elev_expr_dx = -elev_warp_fact*depth*3/lx*sin(3*(xyz[0]/lx-0.3))*sin(2*xyz[1]/ly+0.3)
    elev_expr_dy = elev_warp_fact*depth*2/ly*cos(3*(xyz[0]/lx-0.3))*cos(2*xyz[1]/ly+0.3)
    c_xy_dx = elev_expr_dx*cos((elev_expr/depth))
    c_xy_dy = elev_expr_dy*cos((elev_expr/depth))
    bhead_dx_expr = a*b_dx*(c_xy + c_z) + a*b*c_xy_dx
    bhead_dy_expr = a*b_dy*(c_xy + c_z) + a*b*c_xy_dy

    int_pg_expr = g_grav*as_vector((bhead_dx_expr, bhead_dy_expr))

    # error norms
    volume = comp_volume_3d(mesh)
    l2_err_density = errornorm(density_expr, density_3d, degree_rise=2)/numpy.sqrt(volume)
    print_output('Density L2 error: {:}'.format(l2_err_density))
    l2_err_bhead = errornorm(baroc_head_expr, baroc_head_3d, degree_rise=2)/numpy.sqrt(volume)
    print_output('B.head  L2 error: {:}'.format(l2_err_bhead))
    l2_err_pg = errornorm(int_pg_expr, int_pg_3d, degree_rise=2)/numpy.sqrt(volume)
    print_output('Int.PG  L2 error: {:}'.format(l2_err_pg))

    if not no_exports:
        out_temp = File('temperature.pvd')
        out_density = File('density.pvd')
        out_bhead = File('baroc_head.pvd')
        out_pg = File('int_pg.pvd')

        # export numerical solution
        out_temp.write(temp_3d)
        out_density.write(density_3d)
        out_bhead.write(baroc_head_3d)
        out_pg.write(int_pg_3d)

        # export projected analytical solution
        out_temp.write(temp_3d)
        out_density.write(density_3d.project(density_expr))
        out_bhead.write(baroc_head_3d.project(baroc_head_expr))
        out_pg.write(int_pg_3d.project(int_pg_expr))

    return l2_err_density, l2_err_density, l2_err_pg


def run_convergence(ref_list, save_plot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(compute_l2_error(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_density = y_log[:, 0]
    y_log_bhead = y_log[:, 1]
    y_log_intpg = y_log[:, 2]

    setup_name = 'intpg-stack'
    order = 1

    def check_convergence(x_log, y_log, expected_slope, field_str, save_plot):
        slope_rtol = 0.2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if save_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            n = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, n)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*n/3], yy[2*n/3], '{:4.2f}'.format(slope),
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

    check_convergence(x_log, y_log_density, 2, 'density', save_plot)
    check_convergence(x_log, y_log_bhead, 2, 'bhead', save_plot)
    check_convergence(x_log, y_log_intpg, 1, 'intpg', save_plot)


@pytest.mark.parametrize(('quad_p'), [True, False], ids=['p2_pressure', 'p1_pressure'])
def test_int_pg(quad_p):
    run_convergence([1, 2, 3], quadratic_pressure=quad_p, quadratic_density=False,
                    full_eos=False, project_density=False, no_exports=True,
                    save_plot=False)


if __name__ == '__main__':
    run_convergence([1, 2, 3, 4], quadratic_pressure=True, quadratic_density=False,
                    full_eos=False, project_density=False, no_exports=True,
                    save_plot=True)
