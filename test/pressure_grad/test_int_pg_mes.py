"""
Unit tests for computing the internal pressure gradient

Runs MES convergence tests against a non-trivial analytical solution in a
deformed geometry.

P1DGxP2 space yields 1st order convergence. For second order convergence both
the scalar fields and its gradient must be in P2DGxP2 space.
"""
from thetis import *
from thetis.momentum_eq import InternalPressureGradientCalculator
from scipy import stats
import pytest


def compute_l2_error(refinement=1, quadratic=False, no_exports=True):
    """
    Computes pressure gradient in a setting where bathymetry, mesh surface
    elevation, and pressure are analytical, non-trivial functions.
    """
    print_output(' ---- running refinement {:}'.format(refinement))

    # create mesh
    rho_0 = 1000.0
    physical_constants['rho0'] = rho_0

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

    xy = SpatialCoordinate(mesh2d)
    depth = 3600.
    bath_expr = 0.5*(depth + depth)*(1 - 0.6*tanh(4*(xy[1]-ly/2)/ly)*sin(1.5*xy[0]/ly+0.2))
    bathymetry_2d.project(bath_expr)

    mesh = extrude_mesh_sigma(mesh2d, layers, bathymetry_2d)
    bnd_len = compute_boundary_length(mesh2d)
    mesh2d.boundary_len = bnd_len
    mesh.boundary_len = bnd_len

    # make function spaces and fields
    p1 = get_functionspace(mesh, 'CG', 1)
    if quadratic:
        # NOTE for 3rd order convergence both the scalar and grad must be p2
        fs_pg = get_functionspace(mesh, 'DG', 2, 'CG', 2, vector=True, dim=2)
        fs_scalar = get_functionspace(mesh, 'DG', 2, vfamily='CG', vdegree=2)
    else:
        # the default function spaces in Thetis
        fs_pg = get_functionspace(mesh, 'DG', 1, 'CG', 2, vector=True, dim=2)
        fs_scalar = get_functionspace(mesh, 'DG', 1, vfamily='CG', vdegree=2)

    density_3d = Function(fs_scalar, name='density')
    baroc_head_3d = Function(fs_scalar, name='baroclinic head')
    int_pg_3d = Function(fs_pg, name='pressure gradient')
    elev_3d = Function(p1, name='elevation')
    bathymetry_3d = Function(p1, name='elevation')
    ExpandFunctionTo3d(bathymetry_2d, bathymetry_3d).solve()

    # analytic expressions
    xyz = SpatialCoordinate(mesh)
    elev_expr = 2000.0*sin(0.3 + 1.5*xyz[1]/ly)*cos(2*xyz[0]/lx)
    density_expr = sin((xyz[1] - 0.3)/lx)*cos(2*xyz[2]/depth)*cos(2*xyz[0]/lx)
    baroc_head_expr = -depth*sin((2*xyz[2] - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos(2*xyz[0]/lx))/depth)*sin((xyz[1] - 0.3)/lx)*cos(2*xyz[0]/lx)/2
    baroc_head_expr_dx = (depth*sin((2*xyz[2] - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos(2*xyz[0]/lx))/depth) - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos((2*xyz[2] - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos(2*xyz[0]/lx))/depth)*cos(2*xyz[0]/lx))*sin(2*xyz[0]/lx)*sin((xyz[1] - 0.3)/lx)/lx
    baroc_head_expr_dy = (-depth*ly*sin((2*xyz[2] - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos(2*xyz[0]/lx))/depth)*cos((xyz[1] - 0.3)/lx) + 6000.0*lx*sin((xyz[1] - 0.3)/lx)*cos((2*xyz[2] - 4000.0*sin((0.3*ly + 1.5*xyz[1])/ly)*cos(2*xyz[0]/lx))/depth)*cos(2*xyz[0]/lx)*cos((0.3*ly + 1.5*xyz[1])/ly))*cos(2*xyz[0]/lx)/(2*lx*ly)

    # deform mesh by elevation
    elev_3d.project(elev_expr)
    z_ref = mesh.coordinates.dat.data[:, 2]
    bath = bathymetry_3d.dat.data[:]
    eta = elev_3d.dat.data[:]
    new_z = eta*(z_ref + bath)/bath + z_ref
    mesh.coordinates.dat.data[:, 2] = new_z

    if not no_exports:
        out_density = File('density.pvd')
        out_bhead = File('baroc_head.pvd')
        out_pg = File('int_pg.pvd')

    # project initial scalar
    density_3d.project(density_expr)
    baroc_head_3d.project(baroc_head_expr)

    # compute int_pg
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

    if not no_exports:
        out_density.write(density_3d)
        out_bhead.write(baroc_head_3d)
        out_pg.write(int_pg_3d)

    g_grav = physical_constants['g_grav']

    ana_sol_expr = g_grav*as_vector((
        baroc_head_expr_dx,
        baroc_head_expr_dy,))

    volume = comp_volume_3d(mesh)
    l2_err = errornorm(ana_sol_expr, int_pg_3d, degree_rise=2)/numpy.sqrt(volume)
    print_output('L2 error {:}'.format(l2_err))

    if not no_exports:
        out_density.write(density_3d)
        out_bhead.write(baroc_head_3d)
        out_pg.write(int_pg_3d.project(ana_sol_expr))

    return l2_err


def run_convergence(ref_list, save_plot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(compute_l2_error(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))

    setup_name = 'intpg'
    quadratic = options.get('quadratic', False)
    order = 2 if quadratic else 1

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
            ax.text(xx[int(2*n/3)], yy[int(2*n/3)], '{:4.2f}'.format(slope),
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

    check_convergence(x_log, y_log, order, 'intpg', save_plot)


@pytest.mark.parametrize(('quadratic'), [True, False], ids=['quadratic', 'linear'])
def test_int_pg(quadratic):
    run_convergence([1, 2, 3], quadratic=quadratic, save_plot=False, no_exports=True)


if __name__ == '__main__':
    # compute_l2_error(refinement=3, quadratic=False, no_exports=False)
    run_convergence([1, 2, 3, 4, 6, 8], quadratic=False, save_plot=True, no_exports=True)
