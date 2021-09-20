"""
Unit tests for computing the baroclinic head

Runs MES convergence tests against a non-trivial analytical solution in a
deformed geometry.

Here P1DGxP1DG yields convergence rate ~2.0, while P2DGxP2 yields ~2.2.
In the latter case the magnitude of the error is 10x smaller.
"""
from thetis import *
from scipy import stats
import pytest


def compute_l2_error(refinement=1, fs_type='P1DGxP1DG', no_exports=True):
    """
    Computes baroclinic head in a setting where bathymetry, mesh surface
    elevation, and density are analytical, non-trivial functions.
    """
    print_output(' ---- running refinement {:}'.format(refinement))

    # create mesh
    rho_0 = 1000.0
    physical_constants['rho0'] = rho_0

    delta_x = 120e3/refinement
    lx = 480e3
    ly = 480e3
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

    # make function spaces and fields
    p1 = get_functionspace(mesh, 'CG', 1)
    p1dg = get_functionspace(mesh, 'DG', 1)
    if fs_type == 'P2DGxP2':
        fs_bhead = get_functionspace(mesh, 'DG', 2, vfamily='CG', vdegree=2)
    elif fs_type == 'P1DGxP2':
        fs_bhead = get_functionspace(mesh, 'DG', 1, vfamily='CG', vdegree=2)
    elif fs_type == 'P1DGxP1DG':
        fs_bhead = get_functionspace(mesh, 'DG', 1, vfamily='DG', vdegree=1)
    else:
        raise Exception('Unsupported function space type {:}'.format(fs_type))

    density_3d = Function(p1dg, name='density')
    baroc_head_3d = Function(fs_bhead, name='baroclinic head')
    elev_3d = Function(p1, name='elevation')
    bathymetry_3d = Function(p1, name='elevation')
    ExpandFunctionTo3d(bathymetry_2d, bathymetry_3d).solve()

    # deform mesh by elevation
    xyz = SpatialCoordinate(mesh)
    elev_expr = 1000.*tanh(2*(xyz[0]-lx/2)/lx)*sin(1.5*xyz[1]/ly+0.3)
    elev_3d.project(elev_expr)
    z_ref = mesh.coordinates.dat.data[:, 2]
    bath = bathymetry_3d.dat.data[:]
    eta = elev_3d.dat.data[:]
    new_z = eta*(z_ref + bath)/bath + z_ref
    mesh.coordinates.dat.data[:, 2] = new_z

    if not no_exports:
        out_density = File('density.pvd')
        out_baroc_head = File('baroc_head.pvd')

    # project initial density
    beta = -1.5/depth
    density_expr = 10*cos(0.5*(xyz[0]+0.3*(xyz[1]-0.3))/lx)*sin(beta*xyz[2])
    density_3d.project(density_expr)

    # compute baroclinic head as a vertical integral
    VerticalIntegrator(density_3d,
                       baroc_head_3d,
                       bottom_to_top=False,
                       average=False).solve()
    baroc_head_3d *= -physical_constants['rho0_inv']

    if not no_exports:
        out_density.write(density_3d)
        out_baroc_head.write(baroc_head_3d)

    ana_int = 10*cos(0.5*(xyz[0]+0.3*(xyz[1]-0.3))/lx)*(cos(beta*elev_expr) - cos(beta*xyz[2]))/beta
    ana_sol_expr = -physical_constants['rho0_inv']*ana_int

    volume = comp_volume_3d(mesh)
    l2_err = errornorm(ana_sol_expr, baroc_head_3d, degree_rise=2)/numpy.sqrt(volume)
    print_output('L2 error {:}'.format(l2_err))

    if not no_exports:
        out_density.write(density_3d)
        out_baroc_head.write(baroc_head_3d.project(ana_sol_expr))

    return l2_err


def run_convergence(ref_list, save_plot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(compute_l2_error(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))

    setup_name = 'barochead'
    fs_type = options['fs_type']
    expected_rate = 2

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
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, fs_type])
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

    check_convergence(x_log, y_log, expected_rate, 'barochead', save_plot)


@pytest.mark.parametrize('fs_type', ['P1DGxP1DG', 'P1DGxP2', 'P2DGxP2'])
def test_baroclinic_head(fs_type):
    run_convergence([1, 2, 3, 4], fs_type=fs_type, save_plot=False, no_exports=True)


if __name__ == '__main__':
    run_convergence([1, 2, 3, 4], fs_type='P1DGxP1DG', save_plot=True, no_exports=False)
