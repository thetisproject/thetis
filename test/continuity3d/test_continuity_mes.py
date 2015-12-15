"""
Testing 3D continuity equation with method of exact solution (MES).

Tuomas Karna 2015-10-23
"""
from cofs import *
import numpy
from scipy import stats
import pytest


def setup1(Lx, h0, mimetic=True):
    """
    Linear bath, zero elev, constant uv

    w is constant, tests bottom boundary condition only
    """
    out = {}
    out['bath_expr'] = Expression(
        '0.5*h0*(1.0 + x[0]/Lx)',
        Lx=Lx, h0=h0)
    out['elev_expr'] = Expression(
        '0.0',
        Lx=Lx, h0=h0)
    u_str = '1.0'
    v_str = '0.0'
    w_str = '-0.5*h0/Lx'
    out['uv_expr'] = Expression(
        (
            u_str,
            v_str,
            '0.0',
        ), Lx=Lx, h0=h0)
    out['w_expr'] = Expression(
        (
            '0.0',
            '0.0',
            w_str,
        ), Lx=Lx, h0=h0)
    out['uvw_expr'] = Expression(
        (
            u_str,
            v_str,
            w_str,
        ), Lx=Lx, h0=h0)
    out['options'] = {'mimetic': mimetic}
    return out


def setup1dg(Lx, h0):
    """Linear bath, zero elev, constant uv"""
    return setup1(Lx, h0, mimetic=False)


def setup2(Lx, h0, mimetic=True):
    """
    Constant bath and elev, uv depends on (x,y)

    Tests div(uv) terms only
    """
    out = {}
    out['bath_expr'] = Expression(
        'h0',
        h0=h0)
    out['elev_expr'] = Expression(
        '0.0',
        )
    out['uv_expr'] = Expression(
        (
        'sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        '0.2*sin(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)',
        '0.0',
        ),
        Lx=Lx)
    out['w_expr'] = Expression(
        (
        '0.0',
        '0.0',
        'h0*(-0.12*pi*cos(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)/Lx - 0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx) + x[2]*(-0.12*pi*cos(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)/Lx - 0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)',
        ),
        h0=h0, Lx=Lx)
    out['uvw_expr'] = Expression(
        (
        'sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        '0.2*sin(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)',
        'h0*(-0.12*pi*cos(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)/Lx - 0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx) + x[2]*(-0.12*pi*cos(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)/Lx - 0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)',
        ),
        h0=h0, Lx=Lx)
    out['div_uv_expr'] = Expression(
        (
        '0.0',
        '0.0',
        '-(0.12*pi*cos(0.2*pi*(1.0*x[0] + 3.0*x[1])/Lx)/Lx + 0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)',
        ),
        Lx=Lx)
    out['options'] = {'mimetic': mimetic}
    return out


def setup2dg(Lx, h0):
    """Constant bath and elev, uv depends on (x,y)"""
    return setup2(Lx, h0, mimetic=False)


def setup3(Lx, h0, mimetic=True):
    """Non-trivial bath and elev, u=1, v=0"""
    out = {}
    out['bath_expr'] = Expression(
        '0.25*h0*(cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 3.0)',
        Lx=Lx, h0=h0)
    out['elev_expr'] = Expression(
        '5.0*sin(0.25*pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)',
        Lx=Lx, h0=h0)
    u_str = '1.0'
    v_str = '0.0'
    w_str = '0.25*pi*h0*x[0]*sin(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)/(Lx*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0))'
    out['uv_expr'] = Expression(
        (
            u_str,
            v_str,
            '0.0',
        ), Lx=Lx, h0=h0)
    out['w_expr'] = Expression(
        (
            '0.0',
            '0.0',
            w_str,
        ), Lx=Lx, h0=h0)
    out['uvw_expr'] = Expression(
        (
            u_str,
            v_str,
            w_str,
        ), Lx=Lx, h0=h0)
    out['options'] = {'mimetic': mimetic}
    return out


def setup3dg(Lx, h0):
    """Non-trivial bath and elev, u=1, v=0"""
    return setup2(Lx, h0, mimetic=False)


def setup4(Lx, h0, mimetic=True):
    """Non-trivial bath and elev, uv depends on (x,y)"""
    out = {}
    out['bath_expr'] = Expression(
        '0.25*h0*(cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 3.0)',
        Lx=Lx, h0=h0)
    out['elev_expr'] = Expression(
        '5.0*sin(0.25*pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)',
        Lx=Lx, h0=h0)
    u_str = 'sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)'
    v_str = '0.2*sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)'
    w_str = '0.25*pi*h0*x[0]*sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)*sin(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)/(Lx*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)) + 0.05*pi*h0*x[1]*sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)*sin(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)/(Lx*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)) - 0.16*pi*h0*(cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 3.0)*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 0.64*pi*x[2]*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx'
    out['uv_expr'] = Expression(
        (
            u_str,
            v_str,
            '0.0',
        ), Lx=Lx, h0=h0)
    out['w_expr'] = Expression(
        (
            '0.0',
            '0.0',
            w_str,
        ), Lx=Lx, h0=h0)
    out['uvw_expr'] = Expression(
        (
            u_str,
            v_str,
            w_str,
        ), Lx=Lx, h0=h0)
    out['options'] = {'mimetic': mimetic}
    return out


def setup4dg(Lx, h0):
    """Non-trivial bath and elev, uv depends on (x,y)"""
    return setup3(Lx, h0, mimetic=False)


def setup5(Lx, h0, mimetic=True):
    """Non-trivial bath and elev, uv depends on (x,y,z)"""
    out = {}
    out['bath_expr'] = Expression(
        '0.25*h0*(cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 3.0)',
        Lx=Lx, h0=h0)
    out['elev_expr'] = Expression(
        '5.0*sin(0.25*pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)',
        Lx=Lx, h0=h0)
    u_str = 'sin(0.5*pi*(3.0*x[0] + 1.0*x[1])/Lx)*cos(2.0*pi*x[2]/h0)'
    v_str = '0.2*sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)*cos(2.0*pi*x[2]/h0)'
    w_str = '0.25*pi*h0*x[0]*sin(0.5*pi*(3.0*x[0] + 1.0*x[1])/Lx)*sin(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)*cos(pi*(0.5*cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 1.5))/(Lx*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)) + 0.05*pi*h0*x[1]*sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)*sin(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx)*cos(pi*(0.5*cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 1.5))/(Lx*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)) - 0.02*h0*sin(pi*(0.5*cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 1.5))*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 0.75*h0*sin(pi*(0.5*cos(pi*sqrt(x[0]*x[0] + x[1]*x[1] + 1.0)/Lx) + 1.5))*cos(0.5*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 0.02*h0*sin(2.0*pi*x[2]/h0)*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 0.75*h0*sin(2.0*pi*x[2]/h0)*cos(0.5*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx'
    out['uv_expr'] = Expression(
        (
            u_str,
            v_str,
            '0.0',
        ), Lx=Lx, h0=h0)
    out['w_expr'] = Expression(
        (
            '0.0',
            '0.0',
            w_str,
        ), Lx=Lx, h0=h0)
    out['uvw_expr'] = Expression(
        (
            u_str,
            v_str,
            w_str,
        ), Lx=Lx, h0=h0)
    out['options'] = {'mimetic': mimetic}
    return out


def setup5dg(Lx, h0):
    """Non-trivial bath and elev, uv depends on (x,y,z)"""
    return setup4(Lx, h0, mimetic=False)


def run(setup, refinement, order, export=True):
    """Run single test and return L2 error"""
    print '--- running {:} refinement {:}'.format(setup.__name__, refinement)
    setup_name = setup.__name__
    # domain dimensions
    Lx = 15e3
    Ly = 10e3
    area = Lx*Ly
    depth = 40.0

    S = setup(Lx, depth)

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, Lx, Ly)

    # outputs
    outputDir = createDirectory('outputs')

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.project(S['bath_expr'])

    solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
    solverObj.options.order = order
    solverObj.options.mimetic = False
    solverObj.options.uAdvection = Constant(1.0)
    solverObj.options.outputDir = outputDir
    solverObj.options.dt = 30.0
    solverObj.options.dt_2d = 10.0
    solverObj.options.update(S['options'])

    assert solverObj.options.mimetic is False, ('this test is not suitable '
                                                'for mimetic elements')

    solverObj.createEquations()
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}
    # elevation field
    solverObj.fields.elev_2d.project(S['elev_expr'])
    # update mesh and fields
    copy2dFieldTo3d(solverObj.fields.elev_2d, solverObj.fields.elev_3d)
    updateCoordinates(solverObj.mesh,
                      solverObj.fields.elev_3d,
                      solverObj.fields.bathymetry_3d,
                      solverObj.fields.z_coord_3d,
                      solverObj.fields.z_coord_ref_3d)
    computeElemHeight(solverObj.fields.z_coord_3d, solverObj.fields.v_elem_size_3d)
    copy3dFieldTo2d(solverObj.fields.v_elem_size_3d, solverObj.fields.v_elem_size_2d)
    # velocity field
    solverObj.fields.uv_3d.project(S['uv_expr'])  # NOTE for DG only
    uv_analytical = Function(solverObj.function_spaces.P1DGv, name='uv_ana_3d')
    uv_analytical.project(S['uv_expr'])
    # analytical solution
    w_analytical = Function(solverObj.function_spaces.P1DGv, name='w_ana_3d')
    w_analytical.project(S['w_expr'])
    # analytical solution in high-order space for computing L2 norms
    P1DG_ho = VectorFunctionSpace(solverObj.mesh, 'DG', order+3)
    w_ana_ho = Function(P1DG_ho, name='Analytical w')
    w_ana_ho.project(S['w_expr'])

    if export:
        out_w = File(os.path.join(outputDir, 'w.pvd'))
        out_w_ana = File(os.path.join(outputDir, 'w_ana.pvd'))
        out_uv = File(os.path.join(outputDir, 'uv.pvd'))

    # w needs to be projected to cartesian vector field for sanity check
    w_proj_3d = Function(solverObj.function_spaces.P1DGv, name='w_proj_3d')

    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    uvw = solverObj.fields.uv_3d + solverObj.fields.w_3d
    w_proj_3d.project(uvw)  # This needed for HDiv elements
    # discard u,v components
    w_proj_3d.dat.data[:, :2] = 0
    if export:
        out_w << w_proj_3d
        out_w_ana << w_analytical
        out_uv << uv_analytical
        solverObj.export()

    print 'w_pro', w_proj_3d.dat.data[:, 2].min(), w_proj_3d.dat.data[:, 2].max()
    print 'w_ana', w_analytical.dat.data[:, 2].min(), w_analytical.dat.data[:, 2].max()

    # compute flux through bottom boundary
    normal = FacetNormal(solverObj.mesh)
    bottom_flux = assemble(inner(uvw, normal)*solverObj.eq_momentum.ds_bottom)
    bottom_flux_ana = assemble(inner(w_analytical, normal)*solverObj.eq_momentum.ds_bottom)
    print 'flux through bot', bottom_flux, bottom_flux_ana
    
    err_msg = '{:}: Bottom impermeability violated: bottom flux {:.4g}'.format(setup_name, bottom_flux)
    assert abs(bottom_flux) < 1e-6, err_msg

    L2_err_w = errornorm(w_ana_ho, w_proj_3d)/numpy.sqrt(area)
    print 'L2 error w  {:.12f}'.format(L2_err_w)
    w_ana_ho.project(S['uv_expr'])
    L2_err_uv = errornorm(w_ana_ho, solverObj.fields.uv_3d)/numpy.sqrt(area)
    print 'L2 error uv {:.12f}'.format(L2_err_uv)

    linProblemCache.clear()  # NOTE must destroy all cached solvers for next simulation
    tmpFunctionCache.clear()
    return L2_err_w, L2_err_uv


def run_convergence(setup, ref_list, order, export=False, savePlot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, export=export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_w = y_log[:, 0]
    y_log_uv = y_log[:, 1]
    expected_slope = order + 1

    def check_convergence(x_log, y_log, expected_slope, field_str, savePlot):
        slope_rtol = 0.2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        setup_name = setup.__name__
        if savePlot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            N = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, N)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*N/3], yy[2*N/3], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(field_str)
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str])
            imgfile += '.png'
            imgDir = createDirectory('plots')
            imgfile = os.path.join(imgDir, imgfile)
            print 'saving figure', imgfile
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print '{:}: {:} convergence rate {:.4f} PASSED'.format(setup_name, field_str, slope)
        else:
            print '{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope)
        return slope

    check_convergence(x_log, y_log_w, order, 'w', savePlot)
    check_convergence(x_log, y_log_uv, order + 1, 'uv', savePlot)


# NOTE setup1 does not converge: solution is ~exact for all meshes
# NOTE all tests converge with rate 1 instead of 2 ...
# NOTE these tests are not valid for mimetic elements:
#      - uv and w cannot be properly represented in HDiv space
#      - should derive analytical expressions for HDiv uv and w

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.mark.not_travis
def test_setup5_dg():
    run_convergence(setup5dg, [1, 2, 3], 1, savePlot=False)

# ---------------------------
# run individual setup for debugging
# ---------------------------

#run(setup2, 2, 1)

# ---------------------------
# run individual scaling test
# ---------------------------

#run_convergence(setup5dg, [1, 2, 3], 1, savePlot=True)

# ---------------------------
# run all defined setups
# ---------------------------

#import inspect
#all_setups = [obj for name,obj in inspect.getmembers(sys.modules[__name__])
              #if inspect.isfunction(obj) and obj.__name__.startswith('setup')]
#for s in all_setups:
    #run(s, 2, 1)

