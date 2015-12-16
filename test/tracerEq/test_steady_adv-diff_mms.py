"""
Testing 3D tracer advection-diffusion equation with method of manufactured solution (MMS).

Tuomas Karna 2015-11-28
"""
from cofs import *
import numpy
from scipy import stats
import pytest


def setup1(Lx, Ly, h0, kappa0, mimetic=True):
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
        'sin(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx)
    out['kappa_expr'] = Expression('0.0')
    out['res_expr'] = Expression(
        '0.6*pi*cos(0.2*pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx)
    out['options'] = {'mimetic': mimetic}
    return out


def setup1dg(Lx, Ly, h0, kappa0):
    """
    Constant bathymetry and u velocty, zero diffusivity, non-trivial tracer
    """
    return setup1(Lx, Ly, h0, kappa0, mimetic=False)


def setup2(Lx, Ly, h0, kappa0, mimetic=True):
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
        'sin(3*pi*x[0]/Lx)',
        Lx=Lx)
    out['kappa_expr'] = Expression(
        'kappa0',
        kappa0=kappa0)
    out['res_expr'] = Expression(
        '9*(pi*pi)*kappa0*sin(3*pi*x[0]/Lx)/(Lx*Lx)',
        kappa0=kappa0, Lx=Lx)
    out['options'] = {'mimetic': mimetic}
    return out


def setup2dg(Lx, Ly, h0, kappa0):
    """
    constant bathymetry, zero velocity, constant kappa, x-varying T
    """
    return setup2(Lx, Ly, h0, kappa0, mimetic=False)


def setup3(Lx, Ly, h0, kappa0, mimetic=True):
    """
    constant bathymetry, zero kappa, non-trivial velocity and T
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('sin(pi*(x[1]/Ly + 2*x[0]/Lx))*sin(pi*x[2]/h0)',
         'sin(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*sin(pi*x[2]/h0)',
         '0.0', ),
        h0=h0, Lx=Lx, Ly=Ly)
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*cos(pi*x[2]/h0)/Ly + 0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))/Ly + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))*cos(pi*x[2]/h0)/Lx + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))/Lx', ),
        h0=h0, Lx=Lx, Ly=Ly)
    out['tracer_expr'] = Expression(
        '(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))',
        h0=h0, Lx=Lx, Ly=Ly)
    out['kappa_expr'] = Expression('0.0')
    out['res_expr'] = Expression(
        '-0.4*pi*(0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*cos(pi*x[2]/h0)/Ly + 0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))/Ly + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))*cos(pi*x[2]/h0)/Lx + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))/Lx)*sin(0.5*pi*x[2]/h0)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))/h0 - 0.75*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*sin(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))*sin(pi*x[2]/h0)/Ly - 1.5*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))*sin(pi*(x[1]/Ly + 2*x[0]/Lx))*sin(pi*x[2]/h0)/Lx',
        h0=h0, Lx=Lx, Ly=Ly)
    out['options'] = {'mimetic': mimetic}
    return out


def setup3dg(Lx, Ly, h0, kappa0):
    """
    constant bathymetry, zero kappa, non-trivial velocity and T
    """
    return setup3(Lx, Ly, h0, kappa0, mimetic=False)


def setup4(Lx, Ly, h0, kappa0, mimetic=True):
    """
    constant bathymetry, constant kappa, non-trivial velocity and T
    """
    out = {}
    out['bath_expr'] = Expression('h0', h0=h0)
    out['elev_expr'] = Expression('0.0')
    out['uv_expr'] = Expression(
        ('sin(pi*(x[1]/Ly + 2*x[0]/Lx))*sin(pi*x[2]/h0)',
         'sin(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*sin(pi*x[2]/h0)',
         '0.0', ),
        h0=h0, Lx=Lx, Ly=Ly)
    out['w_expr'] = Expression(
        ('0.0',
         '0.0',
         '0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*cos(pi*x[2]/h0)/Ly + 0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))/Ly + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))*cos(pi*x[2]/h0)/Lx + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))/Lx', ),
        h0=h0, Lx=Lx, Ly=Ly)
    out['tracer_expr'] = Expression(
        '(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))',
        h0=h0, Lx=Lx, Ly=Ly)
    out['kappa_expr'] = Expression(
        'kappa0',
        kappa0=kappa0)
    out['res_expr'] = Expression(
        '-0.4*pi*(0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*cos(pi*x[2]/h0)/Ly + 0.3*h0*cos(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))/Ly + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))*cos(pi*x[2]/h0)/Lx + 2*h0*cos(pi*(x[1]/Ly + 2*x[0]/Lx))/Lx)*sin(0.5*pi*x[2]/h0)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))/h0 - 0.75*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.3*x[1]/Ly + 0.3*x[0]/Lx))*sin(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))*sin(pi*x[2]/h0)/Ly - 0.5625*(pi*pi)*kappa0*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))/(Ly*Ly) - 1.5*pi*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*sin(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))*sin(pi*(x[1]/Ly + 2*x[0]/Lx))*sin(pi*x[2]/h0)/Lx + 2.25*(pi*pi)*kappa0*(0.8*cos(0.5*pi*x[2]/h0) + 0.2)*cos(pi*(0.75*x[1]/Ly + 1.5*x[0]/Lx))/(Lx*Lx)',
        h0=h0, kappa0=kappa0, Lx=Lx, Ly=Ly)
    out['options'] = {'mimetic': mimetic}
    return out


def setup4dg(Lx, Ly, h0, kappa0):
    """
    constant bathymetry, constant kappa, non-trivial velocity and T
    """
    return setup4(Lx, Ly, h0, kappa0, mimetic=False)


def run(setup, refinement, order, export=True):
    """Run single test and return L2 error"""
    print '--- running {:} refinement {:}'.format(setup.__name__, refinement)
    # domain dimensions
    Lx = 15e3
    Ly = 10e3
    area = Lx*Ly
    depth = 40.0
    kappa0 = 1.0e3  # TODO what is the max stable diffusivity?
    # set time steps
    dt = 10.0/refinement
    dt_2d = dt/2
    # simulation run time
    iterations = 2*refinement
    T = dt*iterations

    SET = setup(Lx, Ly, depth, kappa0)

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, Lx, Ly)

    # outputs
    outputDir = createDirectory('outputs')
    if export:
        out_T = File(os.path.join(outputDir, 'T.pvd'))

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.project(SET['bath_expr'])

    solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
    solverObj.options.order = order
    solverObj.options.uAdvection = Constant(1.0)
    solverObj.options.outputDir = outputDir
    solverObj.options.T = T
    solverObj.options.dt = dt
    solverObj.options.dt_2d = dt_2d
    solverObj.options.fieldsToExport = ['salt_3d', 'uv_3d', 'w_3d']
    solverObj.options.update(SET['options'])

    solverObj.createFunctionSpaces()

    # functions for source terms
    source_salt = Function(solverObj.function_spaces.H, name='salinity source')
    source_salt.project(SET['res_expr'])
    solverObj.options.salt_source_3d = source_salt

    # diffusivuty
    kappa = Function(solverObj.function_spaces.P1, name='diffusivity')
    kappa.project(SET['kappa_expr'])
    solverObj.options.hDiffusivity = kappa

    # analytical solution in high-order space for computing L2 norms
    H_HO = FunctionSpace(solverObj.mesh, 'DG', order+3)
    T_ana_ho = Function(H_HO, name='Analytical T')
    T_ana_ho.project(SET['tracer_expr'])
    # analytical solution
    T_ana = Function(solverObj.function_spaces.H, name='Analytical T')
    T_ana.project(SET['tracer_expr'])

    bnd_salt = {'value': T_ana}
    solverObj.bnd_functions['salt'] = {1: bnd_salt, 2: bnd_salt,
                                       3: bnd_salt, 4: bnd_salt}

    solverObj.createEquations()
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}
    # elevation field
    solverObj.fields.elev_2d.project(SET['elev_expr'])
    # update mesh and fields
    copy2dFieldTo3d(solverObj.fields.elev_2d, solverObj.fields.elev_3d)
    updateCoordinates(solverObj.mesh,
                      solverObj.fields.elev_3d,
                      solverObj.fields.bathymetry_3d,
                      solverObj.fields.z_coord_3d,
                      solverObj.fields.z_coord_ref_3d)
    computeElemHeight(solverObj.fields.z_coord_3d, solverObj.fields.v_elem_size_3d)
    copy3dFieldTo2d(solverObj.fields.v_elem_size_3d, solverObj.fields.v_elem_size_2d)

    # salinity field
    solverObj.fields.salt_3d.project(SET['tracer_expr'])
    # velocity field
    solverObj.fields.uv_3d.project(SET['uv_expr'])
    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    if export:
        out_T << T_ana
        solverObj.export()

    # solve salinity advection-diffusion equation with residual source term
    ti = solverObj.timeStepper
    ti.timeStepper_salt_3d.initialize(ti.fields.salt_3d)
    t = 0
    for i in range(iterations):
        for k in range(ti.nStages):
            lastStep = k == ti.nStages - 1
            ti.timeStepper_salt_3d.solveStage(k, t, ti.solver.dt, ti.fields.salt_3d)
            if ti.options.useLimiterForTracers and lastStep:
                ti.solver.tracerLimiter.apply(ti.fields.salt_3d)
        t += dt

    if export:
        out_T << T_ana
        solverObj.export()

    L2_err = errornorm(T_ana_ho, solverObj.fields.salt_3d)/numpy.sqrt(area)
    print 'L2 error {:.12f}'.format(L2_err)

    linProblemCache.clear()  # NOTE must destroy all cached solvers for next simulation
    tmpFunctionCache.clear()
    return L2_err


def run_convergence(setup, ref_list, order, export=False, savePlot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, export=export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))

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
            print '{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope)
        else:
            print '{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope)
        return slope

    check_convergence(x_log, y_log, order+1, 'tracer', savePlot)

# NOTE here mimetic option has no effect -- both use p1dg for tracers
# NOTE with diffusivity convergence rate is only 1.7, 2.0 without
# NOTE external-tracer-value BC and symmetric diffusion flux work fine
# TODO add more BCs

# ---------------------------
# standard tests for pytest
# ---------------------------


def test_setup3_dg():
    run_convergence(setup3dg, [1, 2, 3], 1, savePlot=False)


@pytest.mark.skipif(True, reason='under development')
def test_setup4_dg():
    run_convergence(setup4dg, [1, 2, 3], 1, savePlot=False)

# ---------------------------
# run individual setup for debugging
# ---------------------------

# run(setup3, 3, 1)

# ---------------------------
# run individual scaling test
# ---------------------------

# run_convergence(setup4dg, [1, 2, 3], 1, savePlot=True)
