"""
MMS test for 2d shallow water equations

Tuomas Karna 2015-10-29
"""
from cofs import *
import numpy
from scipy import stats

parameters['coffee'] = {}


def setup1(Lx, depth, f0, g):
    """
    Tests the pressure gradient only

    Constant bath, zero velocity, no Coriolis
    """
    bath_expr = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    cori_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    elev_expr = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    uv_expr = Expression(
        (
            '0',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    res_elev_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    res_uv_expr = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    options = {}
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def setup2(Lx, depth, f0, g):
    """
    Tests the advection and div(Hu) terms

    Constant bath, x velocity, zero elevation, no Coriolis
    """
    bath_expr = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    cori_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    elev_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    uv_expr = Expression(
        (
            'sin(2*pi*x[0]/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    res_elev_expr = Expression(
        '2*pi*h0*cos(2*pi*x[0]/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    res_uv_expr = Expression(
        (
            '2*pi*sin(2*pi*x[0]/Lx)*cos(2*pi*x[0]/Lx)/Lx',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    options = {}
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def setup3(Lx, depth, f0, g):
    """
    Tests and div(Hu) terms for nonlin=False option

    Constant bath, x velocity, zero elevation, no Coriolis
    """
    bath_expr = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    cori_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    elev_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    uv_expr = Expression(
        (
            'sin(2*pi*x[0]/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    res_elev_expr = Expression(
        '2*pi*h0*cos(2*pi*x[0]/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    res_uv_expr = Expression(
        (
            '0',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    options = {'nonlin': False}
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def setup4(Lx, depth, f0, g):
    """
    Constant bath, no Coriolis, non-trivial elev and u
    """
    bath_expr = Expression(
        'h0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    cori_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    elev_expr = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    uv_expr = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)',
            '0',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    res_elev_expr = Expression(
        '-2.0*pi*(h0 + cos(pi*(3.0*x[0] + 1.0*x[1])/Lx))*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx - 3.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    res_uv_expr = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    options = {}
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def setup5(Lx, depth, f0, g):
    """
    No Coriolis, non-trivial bath, elev, u and v
    """
    bath_expr = Expression(
        '4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    cori_expr = Expression(
        '0',
        Lx=Lx, h0=depth, f0=f0, g=g)
    elev_expr = Expression(
        'cos(pi*(3.0*x[0] + 1.0*x[1])/Lx)',
        Lx=Lx, h0=depth, f0=f0, g=g)
    uv_expr = Expression(
        (
            'sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)',
            '0.5*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    res_elev_expr = Expression(
        '(0.3*h0*x[0]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx) + 0.5*(0.2*h0*x[1]/(Lx*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx) + 0.5*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*(cos(pi*(3.0*x[0] + 1.0*x[1])/Lx) + 4.0 + h0*sqrt(0.3*x[0]*x[0] + 0.2*x[1]*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
        Lx=Lx, h0=depth, f0=f0, g=g)
    res_uv_expr = Expression(
        (
            '-3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/Lx)/Lx',
            '-1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/Lx)/Lx + 0.25*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx - 1.5*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/Lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/Lx)/Lx',
        ), Lx=Lx, h0=depth, f0=f0, g=g)
    options = {}
    return bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, options


def run(setup, refinement, order, export=True):
    """Run single test and return L2 error"""
    print '--- running refinement', refinement
    # domain dimensions
    Lx = 15e3
    Ly = 10e3
    area = Lx*Ly
    f0 = 1e-4
    g = physical_constants['g_grav']
    depth = 40.0
    T_period = 5000.0        # period of signals
    T = 1000.0  # 500.0  # 3*T_period           # simulation duration
    TExport = T_period/100.0  # export interval
    if not export:
        TExport = 1.0e12  # high enough

    bath_expr, elev_expr, uv_expr, cori_expr, res_elev_expr, res_uv_expr, opt = setup(Lx, depth, f0, g)

    # mesh
    nx = 5*refinement
    ny = 5*refinement
    mesh2d = RectangleMesh(nx, ny, Lx, Ly)
    dt = 4.0/refinement

    # outputs
    outputDir = createDirectory('outputs')
    if export:
        out_elev = File(os.path.join(outputDir, 'elev.pvd'))
        out_elev_ana = File(os.path.join(outputDir, 'elev_ana.pvd'))

   # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.project(bath_expr)
    if bathymetry_2d.dat.data.min() < 0.0:
        print 'bath', bathymetry_2d.dat.data.min(), bathymetry_2d.dat.data.max()
        raise Exception('Negative bathymetry')

    solverObj = solver2d.flowSolver2d(mesh2d, bathymetry_2d)
    #solverObj.options.nonlin = False
    solverObj.options.order = order
    solverObj.options.mimetic = True
    solverObj.options.uAdvection = Constant(1.0)
    #solverObj.options.coriolis = Constant(f0)
    solverObj.options.outputDir = outputDir
    solverObj.options.T = T
    solverObj.options.dt = dt
    solverObj.options.TExport = TExport
    solverObj.options.timerLabels = []
    #solverObj.options.uvLaxFriedrichs = None
    #solverObj.options.timeStepperType = 'SSPIMEX'
    solverObj.options.update(opt)

    solverObj.createFunctionSpaces()

    elev_ana = Function(solverObj.function_spaces.H_2d, name='Analytical elevation')
    elev_ana.project(elev_expr)
    uv_ana = Function(solverObj.function_spaces.U_2d, name='Analytical velocity')
    uv_ana.project(uv_expr)
    flux_ana = Function(solverObj.function_spaces.U_2d, name='Analytical velocity')
    flux_ana.project(uv_ana*(bathymetry_2d + elev_ana))
    source_uv = Function(solverObj.function_spaces.U_2d, name='momentum source')
    source_uv.project(res_uv_expr)
    source_elev = Function(solverObj.function_spaces.H_2d, name='continuity source')
    source_elev.project(res_elev_expr)

    bnd = {'elev': elev_ana, 'uv': uv_ana}  # NOTE implemented and works

    #bnd = {'elev': elev_ana}  # NOTE unstable?
    #bnd = {'elev': elev_ana, 'flux': flux_ana}  # NOTE implemented and works
    #bnd = {'uv': uv_ana}  # NOTE works for pressure gradient
    #bnd = {'flux': flux_ana}  # NOTE works for pressure gradient

    solverObj.bnd_functions['shallow_water'] = {1: bnd,
                                                2: bnd,
                                                3: bnd,
                                                4: bnd,
                                                }
    solverObj.options.uv_source_2d = source_uv
    solverObj.options.elev_source_2d = source_elev

    solverObj.assignInitialConditions(elev=elev_ana, uv_init=uv_ana)
    solverObj.iterate()

    #elev_L2_err = errornorm(elev_ana, solverObj.fields.solution2d.split()[1])/numpy.sqrt(area)
    #uv_L2_err = errornorm(uv_ana, solverObj.fields.solution2d.split()[0])/numpy.sqrt(area)
    elev_diff = solverObj.fields.solution2d.split()[1] - elev_ana
    elev_L2_err = numpy.sqrt(assemble(inner(elev_diff, elev_diff)*dx)/area)
    uv_diff = solverObj.fields.solution2d.split()[0] - uv_ana
    uv_L2_err = numpy.sqrt(assemble(inner(uv_diff, uv_diff)*dx)/area)
    print 'elev L2 error {:.12f}'.format(elev_L2_err)
    print 'uv L2 error {:.12f}'.format(uv_L2_err)
    linProblemCache.clear()  # NOTE must destroy all cached solvers for next simulation
    tmpFunctionCache.clear()
    return elev_L2_err, uv_L2_err


def run_scaling(setup, ref_list, order, export=False, savePlot=False):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, export=export))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_elev = y_log[:, 0]
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
            imgfile = '_'.join(['convergence', field_str, setup_name, ref_str, order_str])
            imgfile += '.png'
            print 'saving figure', imgfile
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print '{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope)
        else:
            print '{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope)
        return slope

    check_convergence(x_log, y_log_elev, None, 'elev', savePlot)
    check_convergence(x_log, y_log_uv, None, 'uv', savePlot)

# NOTE had to change boundary conditions to get convergence
# NOTE nontrivial velocity implies slower convergence
# NOTE implement and test other boundary conditions as well
#      flux, eta + flux, eta + un
# NOTE try time dependent solution: need to update source terms

# run individual setup
#run(setup5, 1, 1)
#run(setup3, 1, 1)

run_scaling(setup5, [1, 2, 4], 1, savePlot=True)
#run_scaling(setup5, [2, 4, 8, 10], 1, savePlot=True)
#run_scaling(setup2, [1, 2, 4, 8], 1, savePlot=True)

#run_scaling(setup2, [1, 2, 4, 8], 1, savePlot=True)

## run convergence test for all the setups
#ref_list = [1, 2, 4]
#order = 1
#for t in [setup1, setup2, setup3, setup4]:
    #run_scaling(t, ref_list, order)

## NOTE to prove integrity it is sufficient to only test the most complex case
#ref_list = [1, 2, 4]
#order = 1
#run_scaling(setup2, ref_list, order)
