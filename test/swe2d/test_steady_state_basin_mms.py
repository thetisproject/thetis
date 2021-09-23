"""
MMS test for 2d shallow water equations.

- setup functions define analytical expressions for fields and boundary
  conditions. Expressions were derived with sympy.
- run function runs the MMS setup with a single mesh resolution, returning
  L2 errors.
- run_convergence runs a scaling test, computes and asserts convergence rate.
"""
from thetis import *
from scipy import stats
import pytest


def setup7(x, lx, ly, h0, f0, nu0, g):
    """
    Non-trivial Coriolis, bath, elev, u and v, tangential velocity is zero at bnd to test flux BCs
    """
    out = {}
    out['bath_expr'] = h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + 4.0
    out['cori_expr'] = f0*cos(pi*(x[0] + x[1])/lx)
    out['elev_expr'] = cos(pi*(3.0*x[0] + 1.0*x[1])/lx)
    out['uv_expr'] = as_vector(
        [
            sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*sin(pi*x[1]/ly),
            0.5*sin(pi*x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx),
        ])
    out['res_elev_expr'] = (0.3*h0*x[0]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*sin(pi*x[1]/ly) + 0.5*(0.2*h0*x[1]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx) + 0.5*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*sin(pi*x[0]/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx - 2.0*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*sin(pi*x[1]/ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx
    out['res_uv_expr'] = as_vector(
        [
            -0.5*f0*sin(pi*x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(x[0] + x[1])/lx) - 3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx + 0.5*(pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*x[1]/ly)/ly + 1.0*pi*sin(pi*x[1]/ly)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx) - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*sin(pi*x[1]/ly)**2*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx,
            f0*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*sin(pi*x[1]/ly)*cos(pi*(x[0] + x[1])/lx) - 1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx + (-1.5*pi*sin(pi*x[0]/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*x[0]/lx)/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*sin(pi*x[1]/ly) + 0.25*pi*sin(pi*x[0]/lx)**2*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx,
        ])
    out['bnd_funcs'] = {1: {'elev': None, 'flux_left': None},
                        2: {'flux_right': None},
                        3: {'elev': None, 'flux_lower': None},
                        4: {'un_upper': None},
                        }
    return out


def setup8(x, lx, ly, h0, f0, nu0, g):
    """
    Non-trivial Coriolis, bath, elev, u and v, tangential velocity is non-zero at bnd, must prescribe uv at boundary.
    """
    out = {}
    out['bath_expr'] = h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + 4.0
    out['cori_expr'] = f0*cos(pi*(x[0] + x[1])/lx)
    out['elev_expr'] = cos(pi*(3.0*x[0] + 1.0*x[1])/lx)
    out['uv_expr'] = as_vector(
        [
            sin(pi*(-2.0*x[0] + 1.0*x[1])/lx),
            0.5*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx),
        ])
    out['res_elev_expr'] = (0.3*h0*x[0]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx) + 0.5*(0.2*h0*x[1]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx) + 0.5*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx - 2.0*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx
    out['res_uv_expr'] = as_vector(
        [
            -0.5*f0*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(x[0] + x[1])/lx) - 3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx,
            f0*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*(x[0] + x[1])/lx) - 1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx + 0.25*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx - 1.5*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx,
        ])

    # NOTE uv condition alone does not work
    out['bnd_funcs'] = {1: {'elev': None, 'uv': None},
                        2: {'elev': None, 'uv': None},
                        3: {'elev': None, 'uv': None},
                        4: {'elev': None, 'uv': None},
                        }
    return out


def setup9(x, lx, ly, h0, f0, nu0, g):
    """
    No Coriolis, non-trivial bath, viscosity, elev, u and v.
    """
    out = {}
    out['bath_expr'] = h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + 4.0
    out['visc_expr'] = nu0*(1.0 + x[0]/lx)
    out['elev_expr'] = cos(pi*(3.0*x[0] + 1.0*x[1])/lx)
    out['uv_expr'] = as_vector(
        [
            sin(pi*(-2.0*x[0] + 1.0*x[1])/lx),
            0.5*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx),
        ])
    out['res_elev_expr'] = (0.3*h0*x[0]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx) + 0.5*(0.2*h0*x[1]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) - 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx) + 0.5*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx - 2.0*pi*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx
    out['res_uv_expr'] = as_vector(
        [
            -3.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx - 4.0*pi*nu0*(1.0 + x[0]/lx)*(-0.3*h0*x[0]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) + 3.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/(lx*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)) + 0.5*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx - 2.0*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx - 1.5*pi**2*nu0*(1.0 + x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx**2 + 9.0*pi**2*nu0*(1.0 + x[0]/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx**2 + 4.0*pi*nu0*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx**2,
            -1.0*pi*g*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx + 1.0*pi*nu0*(1.0 + x[0]/lx)*(-0.2*h0*x[1]/(lx*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)) + 1.0*pi*sin(pi*(3.0*x[0] + 1.0*x[1])/lx)/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/(lx*(h0*sqrt(0.3*x[0]**2 + 0.2*x[1]**2 + 0.1)/lx + cos(pi*(3.0*x[0] + 1.0*x[1])/lx) + 4.0)) + 0.25*pi*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx - 1.5*pi*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx + 5.5*pi**2*nu0*(1.0 + x[0]/lx)*sin(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx**2 - 2.0*pi**2*nu0*(1.0 + x[0]/lx)*sin(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx**2 + 1.5*pi*nu0*cos(pi*(-3.0*x[0] + 1.0*x[1])/lx)/lx**2 - 1.0*pi*nu0*cos(pi*(-2.0*x[0] + 1.0*x[1])/lx)/lx**2,
        ])

    out['bnd_funcs'] = {1: {'uv': None},
                        2: {'uv': None},
                        3: {'uv': None},
                        4: {'uv': None},
                        }
    out['options'] = {
        'use_grad_div_viscosity_term': True,
        'use_grad_depth_viscosity_term': True,
    }

    return out


def run(setup, refinement, order, do_export=True, options=None,
        solver_parameters=None):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))
    # domain dimensions
    lx = 15e3
    ly = 10e3
    area = lx*ly
    f0 = 5e-3  # NOTE large value to make Coriolis terms larger
    nu0 = 100.0
    g = physical_constants['g_grav']
    depth = 10.0
    t_period = 5000.0        # period of signals
    t_end = 1000.0  # 500.0  # 3*T_period
    t_export = t_period/100.0  # export interval

    # mesh
    nx = 5*refinement
    ny = 5*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    dt = 4.0/refinement
    if options is not None and options.get('swe_timestepper_type') == 'CrankNicolson':
        dt *= 100.

    x = SpatialCoordinate(mesh2d)
    sdict = setup(x, lx, ly, depth, f0, nu0, g)

    # outputs
    outputdir = 'outputs'

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(sdict['bath_expr'])
    if bathymetry_2d.dat.data.min() < 0.0:
        print_output('bath {:} {:}'.format(bathymetry_2d.dat.data.min(), bathymetry_2d.dat.data.max()))
        raise Exception('Negative bathymetry')

    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.polynomial_degree = order
    solver_obj.options.element_family = 'rt-dg'
    solver_obj.options.horizontal_velocity_scale = Constant(1.0)
    solver_obj.options.no_exports = not do_export
    solver_obj.options.output_directory = outputdir
    solver_obj.options.simulation_end_time = t_end
    solver_obj.options.timestep = dt
    solver_obj.options.simulation_export_time = t_export
    if 'options' in sdict:
        solver_obj.options.update(sdict['options'])
    if options is not None:
        solver_obj.options.update(options)
    if hasattr(solver_obj.options.swe_timestepper_options, 'use_automatic_timestep'):
        solver_obj.options.swe_timestepper_options.use_automatic_timestep = False

    solver_obj.create_function_spaces()

    # analytical solution in high-order space for computing L2 norms
    h_2d_ho = FunctionSpace(solver_obj.mesh2d, 'DG', order+3)
    u_2d_ho = VectorFunctionSpace(solver_obj.mesh2d, 'DG', order+4)
    elev_ana_ho = Function(h_2d_ho, name='Analytical elevation')
    elev_ana_ho.project(sdict['elev_expr'])
    uv_ana_ho = Function(u_2d_ho, name='Analytical velocity')
    uv_ana_ho.project(sdict['uv_expr'])

    # functions for source terms
    source_uv = Function(solver_obj.function_spaces.U_2d, name='momentum source')
    source_uv.project(sdict['res_uv_expr'])
    source_elev = Function(solver_obj.function_spaces.H_2d, name='continuity source')
    source_elev.project(sdict['res_elev_expr'])
    solver_obj.options.momentum_source_2d = source_uv
    solver_obj.options.volume_source_2d = source_elev
    if 'cori_expr' in sdict:
        coriolis_func = Function(solver_obj.function_spaces.H_2d, name='coriolis')
        coriolis_func.project(sdict['cori_expr'])
        solver_obj.options.coriolis_frequency = coriolis_func
    if 'visc_expr' in sdict:
        viscosity_space = FunctionSpace(solver_obj.mesh2d, "CG", order)
        viscosity_func = Function(viscosity_space, name='viscosity')
        viscosity_func.project(sdict['visc_expr'])
        solver_obj.options.horizontal_viscosity = viscosity_func

    # functions for boundary conditions
    # analytical elevation
    elev_ana = Function(solver_obj.function_spaces.H_2d, name='Analytical elevation')
    elev_ana.project(sdict['elev_expr'])
    # analytical uv
    uv_ana = Function(solver_obj.function_spaces.U_2d, name='Analytical velocity')
    uv_ana.project(sdict['uv_expr'])
    # normal velocity (scalar field, will be interpreted as un*normal vector)
    # left/right bnds
    un_ana_x = Function(solver_obj.function_spaces.H_2d, name='Analytical normal velocity x')
    un_ana_x.project(uv_ana[0])
    # lower/uppser bnds
    un_ana_y = Function(solver_obj.function_spaces.H_2d, name='Analytical normal velocity y')
    un_ana_y.project(uv_ana[1])
    # flux through left/right bnds
    flux_ana_x = Function(solver_obj.function_spaces.H_2d, name='Analytical x flux')
    flux_ana_x.project(uv_ana[0]*(bathymetry_2d + elev_ana)*ly)
    # flux through lower/upper bnds
    flux_ana_y = Function(solver_obj.function_spaces.H_2d, name='Analytical x flux')
    flux_ana_y.project(uv_ana[1]*(bathymetry_2d + elev_ana)*lx)

    # construct bnd conditions from setup
    bnd_funcs = sdict['bnd_funcs']
    # correct values to replace None in bnd_funcs
    # NOTE: scalar velocity (flux|un) sign def: positive out of domain
    bnd_field_mapping = {'symm': None,
                         'elev': elev_ana,
                         'uv': uv_ana,
                         'un_left': -un_ana_x,
                         'un_right': un_ana_x,
                         'un_lower': -un_ana_y,
                         'un_upper': un_ana_y,
                         'flux_left': -flux_ana_x,
                         'flux_right': flux_ana_x,
                         'flux_lower': -flux_ana_y,
                         'flux_upper': flux_ana_y,
                         }
    for bnd_id in bnd_funcs:
        d = {}  # values for this bnd e.g. {'elev': elev_ana, 'uv': uv_ana}
        for bnd_field in bnd_funcs[bnd_id]:
            field_name = bnd_field.split('_')[0]
            d[field_name] = bnd_field_mapping[bnd_field]
        # set to the correct bnd_id
        solver_obj.bnd_functions['shallow_water'][bnd_id] = d
        # # print a fancy description
        # bnd_str = ''
        # for k in d:
        #     if isinstance(d[k], ufl.algebra.Product):
        #         a, b = d[k].operands()
        #         name = '{:} * {:}'.format(a.value(), b.name())
        #     else:
        #         if d[k] is not None:
        #             name = d[k].name()
        #         else:
        #             name = str(d[k])
        #     bnd_str += '{:}: {:}, '.format(k, name)
        # print_output('bnd {:}: {:}'.format(bnd_id, bnd_str))

    solver_obj.assign_initial_conditions(elev=elev_ana, uv=uv_ana)
    if solver_parameters is not None:
        # HACK: need to change prefix of solver options in order to overwrite them
        solver_obj.timestepper.name += '_'
        solver_obj.timestepper.solver_parameters.update(solver_parameters)
        solver_obj.timestepper.update_solver()

    solver_obj.iterate()

    elev_l2_err = errornorm(elev_ana_ho, solver_obj.fields.solution_2d.split()[1])/numpy.sqrt(area)
    uv_l2_err = errornorm(uv_ana_ho, solver_obj.fields.solution_2d.split()[0])/numpy.sqrt(area)
    print_output('elev L2 error {:.12f}'.format(elev_l2_err))
    print_output('uv L2 error {:.12f}'.format(uv_l2_err))
    return elev_l2_err, uv_l2_err


def run_convergence(setup, ref_list, order, do_export=False, save_plot=False,
                    options=None, solver_parameters=None):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, do_export=do_export,
                          options=options, solver_parameters=solver_parameters))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    y_log_elev = y_log[:, 0]
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

    check_convergence(x_log, y_log_elev, order+1, 'elev', save_plot)
    check_convergence(x_log, y_log_uv, order+1, 'uv', save_plot)

# NOTE nontrivial velocity implies slower convergence
# NOTE try time dependent solution: need to update source terms
# NOTE using Lax-Friedrichs stabilization in mom. advection term improves convergence of velocity

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.fixture(params=[setup7, setup8, setup9],
                ids=["Setup7", "Setup8", "Setup9"])
def setup(request):
    return request.param


@pytest.fixture(params=['rt-dg', 'dg-dg', 'dg-cg', 'bdm-dg'])
def element_family(request):
    return request.param


@pytest.fixture(params=['CrankNicolson'])
def timestepper_type(request):
    return request.param


def test_steady_state_basin_convergence(setup, element_family, timestepper_type):
    sp = {'ksp_type': 'preonly', 'pc_type': 'lu', 'snes_monitor': None,
          'mat_type': 'aij'}
    options = {
        'element_family': element_family,
        'swe_timestepper_type': timestepper_type
    }
    run_convergence(setup, [1, 2, 4, 6], 1, options=options,
                    solver_parameters=sp, save_plot=False)
