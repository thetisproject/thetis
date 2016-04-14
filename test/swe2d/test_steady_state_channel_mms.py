from thetis import *
import math
import numpy
import pytest


@pytest.mark.parametrize("options", [
    {"no_exports": True},  # default: mimetic
    {"no_exports": True, "mimetic": False, "continuous_pressure": True},
], ids=["mimetic", "pndp(n+1)"])
def test_steady_state_channel_mms(options):
    lx = 5e3
    ly = 1e3

    order = 1
    # minimum resolution
    min_cells = 16
    n = 1  # number of timesteps
    dt = 10.
    g = physical_constants['g_grav'].dat.data[0]
    h0 = 10.  # depth at rest
    area = lx*ly

    k = 4.0*math.pi/lx
    q = h0*1.0  # flux (depth-integrated velocity)
    eta0 = 1.0  # free surface amplitude

    eta_expr = Expression("eta0*cos(k*x[0])", k=k, eta0=eta0)
    depth_expr = "H0+eta0*cos(k*x[0])"
    u_expr = Expression(("Q/({H})".format(H=depth_expr), 0.), k=k, Q=q, eta0=eta0, H0=h0)
    source_expr = Expression("k*eta0*(pow(Q,2)/pow({H},3)-g)*sin(k*x[0])".format(H=depth_expr),
                             k=k, g=g, Q=q, eta0=eta0, H0=h0)
    u_bcval = q/(h0+eta0)
    eta_bcval = eta0

    do_exports = not options['no_exports']
    if do_exports:
        diff_pvd = File('diff.pvd')
        udiff_pvd = File('udiff.pvd')
        source_pvd = File('source.pvd')

    eta_errs = []
    u_errs = []
    for i in range(5):
        mesh2d = RectangleMesh(min_cells*2**i, 1, lx, ly)

        # bathymetry
        p1_2d = FunctionSpace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(p1_2d, name="bathymetry")
        bathymetry_2d.assign(h0)

        # --- create solver ---
        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d, order=order)
        solver_obj.options.nonlin = True
        solver_obj.options.t_export = dt
        solver_obj.options.t_end = n*dt
        solver_obj.options.timestepper_type = 'cranknicolson'
        solver_obj.options.timer_labels = []
        solver_obj.options.dt = dt
        solver_obj.options.update(options)

        # boundary conditions
        inflow_tag = 1
        outflow_tag = 2
        inflow_func = Function(p1_2d)
        inflow_func.interpolate(Expression(-u_bcval))
        inflow_bc = {'un': inflow_func}
        outflow_func = Function(p1_2d)
        outflow_func.interpolate(Expression(eta_bcval))
        outflow_bc = {'elev': outflow_func}
        solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
        # parameters['quadrature_degree']=5

        solver_obj.create_equations()
        solver_parameters = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_package': 'mumps',
            'snes_type': 'newtonls',
        }
        # reinitialize the timestepper so we can set our own solver parameters and gamma
        # setting gamma to 1.0 converges faster to
        solver_obj.timestepper = timeintegrator.CrankNicolson(solver_obj.eq_sw, solver_obj.dt,
                                                              solver_parameters, gamma=1.0)
        # hack to avoid picking up prefixed petsc options from other py.test tests:
        solver_obj.timestepper.name = 'test_steady_state_channel_mms'
        solver_obj.timestepper.solver_parameters.update(solver_parameters)
        solver_obj.timestepper.update_solver()
        solver_obj.assign_initial_conditions(uv_init=Expression(("1.0", "0.0")))

        source_space = FunctionSpace(mesh2d, 'DG', order+1)
        source_func = project(source_expr, source_space)
        if do_exports:
            source_pvd << source_func
        solver_obj.timestepper.F -= solver_obj.timestepper.dt_const*solver_obj.eq_sw.U_test[0]*source_func*dx
        # subtract out time derivative
        solver_obj.timestepper.F -= (solver_obj.eq_sw.mass_term(solver_obj.eq_sw.solution)-solver_obj.eq_sw.mass_term(solver_obj.timestepper.solution_old))
        solver_obj.timestepper.update_solver()

        solver_obj.iterate()

        uv, eta = solver_obj.fields.solution_2d.split()

        eta_ana = project(eta_expr, solver_obj.function_spaces.H_2d)
        if do_exports:
            diff_pvd << project(eta_ana-eta, solver_obj.function_spaces.H_2d, name="diff")
        eta_l2norm = assemble(pow(eta-eta_ana, 2)*dx)
        eta_errs.append(math.sqrt(eta_l2norm/area))

        u_ana = project(u_expr, solver_obj.function_spaces.U_2d)
        if do_exports:
            udiff_pvd << project(u_ana-uv, solver_obj.function_spaces.U_2d, name="diff")
        u_l2norm = assemble(inner(u_ana-uv, u_ana-uv)*dx)
        u_errs.append(math.sqrt(u_l2norm/area))

    # NOTE: these currently only pass for order==1
    expected_order = order + 1
    eta_errs = numpy.array(eta_errs)
    print 'eta errors:', eta_errs
    print 'convergence:', eta_errs[:-1]/eta_errs[1:], eta_errs[0]/eta_errs[-1]
    assert(all(eta_errs[:-1]/eta_errs[1:] > 2.**expected_order*0.75))
    assert(eta_errs[0]/eta_errs[-1] > (2.**expected_order)**(len(eta_errs)-1)*0.75)
    print "PASSED"

    expected_order = order + 1
    u_errs = numpy.array(u_errs)
    print 'u errors:', u_errs
    print 'convergence:', u_errs[:-1]/u_errs[1:], u_errs[0]/u_errs[-1]
    assert(all(u_errs[:-1]/u_errs[1:] > 2.**expected_order*0.75))
    assert(u_errs[0]/u_errs[-1] > (2.**expected_order)**(len(u_errs)-1)*0.75)
    print "PASSED"


if __name__ == '__main__':
    test_steady_state_channel_mms()
