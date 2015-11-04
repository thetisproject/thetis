from cofs import *
import math
import numpy

Lx=5e3
Ly=1e3

order = 1
# minimum resolution
if order==0:
    min_cells = 16
else:
    min_cells = 8
N = 100 # number of timesteps
dt = 1000.
g = physical_constants['g_grav'].dat.data[0]
H0 = 10. # depth at rest
area = Lx*Ly

k=4.0*math.pi/Lx
Q=H0*1.0 # flux (depth-integrated velocity)
eta0=1.0 # free surface amplitude

eta_expr = Expression("eta0*cos(k*x[0])", k=k, eta0=eta0)
depth_expr = "H0+eta0*cos(k*x[0])"
u_expr = Expression(("Q/({H})".format(H=depth_expr), 0.), k=k, Q=Q, eta0=eta0, H0=H0)
source_expr = Expression("k*eta0*(pow(Q,2)/pow({H},3)-g)*sin(k*x[0])".format(H=depth_expr),
        k=k, g=g, Q=Q, eta0=eta0, H0=H0)
u_bcval = Q/(H0+eta0)
eta_bcval = eta0

diff_pvd = File('diff.pvd')
udiff_pvd = File('udiff.pvd')

eta_errs = []
u_errs = []
for i in range(5):
    mesh2d = RectangleMesh(min_cells*2**i,1,Lx,Ly)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name="bathymetry")
    bathymetry_2d.assign(H0)


    # --- create solver ---
    solverObj = solver2d.flowSolver2d(mesh2d, bathymetry_2d, order=order)
    solverObj.options.nonlin = True
    solverObj.options.TExport = dt
    solverObj.options.T = N*dt
    solverObj.options.timeStepperType = 'CrankNicolson'
    solverObj.options.dt = dt

    # boundary conditions
    inflow_tag = 1
    outflow_tag = 2
    inflow_func=Function(P1_2d)
    inflow_func.interpolate(Expression(-u_bcval))
    inflow_bc = {'un': inflow_func}
    outflow_func=Function(P1_2d)
    outflow_func.interpolate(Expression(eta_bcval))
    outflow_bc = {'elev': outflow_func}
    solverObj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
    #parameters['quadrature_degree']=5

    solverObj.assignInitialConditions(uv_init=Expression(("1.0","0.0")))
    solver_parameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_package': 'mumps',
        'snes_monitor': False,
        'snes_type': 'newtonls'}
    # reinitialize the timestepper so we can set our own solver parameters and gamma
    # setting gamma to 1.0 converges faster to
    solverObj.timeStepper = timeIntegrator.CrankNicolson(solverObj.eq_sw, solverObj.options.dt,
                                                         solver_parameters, gamma=1.0)

    source_space=FunctionSpace(mesh2d, 'DG', order+1)
    source_func = project(source_expr, source_space)
    File('source.pvd') << source_func
    solverObj.timeStepper.F -= solverObj.timeStepper.dt_const*solverObj.eq_sw.U_test[0]*source_func*solverObj.eq_sw.dx
    solverObj.timeStepper.updateSolver()

    solverObj.iterate()

    uv, eta = solverObj.fields.solution2d.split()

    eta_ana = project(eta_expr, solverObj.function_spaces.H_2d)
    diff_pvd << project(eta_ana-eta, solverObj.function_spaces.H_2d, name="diff")
    eta_l2norm = assemble(pow(eta-eta_ana,2)*dx)
    eta_errs.append(math.sqrt(eta_l2norm/area))

    u_ana = project(u_expr, solverObj.function_spaces.U_2d)
    udiff_pvd << project(u_ana-uv, solverObj.function_spaces.U_2d, name="diff")
    u_l2norm = assemble(inner(u_ana-uv,u_ana-uv)*dx)
    u_errs.append(math.sqrt(u_l2norm/area))

# NOTE: these currently only pass for order==1
expected_order=order+1
eta_errs=numpy.array(eta_errs)
print 'eta errors:', eta_errs
print 'convergence:', eta_errs[:-1]/eta_errs[1:], eta_errs[0]/eta_errs[-1]
assert(all(eta_errs[:-1]/eta_errs[1:]>2.**expected_order*0.75))
assert(eta_errs[0]/eta_errs[-1]>(2.**expected_order)**(len(eta_errs)-1)*0.75)
print "PASSED"

expected_order=order+1
u_errs=numpy.array(u_errs)
print 'u errors:', u_errs
print 'convergence:', u_errs[:-1]/u_errs[1:], u_errs[0]/u_errs[-1]
assert(all(u_errs[:-1]/u_errs[1:]>2.**expected_order*0.75))
assert(u_errs[0]/u_errs[-1]>(2.**expected_order)**(len(u_errs)-1)*0.75)
print "PASSED"
