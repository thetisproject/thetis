# Open tidal farm example
# =======================
#
# Implements open tidal farm example "Farm power production" forward model
# with a dummy friction field.
#
# http://opentidalfarm.readthedocs.io/en/latest/examples/farm-performance/farm-performance.html
#
from thetis_adjoint import *
op2.init(log_level=INFO)

outputdir = 'outputs_adjoint'

lx = 100.0
ly = 50.0
nx = 20.0
ny = 10.0
mesh2d = RectangleMesh(nx, ny, lx, ly)
print_output('Exporting to ' + outputdir)

# total duration in seconds
t_end = 20.
# estimate of max advective velocity used to estimate time step
u_mag = Constant(4.0)
# export interval in seconds
t_export = 0.5
timestep = 0.5

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

depth = 50.0
bathymetry_2d.assign(depth)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
# options.use_nonlinear_equations = False
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
solver_obj.options.timestepper_type = 'CrankNicolson'
solver_obj.options.shallow_water_theta = 1.0
solver_obj.options.solver_parameters_sw = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_package': 'mumps',
    'snes_monitor': False,
    'snes_type': 'newtonls',
}
options.use_automatic_timestep = False
options.timestep = timestep
options.horizontal_viscosity = Constant(2.0)

# create function spaces
solver_obj.create_function_spaces()

# create drag function and set something there
drag_func = Function(solver_obj.function_spaces.P1_2d, name='bottomdrag')
x = SpatialCoordinate(mesh2d)
drag_center = 12.0
drag_bg = 0.0025
x0 = lx/2
y0 = ly/2
sigma = 20.0
drag_func.project(drag_center*exp(-((x[0]-x0)**2 + (x[1]-y0)**2)/sigma**2) + drag_bg, annotate=False)
# assign fiction field
options.quadratic_drag_coefficient = drag_func

velocity_u = 2.0
# assign boundary conditions
inflow_tag = 1
outflow_tag = 2
inflow_bc = {'un': Constant(-velocity_u)}  # NOTE negative into domain
outflow_bc = {'elev': Constant(0.0)}

solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc,
                                             outflow_tag: outflow_bc}

solver_obj.assign_initial_conditions(uv=as_vector((velocity_u, 0.0)))
solver_obj.iterate()

adj_html("forward.html", "forward")
adj_html("adjoint.html", "adjoint")


integral = solver_obj.fields.solution_2d[0]*dx
J = Functional(integral*dt[FINISH_TIME], name="MyFunctional")
c = Control(drag_func)
dJdc = compute_gradient(J, c, forget=False)
out = File('gradient_J.pvd')
out.write(dJdc)
J0 = assemble(integral)
print_output("Functional evaluated by hand: ", J0)

parameters["adjoint"]["stop_annotating"] = True


def jfunc(m):
    drag_func.assign(m)
    solver_obj.simulation_time = 0.
    solver_obj.iteration = 0
    solver_obj.i_export = 0
    solver_obj.assign_initial_conditions(uv=as_vector((velocity_u, 0.0)), elev=Constant(0.0))
    solver_obj.iterate()
    Jm = assemble(integral)
    return Jm


success = replay_dolfin(tol=0.0, stop=False)
print_output(solver_obj.fields.solution_2d.vector().array()[0:10])
Jhat = ReducedFunctional(J, c)
print_output("Output of Jhat: ", Jhat(drag_func))
minconv = taylor_test(jfunc, c, J0, dJdc, seed=1e-4)
assert minconv > 1.95
