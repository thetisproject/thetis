from thetis import *


mesh2d = UnitSquareMesh(10, 10)
P1_2d = FunctionSpace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d).assign(1.0)
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solver_obj.options
options.solve_tracer = True
solver_obj.assign_initial_conditions()

V_2d = solver_obj.function_spaces.V_2d
sol = solver_obj.fields.solution_2d
sol_old = solver_obj.timestepper.timesteppers.swe2d.solution_old
arg = sol.copy(deepcopy=True)
arg_old = sol_old.copy(deepcopy=True)
fields = solver_obj.timestepper.timesteppers.swe2d.fields
fields_old = solver_obj.timestepper.timesteppers.swe2d.fields_old
bcs = None

error_estimator = error_estimation_2d.ShallowWaterErrorEstimator(V_2d, bathymetry2d, options)
res = error_estimator.residual('all', sol, sol_old, arg, arg_old, fields, fields_old, bcs)
print_output(res.dat.data)

Q_2d = solver_obj.function_spaces.Q_2d
sol = solver_obj.fields.tracer_2d
sol_old = solver_obj.timestepper.timesteppers.tracer.solution_old
arg = sol.copy(deepcopy=True)
arg_old = sol_old.copy(deepcopy=True)
fields = solver_obj.timestepper.timesteppers.tracer.fields
fields_old = solver_obj.timestepper.timesteppers.tracer.fields_old
bcs = None

error_estimator = error_estimation_2d.TracerErrorEstimator(Q_2d, bathymetry=bathymetry2d, use_lax_friedrichs=options.use_lax_friedrichs_tracer, sipg_parameter=options.sipg_parameter_tracer)
res = error_estimator.residual('all', sol, sol_old, arg, arg_old, fields, fields_old, bcs)
print_output(res.dat.data)
