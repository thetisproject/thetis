# simple set up with a single "resolved" turbine
from thetis import *
import math
op2.init(log_level=INFO)

mesh2d = Mesh('channel.msh')

# if we solve with PressureProjectionPicard (theta=1.0) it seems to converge (power output to 7 digits) in roughly 
# 800 timesteps of 20s
# with SteadyState we only do 1 timestep (t_end should be slightly smaller than timestep to achieve this)
timestep = 20
t_end = 0.9*timestep

H = 40  # water depth

# turbine parameters:
D = 18  # turbine diameter
C_T = 0.8  # thrust coefficient

# correction to account for the fact that the thrust coefficient is based on an upstream velocity
# whereas we are using a depth averaged at-the-turbine velocity (see Kramer and Piggott 2016, eq. (15))
A_T = math.pi*(D/2)**2
correction = 4/(1+math.sqrt(1-A_T/(H*D)))**2
# NOTE, that we're not yet correcting power output here, so that will be overestimated

# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
options.timestep = timestep
options.simulation_export_time = timestep
options.simulation_end_time = t_end
options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
options.element_family = 'dg-dg'
options.timestepper_type = 'PressureProjectionPicard'
options.timestepper_type = 'SteadyState'
options.timestepper_options.solver_parameters['pc_factor_mat_solver_type'] = 'mumps'
options.timestepper_options.solver_parameters['snes_monitor'] = True
#options.timestepper_options.implicitness_theta = 1.0
options.horizontal_viscosity = Constant(1.)
options.quadratic_drag_coefficient = Constant(0.0025)

# assign boundary conditions
left_tag = 1
right_tag = 2
# noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
freeslip_bc = {'un': Constant(0.0)}
solver_obj.bnd_functions['shallow_water'] = {
    left_tag: {'uv': Constant((3., 0.))},
    right_tag: {'un': Constant(3.), 'elev': Constant(0.)}
}

# we've meshed the turbine as a DxD square, so we can treat it
# as a turbine "farm" with turbine density of 1 turbine per D^2 area
turbine_density = Constant(1.0/D**2, domain=mesh2d)
farm_options = TidalTurbineFarmOptions()
farm_options.turbine_density = turbine_density
farm_options.turbine_options.diameter = D
farm_options.turbine_options.thrust_coefficient = C_T*correction
# assign ID 2 with the "farm"
options.tidal_turbine_farms[2] = farm_options


cb = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb, 'timestep')


# run as normal (this run will be annotated by firedrake_adjoint)
solver_obj.assign_initial_conditions(uv=as_vector((3.0, 0.0)))
solver_obj.iterate()
