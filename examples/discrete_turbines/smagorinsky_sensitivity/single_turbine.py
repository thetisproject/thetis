"""Run one single-turbine wake-recovery sensitivity case.

The flow is initialized uniformly at 3 m/s.  Results are written as a
centreline transect at the turbine centreline (y = 150 m).
"""
import argparse
import os

import numpy as np
from thetis import *


parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True,
                    help='Name used for the transect CSV file')
parser.add_argument('--cs', type=float, default=None,
                    help='Enable Smagorinsky with this coefficient')
parser.add_argument('--depth', type=float, default=50.0,
                    help='Uniform water depth [m]')
parser.add_argument('--results-dir', default='results',
                    help='Result directory relative to this script')
parser.add_argument('--end-time', type=float, default=600.0,
                    help='Simulation duration [s]; enough for the wake to reach outflow')
parser.add_argument('--timestep', type=float, default=20.0)
parser.add_argument('--average-steps', type=int, default=5,
                    help='Number of final timestep solutions to average')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
example_dir = os.path.dirname(script_dir)
mesh_path = os.path.join(script_dir, 'single_turbine.msh')
results_dir = os.path.join(script_dir, args.results_dir)
os.makedirs(results_dir, exist_ok=True)

# This geometry retains Physical Surface 100 around the turbine bump.
channel_length = 1200.0
channel_width = 300.0
channel_depth = args.depth
inflow_velocity = 2.0
turbine_x = 425.0
turbine_y = 150.0

mesh2d = Mesh(mesh_path)
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry').assign(channel_depth)

farm_options = turbines.load_turbine(
    os.path.join(example_dir, 'turbine_files', 'AR2000.yaml'), mesh2d)
farm_options.upwind_correction = True
farm_options.turbine_coordinates = [[Constant(turbine_x), Constant(turbine_y)]]

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.output_directory = os.path.join(results_dir, 'outputs_' + args.label)
options.no_exports = True
options.timestep = args.timestep
options.simulation_export_time = args.end_time
options.simulation_end_time = args.end_time
options.element_family = 'dg-cg'
options.swe_timestepper_type = 'CrankNicolson'
options.swe_timestepper_options.implicitness_theta = 1.0
options.swe_timestepper_options.solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}
options.horizontal_viscosity = Constant(1.0)
options.quadratic_drag_coefficient = Constant(0.0025)
options.use_smagorinsky_viscosity = args.cs is not None
if args.cs is not None:
    options.smagorinsky_coefficient = Constant(args.cs)
    # The sensitivity deliberately compares the unbounded closure response.
    options.use_automatic_smagorinsky_viscosity_cap = False
options.discrete_tidal_turbine_farms[100] = [farm_options]

solver_obj.bnd_functions['shallow_water'] = {
    1: {'uv': Constant((inflow_velocity, 0.0))},
    2: {'elev': Constant(0.0)},
    3: {'un': Constant(0.0)},
    4: {'un': Constant(0.0)},
}

solver_obj.assign_initial_conditions(uv=Constant((inflow_velocity, 0.0)))

num_steps = int(round(args.end_time/args.timestep))
if not np.isclose(num_steps*args.timestep, args.end_time):
    raise ValueError('end-time must be an integer multiple of timestep')
if not 1 <= args.average_steps <= num_steps:
    raise ValueError('average-steps must lie between 1 and the number of timesteps')

uv_average = Function(solver_obj.fields.uv_2d.function_space(),
                      name='Mean velocity')
for step, _ in enumerate(solver_obj.create_iterator()):
    if step >= num_steps - args.average_steps:
        uv_average.dat.data[:] += solver_obj.fields.uv_2d.dat.data_ro
uv_average.dat.data[:] /= args.average_steps

# Avoid boundary evaluation ambiguity while retaining the full wake transect.
x = np.linspace(1.0e-6, channel_length - 1.0e-6, 1001)
points = np.column_stack((x, np.full_like(x, turbine_y)))
uv = np.asarray(uv_average.at(tuple(points)))
u = uv[:, 0]
v = uv[:, 1]
deficit = 1.0 - u/inflow_velocity
output = np.column_stack((x, u, v, deficit))
np.savetxt(os.path.join(results_dir, args.label + '.csv'), output,
           delimiter=',', header='x_m,u_m_per_s,v_m_per_s,velocity_deficit',
           comments='')
print_output('Wrote {}-timestep-mean centreline transect: {:}'.format(
    args.average_steps, args.label + '.csv'))
