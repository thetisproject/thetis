# 2D shallow water equations in a closed channel with (currently) a single turbine
# ================================================================================
from thetis import *

output_dir = create_directory('outputs_tmp')
mesh2d = Mesh('domain.msh')

d = 0.04
H = 4*d

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Constant(H)

C_s = Constant(0.2)
h_viscosity = Constant(0.0)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solver_obj.options
options.timestep = 0.01
options.simulation_export_time = 0.01
options.simulation_end_time = 15
options.output_directory = output_dir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'smag_visc_3d']
options.timestepper_type = 'CrankNicolson'
options.horizontal_viscosity = h_viscosity
options.timestepper_options.implicitness_theta = 0.5
options.timestepper_options.solver_parameters = {'snes_monitor': None,
                                                 'snes_rtol': 1e-9,
                                                 'ksp_type': 'preonly',
                                                 'pc_type': 'lu',
                                                 'pc_factor_mat_solver_type': 'mumps',
                                                 'mat_type': 'aij'
                                                 }

options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = C_s

# boundary conditions
un_ramp = Function(P1_2d)
un_ramp.assign(0.0)
solver_obj.bnd_functions['shallow_water'] = {
    1: {'un': un_ramp},
    2: {'elev': 0.0, 'un': -un_ramp},
    3: {'un': 0.0},
    4: {'uv': Constant((0.0, 0.0))}
}


def update_forcings(t):
    un_ramp.assign(tanh(t)*-0.535)


solver_obj.assign_initial_conditions()

solver_obj.iterate(update_forcings=update_forcings)
