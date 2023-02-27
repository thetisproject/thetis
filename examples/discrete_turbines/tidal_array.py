from thetis import *

# Set output directory, load mesh, set simulation export and end times
outputdir = 'outputs'

# To play with a headland case, create the mesh file using the commented out command below and use gmsh to check the
# PhysIDs of the inflows and outflows
# os.system('gmsh -2  headland3.geo -o headland3.msh')
mesh2d = Mesh('tidal_mesh.msh')
print_output('Loaded mesh ' + mesh2d.name)
print_output('Exporting to ' + outputdir)

t_end = 3 * 3600
t_export = 200.0


# bathymetry and viscosity sponge
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(Constant(50.0))
x = SpatialCoordinate(mesh2d)
h_viscosity = Function(P1_2d).interpolate(conditional(le(x[0], 50), 54-x[0], 1.0))


# Some realistic thrust and speed curves if we want to use a tabulated turbine approach
thrusts_AR2000 = [0., 0.75, 0.85, 0.95, 1., 3.05, 3.3, 3.55, 3.8, 4.05, 4.3, 4.55, 4.8, 5., 5.001, 5.05, 5.25, 5.5,
                  5.75, 6.0, 6.25, 6.5, 6.75, 7.0]
speeds_AR2000 = [0.010531, 0.032281, 0.038951, 0.119951, 0.516484, 0.516484, 0.387856, 0.302601, 0.242037, 0.197252,
                 0.16319, 0.136716, 0.115775, 0.102048, 0.060513, 0.005112, 0.00151, 0.00089, 0.000653, 0.000524,
                 0.000442, 0.000384, 0.000341, 0.000308]

# Initialise discrete turbine farm characteristics
turbine_density = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_density').assign(0.0)
farm_options = DiscreteTidalTurbineFarmOptions()
# farm_options.turbine_type = 'constant'
# farm_options.turbine_options.thrust_coefficient = 0.6
farm_options.turbine_type = 'table'
farm_options.turbine_options.thrust_coefficients = speeds_AR2000
farm_options.turbine_options.thrust_speeds = thrusts_AR2000
farm_options.turbine_options.C_support = 10  # support structure thrust coefficient
farm_options.turbine_options.A_support = 3.5*10.0  # cross-sectional area of support structure
farm_options.turbine_options.diameter = 20
farm_options.upwind_correction = False
farm_options.turbine_density = turbine_density
farm_options.turbine_coordinates = [[213.0, 64.0], [213.0, 128.0], [213.0, 192.0], [213.0, 256.0],
                                    [277.0, 96.0], [277.0, 160.0], [277.0, 224.0]]


# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.quadratic_drag_coefficient = Constant(0.0025)
options.swe_timestepper_type = 'CrankNicolson'
options.horizontal_viscosity = h_viscosity
options.use_wetting_and_drying = True
options.wetting_and_drying_alpha = Constant(0.5)
options.discrete_tidal_turbine_farms[1] = farm_options
if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.timestep = 50.0

options.swe_timestepper_options.solver_parameters = {
    'snes_type': 'newtonls',
    'snes_rtol': 1e-3,
    'snes_linesearch_type': 'bt',
    'snes_max_it': 20,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

# Boundary conditions - steady state case
tidal_elev = Function(P1_2d).assign(0.0)
tidal_vel = Function(P1_2d).assign(0.0)
solver_obj.bnd_functions['shallow_water'] = {1: {'un': tidal_vel},
                                             3: {'elev': tidal_elev}}

# initial conditions, piecewise linear function
elev_init = Function(P1_2d)
elev_init.assign(0.0)
solver_obj.assign_initial_conditions(elev=elev_init, uv=(as_vector((1e-3, 0.0))))
# try moving the solver_obj.assign to after the turbines callback if it fails to run (or doubly assign initial conds)

# Operation of tidal turbine farm through a callback
cb_turbines = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb_turbines, 'timestep')


def update_forcings(t_new):
    ramp = tanh(t_new / 2000.)
    tidal_vel.project(Constant(ramp * 3.5))


# See channel-optimisation example for a completely steady state simulation (no ramp)
solver_obj.iterate(update_forcings=update_forcings)
