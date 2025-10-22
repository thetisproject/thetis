"""
Basic example of a discrete turbine array placed in the high energy region
of flow past a headland. Turbines are based on the SIMEC Atlantis AR1500 and
AR2000 models with cut-in, rated and cut-out speeds.
Flow becomes steady after an initial ramp up.
Two callbacks are used - one for the overall farm and one for discrete turbines.
"""

from thetis import *
from firedrake.output.vtk_output import VTKFile
from turbine_callback import TidalPowerCallback
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Set output directory, load mesh, set simulation export and end times
outputdir = 'outputs'
if not os.path.exists('headland.msh'):
    os.system('gmsh -2 headland.geo -o headland.msh')
mesh2d = Mesh('headland.msh')
site_ID = 2  # mesh PhysID for subdomain where turbines are to be sited
print_output(f'Loaded mesh {mesh2d.name}')
print_output(f'Exporting to {outputdir}')

t_end = 1.5 * 3600
t_export = 200.0

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# Bathymetry and viscosity sponges
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(Constant(50.0))
x = SpatialCoordinate(mesh2d)
h_viscosity = Function(P1_2d).interpolate(conditional(le(x[0], 50), 51-x[0], 1.0))
VTKFile(outputdir + '/viscosity/viscosity.pvd').write(h_viscosity)


# Turbine options
turbine_thrust_def = 'table'  # 'table' or 'constant'
include_support_structure = True  # add additional drag due to the support structure?

# Define the thrust curve of the turbine using a tabulated approach:
# speeds_AR2000: speeds for corresponding thrust coefficients - thrusts_AR2000
# powers_AR2000: list of idealised power coefficients of an AR2000 tidal turbine
# thrusts_AR2000: list of idealised thrust coefficients of an AR2000 tidal turbine using a curve fitting technique with:
#   * cut-in speed = 1 m/s
#   * rated speed = 3.05 m/s
#   * cut-out speed = 5 m/s
# (ramp up and down to cut-in and at cut-out speeds for model stability)
speeds_AR2000 = [0., 0.75, 0.85, 0.95, 1., 3.05, 3.3, 3.55, 3.8, 4.05, 4.3, 4.55, 4.8, 5., 5.001, 5.05, 5.25, 5.5, 5.75,
                 6.0, 6.25, 6.5, 6.75, 7.0]
powers_AR2000 = [0.0105, 0.032, 0.0385, 0.116, 0.437, 0.437, 0.345, 0.277, 0.226, 0.187, 0.156, 0.132, 0.112, 0.0993,
                 0.0595, 0.0051, 0.00151, 0.000889, 0.000652, 0.000523, 0.000441, 0.000384, 0.000341, 0.000308]
thrusts_AR2000 = [0.010531, 0.032281, 0.038951, 0.119951, 0.516484, 0.516484, 0.387856, 0.302601, 0.242037, 0.197252,
                  0.16319, 0.136716, 0.115775, 0.102048, 0.060513, 0.005112, 0.00151, 0.00089, 0.000653, 0.000524,
                  0.000442, 0.000384, 0.000341, 0.000308]

# Set the water density to match the thrust and power curves (defaults to 1000kg/m3)
physical_constants['rho0'] = 1026.0

# initialise discrete turbine farm characteristics
farm_options_AR2000 = DiscreteTidalTurbineFarmOptions()
farm_options_AR2000.turbine_type = turbine_thrust_def
if turbine_thrust_def == 'table':
    farm_options_AR2000.turbine_options.thrust_speeds = speeds_AR2000
    farm_options_AR2000.turbine_options.thrust_coefficients = thrusts_AR2000
    farm_options_AR2000.turbine_options.power_coefficients = powers_AR2000
else:
    farm_options_AR2000.turbine_options.thrust_coefficient = 0.6
    farm_options_AR2000.turbine_options.power_coefficient = 0.55
if include_support_structure:
    farm_options_AR2000.turbine_options.C_support = 0.7  # support structure thrust coefficient
    farm_options_AR2000.turbine_options.A_support = 2.6 * 14.0  # cross-sectional area of support structure
farm_options_AR2000.turbine_options.diameter = 20
farm_options_AR2000.upwind_correction = True  # See https://arxiv.org/abs/1506.03611 for more details
turbine_density = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_density_AR2000')
farm_options_AR2000.turbine_density = turbine_density
farm_options_AR2000.turbine_coordinates = [[Constant(x), Constant(y)]
                                           for x in numpy.arange(1000, 1061, 60)
                                           for y in numpy.arange(260, 341, 40)]
# We can also use estimate the rotor-averaged velocity using a power law, albeit with some significant caveats when
# turbines are in one another's wake. Note that this is still a depth-averaged example and this is just an estimation!
farm_options_AR2000.turbine_options.apply_shear_profile = True
farm_options_AR2000.turbine_options.structure_type = "bottom-fixed"  # or floating
farm_options_AR2000.turbine_options.rel_hub_height = 14.0  # from seabed, or from water surface if floating
farm_options_AR2000.turbine_options.shear_alpha = 7.0   # 1/7th power law
farm_options_AR2000.turbine_options.shear_beta = 0.4

# we can tidy up data loading by using YAML or JSON files
farm_options_AR1500 = turbine_loader.load_turbine(os.path.join(script_dir, "turbine_files", "AR1500.yaml"), mesh2d)
farm_options_AR1500.turbine_coordinates = [[Constant(940), Constant(y)] for y in numpy.arange(260, 341, 40)]
farm_options_AR1500.turbine_options.apply_shear_profile = True
farm_options_AR1500.turbine_options.shear_alpha = 7.0   # 1/7th power law
farm_options_AR1500.turbine_options.shear_beta = 0.4

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
options.swe_timestepper_options.implicitness_theta = 0.5
options.horizontal_viscosity = h_viscosity
options.use_wetting_and_drying = True
options.wetting_and_drying_alpha = Constant(0.5)
if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
    options.timestep = 50.0
options.discrete_tidal_turbine_farms[site_ID] = [farm_options_AR1500, farm_options_AR2000]

# Use direct solver instead of default iterative settings
# (see SemiImplicitSWETimeStepperOptions2d in thetis/options.py)
# to make it more robust for larger timesteps and low viscosity
options.swe_timestepper_options.solver_parameters = {'ksp_type': 'preonly',
                                                     'pc_type': 'lu',
                                                     'pc_factor_mat_solver_type': 'mumps'}

# Boundary conditions - steady state case
left_tag = 1
right_tag = 2
tidal_elev = Function(P1_2d).assign(0.0)
tidal_vel = Function(P1_2d).assign(0.0)
solver_obj.bnd_functions['shallow_water'] = {right_tag: {'un': tidal_vel},
                                             left_tag: {'elev': tidal_elev}}

# Initial conditions, piecewise linear function
elev_init = Function(P1_2d)
elev_init.assign(0.0)
solver_obj.assign_initial_conditions(elev=elev_init, uv=(as_vector((1e-3, 0.0))))

print_output(f'{options.swe_timestepper_type} solver options:')
print_output(options.swe_timestepper_options.solver_parameters)

# Operation of tidal turbine farm through a callback (density assumed = 1000kg/m3)
# 1. In-built farm callback
cb_farm = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb_farm, 'timestep')
power_farm = []  # create empty list to append instantaneous powers to
# export the turbine density
turbine_density_function = Function(P1_2d, name="Turbine Density")
turbine_density_function.project(solver_obj.tidal_farms[0].turbine_density)
VTKFile(outputdir + '/turbine_density_AR1500.pvd').write(turbine_density_function)
turbine_density_function.project(solver_obj.tidal_farms[1].turbine_density)
VTKFile(outputdir + '/turbine_density_AR2000.pvd').write(turbine_density_function)

# 2. Tracking of individual turbines through the callback defined in turbine_callback.py
#    This can be used even if the turbines are not included in the simulation.
#    This method becomes slow when the domain becomes very large (e.g. 100k+ elements).
num_AR1500s = len(farm_options_AR1500.turbine_coordinates)
turbine_names = [f'AR1500_{i+1}' for i in range(num_AR1500s)]
cb_turbines_AR1500 = TidalPowerCallback(solver_obj, site_ID, farm_options_AR1500, ['pow'], name='AR1500array',
                                        turbine_names=turbine_names)
solver_obj.add_callback(cb_turbines_AR1500, 'timestep')
turbine_names = [f'AR2000_{i+1}' for i in range(len(farm_options_AR2000.turbine_coordinates))]
cb_turbines_AR2000 = TidalPowerCallback(solver_obj, site_ID, farm_options_AR2000, ['pow'], name='AR2000array',
                                        turbine_names=turbine_names)
solver_obj.add_callback(cb_turbines_AR2000, 'timestep')
powers_turbines = []

print_output("\nMonitoring the following turbines:")
for i, name in enumerate(cb_turbines_AR1500.variable_names):
    print_output(f'{name}: ' f'({cb_turbines_AR1500.get_turbine_coordinates[i][0]}, '
                 f'{cb_turbines_AR1500.get_turbine_coordinates[i][1]})')
for i, name in enumerate(cb_turbines_AR2000.variable_names):
    print_output(f'{name}: ' f'({cb_turbines_AR2000.get_turbine_coordinates[i][0]}, '
                 f'{cb_turbines_AR2000.get_turbine_coordinates[i][1]})')
print_output("")


def update_forcings(t_new):
    ramp = tanh(t_new / 2000.)
    tidal_vel.project(Constant(ramp * 3.))
    power_farm.append(cb_farm.instantaneous_power.copy())
    powers_turbines.append(numpy.concat((cb_turbines_AR1500(), cb_turbines_AR2000())))


# See channel-optimisation example for a completely steady state simulation (no ramp)
solver_obj.iterate(update_forcings=update_forcings)

power_farm.append(cb_farm.instantaneous_power)  # add final powers, should be the same as callback hdf5 file!
power_farm = np.array(power_farm).T
AR1500farm_energy = np.sum(power_farm[0]) * options.timestep / 3600
AR2000farm_energy = np.sum(power_farm[1]) * options.timestep / 3600
farm_energy = AR1500farm_energy + AR2000farm_energy
powers_turbines.append(np.concat((cb_turbines_AR1500(), cb_turbines_AR2000())))
turbine_energies = np.sum(np.array(powers_turbines), axis=0) * options.timestep / 3600
callback_diff = 100 * (farm_energy - np.sum(turbine_energies, axis=0)) / farm_energy

print_output("\nTurbine energies:")
for i, name in enumerate(cb_turbines_AR1500.variable_names):
    print_output(f'{name}: {turbine_energies[i][0]*10**(-3):.2f}kWh')
for i, name in enumerate(cb_turbines_AR2000.variable_names):
    print_output(f'{name}: {turbine_energies[num_AR1500s+i][0]*10**(-3):.2f}kWh')

print_output("Check callbacks sum to the same energy")
print_output(f"Farm callback total energy recorded: {farm_energy*10**(-3):.2f}kWh")
print_output(f"Turbines callback total energy recorded: {np.sum(turbine_energies*10**(-3), axis=0)[0]:.2f}kWh")
print_output(f"Difference: {callback_diff[0]:.5f}%")
print_output("")
