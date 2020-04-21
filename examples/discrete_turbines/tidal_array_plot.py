from thetis import *
import os

outputdir = 'outputs_plot'
mesh2d = Mesh('tidal_mesh_plot.msh')
print_output('Loaded mesh ' + mesh2d.name)
print_output('Exporting to ' + outputdir)

# Simulation duration
t_end = 10 * 3600
t_export = 200.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
turbine_density = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_density').assign(0.0)

# Initialise Discrete turbine farm characteristics
farm_options = DiscreteTidalTurbineFarmOptions()
farm_options.turbine_density = turbine_density
farm_options.thrust_coefficient = Function(FunctionSpace(mesh2d, "CG", 1), name='thrust_coefficient').assign(0.6)
farm_options.power_coefficient = Function(FunctionSpace(mesh2d, "CG", 1), name='power_coefficient').assign(0.0)
farm_options.turbine_drag = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_drag_coefficient').assign(0.0)
# farm_options.upwind_correction = False

# Add viscosity sponge (depending on condition)
x = SpatialCoordinate(mesh2d)
h_viscosity = Function(P1_2d).interpolate(conditional(le(x[0], 50), 50.5-x[0], 0.5))
bathymetry_2d.assign(Constant(50.0))

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    options.simulation_end_time = 1000.
else:
    options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'smag_visc_2d']
options.quadratic_drag_coefficient = Constant(0.0025)
options.timestepper_type = 'CrankNicolson'
options.horizontal_viscosity = h_viscosity
options.use_smagorinsky_viscosity = True
options.smagorinsky_coefficient = Constant(0.2)
options.use_wetting_and_drying = True
options.wetting_and_drying_alpha = Constant(0.5)
options.discrete_tidal_turbine_farms[2] = farm_options
if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestep = 50.0

options.timestepper_options.solver_parameters = {
    'snes_type': 'newtonls',
    'snes_rtol': 1e-3,
    'snes_linesearch_type': 'bt',
    'snes_max_it': 20,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_package': 'mumps',
}

# Boundary conditions - Steady state case
tidal_elev = Function(P1_2d).assign(0.0)
tidal_vel = Function(P1_2d).assign(0.0)
solver_obj.bnd_functions['shallow_water'] = {1: {'un': tidal_vel},
                                             3: {'elev': tidal_elev}}

# initial conditions, piecewise linear function
elev_init = Function(P1_2d)
elev_init.assign(0.0)

#  Addition of turbines in the domain
turbine = ThrustTurbine(diameter=20, swept_diameter=20)
farm_options.turbine_options = turbine
turbine_coordinates = np.load("Turbine_coords.npy")
turbine_farm = DiscreteTidalfarm(solver_obj, turbine, turbine_coordinates, farm_options.turbine_density, 2)

File(outputdir+'/Farm.pvd').write(turbine_farm.farm_density)

viscosity_vtk_output = File(outputdir + '/viscosity.pvd')
smag_viscosity_vtk_output = File(outputdir + '/smag_viscosity.pvd')

solver_obj.assign_initial_conditions(elev=elev_init, uv=(as_vector((1e-3, 0.0))))

# Operation of tidal turbine farm through a callback
cb = DiscreteTurbineOperation(solver_obj, 2, farm_options, support_structure={"C_sup": 0.6, "A_sup": 10 * 3.5})
solver_obj.add_callback(cb, 'timestep')


def update_forcings(t_new):
    ramp = tanh(t_new / 2000.)
    tidal_vel.project(Constant(ramp * 3.5))

    viscosity_vtk_output.write(Function(P1_2d).interpolate(solver_obj.fields.get('viscosity_h')))
    smag_viscosity_vtk_output.write(solver_obj.fields.smag_visc_2d)


# No update_forcings for steady state case
solver_obj.iterate(update_forcings=update_forcings)
