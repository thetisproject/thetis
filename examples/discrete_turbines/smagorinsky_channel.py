"""
Straight-channel discrete-turbine array with 2D Smagorinsky mixing.

Two rows of two AR2000 turbines are separated by five rotor diameters within
a row.  The second row is shifted downstream by 2.5 rotor diameters.  The
transient run lets turbine wakes develop; ``smag_visc_2d`` in the VTK output
shows the resulting subgrid-scale mixing.
"""
import os

from firedrake.output.vtk_output import VTKFile
from thetis import *


script_dir = os.path.dirname(os.path.abspath(__file__))
outputdir = 'outputs_smagorinsky_channel_2rows'

# Channel and array geometry
diameter = 20.0
separation = 5.0*diameter
channel_length = 850.0
channel_width = 300.0
channel_depth = 50.0
inflow_velocity = 3.0
# The two-row array is centred in the channel.  The final staggered turbine
# still has 15D of wake development and a 50 m outflow buffer.
array_x0 = 350.0

mesh_path = os.path.join(script_dir, 'smagorinsky_channel.msh')
if not os.path.exists(mesh_path):
    raise IOError(
        'Generate the channel mesh first with: '
        'gmsh -2 -format msh2 -o smagorinsky_channel.msh '
        'smagorinsky_channel.geo')
mesh2d = Mesh(mesh_path)
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry').assign(channel_depth)

# The AR2000 table supplies both thrust and power coefficients.
farm_options = turbines.load_turbine(
    os.path.join(script_dir, 'turbine_files', 'AR2000.yaml'), mesh2d)
farm_options.upwind_correction = True
farm_options.turbine_coordinates = [
    [Constant(array_x0 + i*separation + row_offset), Constant(array_y)]
    for array_y, row_offset in (
        (channel_width/3.0, 0.0),
        (2.0*channel_width/3.0, 0.5*separation),
    )
    for i in range(2)
]

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.output_directory = outputdir
options.no_exports = False
options.timestep = 10.0
options.simulation_export_time = 100.0
options.simulation_end_time = 1000.0
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
options.use_smagorinsky_viscosity = True
# Smagorisky coefficient should normally be in the range 0.1-0.2 - but we use a larger value here to demonstrate the effect of subgrid-scale mixing on the turbine wakes.
options.smagorinsky_coefficient = Constant(1.0)
# Export the unbounded wake-shear response; choose the timestep accordingly.
options.use_automatic_smagorinsky_viscosity_cap = False
options.fields_to_export = ['uv_2d', 'elev_2d', 'smag_visc_2d']
# Physical Surface 100 ("array_refinement") is the turbine-operating region.
options.discrete_tidal_turbine_farms[100] = [farm_options]

# Gmsh physical boundary IDs: inflow=1, outflow=2, south=3, north=4.
solver_obj.bnd_functions['shallow_water'] = {
    1: {'uv': Constant((inflow_velocity, 0.0))},
    2: {'elev': Constant(0.0)},
    3: {'un': Constant(0.0)},
    4: {'un': Constant(0.0)},
}

# Build the farm before exporting its resolved turbine-density field.
solver_obj.create_equations()
turbine_density = Function(P1_2d, name='Turbine density')
turbine_density.interpolate(solver_obj.tidal_farms[0].turbine_density)
VTKFile(os.path.join(outputdir, 'turbine_density.pvd')).write(turbine_density)

solver_obj.assign_initial_conditions(uv=Constant((inflow_velocity, 0.0)))
solver_obj.iterate()
