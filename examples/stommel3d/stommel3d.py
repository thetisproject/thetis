"""
Stommel gyre test case in 3D
============================

Wind-driven geostrophic gyre in large basin.
Setup is according to [1]. This version us for 3D equations. As the problem
is purely baroclinic the solution is the same as in 2D.

[1] Comblen, R., Lambrechts, J., Remacle, J.-F., and Legat, V. (2010).
    Practical evaluation of five partly discontinuous finite element pairs
    for the non-conservative shallow water equations. International Journal
    for Numerical Methods in Fluids, 63(6):701-724.
"""
from thetis import *

lx = 1.0e6
nx = 20
mesh2d = RectangleMesh(nx, nx, lx, lx)
outputdir = 'outputs'
print_output('Exporting to '+outputdir)
depth = 1000.0
layers = 6
t_end = 75*12*2*3600.
t_export = 3600.*2

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry
P1_2d = get_functionspace(mesh2d, 'CG', 1)
P1v_2d = get_functionspace(mesh2d, 'CG', 1, vector=True)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
coriolis_2d = Function(P1_2d)
f0, beta = 1.0e-4, 2.0e-11
y_0 = 0.0
x, y = SpatialCoordinate(mesh2d)
coriolis_2d.interpolate(f0 + beta*(y-y_0))

# linear dissipation: tau_bot/(h*rho) = -bf_gamma*u
linear_drag_coefficient = Constant(1e-6)

# --- create solver ---
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
options = solver_obj.options
options.use_nonlinear_equations = False
options.solve_salinity = False
options.solve_temperature = False
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = False
options.use_baroclinic_formulation = False
options.coriolis_frequency = coriolis_2d
options.linear_drag_coefficient = linear_drag_coefficient
options.vertical_viscosity = Constant(1e-2)
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.timestepper_type = 'SSPRK22'
options.timestepper_options.swe_options.solver_parameters['snes_type'] = 'ksponly'
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(0.5)
options.check_volume_conservation_2d = True
options.check_volume_conservation_3d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'uv_dav_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d']

solver_obj.create_function_spaces()
tau_max = 0.1
x, y, z = SpatialCoordinate(solver_obj.mesh)
wind_stress_3d = Function(solver_obj.function_spaces.P1v, name='wind stress')
wind_stress_3d.interpolate(as_vector((tau_max*sin(pi*(y/lx - 0.5)), 0, 0)))
options.wind_stress = wind_stress_3d

solver_obj.iterate()
