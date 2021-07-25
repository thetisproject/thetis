"""
Stommel gyre test case in 2D
============================

Wind-driven geostrophic gyre in large basin.
Setup is according to [1].

This is a version that runs much faster using much larger timesteps (2 hr
instead of 45 s) using an implicit time integration scheme
(PressureProjectionPicard).

[1] Comblen, R., Lambrechts, J., Remacle, J.-F., and Legat, V. (2010).
    Practical evaluation of five partly discontinuous finite element pairs
    for the non-conservative shallow water equations. International Journal
    for Numerical Methods in Fluids, 63(6):701-724.
"""
from thetis import *

lx = 1.0e6
nx = 20
mesh2d = RectangleMesh(nx, nx, lx, lx)
outputdir = 'outputs_picard'
print_output('Exporting to ' + outputdir)
depth = 1000.0
t_end = 75*12*2*3600
t_export = 3600*2

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

# Wind stress
wind_stress_2d = Function(P1v_2d, name='wind stress')
tau_max = 0.1
wind_stress_2d.interpolate(as_vector((tau_max*sin(pi*(y/lx - 0.5)), 0)))

# linear dissipation: tau_bot/(h*rho) = -bf_gamma*u
linear_drag_coefficient = Constant(1e-6)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-cg'
options.use_nonlinear_equations = False
options.coriolis_frequency = coriolis_2d
options.wind_stress = wind_stress_2d
options.linear_drag_coefficient = linear_drag_coefficient
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.timestep = 3600.*2.
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(0.01)
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
options.swe_timestepper_type = 'PressureProjectionPicard'
options.swe_timestepper_options.implicitness_theta = 1.0

solver_obj.iterate()
