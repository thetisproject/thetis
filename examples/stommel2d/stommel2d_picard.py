# Stommel gyre test case in 2D
# ============================
#
# Wind-driven geostrophic gyre in large basin.
# Setup is according to [1].
#
# [1] Comblen, R., Lambrechts, J., Remacle, J.-F., and Legat, V. (2010).
#     Practical evaluation of five partly discontinuous finite element pairs
#     for the non-conservative shallow water equations. International Journal
#     for Numerical Methods in Fluids, 63(6):701-724.
#
# Tuomas Karna 2015-04-28

# This is a version that runs much faster using much larger timesteps (2 hr instead of 45s.)
# using an implicit time integration scheme (PressureProjectionPicard)

from thetis import *

lx = 1.0e6
nx = 20
mesh2d = RectangleMesh(nx, nx, lx, lx)
outputdir = 'outputs_picard'
print_output('Loaded mesh '+mesh2d.name)
print_output('Exporting to '+outputdir)
depth = 1000.0
t_end = 75*12*2*3600
t_export = 3600*2

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Coriolis forcing
coriolis_2d = Function(P1_2d)
f0, beta = 1.0e-4, 2.0e-11
coriolis_2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0))

# Wind stress
wind_stress_2d = Function(P1v_2d, name='wind stress')
tau_max = 0.1
wind_stress_2d.interpolate(Expression(('tau_max*sin(pi*(x[1]/L - 0.5))', '0'), tau_max=tau_max, L=lx))

# linear dissipation: tau_bot/(h*rho) = -bf_gamma*u
linear_drag_coefficient = Constant(1e-6)

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.use_nonlinear_equations = False
options.coriolis_frequency = coriolis_2d
options.wind_stress = wind_stress_2d
options.linear_drag_coefficient = linear_drag_coefficient
options.t_export = t_export
options.t_end = t_end
options.timestep = 3600.*2.
options.output_directory = outputdir
options.horizontal_velocity_scale = Constant(0.01)
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
options.timestepper_type = 'PressureProjectionPicard'
options.use_linearized_semi_implicit_2d = True
options.shallow_water_theta = 1.0
options.solver_parameters_sw = {
    'snes_type': 'ksponly',
    'ksp_type': 'preonly',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',
    'fieldsplit_0_ksp_type': 'gmres',
    'fieldsplit_0_pc_type': 'sor',
    'fieldsplit_1_ksp_type': 'gmres',
    'fieldsplit_1_ksp_converged_reason': True,
    'fieldsplit_1_pc_type': 'hypre',
}
options.solver_parameters_sw_momentum = {
    'ksp_type': 'gmres',
    'ksp_converged_reason': True,
    'pc_type': 'sor',
    'pc_factor_mat_solver_package': 'mumps',
}

solver_obj.iterate()
