"""
Balzano wetting-drying test case
================================

Solves shallow water equations with wetting and drying in
rectangular domain with sloping bathymetry, with periodic
free surface boundary condition applied at deep end.

Initial water elevation and velocity are zero everywhere.

Further details can be found in, e.g, [1]

Demonstrates the use of wetting and drying within Thetis

In addition to bathymetry, elevation and velocity fields,
user-specified outputs are used for the moving bathymetry
(h_tilde) and total water depth imposed on original
bathymetry (eta_tilde), the latter being useful for
comparisons with other WD models.

[1] O. Gourgue, R. Comblen, J. Lambrechts, T. Karna, V.
    Legat, and E. Deleersnijder. A flux-limiting wetting-
    drying method for finite-element shallow-water models,
    with application to the scheldt estuary. Advances in
    Water Resources, 32:1726 - 1739, 2009.
    doi: 10.1016/j.advwatres.2009.09.005.
"""
from thetis import *

outputdir = 'outputs'
mesh2d = RectangleMesh(12, 6, 13800, 7200)

# Balzano testcase 3 with isolated lake
lake_test = False

# time step in seconds
dt = 600.
# total duration in seconds: 4 periods
t_end = 2*24*3600.
# export interval in seconds
t_export = 600.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

if lake_test:
    outputdir += '_lake'

print_output('Exporting to '+outputdir)


# bathymetry: uniform slope with gradient 1/2760
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry = Function(P1_2d, name='Bathymetry')
x, y = SpatialCoordinate(mesh2d)
bath_expr = x / 2760.
if lake_test:
    bath_expr = conditional(x < 3600, bath_expr, -x / 2760. + 60./23)
    bath_expr = conditional(x < 4800, bath_expr, x / 920. - 100./23)
    bath_expr = conditional(x < 6000, bath_expr, x / 2760.)

bathymetry.interpolate(bath_expr)

# bottom friction suppresses reflection from wet-dry front
manning_drag_coefficient = Constant(0.02)
# wetting-drying options
wetting_and_drying_alpha = Constant(0.4)

# --- create solver ---
solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
options = solverObj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d']
#options.timestepper_type = 'SSPRK33'
#options.timestepper_options.use_automatic_timestep = False
#options.timestep = 10.0
#options.timestepper_type = 'CrankNicolson'
options.timestepper_type = 'DIRK22'
options.timestep = dt
options.horizontal_viscosity = Constant(1000.)
options.use_lax_friedrichs_velocity = False
options.use_wetting_and_drying = False
options.wetting_and_drying_alpha = wetting_and_drying_alpha
options.manning_drag_coefficient = manning_drag_coefficient
options.horizontal_velocity_scale = Constant(1.0)

# elevation at open boundary
bnd_time = Constant(0)
h_amp = 2.0
h_T = 12*3600.

if lake_test:
    ramp_t = 6*3600.
    f = h_amp * cos(2 * pi * bnd_time / h_T)
    ocean_elev_expr = conditional(le(bnd_time, ramp_t), f, -h_amp)
else:
    ocean_elev_expr = -h_amp * sin(2 * pi * bnd_time / h_T)

solverObj.bnd_functions['shallow_water'] = {
    2: {'elev': ocean_elev_expr}
}

# # User-defined output: moving bathymetry and eta_tilde
# wd_bathfile = File(os.path.join(outputdir, 'moving_bath.pvd'))
# moving_bath = Function(P1_2d, name="moving_bath")
# eta_tildefile = File(os.path.join(outputdir, 'eta_tilde.pvd'))
# eta_tilde = Function(P1_2d, name="eta_tilde")
#
#
# def export_func():
#     wd_bath_displacement = solverObj.eq_sw.bathymetry_displacement_mass_term.wd_bathymetry_displacement
#     eta = solverObj.fields.elev_2d
#     moving_bath.project(bathymetry + wd_bath_displacement(eta))
#     wd_bathfile.write(moving_bath)
#     eta_tilde.project(eta+wd_bath_displacement(eta))
#     eta_tildefile.write(eta_tilde)


# callback function to update boundary forcing
def update_forcings(t):
    bnd_time.assign(t)


# initial condition: assign non-zero velocity
elev_init = Constant(0.10)
if lake_test:
    elev_init = Constant(2.0)
solverObj.assign_initial_conditions(uv=Constant((1e-7, 0.)), elev=elev_init)

solverObj.iterate(update_forcings=update_forcings)
