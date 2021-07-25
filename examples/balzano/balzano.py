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
print_output('Exporting to '+outputdir)

# time step in seconds
dt = 600.
# total duration in seconds: 4 periods
t_end = 2*24*3600.
# export interval in seconds
t_export = 600.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry: uniform slope with gradient 1/2760
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry = Function(P1_2d, name='Bathymetry')
x = SpatialCoordinate(mesh2d)
bathymetry.interpolate(x[0] / 2760.0)

# bottom friction suppresses reflection from wet-dry front
manning_drag_coefficient = Constant(0.02)
# wetting-drying options
use_wetting_and_drying = True
wetting_and_drying_alpha = Constant(0.4)

# --- create solver ---
solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
options = solverObj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.swe_timestepper_type = 'CrankNicolson'
options.swe_timestepper_options.implicitness_theta = 0.5
options.use_wetting_and_drying = use_wetting_and_drying
options.wetting_and_drying_alpha = wetting_and_drying_alpha
options.manning_drag_coefficient = manning_drag_coefficient
options.timestep = dt

# boundary conditions
h_amp = -2.0      # ocean boundary forcing amplitude
h_T = 12*3600.    # ocean boundary forcing period
ocean_elev_func = lambda t: h_amp * sin(2 * pi * t / h_T)
solverObj.create_function_spaces()
H_2d = solverObj.function_spaces.H_2d
ocean_elev = Function(H_2d, name="ocean boundary elevation").assign(ocean_elev_func(0.))
ocean_funcs = {'elev': ocean_elev}
solverObj.bnd_functions['shallow_water'] = {
    2: ocean_funcs
}

# User-defined output: moving bathymetry and eta_tilde
wd_bathfile = File(os.path.join(outputdir, 'moving_bath.pvd'))
moving_bath = Function(P1_2d, name="moving_bath")
eta_tildefile = File(os.path.join(outputdir, 'eta_tilde.pvd'))
eta_tilde = Function(P1_2d, name="eta_tilde")


# user-specified export function
def export_func():
    wd_bath_displacement = solverObj.depth.wd_bathymetry_displacement
    eta = solverObj.fields.elev_2d
    moving_bath.project(bathymetry + wd_bath_displacement(eta))
    wd_bathfile.write(moving_bath)
    eta_tilde.project(eta+wd_bath_displacement(eta))
    eta_tildefile.write(eta_tilde)


# callback function to update boundary forcing
def update_forcings(t):
    print_output("Updating boundary condition at t={}".format(t))
    ocean_elev.assign(ocean_elev_func(t))


# initial condition: assign non-zero velocity
solverObj.assign_initial_conditions(uv=Constant((1e-7, 0.)))

solverObj.iterate(update_forcings=update_forcings, export_func=export_func)
