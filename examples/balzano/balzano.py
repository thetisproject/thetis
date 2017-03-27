#  Balzano wetting-drying test case
# ==============================================
#
# Details from e.g. Gourgue et al (2009)
#
# Solves shallow water equations with wetting and drying in
# rectangular domain with sloping bathymetry, with periodic
# free surface boundary condition applied at deep end.
#
# Initial water elevation and velocity are zero everywhere.
#
# Demonstrates the use of wetting and drying within Thetis
#
#
# Simon Warder 2017-03-21

from thetis import *

outputdir = 'outputs'
mesh2d = RectangleMesh(12, 6, 13800, 7200)
print_output('Loaded mesh '+mesh2d.name)
print_output('Exporting to '+outputdir)

# time step in seconds
dt = 600.
# total duration in seconds: 4 periods
t_end = 2*24*3600.
# export interval in seconds
t_export = 600.

# bathymetry: uniform slope with gradient 1/2760
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry = Function(P1_2d, name='Bathymetry')
x = SpatialCoordinate(mesh2d)
bathymetry.interpolate(x[0] / 2760.0)

# bottom friction suppresses reflection from wet-dry front
mu_manning = 0.02
# wetting-drying options
wetting_and_drying = True
wd_alpha = 0.4

# --- create solver ---
solverObj = solver2d.FlowSolver2d(mesh2d, bathymetry)
options = solverObj.options
options.t_export = t_export
options.t_end = t_end
options.outputdir = outputdir
options.check_vol_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.timestepper_type = 'cranknicolson'
options.shallow_water_theta = 0.5
options.wetting_and_drying = wetting_and_drying
options.wd_alpha = wd_alpha
options.mu_manning = mu_manning
options.dt = dt
options.solver_parameters_sw = {
    'snes_type': 'newtonls',
    'snes_monitor': True,
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
}

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


# callback function to update boundary forcing
def update_forcings(t):
    print_output("Updating boundary condition at t={}".format(t))
    ocean_elev.assign(ocean_elev_func(t))


# initial condition: assign non-zero velocity
solverObj.assign_initial_conditions(uv=Constant((1e-7, 0.)))

solverObj.iterate(update_forcings=update_forcings)
