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

outputdir = 'outputs_balzano_cf'
mesh2d = RectangleMesh(24, 1, 13800, 7200)
print_output('Exporting to '+outputdir)

# time step in seconds
dt = 10.
# total duration in seconds: 4 periods
t_end = 2*24*3600.
# export interval in seconds
t_export = 600.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

# bathymetry: uniform slope with gradient 1/2760
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
x = SpatialCoordinate(mesh2d)
bathymetry_2d.interpolate(x[0] / 2760.0)

# --- create solver ---
solver_obj = solver2d_cf.FlowSolverCF(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
# time stepper
options.timestepper_type = 'SSPRK33'
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = False
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['elev_2d']
options.fields_to_export_hdf5 = ['elev_2d']
# bottom friction suppresses reflection from wet-dry front
options.manning_drag_coefficient = Constant(0.0)
# wetting and drying
options_nh = options.nh_model_options
options_nh.use_explicit_wetting_and_drying = True
options_nh.wetting_and_drying_threshold = 1e-2

# boundary conditions
h_amp = -2.0      # ocean boundary forcing amplitude
h_T = 12*3600.    # ocean boundary forcing period
ocean_elev_func = lambda t: h_amp * sin(2 * pi * t / h_T)
solver_obj.create_function_spaces()
H_2d = solver_obj.function_spaces.H_2d
ocean_elev = Function(H_2d, name="ocean boundary elevation").assign(ocean_elev_func(0.))
ocean_funcs = {'elev': ocean_elev}
solver_obj.bnd_functions['shallow_water'] = {
    2: ocean_funcs
}

# User-defined output
bath_minus = Function(P1_2d, name="bath_minus").assign(-bathymetry_2d)
File(os.path.join(outputdir, 'minus_bath.pvd')).write(bath_minus)
bath_dg = Function(H_2d, name="bath_dg").project(bathymetry_2d)


# user-specified export function
def export_func():
    solver_obj.fields.elev_2d.assign(solver_obj.fields.h_2d - bath_dg)


# callback function to update boundary forcing
def update_forcings(t):
    ocean_elev.assign(ocean_elev_func(solver_obj.simulation_time))
    if options_nh.use_explicit_wetting_and_drying:
        solver_obj.wd_modification.apply(
            solver_obj.fields.solution_2d,
            options_nh.wetting_and_drying_threshold
        )


# initial condition: assign non-zero velocity
solver_obj.assign_initial_conditions(uv=Constant((1e-7, 0.)))

solver_obj.iterate(update_forcings=update_forcings, export_func=export_func)
