from thetis import *
from firedrake_adjoint import *
import thetis.inversion_tools as inversion_tools
from model_config import *
import argparse
import numpy
import os
import sys


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami source inversion problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-o",
    "--okada-parameters",
    help="Okada parameters to invert for in the Okada model case",
    nargs="+",
    choices=["depth", "dip", "slip", "rake"],
    default=["depth", "dip", "slip", "rake"],
)
parser.add_argument(
    "--maxiter",
    help="Maximum number of iterations for optimisation routine",
    type=int,
    default=100,
)
parser.add_argument(
    "--ftol",
    help="Convergence criterion for optimisation routine",
    type=float,
    default=1.0e-05,
)
parser.add_argument(
    "--no-consistency-test",
    help="Test the consistency of the cost function re-evaluation",
    action="store_true",
)
parser.add_argument(
    "--no-taylor-test",
    help="Test the consistency of the computed gradients",
    action="store_true",
)
args = parser.parse_args()
active_controls = args.okada_parameters
if len(active_controls) == 0:
    print_output("Nothing to do.")
    sys.exit(0)
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test

# Setup initial condition
pwd = os.path.abspath(os.path.dirname(__file__))
mesh2d = Mesh(f"{pwd}/japan_sea.msh")
elev_init, controls = initial_condition(mesh2d, okada_parameters=active_controls)

# Setup PDE
output_dir = f"{pwd}/outputs_elev-init-optimization_okada"
solver_obj = construct_solver(elev_init, output_directory=output_dir)
options = solver_obj.options
if not options.no_exports:
    print_output(f"Exporting to {options.output_directory}")

# Set the bounds for the controls
control = "elev_init"
nc = len(controls)
na = len(active_controls)
nb = nc // na
bounds = numpy.transpose([okada_bounds[c] for c in active_controls]).flatten()
control_bounds = numpy.kron(bounds, numpy.ones(nb)).reshape((2, na * nb))
if nc == 1 and len(control_bounds[0]) == 1:
    control_bounds = [control_bounds[0][0], control_bounds[1][0]]

# Set up a StationObservationManager, which handles the observation data
observation_data_dir = f"{pwd}/observations"
variable = "elev"
stations = read_station_data()
station_names = list(stations.keys())
sta_manager = inversion_tools.StationObservationManager(mesh2d, output_directory=output_dir)
sta_manager.load_observation_data(
    observation_data_dir,
    station_names,
    variable,
    start_times=[dat["start"] for dat in stations.values()],
    end_times=[dat["end"] for dat in stations.values()],
)
sta_manager.set_model_field(solver_obj.fields.elev_2d)

# Define the scaling for the cost function so that J ~ O(1)
J_scalar = Constant(solver_obj.dt / options.simulation_end_time)

# Create an InversionManager and tell it what the controls are
inv_manager = inversion_tools.InversionManager(
    sta_manager, real=True, cost_function_scaling=J_scalar,
    output_dir=options.output_directory, no_exports=options.no_exports,
    test_consistency=do_consistency_test, test_gradient=do_taylor_test)
for c in controls:
    inv_manager.add_control(c)

# Extract the cost function from the InversionManager
cost_function = inv_manager.get_cost_function(solver_obj)

# Solve and setup the reduced functional
solver_obj.iterate(update_forcings=cost_function)

# Tell the InversionManager to stop recording operations
#   At this point it will do some testing for consistency of the
#   cost function and its gradient
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1
opt_options = {
    "maxiter": args.maxiter,
    "ftol": args.ftol,
    "disp": opt_verbose if mesh2d.comm.rank == 0 else -1,
}
control_opt_list = inv_manager.minimize(
    opt_method="L-BFGS-B", bounds=control_bounds, **opt_options)
if options.no_exports:
    sys.exit(0)
oc = initial_condition(
    mesh2d,
    controls=inv_manager.control_coeff_list,
    okada_parameters=active_controls,
)[0]
outfile = File(f"{options.output_directory}/elevation_optimised.pvd")
print_function_value_range(oc, prefix="Optimal")
outfile.write(oc)
