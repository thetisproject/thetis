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
parser.add_argument("-s", "--source-model", type=str, default="CG1")
parser.add_argument("--maxiter", type=int, default=40)
parser.add_argument("--ftol", type=float, default=1.0e-05)
parser.add_argument("--no-consistency-test", action="store_true")
parser.add_argument("--no-taylor-test", action="store_true")
parser.add_argument("--suffix", type=str, default=None)
args = parser.parse_args()
source_model = args.source_model
do_consistency_test = not args.no_consistency_test
do_taylor_test = not args.no_taylor_test
suffix = args.suffix

# Setup initial condition
pwd = os.path.abspath(os.path.dirname(__file__))
mesh2d = Mesh(f"{pwd}/japan_sea.msh")
elev_init, controls = initial_condition(mesh2d, source_model=source_model)

# Setup PDE
output_dir = f"{pwd}/outputs_elev-init-optimization_{source_model}"
if suffix is not None:
    output_dir = "_".join([output_dir, suffix])
solver_obj = construct_solver(
    elev_init,
    output_directory=output_dir,
    store_station_time_series=False,
    no_exports=os.getenv("THETIS_REGRESSION_TEST") is not None,
)
options = solver_obj.options
if not options.no_exports:
    print_output(f"Exporting to {options.output_directory}")

# Setup controls
control = "elev_init"
nc = len(controls)
if nc == 1:
    control_bounds = [-numpy.inf, numpy.inf]
else:
    control_bounds = [[-numpy.inf] * nc, [numpy.inf] * nc]

# Set up observation and regularization managers
observation_data_dir = f"{pwd}/observations"
variable = "elev"
station_names = list(stations.keys())
start_times = [dat["interval"][0] for sta, dat in stations.items()]
end_times = [dat["interval"][1] for sta, dat in stations.items()]
sta_manager = inversion_tools.StationObservationManager(
    mesh2d, output_directory=options.output_directory
)
sta_manager.load_observation_data(
    observation_data_dir,
    station_names,
    variable,
    start_times=start_times,
    end_times=end_times,
)
sta_manager.set_model_field(solver_obj.fields.elev_2d)

# Set regularization parameter
gamma_hessian_list = []
if source_model[:2] in ("CG", "DG"):
    gamma_hessian_list.append(Constant(0.1))

# Define the scaling for the cost function so that J ~ O(1)
J_scalar = Constant(solver_obj.dt / options.simulation_end_time)

# Create inversion manager and add controls
inv_manager = inversion_tools.InversionManager(
    sta_manager, real=source_model[:2] not in ("CG", "DG"),
    output_dir=options.output_directory, no_exports=options.no_exports,
    penalty_parameters=gamma_hessian_list, cost_function_scaling=J_scalar,
    test_consistency=do_consistency_test, test_gradient=do_taylor_test)
for c in controls:
    inv_manager.add_control(c)

# Extract the regularized cost function
cost_function = inv_manager.get_cost_function(solver_obj)

# Solve and setup the reduced functional
solver_obj.iterate(update_forcings=cost_function)
inv_manager.stop_annotating()

# Run inversion
opt_verbose = -1
opt_options = {
    "maxiter": args.maxiter,
    "ftol": args.ftol,
    "disp": opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv("THETIS_REGRESSION_TEST") is not None:
    opt_options["maxiter"] = 1
control_opt_list = inv_manager.minimize(
    opt_method="L-BFGS-B", bounds=control_bounds, **opt_options)
if options.no_exports:
    sys.exit(0)
if source_model[:2] in ("CG", "DG"):
    cc = control_opt_list
    if not isinstance(control_opt_list, Function):
        cc = cc[0]
    oc = inv_manager.control_coeff_list[0]
    name = cc.name()
    oc.rename(name)
else:
    oc = initial_condition(
        mesh2d, source_model=source_model, controls=inv_manager.control_coeff_list
    )[0]
outfile = File(f"{options.output_directory}/elevation_optimised.pvd")
print_function_value_range(oc, prefix="Optimal")
outfile.write(oc)
