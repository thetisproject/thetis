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

# Setup PDE
pwd = os.path.abspath(os.path.dirname(__file__))
output_dir = f"{pwd}/outputs_elev-init-optimization_{source_model}"
if suffix is not None:
    output_dir = "_".join([output_dir, suffix])
solver_obj = construct_solver(
    output_directory=output_dir,
    store_station_time_series=False,
    no_exports=os.getenv("THETIS_REGRESSION_TEST") is not None,
)
mesh2d = solver_obj.mesh2d
elev_init, controls = initial_condition(mesh2d, source_model=source_model)
options = solver_obj.options
mesh2d = solver_obj.mesh2d
bathymetry_2d = solver_obj.fields.bathymetry_2d

# Assign initial conditions
if not options.no_exports:
    print_output(f"Exporting to {options.output_directory}")
solver_obj.assign_initial_conditions(elev=elev_init)

# Choose optimisation parameters
control = "elev_init"
nc = len(controls)
if nc == 1:
    control_bounds = [-numpy.inf, numpy.inf]
else:
    control_bounds = [[-numpy.inf] * nc, [numpy.inf] * nc]
op = inversion_tools.OptimisationProgress(
    options.output_directory,
    real=source_model[:2] not in ("CG", "DG"),
    no_exports=options.no_exports,
)
for c in controls:
    op.add_control(c)

# Define the (appropriately scaled) cost function
dt_const = Constant(solver_obj.dt)
total_time_const = Constant(options.simulation_end_time)
J_scalar = dt_const / total_time_const

# Set up observation and regularization managers
observation_data_dir = f"{pwd}/observations"
variable = "elev"
station_names = list(stations.keys())
start_times = [dat["interval"][0] for sta, dat in stations.items()]
end_times = [dat["interval"][1] for sta, dat in stations.items()]
stationmanager = inversion_tools.StationObservationManager(
    mesh2d, J_scalar=J_scalar, output_directory=options.output_directory
)
stationmanager.load_observation_data(
    observation_data_dir,
    station_names,
    variable,
    start_times=start_times,
    end_times=end_times,
)
stationmanager.set_model_field(solver_obj.fields.elev_2d)

# Compute regularization term
if source_model[:2] in ("CG", "DG"):
    gamma_hessian_list = [Constant(0.1)]
    reg_manager = inversion_tools.ControlRegularizationManager(
        op.control_coeff_list,
        gamma_hessian_list,
        J_scalar=J_scalar,
    )
    op.J = reg_manager.eval_cost_function()
else:
    reg_manager = None

# Extract the regularized cost function
cost_function = inversion_tools.get_cost_function(
    solver_obj, op, stationmanager, reg_manager=reg_manager, weight_by_variance=True
)

# Solve and setup the reduced functional
solver_obj.iterate(export_func=cost_function)
Jhat = ReducedFunctional(op.J, op.control_list, **op.rf_kwargs)
pause_annotation()

if do_consistency_test:
    print_output("Running consistency test")
    J = Jhat(op.control_coeff_list)
    assert numpy.isclose(J, op.J)
    print_output("Consistency test passed!")

if do_taylor_test:
    func_list = []
    for f in op.control_coeff_list:
        dc = Function(f.function_space()).assign(f)
        func_list.append(dc)
    minconv = taylor_test(Jhat, op.control_coeff_list, func_list)
    assert minconv > 1.9
    print_output("Taylor test passed!")


def optimisation_callback(m):
    """
    Stash optimisation progress after successful line search.
    """
    op.update_progress()
    stationmanager.dump_time_series()


# Run inversion
opt_method = "L-BFGS-B"
opt_verbose = -1
opt_options = {
    "maxiter": args.maxiter,
    "ftol": args.ftol,
    "disp": opt_verbose if mesh2d.comm.rank == 0 else -1,
}
if os.getenv("THETIS_REGRESSION_TEST") is not None:
    opt_options["maxiter"] = 1
print_output(f"Running {opt_method} optimisation")
op.reset_counters()
op.start_clock()
J = float(Jhat(op.control_coeff_list))
op.set_initial_state(J, Jhat.derivative(), op.control_coeff_list)
control_opt_list = minimize(
    Jhat,
    method=opt_method,
    bounds=control_bounds,
    callback=optimisation_callback,
    options=opt_options,
)
if options.no_exports:
    sys.exit(0)
if source_model[:2] in ("CG", "DG"):
    cc = control_opt_list
    if not isinstance(control_opt_list, Function):
        cc = cc[0]
    oc = op.control_coeff_list[0]
    name = cc.name()
    oc.rename(name)
else:
    oc = initial_condition(
        mesh2d, source_model=source_model, controls=op.control_coeff_list
    )[0]
outfile = File(f"{options.output_directory}/elevation_optimised.pvd")
print_function_value_range(oc, prefix="Optimal")
outfile.write(oc)
