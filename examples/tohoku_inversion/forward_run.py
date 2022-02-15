from thetis import *
import time as time_mod
from model_config import *
import argparse
import os


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami propagation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--source_model", type=str, default="CG1")
args = parser.parse_args()
source_model = args.source_model
no_exports = os.getenv("THETIS_REGRESSION_TEST") is not None

# Solve forward
pwd = os.path.abspath(os.path.dirname(__file__))
solver_obj = construct_solver(
    output_directory=f"{pwd}/outputs_forward_{source_model}",
    store_station_time_series=not no_exports,
    no_exports=no_exports,
)
mesh2d = solver_obj.mesh2d
elev_init = initial_condition(mesh2d, source_model=source_model)
elev = Function(elev_init.function_space())
elev.project(mask(mesh2d) * elev_init)
print_output(f"Exporting to {solver_obj.options.output_directory}")
solver_obj.assign_initial_conditions(elev=elev)
tic = time_mod.perf_counter()
solver_obj.iterate()
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
