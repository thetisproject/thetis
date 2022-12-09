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
parser.add_argument("-s", "--source-model", type=str, default="okada")
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--load", action="store_true")
args = parser.parse_args()
source_model = args.source_model
suffix = args.suffix
no_exports = os.getenv("THETIS_REGRESSION_TEST") is not None
pwd = os.path.abspath(os.path.dirname(__file__))
input_dir = f"{pwd}/outputs_elev-init-optimization_{source_model}"
output_dir = f"{pwd}/outputs_forward_{source_model}"
if suffix != "":
    input_dir = "_".join([input_dir, suffix])
    output_dir = "_".join([output_dir, suffix])

# Setup initial condition
pwd = os.path.abspath(os.path.dirname(__file__))
with CheckpointFile(f"{pwd}/japan_sea_bathymetry.h5", "r") as f:
    mesh2d = f.load_mesh("firedrake_default")
initial_guess = None
if args.load:
    print_output(f"Loading controls from {input_dir}")
    if source_model in ("box", "radial", "okada"):
        initial_guess = f"{input_dir}/m_progress.npy"
    else:
        initial_guess = f"{input_dir}/hdf5/control_00_{maxiter:02d}"
source = get_source(mesh2d, source_model, initial_guess=initial_guess)

# Solve forward
solver_obj = construct_solver(
    source.elev_init,
    output_directory=output_dir,
    store_station_time_series=not no_exports,
    no_exports=no_exports,
)
print_output(f"Exporting to {solver_obj.options.output_directory}")
tic = time_mod.perf_counter()
solver_obj.iterate()
toc = time_mod.perf_counter()
print_output(f"Total duration: {toc-tic:.2f} seconds")
