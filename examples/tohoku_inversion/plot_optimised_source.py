from thetis import *
from model_config import *
import argparse
import numpy


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami source inversion problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-s", "--source-model", type=str, default="CG1")
parser.add_argument("--maxiter", type=int, default=40)
args = parser.parse_args()
source_model = args.source_model
maxiter = args.maxiter

# Load optimised controls
mesh2d = Mesh("japan_sea.msh")
output_directory = f"outputs_elev-init-optimization_{source_model}"
source_model = source_model.split("_")[0]
if source_model[:2] in ("CG", "DG"):
    fname = f"{output_directory}/hdf5/control_00_{maxiter:02d}.h5"
    with CheckpointFile(fname, "r") as chk:
        m = chk.load_mesh("firedrake_default")
        elev = chk.load_function(m, "elev_2d")
else:
    c = numpy.load(f"{output_directory}/m_progress.npy")[-1]
    elev = initial_condition(mesh2d, source_model=source_model, initial_guess=c)[0]

# Write to vtu
outfile = File(f"{output_directory}/elevation_optimised.pvd")
print_function_value_range(elev, prefix="Optimal")
outfile.write(elev)
