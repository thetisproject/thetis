from thetis import *
from model_config import *
import argparse
import matplotlib.pyplot as plt
import numpy


# Parse user input
parser = argparse.ArgumentParser(
    description="Tohoku tsunami source inversion problem",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-s", "--source-model", type=str, default="okada")
parser.add_argument(
    "-o",
    "--okada-parameters",
    help="Okada parameters to invert for in the Okada model case",
    nargs="+",
    choices=["depth", "dip", "slip", "rake"],
    default=["depth", "dip", "slip", "rake"],
)
parser.add_argument("--maxiter", type=int, default=40)
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--plot-subfaults", action="store_true")
args = parser.parse_args()
source_model = args.source_model
maxiter = args.maxiter
suffix = args.suffix

# Load optimized controls
print_output("Loading optimized controls")
mesh2d = Mesh("japan_sea.msh")
output_dir = f"outputs_elev-init-optimization_{source_model}"
if suffix is not None:
    output_dir = "_".join([output_dir, suffix])
source_model = source_model.split("_")[0]
if source_model in ("box", "radial", "okada"):
    initial_guess = f"{output_dir}/m_progress.npy"
else:
    initial_guess = f"{output_dir}/hdf5/control_00_{maxiter:02d}"
source = get_source(mesh2d, source_model, initial_guess=initial_guess)
if source_model == "okada":
    source.subfault_variables = args.okada_parameters
elev = source.elev_init

# Plot in vtu format
print_output("Writing optimized elevation to vtu")
outfile = File(f"{output_dir}/elevation_optimized.pvd")
print_function_value_range(elev, prefix="Optimal")
outfile.write(elev)

# Plot in png format
print_output("Writing optimized elevation to png")
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
tc = tricontourf(elev, axes=axes, levels=51, cmap="RdBu_r")
cb = fig.colorbar(tc, ax=axes)
cb.set_label("Initial elevation (m)")
axes.axis(False)

# Annotate with subfault array
if args.plot_subfaults and source_model in ("box", "radial", "okada"):
    nx = source.num_subfaults_par + 1
    ny = source.num_subfaults_perp + 1
    nn = nx * ny  # Total number of nodes
    array_centroid = source.fault_centroid
    xy0 = numpy.kron(numpy.array(array_centroid), numpy.ones(nn)).reshape(2, nn)
    R = source.rotation_matrix(backend=numpy)
    L, W = source.fault_length, source.fault_width
    l, w = source.subfault_length, source.subfault_width
    x = numpy.linspace(-0.5 * L, 0.5 * L, nx)
    y = numpy.linspace(-0.5 * W, 0.5 * W, ny)
    xy = numpy.hstack((numpy.kron(x, numpy.ones(ny)), numpy.kron(numpy.ones(nx), y)))
    X, Y = xy0 + numpy.dot(R, xy.reshape(2, nn))
    for start in range(0, nn, ny):
        end = start + ny
        axes.plot(X[start:end], Y[start:end], "-", color="k", linewidth=0.5)
    for start in range(ny):
        axes.plot(X[start::ny], Y[start::ny], "-", color="k", linewidth=0.5)
axes.set_xlim([300e3, 1100e3])
axes.set_ylim([3700e3, 4700e3])
plt.tight_layout()
plt.savefig(f"elevation_optimized_{source_model}.png", dpi=300)
