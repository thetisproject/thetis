from thetis import *
from model_config import *
import argparse
import matplotlib.pyplot as plt
import numpy


# Parse user input
parser = argparse.ArgumentParser(
    description="Plot optimised source for the Tohoku tsunami",
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
parser.add_argument("--maxiter", type=int, default=40)
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--plot-subfaults", action="store_true")
args = parser.parse_args()
active_controls = args.okada_parameters
if len(active_controls) == 0:
    print_output("Nothing to do.")
    sys.exit(0)

# Load optimised controls
print_output("Loading optimised controls")
mesh2d = Mesh("japan_sea.msh")
output_dir = "outputs_elev-init-optimization_okada"
c = numpy.load(f"{output_dir}/m_progress.npy")[-1]
elev = initial_condition(
    mesh2d,
    initial_guess=c,
    okada_parameters=active_controls,
)[0]

# Plot in vtu format
print_output("Writing to vtu")
outfile = File(f"{output_dir}/elevation_optimised.pvd")
print_function_value_range(elev, prefix="Optimal")
outfile.write(elev)

# Plot in png format
print_output("Writing to png")
fig, axes = plt.subplots(figsize=(6.4, 4.8))
triplot(mesh2d, axes=axes, boundary_kw={"linewidth": 0.5, "edgecolor": "k"})
tc = tricontourf(elev, axes=axes, levels=51, cmap="RdBu_r")
cb = fig.colorbar(tc, ax=axes)
cb.set_label("Initial elevation (m)")
axes.axis(False)

# Annotate with subfault array
if args.plot_subfaults:
    nx += 1  # Number of nodes parallel to fault
    ny += 1  # Number of nodes perpendicular to fault
    nn = nx * ny  # total number of nodes
    x0, y0 = array_centre
    xy0 = numpy.kron(numpy.array([x0, y0]), numpy.ones(nn)).reshape(2, nn)
    theta = strike_angle
    R = numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])
    l = Dx / nx
    w = Dy / ny
    x = numpy.linspace(-0.5 * (Dx + l), 0.5 * (Dx + l), nx)
    y = numpy.linspace(-0.5 * (Dy + w), 0.5 * (Dy + w), ny)
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
plt.savefig("elevation_optimised_okada.png", dpi=300)
