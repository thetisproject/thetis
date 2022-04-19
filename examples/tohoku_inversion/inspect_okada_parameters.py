from thetis import *
from model_config import okada_defaults
import argparse
import matplotlib.pyplot as plt
import numpy


# Parse user input
parser = argparse.ArgumentParser(
    description="Inspect optimised Okada parameters",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-o",
    "--okada-parameters",
    help="Okada parameters to invert for in the Okada model case",
    nargs="+",
    default=["depth", "dip", "slip", "rake"],
)
parser.add_argument("--suffix", type=str, default=None)
args = parser.parse_args()
active_controls = args.okada_parameters
suffix = args.suffix
nc = len(active_controls)

# Load optimised controls
mesh2d = Mesh("japan_sea.msh")
output_dir = "outputs_elev-init-optimization_okada"
if suffix is not None:
    output_dir = "_".join([output_dir, suffix])
c = numpy.load(f"{output_dir}/m_progress.npy")[-1].reshape((nc, 130))

# Plot histogram for each
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
for i, (control, val) in enumerate(zip(active_controls, c)):
    ax = axes[i // 2, i % 2]
    unit = r" ($\mathrm{m}$)" if control in ("depth", "slip") else ""
    ax.hist(val)
    ax.set_xlabel(f"{control.capitalize()}{unit}")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    if control in ("dip", "rake"):
        ticks = ax.get_xticks()
        ax.set_xticklabels([r"$%.0f^\circ$" % t for t in ticks])
    ylim = ax.get_ylim()
    ax.axvline(okada_defaults[control], *ylim, color="k", linestyle="--")
    ax.set_ylim(ylim)
plt.tight_layout()
imgfile = "optimised_okada_parameters"
if suffix is not None:
    imgfile = "_".join([imgfile, suffix])
imgfile += ".png"
print(f"Saving to {imgfile}")
plt.savefig(imgfile, dpi=200, bbox_inches="tight")
