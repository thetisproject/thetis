import glob
import matplotlib.pyplot as plt
import numpy


colors = {
    "CG1": "C0",
    "DG0": "C1",
    "DG1": "C2",
    "box": "C3",
    "radial": "C4",
    "okada": "C5",
}
markers = ["x", "+", "^", "v", "o", "h"]
lines = ["-", "--", ":"] * 2
counts = {approach: 0 for approach in colors}

fig, axes = plt.subplots()
inv_dir = "outputs_elev-init-optimization"
fpaths = glob.glob(f"{inv_dir}_*")
if len(fpaths) == 0:
    raise ValueError("Nothing to plot!")
for fpath in fpaths:
    source_model = fpath.split(inv_dir + "_")[-1]
    color = "k"
    marker = ""
    line = "-"
    kw = {"label": source_model}
    for approach, c in colors.items():
        if source_model.startswith(approach):
            kw["color"] = c
            kw["marker"] = markers[counts[approach]]
            kw["linestyle"] = lines[counts[approach]]
            counts[approach] += 1
            break
    J_progress = numpy.load(f"{fpath}/J_progress.npy")
    it = numpy.arange(1, len(J_progress) + 1)
    axes.loglog(it, J_progress, markevery=5, **kw)
axes.set_xlabel("Iteration")
axes.set_ylabel("Mean square error")
axes.grid(True, which="both")
axes.legend(ncol=2)
imgfile = "optimization_progress.png"
print(f"Saving to {imgfile}")
plt.savefig(imgfile, dpi=200, bbox_inches="tight")
