import glob
import matplotlib.pyplot as plt


fig, axes = plt.subplots()
inv_dir = "outputs_elev-init-optimization"
fpaths = glob.glob(f"{inv_dir}_*")
if len(fpaths) == 0:
    raise ValueError("Nothing to plot!")
for fpath in fpaths:
    source_model = fpath.split(inv_dir + "_")[-1]
    J_progress = []
    for line in open(f"{fpath}/log", "r"):
        if not line.startswith("line search"):
            continue
        J = line.split("J=")[1].split(",")[0]
        J_progress.append(float(J))
    axes.plot(J_progress, label=source_model)
axes.set_xlabel("Iteration")
axes.set_ylabel("Mean square error")
axes.grid(True)
axes.legend()
imgfile = "optimization_progress.png"
print(f"Saving to {imgfile}")
plt.savefig(imgfile, dpi=200, bbox_inches="tight")
