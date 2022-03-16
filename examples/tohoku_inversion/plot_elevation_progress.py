from model_config import read_station_data

import glob
import h5py
import matplotlib.pyplot as plt


inv_dir = "outputs_elev-init-optimization"
fpaths = glob.glob(f"{inv_dir}_*")
if len(fpaths) == 0:
    raise ValueError("Nothing to plot!")
stations = read_station_data().keys()
for fpath in fpaths:
    fig = plt.figure(figsize=(40, 20))
    axes = fig.subplots(4, 4)
    source_model = fpath.split(inv_dir + "_")[-1]

    for i, sta in enumerate(stations):

        o = f"observations/diagnostic_timeseries_{sta}_elev.hdf5"
        with h5py.File(o, "r") as h5file:
            time_obs = h5file["time"][:].flatten() / 60.0
            vals_obs = h5file["elev"][:].flatten()

        f = f"{fpath}/diagnostic_timeseries_progress_{sta}_elev.hdf5"
        with h5py.File(f, "r") as h5file:
            time = h5file["time"][:].flatten() / 60.0
            vals = h5file["elev"][:]

        ax = axes[i // 4, i % 4]
        ax.plot(time_obs, vals_obs, "k", zorder=3, label="Observation", lw=1.3)
        for i, v in enumerate(vals):
            v -= v[time >= time_obs[0]][0]
            ax.plot(time, v, label=f"_iter {i}", lw=0.5)
        ax.set_title(sta)
        ax.set_xlim([time_obs[0], time_obs[-1]])
        if i // 4 == 3:
            ax.set_xlabel("Time (min)")
        if i % 4 == 0:
            ax.set_ylabel("Elevation (m)")
        ax.legend(ncol=1)
        ax.grid(True)

    imgfile = f"optimization_progress_elev_ts_{source_model}.png"
    print(f"Saving to {imgfile}")
    plt.savefig(imgfile, dpi=200, bbox_inches="tight")
