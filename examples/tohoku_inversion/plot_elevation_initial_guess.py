from model_config import read_station_data

import glob
import h5py
import matplotlib.pyplot as plt


fwd_dir = "outputs_forward"
fpaths = glob.glob(f"{fwd_dir}*")
if len(fpaths) == 0:
    raise ValueError("Nothing to plot!")
stations = read_station_data().keys()
for fpath in fpaths:
    source_model = fpath.split(fwd_dir + "_")[-1]
    fig = plt.figure(figsize=(40, 20))
    axes = fig.subplots(4, 4)

    for i, sta in enumerate(stations):
        ax = axes[i // 4, i % 4]
        ax.set_title(sta)

        o = f"observations/diagnostic_timeseries_{sta}_elev.hdf5"
        with h5py.File(o, "r") as h5file:
            time_obs = h5file["time"][:].flatten() / 60.0
            vals_obs = h5file["elev"][:].flatten()
        ax.plot(time_obs, vals_obs, "k", zorder=3, label="Observation", lw=1.3)

        f = f"{fpath}/diagnostic_timeseries_{sta}_elev.hdf5"
        with h5py.File(f, "r") as h5file:
            time = h5file["time"][:].flatten() / 60.0
            vals = h5file["elev"][:].flatten()
        if len(vals) > 0:
            vals -= vals[time >= time_obs[0]][0]
            ax.plot(time, vals, "r:", zorder=3, label="Initial guess", lw=1.5)

        ax.set_xlim([time_obs[0], time_obs[-1]])
        if i // 4 == 3:
            ax.set_xlabel("Time (min)")
        if i % 4 == 0:
            ax.set_ylabel("Elevation (m)")
        ax.legend(ncol=1)
        ax.grid(True)

    imgfile = f"initial_guess_ts_{source_model}.png"
    print(f"Saving to {imgfile}")
    plt.savefig(imgfile, dpi=200, bbox_inches="tight")
