from model_config import read_station_data, sim_tz

import datetime
import h5py
import matplotlib.pyplot as plt
import netCDF4
import numpy


start_date = datetime.datetime(2022, 1, 15, tzinfo=sim_tz)
dates = [(start_date + datetime.timedelta(days=x)).strftime("%d/%m") for x in range(4)]

fig = plt.figure(figsize=(30, 10))
axes = fig.subplots(ncols=4, nrows=2)
for i, (sta, data) in enumerate(read_station_data().items()):
    region = data["region"]
    ax = axes[i // 4, i % 4]
    if sta.startswith("St"):
        ax.set_title(". ".join([sta[:2], sta[2:]]))
    elif sta.endswith("Port"):
        ax.set_title(" ".join([sta[:-4], sta[-4:]]))
    else:
        ax.set_title(sta)

    # Load the observation data
    if region == "NO":
        o = f"observations/{region}_TS_TG_{sta}TG_202201.nc"
    else:
        o = f"observations/{region}_TS_TG_{sta}_202201.nc"
    with netCDF4.Dataset(o, "r") as nc:
        time_obs = nc.variables["TIME"][:]
        vals_obs = nc.variables["SLEV"][:]

    # Load the simulation data
    f = f"outputs/diagnostic_timeseries_{sta}_elev.hdf5"
    with h5py.File(f, "r") as h5file:
        time = h5file["time"][:].flatten() / (24 * 3600.0)
        vals = h5file["elev"][:].flatten()

    # Trim the observation data to the time window of interest
    time_obs -= time_obs[0] + start_date.day
    vals_obs = vals_obs[time[0] <= time_obs]
    time_obs = time_obs[time[0] <= time_obs]
    vals_obs = vals_obs[time_obs <= time[-1]]
    time_obs = time_obs[time_obs <= time[-1]]

    # Subtract the average in each case
    vals_obs -= numpy.mean(vals_obs)
    vals -= numpy.mean(vals)

    # Plot on the same axes
    ax.plot(time_obs, vals_obs, "k-", zorder=3, label="Observation", lw=1.5)
    ax.plot(time, vals, "r:", zorder=3, label="Simulation", lw=1.5)
    ax.set_xticks(list(range(len(dates))))
    ax.set_xticklabels(dates)
    ax.set_xlim([time[0], time[-1]])
    if i // 4 == 1:
        ax.set_xlabel("Date")
    if i % 4 == 0:
        ax.set_ylabel("Elevation (m)")
    ax.grid(True)
    ax.legend(ncol=2)
imgfile = "north_sea_elev_ts.png"
print(f"Saving to {imgfile}")
plt.savefig(imgfile, dpi=300, bbox_inches="tight")
