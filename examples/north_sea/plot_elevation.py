from model_config import read_station_data
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4
import numpy
import cftime

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
        time_raw = nc.variables["TIME"][:]
        time_units = nc.variables["TIME"].getncattr('units')
        time_obs = cftime.num2pydate(time_raw, time_units)
        vals_obs = nc.variables["SLEV"][:]

    # Load the simulation data
    f = f"outputs/diagnostic_timeseries_{sta}_elev.hdf5"
    with h5py.File(f, "r") as h5file:
        time_raw = h5file["time"][:].flatten()
        time_units = h5file["time"].attrs['units']
        time = cftime.num2pydate(time_raw, time_units)
        vals = h5file["elev"][:].flatten()

    # Trim the observation data to the model time window
    filter_ix = (time_obs >= time[0]) * (time_obs <= time[-1])
    time_obs = time_obs[filter_ix]
    vals_obs = vals_obs[filter_ix]

    # Subtract the average in each case
    vals_obs -= numpy.mean(vals_obs)
    vals -= numpy.mean(vals)

    # Plot on the same axes
    ax.plot(time_obs, vals_obs, "k-", zorder=3, label="Observation", lw=1.5)
    ax.plot(time, vals, "r:", zorder=3, label="Simulation", lw=1.5)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
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
