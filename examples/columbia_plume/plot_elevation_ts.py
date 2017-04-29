"""
Plots elevation time series
"""
import h5py
import netCDF4
from timezone import *
from timeseries_forcing import *

import matplotlib.pyplot as plt
import matplotlib


# load observations into a interpolator instance
timezone = FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2016, 5 , 1, tzinfo=timezone)
obs_interp = NetCDFTimeSeriesInterpolator('forcings/stations/tpoin/tpoin.0.A.External/2016*.nc', 'time', 'elevation', init_date)

# load simulated time series
d = h5py.File('outputs_coarse/diagnostic_timeseries_elev_2d_tpoin.hdf5')
d.keys()
time = d['time'][:]
elev = d['value'][:]

mask = np.ones_like(time, dtype=bool)
mask[[1, 3, 5]] = False

time = time[mask]
elev = elev[mask]

def simtime_to_datetime(simtime, init_date):
    return np.array([init_date + datetime.timedelta(seconds=float(t)) for t in simtime])

datetime_arr = simtime_to_datetime(time, init_date)
plot_dates = matplotlib.dates.date2num(datetime_arr)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)

ax.plot(datetime_arr, obs_interp(time), 'r', label='obs tpoin')
ax.plot(datetime_arr, elev, 'k', label='model')
ax.set_ylabel('Elevation [m]')
fig.autofmt_xdate()
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.))
#plt.show()

date_str = '_'.join([d.strftime('%Y-%m-%d') for d in [datetime_arr[0], datetime_arr[-1]]])
imgfn = 'ts_cmop_elev_tpoin_{:}.png'.format(date_str)
print('Saving {:}'.format(imgfn))
fig.savefig(imgfn, dpi=200, bbox_inches='tight')
