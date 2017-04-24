"""
Test netcd time interpolation
"""
from interpolation import *
import numpy as np
import random
from scipy.interpolate import interp1d
import netCDF4

# data set
np.random.seed(2)

# construct data set
x_scale = 100.
ndata = 35
xx = np.linspace(0, x_scale, ndata)
yy = np.random.rand(*xx.shape)
timestep = np.diff(xx).mean()

# construct interpolation points
ninterp = 100
x_interp = np.random.rand(ninterp)*x_scale

# get correct solution with scipy
y_interp = interp1d(xx, yy)(x_interp)

# save into a bunch of netcdf files
ncfile_pattern = 'tmp/testfile_{:}.nc'
basetime = FixedTimeZone(-6, 'FFF').localize(datetime.datetime(1972, 1, 1))
basetime_str = basetime.strftime('%Y-%m-%d %H:%M:%S')+basetime.strftime('%z')[:-2]
#basetime = utc_tz.localize(datetime.datetime(1972, 1, 1))
#basetime_str = basetime.strftime('%Y-%m-%d')

nfiles = 5
for i in range(nfiles):
    n = ndata/nfiles
    ix = np.arange(n*i, n*(i+1))
    fn = ncfile_pattern.format(i)
    d = netCDF4.Dataset(fn, 'w')
    time = d.createDimension('time', None)
    time_var = d.createVariable('time', 'f8', ('time',))
    time_var.long_name = 'Time'
    time_var.standard_name = 'time'
    time_var.units = 'seconds since {:}'.format(basetime_str)

    data_var = d.createVariable('data', 'f8', ('time',))
    time_var[:] = xx[ix]
    data_var[:] = yy[ix]
    print i, epoch_to_datetime(time_var[0] + datetime_to_epoch(basetime)),\
        epoch_to_datetime(time_var[-1] + datetime_to_epoch(basetime))

# test NetCDFTime
for i in range(nfiles):
    nct = NetCDFTime(ncfile_pattern.format(i))
    t_offset = xx[i*ndata/nfiles]
    t_offset_end = xx[(i+1)*ndata/nfiles - 1]
    assert nct.ntimesteps == ndata/nfiles
    assert nct.time_unit == 'seconds'
    assert nct.start_time == basetime + datetime.timedelta(seconds=t_offset)
    assert np.allclose(nct.timestep, timestep)
    assert nct.get_start_time() == epoch_to_datetime(datetime_to_epoch(basetime) + t_offset)
    assert (nct.get_end_time() - basetime).total_seconds() - t_offset_end < 1e-6
    assert nct.find_time_stamp(datetime_to_epoch(basetime) + t_offset + 10., previous=True) == 3
    assert nct.find_time_stamp(datetime_to_epoch(basetime) + t_offset + 10., previous=False) == 4

# test NetCDFTimeSearch
init_date = basetime
nts = NetCDFTimeSearch(ncfile_pattern.format('*'), init_date, NetCDFTime)
assert nts.simulation_time_to_datetime(xx[0]) == init_date
assert nts.simulation_time_to_datetime(xx[10]) == init_date + datetime.timedelta(seconds=xx[10])

for t in np.linspace(1., x_scale-1., 20):
    # search for time index and assert time stamp from nc file
    fn, itime, ftime = nts.find(t, previous=True)
    time_var = netCDF4.Dataset(fn)['time']
    correct = xx[np.searchsorted(xx, t) - 1]
    assert np.allclose(time_var[itime], ftime)
    assert np.allclose(time_var[itime], correct)
    assert time_var[itime] < t

    fn, itime, ftime = nts.find(t, previous=False)
    time_var = netCDF4.Dataset(fn)['time']
    correct = xx[np.searchsorted(xx, t)]
    assert np.allclose(time_var[itime], ftime)
    assert np.allclose(time_var[itime], correct)
    assert time_var[itime] > t

# test full LinearTimeInterpolator!
