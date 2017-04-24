"""
Implements time series interpolators that can be used as forcing terms.
"""
from scipy.interpolate import interp1d
import netCDF4
import numpy as np
import glob
from timezone import *


def read_nc_time_series(filename, time_var, value_var):
    ncd = netCDF4.Dataset(filename)
    time = ncd['time'][:]
    vals = ncd['flux'][:]
    return time, vals


def gather_nc_files(file_pattern, time_var, value_var):
    file_list = sorted(glob.glob(file_pattern))
    time = []
    vals = []
    for fn in file_list:
        t, v = read_nc_time_series(fn, time_var, value_var)
        time.append(t)
        vals.append(v)
    time = np.concatenate(tuple(time))
    vals = np.concatenate(tuple(vals))
    return time, vals


class NetCDFTimeSeriesInterpolator(object):
    """
    Time series interpolator that reads the data from a sequence of netCDF files.
    """
    def __init__(self, file_pattern, time_var, value_var, init_date, t_end,
                 scalar=1.0, data_tz=None):
        """
        :arg string file_pattern: where to look for netCDF file(s), e.g.
            "some/dir/file_*.nc"
        :arg string time_var: name of the time variable in the netCDF file
        :arg string value_var: name of the value variable in the netCDF file
        :arg datetime init_date: simulation start time
        :arg float t_end: simulation duration in seconds
        :kwarg float scalar: value will be scaled by this value (default: 1.0)
        """
        assert init_date.tzinfo is not None, 'init_date must have time zone information'
        if data_tz is None:
            # Assume data is in utc
            # TODO try to sniff from netCDF metadata
            data_tz = utc_tz
        time, vals = gather_nc_files(file_pattern, time_var, value_var)
        #time_offset = datetime_to_epoch(init_date.astimezone(data_tz))
        time_offset = datetime_to_epoch(init_date)
        time_sim = time - time_offset

        self.scalar = scalar
        vals = self.scalar*vals

        self.interpolator = interp1d(time_sim, vals)

    def get(self, t):
        """Evaluate the time series at time t."""
        return self.interpolator(t)


def test():
    sim_tz = FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(1969, 12, 31, 16, tzinfo=sim_tz)
    print datetime_to_epoch(datetime.datetime(1970, 1, 1, tzinfo=utc_tz)), datetime_to_epoch(init_date)

    print datetime_to_epoch(datetime.datetime(2016, 5, 1, 8, tzinfo=utc_tz)), datetime_to_epoch(datetime.datetime(2016, 5, 1, tzinfo=sim_tz))

    init_date = datetime.datetime(2016, 5, 1, tzinfo=sim_tz)
    t_end = 15*24*3600.
    river_flux_interp = NetCDFTimeSeriesInterpolator(
        'forcings/stations/bvao3/bvao3.0.A.FLUX/*.nc',
        'time', 'flux', init_date, t_end, scalar=-1.0)
    print datetime_to_epoch(init_date), init_date, river_flux_interp.get(0.)

    ncd = netCDF4.Dataset('forcings/stations/bvao3/bvao3.0.A.FLUX/201605.nc')
    t = ncd['time'][:]
    v = ncd['flux'][:]
    print t[0], datetime.datetime.fromtimestamp(t[0], tz=utc_tz), v[0]


if __name__ == '__main__':
    test()
