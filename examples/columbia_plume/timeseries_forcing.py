"""
A test case for checking river flux time series interpolation.
"""
import netCDF4
import thetis.timezone as timezone
import thetis.interpolation as interpolation
import datetime


def test():
    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    zero_date = datetime.datetime(1969, 12, 31, 16, tzinfo=sim_tz)
    print('Epoch zero: {:} {:}'.format(
        timezone.datetime_to_epoch(datetime.datetime(1970, 1, 1, tzinfo=timezone.pytz.utc)),
        timezone.datetime_to_epoch(zero_date)))

    init_date = datetime.datetime(2006, 5, 15, tzinfo=sim_tz)
    print('sim. start: {:} {:}'.format(
        timezone.datetime_to_epoch(datetime.datetime(2006, 5, 15, 8, tzinfo=timezone.pytz.utc)),
        timezone.datetime_to_epoch(init_date)))

    river_flux_interp = interpolation.NetCDFTimeSeriesInterpolator(
        'forcings/stations/beaverarmy/flux_*.nc',
        ['flux'], init_date, scalars=[-1.0], allow_gaps=True)
    print('interpolated: {:} {:} {:}'.format(timezone.datetime_to_epoch(init_date), init_date, river_flux_interp(0.)[0]))

    ncd = netCDF4.Dataset('forcings/stations/beaverarmy/flux_2006.nc')
    t = ncd['time'][:]
    v = ncd['flux'][:]
    print('original:     {:} {:} {:}'.format(t[0], datetime.datetime.fromtimestamp(t[0], tz=timezone.pytz.utc), -v[0]))

    dt = 900
    for i in range(96):
        d = init_date + datetime.timedelta(seconds=i*dt)
        print('Time step {:3d}, {:}, flux: {:8.1f}'.format(i, d, river_flux_interp(i*dt)[0]))


if __name__ == '__main__':
    test()
