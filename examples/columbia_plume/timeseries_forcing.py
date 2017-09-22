"""
Implements time series interpolators that can be used as forcing terms.
"""
import netCDF4
import thetis.timezone as timezone
import thetis.interpolation as interpolation
import datetime


def test():
    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(1969, 12, 31, 16, tzinfo=sim_tz)
    print timezone.datetime_to_epoch(datetime.datetime(1970, 1, 1, tzinfo=timezone.utc_tz)), timezone.datetime_to_epoch(init_date)

    print timezone.datetime_to_epoch(datetime.datetime(2016, 5, 1, 8, tzinfo=timezone.utc_tz)), timezone.datetime_to_epoch(datetime.datetime(2016, 5, 1, tzinfo=sim_tz))

    init_date = datetime.datetime(2016, 5, 1, tzinfo=sim_tz)
    river_flux_interp = interpolation.NetCDFTimeSeriesInterpolator(
        'forcings/stations/bvao3/bvao3.0.A.FLUX/*.nc',
        ['flux'], init_date, scalars=[-1.0], allow_gaps=True)
    print timezone.datetime_to_epoch(init_date), init_date, river_flux_interp(0.)

    ncd = netCDF4.Dataset('forcings/stations/bvao3/bvao3.0.A.FLUX/201605.nc')
    t = ncd['time'][:]
    v = ncd['flux'][:]
    print t[0], datetime.datetime.fromtimestamp(t[0], tz=timezone.utc_tz), v[0]


if __name__ == '__main__':
    test()
