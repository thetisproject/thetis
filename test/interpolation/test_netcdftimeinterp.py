"""
Test netcd time interpolation
"""
from thetis.interpolation import *
from scipy.interpolate import interp1d
import netCDF4
import numpy
import pytest
import os


@pytest.fixture()
def dataset(request):
    # create random time series on regular intervals
    numpy.random.seed(2)

    x_scale = 100.
    ndata = 35
    xx = numpy.linspace(0, x_scale, ndata)
    yy = numpy.random.rand(*xx.shape)

    # construct interpolation points
    ninterp = 100
    x_interp = numpy.random.rand(ninterp)*x_scale

    # get correct solution with scipy
    y_interp = interp1d(xx, yy)(x_interp)
    return (ndata, xx, yy, x_interp, y_interp)


@pytest.fixture()
def tmp_outputdir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('outputs')
    return str(fn)


@pytest.fixture(params=[True, False], ids=['custom-timezone', 'utc-timezone'])
def netcdf_files(request, dataset, tmp_outputdir):
    custom_timezone = request.param
    ndata, xx, yy, x_interp, y_interp = dataset

    # save into a bunch of netcdf files
    ncfile_pattern = tmp_outputdir + '/testfile_{:}.nc'
    if custom_timezone:
        basetime = FixedTimeZone(-6, 'FFF').localize(datetime.datetime(1972, 1, 1))
        basetime_str = basetime.strftime('%Y-%m-%d %H:%M:%S')+basetime.strftime('%z')[:-2]
    else:
        basetime = pytz.utc.localize(datetime.datetime(1972, 1, 1))
        basetime_str = basetime.strftime('%Y-%m-%d')

    print(f'Basetime: {basetime}')
    nfiles = 5
    all_files = []
    for i in range(nfiles):
        n = int(ndata/nfiles)
        ix = numpy.arange(n*i, n*(i+1))
        fn = ncfile_pattern.format(i)
        d = netCDF4.Dataset(fn, 'w')
        d.createDimension('time', None)
        time_var = d.createVariable('time', 'f8', ('time', ))
        time_var.long_name = 'Time'
        time_var.standard_name = 'time'
        time_var.units = f'seconds since {basetime_str}'
        time_var.calendar = 'gregorian'

        data_var = d.createVariable('data', 'f8', ('time',))
        time_var[:] = xx[ix]
        data_var[:] = yy[ix]
        first = epoch_to_datetime(time_var[0] + datetime_to_epoch(basetime))
        last = epoch_to_datetime(time_var[-1] + datetime_to_epoch(basetime))
        print(f'{i} {first} {last}')
        all_files.append(fn)

    def teardown():
        for fn in all_files:
            os.remove(fn)
    request.addfinalizer(teardown)

    return basetime, ncfile_pattern, all_files


def test_netcdftime(dataset, netcdf_files):
    ndata, xx, yy, x_interp, y_interp = dataset
    basetime, ncfile_pattern, files = netcdf_files
    nfiles = len(files)
    # test NetCDFTimeParser
    for i in range(nfiles):
        nct = NetCDFTimeParser(files[i])
        nn = int(ndata/nfiles)
        t_offset = xx[i*nn]
        t_offset_end = xx[(i+1)*nn - 1]
        assert nct.start_time == basetime + datetime.timedelta(seconds=t_offset)
        assert nct.get_start_time() == epoch_to_datetime(datetime_to_epoch(basetime) + t_offset)
        assert (nct.get_end_time() - basetime).total_seconds() - t_offset_end < 1e-6
        assert nct.find_time_stamp(datetime_to_epoch(basetime) + t_offset + 10., previous=True) == 3
        assert nct.find_time_stamp(datetime_to_epoch(basetime) + t_offset + 10., previous=False) == 4


def test_netcdftimesearch(dataset, netcdf_files):
    ndata, xx, yy, x_interp, y_interp = dataset
    basetime, ncfile_pattern, files = netcdf_files
    init_date = basetime
    nts = NetCDFTimeSearch(ncfile_pattern.format('*'), init_date, NetCDFTimeParser)
    assert nts.simulation_time_to_datetime(xx[0]) == init_date
    assert nts.simulation_time_to_datetime(xx[10]) == init_date + datetime.timedelta(seconds=xx[10])

    x_max = xx.max()
    for t in numpy.linspace(1., x_max-1., 20):
        # search for time index and assert time stamp from nc file
        fn, itime, ftime = nts.find(t, previous=True)
        time_var = netCDF4.Dataset(fn)['time']
        correct = xx[numpy.searchsorted(xx, t) - 1]
        assert numpy.allclose(time_var[itime], ftime)
        assert numpy.allclose(time_var[itime], correct)
        assert time_var[itime] < t

        fn, itime, ftime = nts.find(t, previous=False)
        time_var = netCDF4.Dataset(fn)['time']
        correct = xx[numpy.searchsorted(xx, t)]
        assert numpy.allclose(time_var[itime], ftime)
        assert numpy.allclose(time_var[itime], correct)
        assert time_var[itime] > t


def test_lineartimeinterpolator(dataset, netcdf_files, plot=False):
    ndata, xx, yy, x_interp, y_interp = dataset
    basetime, ncfile_pattern, files = netcdf_files
    init_date = basetime
    timesearch_obj = NetCDFTimeSearch(ncfile_pattern.format('*'), init_date, NetCDFTimeParser)
    reader = NetCDFTimeSeriesReader(['data'])

    lintimeinterp = LinearTimeInterpolator(timesearch_obj, reader)
    y_interp2 = numpy.zeros_like(y_interp)
    for i in range(len(y_interp2)):
        y_interp2[i] = lintimeinterp(x_interp[i])[0]

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(xx, yy, 'k')
        plt.plot(x_interp, y_interp, 'bo')
        plt.plot(x_interp, y_interp2, 'rx')
        plt.show()

    assert numpy.allclose(y_interp, y_interp2)
