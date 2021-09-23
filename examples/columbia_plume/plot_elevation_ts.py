"""
Plots elevation time series
"""
import h5py
from netCDF4 import Dataset
from thetis.timezone import *

import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict

timezone = FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2006, 5, 10, tzinfo=timezone)

epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


def simtime_to_datetime(simtime, init_date):
    return numpy.array([init_date + datetime.timedelta(seconds=float(t)) for t in simtime])


def simtime_to_epoch(simtime, init_date):
    offset = (init_date - epoch).total_seconds()
    return simtime + offset


def epoch_to_datetime(time):
    if isinstance(time, numpy.ndarray):
        return numpy.array([epoch + datetime.timedelta(seconds=float(t)) for t in time])
    return epoch + datetime.timedelta(seconds=float(time))


def read_netcdf(fn):
    d = Dataset(fn)
    assert 'time' in d.variables.keys(), 'netCDF file does not contain time variable'
    out = OrderedDict()
    # assuming epoch time
    out['time'] = d['time'][:]
    for k in d.variables.keys():
        if k == 'time':
            continue
        out[k] = d[k][:]
    return out


def read_hdf5(fn):
    d = h5py.File(fn)
    assert 'time' in d, 'hdf5 file does not contain time variable'
    out = OrderedDict()
    # assuming simulation time
    time = simtime_to_epoch(d['time'][:], init_date)
    out['time'] = time
    for k in d.keys():
        if k == 'time':
            continue
        out[k] = d[k][:]
    return out


def make_plot(data):

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    t_min = numpy.finfo('d').max
    t_max = numpy.finfo('d').min
    for tag in data:
        d = data[tag]
        time = d['time']
        vals = d[list(d.keys())[1]]

        datetime_arr = epoch_to_datetime(time)
        t_min = min(t_min, time[0])
        t_max = max(t_max, time[-1])

        ax.plot(datetime_arr, vals, label=tag, alpha=0.8)

    ax.set_ylabel('Elevation [m]')
    fig.autofmt_xdate()
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.))

    date_str = '_'.join([epoch_to_datetime(t).strftime('%Y-%m-%d') for t in [t_min, t_max]])
    imgfn = 'ts_cmop_elev_tpoin_{:}.png'.format(date_str)
    print('Saving {:}'.format(imgfn))
    fig.savefig(imgfn, dpi=200, bbox_inches='tight')


def process(file_list):

    files = [f for f in file_list if f[-1] != ':']
    tags = [f[:-1] for f in file_list if f[-1] == ':']
    if len(tags) == 0:
        tags = files
    assert len(tags) == len(files)

    data = OrderedDict()

    for f, t in zip(files, tags):
        if f.endswith('.hdf5'):
            d = read_hdf5(f)
        elif f.endswith('.nc'):
            d = read_netcdf(f)
        else:
            raise IOError('Unknown file format {:}'.format(f))
        data[t] = d

    make_plot(data)


def get_argparser():
    return parser


def parse_options():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='file_list', type=str, nargs='+',
                        help='hdf5 or netcdf file to plot')
    args = parser.parse_args()
    process(args.file_list)


if __name__ == '__main__':
    parse_options()
