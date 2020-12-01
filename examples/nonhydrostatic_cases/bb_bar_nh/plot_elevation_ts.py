"""
Plots elevation time series
"""
import h5py
import os
from netCDF4 import Dataset
from thetis.timezone import *

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from scipy import interpolate

timezone = FixedTimeZone(-8, 'PST')
init_date = datetime.datetime(2006, 5, 10, tzinfo=timezone)

epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)


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
    time = d['time'][:]
    out['time'] = time
    for k in d.keys():
        if k == 'time':
            continue
        out[k] = d[k][:]
    return out


def read_measured_data(dirPath):
    out = OrderedDict()
    AllfileName = os.listdir(dirPath)
    for subName in AllfileName:
        child = os.path.join(dirPath, subName)
        f = open(child, 'r')
        lines = f.readlines()
        timeVals = []
        eleVals = []
        for i in lines:
            timeVals.append(float(i.split()[0]))
            eleVals.append(float(i.split()[1]))
        out[subName] = OrderedDict()
        out[subName]['time'] = timeVals
        out[subName]['elevation'] = eleVals
    return out


def make_plot(data, measured):

    fig = plt.figure(figsize=(10, 6))

    for i in range(8):
        locals()['ax_'+str(i+1)] = fig.add_subplot(4, 2, i+1)
        ax = locals()['ax_'+str(i+1)]

        # experimental data
        g_name = 'gauge' + str(i+4) + '.txt'
        x_exp = measured[g_name]['time']
        y_exp = measured[g_name]['elevation']

        # numerical results
        xshift = -0.1
        x_num = data['time'] + xshift
        y_num = data[list(data.keys())[i+1]]
        tck = interpolate.splrep(x_num, y_num, s=0)
        xnew = np.arange(0, 40, 0.01).reshape(4000, 1)
        ynew = interpolate.splev(xnew, tck, der=0)

        ax.plot(x_exp, y_exp, 'ko', markersize=4, markerfacecolor='w', label='Measured', alpha=0.8)
        ax.plot(xnew, ynew, 'b', linewidth=1, label='Gauge '+str(i+4), alpha=1.0)

        if i == 6 or i == 7:
            ax.set_xlabel('Time (s)')
        if i % 2 == 0:
            ax.set_ylabel('Elevation (m)')

        ax.legend(loc='upper right', fontsize=6, frameon=False)

        ax.set_xlim((33, 39))
        ax.set_ylim((-0.03, 0.05))
        ax.set_yticks([-0.02, 0, 0.02, 0.04])
        plt.tick_params(labelsize=8)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.4)
    plt.suptitle('Surface elevation time series at eight wave gauges for the submerged bar case')

    imgfn = 'ts_cmop_elev.png'
    print('Saving {:}'.format(imgfn))
    fig.savefig(imgfn, dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    DiagnosticFile = './outputs_bbbar_2d/diagnostic_gauges.hdf5'
    MeasuredDataPath = './measured_data'
    if DiagnosticFile.endswith('.hdf5'):
        d = read_hdf5(DiagnosticFile)
    elif DiagnosticFile.endswith('.nc'):
        d = read_netcdf(DiagnosticFile)
    else:
        raise IOError('Unknown file format {:}'.format(DiagnosticFile))
    m = read_measured_data(MeasuredDataPath)
    make_plot(d, m)
