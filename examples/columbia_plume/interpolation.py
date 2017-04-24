"""
Methods for data interpolation
"""
import glob
from timezone import *
import numpy as np
import scipy.spatial.qhull as qhull
import netCDF4


class GridInterpolator(object):
    """
    A reuseable griddata interpolator object

    Based on
    http://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    """
    def __init__(self, grid_xyz, target_xyz):
        # compute interpolation interpolation weights
        d = grid_xyz.shape[1]
        tri = qhull.Delaunay(grid_xyz)
        simplex = tri.find_simplex(target_xyz)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = target_xyz - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        self.vtx = vertices
        self.wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def __call__(self, values, fill_value=np.nan):
        """
        Interpolate values defined on grid_xyz to target_xyz.

        Uses fill_value for points outside grid_xyz
        """
        ret = np.einsum('nj,nj->n', np.take(values, self.vtx), self.wts)
        ret[np.any(self.wts < 0, axis=1)] = fill_value
        return ret


class TimeSearch(object):
    """
    Abstract base class for searching nearest time steps from a database
    """
    #@abstract_property
    def find(self, time, previous=False):
        """
        Find a next (previous) time stamp from a given time

        :arg float time: input time stamp
        :arg bool previous: if True, look for last time stamp before requested
            time. Otherwise returns next time stamp.
        :return: a (descriptor, time_index, time) tuple
        """
        pass


class NetCDFTimeSearch(TimeSearch):
    """
    Finds a nearest time stamp in a collection of netCDF files.
    """
    def __init__(self, file_pattern, init_date, netcdf_class):
        all_files = glob.glob(file_pattern)
        assert len(all_files) > 0, 'No files found: {:}'.format(file_pattern)

        self.netcdf_class = netcdf_class
        self.init_date = init_date
        self.sim_start_time = datetime_to_epoch(self.init_date)
        dates = []
        ncfiles = []
        for fn in all_files:
            print('Reading {:}'.format(fn))
            nc = self.netcdf_class(fn)
            ncfiles.append(nc)
            dates.append(nc.get_start_time())
        sort_ix = np.argsort(dates)
        self.files = np.array(all_files)[sort_ix]
        self.ncfiles = np.array(ncfiles)[sort_ix]
        self.start_datetime = np.array(dates)[sort_ix]
        self.start_times = [(s - self.init_date).total_seconds() for s in self.start_datetime]
        self.start_times = np.array(self.start_times)

    def simulation_time_to_datetime(self, t):
        return epoch_to_datetime(datetime_to_epoch(self.init_date) + t).astimezone(self.init_date.tzinfo)

    def find(self, simulation_time, previous=False):
        """
        Find file and time step within that file for given simulation time

        :arg float simulation_time: simulation time in seconds
        :kwarg bool previous: find previous existing time stamp instead of next
        :return: (filename, time index, simulation time) of found data
        """
        err_msg = 'No file found for time {:}'.format(self.simulation_time_to_datetime(simulation_time))
        ix = np.searchsorted(self.start_times, simulation_time)
        if ix > 0:
            candidates = [ix-1, ix]
        else:
            candidates = [ix]
        itime = None
        for i in candidates:
            try:
                nc = self.ncfiles[i]
                itime = nc.find_time_stamp(self.sim_start_time + simulation_time, previous=previous)
                time = nc.start_epoch + nc.timestep*itime - self.sim_start_time
                break
            except IndexError as e:
                pass
        if itime is None:
            raise Exception(err_msg)
        return self.files[i], itime, time


class NetCDFTime(object):
    """
    Describes the time stamps stored in a netCDF file.
    """
    scalars = {
        'seconds': 1.0,
        'days': 24*3600.0,
    }

    def __init__(self, filename):
        """
        Construct a new object by scraping data from the given netcdf file.

        :arg str filename: name of the netCDF file to read
        """
        self.filename = filename

        with netCDF4.Dataset(filename) as d:
            time_var = d['time']
            assert 'units' in time_var.ncattrs(), 'Time does not have units; {:}'.format(self.filename)
            unit_str = time_var.getncattr('units')
            msg = 'Unknown time unit "{:}" in {:}'.format(unit_str, self.filename)
            words = unit_str.split()
            assert words[0] in ['days', 'seconds'], msg
            self.time_unit = words[0]
            self.time_scalar = self.scalars[self.time_unit]
            assert words[1] == 'since', msg
            if len(words) == 3:
                # assuming format "days since 2000-01-01" in UTC
                base_date_srt = words[2]
                numbers = len(base_date_srt.split('-'))
                assert numbers == 3, msg
                try:
                    self.basetime = datetime.datetime.strptime(base_date_srt, '%Y-%m-%d').replace(tzinfo=utc_tz)
                except ValueError as e:
                    raise ValueError(msg)
            if len(words) == 4:
                # assuming format "days since 2000-01-01 00:00:00" in UTC
                # or "days since 2000-01-01 00:00:00-10"
                base_date_srt = ' '.join(words[2:4])
                assert len(words[2].split('-')) == 3, msg
                assert len(words[3].split(':')) == 3, msg
                if len(words[3].split('-')) == 2:
                    base_date_srt = base_date_srt[:-3]
                    tz_offset = int(words[3][-3:])
                    timezone = FixedTimeZone(tz_offset, 'UTC{:}'.format(tz_offset))
                else:
                    timezone = utc_tz
                try:
                    self.basetime = datetime.datetime.strptime(base_date_srt, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone)
                except ValueError as e:
                    raise ValueError(msg)
            self.start_time = self.basetime + datetime.timedelta(seconds=float(time_var[0]))

            self.ntimesteps = len(time_var)
            dt_arr = np.diff(time_var[:])
            assert np.allclose(dt_arr, dt_arr[0]), 'Time step is not constant. {:}'.format(self.filename)
            #self.timestep = np.round(dt_arr[0]*self.time_scalar, decimals=3)
            self.timestep = dt_arr[0]*self.time_scalar

            self.start_epoch = datetime_to_epoch(self.start_time)

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.start_time + datetime.timedelta(seconds=(self.ntimesteps - 1)*self.timestep)

    def find_time_stamp(self, t, previous=False):
        t_epoch = datetime_to_epoch(t) if isinstance(t, datetime.datetime) else t
        round_op = np.floor if previous else np.ceil
        offset = 0.0 if previous else 1e-8
        itime = int(round_op((t_epoch - self.start_epoch + offset)/self.timestep))
        if itime < 0:
            raise IndexError('Requested time out of bounds {:} < {:}'.format(t_epoch, self.start_epoch))
        if itime > self.ntimesteps - 1:
            raise IndexError('Requested time out of bounds {:} > {:}'.format(t_epoch, datetime_to_epoch(self.get_end_time())))
        return itime


class DBReader(object):
    """
    Abstract base class of file tree reader object
    """
    def __call__(self, descriptor, time_index):
        """
        Reads a time_index from the data base

        :arg str descriptor: a descriptor where to find the data (e.g. filename)
        :arg int time_index: time index to read
        :return: a float or numpy.array_like value
        """
        pass


class LinearTimeInterpolator(object):
    """
    Interpolates time series in time

    User must provide timesearch_obj that can be used to find time stamps from
    a database (e.g. netcdf file tree), and a reader that can read those time
    stamps into numpy arrays.
    """
    def __init__(self, timesearch_obj, reader):
        """
        :arg timesearch_obj: TimeSearch object
        :arg reader: DBReader object
        """
        self.timesearch = timesearch_obj
        self.reader = reader
        self.cache = {}

    def _get_from_cache(self, key):
        """
        Fetch data set from cache, read if not present
        """
        if key not in self.cache:
            self.cache[key] = self.reader(key[0], key[1])
        return self.cache[key]

    def _clean_cache(self, keys_to_keep):
        """
        Remove cached data sets that are no longer needed
        """
        for key in self.cache.keys():
            if key not in keys_to_keep:
                self.cache.pop(key)

    def __call__(self, t):
        """
        Interpolate at time t
        """
        prev_id = self.timesearch.find(t, previous=True)
        next_id = self.timesearch.find(t, previous=False)

        prev = self._get_from_cache(prev_id)
        next = self._get_from_cache(next_id)
        self._clean_cache([prev_id, next_id])

        # interpolate
        t_prev = prev_id[2]
        t_next = next_id[2]
        alpha =  (t - t_prev)/(t_next - t_prev)
        TOL = 1e-6
        assert alpha + TOL >= 0.0 and alpha <= 1.0 + TOL, \
            'Value {:} out of range {:} .. {:}'.format(t, t_prev, t_next)
        val = (1 - alpha)*prev + alpha*next
        return val
