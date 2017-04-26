"""
Methods for data interpolation
"""
import glob
import os
from timezone import *
import numpy as np
import scipy.spatial.qhull as qhull
import netCDF4
from abc import abstractmethod
from firedrake import *

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
    @abstractmethod
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


class FileTreeReader(object):
    """
    Abstract base class of file tree reader object
    """
    @abstractmethod
    def __call__(self, descriptor, time_index):
        """
        Reads a time_index from the data base

        :arg str descriptor: a descriptor where to find the data (e.g. filename)
        :arg int time_index: time index to read
        :return: a list of floats or numpy.array_like objects
        """
        pass


class NetCDFReader(FileTreeReader):
    """
    A simple netCDF reader that returns a time slice of the given variable.
    """
    def __init__(self, variable_list, time_variable_name='time'):
        self.variable_list = variable_list
        self.time_variable_name = time_variable_name
        self.time_dim = None
        self.ndims = None

    def _detect_time_dim(self, ncfile):
        assert self.time_variable_name in ncfile.dimensions
        nc_var = ncfile[self.variable_list[0]]
        assert self.time_variable_name in nc_var.dimensions
        self.time_dim = nc_var.dimensions.index('time')
        self.ndims = len(nc_var.dimensions)

    def _get_slice(self, time_index):
        """
        Returns a slice object that extracts a single time index
        """
        if self.ndims == 1:
            return time_index
        slice_list = [slice(None, None, None)]*self.ndims
        slice_list[self.time_dim] = slice(time_index, time_index+1, None)
        return slice

    def __call__(self, filename, time_index):
        """
        Reads a time_index from the data base

        :arg str filename: netcdf file where to find the data
        :arg int time_index: time index to read
        :return: a float or numpy.array_like value
        """
        assert os.path.isfile(filename), 'File not found: {:}'.format(filename)
        with netCDF4.Dataset(filename) as ncfile:
            if self.time_dim is None:
                self._detect_time_dim(ncfile)
            output = []
            for var in self.variable_list:
                values = ncfile[var][self._get_slice(time_index)]
                output.append(values)
            return output


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
        :arg reader: FileTreeReader object
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

        val = [(1.0 - alpha)*p + alpha*n for p, n in zip(prev, next)]
        return val


class NetCDFSpatialInterpolator(FileTreeReader):
    """
    Wrapper class that provides FileTreeReader API for grid interpolators
    """
    def __init__(self, grid_interpolator, variable_list):
        self.grid_interpolator = grid_interpolator
        self.variable_list = variable_list

    def __call__(self, filename, time_index):
        return self.grid_interpolator.interpolate(filename, self.variable_list, time_index)


class NetCDFLatLonInterpolator2d(object):
    """
    Interpolates netCDF data on local 2D unstructured mesh
    """
    def __init__(self, function_space, to_latlon):
        """
        :arg function_space: target Firedrake FunctionSpace
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        """
        self.function_space = function_space

        # construct local coordinates
        xy = SpatialCoordinate(self.function_space.mesh())
        fsx = Function(self.function_space).interpolate(xy[0]).dat.data_with_halos
        fsy = Function(self.function_space).interpolate(xy[1]).dat.data_with_halos

        mesh_lonlat = []
        for node in range(len(fsx)):
            lat, lon = to_latlon(fsx[node], fsy[node])
            mesh_lonlat.append((lon, lat))
        self.mesh_lonlat = np.array(mesh_lonlat)

        self._initialized = False

    def _get_subset_nodes(self, grid_x, grid_y, target_x, target_y):
        """
        Retuns grid nodes that are necessary for intepolating onto target_x,y
        """
        orig_shape = grid_x.shape
        grid_xy = np.array((grid_x.ravel(), grid_y.ravel())).T
        target_xy = np.array((target_x.ravel(), target_y.ravel())).T
        tri = qhull.Delaunay(grid_xy)
        simplex = tri.find_simplex(target_xy)
        vertices = np.take(tri.simplices, simplex, axis=0)
        nodes = np.unique(vertices.ravel())
        nodes_x, nodes_y = np.unravel_index(nodes, orig_shape)

        # x and y bounds for reading a subset of the netcdf data
        ind_x = np.arange(nodes_x.min(), nodes_x.max() + 1)
        ind_y = np.arange(nodes_y.min(), nodes_y.max() + 1)

        return nodes, ind_x, ind_y

    def _create_interpolator(self, ncfile):
        # create grid interpolator
        grid_lat_full = ncfile['lat'][:]
        grid_lon_full = ncfile['lon'][:]


        self.nodes, self.ind_lon, self.ind_lat = self._get_subset_nodes(
            grid_lon_full,
            grid_lat_full,
            self.mesh_lonlat[:, 0],
            self.mesh_lonlat[:, 1]
        )

        grid_lat = ncfile['lat'][self.ind_lon, self.ind_lat].ravel()
        grid_lon = ncfile['lon'][self.ind_lon, self.ind_lat].ravel()
        grid_lonlat = np.array((grid_lon, grid_lat)).T
        self.interpolator = GridInterpolator(grid_lonlat, self.mesh_lonlat)
        self._initialized = True

        # debug: plot subsets
        # import matplotlib.pyplot as plt
        # plt.plot(grid_lon_full, grid_lat_full, 'k.')
        # plt.plot(grid_lonlat[:, 0], grid_lonlat[:, 1], 'b.')
        # plt.plot(self.mesh_lonlat[:, 0], self.mesh_lonlat[:, 1], 'r.')
        # plt.show()

    def interpolate(self, nc_filename, variable_list, itime):
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                # TODO generalize data dimensions, sniff from netcdf file
                grid_data = ncfile[var][itime, self.ind_lon, self.ind_lat].ravel()
                data = self.interpolator(grid_data)
                output.append(data)
            return output
