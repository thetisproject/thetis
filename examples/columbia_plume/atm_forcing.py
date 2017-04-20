"""
Wind stress from WRF atmospheric model

wind stress is defined as

.. math:
    tau_w = C_D \rho_{air} \|U_{10}\| U_{10}

where :math:`C_D` is the drag coefficient, :math:`\rho_{air}` is the density of
air, and :math:`U_{10}` is wind speed 10 m above the sea surface.

In practice `C_D` depends on the wind speed. A commonly-used formula is that of
Large and Pond (1981):

.. math::
    C_D = 1.2\times10^{-3},\ \text{for}\ 4 < \|U_{10}\| < 11 m/s
    C_D = 10^{-3}(0.49 + 0.065\|U_{10}\|) ,\ \text{for}\ 11 < \|U_{10}\| < 25 m/s

Formulation is based on
Large and Pond (1981), J. Phys. Oceanog., 11, 324-336
"""
import coordsys_spcs
from firedrake import *
import numpy as np
import datetime
import scipy.interpolate
import glob
import os
import pytz
import netCDF4

utc_tz = pytz.timezone('UTC')
epoch = datetime.datetime(1970, 1, 1, tzinfo=utc_tz)


def datetime_to_epoch(t):
    """Convert python datetime object to epoch time stamp.
    By default, time.mktime converts from system's local time zone to UTC epoch.
    Here the input is treated as PST and local time zone information is discarded.
    """
    return (t - epoch).total_seconds()


rho_air = 1.22  # kg/m3


def to_latlon(x, y, positive_lon=False):
    lon, lat = coordsys_spcs.spcs2lonlat(x, y)
    if positive_lon and lon < 0.0:
        lon += 360.
    return lat, lon


def compute_wind_stress(wind_u, wind_v):
    wind_mag = np.hypot(wind_u, wind_v)
    if wind_mag < 11.0:
        C_D = 1.2e-3
    else:
        C_D = 10e-3*(0.49 + 0.065*wind_mag)
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import itertools


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
        ret = np.einsum('nj,nj->n', np.take(values, self.vtx), self.wts)
        ret[np.any(self.wts < 0, axis=1)] = fill_value
        return ret


class WindStressForcing(object):
    def __init__(self, wind_stress_field, init_time):

        self.wind_stress_field = wind_stress_field

        # determine nodes at the boundary
        fs = self.wind_stress_field.function_space()
        self.nodes = np.arange(self.wind_stress_field.dat.data_with_halos.shape[0])

        xy = SpatialCoordinate(fs.mesh())
        fsx = Function(fs).interpolate(xy[0]).dat.data_with_halos
        fsy = Function(fs).interpolate(xy[1]).dat.data_with_halos

        # compute lat lon bounds for each process
        if len(self.nodes) > 0:
            bounds_x = [fsx[self.nodes].min(), fsx[self.nodes].max()]
            bounds_y = [fsy[self.nodes].min(), fsy[self.nodes].max()]
            bounds_lat = [1e20, -1e20]
            bounds_lon = [1e20, -1e20]
            for x in bounds_x:
                for y in bounds_y:
                    atm_lat, atm_lon = to_latlon(x, y)
                    bounds_lat[0] = min(bounds_lat[0], atm_lat)
                    bounds_lat[1] = max(bounds_lat[1], atm_lat)
                    bounds_lon[0] = min(bounds_lon[0], atm_lon)
                    bounds_lon[1] = max(bounds_lon[1], atm_lon)

            ranges = (bounds_lat, bounds_lon)
        else:
            ranges = None

        mesh_lonlat = []
        for node in self.nodes:
            lat, lon = to_latlon(fsx[node], fsy[node])
            mesh_lonlat.append((lon, lat))
        mesh_lonlat = np.array(mesh_lonlat)

        # get and example file FIXME
        nc_atm_file = 'forcings/atm/wrf/wrf_air.2016_05_01.nc'
        itime = 0


        # create grid interpolator
        d = netCDF4.Dataset(nc_atm_file)
        atm_shape = d['lat'].shape
        atm_lat = d['lat'][:]
        atm_lon = d['lon'][:]
        #atm_lonlat_orig = np.array((atm_lon.ravel(), atm_lat.ravel())).T

        def get_subset_nodes(grid_x, grid_y, target_x, target_y):
            """
            Retuns grid nodes that are necessary for intepolating onto target_xyz
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

        self.nodes, self.ind_x, self.ind_y = get_subset_nodes(atm_lon, atm_lat, mesh_lonlat[:, 0], mesh_lonlat[:, 1])

        atm_lat = d['lat'][self.ind_x, self.ind_y].ravel()
        atm_lon = d['lon'][self.ind_x, self.ind_y].ravel()
        atm_lonlat = np.array((atm_lon, atm_lat)).T
        self.interpolator = GridInterpolator(atm_lonlat, mesh_lonlat)

        #import matplotlib.pyplot as plt
        #plt.plot(atm_lonlat_orig[:, 0], atm_lonlat_orig[:, 1], 'k.')
        #plt.plot(atm_lonlat[:, 0], atm_lonlat[:, 1], 'b.')
        #plt.plot(lonlat[:, 0], lonlat[:, 1], 'r.')
        #plt.show()


        #atm_val = d['uwind'][itime, self.ind_x, self.ind_y].ravel()
        #import time
        #t0 = time.clock()
        #vals = scipy.interpolate.griddata(atm_lonlat, atm_val, lonlat, method='linear')
        #print 'duration', time.clock() - t0
        #t0 = time.clock()
        #self.interpolator = GridInterpolator(atm_lonlat, lonlat)
        #print 'duration', time.clock() - t0
        #t0 = time.clock()
        #vals2 = self.interpolator(atm_val)
        #print 'duration', time.clock() - t0
        #print np.allclose(vals, vals2)

        wind_u = self.interpolator(d['uwind'][itime, self.ind_x, self.ind_y].ravel())
        # TODO add coordsys vector rotation

    def set_tidal_field(self, t):
        pass
        # given time t
        # figure out the file and time index for previous time stamp
        # read values from netcdf
        # interpolate on mesh
        # store field in cache, prev_vals
        # repeat same procedure for the next time stamp
        # invoke temporal interpolator to evaluate field at time t

        # TODO needed:
        # - class that gets the files and time steps, prev or next
        #   - returns fn, itime, time
        # - time interpolator object for each field
        #   - stores previous values, does time interp
        #   - uses self.interpolator


class WRFNetCDFFile(object):
    """
    NetCDF file that handles time conversion
    """
    scalars = {
        'seconds': 1.0,
        'days': 24*3600.0,
    }

    def __init__(self, filename):
        self.filename = filename

        with netCDF4.Dataset(filename) as d:
            time_var = d['time']
            assert 'units' in time_var.ncattrs(), 'Time does not have units; {:}'.format(self.filename)
            unit_str = time_var.getncattr('units')
            msg = 'Unknown time unit {:} in {:}'.format(unit_str, self.filename)
            words = unit_str.split()
            assert len(words) == 3, msg
            assert words[1] == 'since', msg
            assert words[0] in ['days', 'seconds'], msg
            self.time_unit = words[0]
            self.time_scalar = self.scalars[self.time_unit]
            base_date_srt = words[2]
            numbers = len(base_date_srt.split('-'))
            if numbers == 3:
                # assume utc
                self.start_time = datetime.datetime.strptime(base_date_srt, '%Y-%m-%d').replace(tzinfo=utc_tz)
            else:
                raise Exception(msg)

            self.ntimesteps = len(time_var)
            dt_arr = np.diff(time_var[:])
            assert np.allclose(dt_arr, dt_arr[0]), 'Time step is not constant. {:}'.format(self.filename)
            self.timestep = dt_arr[0]*self.time_scalar

            self.start_epoch = datetime_to_epoch(self.start_time)
        # HACK these are daily forecast files, limit time steps to one day
        self.ntimesteps = 24

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.start_time + datetime.timedelta(seconds=self.ntimesteps*self.timestep)

    def find_time_stamp(self, t, previous=False):
        t_epoch = datetime_to_epoch(t) if isinstance(t, datetime.datetime) else t
        round_op = np.floor if previous else np.ceil
        itime = int(round_op((t_epoch - self.start_epoch)/self.timestep))
        if itime < 0:
            raise IndexError('Requested time out of bounds {:} < {:}'.format(t_epoch, self.start_epoch))
        if itime > self.ntimesteps - 1:
            raise IndexError('Requested time out of bounds {:} > {:}'.format(t_epoch, datetime_to_epoch(self.get_end_time())))
        return itime


class NetCDFTimeSearch(object):
    """
    Finds a certain time stamp in a collection of netCDF files.
    """
    def __init__(self, file_pattern, init_time, netcdf_class):
        all_files = glob.glob(file_pattern)
        assert len(all_files) > 0, 'No files found: {:}'.format(file_pattern)

        dates = []
        for fn in all_files:
            nc = netcdf_class(fn)
            dates.append(nc.get_start_time())
        sort_ix = np.argsort(dates)
        print sort_ix
        self.files = np.array(all_files)[sort_ix]
        self.start_datetime = np.array(dates)[sort_ix]
        self.start_times = [(s - init_time).total_seconds() for s in self.start_datetime]
        self.start_times = np.array(self.start_times)
        print self.start_datetime
        print self.start_times

        # glob all files, make a list with start times
        # convert start times to simulation start times
        # store time step, ensure time step is the same in all files
        pass

    def get_time_stamp(self, t, previous=False):
        """
        Returns the name of netCDF file that contains the next time step
        """

        return fn, itime, time

import datetime
mesh2d = Mesh('mesh_cre-plume002.msh')
p1 = FunctionSpace(mesh2d, 'CG', 1)
m2_field = Function(p1, name='elevation')

wsf = WindStressForcing(m2_field, datetime.datetime(2016, 5, 1))
#tbnd.set_tidal_field(0.0)
