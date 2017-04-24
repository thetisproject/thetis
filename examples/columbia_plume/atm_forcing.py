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
import scipy.interpolate
import glob
import os
from timezone import *
from interpolation import *

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




class WindStressForcing(object):
    def __init__(self, wind_stress_field, init_date):

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


class WRFNetCDFTime(NetCDFTime):
    """
    Custom class to handle WRF atmospheric model forecast files
    """
    def __init__(self, filename):
        super(WRFNetCDFTime, self).__init__(filename)
        # NOTE these are daily forecast files, limit time steps to one day
        self.ntimesteps = 24


def test():
    #timezone = FixedTimeZone(-8, 'PST')
    timezone = pytz.timezone('UTC')
    init_date = datetime.datetime(2016, 5 , 1, tzinfo=timezone)

    pattern = 'forcings/atm/wrf/wrf_air.2016_*_*.nc'
    nts = NetCDFTimeSearch(pattern, init_date, WRFNetCDFTime)

    print nts.find(3600.0, previous=True)
    print nts.find(3600.0, previous=False)

    hours = 24*3
    granule = 4
    simtime = np.arange(granule*hours)*3600./granule
    for t in simtime:
        print t, nts.simulation_time_to_datetime(t)
        print nts.find(t, previous=True)
        print nts.find(t, previous=False)

    #import datetime
    #mesh2d = Mesh('mesh_cre-plume002.msh')
    #p1 = FunctionSpace(mesh2d, 'CG', 1)
    #m2_field = Function(p1, name='elevation')

    #wsf = WindStressForcing(m2_field, datetime.datetime(2016, 5, 1))


if __name__ == '__main__':
    test()
