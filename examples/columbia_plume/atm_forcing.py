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
    CD_LOW = 1.2e-3
    C_D = np.ones_like(wind_u)*CD_LOW
    high_wind = wind_mag > 11.0
    C_D[high_wind] = 10e-3*(0.49 + 0.065*wind_mag[high_wind])
    #if wind_mag < 11.0:
        #C_D = 1.2e-3
    #else:
        #C_D = 10e-3*(0.49 + 0.065*wind_mag)
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


class WRFNetCDFTime(NetCDFTime):
    """
    Custom class to handle WRF atmospheric model forecast files
    """
    def __init__(self, filename):
        super(WRFNetCDFTime, self).__init__(filename)
        # NOTE these are daily forecast files, limit time steps to one day
        self.ntimesteps = 24


class WRFInterpolator(object):
    """
    Interpolates WRF atmospheric model data on 2D fields
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, ncfile_pattern, init_date, to_latlon):
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field

        # construct interpolators
        self.grid_interpolator = NetCDFLatLonInterpolator2d(self.function_space, to_latlon)
        self.reader = NetCDFSpatialInterpolator(self.grid_interpolator, ['uwind', 'vwind', 'prmsl'])
        self.timesearch_obj = NetCDFTimeSearch(ncfile_pattern, init_date, WRFNetCDFTime)
        self.interpolator = LinearTimeInterpolator(self.timesearch_obj, self.reader)

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        uwind, vwind, prmsl = self.interpolator(time)
        u_stress, v_stress = compute_wind_stress(uwind, vwind)
        self.wind_stress_field.dat.data_with_halos[:, 0] = u_stress
        self.wind_stress_field.dat.data_with_halos[:, 1] = v_stress
        self.atm_pressure_field.dat.data_with_halos[:] = prmsl


def test():
    mesh2d = Mesh('mesh_cre-plume002.msh')
    comm = mesh2d.comm
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    p1v = VectorFunctionSpace(mesh2d, 'CG', 1)
    windstress_2d = Function(p1v, name='wind stress')
    atmpressure_2d = Function(p1, name='atm pressure')

    timezone = FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2016, 5 , 1, tzinfo=timezone)
    pattern = 'forcings/atm/wrf/wrf_air.2016_*_*.nc'

    wrf = WRFInterpolator(p1, windstress_2d, atmpressure_2d, pattern, init_date, to_latlon)

    # create a naive interpolation for first file
    xy = SpatialCoordinate(p1.mesh())
    fsx = Function(p1).interpolate(xy[0]).dat.data_with_halos
    fsy = Function(p1).interpolate(xy[1]).dat.data_with_halos

    mesh_lonlat = []
    for node in range(len(fsx)):
        lat, lon = to_latlon(fsx[node], fsy[node])
        mesh_lonlat.append((lon, lat))
    mesh_lonlat = np.array(mesh_lonlat)

    ncfile = netCDF4.Dataset('forcings/atm/wrf/wrf_air.2016_05_01.nc')
    itime = 10
    grid_lat = ncfile['lat'][:].ravel()
    grid_lon = ncfile['lon'][:].ravel()
    grid_lonlat = np.array((grid_lon, grid_lat)).T
    grid_pres = ncfile['prmsl'][itime, :, :].ravel()
    pres = scipy.interpolate.griddata(grid_lonlat, grid_pres, mesh_lonlat, method='linear')
    grid_uwind = ncfile['uwind'][itime, :, :].ravel()
    uwind = scipy.interpolate.griddata(grid_lonlat, grid_uwind, mesh_lonlat, method='linear')
    grid_vwind = ncfile['vwind'][itime, :, :].ravel()
    vwind = scipy.interpolate.griddata(grid_lonlat, grid_vwind, mesh_lonlat, method='linear')
    u_stress, v_stress = compute_wind_stress(uwind, vwind)

    # compare
    wrf.set_fields((itime - 8)*3600.)  # NOTE timezone offset
    assert np.allclose(pres, atmpressure_2d.dat.data_with_halos)
    assert np.allclose(u_stress, windstress_2d.dat.data_with_halos[:, 0])

    # write fields to disk for visualization
    out_pres = File('tmp/atm_pressure.pvd')
    out_wind = File('tmp/wind_stress.pvd')
    hours = 24*3
    granule = 4
    simtime = np.arange(granule*hours)*3600./granule
    i = 0
    for t in simtime:
        wrf.set_fields(t)
        norm_atm = norm(atmpressure_2d)
        norm_wind = norm(windstress_2d)
        if comm.rank == 0:
            print('{:} {:} {:} {:}'.format(i, t, norm_atm, norm_wind))
        out_pres.write(atmpressure_2d)
        out_wind.write(windstress_2d)
        i += 1


if __name__ == '__main__':
    test()