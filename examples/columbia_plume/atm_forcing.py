r"""
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
from firedrake import *
import numpy as np
import scipy.interpolate
import thetis.timezone as timezone
import thetis.interpolation as interpolation
import thetis.coordsys as coordsys
import datetime
import netCDF4

rho_air = 1.22  # kg/m3

COORDSYS = coordsys.UTM_ZONE10

def to_latlon(x, y, positive_lon=False):
    lon, lat = coordsys.convert_coords(COORDSYS,
                                       coordsys.LL_WGS84, x, y)
    if positive_lon and lon < 0.0:
        lon += 360.
    return lat, lon


def compute_wind_stress(wind_u, wind_v):
    """
    Wind stress formulation by Large and Pond (1981).

    Large and Pond (1981), J. Phys. Oceanog., 11, 324-336.
    """
    wind_mag = np.hypot(wind_u, wind_v)
    CD_LOW = 1.2e-3
    C_D = np.ones_like(wind_u)*CD_LOW
    high_wind = wind_mag > 11.0
    C_D[high_wind] = 1.0e-3*(0.49 + 0.065*wind_mag[high_wind])
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


class WRFNetCDFTime(interpolation.NetCDFTimeParser):
    """
    Custom class to handle WRF atmospheric model forecast files
    """
    def __init__(self, filename):
        super(WRFNetCDFTime, self).__init__(filename, time_variable_name='time')
        # NOTE these are daily forecast files, limit time steps to one day
        self.time_array = self.time_array[:24]
        self.start_time = timezone.epoch_to_datetime(float(self.time_array[0]))
        self.end_time = timezone.epoch_to_datetime(float(self.time_array[-1]))


class WRFInterpolator(object):
    """
    Interpolates WRF atmospheric model data on 2D fields
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, ncfile_pattern, init_date):
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field

        # construct interpolators
        self.grid_interpolator = interpolation.NetCDFLatLonInterpolator2d(self.function_space, to_latlon)
        self.reader = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, ['uwind', 'vwind', 'prmsl'])
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile_pattern, init_date, WRFNetCDFTime)
        self.time_interpolator = interpolation.LinearTimeInterpolator(self.timesearch_obj, self.reader)
        lon = self.grid_interpolator.mesh_lonlat[:, 0]
        lat = self.grid_interpolator.mesh_lonlat[:, 1]
        self.vect_rotator = coordsys.VectorCoordSysRotation(coordsys.LL_WGS84, COORDSYS, lon, lat)

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        lon_wind, lat_wind, prmsl = self.time_interpolator(time)
        u_wind, v_wind = self.vect_rotator(lon_wind, lat_wind)
        u_stress, v_stress = compute_wind_stress(u_wind, v_wind)
        self.wind_stress_field.dat.data_with_halos[:, 0] = u_stress
        self.wind_stress_field.dat.data_with_halos[:, 1] = v_stress
        self.atm_pressure_field.dat.data_with_halos[:] = prmsl


def test():
    """
    Tests atmospheric model data interpolation.

    .. note::
        The following files must be present
        forcings/atm/wrf/wrf_air.2015_05_16.nc
        forcings/atm/wrf/wrf_air.2015_05_17.nc
    """
    mesh2d = Mesh('mesh_cre-plume_02.msh')
    comm = mesh2d.comm
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    p1v = VectorFunctionSpace(mesh2d, 'CG', 1)
    windstress_2d = Function(p1v, name='wind stress')
    atmpressure_2d = Function(p1, name='atm pressure')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)
    pattern = 'forcings/atm/wrf/wrf_air.2015_*_*.nc'

    wrf = WRFInterpolator(p1, windstress_2d, atmpressure_2d, pattern,
                          init_date)

    # create a naive interpolation for first file
    xy = SpatialCoordinate(p1.mesh())
    fsx = Function(p1).interpolate(xy[0]).dat.data_with_halos
    fsy = Function(p1).interpolate(xy[1]).dat.data_with_halos

    mesh_lonlat = []
    for node in range(len(fsx)):
        lat, lon = to_latlon(fsx[node], fsy[node])
        mesh_lonlat.append((lon, lat))
    mesh_lonlat = np.array(mesh_lonlat)

    ncfile = netCDF4.Dataset('forcings/atm/wrf/wrf_air.2015_05_16.nc')
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
    vrot = coordsys.VectorCoordSysRotation(coordsys.LL_WGS84, COORDSYS, mesh_lonlat[:, 0], mesh_lonlat[:, 1])
    uwind, vwind = vrot(uwind, vwind)
    u_stress, v_stress = compute_wind_stress(uwind, vwind)

    # compare
    wrf.set_fields((itime - 8)*3600.)  # NOTE timezone offset
    assert np.allclose(pres, atmpressure_2d.dat.data_with_halos)
    assert np.allclose(u_stress, windstress_2d.dat.data_with_halos[:, 0])

    # write fields to disk for visualization
    pres_fn = 'tmp/atm_pressure.pvd'
    wind_fn = 'tmp/wind_stress.pvd'
    print('Saving output to {:} {:}'.format(pres_fn, wind_fn))
    out_pres = File(pres_fn)
    out_wind = File(wind_fn)
    hours = 24*1.5
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
