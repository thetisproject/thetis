r"""
Wind stress from WRF/NAM atmospheric model

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
from thetis.log import *
import datetime
import netCDF4

rho_air = 1.22  # kg/m3

COORDSYS = coordsys.UTM_ZONE10


def to_latlon(x, y, positive_lon=False):
    lon, lat = coordsys.convert_coords(COORDSYS,
                                       coordsys.LL_WGS84, x, y)
    if positive_lon:
        if isinstance(lon, np.ndarray):
            ix = lon < 0.0
            lon[ix] += 360.
        else:  # assume float
            lon += 360.
    return lat, lon


def compute_wind_stress(wind_u, wind_v, method='LargePond1981'):
    """
    Compute wind stress from atmospheric 10 m wind.

    Two formulation are currently implemented:

    - "LargePond1981":
        Wind stress formulation by [1]
    - "SmithBanke1975":
        Wind stress formulation by [2]

    [1] Large and Pond (1981). Open Ocean Momentum Flux Measurements in
        Moderate to Strong Winds. Journal of Physical Oceanography,
        11(3):324-336.
        https://doi.org/10.1175/1520-0485(1981)011%3C0324:OOMFMI%3E2.0.CO;2
    [2] Smith and Banke (1975). Variation of the sea surface drag coefficient with
        wind speed. Q J R Meteorol Soc., 101(429):665-673.
        https://doi.org/10.1002/qj.49710142920
    """
    wind_mag = np.hypot(wind_u, wind_v)
    if method == 'LargePond1981':
        CD_LOW = 1.2e-3
        C_D = np.ones_like(wind_u)*CD_LOW
        high_wind = wind_mag > 11.0
        C_D[high_wind] = 1.0e-3*(0.49 + 0.065*wind_mag[high_wind])
    elif method == 'SmithBanke1975':
        C_D = (0.63 + 0.066 * wind_mag)/1000.
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


class ATMNetCDFTime(interpolation.NetCDFTimeParser):
    """
    Custom class to handle WRF/NAM atmospheric model forecast files
    """
    def __init__(self, filename, max_duration=24.*3600., verbose=False):
        super(ATMNetCDFTime, self).__init__(filename, time_variable_name='time')
        # NOTE these are daily forecast files, limit time steps to one day
        self.start_time = timezone.epoch_to_datetime(float(self.time_array[0]))
        self.end_time_raw = timezone.epoch_to_datetime(float(self.time_array[-1]))
        self.time_step = np.mean(np.diff(self.time_array))
        self.max_steps = int(max_duration / self.time_step)
        self.time_array = self.time_array[:self.max_steps]
        self.end_time = timezone.epoch_to_datetime(float(self.time_array[-1]))
        if verbose:
            print_output('Parsed file {:}'.format(filename))
            print_output('  Raw time span: {:} -> {:}'.format(self.start_time, self.end_time_raw))
            print_output('  Time step: {:} h'.format(self.time_step/3600.))
            print_output('  Restricting duration to {:} h -> keeping {:} steps'.format(max_duration/3600., self.max_steps))
            print_output('  New time span: {:} -> {:}'.format(self.start_time, self.end_time))


class ATMInterpolator(object):
    """
    Interpolates WRF/NAM atmospheric model data on 2D fields
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, ncfile_pattern, init_date, verbose=False):
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field

        # construct interpolators
        self.grid_interpolator = interpolation.NetCDFLatLonInterpolator2d(self.function_space, to_latlon)
        self.reader = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, ['uwind', 'vwind', 'prmsl'])
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile_pattern, init_date, ATMNetCDFTime, verbose=verbose)
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

        forcings/atm/nam/nam_air.local.2006_04_19.nc
        forcings/atm/nam/nam_air.local.2006_04_20.nc
    """
    mesh2d = Mesh('mesh_cre-plume_02.msh')
    comm = mesh2d.comm
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    p1v = VectorFunctionSpace(mesh2d, 'CG', 1)
    windstress_2d = Function(p1v, name='wind stress')
    atmpressure_2d = Function(p1, name='atm pressure')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')

    # WRF
    # init_date = datetime.datetime(2015, 5, 16, tzinfo=sim_tz)
    # pattern = 'forcings/atm/wrf/wrf_air.2015_*_*.nc'
    # atm_time_step = 3600.  # for verification only
    # test_atm_file = 'forcings/atm/wrf/wrf_air.2015_05_16.nc'

    # NAM
    init_date = datetime.datetime(2006, 4, 19, tzinfo=sim_tz)
    pattern = 'forcings/atm/nam/nam_air.local.2006_*_*.nc'
    atm_time_step = 3*3600.
    test_atm_file = 'forcings/atm/nam/nam_air.local.2006_04_19.nc'

    atm_interp = ATMInterpolator(p1, windstress_2d, atmpressure_2d,
                                 pattern, init_date, verbose=True)

    # create a naive interpolation for first file
    xy = SpatialCoordinate(p1.mesh())
    fsx = Function(p1).interpolate(xy[0]).dat.data_with_halos
    fsy = Function(p1).interpolate(xy[1]).dat.data_with_halos

    mesh_lonlat = []
    for node in range(len(fsx)):
        lat, lon = to_latlon(fsx[node], fsy[node])
        mesh_lonlat.append((lon, lat))
    mesh_lonlat = np.array(mesh_lonlat)

    ncfile = netCDF4.Dataset(test_atm_file)
    itime = 6
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
    atm_interp.set_fields(itime*atm_time_step - 8*3600.)  # NOTE timezone offset
    assert np.allclose(pres, atmpressure_2d.dat.data_with_halos)
    assert np.allclose(u_stress, windstress_2d.dat.data_with_halos[:, 0])

    # write fields to disk for visualization
    pres_fn = 'tmp/AtmPressure2d.pvd'
    wind_fn = 'tmp/WindStress2d.pvd'
    print('Saving output to {:} {:}'.format(pres_fn, wind_fn))
    out_pres = File(pres_fn)
    out_wind = File(wind_fn)
    hours = 24*1.5
    granule = 4
    simtime = np.arange(granule*hours)*3600./granule
    i = 0
    for t in simtime:
        atm_interp.set_fields(t)
        norm_atm = norm(atmpressure_2d)
        norm_wind = norm(windstress_2d)
        if comm.rank == 0:
            print('{:} {:} {:} {:}'.format(i, t, norm_atm, norm_wind))
        out_pres.write(atmpressure_2d)
        out_wind.write(windstress_2d)
        i += 1


if __name__ == '__main__':
    test()
