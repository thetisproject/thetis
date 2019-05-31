"""
Routines for interpolating forcing fields for the 3D solver.
"""
from firedrake import *
import numpy as np
import scipy.interpolate
import thetis.timezone as timezone
import thetis.interpolation as interpolation
import thetis.coordsys as coordsys
from .log import *
import datetime
import netCDF4
import thetis.physical_constants as physical_constants


def compute_wind_stress(wind_u, wind_v, method='LargePond1981'):
    """
    Compute wind stress from atmospheric 10 m wind.

    wind stress is defined as

    .. math:
        tau_w = C_D \rho_{air} \|U_{10}\| U_{10}

    where :math:`C_D` is the drag coefficient, :math:`\rho_{air}` is the density of
    air, and :math:`U_{10}` is wind speed 10 m above the sea surface.

    In practice `C_D` depends on the wind speed.

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

    :arg wind_u, wind_v: Wind u and v components as numpy arrays
    :kwarg method: Choose the stress formulation. Currently supports:
        'LargePond1981' (default) or 'SmithBanke1975'.
    :returns: (tau_x, tau_y) wind stress x and y components as numpy arrays
    """
    rho_air = float(physical_constants['rho_air'])
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
    A TimeParser class for reading WRF/NAM atmospheric forecast files.
    """
    def __init__(self, filename, max_duration=24.*3600., verbose=False):
        """
        :arg filename:
        :kwarg max_duration: Time span to read from each file (in secords,
            default one day). Forecast files are usually daily files that
            contain forecast for > 1 days.
        :kwarg bool verbose: Se True to print debug information.
        """
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
    Interpolates WRF/NAM atmospheric model data on 2D fields.
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, to_latlon,
                 ncfile_pattern, init_date, target_coordsys, verbose=False):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg wind_stress_field: A 2D vector :class:`Function` where the output
            wind stress will be stored.
        :arg atm_pressure_field: A 2D scalar :class:`Function` where the output
            atmospheric pressure will be stored.
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        :arg ncfile_pattern: A file name pattern for reading the atmospheric
            model output files. E.g. 'forcings/nam_air.local.2006_*.nc'
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :arg target_coordsys: coordinate system in which the model grid is
            defined. This is used to rotate vectors to local coordinates.
        :kwarg bool verbose: Se True to print debug information.
        """
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
        self.vect_rotator = coordsys.VectorCoordSysRotation(
            coordsys.LL_WGS84, target_coordsys, lon, lat)

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time.

        Performs interpolation and updates the output wind stress and
        atmospheric pressure fields in place.

        :arg float time: Thetis simulation time in seconds.
        """
        lon_wind, lat_wind, prmsl = self.time_interpolator(time)
        u_wind, v_wind = self.vect_rotator(lon_wind, lat_wind)
        u_stress, v_stress = compute_wind_stress(u_wind, v_wind)
        self.wind_stress_field.dat.data_with_halos[:, 0] = u_stress
        self.wind_stress_field.dat.data_with_halos[:, 1] = v_stress
        self.atm_pressure_field.dat.data_with_halos[:] = prmsl
