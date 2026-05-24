"""
Routines for interpolating forcing fields for the 3D solver.
"""
from firedrake import *
from firedrake.petsc import PETSc
import scipy.spatial.qhull as qhull
import thetis.timezone as timezone
import thetis.interpolation as interpolation
from .log import *
import netCDF4
import thetis.physical_constants as physical_constants
import uptide
import uptide.tidal_netcdf
from abc import ABC, abstractmethod, abstractproperty
import os
import numpy


def compute_wind_stress(wind_u, wind_v, method='LargeYeager2009'):
    r"""
    Compute wind stress from atmospheric 10 m wind.

    wind stress is defined as

    .. math:
        tau_w = C_D \rho_{air} \|U_{10}\| U_{10}

    where :math:`C_D` is the drag coefficient, :math:`\rho_{air}` is the
    density of air, and :math:`U_{10}` is wind speed 10 m above the sea
    surface.

    In practice `C_D` depends on the wind speed.

    Two formulation are currently implemented:

    - "LargePond1981":
        Wind stress formulation by [1]
    - "SmithBanke1975":
        Wind stress formulation by [2]
    - "LargeYeager2009":
        Wind stress formulation by [3]

    [1] Large and Pond (1981). Open Ocean Momentum Flux Measurements in
        Moderate to Strong Winds. Journal of Physical Oceanography,
        11(3):324-336.
        https://doi.org/10.1175/1520-0485(1981)011%3C0324:OOMFMI%3E2.0.CO;2
    [2] Smith and Banke (1975). Variation of the sea surface drag coefficient
        with wind speed. Q J R Meteorol Soc., 101(429):665-673.
        https://doi.org/10.1002/qj.49710142920
    [3] Large and Yeager (2009). The global climatology of an interannually
        varying air–sea flux data set. Climate Dynamics,
        33:341–364. http://dx.doi.org/10.1007/s00382-008-0441-3

    :arg wind_u, wind_v: Wind u and v components as numpy arrays
    :kwarg method: Choose the stress formulation. Currently supports:
        'LargeYeager2009' (default), 'LargePond1981', or 'SmithBanke1975'.
    :returns: (tau_x, tau_y) wind stress x and y components as numpy arrays
    """
    rho_air = float(physical_constants['rho_air'])
    wind_mag = numpy.hypot(wind_u, wind_v)
    if method == 'LargePond1981':
        CD_LOW = 1.2e-3
        C_D = numpy.ones_like(wind_u)*CD_LOW
        high_wind = wind_mag > 11.0
        C_D[high_wind] = 1.0e-3*(0.49 + 0.065*wind_mag[high_wind])
    elif method == 'SmithBanke1975':
        C_D = (0.63 + 0.066 * wind_mag)/1000.
    elif method == 'LargeYeager2009':
        # NOTE wind velocity should be shifted to 10 m neutral equivalent
        # but it requires air temperature, humidity and iteration, see [3]
        high_wind = wind_mag > 33.0
        eps = 1e-3
        C_D = 1.e-3 * (2.7 / (wind_mag + eps) + 0.142
                       + wind_mag / 13.09 - 3.14807e-10*wind_mag**6)
        C_D[high_wind] = 2.34e-3
    tau = C_D*rho_air*wind_mag
    tau_x = tau*wind_u
    tau_y = tau*wind_v
    return tau_x, tau_y


class AtmosphericForcingInterpolator(object):
    """
    Generic class for atmospheric forcing (pressure and wind stress)
    """
    @PETSc.Log.EventDecorator("thetis.AtmosForcingInterpolator.__init__")
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, coord_system,
                 ncfile, init_date,
                 vect_rotator=None,
                 east_wind_var_name='u10', north_wind_var_name='v10',
                 pressure_var_name='msl', time_var_name='valid_time',
                 pressure_units="pa", temporal_interpolator=interpolation.NetCDFTimeParser,
                 fill_mode=None, fill_value=numpy.nan,
                 verbose=False):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg wind_stress_field: A 2D vector :class:`Function` where the output
            wind stress will be stored.
        :arg atm_pressure_field: A 2D scalar :class:`Function` where the output
            atmospheric pressure will be stored.
        :arg coord_system: :class:`CoordinateSystem` object
        :arg ncfile: A file name or file pattern for reading the atmospheric data
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :arg temporal_interpolator: A :class:`TimeParser` object to handle temporal interpolation.
        :kwarg vect_rotator: function that rotates vectors from ENU coordinates
            to target function space (optional).
        :kwarg east_wind_var_name, north_wind_var_name, pressure_var_name, time_var_name:
            wind components, pressure, and time field names in netCDF file.
        :kwarg fill_mode: Determines how points outside the source grid will be
            treated. If 'nearest', value of the nearest source point will be
            used. Otherwise a constant fill value will be used (default).
        :kwarg float fill_value: Set the fill value (default: NaN)
        :kwarg bool verbose: Set True to print debug information.
        """
        self.function_space = function_space
        self.wind_stress_field = wind_stress_field
        self.atm_pressure_field = atm_pressure_field
        self.pressure_units = pressure_units

        # construct interpolators
        self.grid_interpolator = interpolation.NetCDFLatLonInterpolator2d(
            self.function_space, coord_system, fill_mode=fill_mode,
            fill_value=fill_value)
        var_list = [east_wind_var_name, north_wind_var_name, pressure_var_name]
        self.reader = interpolation.NetCDFSpatialInterpolator(
            self.grid_interpolator, var_list)
        self.timesearch_obj = \
            interpolation.NetCDFTimeSearch(ncfile,
                                           init_date,
                                           temporal_interpolator,
                                           verbose=verbose,
                                           time_variable_name=time_var_name)
        self.time_interpolator = \
            interpolation.LinearTimeInterpolator(self.timesearch_obj,
                                                 self.reader)
        lon = self.grid_interpolator.mesh_lonlat[:, 0]
        lat = self.grid_interpolator.mesh_lonlat[:, 1]
        if vect_rotator is None:
            self.vect_rotator = coord_system.get_vector_rotator(lon, lat)
        else:
            self.vect_rotator = vect_rotator

    @PETSc.Log.EventDecorator("thetis.AtmosForcingInterpolator.set_fields")
    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time.
        Performs interpolation and updates the output wind stress and
        atmospheric pressure fields in place.
        :arg float time: Thetis simulation time in seconds.
        """
        east_wind, north_wind, prmsl = self.time_interpolator(time)
        east_strs, north_strs = compute_wind_stress(east_wind, north_wind)
        if self.wind_stress_field.ufl_shape == (3,):
            u_strs, v_strs, z_strs = self.vect_rotator(east_strs, north_strs)
            self.wind_stress_field.dat.data_with_halos[:, 0] = u_strs
            self.wind_stress_field.dat.data_with_halos[:, 1] = v_strs
            self.wind_stress_field.dat.data_with_halos[:, 2] = z_strs
        else:
            u_strs, v_strs = self.vect_rotator(east_strs, north_strs)
            self.wind_stress_field.dat.data_with_halos[:, 0] = u_strs
            self.wind_stress_field.dat.data_with_halos[:, 1] = v_strs
        if self.pressure_units == "pa":
            self.atm_pressure_field.dat.data_with_halos[:] = prmsl
        elif self.pressure_units == "hpa":
            self.atm_pressure_field.dat.data_with_halos[:] = prmsl * 100  # hPa to Pa


class ERA5Interpolator(AtmosphericForcingInterpolator):
    """
    Class to use ERA5 forcing. Note most defaults are for this kind of forcing
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, coord_system,
                 ncfile, init_date,
                 vect_rotator=None,
                 fill_mode=None, fill_value=numpy.nan,
                 verbose=False):

        super().__init__(function_space, wind_stress_field,
                         atm_pressure_field, coord_system,
                         ncfile, init_date,
                         vect_rotator,
                         fill_mode=fill_mode, fill_value=fill_value,
                         verbose=verbose)


class TCHazardsInterpolator(AtmosphericForcingInterpolator):
    """
    Class to use atmospheric forcing from the TCHazaRds package
    https://github.com/AusClimateService/TCHazaRds
    O’Grady, J. et al. (2024) ‘Evaluation of parametric tropical cyclone surface winds
    over the Eastern Australian region’,
    Monthly weather review, 152(1), pp. 345–361.
    Available at: https://doi.org/10.1175/mwr-d-23-0063.1.
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, coord_system,
                 ncfile, init_date,
                 vect_rotator=None,
                 fill_mode=None, fill_value=numpy.nan,
                 verbose=False):

        super().__init__(function_space, wind_stress_field,
                         atm_pressure_field, coord_system,
                         ncfile, init_date,
                         vect_rotator,
                         east_wind_var_name="Uw", north_wind_var_name="Vw",
                         pressure_var_name="Pr", time_var_name="time",
                         pressure_units="hpa",
                         fill_mode=fill_mode, fill_value=fill_value,
                         verbose=verbose)


class ATMNetCDFTime(interpolation.NetCDFTimeParser):
    """
    A TimeParser class for reading atmosphere model output files.
    """
    @PETSc.Log.EventDecorator("thetis.ATMNetCDFTime.__init__")
    def __init__(self, filename, max_duration=None, verbose=False, time_variable_name='time'):
        """
        :arg filename:
        :kwarg max_duration: Time span to read from each file (in seconds).
            E.g. forecast files can consist of daily files with > 1 day of
            data. In this case max_duration should be set to 24 h. If None,
            all time steps are loaded. Default: None.
        :kwarg bool verbose: Se True to print debug information.
        """
        super().__init__(filename, time_variable_name=time_variable_name)
        # NOTE these are daily forecast files, limit time steps to one day
        self.start_time = timezone.epoch_to_datetime(float(self.time_array[0]))
        self.end_time_raw = timezone.epoch_to_datetime(float(self.time_array[-1]))
        self.time_step = numpy.mean(numpy.diff(self.time_array))
        if max_duration is not None:
            self.max_steps = int(max_duration / self.time_step)
        else:
            self.max_steps = self.nb_steps
        self.time_array = self.time_array[:self.max_steps]
        self.end_time = timezone.epoch_to_datetime(float(self.time_array[-1]))
        if verbose:
            print_output('Parsed file %s', filename)
            print_output('  Time span: %s -> %s', self.start_time, self.end_time_raw)
            print_output('  Time step: %.1f h', self.time_step/3600.)
            if max_duration is not None:
                print_output('  Restricting duration to %.1f h -> keeping %d steps',
                             max_duration/3600.,
                             self.max_steps)
                print_output('  New time span: %s -> %s', self.start_time, self.end_time)


class ATMInterpolator(AtmosphericForcingInterpolator):
    """
    Interpolates WRF/NAM atmospheric model data on 2D fields.
    """
    def __init__(self, function_space, wind_stress_field,
                 atm_pressure_field, coord_system,
                 ncfile, init_date,
                 vect_rotator=None,
                 fill_mode=None, fill_value=numpy.nan,
                 verbose=False):

        super().__init__(function_space, wind_stress_field,
                         atm_pressure_field, coord_system,
                         ncfile, init_date,
                         vect_rotator,
                         east_wind_var_name="uwind", north_wind_var_name="vwind",
                         pressure_var_name="prmsl",
                         temporal_interpolator=ATMNetCDFTime,
                         fill_mode=fill_mode, fill_value=fill_value,
                         verbose=verbose)


class SpatialInterpolatorNCOMBase(interpolation.SpatialInterpolator):
    """
    Base class for 2D and 3D NCOM spatial interpolators.
    """
    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOMBase.__init__")
    def __init__(self, function_space, coord_system, grid_path):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg coord_system: :class:`CoordinateSystem` object
        :arg grid_path: File path where the NCOM model grid files
            ('model_lat.nc', 'model_lon.nc', 'model_zm.nc') are located.
        """
        self.function_space = function_space
        self.grid_path = grid_path
        self._initialized = False

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOMBase._create_2d_mapping")
    def _create_2d_mapping(self, ncfile):
        """
        Create map for 2D nodes.
        """
        # read source lat lon grid
        lat_full = self._get_forcing_grid('model_lat.nc', 'Lat')
        lon_full = self._get_forcing_grid('model_lon.nc', 'Long')
        x_ind = ncfile['X_Index'][:].astype(int)
        y_ind = ncfile['Y_Index'][:].astype(int)
        lon = lon_full[y_ind, :][:, x_ind]
        lat = lat_full[y_ind, :][:, x_ind]

        # find where data values are not defined
        varkey = None
        for k in ncfile.variables.keys():
            if k not in ['X_Index', 'Y_Index', 'level']:
                varkey = k
                break
        assert varkey is not None, 'Could not find variable in file'
        vals = ncfile[varkey][:]  # shape (nz, lat, lon) or (lat, lon)
        is3d = len(vals.shape) == 3
        land_mask = numpy.all(vals.mask, axis=0) if is3d else vals.mask

        # build 2d mask
        mask_good_values = ~land_mask
        # neighborhood mask with bounding box
        mask_cover = numpy.zeros_like(mask_good_values)
        buffer = 0.2
        lat_min = self.latlonz_array[:, 0].min() - buffer
        lat_max = self.latlonz_array[:, 0].max() + buffer
        lon_min = self.latlonz_array[:, 1].min() - buffer
        lon_max = self.latlonz_array[:, 1].max() + buffer
        mask_cover[(lat >= lat_min)
                   * (lat <= lat_max)
                   * (lon >= lon_min)
                   * (lon <= lon_max)] = True
        mask_cover *= mask_good_values
        # include nearest valid neighbors
        # needed for nearest neighbor filling
        from scipy.spatial import cKDTree
        good_lat = lat[mask_good_values]
        good_lon = lon[mask_good_values]
        ll = numpy.vstack([good_lat.ravel(), good_lon.ravel()]).T
        dist, ix = cKDTree(ll).query(self.latlonz_array[:, :2])
        ix = numpy.unique(ix)
        ix = numpy.nonzero(mask_good_values.ravel())[0][ix]
        a, b = numpy.unravel_index(ix, lat.shape)
        mask_nn = numpy.zeros_like(mask_good_values)
        mask_nn[a, b] = True
        # final mask
        mask = mask_cover + mask_nn

        self.nodes = numpy.nonzero(mask.ravel())[0]
        self.ind_lat, self.ind_lon = numpy.unravel_index(self.nodes, lat.shape)

        lat_subset = lat[self.ind_lat, self.ind_lon]
        lon_subset = lon[self.ind_lat, self.ind_lon]

        assert len(lat_subset) > 0, 'rank {:} has no source lat points'
        assert len(lon_subset) > 0, 'rank {:} has no source lon points'

        return lon_subset, lat_subset, x_ind, y_ind, vals

    def _get_forcing_grid(self, filename, varname):
        """
        Helper function to load NCOM grid files.
        """
        v = None
        with netCDF4.Dataset(os.path.join(self.grid_path, filename), 'r') as ncfile:
            v = ncfile[varname][:]
        return v


class SpatialInterpolatorNCOM3d(SpatialInterpolatorNCOMBase):
    """
    Spatial interpolator class for interpolating NCOM ocean model 3D fields.
    """
    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM3d.__init__")
    def __init__(self, function_space, coord_system, grid_path):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg coord_system: :class:`CoordinateSystem` object
        :arg grid_path: File path where the NCOM model grid files
            ('model_lat.nc', 'model_lon.nc', 'model_zm.nc') are located.
        """
        super().__init__(function_space, coord_system, grid_path)

        # construct local coordinates
        xyz = SpatialCoordinate(self.function_space.mesh())
        tmp_func = self.function_space.get_work_function()
        xyz_array = numpy.zeros((tmp_func.dat.data_with_halos.shape[0], 3))
        for i in range(3):
            tmp_func.interpolate(xyz[i])
            xyz_array[:, i] = tmp_func.dat.data_with_halos[:]
        self.function_space.restore_work_function(tmp_func)

        self.latlonz_array = numpy.zeros_like(xyz_array)
        lon, lat = coord_system.to_lonlat(
            xyz_array[:, 0], xyz_array[:, 1], positive_lon=True
        )
        self.latlonz_array[:, 0] = lat
        self.latlonz_array[:, 1] = lon
        self.latlonz_array[:, 2] = xyz_array[:, 2]

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM3d._create_interpolator")
    def _create_interpolator(self, ncfile):
        """
        Create a compact interpolator by finding the minimal necessary support
        """
        lon_subset, lat_subset, x_ind, y_ind, vals = self._create_2d_mapping(ncfile)

        # find 3d mask where data is not defined
        vals = vals[:, self.ind_lat, self.ind_lon]
        self.good_mask_3d = ~vals.mask

        # construct vertical grid
        zm = self._get_forcing_grid('model_zm.nc', 'zm')
        zm = zm[:, y_ind, :][:, :, x_ind]
        grid_z = zm[:, self.ind_lat, self.ind_lon]  # shape (nz, nlatlon)
        grid_z = grid_z.filled(-5000.)
        # nudge water surface higher for interpolation
        grid_z[0, :] = 1.5
        nz = grid_z.shape[0]

        # data shape is [nz, neta*nxi]
        grid_lat = numpy.tile(lat_subset, (nz, 1))[self.good_mask_3d]
        grid_lon = numpy.tile(lon_subset, (nz, 1))[self.good_mask_3d]
        grid_z = grid_z[self.good_mask_3d]
        if numpy.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if numpy.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        if numpy.ma.isMaskedArray(grid_z):
            grid_z = grid_z.filled(0.0)
        grid_latlonz = numpy.vstack((grid_lat, grid_lon, grid_z)).T

        # building 3D interpolator, this can take a long time (minutes)
        print_output('Constructing 3D GridInterpolator...')
        self.interpolator = interpolation.GridInterpolator(
            grid_latlonz, self.latlonz_array,
            normalize=True, fill_mode='nearest', dont_raise=True
        )
        print_output('done.')
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM3d.interpolate")
    def interpolate(self, nc_filename, variable_list, itime):
        """
        Calls the interpolator object
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                grid_data = ncfile[var][:][:, self.ind_lat, self.ind_lon][self.good_mask_3d]
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class SpatialInterpolatorNCOM2d(SpatialInterpolatorNCOMBase):
    """
    Spatial interpolator class for interpolating NCOM ocean model 2D fields.
    """
    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM2d.__init__")
    def __init__(self, function_space, coord_system, grid_path):
        """
        :arg function_space: Target (scalar) :class:`FunctionSpace` object onto
            which data will be interpolated.
        :arg coord_system: :class:`CoordinateSystem` object
        :arg grid_path: File path where the NCOM model grid files
            ('model_lat.nc', 'model_lon.nc', 'model_zm.nc') are located.
        """
        super().__init__(function_space, coord_system, grid_path)
        # construct local coordinates
        xyz = SpatialCoordinate(self.function_space.mesh())
        tmp_func = self.function_space.get_work_function()
        xy_array = numpy.zeros((tmp_func.dat.data_with_halos.shape[0], 2))
        for i in range(2):
            tmp_func.interpolate(xyz[i])
            xy_array[:, i] = tmp_func.dat.data_with_halos[:]
        self.function_space.restore_work_function(tmp_func)

        self.latlonz_array = numpy.zeros_like(xy_array)
        lon, lat = coord_system.to_lonlat(
            xy_array[:, 0], xy_array[:, 1], positive_lon=True
        )
        self.latlonz_array[:, 0] = lat
        self.latlonz_array[:, 1] = lon

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM2d._create_interpolator")
    def _create_interpolator(self, ncfile):
        """
        Create a compact interpolator by finding the minimal necessary support
        """
        lon_subset, lat_subset, x_ind, y_ind, vals = self._create_2d_mapping(ncfile)

        grid_lat = lat_subset
        grid_lon = lon_subset
        if numpy.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if numpy.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        grid_latlon = numpy.vstack((grid_lat, grid_lon)).T

        # building 3D interpolator, this can take a long time (minutes)
        self.interpolator = interpolation.GridInterpolator(
            grid_latlon, self.latlonz_array,
            normalize=False, fill_mode='nearest', dont_raise=True
        )
        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorNCOM2d.interpolate")
    def interpolate(self, nc_filename, variable_list, itime):
        """
        Calls the interpolator object
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                grid_data = ncfile[var][:][self.ind_lat, self.ind_lon]
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class NCOMInterpolator(object):
    """
    Interpolates NCOM model data on 3D fields.

    .. note::
        The following NCOM output files must be present:
        ./forcings/ncom/model_h.nc
        ./forcings/ncom/model_lat.nc
        ./forcings/ncom/model_ang.nc
        ./forcings/ncom/model_lon.nc
        ./forcings/ncom/model_zm.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/t3d/t3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/u3d/u3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/v3d/v3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/ssh/ssh.glb8_2f_2006042000.nc
    """
    @PETSc.Log.EventDecorator("thetis.NCOMInterpolator.__init__")
    def __init__(self, function_space_2d, function_space_3d, fields, field_names, field_fnstr,
                 coord_system, basedir, file_pattern, init_date, verbose=False):
        """
        :arg function_space_2d: Target (scalar) :class:`FunctionSpace` object onto
            which 2D data will be interpolated.
        :arg function_space_3d: Target (scalar) :class:`FunctionSpace` object onto
            which 3D data will be interpolated.
        :arg fields: list of :class:`Function` objects where data will be
            stored.
        :arg field_names: List of netCDF variable names for the fields. E.g.
            ['Salinity', 'Temperature'].
        :arg field_fnstr: List of variables in netCDF file names. E.g.
            ['s3d', 't3d'].
        :arg coord_system: :class:`CoordinateSystem` object
        :arg basedir: Root dir where NCOM files are stored.
            E.g. '/forcings/ncom'.
        :arg file_pattern: A file name pattern for reading the NCOM output
            files (excluding the basedir). E.g.
            {year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc'.
        :arg init_date: A :class:`datetime` object that indicates the start
            date/time of the Thetis simulation. Must contain time zone. E.g.
            'datetime(2006, 5, 1, tzinfo=pytz.utc)'
        :kwarg bool verbose: Se True to print debug information.
        """
        self.function_space_2d = function_space_2d
        self.function_space_3d = function_space_3d
        for f in fields:
            assert f.function_space() in [self.function_space_2d, self.function_space_3d], 'field \'{:}\' does not belong to given function space.'.format(f.name())
        assert len(fields) == len(field_names)
        assert len(fields) == len(field_fnstr)
        self.field_names = field_names
        self.fields = dict(zip(self.field_names, fields))

        # construct interpolators
        self.grid_interpolator_2d = SpatialInterpolatorNCOM2d(self.function_space_2d, coord_system, basedir)
        self.grid_interpolator_3d = SpatialInterpolatorNCOM3d(self.function_space_3d, coord_system, basedir)
        # each field is in different file
        # construct time search and interp objects separately for each
        self.time_interpolator = {}
        for ncvarname, fnstr in zip(field_names, field_fnstr):
            gi = self.grid_interpolator_2d if fnstr == 'ssh' else self.grid_interpolator_3d
            r = interpolation.NetCDFSpatialInterpolator(gi, [ncvarname])
            pat = file_pattern.replace('{fieldstr:}', fnstr)
            pat = os.path.join(basedir, pat)
            ts = interpolation.DailyFileTimeSearch(pat, init_date, verbose=verbose)
            ti = interpolation.LinearTimeInterpolator(ts, r)
            self.time_interpolator[ncvarname] = ti
        # construct velocity rotation object
        self.rotate_velocity = ('U_Velocity' in field_names
                                and 'V_Velocity' in field_names)
        self.scalar_field_names = list(self.field_names)
        if self.rotate_velocity:
            self.scalar_field_names.remove('U_Velocity')
            self.scalar_field_names.remove('V_Velocity')
            lat = self.grid_interpolator_3d.latlonz_array[:, 0]
            lon = self.grid_interpolator_3d.latlonz_array[:, 1]
            self.vect_rotator = coord_system.get_vector_rotator(lon, lat)

    @PETSc.Log.EventDecorator("thetis.NCOMInterpolator.set_fields")
    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        if self.rotate_velocity:
            # water_u (meter/sec) = Eastward Water Velocity
            # water_v (meter/sec) = Northward Water Velocity
            lon_vel = self.time_interpolator['U_Velocity'](time)[0]
            lat_vel = self.time_interpolator['V_Velocity'](time)[0]
            u, v = self.vect_rotator(lon_vel, lat_vel)
            self.fields['U_Velocity'].dat.data_with_halos[:] = u
            self.fields['V_Velocity'].dat.data_with_halos[:] = v

        for fname in self.scalar_field_names:
            vals = self.time_interpolator[fname](time)[0]
            self.fields[fname].dat.data_with_halos[:] = vals


class SpatialInterpolatorROMS3d(interpolation.SpatialInterpolator):
    """
    Abstract spatial interpolator class that can interpolate onto a Function
    """
    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorROMS3d.__init__")
    def __init__(self, function_space, coord_system):
        """
        :arg function_space: target Firedrake FunctionSpace
        :arg coord_system: :class:`CoordinateSystem` object
        """
        self.function_space = function_space

        # construct local coordinates
        xyz = SpatialCoordinate(self.function_space.mesh())
        tmp_func = self.function_space.get_work_function()
        xyz_array = numpy.zeros((tmp_func.dat.data_with_halos.shape[0], 3))
        for i in range(3):
            tmp_func.interpolate(xyz[i])
            xyz_array[:, i] = tmp_func.dat.data_with_halos[:]
        self.function_space.restore_work_function(tmp_func)

        self.latlonz_array = numpy.zeros_like(xyz_array)
        lon, lat = coord_system.to_lonlat(
            xyz_array[:, 0], xyz_array[:, 1]
        )
        self.latlonz_array[:, 0] = lat
        self.latlonz_array[:, 1] = lon
        self.latlonz_array[:, 2] = xyz_array[:, 2]

        self._initialized = False

    def _get_subset_nodes(self, grid_x, grid_y, target_x, target_y):
        """
        Returns grid nodes that are necessary for intepolating onto target_x,y
        """
        orig_shape = grid_x.shape
        grid_xy = numpy.array((grid_x.ravel(), grid_y.ravel())).T
        target_xy = numpy.array((target_x.ravel(), target_y.ravel())).T
        tri = qhull.Delaunay(grid_xy)
        simplex = tri.find_simplex(target_xy)
        vertices = numpy.take(tri.simplices, simplex, axis=0)
        nodes = numpy.unique(vertices.ravel())
        nodes_x, nodes_y = numpy.unravel_index(nodes, orig_shape)

        return nodes, nodes_x, nodes_y

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorROMS3d._compute_roms_z_coord")
    def _compute_roms_z_coord(self, ncfile, constant_zeta=None):
        zeta = ncfile['zeta'][0, :, :]
        bath = ncfile['h'][:]
        # NOTE compute z coordinates for full levels (w)
        cs = ncfile['Cs_w'][:]
        s = ncfile['s_w'][:]
        hc = ncfile['hc'][:]

        # ROMS transformation ver. 2:
        # z(x, y, sigma, t) = zeta(x, y, t) + (zeta(x, y, t) +  h(x, y))*S(x, y, sigma)
        zeta = zeta[self.ind_lat, self.ind_lon][self.mask].filled(0.0)
        bath = bath[self.ind_lat, self.ind_lon][self.mask]
        if constant_zeta:
            zeta = numpy.ones_like(bath)*constant_zeta
        ss = (hc*s[:, numpy.newaxis] + bath[numpy.newaxis, :]*cs[:, numpy.newaxis])/(hc + bath[numpy.newaxis, :])
        grid_z_w = zeta[numpy.newaxis, :]*(1 + ss) + bath[numpy.newaxis, :]*ss
        grid_z = 0.5*(grid_z_w[1:, :] + grid_z_w[:-1, :])
        grid_z[0, :] = grid_z_w[0, :]
        grid_z[-1, :] = grid_z_w[-1, :]
        return grid_z

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorROMS3d._create_interpolator")
    def _create_interpolator(self, ncfile):
        """
        Create compact interpolator by finding the minimal necessary support
        """
        lat = ncfile['lat_rho'][:]
        lon = ncfile['lon_rho'][:]
        self.mask = ncfile['mask_rho'][:].astype(bool)
        self.nodes, self.ind_lat, self.ind_lon = self._get_subset_nodes(lat, lon, self.latlonz_array[:, 0], self.latlonz_array[:, 1])
        lat_subset = lat[self.ind_lat, self.ind_lon]
        lon_subset = lon[self.ind_lat, self.ind_lon]
        self.mask = self.mask[self.ind_lat, self.ind_lon]

        # COMPUTE z coords for constant elevation=0.1
        grid_z = self._compute_roms_z_coord(ncfile, constant_zeta=0.1)

        # omit land mask
        lat_subset = lat_subset[self.mask]
        lon_subset = lon_subset[self.mask]

        nz = grid_z.shape[0]

        # data shape is [nz, neta, nxi]
        grid_lat = numpy.tile(lat_subset, (nz, 1, 1)).ravel()
        grid_lon = numpy.tile(lon_subset, (nz, 1, 1)).ravel()
        grid_z = grid_z.ravel()
        if numpy.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if numpy.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        if numpy.ma.isMaskedArray(grid_z):
            grid_z = grid_z.filled(0.0)
        grid_latlonz = numpy.vstack((grid_lat, grid_lon, grid_z)).T

        # building 3D interpolator, this can take a long time (minutes)
        print_output('Constructing 3D GridInterpolator...')
        self.interpolator = interpolation.GridInterpolator(
            grid_latlonz, self.latlonz_array, normalize=True,
            fill_mode='nearest'
        )
        print_output('done.')

        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.SpatialInterpolatorROMS3d.interpolate")
    def interpolate(self, nc_filename, variable_list, itime):
        """
        Calls the interpolator object
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                grid_data = ncfile[var][itime, :, :, :][:, self.ind_lat, self.ind_lon][:, self.mask].filled(numpy.nan).ravel()
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class LiveOceanInterpolator(object):
    """
    Interpolates LiveOcean (ROMS) model data on 3D fields
    """
    @PETSc.Log.EventDecorator("thetis.LiveOceanInterpolator.__init__")
    def __init__(self, function_space, fields, field_names, ncfile_pattern, init_date, coord_system):
        self.function_space = function_space
        for f in fields:
            assert f.function_space() == self.function_space, 'field \'{:}\' does not belong to given function space {:}.'.format(f.name(), self.function_space.name)
        assert len(fields) == len(field_names)
        self.fields = fields
        self.field_names = field_names

        # construct interpolators
        self.grid_interpolator = SpatialInterpolatorROMS3d(self.function_space, coord_system)
        self.reader = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, field_names)
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile_pattern, init_date, interpolation.NetCDFTimeParser, time_variable_name='ocean_time', verbose=False)
        self.time_interpolator = interpolation.LinearTimeInterpolator(self.timesearch_obj, self.reader)

    @PETSc.Log.EventDecorator("thetis.LiveOceanInterpolator.set_fields")
    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        vals = self.time_interpolator(time)
        for i in range(len(self.fields)):
            self.fields[i].dat.data_with_halos[:] = vals[i]


class GenericSpatialInterpolator2D(interpolation.SpatialInterpolator2d):
    """
    Spatial interpolator class for interpolating netCDF 2D fields.
    """
    def _get_nc_var_name(self, ncfile, standard_name):
        """
        Find netCDF variable name that matches CF standard_name.

        Raises an AssertionError if standard_name is not found.
        :arg ncfile: netCDF Dataset object
        :arg standard_name: standard_name to look for
        :returns: name of the netCDF variable
        """
        name = None
        for var in ncfile.variables.values():
            if hasattr(var, 'standard_name') and var.standard_name == standard_name:
                name = var.name
                break
        msg = f'Variable {standard_name} not found in {ncfile.filepath()}'
        assert name is not None, msg
        return name

    def _get_subset_nodes(self, grid_x, grid_y, target_x, target_y):
        """
        Returns grid nodes that are necessary for intepolating onto target_x,y
        """
        orig_shape = grid_x.shape
        grid_xy = numpy.array((grid_x.ravel(), grid_y.ravel())).T
        target_xy = numpy.array((target_x.ravel(), target_y.ravel())).T
        tri = qhull.Delaunay(grid_xy)
        simplex = tri.find_simplex(target_xy)
        vertices = numpy.take(tri.simplices, simplex, axis=0)
        nodes = numpy.unique(vertices.ravel())
        nodes_x, nodes_y = numpy.unravel_index(nodes, orig_shape)

        return nodes, nodes_x, nodes_y

    @PETSc.Log.EventDecorator("thetis.GenericSpatialInterpolator2D._create_interpolator")
    def _create_interpolator(self, ncfile):
        """
        Create compact interpolator by finding the minimal necessary support
        """
        lat_name = self._get_nc_var_name(ncfile, 'latitude')
        lon_name = self._get_nc_var_name(ncfile, 'longitude')
        lat1d = ncfile[lat_name][:]
        lon1d = ncfile[lon_name][:]
        lat, lon = numpy.meshgrid(lat1d, lon1d, indexing='ij')
        # find valid mask
        self.valid_mask = None
        # read a variable and take inverse of its mask
        for name, var in ncfile.variables.items():
            if len(var.shape) == 3:
                v = var[0, ...]  # read first time index
                self.valid_mask = ~v.mask
                break
        assert self.valid_mask is not None, 'could not determine mask'
        assert self.valid_mask.shape == lat.shape, 'mask has wrong shape {self.mask.shape} {lat.shape}'

        self.nodes, self.ind_lat, self.ind_lon = self._get_subset_nodes(lat, lon, self.mesh_lonlat[:, 1], self.mesh_lonlat[:, 0])
        lat_subset = lat[self.ind_lat, self.ind_lon]
        lon_subset = lon[self.ind_lat, self.ind_lon]

        self.valid_mask = self.valid_mask[self.ind_lat, self.ind_lon]

        # omit land mask
        lat_subset = lat_subset[self.valid_mask]
        lon_subset = lon_subset[self.valid_mask]

        grid_lat = lat_subset.ravel()
        grid_lon = lon_subset.ravel()
        if numpy.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if numpy.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        grid_latlon = numpy.vstack((grid_lat, grid_lon)).T

        # building 2D interpolator, this can take a long time (minutes)
        print_output('Constructing 2D GridInterpolator...')
        mesh_latlon = self.mesh_lonlat[:, [1, 0]]
        self.interpolator = interpolation.GridInterpolator(
            grid_latlon, mesh_latlon, normalize=False,
            fill_mode='nearest'
        )
        print_output('done.')

        self._initialized = True

    @PETSc.Log.EventDecorator("thetis.GenericSpatialInterpolator2D.interpolate")
    def interpolate(self, nc_filename, variable_list, itime):
        """
        Calls the interpolator object
        """
        with netCDF4.Dataset(nc_filename, 'r') as ncfile:
            if not self._initialized:
                self._create_interpolator(ncfile)
            output = []
            for var in variable_list:
                assert var in ncfile.variables
                grid_data = ncfile[var][itime, ...][self.ind_lat, self.ind_lon][self.valid_mask]
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class GenericInterpolator2D(object):
    """
    Interpolates 2D fields from netCDF files.

    The grid latitude, longitude coordinates must be defined in 1D arrays
    with CF standard_name attributes "latitude" and "longitude". Time must be
    defined with cftime compliant units and metadata.
    """
    @PETSc.Log.EventDecorator("thetis.GenericInterpolator2D.__init__")
    def __init__(self, function_space, fields, field_names, ncfile_pattern,
                 init_date, coord_system, vector_field=None,
                 vector_components=None, vector_rotator=None):
        self.function_space = function_space
        for f in fields:
            assert f.function_space() == self.function_space, 'field \'{:}\' does not belong to given function space {:}.'.format(f.name(), self.function_space.name)
        assert len(fields) == len(field_names)
        self.fields = fields
        self.field_names = list(field_names)
        self.scalar_field_index = list(range(len(field_names)))
        # construct interpolators
        self.grid_interpolator = GenericSpatialInterpolator2D(self.function_space, coord_system)
        self.reader = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, self.field_names)
        # TODO generalize _get_nc_var_name and use it for time dimension as well
        self.timesearch_obj = interpolation.NetCDFTimeSearch(ncfile_pattern, init_date, interpolation.NetCDFTimeParser, time_variable_name='time', verbose=False)
        self.time_interpolator = interpolation.LinearTimeInterpolator(self.timesearch_obj, self.reader)

        self.rotate_velocity = vector_components is not None
        if self.rotate_velocity:
            assert vector_field is not None, 'vector_field must be provided'
            self.field_names += list(vector_components)
            self.vector_field_index = [self.field_names.index(c) for c in vector_components]
            self.vector_field = vector_field

            lon = self.grid_interpolator.mesh_lonlat[:, 0]
            lat = self.grid_interpolator.mesh_lonlat[:, 1]
            if vector_rotator is None:
                self.vect_rotator = coord_system.get_vector_rotator(lon, lat)
            else:
                self.vect_rotator = vector_rotator

    @PETSc.Log.EventDecorator("thetis.GenericInterpolator2D.set_fields")
    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        vals = self.time_interpolator(time)
        if self.rotate_velocity:
            i, j = self.vector_field_index
            east_comp = vals[i]
            north_comp = vals[j]
            if self.vector_field.ufl_shape == (3,):
                u, v, w = self.vect_rotator(east_comp, north_comp)
                self.vector_field.dat.data_with_halos[:, 0] = u
                self.vector_field.dat.data_with_halos[:, 1] = v
                self.vector_field.dat.data_with_halos[:, 2] = w
            else:
                u, v = self.vect_rotator(east_comp, north_comp)
                self.vector_field.dat.data_with_halos[:, 0] = u
                self.vector_field.dat.data_with_halos[:, 1] = v
        for i in self.scalar_field_index:
            self.fields[i].dat.data_with_halos[:] = vals[i]


class TidalBoundaryForcing(ABC):
    """Base class for tidal boundary interpolators."""

    @abstractproperty
    def coord_layout():
        """
        Data layout in the netcdf files.

        Either 'lon,lat' or 'lat,lon'.
        """
        return 'lon,lat'

    @abstractproperty
    def compute_velocity():
        """If True, compute tidal currents as well."""
        return False

    @PETSc.Log.EventDecorator("thetis.TidalBoundaryForcing.__init__")
    def __init__(self, elev_field, init_date, coord_system, vect_rotator=None,
                 uv_field=None, constituents=None, boundary_ids=None,
                 data_dir=None):
        """
        :arg elev_field: Function where tidal elevation will be interpolated.
        :arg init_date: Datetime object defining the simulation init time.
        :arg coord_system: :class:`CoordinateSystem` object
        :kwarg vect_rotator: User-defined vector rotator function
        :kwarg uv_field: Function where tidal transport will be interpolated.
        :kwarg constituents: list of tidal constituents, e.g. ['M2', 'K1']
        :kwarg boundary_ids: list of boundary_ids where tidal data will be
            evaluated. If not defined, tides will be in evaluated in the entire
            domain.
        :kwarg data_dir: path to directory where tidal model netCDF files are
            located.
        """
        assert init_date.tzinfo is not None, 'init_date must have time zone information'
        if constituents is None:
            constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']

        self.data_dir = data_dir if data_dir is not None else ''

        if not self.compute_velocity and uv_field is not None:
            warning('{:}: uv_field is defined but velocity computation is not supported. uv_field will be ignored.'.format(__class__.__name__))
        self.compute_velocity = self.compute_velocity and uv_field is not None

        # determine nodes at the boundary
        self.elev_field = elev_field
        self.uv_field = uv_field
        function_space = elev_field.function_space()
        if boundary_ids is None:
            # interpolate in the whole domain
            self.nodes = numpy.arange(self.elev_field.dat.data_with_halos.shape[0])
        else:
            bc = DirichletBC(function_space, 0., boundary_ids)
            self.nodes = bc.nodes
        self._empty_set = self.nodes.size == 0

        # construct local coordinates
        on_sphere = function_space.mesh().geometric_dimension == 3

        if on_sphere:
            x, y, z = SpatialCoordinate(function_space.mesh())
            fsx = Function(function_space).interpolate(x).dat.data_with_halos
            fsy = Function(function_space).interpolate(y).dat.data_with_halos
            fsz = Function(function_space).interpolate(z).dat.data_with_halos
            coords = (fsx, fsy, fsz)
        else:
            x, y = SpatialCoordinate(function_space.mesh())
            fsx = Function(function_space).interpolate(x).dat.data_with_halos
            fsy = Function(function_space).interpolate(y).dat.data_with_halos
            coords = (fsx, fsy)

        lon, lat = coord_system.to_lonlat(*coords, positive_lon=True)
        self.latlon = numpy.array([lat, lon]).T

        if not self._empty_set:
            # compute bounding box
            bounds_lat = [self.latlon[:, 0].min(), self.latlon[:, 0].max()]
            bounds_lon = [self.latlon[:, 1].min(), self.latlon[:, 1].max()]
            if self.coord_layout == 'lon,lat':
                self.ranges = (bounds_lon, bounds_lat)
            else:
                self.ranges = (bounds_lat, bounds_lon)

            self.tide = uptide.Tides(constituents)
            self.tide.set_initial_time(init_date)
            self._create_readers()

            if self.compute_velocity:
                lat = self.latlon[:, 0]
                lon = self.latlon[:, 1]
                if vect_rotator is None:
                    self.vect_rotator = coord_system.get_vector_rotator(lon, lat)
                else:
                    self.vect_rotator = vect_rotator

    @abstractmethod
    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        pass

    @PETSc.Log.EventDecorator("thetis.TidalBoundaryForcing.set_tidal_field")
    def set_tidal_field(self, t):
        elev_data = self.elev_field.dat.data_with_halos
        if self.compute_velocity:
            uv_data = self.uv_field.dat.data_with_halos
        if self._empty_set:
            return
        self.tnci.set_time(t)
        if self.compute_velocity:
            self.tnciu.set_time(t)
            self.tnciv.set_time(t)
        if self.compute_velocity:
            lat_vel = numpy.zeros_like(elev_data)
            lon_vel = numpy.zeros_like(elev_data)
        for i, node in enumerate(self.nodes):
            lat, lon = self.latlon[node, :]
            point = (lon, lat) if self.coord_layout == 'lon,lat' else (lat, lon)
            try:
                elev = self.tnci.get_val(point, allow_extrapolation=True)
                elev_data[node] = elev
            except uptide.netcdf_reader.CoordinateError:
                elev_data[node] = 0.
            if self.compute_velocity:
                try:
                    lon_vel[node] = self.tnciu.get_val(point, allow_extrapolation=True)
                    lat_vel[node] = self.tnciv.get_val(point, allow_extrapolation=True)
                except uptide.netcdf_reader.CoordinateError:
                    uv_data[node, 0] = 0
                    uv_data[node, 1] = 0
        if self.compute_velocity:
            uv = self.vect_rotator(lon_vel, lat_vel)
            uv_data[:, 0] = uv[0]
            uv_data[:, 1] = uv[1]


class TPXOTidalBoundaryForcing(TidalBoundaryForcing):
    """
    Interpolator for TPXO global tidal model data.

    See the `TPXO website <https://www.tpxo.net/global/tpxo9-atlas>`_ for data
    access. By default, this routine assumes the `tpxo9v5a` data set in NetCDF
    format. Other versions can be used by setting correct `elev_file`,
    `uv_file`, and `grid_file` arguments.
    """
    coord_layout = 'lon,lat'
    compute_velocity = True

    @PETSc.Log.EventDecorator("thetis.TPXOTidalBoundaryForcing.__init__")
    def __init__(self, elev_field, init_date, coord_system,
                 vect_rotator=None, uv_field=None, constituents=None,
                 boundary_ids=None, data_dir=None, elev_file='h_tpxo9.v5a.nc',
                 uv_file='u_tpxo9.v5a.nc', grid_file='gridtpxo9v5a.nc'):
        """
        :arg elev_field: Function where tidal elevation will be interpolated.
        :arg init_date: Datetime object defining the simulation init time.
        :arg coord_system: :class:`CoordinateSystem` object
        :kwarg vect_rotator: User-defined vector rotator function
        :kwarg uv_field: Function where tidal transport will be interpolated.
        :kwarg constituents: list of tidal constituents, e.g. ['M2', 'K1']
        :kwarg boundary_ids: list of boundary_ids where tidal data will be
            evaluated. If not defined, tides will be in evaluated in the entire
            domain.
        :kwarg data_dir: path to directory where tidal model netCDF files are
            located.
        :kwarg elev_file: TPXO tidal elevation file
        :kwarg uv_file: TPXO transport/velocity file
        :kwarg grid_file: TPXO grid file
        """
        self.compute_velocity = uv_field is not None
        self.elev_nc_file = elev_file
        self.uv_nc_file = uv_file
        self.grid_nc_file = grid_file
        super().__init__(
            elev_field, init_date, coord_system,
            vect_rotator=vect_rotator, uv_field=uv_field,
            constituents=constituents, boundary_ids=boundary_ids,
            data_dir=data_dir
        )

    @PETSc.Log.EventDecorator("thetis.TPXOTidalBoundaryForcing._create_readers")
    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        msg = 'File {:} not found.'
        f_grid = os.path.join(self.data_dir, self.grid_nc_file)
        assert os.path.exists(f_grid), msg.format(f_grid)
        f_elev = os.path.join(self.data_dir, self.elev_nc_file)
        assert os.path.exists(f_elev), msg.format(f_elev)
        self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(self.tide, f_grid, f_elev, ranges=self.ranges)
        if self.uv_field is not None:
            f_uv = os.path.join(self.data_dir, self.uv_nc_file)
            assert os.path.exists(f_uv), msg.format(f_uv)
            self.tnciu = uptide.tidal_netcdf.OTPSncTidalComponentInterpolator(self.tide, f_grid, f_uv, 'u', 'u', ranges=self.ranges)
            self.tnciv = uptide.tidal_netcdf.OTPSncTidalComponentInterpolator(self.tide, f_grid, f_uv, 'v', 'v', ranges=self.ranges)


class FES2004TidalBoundaryForcing(TidalBoundaryForcing):
    """Tidal boundary interpolator for FES2004 tidal model."""
    elev_nc_file = 'tide.fes2004.nc'
    uv_nc_file = None
    grid_nc_file = None
    coord_layout = 'lat,lon'
    compute_velocity = False

    @PETSc.Log.EventDecorator("thetis.FES2004TidalBoundaryForcing._create_readers")
    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        f_elev = os.path.join(self.data_dir, self.elev_nc_file)
        msg = 'File {:} not found, download it from \nftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/'.format(f_elev)
        assert os.path.exists(f_elev), msg
        self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(self.tide, f_elev, ranges=self.ranges)
