"""
Methods for reading NCOM ocean model outputs
"""
from thetis import *
from atm_forcing import to_latlon
from thetis.timezone import *
import netCDF4
from thetis.interpolation import GridInterpolator, SpatialInterpolator


class SpatialInterpolatorNCOM3d(SpatialInterpolator):
    """
    Abstract spatial interpolator class that can interpolate onto a Function
    """
    def __init__(self, function_space, to_latlon, grid_path):
        """
        :arg function_space: target Firedrake FunctionSpace
        :arg to_latlon: Python function that converts local mesh coordinates to
            latitude and longitude: 'lat, lon = to_latlon(x, y)'
        """
        self.function_space = function_space
        self.grid_path = grid_path

        # construct local coordinates
        xyz = SpatialCoordinate(self.function_space.mesh())
        tmp_func = self.function_space.get_work_function()
        xyz_array = np.zeros((tmp_func.dat.data_with_halos.shape[0], 3))
        for i in range(3):
            tmp_func.interpolate(xyz[i])
            xyz_array[:, i] = tmp_func.dat.data_with_halos[:]
        self.function_space.restore_work_function(tmp_func)

        self.latlonz_array = np.zeros_like(xyz_array)
        lat, lon = to_latlon(xyz_array[:, 0], xyz_array[:, 1], positive_lon=True)
        self.latlonz_array[:, 0] = lat
        self.latlonz_array[:, 1] = lon
        self.latlonz_array[:, 2] = xyz_array[:, 2]

        self._initialized = False

    def _get_forcing_grid(self, filename, varname):
        v = None
        with netCDF4.Dataset(os.path.join(self.grid_path, filename), 'r') as ncfile:
            v = ncfile[varname][:]
        return v

    def _create_interpolator(self, ncfile):
        """
        Create compact interpolator by finding the minimal necessary support
        """
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
        vals = ncfile[varkey][:]  # shape nz, lat, lon
        land_mask = np.all(vals.mask, axis=0)

        # build 2d mask
        mask_good_values = ~land_mask
        # neighborhood mask with bounding box
        mask_cover = np.zeros_like(mask_good_values)
        buffer = 0.2
        lat_min = self.latlonz_array[:, 0].min() - buffer
        lat_max = self.latlonz_array[:, 0].max() + buffer
        lon_min = self.latlonz_array[:, 1].min() - buffer
        lon_max = self.latlonz_array[:, 1].max() + buffer
        mask_cover[(lat >= lat_min) *
                   (lat <= lat_max) *
                   (lon >= lon_min) *
                   (lon <= lon_max)] = True
        mask_cover *= mask_good_values
        # include nearest valid neighbors
        # needed for nearest neighbor filling
        from scipy.spatial import cKDTree
        good_lat = lat[mask_good_values]
        good_lon = lon[mask_good_values]
        ll = np.vstack([good_lat.ravel(), good_lon.ravel()]).T
        dist, ix = cKDTree(ll).query(self.latlonz_array[:, :2])
        ix = np.unique(ix)
        ix = np.nonzero(mask_good_values.ravel())[0][ix]
        a, b = np.unravel_index(ix, lat.shape)
        mask_nn = np.zeros_like(mask_good_values)
        mask_nn[a, b] = True
        # final mask
        mask = mask_cover + mask_nn

        self.nodes = np.nonzero(mask.ravel())[0]
        self.ind_lat, self.ind_lon = np.unravel_index(self.nodes, lat.shape)

        # find 3d mask where data is not defined
        vals = vals[:, self.ind_lat, self.ind_lon]
        self.good_mask_3d = ~vals.mask

        lat_subset = lat[self.ind_lat, self.ind_lon]
        lon_subset = lon[self.ind_lat, self.ind_lon]

        assert len(lat_subset) > 0, 'rank {:} has no source lat points'
        assert len(lon_subset) > 0, 'rank {:} has no source lon points'

        # construct vertical grid
        zm = self._get_forcing_grid('model_zm.nc', 'zm')
        zm = zm[:, y_ind, :][:, :, x_ind]
        grid_z = zm[:, self.ind_lat, self.ind_lon]  # shape (nz, nlatlon)
        grid_z = grid_z.filled(-5000.)
        # nudge water surface higher for interpolation
        grid_z[0, :] = 1.5
        nz = grid_z.shape[0]

        # data shape is [nz, neta*nxi]
        grid_lat = np.tile(lat_subset, (nz, 1))[self.good_mask_3d]
        grid_lon = np.tile(lon_subset, (nz, 1))[self.good_mask_3d]
        grid_z = grid_z[self.good_mask_3d]
        if np.ma.isMaskedArray(grid_lat):
            grid_lat = grid_lat.filled(0.0)
        if np.ma.isMaskedArray(grid_lon):
            grid_lon = grid_lon.filled(0.0)
        if np.ma.isMaskedArray(grid_z):
            grid_z = grid_z.filled(0.0)
        grid_latlonz = np.vstack((grid_lat, grid_lon, grid_z)).T

        # building 3D interpolator, this can take a long time (minutes)
        print_output('Constructing 3D GridInterpolator...')
        self.interpolator = GridInterpolator(grid_latlonz, self.latlonz_array,
                                             normalize=True,
                                             fill_mode='nearest',
                                             dont_raise=True)
        print_output('done.')
        self._initialized = True

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
                # TODO generalize data dimensions, sniff from netcdf file
                grid_data = ncfile[var][:][:, self.ind_lat, self.ind_lon][self.good_mask_3d]
                data = self.interpolator(grid_data)
                output.append(data)
        return output


class NCOMInterpolator(object):
    """
    Interpolates NCOM model data on 3D fields
    """
    def __init__(self, function_space, fields, field_names, field_fnstr,
                 file_pattern, init_date, verbose=False):
        self.function_space = function_space
        for f in fields:
            assert f.function_space() == self.function_space, 'field \'{:}\' does not belong to given function space {:}.'.format(f.name(), self.function_space.name)
        assert len(fields) == len(field_names)
        assert len(fields) == len(field_fnstr)
        self.fields = fields
        self.field_names = field_names

        # construct interpolators
        self.grid_interpolator = SpatialInterpolatorNCOM3d(self.function_space, to_latlon, 'forcings/ncom/')
        # each field is in different file
        # construct time search and interp objects separately for each
        self.time_interpolator = {}
        for ncvarname, fnstr in zip(field_names, field_fnstr):
            r = interpolation.NetCDFSpatialInterpolator(self.grid_interpolator, [ncvarname])
            pat = file_pattern.replace('{fieldstr:}', fnstr)
            ts = interpolation.DailyFileTimeSearch(pat, init_date, verbose=verbose)
            ti = interpolation.LinearTimeInterpolator(ts, r)
            self.time_interpolator[ncvarname] = ti

    def set_fields(self, time):
        """
        Evaluates forcing fields at the given time
        """
        for i in range(len(self.fields)):
            vals = self.time_interpolator[self.field_names[i]](time)
            self.fields[i].dat.data_with_halos[:] = vals[0]


def test_time_search():
    """
    Tests time search object.

    Time stamps are deduced from NCOM output files.

    .. note::
        The following NCOM output files must be present:
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006041900.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006042000.nc
        ./forcings/ncom/2006/s3d/s3d.glb8_2f_2006042100.nc
    """
    # test time search
    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 4, 20, tzinfo=sim_tz)
    pattern = 'forcings/ncom/{year:04d}/s3d/s3d.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc'
    timesearch_obj = interpolation.DailyFileTimeSearch(pattern, init_date, verbose=True)

    sim_time = 100.0
    fn, itime, time = timesearch_obj.find(sim_time, previous=True)
    assert fn == 'forcings/ncom/2006/s3d/s3d.glb8_2f_2006041900.nc'
    assert itime == 0
    assert time == -72000.0
    fn, itime, time = timesearch_obj.find(sim_time)
    assert fn == 'forcings/ncom/2006/s3d/s3d.glb8_2f_2006042000.nc'
    assert itime == 0
    assert time == 14400.0

    dt = 3*3600.
    for i in range(8):
        d = init_date + datetime.timedelta(seconds=i*dt)
        print('Time step {:}, {:}'.format(i, d))
        fn, itime, time = timesearch_obj.find(i*dt, previous=True)
        print('  prev: {:} {:}'.format(fn, itime))
        fn, itime, time = timesearch_obj.find(i*dt, previous=False)
        print('  next: {:} {:}'.format(fn, itime))


def test_interpolator():
    """
    Test NCOM 3d interpolator.

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

    # load and extrude mesh
    from bathymetry import get_bathymetry, smooth_bathymetry, smooth_bathymetry_at_bnd
    nlayers, surf_elem_height, max_z_stretch = (9, 5.0, 4.0)
    mesh2d = Mesh('mesh_cre-plume_02.msh')

    # interpolate bathymetry and smooth it
    bathymetry_2d = get_bathymetry('bathymetry_utm.nc', mesh2d, project=False)
    bathymetry_2d = smooth_bathymetry(
        bathymetry_2d, delta_sigma=1.0, bg_diff=0,
        alpha=1e2, exponent=2.5,
        minimum_depth=3.5, niter=30)
    bathymetry_2d = smooth_bathymetry_at_bnd(bathymetry_2d, [2, 7])

    # 3d mesh vertical stretch factor
    z_stretch_fact_2d = Function(bathymetry_2d.function_space(), name='z_stretch')
    # 1.0 (sigma mesh) in shallow areas, 4.0 in deep ocean
    z_stretch_fact_2d.project(-ln(surf_elem_height/bathymetry_2d)/ln(nlayers))
    z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data < 1.0] = 1.0
    z_stretch_fact_2d.dat.data[z_stretch_fact_2d.dat.data > max_z_stretch] = max_z_stretch

    extrude_options = {
        'z_stretch_fact': z_stretch_fact_2d,
    }
    mesh = extrude_mesh_sigma(mesh2d, nlayers, bathymetry_2d,
                              **extrude_options)
    p1 = FunctionSpace(mesh, 'CG', 1)

    # make functions
    salt = Function(p1, name='salinity')
    temp = Function(p1, name='temperature')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 4, 20, tzinfo=sim_tz)
    interp = NCOMInterpolator(
        p1, [salt, temp], ['Salinity', 'Temperature'], ['s3d', 't3d'],
        'forcings/ncom/{year:04d}/{fieldstr:}/{fieldstr:}.glb8_2f_{year:04d}{month:02d}{day:02d}00.nc',
        init_date, verbose=True
    )
    interp.set_fields(0.0)
    salt_fn = 'tmp/salt.pvd'
    temp_fn = 'tmp/temp.pvd'
    print('Saving output to {:} {:}'.format(salt_fn, temp_fn))
    out_salt = File(salt_fn)
    out_temp = File(temp_fn)

    out_salt.write(salt)
    out_temp.write(temp)

    dt = 900.
    for i in range(8):
        print('Time step {:}'.format(i))
        interp.set_fields(i*dt)
        out_salt.write(salt)
        out_temp.write(temp)


if __name__ == '__main__':
    test_time_search()
    test_interpolator()
