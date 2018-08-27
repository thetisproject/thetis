from firedrake import *
import thetis.timezone as timezone
from thetis.log import *
from atm_forcing import to_latlon
import uptide
import uptide.tidal_netcdf
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import os


class TidalBoundaryForcing(object):
    """Base class for tidal boundary interpolators."""
    __metaclass__ = ABCMeta

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

    @abstractproperty
    def elev_nc_file():
        """Tidal elavation NetCDF file name."""
        return None

    @abstractproperty
    def uv_nc_file():
        """Tidal velocity NetCDF file name."""
        return None

    @abstractproperty
    def grid_nc_file():
        """Grid NetCDF file name."""
        return None

    def __init__(self, elev_field, init_date, uv_field=None,
                 constituents=None, boundary_ids=None, data_dir=None):
        """
        :arg elev_field: Function where tidal elevation will be interpolated.
        :arg init_date: Datetime object defining the simulation init time.
        :kwarg uv_field: Function where tidal transport will be interpolated.
        :kwarg constituents: list of tidal constituents, e.g. ['M2', 'K1']
        :kwarg boundary_ids: list of boundary_ids where tidal data will be
            evaluated. If not defined, tides will be in evaluated in the entire
            domain.
        :kward data_dir: path to directory where tidal model netCDF files are
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
        fs = elev_field.function_space()
        if boundary_ids is None:
            # interpolate in the whole domain
            self.nodes = np.arange(self.elev_field.dat.data_with_halos.shape[0])
        else:
            bc = DirichletBC(fs, 0., boundary_ids, method='geometric')
            self.nodes = bc.nodes
        self._empty_set = self.nodes.size == 0

        xy = SpatialCoordinate(fs.mesh())
        fsx = Function(fs).interpolate(xy[0]).dat.data_ro_with_halos
        fsy = Function(fs).interpolate(xy[1]).dat.data_ro_with_halos
        if not self._empty_set:

            latlon = []
            for node in self.nodes:
                x, y = fsx[node], fsy[node]
                lat, lon = to_latlon(x, y, positive_lon=True)
                latlon.append((lat, lon))
            self.latlon = np.array(latlon)

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

    @abstractmethod
    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        pass

    def set_tidal_field(self, t):
        if not self._empty_set:
            self.tnci.set_time(t)
            if self.compute_velocity:
                self.tnciu.set_time(t)
                self.tnciv.set_time(t)
        elev_data = self.elev_field.dat.data_with_halos
        if self.compute_velocity:
            uv_data = self.uv_field.dat.data_with_halos
        for i, node in enumerate(self.nodes):
            lat, lon = self.latlon[i, :]
            point = (lon, lat) if self.coord_layout == 'lon,lat' else (lat, lon)
            try:
                elev = self.tnci.get_val(point, allow_extrapolation=True)
                elev_data[node] = elev
            except uptide.netcdf_reader.CoordinateError:
                elev_data[node] = 0.
            if self.compute_velocity:
                try:
                    u = self.tnciu.get_val(point, allow_extrapolation=True)
                    v = self.tnciv.get_val(point, allow_extrapolation=True)
                    uv_data[node, :] = (u, v)
                except uptide.netcdf_reader.CoordinateError:
                    uv_data[node, :] = (0, 0)


class TPXOTidalBoundaryForcing(TidalBoundaryForcing):
    """Tidal boundary interpolator for TPXO tidal model."""
    elev_nc_file = 'h_tpxo9.v1.nc'
    uv_nc_file = 'u_tpxo9.v1.nc'
    grid_nc_file = 'grid_tpxo9.nc'
    coord_layout = 'lon,lat'
    compute_velocity = True

    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        msg = 'File {:} not found, download it from \nftp://ftp.oce.orst.edu/dist/tides/Global/tpxo9_netcdf.tar.gz'
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

    def _create_readers(self, ):
        """Create uptide netcdf reader objects."""
        f_elev = os.path.join(self.data_dir, self.elev_nc_file)
        msg = 'File {:} not found, download it from \nftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/'.format(f_elev)
        assert os.path.exists(f_elev), msg
        self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(self.tide, f_elev, ranges=self.ranges)


def test():
    import datetime
    mesh2d = Mesh('mesh_cre-plume_02.msh')
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    p1v = VectorFunctionSpace(mesh2d, 'CG', 1)
    elev_field = Function(p1, name='elevation')
    uv_field = Function(p1v, name='transport')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 5, 15, tzinfo=sim_tz)

    # tide_cls = FES2004TidalBoundaryForcing
    tide_cls = TPXOTidalBoundaryForcing
    tbnd = tide_cls(
        elev_field, init_date, uv_field=uv_field,
        data_dir='forcings',
        constituents=['M2', 'K1'],
        boundary_ids=[2, 5, 7])
    tbnd.set_tidal_field(0.0)

    elev_outfn = 'tmp/tidal_elev.pvd'
    uv_outfn = 'tmp/tidal_uv.pvd'
    print('Saving to {:} {:}'.format(elev_outfn, uv_outfn))
    elev_out = File(elev_outfn)
    uv_out = File(uv_outfn)
    for t in np.linspace(0, 12*3600., 49):
        tbnd.set_tidal_field(t)
        if elev_field.function_space().mesh().comm.rank == 0:
            print('t={:7.1f} elev: {:7.1f} uv: {:7.1f}'.format(t, norm(elev_field), norm(uv_field)))
        elev_out.write(elev_field)
        uv_out.write(uv_field)


if __name__ == '__main__':
    test()
