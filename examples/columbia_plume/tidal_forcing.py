import coordsys_spcs
import uptide
import uptide.tidal_netcdf
from firedrake import *
import numpy as np
import os
import pytz

utc_tz = pytz.timezone('UTC')

tide_file = 'tide.fes2004.nc'
msg = 'File {:} not found, download it from \nftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/'.format(tide_file)
assert os.path.exists(tide_file), msg


def to_latlon(x, y):
    lon, lat = coordsys_spcs.spcs2lonlat(x, y)
    if lon < 0.0:
        lon += 360.
    return lat, lon


class TidalBoundaryForcing(object):
    def __init__(self, tidal_field, init_date,
                 constituents=None, boundary_ids=None):
        assert init_date.tzinfo is not None, 'init_date must have time zone information'
        if constituents is None:
            constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']

        # determine nodes at the boundary
        fs = tidal_field.function_space()
        if boundary_ids is None:
            # interpolate in the whole domain
            self.nodes = np.arange(self.tidal_field.dat.data_with_halos.shape[0])
        else:
            bc = DirichletBC(fs, 0., boundary_ids, method='geometric')
            self.nodes = bc.nodes

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
                    lat, lon = to_latlon(x, y)
                    bounds_lat[0] = min(bounds_lat[0], lat)
                    bounds_lat[1] = max(bounds_lat[1], lat)
                    bounds_lon[0] = min(bounds_lon[0], lon)
                    bounds_lon[1] = max(bounds_lon[1], lon)

            ranges = (bounds_lat, bounds_lon)
        else:
            ranges = None

        tide = uptide.Tides(constituents)
        tide.set_initial_time(init_date)
        self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(tide, tide_file, ranges=ranges)
        self.tidal_field = tidal_field

        latlon = []
        for node in self.nodes:
            x, y = fsx[node], fsy[node]
            lat, lon = to_latlon(x, y)
            latlon.append((lat, lon))
        self.nll = zip(self.nodes, latlon)

    def set_tidal_field(self, t):
        self.tnci.set_time(t)
        data = self.tidal_field.dat.data_with_halos
        for node, (lat, lon) in self.nll:
            try:
                val = self.tnci.get_val((lat, lon), allow_extrapolation=True)
                data[node] = val
            except uptide.netcdf_reader.CoordinateError:
                data[node] = 0.


def test():
    import datetime
    mesh2d = Mesh('mesh_cre-plume002.msh')
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    m2_field = Function(p1, name='elevation')

    timezone = pytz.timezone('Etc/GMT+8')
    init_date = datetime.datetime(2016, 5 , 1, tzinfo=timezone)

    tbnd = TidalBoundaryForcing(m2_field,
                                init_date,
                                constituents=['M2'],
                                boundary_ids=[1, 2, 3])
    tbnd.set_tidal_field(0.0)

    out = File('tmp/tidal_elev.pvd')
    for t in np.linspace(0, 12*3600., 60):
        tbnd.set_tidal_field(t)
        n = norm(m2_field)
        if m2_field.function_space().mesh().comm.rank == 0:
            print n
        out.write(m2_field)


if __name__ == '__main__':
    test()
