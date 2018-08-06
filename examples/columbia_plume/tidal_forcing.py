import thetis.timezone as timezone
from atm_forcing import to_latlon
import uptide
import uptide.tidal_netcdf
from firedrake import *
import numpy as np
import os

tide_file = 'forcings/tide.fes2004.nc'
msg = 'File {:} not found, download it from \nftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/'.format(tide_file)
assert os.path.exists(tide_file), msg


class TidalBoundaryForcing(object):
    def __init__(self, tidal_field, init_date,
                 constituents=None, boundary_ids=None):
        assert init_date.tzinfo is not None, 'init_date must have time zone information'
        if constituents is None:
            constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']

        # determine nodes at the boundary
        self.tidal_field = tidal_field
        fs = tidal_field.function_space()
        if boundary_ids is None:
            # interpolate in the whole domain
            self.nodes = np.arange(self.tidal_field.dat.data_with_halos.shape[0])
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
            ranges = (bounds_lat, bounds_lon)

            tide = uptide.Tides(constituents)
            tide.set_initial_time(init_date)
            self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(tide, tide_file, ranges=ranges)

    def set_tidal_field(self, t):
        if not self._empty_set:
            self.tnci.set_time(t)
        data = self.tidal_field.dat.data_with_halos
        for i, node in enumerate(self.nodes):
            lat, lon = self.latlon[i, :]
            try:
                val = self.tnci.get_val((lat, lon), allow_extrapolation=True)
                data[node] = val
            except uptide.netcdf_reader.CoordinateError:
                data[node] = 0.


def test():
    import datetime
    mesh2d = Mesh('mesh_cre-plume_02.msh')
    p1 = FunctionSpace(mesh2d, 'CG', 1)
    m2_field = Function(p1, name='elevation')

    sim_tz = timezone.FixedTimeZone(-8, 'PST')
    init_date = datetime.datetime(2006, 5, 15, tzinfo=sim_tz)

    tbnd = TidalBoundaryForcing(m2_field,
                                init_date,
                                constituents=['M2'],
                                boundary_ids=[2, 5, 7])
    tbnd.set_tidal_field(0.0)

    outfn = 'tmp/tidal_elev.pvd'
    print('Saving to {:}'.format(outfn))
    out = File(outfn)
    for t in np.linspace(0, 12*3600., 60):
        tbnd.set_tidal_field(t)
        n = norm(m2_field)
        if m2_field.function_space().mesh().comm.rank == 0:
            print(n)
        out.write(m2_field)


if __name__ == '__main__':
    test()
