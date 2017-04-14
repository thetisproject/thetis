import coordsys_spcs
import uptide
import uptide.tidal_netcdf
from firedrake import *
import numpy as np
import os

tide_file = 'tide.fes2004.nc'
msg = 'File {:} not found, download it from \nftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/'.format(tide_file)
assert os.path.exists(tide_file), msg


def to_latlon(x, y):
    lon, lat = coordsys_spcs.spcs2lonlat(x, y)
    return lat, lon


class TidalBoundaryForcing(object):
    def __init__(self, tidal_field, init_time,
                 constituents=None, boundary_ids=None):
        if constituents is None:
            constituents = ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
        tide = uptide.Tides(constituents)
        tide.set_initial_time(init_time)
        self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(tide, tide_file, ranges=None)
        self.tidal_field = tidal_field
        fs = tidal_field.function_space()
        if boundary_ids is None:
            # interpolate for the whole domain
            self.nodes = np.arange(self.tidal_field.dat.data.shape[0])
        else:
            self.bc = DirichletBC(fs, 0., boundary_ids)
            self.nodes = self.bc.nodes

        xy = SpatialCoordinate(fs.mesh())
        fsx = Function(fs).interpolate(xy[0]).dat.data
        fsy = Function(fs).interpolate(xy[1]).dat.data
        latlon = []
        for node in self.nodes:
            x, y = fsx[node], fsy[node]
            lat, lon = to_latlon(x, y)
            if lon < 0.0:
                lon += 360.
            latlon.append((lat, lon))
        self.nll = zip(self.nodes, latlon)

    def set_tidal_field(self, t):
        self.tnci.set_time(t)
        data = self.tidal_field.dat.data
        for node, (lat, lon) in self.nll:
            try:
                val = self.tnci.get_val((lat, lon), allow_extrapolation=True)
                data[node] = val
            except uptide.netcdf_reader.CoordinateError:
                data[node] = 0.

# x = 333000.
# y = 295000.
# print x, y, to_latlon(x, y)

# mesh2d = Mesh('mesh_cre-plume002.msh')
# p1 = FunctionSpace(mesh2d, 'CG', 1)
# m2_field = Function(p1, name='elevation')
#
# tbnd = TidalBoundaryForcing(tnci, m2_field)
# tbnd.set_tidal_field(0.0)
#
# out = File('tidal_elev.pvd')
# for t in np.linspace(0, 12*3600., 60):
#     tbnd.set_tidal_field(t)
#     out.write(m2_field)
#