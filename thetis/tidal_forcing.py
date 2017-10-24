import thetis.coordsys as coordsys
import uptide
import uptide.tidal_netcdf
from firedrake import *
import numpy as np
import os.path

def to_latlon(coord_sys, x, y):
    lon, lat = coordsys.convert_coords(coord_sys, coordsys.LL_WGS84, x, y)
    if lon < 0.0:
        lon += 360.
    return lat, lon

def check_tidal_file(tidal_file, name, download_location):
    if not os.path.exists(tidal_file):
        raise IOError("Could not access {} '{}'. Fix path or download from {}".format(name, tidal_file, download_location))

class TidalBoundaryForcing(object):
    """
    Sets the tidal_field to the tidal elevation reconstructed from a global solution like FES or TPXO which can be used as the forcing for a regional tidal model.

    :arg tidal_field: scalar Function to store the tidal elevation
    :arg initial_datetime: datetime object (should have tzinfo set) that corresponds to t=0 in the model
    :arg coord_sys: coordinate system (pyproj projection) used in the mesh of tidal_field
    :arg constituents: list of tidal constituents to include, default: ['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2']
    :arg boundary_ids: if specified tidal elevation is only calculated on the nodes of this part of the boundary,
      otherwise the tidal elevation is calculated everywhere

    specify one of:
    :arg fes2004_file: path to the tide.fes2004.nc file (download from ftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/tide/)
    :arg tpxo_grid_file: and :arg tpxo_hf_file: path to TPXO grid (gridXXX.nc) and elevation (hf.XXX.nc) files (download any of the _netcdf solutions from ftp://ftp.oce.orst.edu/dist/tides/regional/)
    """
    def __init__(self, tidal_field, inital_datetime, coord_sys,
                 constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'],
                 boundary_ids=None,
                 fes2004_file=None,
                 tpxo_grid_file=None, tpxo_hf_file=None):
        assert inital_datetime.tzinfo is not None, 'inital_datetime must have time zone information'

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
        latlon = []

        for node in self.nodes:
            x, y = fsx[node], fsy[node]
            lat, lon = to_latlon(coord_sys, x, y)
            latlon.append((lat, lon))
        self.latlon = np.array(latlon)
        if self._empty_set:
            return

        # compute bounding box
        bounds_lat = [self.latlon[:, 0].min(), self.latlon[:, 0].max()]
        bounds_lon = [self.latlon[:, 1].min(), self.latlon[:, 1].max()]
        ranges = (bounds_lat, bounds_lon)

        tide = uptide.Tides(constituents)
        tide.set_initial_time(inital_datetime)
        if fes2004_file is not None:
            if tpxo_grid_file is not None:
                raise ValueError("Need to specify either fes2004_file or tpxo_grid_file, not both")
            check_tidal_file(fes2004_file, "FES2004 file", "ftp://ftp.legos.obs-mip.fr/pub/soa/maree/tide_model/global_solution/fes2004/tide.")
            self.tnci = uptide.tidal_netcdf.FESTidalInterpolator(tide, fes2004_file, ranges=ranges)
        elif tpxo_grid_file is not None:
            if tpxo_hf_file is None:
                raise ValueError("Need to specify both tpxo_grid_file and tpxo_hf_file.")
            check_tidal_file(tpxo_grid_file, "TPXO grid file", "ftp://ftp.oce.orst.edu/dist/tides/regional/")
            check_tidal_file(tpxo_hf_file, "TPXO elevation file", "ftp://ftp.oce.orst.edu/dist/tides/regional/")
            self.tnci = uptide.tidal_netcdf.OTPSncTidalInterpolator(tide, tpxo_grid_file, tpxo_hf_file, ranges=ranges)

    def set_tidal_field(self, t):
        """Compute tidal elevations at time t."""
        if not self._empty_set:
            self.tnci.set_time(t)
        data = self.tidal_field.dat.data_with_halos
        for (lat, lon), node in zip(self.latlon, self.nodes):
            try:
                val = self.tnci.get_val((lat, lon), allow_extrapolation=True)
                data[node] = val
            except uptide.netcdf_reader.CoordinateError:
                data[node] = 0.
