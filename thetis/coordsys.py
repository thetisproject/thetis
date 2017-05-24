"""
Generic methods for converting data between different spatial coordinate systems.
Uses pyproj library.
"""

import pyproj
import numpy as np

# NOTE this adds pyproj as dependencency
# TODO generalize; define unique names for commonly-used systems


UTM_ZONE10 = pyproj.Proj(
    proj='utm',
    zone=10,
    datum='NAD83',
    units='m',
    errcheck=True)
SPCS_N_OR = pyproj.Proj(init='nad27:3601', errcheck=True)
LL_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84', errcheck=True)
LL_WO = pyproj.Proj(proj='latlong', nadgrids='WO', errcheck=True)  # HARN


def convertCoords(x, y, fromSys, toSys):
    """Converts x,y from fromSys to toSys."""
    if isinstance(x, np.ndarray):
        # proj may give wrong results if nans in the arrays
        lon = np.ones_like(x) * np.nan
        lat = np.ones_like(y) * np.nan
        goodIx = np.logical_and(np.isfinite(x), np.isfinite(y))
        lon[goodIx], lat[goodIx] = pyproj.transform(
            fromSys, toSys, x[goodIx], y[goodIx])
    else:
        lon, lat = pyproj.transform(fromSys, toSys, x, y)
    return lon, lat


def getVectorRotationMatrix(lon, lat, target_csys, source_csys=None,
                            delta=None):
    """
    Estimate rotation matrix that converts vectors from source_csys to
    target_csys at (lon, lat) location.


    A vector can defined at (lon, lat) can then be transformed as

    .. code-block:: python

        R, theta = getVectorRotationMatrix(lon, lat, target_csys)
        v_ll = numpy.array([[v_lon], [v_lat]])
        v_csys = numpy.matmul(R, v_ll)
        v_x, v_y = v_csys

    """
    if source_csys is None:
        source_csys = LL_WO
    if delta is None:
        delta = 1e-6  # ~1 m in LL_WO
    x, y = pyproj.transform(source_csys, target_csys, lon, lat)

    x2, y2 = pyproj.transform(source_csys, target_csys, lon, lat + delta)
    dxdl = (x2 - x) / delta
    dydl = (y2 - y) / delta
    theta = np.arctan2(-dxdl, dydl)

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    return R, theta


class VectorCoordSysRotation(object):
    """
    Rotates vectors defined in (lat,lon) to desired local coordinates
    """
    def __init__(self, lon, lat, target_csys):
        """
        :arg lon: lon coordinates of mesh points
        :arg lat: lat coordinates of mesh points
        :arg target_csys: target pyproj coordinate system object
        """
        self.target_csys = target_csys
        self._compute_rotation(lon, lat)

    def _compute_rotation(self, lon, lat):
        R, theta = getVectorRotationMatrix(lon, lat, self.target_csys)
        self.rotation_sin = np.sin(theta)
        self.rotation_cos = np.cos(theta)

    def __call__(self, v_lon, v_lat):
        """Apply rotation to vectors with components v_lat, v_lon"""
        # | c -s | | v_lon |
        # | s  c | | v_lat |
        u = v_lon * self.rotation_cos - v_lat * self.rotation_sin
        v = v_lon * self.rotation_sin + v_lat * self.rotation_cos
        return u, v
