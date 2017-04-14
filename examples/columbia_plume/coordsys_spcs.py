"""
Generic methods for converting data between different spatial coordinate systems.
Uses pyproj library.

Tuomas Karna 2013-01-15
"""

import pyproj
import numpy as np
import collections

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


def spcs2lonlat(x, y):
    """Converts SPCS to longitude-latitude."""
    return convertCoords(x, y, SPCS_N_OR, LL_WO)


def lonlat2spcs(lon, lat):
    """Converts longitude-latitude to SPCS."""
    return convertCoords(lon, lat, LL_WO, SPCS_N_OR)


def spcs2utm(x, y):
    """Converts SPCS to longitude-latitude."""
    return convertCoords(x, y, SPCS_N_OR, UTM_ZONE10)


def utm2spcs(lon, lat):
    """Converts longitude-latitude to SPCS."""
    return convertCoords(lon, lat, UTM_ZONE10, SPCS_N_OR)


def WGS842spcs(lon, lat):
    """Converts longitude-latitude to SPCS."""
    return convertCoords(lon, lat, LL_WGS84, SPCS_N_OR)
