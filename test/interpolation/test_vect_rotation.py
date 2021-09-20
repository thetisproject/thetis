"""
Test vector rotation
"""
import thetis.coordsys as coordsys
import numpy
import pytest


def compute_rotation_error(direction='lat'):
    delta_mag = 1e-5
    lat, lon = [46.248070, -124.073456]
    delta_lat = delta_lon = 0
    if direction == 'lat':
        delta_lat = delta_mag
    else:
        delta_lon = delta_mag

    # compute direction in xy space
    x, y = coordsys.convert_coords(coordsys.LL_WGS84, coordsys.UTM_ZONE10,
                                   lon, lat)
    x2, y2 = coordsys.convert_coords(coordsys.LL_WGS84, coordsys.UTM_ZONE10,
                                     lon + delta_lon, lat + delta_lat)
    vect_x, vect_y = x2-x, y2-y
    mag = numpy.hypot(vect_x, vect_y)
    nvect_x, nvect_y = vect_x/mag, vect_y/mag

    # we should be able to get the same unit vector through rotation
    R, theta = coordsys.get_vector_rotation_matrix(
        coordsys.LL_WGS84, coordsys.UTM_ZONE10, lon, lat, delta=delta_mag)
    delta_xy = numpy.matmul(R, numpy.array([[delta_lon], [delta_lat]]))
    mag2 = numpy.hypot(delta_xy[0, 0], delta_xy[1, 0])
    nvect_x2, nvect_y2 = delta_xy[0, 0]/mag2, delta_xy[1, 0]/mag2

    assert numpy.allclose(nvect_x, nvect_x2, atol=1e-5)
    assert numpy.allclose(nvect_y, nvect_y2, atol=1e-5)


@pytest.mark.parametrize('direction', ['lat', 'lon'])
def test_vector_rotation(direction):
    compute_rotation_error(direction=direction)
