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
    csys = coordsys.UTMCoordinateSystem(utm_zone=10)

    x, y = csys.to_xy(lon, lat)
    x2, y2 = csys.to_xy(lon + delta_lon, lat + delta_lat)
    vect_x, vect_y = x2 - x, y2 - y
    mag = numpy.hypot(vect_x, vect_y)
    nvect_x, nvect_y = vect_x / mag, vect_y / mag

    # we should be able to get the same unit vector through rotation
    rotator = csys.get_vector_rotator(numpy.array([lon]), numpy.array([lat]))

    vect_x2, vect_y2 = rotator(numpy.array([delta_lon]), numpy.array([delta_lat]))
    mag = numpy.hypot(vect_x2, vect_y2)
    nvect_x2, nvect_y2 = vect_x2 / mag, vect_y2 / mag

    assert numpy.allclose(nvect_x, nvect_x2, atol=1e-5)
    assert numpy.allclose(nvect_y, nvect_y2, atol=1e-5)


@pytest.mark.parametrize('direction', ['lat', 'lon'])
def test_vector_rotation(direction):
    compute_rotation_error(direction=direction)
