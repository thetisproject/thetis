import pytest
from firedrake import *
import thetis.utility as utility
import thetis.utility3d as utility3d
import numpy


@pytest.fixture(scope="module")
def mesh2d():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope="module")
def mesh(mesh2d):
    fs = utility.get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(fs).assign(1.0)
    n_layers = 10
    return utility.extrude_mesh_sigma(mesh2d, n_layers, bathymetry_2d)


@pytest.fixture(params=[
    "P1xP1", "P2xP2",
    "P1DGxP1DG", "P2DGxP2DG",
    "P1xP3", "P1DGxP3DG",
    "P3xP1DG", "P3DGxP1",
])
def spaces(request):
    h_name, v_name = request.param.split('x')

    def get_tuple(name):
        degree = int(name[1:2])
        family = 'DG' if 'DG' in name else 'CG'
        return (family, degree)

    return (get_tuple(h_name), get_tuple(v_name))


@pytest.fixture
def p1_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return utility.get_functionspace(mesh2d, name, order)


@pytest.fixture
def p1(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return utility.get_functionspace(mesh, name, order, vname, vorder)


@pytest.fixture
def u_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return utility.get_functionspace(mesh2d, name, order, vector=True)


@pytest.fixture
def u(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return utility.get_functionspace(mesh, name, order, vname, vorder,
                                     vector=True)


@pytest.fixture
def c3d(p1):
    x, y, z = SpatialCoordinate(p1.mesh())
    return Function(p1, name="Tracer").interpolate(z + 2.0)


@pytest.fixture
def c3d_x(p1):
    x, y, z = SpatialCoordinate(p1.mesh())
    return Function(p1, name="Tracer").interpolate(x + 2.0)


@pytest.fixture
def c2d(p1_2d):
    x, y = SpatialCoordinate(p1_2d.mesh())
    return Function(p1_2d, name="Tracer").interpolate(Constant(4.0))


@pytest.fixture
def c2d_x(p1_2d):
    x, y = SpatialCoordinate(p1_2d.mesh())
    return Function(p1_2d, name="Tracer").interpolate(2*x)


@pytest.fixture
def uv_3d(u):
    x, y, z = SpatialCoordinate(u.mesh())
    return Function(u, name="Velocity").interpolate(as_vector((z + 1.0,
                                                               2.0*z + 4.0,
                                                               3.0*z + 6.0)))


@pytest.fixture
def uv_3d_x(u):
    x, y, z = SpatialCoordinate(u.mesh())
    return Function(u, name="Velocity").interpolate(as_vector((x + 1.0,
                                                               2.0*y + 4.0,
                                                               3.0*x*z + 6.0)))


@pytest.fixture
def uv_2d(u_2d):
    x, y = SpatialCoordinate(u_2d.mesh())
    return Function(u_2d, name="Velocity").interpolate(Constant((4.0, 8.0)))


@pytest.fixture
def uv_2d_x(u_2d):
    x, y = SpatialCoordinate(u_2d.mesh())
    return Function(u_2d, name="Velocity").interpolate(as_vector((4.0*x,
                                                                  8.0*y)))


@pytest.mark.parametrize('params',
                         (['bottom', 'bottom', 1.0],
                          ['bottom', 'top', 1.1],
                          ['bottom', 'average', 1.05],
                          ['top', 'top', 2.0],
                          ['top', 'bottom', 1.9],
                          ['top', 'average', 1.95],
                          ))
def test_copy_3d_field_to_2d(c3d, c2d, params):
    boundary, facet, expect = params
    utility3d.SubFunctionExtractor(c3d, c2d, boundary=boundary, elem_facet=facet).solve()
    assert numpy.allclose(c2d.dat.data_ro[:], expect)


@pytest.mark.parametrize('params',
                         (['bottom', 'bottom', (0.0, 2.0)],
                          ['bottom', 'top', (0.1, 2.2)],
                          ['bottom', 'average', (0.05, 2.1)],
                          ['top', 'top', (1.0, 4.0)],
                          ['top', 'bottom', (0.9, 3.8)],
                          ['top', 'average', (0.95, 3.9)],
                          ))
def test_copy_3d_field_to_2d_vec(uv_3d, uv_2d, params):
    boundary, facet, expect = params
    utility3d.SubFunctionExtractor(uv_3d, uv_2d, boundary=boundary, elem_facet=facet).solve()
    assert numpy.allclose(uv_2d.dat.data_ro, expect)


@pytest.mark.parametrize('boundary', ('top', 'bottom'))
@pytest.mark.parametrize('facet', ('top', 'bottom', 'average'))
def test_copy_3d_field_to_2d_x(c3d_x, c2d_x, boundary, facet):
    utility3d.SubFunctionExtractor(c3d_x, c2d_x, boundary=boundary, elem_facet=facet).solve()
    assert numpy.allclose(c2d_x.dat.data_ro.min(), 2.0)
    assert numpy.allclose(c2d_x.dat.data_ro.max(), 3.0)


@pytest.mark.parametrize('boundary', ('top', 'bottom'))
@pytest.mark.parametrize('facet', ('top', 'bottom', 'average'))
def test_copy_3d_field_to_2d_x_vec(uv_3d_x, uv_2d_x, boundary, facet):
    utility3d.SubFunctionExtractor(uv_3d_x, uv_2d_x, boundary=boundary, elem_facet=facet).solve()
    assert numpy.allclose(uv_2d_x.dat.data_ro[:, 0].min(), 1.0)
    assert numpy.allclose(uv_2d_x.dat.data_ro[:, 0].max(), 2.0)
    assert numpy.allclose(uv_2d_x.dat.data_ro[:, 1].min(), 4.0)
    assert numpy.allclose(uv_2d_x.dat.data_ro[:, 1].max(), 6.0)


def test_copy_2d_field_to_3d(c2d, c3d):
    utility3d.ExpandFunctionTo3d(c2d, c3d).solve()
    assert numpy.allclose(c3d.dat.data_ro[:], 4.0)


def test_copy_2d_field_to_3d_x(c2d_x, c3d_x):
    utility3d.ExpandFunctionTo3d(c2d_x, c3d_x).solve()
    assert numpy.allclose(c3d_x.dat.data_ro.min(), 0.0)
    assert numpy.allclose(c3d_x.dat.data_ro.max(), 2.0)


def test_copy_2d_field_to_3d_x_vec(uv_2d_x, uv_3d_x):
    utility3d.ExpandFunctionTo3d(uv_2d_x, uv_3d_x).solve()
    assert numpy.allclose(uv_3d_x.dat.data_ro[:, 0].min(), 0.0)
    assert numpy.allclose(uv_3d_x.dat.data_ro[:, 0].max(), 4.0)
    assert numpy.allclose(uv_3d_x.dat.data_ro[:, 1].min(), 0.0)
    assert numpy.allclose(uv_3d_x.dat.data_ro[:, 1].max(), 8.0)


def test_copy_2d_field_to_3d_vec(uv_2d, uv_3d):
    utility3d.ExpandFunctionTo3d(uv_2d, uv_3d).solve()
    assert numpy.allclose(uv_3d.dat.data_ro[:, 0], 4.0)
    assert numpy.allclose(uv_3d.dat.data_ro[:, 1], 8.0)


def test_minimum_angle(mesh2d):
    min_angle = utility.get_minimum_angles_2d(mesh2d).vector().gather().min()
    assert numpy.allclose(min_angle, pi/4)


if __name__ == '__main__':
    """Run all tests"""
    import os
    pytest.main(os.path.abspath(__file__))
