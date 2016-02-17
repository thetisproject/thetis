import pytest
from firedrake import *
import thetis.utility as utility
import numpy as np


@pytest.fixture(scope="module")
def mesh2d():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope="module")
def mesh(mesh2d):
    return ExtrudedMesh(mesh2d, layers=10, layer_height=-0.1)


@pytest.fixture(params=["p1xp1", "P2xP2", "p1DGxp1DG", "P2DGxP2DG"])
def spaces(request):
    if request.param == "p1xp1":
        return (("CG", 1), ("CG", 1))
    elif request.param == "P2xP2":
        return (("CG", 2), ("CG", 2))
    elif request.param == "p1DGxp1DG":
        return (("DG", 1), ("DG", 1))
    elif request.param == "P2DGxP2DG":
        return (("DG", 2), ("DG", 2))


@pytest.fixture
def p1_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return FunctionSpace(mesh2d, name, order)


@pytest.fixture
def p1(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return FunctionSpace(mesh, name, order,
                         vfamily=vname, vdegree=vorder)


@pytest.fixture
def u_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return VectorFunctionSpace(mesh2d, name, order)


@pytest.fixture
def u(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return VectorFunctionSpace(mesh, name, order,
                               vfamily=vname, vdegree=vorder)


@pytest.fixture
def c3d(p1):
    return Function(p1, name="Tracer").interpolate(Expression("x[2] + 2.0"))


@pytest.fixture
def c3d_x(p1):
    return Function(p1, name="Tracer").interpolate(Expression("x[0] + 2.0"))


@pytest.fixture
def c2d(p1_2d):
    return Function(p1_2d, name="Tracer").interpolate(Expression("4.0"))


@pytest.fixture
def c2d_x(p1_2d):
    return Function(p1_2d, name="Tracer").interpolate(Expression("2*x[0]"))


@pytest.fixture
def uv_3d(u):
    return Function(u, name="Velocity").interpolate(Expression(('x[2] + 1.0',
                                                                '2.0*x[2] + 4.0',
                                                                '3.0*x[2] + 6.0')))


@pytest.fixture
def uv_3d_x(u):
    return Function(u, name="Velocity").interpolate(Expression(('x[0] + 1.0',
                                                                '2.0*x[1] + 4.0',
                                                                '3.0*x[0]*x[2] + 6.0')))


@pytest.fixture
def uv_2d(u_2d):
    return Function(u_2d, name="Velocity").interpolate(Expression(('4.0', '8.0')))


@pytest.fixture
def uv_2d_x(u_2d):
    return Function(u_2d, name="Velocity").interpolate(Expression(('4.0*x[0]', '8.0*x[1]')))


@pytest.mark.parametrize("bottom",
                         ([True, 1.0],
                          [False, 2.0]))
def test_copy_3d_field_to_2d(c3d, c2d, bottom):
    bottom, expect = bottom
    utility.SubFunctionExtractor(c3d, c2d, use_bottom_value=bottom).solve()
    assert np.allclose(c2d.dat.data_ro[:], expect)


@pytest.mark.parametrize("bottom",
                         ([True, (0.0, 2.0)],
                          [False, (1.0, 4.0)]))
def test_copy_3d_field_to_2d_vec(uv_3d, uv_2d, bottom):
    bottom, expect = bottom
    utility.SubFunctionExtractor(uv_3d, uv_2d, use_bottom_value=bottom).solve()
    assert np.allclose(uv_2d.dat.data_ro, expect)


@pytest.mark.parametrize("bottom", (True, False))
def test_copy_3d_field_to_2d_x(c3d_x, c2d_x, bottom):
    utility.SubFunctionExtractor(c3d_x, c2d_x, use_bottom_value=bottom).solve()
    assert np.allclose(c2d_x.dat.data_ro.min(), 2.0)
    assert np.allclose(c2d_x.dat.data_ro.max(), 3.0)


@pytest.mark.parametrize("bottom", (True, False))
def test_copy_3d_field_to_2d_x_vec(uv_3d_x, uv_2d_x, bottom):
    utility.SubFunctionExtractor(uv_3d_x, uv_2d_x, use_bottom_value=bottom).solve()
    assert np.allclose(uv_2d_x.dat.data_ro[:, 0].min(), 1.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 0].max(), 2.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 1].min(), 4.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 1].max(), 6.0)


def test_copy_2d_field_to_3d(c2d, c3d):
    utility.ExpandFunctionTo3d(c2d, c3d).solve()
    assert np.allclose(c3d.dat.data_ro[:], 4.0)


def test_copy_2d_field_to_3d_x(c2d_x, c3d_x):
    utility.ExpandFunctionTo3d(c2d_x, c3d_x).solve()
    assert np.allclose(c3d_x.dat.data_ro.min(), 0.0)
    assert np.allclose(c3d_x.dat.data_ro.max(), 2.0)


def test_copy_2d_field_to_3d_x_vec(uv_2d_x, uv_3d_x):
    utility.ExpandFunctionTo3d(uv_2d_x, uv_3d_x).solve()
    assert np.allclose(uv_3d_x.dat.data_ro[:, 0].min(), 0.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 0].max(), 4.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 1].min(), 0.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 1].max(), 8.0)


def test_copy_2d_field_to_3d_vec(uv_2d, uv_3d):
    utility.ExpandFunctionTo3d(uv_2d, uv_3d).solve()
    assert np.allclose(uv_3d.dat.data_ro[:, 0], 4.0)
    assert np.allclose(uv_3d.dat.data_ro[:, 1], 8.0)

if __name__ == '__main__':
    """Run all tests"""
    import os
    pytest.main(os.path.abspath(__file__))
