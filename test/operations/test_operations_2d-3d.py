import pytest
from firedrake import *
import cofs.utility as utility
import numpy as np


@pytest.fixture(scope="module")
def mesh2d():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope="module")
def mesh(mesh2d):
    return ExtrudedMesh(mesh2d, layers=10, layer_height=-0.1)


@pytest.fixture(params=["P1xP1", "P2xP2", "P1DGxP1DG", "P2DGxP2DG"])
def spaces(request):
    if request.param == "P1xP1":
        return (("CG", 1), ("CG", 1))
    elif request.param == "P2xP2":
        return (("CG", 2), ("CG", 2))
    elif request.param == "P1DGxP1DG":
        return (("DG", 1), ("DG", 1))
    elif request.param == "P2DGxP2DG":
        return (("DG", 2), ("DG", 2))


@pytest.fixture
def P1_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return FunctionSpace(mesh2d, name, order)


@pytest.fixture
def P1(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return FunctionSpace(mesh, name, order,
                         vfamily=vname, vdegree=vorder)


@pytest.fixture
def U_2d(mesh2d, spaces):
    (name, order), (vname, vorder) = spaces
    return VectorFunctionSpace(mesh2d, name, order)


@pytest.fixture
def U(mesh, spaces):
    (name, order), (vname, vorder) = spaces
    return VectorFunctionSpace(mesh, name, order,
                               vfamily=vname, vdegree=vorder)


@pytest.fixture
def c3d(P1):
    return Function(P1, name="Tracer").interpolate(Expression("x[2] + 2.0"))


@pytest.fixture
def c3d_x(P1):
    return Function(P1, name="Tracer").interpolate(Expression("x[0] + 2.0"))


@pytest.fixture
def c2d(P1_2d):
    return Function(P1_2d, name="Tracer").interpolate(Expression("4.0"))


@pytest.fixture
def c2d_x(P1_2d):
    return Function(P1_2d, name="Tracer").interpolate(Expression("2*x[0]"))


@pytest.fixture
def uv_3d(U):
    return Function(U, name="Velocity").interpolate(Expression(('x[2] + 1.0',
                                                                '2.0*x[2] + 4.0',
                                                                '3.0*x[2] + 6.0')))


@pytest.fixture
def uv_3d_x(U):
    return Function(U, name="Velocity").interpolate(Expression(('x[0] + 1.0',
                                                                '2.0*x[1] + 4.0',
                                                                '3.0*x[0]*x[2] + 6.0')))


@pytest.fixture
def uv_2d(U_2d):
    return Function(U_2d, name="Velocity").interpolate(Expression(('4.0', '8.0')))


@pytest.fixture
def uv_2d_x(U_2d):
    return Function(U_2d, name="Velocity").interpolate(Expression(('4.0*x[0]', '8.0*x[1]')))


@pytest.mark.parametrize("bottom",
                         ([True, 1.0],
                          [False, 2.0]))
def test_copy3dFieldTo2d(c3d, c2d, bottom):
    bottom, expect = bottom
    utility.copy3dFieldTo2d(c3d, c2d, useBottomValue=bottom)
    assert np.allclose(c2d.dat.data_ro[:], expect)


@pytest.mark.parametrize("bottom",
                         ([True, (0.0, 2.0)],
                          [False, (1.0, 4.0)]))
def test_copy3dFieldTo2d_vec(uv_3d, uv_2d, bottom):
    bottom, expect = bottom
    utility.copy3dFieldTo2d(uv_3d, uv_2d, useBottomValue=bottom)
    assert np.allclose(uv_2d.dat.data_ro, expect)


@pytest.mark.parametrize("bottom", (True, False))
def test_copy3dFieldTo2d_x(c3d_x, c2d_x, bottom):
    utility.copy3dFieldTo2d(c3d_x, c2d_x, useBottomValue=bottom)
    assert np.allclose(c2d_x.dat.data_ro.min(), 2.0)
    assert np.allclose(c2d_x.dat.data_ro.max(), 3.0)


@pytest.mark.parametrize("bottom", (True, False))
def test_copy3dFieldTo2d_x_vec(uv_3d_x, uv_2d_x, bottom):
    utility.copy3dFieldTo2d(uv_3d_x, uv_2d_x, useBottomValue=bottom)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 0].min(), 1.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 0].max(), 2.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 1].min(), 4.0)
    assert np.allclose(uv_2d_x.dat.data_ro[:, 1].max(), 6.0)


def test_copy2dFieldTo3d(c2d, c3d):
    utility.copy2dFieldTo3d(c2d, c3d)
    assert np.allclose(c3d.dat.data_ro[:], 4.0)


def test_copy2dFieldTo3d_x(c2d_x, c3d_x):
    utility.copy2dFieldTo3d(c2d_x, c3d_x)
    assert np.allclose(c3d_x.dat.data_ro.min(), 0.0)
    assert np.allclose(c3d_x.dat.data_ro.max(), 2.0)


def test_copy2dFieldTo3d_x_vec(uv_2d_x, uv_3d_x):
    utility.copy2dFieldTo3d(uv_2d_x, uv_3d_x)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 0].min(), 0.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 0].max(), 4.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 1].min(), 0.0)
    assert np.allclose(uv_3d_x.dat.data_ro[:, 1].max(), 8.0)


def test_copy2dFieldTo3d_vec(uv_2d, uv_3d):
    utility.copy2dFieldTo3d(uv_2d, uv_3d)
    assert np.allclose(uv_3d.dat.data_ro[:, 0], 4.0)
    assert np.allclose(uv_3d.dat.data_ro[:, 1], 8.0)

if __name__ == '__main__':
    """Run all tests"""
    import os
    pytest.main(os.path.abspath(__file__))
