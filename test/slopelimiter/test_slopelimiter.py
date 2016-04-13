"""
Tests slope limiters
"""
from thetis import *
from thetis.limiter import VertexBasedP1DGLimiter
import pytest


def vertex_limiter_test(dim=3, type='linear', direction='x', export=False):
    """
    type == 'linear': Tests that a linear field is not altered by the limiter.
        Tracer is a linear field in x|y|z|xz direction, projected to p1dg.
    type == 'jump': Tests that limiter conserves mass and eliminates overshoots
        Tracer is a jump in x|y|z|xz direction, projected to p1dg.

    """
    mesh2d = UnitSquareMesh(5, 5)
    if dim == 3:
        nlayers = 5
        mesh = ExtrudedMesh(mesh2d, nlayers, 1.0/nlayers)
        # slanted prisms
        xyz = mesh.coordinates
        xyz.dat.data[:, 2] *= 1.0 + 0.25 - 0.5*xyz.dat.data[:, 0]
        p1dg = FunctionSpace(mesh, 'DP', 1, vfamily='DP', vdegree=1)
        p1 = FunctionSpace(mesh, 'P', 1, vfamily='P', vdegree=1)
        p0 = FunctionSpace(mesh, 'DP', 0, vfamily='DP', vdegree=0)
    else:
        p1dg = FunctionSpace(mesh2d, 'DP', 1)
        p1 = FunctionSpace(mesh2d, 'P', 1)
        p0 = FunctionSpace(mesh2d, 'DP', 0)

    coordname = {'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]', 'xz': 'x[0]*x[2]'}

    tracer = Function(p1dg, name='tracer')
    tracer_original = Function(p1dg, name='tracer original')
    if type == 'linear':
        tracer_original.project(Expression(coordname[direction]))
    if type == 'jump':
        tracer_original.project(Expression('0.5 + 0.5*tanh(20*({x}-0.5))'.format(x=coordname[direction])))
    tracer.project(tracer_original)

    if export:
        tracer_file = File('tracer.pvd')
        tracer_file.write(tracer)

    limiter = VertexBasedP1DGLimiter(p1dg, p1, p0)
    limiter.apply(tracer)
    if export:
        tracer_file.write(tracer)

    if type == 'linear':
        l2_err = errornorm(tracer_original, tracer)
        assert l2_err < 1e-12
    if type == 'jump':
        mass_orig = assemble(tracer_original*dx)
        mass = assemble(tracer*dx)
        assert abs(mass - mass_orig) < 1e-12
        assert tracer.dat.data.min() > 0.0


@pytest.fixture(params=['x', 'y'])
def direction_2d(request):
    return request.param


@pytest.fixture(params=['x', 'y', 'z'])
def direction_3d(request):
    return request.param


@pytest.fixture(params=['linear', 'jump'])
def type(request):
    return request.param


def test_limiter_2d(type, direction_2d):
    vertex_limiter_test(dim=2, type=type, direction=direction_2d)


def test_limiter_3d(type, direction_3d):
    vertex_limiter_test(dim=3, type=type, direction=direction_3d)


if __name__ == '__main__':
    vertex_limiter_test(dim=2, type='linear', direction='x', export=True)
    vertex_limiter_test(dim=3, type='jump', direction='z', export=True)
    vertex_limiter_test(dim=3, type='linear', direction='z', export=True)
