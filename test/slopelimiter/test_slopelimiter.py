"""
Tests slope limiters
"""
from thetis import *
from thetis.limiter import VertexBasedP1DGLimiter, OptimalP1DGLimiter
import pytest


def vertex_limiter_test(limiter_kind='standard', dim=3,
                        init_field='linear', direction='x',
                        elem_type='triangle', export=False):
    """
    init_field == 'linear': Verify that a linear field is not altered.
        Tracer is a linear field in x|y|z|xz direction, projected to p1dg.
    init_field == 'jump': Verify mass conservation and overshoot suppression
        Tracer is a jump in x|y|z|xz direction, projected to p1dg.
    """
    mesh2d = UnitSquareMesh(5, 5, quadrilateral=elem_type == 'quadrilateral')
    if dim == 3:
        nlayers = 5
        mesh = ExtrudedMesh(mesh2d, nlayers, 1.0/nlayers)
        # slanted prisms
        xyz = mesh.coordinates
        xyz.dat.data[:, 2] *= 1.0 + 0.25 - 0.5*xyz.dat.data[:, 0]
        p1dg = get_functionspace(mesh, 'DP', 1, vfamily='DP', vdegree=1)
        x, y, z = SpatialCoordinate(mesh)
        z_func = Function(p1dg)
        z_func.interpolate(z)
        elem_height = Function(p1dg)
        compute_elem_height(z_func, elem_height)
    else:
        p1dg = get_functionspace(mesh2d, 'DP', 1)
        x, y = SpatialCoordinate(mesh2d)
        z = Constant(0)
        elem_height = None

    coord_expr = {'x': x, 'y': y, 'xy': x + 0.5*y - 0.25, 'z': z, 'xz': x * z}

    tracer = Function(p1dg, name='tracer')
    tracer_original = Function(p1dg, name='tracer original')
    if init_field == 'linear':
        tracer_original.project(coord_expr[direction])
    if init_field == 'jump':
        tracer_original.project(0.5 + 0.5*tanh(20*(coord_expr[direction]-0.5)))
    tracer.project(tracer_original)

    if export:
        tracer_file = File('tracer.pvd')
        tracer_file.write(tracer)

    if limiter_kind == 'standard':
        limiter = VertexBasedP1DGLimiter(p1dg, elem_height)
    else:
        limiter = OptimalP1DGLimiter(p1dg, elem_height)

    limiter.apply(tracer)
    if export:
        tracer_file.write(tracer)

    if init_field == 'linear':
        l2_err = errornorm(tracer_original, tracer)
        assert l2_err < 1e-12
    if init_field == 'jump':
        assert tracer.dat.data.min() > -1e-6
    mass_orig = assemble(tracer_original*dx)
    mass = assemble(tracer*dx)
    assert abs(mass - mass_orig) < 1e-12


@pytest.fixture(params=['standard', 'optimal'])
def limiter_kind(request):
    return request.param

@pytest.fixture(params=[
    'x', 'y',
    pytest.param('xy', marks=pytest.mark.xfail(reason='corner elements will be limited'))
])
def direction_2d(request):
    return request.param


@pytest.fixture(params=[
    'x', 'y',
    pytest.param('z', marks=pytest.mark.xfail(reason='surface corner elements will be limited')),
    pytest.param('xz', marks=pytest.mark.xfail(reason='corner elements will be limited'))
])
def direction_3d(request):
    return request.param


@pytest.fixture(params=['linear', 'jump'])
def init_field(request):
    return request.param


@pytest.fixture(params=['triangle', 'quadrilateral'])
def elem_type(request):
    return request.param


def test_limiter_2d(limiter_kind, init_field, direction_2d, elem_type):
    vertex_limiter_test(dim=2, limiter_kind=limiter_kind,
                        init_field=init_field, direction=direction_2d,
                        elem_type=elem_type)


def test_limiter_3d(init_field, direction_3d, elem_type):
    vertex_limiter_test(dim=3, limiter_kind=limiter_kind,
                        init_field=init_field, direction=direction_3d,
                        elem_type=elem_type)


if __name__ == '__main__':
    vertex_limiter_test(dim=2, init_field='linear', direction='x', export=True)
    vertex_limiter_test(dim=3, init_field='linear', direction='x', export=True)
