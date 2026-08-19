import numpy
import pytest

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import get_working_tape

from thetis.inversion_tools import IndependentPointControlMapping


def make_real_controls(mesh, values, name):
    R = FunctionSpace(mesh, 'R', 0)
    controls = []
    for i, value in enumerate(values):
        control = Function(R, name=f'{name}_{i:02d}')
        control.assign(float(value))
        controls.append(control)
    return controls


@pytest.mark.parametrize('method', ['linear', 'cubic', 'rbf'])
def test_independent_point_mapping_gradient(method):
    get_working_tape().clear_tape()
    continue_annotation()

    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, 'CG', 1)
    x, y = SpatialCoordinate(mesh)
    target = Function(V).interpolate(0.022 + 0.004*sin(pi*x)*cos(pi*y) + 0.002*x)

    points = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.2, 0.25],
        [0.75, 0.2],
        [0.25, 0.75],
        [0.8, 0.8],
        [0.5, 0.5],
    ])
    values = numpy.array([0.020, 0.024, 0.023, 0.021, 0.026, 0.019, 0.025, 0.022, 0.027])
    direction_values = numpy.array([0.30, -0.20, 0.15, -0.10, 0.25, -0.35, 0.40, -0.30, 0.20])

    controls = make_real_controls(mesh, values, f'{method}_control')
    directions = make_real_controls(mesh, direction_values, f'{method}_direction')
    mapping = IndependentPointControlMapping(V, points, method=method)

    field = Function(V, name=f'{method}_field')
    mapping.assign(field, controls)
    J = assemble(0.5*(field - target)**2*dx)
    reduced_functional = ReducedFunctional(J, [Control(c) for c in controls])

    assert taylor_test(reduced_functional, controls, directions) > 1.9
