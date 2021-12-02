from thetis import *
from thetis.diagnostics import *
import pytest


@pytest.fixture(params=[True, False])
def interp(request):
    return request.param


def test_hessian_recovery2d(interp, tol=1.0e-08):
    r"""
    Apply Hessian recovery techniques to the quadratic
    polynomial :math:`f(x, y) = \frac12(x^2 + y^2)` and
    check that the result is the identity matrix.

    We get the gradient approximation as part of the
    recovery procedure, so check that this is also as
    expected.

    :arg interp: should the polynomial be interpolated as
        as a :class:`Function`, or left as a UFL expression?
    :kwarg tol: relative tolerance for value checking
    """
    mesh2d = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh2d)
    bowl = 0.5*(x**2 + y**2)
    if interp:
        P2_2d = get_functionspace(mesh2d, 'CG', 2)
        bowl = interpolate(bowl, P2_2d)
    P1v_2d = get_functionspace(mesh2d, 'CG', 1, vector=True)
    P1t_2d = get_functionspace(mesh2d, 'CG', 1, tensor=True)

    # Recover derivatives
    g = Function(P1v_2d, name='Gradient')
    H = Function(P1t_2d, name='Hessian')
    hessian_recoverer = HessianRecoverer2D(bowl, H, g)
    hessian_recoverer.solve()

    # Check values
    g_expect = Function(P1v_2d, name='Expected gradient')
    g_expect.interpolate(as_vector([x, y]))
    H_expect = Function(P1t_2d, name='Expected Hessian')
    H_expect.interpolate(Identity(2))
    err_g = errornorm(g, g_expect)/norm(g_expect)
    assert err_g < tol, f'Gradient approximation error {err_g:.4e}'
    err_H = errornorm(H, H_expect)/norm(H_expect)
    assert err_H < tol, f'Hessian approximation error {err_H:.4e}'


def test_vorticity_calculation2d(interp, tol=1.0e-08):
    r"""
    Calculate the vorticity of the velocity field
    :math:`\mathbf u(x, y) = \frac12(y, -x)` and check
    that the result is unity.

    :arg interp: should the velocity be interpolated as
        as a :class:`Function`, or left as a UFL expression?
    :kwarg tol: relative tolerance for value checking
    """
    mesh2d = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh2d)
    uv = 0.5*as_vector([y, -x])
    if interp:
        P1v_2d = get_functionspace(mesh2d, 'CG', 1, vector=True)
        uv = interpolate(uv, P1v_2d)
    P1_2d = get_functionspace(mesh2d, 'CG', 1)

    # Recover vorticity
    omega = Function(P1_2d, name='Vorticity')
    vorticity_calculator = VorticityCalculator2D(uv, omega)
    vorticity_calculator.solve()

    # Check values
    omega_expect = Function(P1_2d, name='Expected vorticity')
    omega_expect.assign(1.0)
    err = errornorm(omega, omega_expect)/norm(omega_expect)
    assert err < tol, f'Vorticity approximation error {err:.4e}'
