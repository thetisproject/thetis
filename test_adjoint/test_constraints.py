"""
Test minimum distance constraints
"""
# import pytest
from thetis import *
import numpy
# import pyadjoint


def taylor_test(f, dfdx, x, h, n=8):
    import pdb
    pdb.set_trace()
    df_h = dfdx(x) @ h
    fx = f(x)
    residuals = []
    eps = 1.
    for i in range(n):
        residuals.append(numpy.linalg.norm(f(x+eps*h)-fx-eps*df_h))
        eps /= 2
    residuals = numpy.array(residuals)
    convergence = numpy.log(residuals[0:-1]/residuals[1:])/numpy.log(2)
    print_output(f"Residuals: {residuals}")
    print_output(f"Convergence: {convergence}")
    return convergence.min()


def test_distance_constraints():
    pos = [[x, y] for x in numpy.arange(880, 1121, 60) for y in numpy.arange(160, 341, 60)]
    pos = numpy.array(pos)
    mdc = turbines.MinimumDistanceConstraints(pos, 50)
    posf = pos.flatten()
    h = numpy.random.random(pos.shape).flatten()*20 - 10
    assert taylor_test(mdc.function, mdc.jacobian, posf, h) > 1.95
