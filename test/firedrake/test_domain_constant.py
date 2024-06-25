from thetis import domain_constant
import firedrake as fd
import numpy as np


def test_domain_constant():
    mesh = fd.UnitSquareMesh(1, 1)
    arr = [1, [1, 2], [[1, 2], [3, 4]]]
    for value in arr:
        shape = np.shape(value)
        fl_arr = np.asarray(value).flatten()

        # create from float or (nested) list)
        dc = domain_constant(value, mesh)
        assert dc.ufl_shape == shape
        np.testing.assert_equal(dc.dat.data, fl_arr)

        # create from Constant
        dc = domain_constant(fd.Constant(value), mesh)
        assert dc.ufl_shape == shape
        np.testing.assert_equal(dc.dat.data, fl_arr)

        # create from domain_constant
        dc = domain_constant(dc, mesh)
        assert dc.ufl_shape == shape
        np.testing.assert_equal(dc.dat.data, fl_arr)
