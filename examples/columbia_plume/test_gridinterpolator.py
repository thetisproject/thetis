"""
Test GridInterpolator object
"""

from interpolation import *
import numpy as np
from scipy.interpolate import griddata
import pytest


def do_interpolation(dataset='random', plot=False):
    """
    Compare GridInterpolator against scipy.griddata
    """
    np.random.seed(2)

    # fabricate dataset
    x_scale = 100.
    ndata = 35
    x = np.linspace(0, x_scale, ndata)
    y = np.linspace(0, x_scale, ndata)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack((xx.ravel(), yy.ravel())).T
    if dataset == 'sin':
        zz = np.sin(2*np.pi*xx/x_scale)*np.sin(1.5*2*np.pi*yy/x_scale)
    elif dataset == 'gauss':
        zz = np.exp(-(((xx - 50.)/20.)**2 + ((yy - 50.)/40.)**2))
    else:
        zz = np.random.rand(*xx.shape)
    z = zz.ravel()

    # generate 2D mesh points
    x_lim = [20., 70.]
    y_lim = [10., 90.]
    npoints = 120
    mesh_x = (x_lim[1] - x_lim[0])*np.random.rand(npoints) + x_lim[0]
    mesh_y = (y_lim[1] - y_lim[0])*np.random.rand(npoints) + y_lim[0]
    mesh_xy = np.vstack((mesh_x, mesh_y)).T

    # interpolate with scipy
    result = griddata(xy, z, mesh_xy, method='linear')

    # interpolate with GridInterpolator
    interp = GridInterpolator(xy, mesh_xy)
    result2 = interp(z)
    assert np.allclose(result, result2)

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolorfast(x, y, zz, cmap=plt.get_cmap('RdBu_r'))
        ax.plot(mesh_x, mesh_y, 'k.')
        plt.show()


@pytest.mark.parametrize('dataset', ['random', 'sin', 'gauss'])
def test_gridinterpolator(dataset):
    do_interpolation(dataset=dataset)


if __name__ == '__main__':
    do_interpolation(dataset='sin', plot=True)
