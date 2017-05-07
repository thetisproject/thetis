"""
Test LinearTimeInterpolator object
"""

from interpolation import *
import numpy as np
from scipy.interpolate import interp1d


def do_interpolation(plot=False):
    np.random.seed(2)

    # construct data set
    x_scale = 100.
    ndata = 35
    xx = np.linspace(0, x_scale, ndata)
    yy = np.random.rand(*xx.shape)

    # construct interpolation points
    ninterp = 100
    x_interp = np.random.rand(ninterp)*x_scale

    # get correct solution with scipy
    y_interp = interp1d(xx, yy)(x_interp)

    class TimeSeriesReader(DBReader):
        def __init__(self, y):
            self.y = y

        def __call__(self, descriptor, time_index):
            return self.y[time_index]

    class SimpleTimeSearch(TimeSearch):
        def __init__(self, t):
            self.t = t

        def find(self, time, previous=False):
            ix = np.searchsorted(self.t, time)
            if previous:
                ix -= 1
            if ix < 0:
                raise Exception('Index out of bounds')
            tstamp = self.t[ix]
            return ('cat', ix, tstamp)

    timesearch_obj = SimpleTimeSearch(xx)
    reader = TimeSeriesReader(yy)
    lintimeinterp = LinearTimeInterpolator(timesearch_obj, reader)
    y_interp2 = np.zeros_like(y_interp)
    for i in range(len(y_interp2)):
        y_interp2[i] = lintimeinterp(x_interp[i])

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(xx, yy, 'k')
        plt.plot(x_interp, y_interp, 'bo')
        plt.plot(x_interp, y_interp2, 'rx')
        plt.show()

    assert np.allclose(y_interp, y_interp2)


def test_linearinterpolator():
    do_interpolation()


if __name__ == '__main__':
    do_interpolation(plot=True)
