# Test vertical velocity computation
# ==================================
#
# Solves 3D continuity equation for simple horzontal velocity fields.
#
# Tuomas Karna 2015-03-03

from scipy.interpolate import interp1d
from cofs import *


def test1():
    # ---- test 1: constant bathymetry
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = UnitSquareMesh(10,10)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name='Bathymetry')
    bathymetry2d.interpolate(Expression('1.0'))

    # create solver
    solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers)
    solverObj.uAdvection = Constant(1e-3)
    solverObj.TExport = 100.0
    solverObj.T = 100.0
    solverObj.outputDir = 'tmp'

    solverObj.mightyCreator()

    solverObj.uv3d.interpolate(Expression(('1e-3', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d)
    print 'w', solverObj.w3d.dat.data.min(), solverObj.w3d.dat.data.max()
    assert(np.allclose(solverObj.w3d.dat.data, 0.0))
    print 'PASSED'

    solverObj.uv3d.interpolate(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d)
    print 'w', solverObj.w3d.dat.data.min(), solverObj.w3d.dat.data.max()
    assert(np.allclose(solverObj.w3d.dat.data.min(), -1e-3))
    print 'PASSED'
    linProblemCache.clear()

def test2():
    # ---- test 2: sloping bathymetry
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = UnitSquareMesh(10,10)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name='Bathymetry')
    bathymetry2d.interpolate(Expression('1.0 + x[0]'))

    solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers)
    solverObj.uAdvection = Constant(1e-3)
    solverObj.TExport = 100.0
    solverObj.T = 100.0
    solverObj.outputDir = 'tmp'

    solverObj.mightyCreator()

    solverObj.uv3d.interpolate(Expression(('1e-3', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d)
    print 'w', solverObj.w3d.dat.data.min(), solverObj.w3d.dat.data.max()
    assert(np.allclose(solverObj.w3d.dat.data, -1e-3))
    print 'PASSED'
    solverObj.exporter.export()

    solverObj.uv3d.interpolate(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d)
    print 'w', solverObj.w3d.dat.data.min(), solverObj.w3d.dat.data.max()
    assert(np.allclose(solverObj.w3d.dat.data.min(), -3e-3, rtol=1e-2))
    print 'PASSED'
    solverObj.exporter.export()
    linProblemCache.clear()

test1()
test2()
