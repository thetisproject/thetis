# Test vertical velocity computation
# ==================================
#
# Solves 3D continuity equation for simple horzontal velocity fields.
#
# Tuomas Karna 2015-07-10
from cofs import *


def test1():
    # ---- test 1: constant bathymetry
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = UnitSquareMesh(10, 10)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name='Bathymetry')
    bathymetry2d.assign(1.0)

    # create solver
    solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers)
    solverObj.uAdvection = Constant(1e-3)
    solverObj.TExport = 100.0
    solverObj.T = 100.0
    solverObj.outputDir = 'tmp'

    solverObj.mightyCreator()
    # w needs to be projected to cartesian vector field for sanity check
    w3d_proj = Function(solverObj.P1DGv, name='projected w')
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}

    solverObj.uv3d.project(Expression(('1e-3', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    w3d_proj.project(solverObj.w3d)
    print 'w', w3d_proj.dat.data.min(), w3d_proj.dat.data.max()
    assert(np.allclose(w3d_proj.dat.data, 0.0))
    print 'PASSED'

    solverObj.uv3d.project(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d)
    w3d_proj.project(solverObj.w3d)
    print 'w', w3d_proj.dat.data.min(), w3d_proj.dat.data.max()
    assert(np.allclose(w3d_proj.dat.data.min(), -1e-3))
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
    # w needs to be projected to cartesian vector field for sanity check
    w3d_proj = Function(solverObj.P1DGv, name='projected w')
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}

    solverObj.uv3d.project(Expression(('1e-3', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    w3d_proj.project(solverObj.w3d)
    print 'w', w3d_proj.dat.data.min(), w3d_proj.dat.data.max()
    assert(np.allclose(w3d_proj.dat.data[:, 2], -1e-3))
    print 'PASSED'
    solverObj.exporter.export()

    solverObj.uv3d.project(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.w3d, solverObj.uv3d, solverObj.bathymetry3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    w3d_proj.project(solverObj.w3d)
    print 'w', w3d_proj.dat.data.min(), w3d_proj.dat.data.max()
    assert(np.allclose(w3d_proj.dat.data.min(), -3e-3, rtol=1e-2))
    print 'PASSED'
    solverObj.exporter.export()
    linProblemCache.clear()

test1()
test2()
