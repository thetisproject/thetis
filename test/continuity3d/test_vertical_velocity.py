# Test vertical velocity computation
# ==================================
#
# Solves 3D continuity equation for simple horzontal velocity fields.
#
# Tuomas Karna 2015-07-10
from cofs import *


def setup1(mimetic=False):
    """
    Compute vertical velocity in case of constant
    """
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = UnitSquareMesh(10, 10)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.assign(1.0)

    # create solver
    solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
    solverObj.options.mimetic = mimetic
    solverObj.options.uAdvection = Constant(1e-3)
    solverObj.options.TExport = 100.0
    solverObj.options.T = 100.0
    solverObj.options.dt = 30.0
    solverObj.options.dt_2d = 10.0
    solverObj.options.outputDir = outputDir

    solverObj.createEquations()
    # w needs to be projected to cartesian vector field for sanity check
    w_3d_proj = Function(solverObj.function_spaces.P1DGv, name='projected w')
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}

    solverObj.fields.uv_3d.project(Expression(('1e-3', '0.0', '0.0')))
    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    w_3d_proj.project(solverObj.fields.w_3d)
    print 'w', w_3d_proj.dat.data.min(), w_3d_proj.dat.data.max()
    assert(np.allclose(w_3d_proj.dat.data, 0.0))
    print 'PASSED'

    solverObj.fields.uv_3d.project(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d)
    w_3d_proj.project(solverObj.fields.w_3d)
    print 'w', w_3d_proj.dat.data.min(), w_3d_proj.dat.data.max()
    assert(np.allclose(w_3d_proj.dat.data.min(), -1e-3))
    print 'PASSED'
    linProblemCache.clear()


def setup2(mimetic=False):
    """
    Compute vertical velocity in case of sloping bathymetry
    """
    n_layers = 6
    outputDir = createDirectory('outputs')
    mesh2d = UnitSquareMesh(10, 10)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.interpolate(Expression('1.0 + x[0]'))

    solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
    solverObj.options.mimetic = mimetic
    solverObj.options.uAdvection = Constant(1e-3)
    solverObj.options.TExport = 100.0
    solverObj.options.T = 100.0
    solverObj.options.dt = 30.0
    solverObj.options.dt_2d = 10.0
    solverObj.options.outputDir = outputDir

    solverObj.createEquations()
    # w needs to be projected to cartesian vector field for sanity check
    w_3d_proj = Function(solverObj.function_spaces.P1DGv, name='projected w')
    # use symmetry condition at all boundaries
    bnd_markers = solverObj.eq_sw.boundary_markers
    bnd_funcs = {}
    for k in bnd_markers:
        bnd_funcs[k] = {'symm': None}

    # NOTE need to project the full 3d field on uv
    # NOTE is it possible to use a single uvw field instead of uv and w?
    solverObj.fields.uv_3d.project(Expression(('1e-3', '0.0', '-1.0e-3')))
    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    solverObj.export()
    # in the case of mimetic elements real w is the vertical component of (uv + w)
    w_3d_proj.project(solverObj.fields.w_3d)
    print '   w', w_3d_proj.dat.data[:, 2].min(), w_3d_proj.dat.data[:, 2].max()
    w_3d_proj.project(solverObj.fields.w_3d + solverObj.fields.uv_3d)
    f = File('outputs/w.pvd')
    f << w_3d_proj
    print 'uv+w', w_3d_proj.dat.data[:, 2].min(), w_3d_proj.dat.data[:, 2].max()
    assert(np.allclose(w_3d_proj.dat.data[:, 2], -1e-3, rtol=3e-1))
    print 'PASSED'

    solverObj.fields.uv_3d.project(Expression(('1e-3*x[0]', '0.0', '0.0')))
    computeVertVelocity(solverObj.fields.w_3d, solverObj.fields.uv_3d, solverObj.fields.bathymetry_3d,
                        boundary_markers=bnd_markers, boundary_funcs=bnd_funcs)
    solverObj.export()
    w_3d_proj.project(solverObj.fields.w_3d)
    print '   w', w_3d_proj.dat.data[:, 2].min(), w_3d_proj.dat.data[:, 2].max()
    w_3d_proj.project(solverObj.fields.w_3d + solverObj.fields.uv_3d)
    print 'uv+w', w_3d_proj.dat.data[:, 2].min(), w_3d_proj.dat.data[:, 2].max()
    assert(np.allclose(w_3d_proj.dat.data.min(), -3e-3, rtol=3e-1))
    print 'PASSED'
    linProblemCache.clear()


def test_setup1_dg():
    setup1(mimetic=False)


def test_setup2_dg():
    setup2(mimetic=False)


def test_setup1_rt():
    setup1(mimetic=True)


def test_setup2_rt():
    setup2(mimetic=True)


if __name__ == '__main__':
    #test_setup1_dg()
    #test_setup1_rt()

    #test_setup2_dg()
    test_setup2_rt()
