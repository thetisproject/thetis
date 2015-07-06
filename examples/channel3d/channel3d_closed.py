# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Tuomas Karna 2015-03-03

from scipy.interpolate import interp1d
from cofs import *

n_layers = 6
outputDir = createDirectory('outputs_closed')
mesh2d = Mesh('channel_mesh.msh')
printInfo('Loaded mesh '+mesh2d.name)
printInfo('Exporting to '+outputDir)
T = 48 * 3600
Umag = Constant(4.2)
TExport = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 7.0
bathymetry2d.interpolate(Expression('ho - (ho-hr)*x[0]/100e3',
                                    ho=depth_oce, hr=depth_riv))
#bathymetry2d.interpolate(Expression('ho - (ho-hr)*0.5*(1+tanh((x[0]-50e3)/15e3))',
                                    #ho=depth_oce, hr=depth_riv))

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers, order=1)
#solverObj.nonlin = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = False
solverObj.uvLaxFriedrichs = Constant(1.0)
solverObj.tracerLaxFriedrichs = Constant(1.0)
#solverObj.useSemiImplicit2D = False
#solverObj.useModeSplit = False
#solverObj.baroclinic = True
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.checkSaltConservation = True
solverObj.checkSaltDeviation = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'uv2d_bot', 'nuv3d']
solverObj.timerLabels = []

# initial conditions
elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])


def elevation(x, y, z, x_array, val_array):
    padval = 1e20
    x0 = np.hstack(([-padval], x_array, [padval]))
    vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
    return interp1d(x0, vals0)(x)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
elev_init = Function(P1_2d)
elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0,
                                  elev_x, elev_v)
salt_init3d = Constant(4.5)
#solverObj.mightyCreator()
#salt_init3d = Function(solverObj.P1).interpolate(Expression('4.5*0.5*(1.0 - tanh((x[0]-20.0e3)/5000.0))'))

solverObj.assignInitialConditions(elev=elev_init, salt=salt_init3d)
solverObj.iterate()
