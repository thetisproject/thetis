# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a rectangular channel.
#
# Tuomas Karna 2015-03-03
from scipy.interpolate import interp1d
from cofs import *

# set physical constants
physical_constants['z0_friction'].assign(5.0e-5)

use_wd = False
nonlin = True
n_layers = 6
outputDir = createDirectory('outputs')
mesh2d = Mesh('channel_mesh.msh')
T = 48 * 3600
Umag = Constant(6.0)  # 4.2 closed
TExport = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 5.0  # 5.0 closed
bath_x = np.array([0, 100e3])
bath_v = np.array([depth_oce, depth_riv])


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
bathymetry2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

# --- create solver ---
solverObj = solver.flowSolver2dMimetic(mesh2d, bathymetry2d, order=1)
solverObj.cfl_2d = 1.0
#solverObj.nonlin = False
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.fieldsToExport = ['uv2d', 'elev2d']
solverObj.timerLabels = []
#solverObj.timeStepperType = 'CrankNicolson'
solverObj.timeStepperType = 'SSPRK33'

solverObj.mightyCreator()
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
solverObj.assignInitialConditions(elev=elev_init)

solverObj.iterate()
