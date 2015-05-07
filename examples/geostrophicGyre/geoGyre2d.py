# Geostrophic gyre test case
# ==========================
#
# Setup is according to Comblen et al. (2010).
#
# Tuomas Karna 2015-04-28

from cofs import *
import cofs.timeIntegration as timeIntegration
import time as timeMod

# set physical constants
physical_constants['z0_friction'].assign(0.0)

mesh2d = Mesh('stommel_square.msh')
nonlin = False
depth = 1000.0
elev_amp = 3.0
outputDir = createDirectory('outputs')
T = 75*12*2*3600
TExport = 3600*2

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# Coriolis forcing
coriolis2d = Function(P1_2d)
f0, beta = 1.0e-4, 0.0
coriolis2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0)
    )

# --- create solver ---
solverObj = solver.flowSolver2d(mesh2d, bathymetry2d)
solverObj.cfl_2d = 1.0
solverObj.nonlin = False
solverObj.coriolis = coriolis2d
solverObj.TExport = TExport
solverObj.T = T
solverObj.dt = 20.0
solverObj.outputDir = outputDir
solverObj.uAdvection = Constant(0.01)
solverObj.checkVolConservation2d = True
solverObj.fieldsToExport = ['uv2d', 'elev2d']
solverObj.timerLabels = []
solverObj.timeStepperType = 'CrankNicolson'

solverObj.mightyCreator()
sigma = 160.0e3
elev_init = Function(solverObj.H_2d)
elev_init.project(Expression('eta_amp*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)', eta_amp=elev_amp, s=sigma))

# initial velocity: u = -g/f deta/dy, v = g/f deta/dx
uv_init = Function(solverObj.U_2d)
uv_init.project(Expression(('g/f*eta_amp*2*x[1]/s/s*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)',
                            '-g/f*eta_amp*2*x[0]/s/s*exp(-(x[0]*x[0]+x[1]*x[1])/s/s)'), eta_amp=elev_amp, s=sigma, g=9.81, f=f0))

solverObj.assignInitialConditions(elev=elev_init, uv_init=uv_init)

solverObj.iterate()
