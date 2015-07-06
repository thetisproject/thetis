# Wave equation in 2D
# ===================
#
# Rectangular channel geometry.
#
# Tuomas Karna 2015-03-11
from cofs import *
import cofs.timeIntegration as timeIntegration
import time as timeMod

# set physical constants
physical_constants['z0_friction'].assign(0.0)

mesh2d = Mesh('channel_waveEq.msh')
nonlin = False
depth = 20.0
outputDir = createDirectory('outputs_waveEq2d')
printInfo('Loaded mesh '+mesh2d.name)
printInfo('Exporting to '+outputDir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
P1v_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

# set time step, and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = Lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
TExport = dt
T = 10*T_cycle + 1e-3
# explicit model
Umag = Constant(0.5)

# --- create solver ---
solverObj = solver.flowSolver2d(mesh2d, bathymetry2d)
solverObj.cfl_2d = 1.0
solverObj.nonlin = False
solverObj.TExport = TExport
solverObj.T = T
solverObj.dt = dt
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.fieldsToExport = ['uv2d', 'elev2d']
solverObj.timerLabels = []
solverObj.timeStepperType = 'CrankNicolson'
#solverObj.timeStepperType = 'SSPRK33'

solverObj.mightyCreator()
elev_init = Function(solverObj.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=1.0,
                             Lx=Lx))
solverObj.assignInitialConditions(elev=elev_init)
solverObj.iterate()
