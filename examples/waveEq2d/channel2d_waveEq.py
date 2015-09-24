# Wave equation in 2D
# ===================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
# This example tests dispersion of surface waves and dissipation of time
# integrators.
#
# Tuomas Karna 2015-03-11
from cofs import *

mesh2d = Mesh('channel_waveEq.msh')
depth = 50.0
elev_amp = 1.0
# estimate of max advective velocity used to estimate time step
Umag = Constant(0.5)

outputDir = createDirectory('outputs_waveEq2d')
printInfo('Loaded mesh '+mesh2d.name)
printInfo('Exporting to '+outputDir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# Compute lenght of the domain
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = Lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
TExport = dt
T = 10*T_cycle + 1e-3

# --- create solver ---
solverObj = solver.flowSolver2d(mesh2d, bathymetry2d)
solverObj.cfl_2d = 1.0
solverObj.nonlin = False  # use linear wave equation
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.fieldsToExport = ['uv2d', 'elev2d']
solverObj.timerLabels = []
#solverObj.timeStepperType = 'SSPRK33'
#solverObj.dt = dt/40.0  # for explicit schemes
# solverObj.timeStepperType = 'CrankNicolson'
# solverObj.dt = 10.0  # override dt for CrankNicolson (semi-implicit)
solverObj.timeStepperType = 'SSPIMEX'
solverObj.dt = 100.0  # override dt for IMEX (semi-implicit)

# need to call creator to create the function spaces
solverObj.mightyCreator()
# set initial elevation to first standing wave mode
elev_init = Function(solverObj.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=elev_amp,
                             Lx=Lx))
solverObj.assignInitialConditions(elev=elev_init)
solverObj.iterate()
