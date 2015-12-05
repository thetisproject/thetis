# Tracer box in 3D
# ================
#
# Solves a standing wave in a rectangular basin using wave equation.
#
# This version uses the ALE moving mesh and a constant tracer to check
# tracer local/global tracer conservation.
# NOTE ALE tracer conservation is currently broken
#
# Initial condition for elevation corresponds to a standing wave.
# Time step and export interval are chosen based on theorethical
# oscillation frequency. Initial condition repeats every 20 exports.
#
#
# Tuomas Karna 2015-03-11
from cofs import *

mesh2d = Mesh('channel_waveEq.msh')
depth = 50.0
elev_amp = 1.0
n_layers = 6
# estimate of max advective velocity used to estimate time step
Umag = Constant(0.5)
sloped = True

suffix = ''
if sloped:
    suffix = '_sloped'
outputDir = createDirectory('outputs'+suffix)

printInfo('Loaded mesh '+mesh2d.name)
printInfo('Exporting to '+outputDir)

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# Compute lenght of the domain
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, op=MPI.MAX)
Lx = x_max - x_min

if sloped:
    bathymetry_2d.interpolate(Expression('h + 8.0*x[0]/Lx', h=depth, Lx=Lx))

# set time step, export interval and run duration
c_wave = float(np.sqrt(9.81*depth))
T_cycle = Lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
TExport = dt
T = 10*T_cycle + 1e-3

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry_2d, n_layers)
options = solverObj.options
options.nonlin = False
#options.mimetic = False
options.solveSalt = True
options.solveVertDiffusion = False
options.useBottomFriction = False
options.useALEMovingMesh = False
# options.useSemiImplicit2D = False
# options.useModeSplit = False
if options.useModeSplit:
    options.dt = dt/5.0
else:
    options.dt = dt/40.0
options.TExport = TExport
options.T = T
options.uAdvection = Umag
options.checkVolConservation2d = True
options.checkVolConservation3d = True
options.checkSaltConservation = True
options.checkSaltDeviation = True
options.outputDir = outputDir
options.timerLabels = []
#options.timerLabels = ['mode2d', 'momentumEq', 'continuityEq',
                         #'aux_functions']
options.fieldsToExport = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                          'w_3d', 'w_mesh_3d', 'salt_3d',
                          'uv_dav_2d', 'uv_bottom_2d']

# need to call creator to create the function spaces
solverObj.createEquations()
elev_init = Function(solverObj.function_spaces.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=elev_amp,
                             Lx=Lx))
if options.solveSalt:
    salt_init3d = Function(solverObj.function_spaces.H, name='initial salinity')
    salt_init3d.assign(4.5)
else:
    salt_init3d = None

solverObj.assignInitialConditions(elev=elev_init, salt=salt_init3d)
solverObj.iterate()
