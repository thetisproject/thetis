# Wave equation in 2D
# ===================
#
# Rectangular channel geometry.
#
# Tuomas Karna 2015-03-11

from scipy.interpolate import interp1d
from cofs import *

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# set physical constants
physical_constants['z0_friction'].assign(0.0)

n_layers = 6
outputDir = createDirectory('outputs')
mesh2d = Mesh('channel_waveEq.msh')
Umag = Constant(0.5)
depth = 50.0
elev_amp = 1.0

# Function spaces for 2d mode
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
# Mean free surface height (bathymetry)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

# compute time step that matches the oscillation period
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

c_wave = float(np.sqrt(9.81*depth))
T_cycle = Lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
TExport = dt
T = 10*T_cycle + 1e-3

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers)
solverObj.nonlin = False
solverObj.use_wd = False
solverObj.solveSalt = False
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = False
solverObj.useModeSplit = False
#solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.uAdvection = Umag
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.timerLabels = []
#solverObj.timerLabels = ['mode2d', 'momentumEq', 'continuityEq',
                         #'aux_functions']
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'uv2d_bot', 'nuv3d']

solverObj.mightyCreator()
elev_init = Function(solverObj.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=elev_amp,
                             Lx=Lx))
salt_init3d = Function(solverObj.H, name='initial salinity')
salt_init3d.interpolate(Expression('x[0]/1.0e5*10.0+2.0'))

solverObj.assingInitialConditions(elev=elev_init, salt=salt_init3d)
solverObj.iterate()
