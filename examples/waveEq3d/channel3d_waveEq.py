# Wave equation in 2D
# ===================
#
# Rectangular channel geometry.
#
# Tuomas Karna 2015-03-11

from scipy.interpolate import interp1d
from cofs.utility import *
from cofs.physical_constants import physical_constants
import cofs.timeIntegration as timeIntegration
import cofs.solver as solverMod

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# set physical constants
physical_constants['z0_friction'].assign(0.0)

use_wd = False
nonlin = False
n_layers = 6
outputDir = createDirectory('outputs_waveEq')
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
solver = solverMod.flowSolver(mesh2d, bathymetry2d, n_layers)
solver.nonlin = nonlin
solver.use_wd = use_wd
solver.solveSalt = False
solver.solveVertDiffusion = False
solver.useBottomFriction = False
solver.useALEMovingMesh = False
solver.dt = dt
solver.TExport = TExport
solver.T = T
solver.uAdvection = Umag
solver.checkVolConservation2d = True
solver.checkVolConservation3d = True
solver.timerLabels = ['mode2d', 'momentumEq', 'continuityEq', 'aux_functions']
solver.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                         'w3d', 'w3d_mesh', 'salt3d',
                         'uv2d_dav', 'uv2d_bot', 'nuv3d']


solver.mightyCreator()
elev_init = Function(solver.H_2d)
elev_init.project(Expression('-eta_amp*cos(2*pi*x[0]/Lx)', eta_amp=elev_amp,
                             Lx=Lx))
salt_init3d = Function(solver.H, name='initial salinity')
salt_init3d.interpolate(Expression('x[0]/1.0e5*10.0+2.0'))

solver.assingInitialConditions(elev=elev_init, salt=salt_init3d)
solver.iterate()
