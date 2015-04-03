# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Tuomas Karna 2015-03-03

from scipy.interpolate import interp1d
from cofs import *

op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# set physical constants
physical_constants['z0_friction'].assign(5.0e-5)

reso_str = 'coarse'
outputDir = createDirectory('outputs_'+reso_str)
layers = {'coarse': 10, 'medium': 40, 'fine': 160}
refinement = {'coarse': 1, 'medium': 4, 'fine': 16}
mesh2d = Mesh('mesh_{0:s}.msh'.format(reso_str))
dt = 450.0/refinement[reso_str]
T = 70 * 3600
TExport = 15*60.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers[reso_str])
solverObj.nonlin = False
solverObj.use_wd = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = True
solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Constant(0.15)
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav']

solverObj.mightyCreator()
salt_init3d = Function(solverObj.H, name='initial salinity')
salt_init3d.interpolate(Expression(('(x[0] > 0.0) ? 1025 : 1020')))

solverObj.assingInitialConditions(salt=salt_init3d)
solverObj.iterate()
