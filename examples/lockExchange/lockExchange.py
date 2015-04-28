# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Tuomas Karna 2015-03-03

from cofs import *

# set physical constants
physical_constants['z0_friction'].assign(5.0e-5)

reso_str = 'coarse2'
outputDir = createDirectory('outputs_'+reso_str)
refinement = {'coarse': 1, 'coarse2': 2, 'medium': 4, 'fine': 16}
layers = 10*refinement[reso_str]
mesh2d = Mesh('mesh_{0:s}.msh'.format(reso_str))
print 'Loaded mesh', mesh2d.name
dt = 45.0/refinement[reso_str]
T = 70 * 3600
TExport = 15*60.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers)
solverObj.cfl_2d = 1.0
solverObj.nonlin = True
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = True
#solverObj.useModeSplit = False
solverObj.baroclinic = True
solverObj.useSUPG = False
solverObj.useGJV = False
#solverObj.uvLaxFriedrichs = None
# how does diffusion scale with mesh size?? nu = Lx^2/dt??
solverObj.hDiffusivity = Constant(100.0)
solverObj.hViscosity = Constant(100.0)
solverObj.vViscosity = Constant(1e-3)
if solverObj.useModeSplit:
    solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Constant(2.5)
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.checkSaltConservation = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'barohead3d',
                            'barohead2d', 'gjvAlphaH3d', 'gjvAlphaV3d']
solverObj.timerLabels = []

solverObj.mightyCreator()
salt_init3d = Function(solverObj.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression(('(x[0] > 0.0) ? 20.0 : 25.0')))
# smooth condition
salt_init3d.interpolate(Expression('22.5 - 2.5*tanh(x[0]/sigma)',
                                   sigma=1000.0))

solverObj.assingInitialConditions(salt=salt_init3d)
solverObj.iterate()
