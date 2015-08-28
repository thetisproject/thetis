# Lock Exchange Test case
# =======================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Dianeutral mixing depends on mesh Reynolds number (Ilicak et al. 2012)
# Re_h = U dx / nu
# U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(gH drho/rho_0)
# dx = horizontal mesh size
# nu = background viscosity
#
# For coarse mesh:
# Re_h = 0.5 2000 / 100 = 10
#
# TODO run medium for Re_h = 250
# => nu = 0.5 500 / 250 = 1.0
#
# Smagorinsky factor should be C_s = 1/sqrt(Re_h)
#
# Tuomas Karna 2015-03-03

from cofs import *

# set physical constants
physical_constants['z0_friction'].assign(5.0e-5)

reso_str = 'coarse2'
outputDir = createDirectory('outputs_struct_'+reso_str)
refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
              'medium2': 8, 'fine': 16}
# set mesh resolution
dx = 2000.0/refinement[reso_str]
layers = int(round(10*refinement[reso_str]))
# generate unit mesh and transform its coords
x_max = 32.0e3
x_min = -32.0e3
n_x = (x_max - x_min)/dx
mesh2d = UnitSquareMesh(n_x, 2)
coords = mesh2d.coordinates
# x in [x_min, x_max], y in [-dx, dx]
coords.dat.data[:, 0] = coords.dat.data[:, 0]*(x_max - x_min) + x_min
coords.dat.data[:, 1] = coords.dat.data[:, 1]*2*dx - dx

printInfo('Exporting to '+outputDir)
dt = 100.0/refinement[reso_str]
if reso_str == 'fine':
    dt /= 2.0
T = 25 * 3600
TExport = 15*60.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers)
solverObj.cfl_2d = 1.0
#solverObj.nonlin = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = False
#solverObj.useSemiImplicit2D = False
#solverObj.useModeSplit = False
solverObj.baroclinic = True
solverObj.uvLaxFriedrichs = Constant(1.0)
solverObj.tracerLaxFriedrichs = Constant(1.0)
Re_h = 2.0
solverObj.smagorinskyFactor = Constant(1.0/np.sqrt(Re_h))
solverObj.saltJumpDiffFactor = None  # Constant(1.0)
solverObj.saltRange = Constant(5.0)
solverObj.useLimiterForTracers = True
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
#solverObj.hViscosity = Constant(100.0/refinement[reso_str])
#solverObj.hViscosity = Constant(10.0)
if solverObj.useModeSplit:
    solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Constant(1.0)
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.checkSaltConservation = True
solverObj.checkSaltOvershoot = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'uv3d_dav', 'barohead3d',
                            'barohead2d',
                            'smagViscosity', 'saltJumpDiff']
solverObj.fieldsToExportNumpy = ['salt3d']
solverObj.timerLabels = []

solverObj.mightyCreator()
salt_init3d = Function(solverObj.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression(('(x[0] > 0.0) ? 20.0 : 25.0')))
# smooth condition
salt_init3d.interpolate(Expression('22.5 - 2.5*tanh(x[0]/sigma)',
                                   sigma=1000.0))

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
