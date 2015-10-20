# Lock Exchange Test case
# =======================
#
# Solves hydrostatic flow in a closed rectangular channel.
#
# Dianeutral mixing depends on mesh Reynolds number [1]
# Re_h = U dx / nu
# U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(gH drho/rho_0)
# dx = horizontal mesh size
# nu = background viscosity
#
#
# Smagorinsky factor should be C_s = 1/sqrt(Re_h)
#
# Mesh resolutions:
# - ilicak [1]:  dx =  500 m,  20 layers
# COMODO lock exchange benchmark [2]:
# - coarse:      dx = 2000 m,  10 layers
# - coarse2 (*): dx = 1000 m,  20 layers
# - medium:      dx =  500 m,  40 layers
# - medium2 (*): dx =  250 m,  80 layers
# - fine:        dx =  125 m, 160 layers
# (*) not part of the original benchmark
#
# [1] Ilicak et al. (2012). Spurious dianeutral mixing and the role of
#     momentum closure. Ocean Modelling, 45-46(0):37-58.
#     http://dx.doi.org/10.1016/j.ocemod.2011.10.003
# [2] COMODO Lock Exchange test.
#     http://indi.imag.fr/wordpress/?page_id=446
#
# Tuomas Karna 2015-03-03

from cofs import *

# --- get run params from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('reso_str', type=str,
                    help='resolution string (coarse, medium, fine)')
parser.add_argument('-j', '--jumpDiffFactor', type=float, default=1.0,
                    help='factor for jump diff')
parser.add_argument('-l', '--useLimiter', action='store_true',
                    help='use slope limiter for tracers instead of diffusion')
parser.add_argument('-p', '--polyOrder', type=int, default=1,
                    help='order of finite element space (0|1)')
parser.add_argument('-m', '--mimetic', action='store_true',
                    help='use mimetic elements for velocity')
parser.add_argument('-Re', '--reynoldsNumber', type=float, default=2.0,
                    help='mesh Reynolds number for Smagorinsky scheme')
args = parser.parse_args()
if args.useLimiter:
    args.jumpDiffFactor = None
argsDict = vars(args)
if commrank == 0:
    print 'Running test case with setup:'
    for k in sorted(argsDict.keys()):
        print ' - {0:15s} : {1:}'.format(k, argsDict[k])

limiterStr = 'limiter' if args.useLimiter else 'jumpDiff'+str(args.jumpDiffFactor)
spaceStr = 'RT' if args.mimetic else 'DG'
outputDir = 'out_{:}_p{:}{:}_Re{:}_{:}'.format(args.reso_str, spaceStr,
                                               args.polyOrder,
                                               args.reynoldsNumber, limiterStr)

outputDir = createDirectory(outputDir)
reso_str = args.reso_str
if args.jumpDiffFactor is not None:
    args.jumpDiffFactor = Constant(args.jumpDiffFactor)

# ---

refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
              'medium2': 8, 'fine': 16, 'ilicak': 4}
# set mesh resolution
dx = 2000.0/refinement[reso_str]
layers = int(round(10*refinement[reso_str]))
if reso_str == 'ilicak':
    layers = 20

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
dt = 75.0/refinement[reso_str]
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
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers,
                              order=args.polyOrder, mimetic=args.mimetic)
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
solverObj.smagorinskyFactor = Constant(1.0/np.sqrt(args.reynoldsNumber))
solverObj.saltJumpDiffFactor = args.jumpDiffFactor
solverObj.saltRange = Constant(5.0)
solverObj.useLimiterForTracers = args.useLimiter
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
#solverObj.hViscosity = Constant(100.0/refinement[reso_str])
solverObj.hViscosity = Constant(1.0)
solverObj.hDiffusivity = Constant(1.0)
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
                            'w3d', 'wMesh3d', 'salt3d',
                            'uvDav2d', 'uvDav3d', 'baroHead3d',
                            'baroHead2d',
                            'smagViscosity', 'saltJumpDiff']
solverObj.fieldsToExportNumpy = ['salt3d']
solverObj.timerLabels = []

solverObj.createEquations()
salt_init3d = Function(solverObj.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression(('(x[0] > 0.0) ? 20.0 : 25.0')))
# smooth condition
salt_init3d.interpolate(Expression('22.5 - 2.5*tanh(x[0]/sigma)',
                                   sigma=1000.0))

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
