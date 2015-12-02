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

parameters['coffee'] = {}

reso_str = 'coarse'
outputDir = createDirectory('outputs_struct_' + reso_str)
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
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry_2d, layers)
options = solverObj.options
options.cfl_2d = 1.0
#options.nonlin = False
options.mimetic = False
options.solveSalt = True
options.solveVertDiffusion = False
options.useBottomFriction = False
options.useALEMovingMesh = False
#options.useIMEX = True
#options.useSemiImplicit2D = False
#options.useModeSplit = False
options.baroclinic = True
options.uvLaxFriedrichs = Constant(1.0)
options.tracerLaxFriedrichs = Constant(1.0)
Re_h = 1.0
options.smagorinskyFactor = Constant(1.0/np.sqrt(Re_h))
options.salt_jump_diffFactor = None  # Constant(1.0)
options.saltRange = Constant(5.0)
options.useLimiterForTracers = True
# To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
#options.hViscosity = Constant(100.0/refinement[reso_str])
options.hViscosity = Constant(1.0)
options.hDiffusivity = Constant(1.0)
if options.useModeSplit:
    options.dt = dt
options.TExport = TExport
options.T = T
options.outputDir = outputDir
options.uAdvection = Constant(1.0)
options.checkVolConservation2d = True
options.checkVolConservation3d = True
options.checkSaltConservation = True
options.checkSaltOvershoot = True
options.fieldsToExport = ['uv_2d', 'elev_2d', 'uv_3d',
                          'w_3d', 'w_mesh_3d', 'salt_3d',
                          'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                          'baroc_head_2d',
                          'smag_visc_3d', 'salt_jump_diff']
options.fieldsToExportNumpy = ['salt_3d']
options.timerLabels = []

solverObj.createEquations()
salt_init3d = Function(solverObj.function_spaces.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression(('(x[0] > 0.0) ? 20.0 : 25.0')))
# smooth condition
salt_init3d.interpolate(Expression('22.5 - 2.5*tanh(x[0]/sigma)',
                                   sigma=1000.0))

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
