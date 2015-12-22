# Geostrophic freshwater cylinder test case
# =========================================
#
# For detailed description and discussion of the test case see
# [1] Tartinville et al. (1998). A coastal ocean model intercomparison study
#     for a three-dimensional idealised test case. Applied Mathematical
#     Modelling, 22(3):165-182.
#     http://dx.doi.org/10.1016/S0307-904X(98)00015-8
#
# Test case setup:
# domain: 30 km x 30 km, 20 m deep
# mesh resolution: 1 km, 20 vertical levels
# coriolis: f=1.15e-4 1/s
# initial salinity: cylinder
#    center: center of domain
#    radius: 3 km
#    depth: surface to 10 m deep
#    salinity inside: 1.1*(r/1000/3)^8 + 33.75 psu
#       (r radial distance in m)
#    salinity outside: 34.85 psu
# equation of state: 1025 + 0.78*(S - 33.75)
# density inside: rho = 1025 + 0.78*1.1*(r/1000/3)^8
# density outside: 1025 + 0.78*1.1 = 1025.858
# initial elevation: zero
# initial velocity: zero
# inertial period: 144 h / 9.5 = 54568.42 s ~= 30 exports
# simulation period: 144 h
#
# S contours are 33.8, 34.0, 34.2, 34.4, 34.6, 34.8
# which correspond to rho' 0.039,  0.195,  0.351,  0.507,  0.663,  0.819
#
# NOTE with SLIM mode-2 instability starts to develop around t=100 h
#
# Tuomas Karna 2015-05-30

from cofs import *

# set physical constants
physical_constants['rho0'].assign(1025.0)

outputdir = create_directory('outputs')
layers = 20
mesh2d = Mesh('tartinville_physical.msh')
print_info('Loaded mesh ' + mesh2d.name)
dt = 25.0
T = 288 * 3600
TExport = 900.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

coriolis_2d = Function(P1_2d)
f0, beta = 1.15e-4, 0.0
coriolis_2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0))

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry_2d, layers)
options = solverObj.options
options.cfl_2d = 1.0
# options.nonlin = False
options.solveSalt = True
options.solveVertDiffusion = False
options.useBottomFriction = False
options.useALEMovingMesh = False
options.useSemiImplicit2D = False
# options.useModeSplit = False
options.baroclinic = True
options.coriolis = coriolis_2d
options.uvLaxFriedrichs = None  # Constant(1e-3)
options.tracerLaxFriedrichs = None  # Constant(1e-3)
options.useLimiterForTracers = True
Re_h = 2.0
options.smagorinskyFactor = Constant(1.0/np.sqrt(Re_h))
# how does diffusion scale with mesh size?? nu = Lx^2/dt??
# options.hDiffusivity = Constant(3.0)
# options.hViscosity = Constant(1e-2)
# options.vViscosity = Constant(1e-5)
if options.useModeSplit:
    options.dt = dt
options.TExport = TExport
options.T = T
options.outputdir = outputdir
options.uAdvection = Constant(1.5)
options.checkVolConservation2d = True
options.checkVolConservation3d = True
options.checkSaltConservation = True
options.checkSaltOvershoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                          'w_3d', 'w_mesh_3d', 'salt_3d',
                          'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                          'baroc_head_2d']
options.fields_to_exportNumpy = ['salt_3d', 'baroc_head_3d', 'elev_2d']
options.timerLabels = ['mode2d', 'momentumEq', 'continuityEq', 'saltEq',
                       'aux_barolinicity', 'aux_mom_coupling',
                       'func_copy2dTo3d', 'func_copy3dTo2d', ]

solverObj.createEquations()
# assign initial salinity
salt_init3d = Function(solverObj.function_spaces.H, name='initial salinity')
# interpolate on P1 field to circumvent overshoots
# impose rho' = rho - 1025.0
tmp = Function(solverObj.function_spaces.P1, name='initial salinity')
tmp.interpolate(Expression('0.78*1.1*pow((sqrt(x[0]*x[0] + x[1]*x[1])/1000/3 + (1.0-tanh(10*(x[2]+10.0)))*0.5), 8)'))
# crop bad values
ix = tmp.dat.data[:] > 0.858
tmp.dat.data[ix] = 0.858
salt_init3d.project(tmp)

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
