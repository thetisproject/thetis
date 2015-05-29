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
# simulation period: 144 h

# Tuomas Karna 2015-03-03

from cofs import *

# set physical constants
physical_constants['rho0'].assign(1025.0)

outputDir = createDirectory('outputs')
layers = 10
mesh2d = Mesh('tartinville_physical.msh')
print 'Loaded mesh', mesh2d.name
dt = 60.0
T = 288 * 3600
TExport = 30*60.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

coriolis2d = Function(P1_2d)
f0, beta = 1.15e-4, 0.0
coriolis2d.interpolate(
    Expression('f0+beta*(x[1]-y_0)', f0=f0, beta=beta, y_0=0.0)
    )

# create solver
solverObj = solver.flowSolverMimetic(mesh2d, bathymetry2d, layers)
solverObj.cfl_2d = 1.0
#solverObj.nonlin = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = False
#solverObj.useSemiImplicit2D = False
#solverObj.useModeSplit = False
solverObj.baroclinic = True
solverObj.coriolis = coriolis2d
#solverObj.uvLaxFriedrichs = None
# how does diffusion scale with mesh size?? nu = Lx^2/dt??
#solverObj.hDiffusivity = Constant(100.0/refinement[reso_str])
#solverObj.hViscosity = Constant(100.0/refinement[reso_str])
#solverObj.vViscosity = Constant(1e-5/refinement[reso_str])
if solverObj.useModeSplit:
    solverObj.dt = dt
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Constant(0.25)
solverObj.checkVolConservation2d = True
solverObj.checkVolConservation3d = True
solverObj.checkSaltConservation = True
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'uv2d_dav', 'uv3d_dav', 'barohead3d',
                            'barohead2d', 'gjvAlphaH3d', 'gjvAlphaV3d']
solverObj.timerLabels = []

solverObj.mightyCreator()
# assign initial salinity
salt_init3d = Function(solverObj.H, name='initial salinity')
# interpolate on P1 field to circumvent overshoots
# impose rho' = rho - 1025.0
tmp = Function(solverObj.P1, name='initial salinity')
tmp.interpolate(Expression('0.78*1.1*pow((sqrt(x[0]*x[0] + x[1]*x[1])/1000/3 + (1.0-tanh(10*(x[2]+10.0)))*0.5), 8)'))
# crop bad values
ix = tmp.dat.data[:] > 0.858
tmp.dat.data[ix] = 0.858
salt_init3d.project(tmp)

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
