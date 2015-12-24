# Overflow Test case
# =======================
#
# Overflow test case from Haidvogel and Beckmann (1999)
#
# 200 km long channel with sloping bathymetry from 200 m to 4 km depth.
# Initially dense water is located on top of the slope.
#
# Horizontal resolution: 1 km
# Vertical layers: 40, 66, or 100 (Ilicak, 2012)
# Baroclinic/barotropic time steps: 10.0 s / 1.0 s
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
# Tuomas Karna 2015-06-10

from cofs import *

reso_str = 'medium'
outputdir = create_directory('outputs_'+reso_str)
refinement = {'medium': 1}
layers = int(round(16*refinement[reso_str]))
mesh2d = Mesh('mesh_{0:s}.msh'.format(reso_str))
print_info('Loaded mesh '+mesh2d.name)
dt = 5.0/refinement[reso_str]
T = 25 * 3600
TExport = 15*60.0
depth = 20.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.interpolate(Expression('hmin + 0.5*(hmax - hmin)*(1 + tanh((x[0] - x0)/Ls))',
                          hmin=200.0, hmax=4000.0, Ls=10.0e3, x0=40.0e3))

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry_2d, layers)
options = solverObj.options
options.cfl_2d = 1.0
# options.nonlin = False
options.mimetic = False
options.solveSalt = True
options.solveVertDiffusion = False
options.useBottomFriction = False
options.useALEMovingMesh = False
# options.useIMEX = True
# options.useSemiImplicit2D = False
# options.useModeSplit = False
options.baroclinic = True
options.uvLaxFriedrichs = None
options.tracerLaxFriedrichs = None
Re_h = 1.0
options.smagorinskyFactor = Constant(1.0/np.sqrt(Re_h))
options.salt_jump_diffFactor = None
options.saltRange = Constant(5.0)
options.useLimiterForTracers = True
options.hViscosity = None
options.hDiffusivity = None
if options.useModeSplit:
    options.dt = dt
options.TExport = TExport
options.T = T
options.outputdir = outputdir
options.uAdvection = Constant(5.0)
options.checkVolConservation2d = True
options.checkVolConservation3d = True
options.checkSaltConservation = True
options.checkSaltOvershoot = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                            'baroc_head_2d',
                            'smag_visc_3d', 'salt_jump_diff']
# options.fields_to_exportNumpy = ['salt_3d']
options.timerLabels = []

solverObj.createEquations()
salt_init3d = Function(solverObj.function_spaces.H, name='initial salinity')
# vertical barrier
# salt_init3d.interpolate(Expression(('(x[0] > 20.0e3) ? 0.0 : 2.0')))
# smooth condition
salt_init3d.interpolate(Expression('drho*0.5*(1.0 - tanh((x[0] - x0)/sigma))',
                                   drho=2.0, x0=20.0e3, sigma=1000.0))

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate()
