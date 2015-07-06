# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a rectangular channel forced by tides.
#
# Tuomas Karna 2015-03-03

from scipy.interpolate import interp1d
from cofs import *

n_layers = 6
outputDir = createDirectory('outputs')
mesh2d = Mesh('channel_mesh.msh')
printInfo('Loaded mesh '+mesh2d.name)
printInfo('Exporting to '+outputDir)
T = 48 * 3600
Umag = Constant(2.5)
TExport = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 7.0
bathymetry2d.interpolate(Expression('ho - (ho-hr)*x[0]/100e3',
                                    ho=depth_oce, hr=depth_riv))

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, n_layers)
#solverObj.nonlin = False
solverObj.solveSalt = True
solverObj.solveVertDiffusion = False
solverObj.useBottomFriction = False
solverObj.useALEMovingMesh = False
solverObj.uvLaxFriedrichs = Constant(1.0)
solverObj.tracerLaxFriedrichs = Constant(1.0)
#solverObj.useSemiImplicit2D = False
#solverObj.useModeSplit = False
#solverObj.baroclinic = True
solverObj.TExport = TExport
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkSaltDeviation = True
solverObj.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion']
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                            'w3d', 'w3d_mesh', 'salt3d',
                            'barohead3d', 'barohead2d',
                            'uv2d_dav', 'uv2d_bot', 'nuv3d']

# initial conditions
salt_init3d = Constant(4.5)


# weak boundary conditions
L_y = 1900
h_amp = 2.0
un_amp = -2.0
flux_amp = L_y*depth_oce*un_amp
h_T = 12 * 3600  # 44714.0
un_river = -0.3
flux_river = L_y*depth_riv*un_river
t = 0.0
T_ramp = 1000.0
ocean_elev_func = lambda t: h_amp * sin(2 * pi * t / h_T)  # + 3*pi/2)
ocean_elev = Constant(ocean_elev_func(t))
ocean_un_func = lambda t: (un_amp*sin(2 * pi * t / h_T) -
                           un_river)*min(t/T_ramp, 1.0)
ocean_un = Constant(ocean_un_func(t))
ocean_flux_func = lambda t: (flux_amp*sin(2 * pi * t / h_T) -
                             flux_river)*min(t/T_ramp, 1.0)
ocean_flux = Constant(ocean_flux_func(t))
river_flux_func = lambda t: flux_river*min(t/T_ramp, 1.0)
river_flux = Constant(river_flux_func(t))

ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}
ocean_funcs_3d = {'flux': ocean_flux}
river_funcs_3d = {'flux': river_flux}
#ocean_funcs_3d = {'symm': None}
#river_funcs_3d = {'symm': None}
ocean_salt_3d = {'value': salt_init3d}
river_salt_3d = {'value': salt_init3d}
solverObj.bnd_functions['shallow_water'] = {2: ocean_funcs, 1: river_funcs}
solverObj.bnd_functions['momentum'] = {2: ocean_funcs_3d, 1: river_funcs_3d}
solverObj.bnd_functions['salt'] = {2: ocean_salt_3d, 1: river_salt_3d}


def updateForcings(t_new):
    ocean_elev.dat.data[:] = ocean_elev_func(t_new)
    ocean_un.dat.data[:] = ocean_un_func(t_new)
    ocean_flux.dat.data[:] = ocean_flux_func(t_new)
    river_flux.dat.data[:] = river_flux_func(t_new)


def updateForcings3d(t_new):
    ocean_elev.dat.data[:] = ocean_elev_func(t_new)
    ocean_un.dat.data[:] = ocean_un_func(t_new)
    ocean_flux.dat.data[:] = ocean_flux_func(t_new)
    river_flux.dat.data[:] = river_flux_func(t_new)

solverObj.assignInitialConditions(salt=salt_init3d)
solverObj.iterate(updateForcings=updateForcings,
                  updateForcings3d=updateForcings3d)
