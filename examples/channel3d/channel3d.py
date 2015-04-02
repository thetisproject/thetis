# Idealised channel flow in 3D
# ============================
#
# Solves hydrostatic flow in a rectangular channel.
#
# Tuomas Karna 2015-03-03

from firedrake import *
import numpy as np
import os
import sys
import time as timeMod
from mpi4py import MPI
from scipy.interpolate import interp1d
import cofs.module_2d as mode2d
import cofs.module_3d as mode3d
from cofs.utility import *
from cofs.physical_constants import physical_constants
import cofs.timeIntegration as timeIntegration
import cofs.solver as solverMod

# HACK to fix unknown node: XXX / (F0) COFFEE errors
op2.init()
parameters['coffee']['O2'] = False

#parameters['form_compiler']['quadrature_degree'] = 6  # 'auto'
parameters['form_compiler']['optimize'] = False
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -xhost'

#from pyop2 import op2
comm = op2.MPI.comm
commrank = op2.MPI.comm.rank
op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# set physical constants
#physical_constants['z0_friction'].assign(0.0)
physical_constants['z0_friction'].assign(1.0e-6)

use_wd = False
nonlin = True
n_layers = 6
outputDir = createDirectory('outputs')
mesh2d = Mesh('channel_mesh.msh')
T = 48 * 3600
Umag = Constant(4.2)
TExport = 100.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')

depth_oce = 20.0
depth_riv = 5.0
bath_x = np.array([0, 100e3])
bath_v = np.array([depth_oce, depth_riv])


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)


#define a bath func depending on x,y,z
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
bathymetry2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

# create solver
solver = solverMod.flowSolver(mesh2d, bathymetry2d, n_layers)
solver.nonlin = nonlin
solver.use_wd = use_wd
solver.TExport = TExport
solver.T = T
solver.uAdvection = Umag

# initial conditions
elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])
def elevation(x, y, z, x_array, val_array):
    padval = 1e20
    x0 = np.hstack(([-padval], x_array, [padval]))
    vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
    return interp1d(x0, vals0)(x)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
elev_init = Function(P1_2d)
elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0,
                                  elev_x, elev_v)

salt_init3d = Constant(4.5)


## weak boundary conditions
#L_y = 1900
#h_amp = 2.0
#un_amp = -2.0
#flux_amp = L_y*depth_oce*un_amp
#h_T = 12 * 3600  # 44714.0
#un_river = -0.3
#flux_river = L_y*depth_riv*un_river
#t = 0.0
##T_ramp = 3600.0
#T_ramp = 1000.0
#ocean_elev_func = lambda t: h_amp * sin(2 * pi * t / h_T)  # + 3*pi/2)
#ocean_elev = Function(swe2d.space.sub(1)).interpolate(Expression(ocean_elev_func(t)))
#ocean_elev_3d = Function(H).interpolate(Expression(ocean_elev_func(t)))
#ocean_un_func = lambda t: (un_amp*sin(2 * pi * t / h_T) -
                           #un_river)*min(t/T_ramp, 1.0)
#ocean_un = Function(H_2d).interpolate(Expression(ocean_un_func(t)))
#ocean_un_3d = Function(H).interpolate(Expression(ocean_un_func(t)))
#ocean_flux_func = lambda t: (flux_amp*sin(2 * pi * t / h_T) -
                             #flux_river)*min(t/T_ramp, 1.0)
##ocean_flux_func = lambda t: (flux_amp)*min(t/T_ramp, 1.0)
#ocean_flux = Function(H_2d).interpolate(Expression(ocean_flux_func(t)))
#ocean_flux_3d = Function(H).interpolate(Expression(ocean_flux_func(t)))
#river_flux_func = lambda t: flux_river*min(t/T_ramp, 1.0)
#river_flux = Function(U_scalar_2d).interpolate(Expression(river_flux_func(t)))
#river_flux_3d = Function(U_scalar).interpolate(Expression(river_flux_func(t)))
#ocean_funcs = {'flux': ocean_flux}
#river_funcs = {'flux': river_flux}
#ocean_funcs_3d = {'flux': ocean_flux_3d}
#river_funcs_3d = {'flux': river_flux_3d}
#ocean_salt_3d = {'value': salt_init3d}
#river_salt_3d = {'value': salt_init3d}
#swe2d.bnd_functions = {2: ocean_funcs, 1: river_funcs}
#mom_eq3d.bnd_functions = {2: ocean_funcs_3d, 1: river_funcs_3d}
#salt_eq3d.bnd_functions = {2: ocean_salt_3d, 1: river_salt_3d}

## exporters
#U_2d_file = exporter(U_visu_2d, 'Depth averaged velocity', outputDir, 'Velocity2d.pvd')
#eta_2d_file = exporter(P1_2d, 'Elevation', outputDir, 'Elevation2d.pvd')
#eta_3d_file = exporter(P1, 'Elevation', outputDir, 'Elevation3d.pvd')
#uv_3d_file = exporter(U_visu, 'Velocity', outputDir, 'Velocity3d.pvd')
#w_3d_file = exporter(P1, 'V.Velocity', outputDir, 'VertVelo3d.pvd')
#w_mesh_3d_file = exporter(P1, 'Mesh Velocity', outputDir, 'MeshVelo3d.pvd')
#salt_3d_file = exporter(P1, 'Salinity', outputDir, 'Salinity3d.pvd')
#uv_dav_2d_file = exporter(U_visu_2d, 'Depth Averaged Velocity', outputDir, 'DAVelocity2d.pvd')
#uv_bot_2d_file = exporter(U_visu_2d, 'Bottom Velocity', outputDir, 'BotVelocity2d.pvd')
#visc_3d_file = exporter(P1, 'Vertical Viscosity', outputDir, 'Viscosity3d.pvd')

# assign initial conditions
solver.assingInitialConditions(elev=elev_init, salt=salt_init3d)

## Export initial conditions
#U_2d_file.export(solution2d.split()[0])
#eta_2d_file.export(solution2d.split()[1])
#eta_3d_file.export(eta3d)
#uv_3d_file.export(uv3d)
#w_3d_file.export(w3d)
#w_mesh_3d_file.export(w_mesh3d)
#salt_3d_file.export(salt3d)
#uv_dav_2d_file.export(uv2d_dav)
#uv_bot_2d_file.export(uv_bottom2d)
#visc_3d_file.export(viscosity_v3d)


#def updateForcings(t_new):
    #ocean_elev.dat.data[:] = ocean_elev_func(t_new)
    #ocean_un.dat.data[:] = ocean_un_func(t_new)
    #ocean_flux.dat.data[:] = ocean_flux_func(t_new)
    #river_flux.dat.data[:] = river_flux_func(t_new)


#def updateForcings3d(t_new):
    #ocean_elev_3d.dat.data[:] = ocean_elev_func(t_new)
    #ocean_un_3d.dat.data[:] = ocean_un_func(t_new)
    #ocean_flux_3d.dat.data[:] = ocean_flux_func(t_new)
    #river_flux_3d.dat.data[:] = river_flux_func(t_new)


solver.iterate()