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

## HACK to fix unknown node: XXX / (F0) COFFEE errors
#op2.init()
#parameters['coffee']['O2'] = False

#parameters['form_compiler']['quadrature_degree'] = 6  # 'auto'
parameters['form_compiler']['optimize'] = False
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -xhost'

#from pyop2 import op2
comm = op2.MPI.comm
commrank = op2.MPI.comm.rank
op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# set physical constants
#physical_constants['z0_friction'].assign(5.0e-5)
physical_constants['z0_friction'].assign(0.0)
#physical_constants['viscosity_h'].assign(0.0)

mesh2d = Mesh('channel_mesh.msh')

# Function spaces for 2d mode
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
U_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
U_visu_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
U_scalar_2d = FunctionSpace(mesh2d, 'DG', 1)
H_2d = FunctionSpace(mesh2d, 'CG', 2)
W_2d = MixedFunctionSpace([U_2d, H_2d])

solution2d = Function(W_2d, name='solution2d')
# Mean free surface height (bathymetry)
bathymetry2d = Function(P1_2d, name='Bathymetry')

uv_bottom2d = Function(U_2d, name='Bottom Velocity')
z_bottom2d = Function(P1_2d, name='Bot. Vel. z coord')
bottom_drag2d = Function(P1_2d, name='Bottom Drag')

use_wd = False
nonlin = True
swe2d = mode2d.freeSurfaceEquations(mesh2d, W_2d, solution2d, bathymetry2d,
                                    #uv_bottom2d, bottom_drag2d,
                                    nonlin=nonlin, use_wd=use_wd)

# TODO advection unstable with SSPRK33, with/-out restartFromAv
# TODO try if DIRK3 is stable ...

#bath_x = np.array([0, 10e3, 30e3, 45e3, 100e3])
#bath_v = np.array([20, 20, 6, 15, 5])
depth_oce = 20.0
depth_riv = 7.0
bath_x = np.array([0, 8e3, 100e3])
bath_v = np.array([depth_oce, depth_oce, depth_riv])
depth = 20.0


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)

#define a bath func depending on x,y,z
x_func = Function(P1_2d).interpolate(Expression('x[0]'))
bathymetry2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

outputDir = createDirectory('outputs')
bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])


def elevation(x, y, z, x_array, val_array):
    padval = 1e20
    x0 = np.hstack(([-padval], x_array, [padval]))
    vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
    return interp1d(x0, vals0)(x)

x_func = Function(H_2d).interpolate(Expression('x[0]'))
elev_init = Function(H_2d)
#elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0,
                                  #elev_x, elev_v)
#elev_init.dat.data[:] = 2.0

T = 18 * 3600  # 100*24*3600
Umag = Constant(3.0)
mesh_dt = swe2d.getTimeStepAdvection(Umag=Umag)
dt = float(np.floor(mesh_dt.dat.data.min()/10.0))*0.80
dt = round(comm.allreduce(dt, dt, op=MPI.MIN))
TExport = 100.0
mesh2d_dt = swe2d.getTimeStep(Umag=Umag)
dt_2d = mesh2d_dt.dat.data.min()/20.0
dt_2d = comm.allreduce(dt_2d, dt_2d, op=MPI.MIN)
M_modesplit = int(np.ceil(dt/dt_2d))
dt_2d = float(dt/M_modesplit)
if commrank == 0:
    print 'dt =', dt
    print '2D dt =', dt_2d, M_modesplit
    sys.stdout.flush()

# weak boundary conditions
h_amp = 2.0
un_amp = -2.0
L_y = 1900
flux_amp = L_y*depth_oce*un_amp
h_T = 12 * 3600  # 44714.0
un_river = -0.3
flux_river = L_y*depth_riv*un_river
t = 0.0
#T_ramp = 3600.0
T_ramp = 1000.0
ocean_elev_func = lambda t: h_amp * sin(2 * pi * t / h_T)  # + 3*pi/2)
ocean_elev = Function(swe2d.space.sub(1)).interpolate(Expression(ocean_elev_func(t)))
ocean_un_func = lambda t: (un_amp*sin(2 * pi * t / h_T) -
                           un_river)*min(t/T_ramp, 1.0)
ocean_un = Function(H_2d).interpolate(Expression(ocean_un_func(t)))
ocean_flux_func = lambda t: (flux_amp*sin(2 * pi * t / h_T) -
                             flux_river)*min(t/T_ramp, 1.0)
ocean_flux = Function(H_2d).interpolate(Expression(ocean_flux_func(t)))
river_flux_func = lambda t: flux_river*min(t/T_ramp, 1.0)
river_flux = Function(U_scalar_2d).interpolate(Expression(river_flux_func(t)))
ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}
swe2d.bnd_functions = {2: ocean_funcs, 1: river_funcs}

solver_parameters = {
    #'ksp_type': 'fgmres',
    #'ksp_monitor': True,
    'ksp_rtol': 1e-12,
    'ksp_atol': 1e-16,
    #'pc_type': 'fieldsplit',
    #'pc_fieldsplit_type': 'multiplicative',
}
subIterator = timeIntegration.SSPRK33(swe2d, dt_2d, solver_parameters)
timeStepper2d = timeIntegration.macroTimeStepIntegrator(subIterator,
                                               M_modesplit,
                                               restartFromAv=False)

U_2d_file = exporter(U_visu_2d, 'Depth averaged velocity', outputDir, 'Velocity2d.pvd')
eta_2d_file = exporter(P1_2d, 'Elevation', outputDir, 'Elevation2d.pvd')

# assign initial conditions
uv2d, eta2d = solution2d.split()
eta2d.assign(elev_init)
timeStepper2d.initialize(solution2d)

# Export initial conditions
U_2d_file.export(solution2d.split()[0])
eta_2d_file.export(solution2d.split()[1])

# The time-stepping loop
T_epsilon = 1.0e-5
cputimestamp = timeMod.clock()
t = 0
i = 0
iExp = 1
next_export_t = t + TExport


def updateForcings(t_new):
    ocean_elev.dat.data[:] = ocean_elev_func(t_new)
    ocean_un.dat.data[:] = ocean_un_func(t_new)
    ocean_flux.dat.data[:] = ocean_flux_func(t_new)
    river_flux.dat.data[:] = river_flux_func(t_new)

from pyop2.profiling import timed_region, timed_function, timing

while t <= T + T_epsilon:

    with timed_region('mode2d'):
        timeStepper2d.advance(t, dt, swe2d.solution, updateForcings)

    # Move to next time step
    t += dt
    i += 1

    # Write the solution to file
    if t >= next_export_t - T_epsilon:
        cputime = timeMod.clock() - cputimestamp
        cputimestamp = timeMod.clock()
        norm_h = norm(solution2d.split()[1])
        norm_u = norm(solution2d.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                              u=norm_u, cpu=cputime))
            sys.stdout.flush()
        U_2d_file.export(solution2d.split()[0])
        eta_2d_file.export(solution2d.split()[1])

        next_export_t += TExport
        iExp += 1

        #if commrank == 0:
            #labels = ['mode2d', 'momentumEq', 'vert_diffusion',
                      #'continuityEq', 'saltEq', 'aux_functions']
            #cost = {}
            #relcost = {}
            #totcost = 0
            #for label in labels:
                #value = timing(label, reset=True)
                #cost[label] = value
                #totcost += value
            #for label in labels:
                #c = cost[label]
                #relcost = c/totcost
                #print '{0:25s} : {1:11.6f} {2:11.2f}'.format(label, c, relcost)
                #sys.stdout.flush()
