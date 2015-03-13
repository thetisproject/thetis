# Wave equation in 2D
# ===================
#
# Rectangular channel geometry.
#
# Tuomas Karna 2015-03-11

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


def createDirectory(path):
    if commrank == 0:
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception('file with same name exists', path)
        else:
            os.makedirs(path)
    return path

# set physical constants
physical_constants['z0_friction'].assign(0.0)
#physical_constants['viscosity_h'].assign(0.0)

mesh2d = Mesh('channel_waveEq.msh')

# Function spaces for 2d mode
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
U_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
U_visu_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
U_scalar_2d = FunctionSpace(mesh2d, 'DG', 1)
H_2d = FunctionSpace(mesh2d, 'CG', 2)

# Mean free surface height (bathymetry)
bathymetry2d = Function(P1_2d, name='Bathymetry')

use_wd = False
nonlin = False
swe2d = mode2d.freeSurfaceEquations(mesh2d, U_2d, H_2d, bathymetry2d,
                                    nonlin=nonlin, use_wd=use_wd)

x_func = Function(P1_2d).interpolate(Expression('x[0]'))
x_min = x_func.dat.data.min()
x_max = x_func.dat.data.max()
x_min = comm.allreduce(x_min, x_min, op=MPI.MIN)
x_max = comm.allreduce(x_max, x_max, op=MPI.MAX)
Lx = x_max - x_min

depth_oce = 50.0
depth_riv = 50.0
bath_x = np.array([0, Lx])
bath_v = np.array([depth_oce, depth_riv])
depth = 20.0


def bath(x, y, z):
    padval = 1e20
    x0 = np.hstack(([-padval], bath_x, [padval]))
    vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
    return interp1d(x0, vals0)(x)

#define a bath func depending on x,y,z
bathymetry2d.dat.data[:] = bath(x_func.dat.data, 0, 0)

outputDir = createDirectory('outputs_waveEq2d')
bathfile = File(os.path.join(outputDir, 'bath.pvd'))
bathfile << bathymetry2d

elev_init = Function(H_2d)
elev_init.project(Expression('eta_amp*sin(pi*x[0]/Lx)', eta_amp=0.5,
                             Lx=Lx))

# set time step, and run duration
c_wave = float(np.sqrt(9.81*depth_oce))
T_cycle = Lx/c_wave
n_steps = 20
dt = round(float(T_cycle/n_steps))
TExport = dt
T = 10*T_cycle + 1e-3
# explicit model
Umag = Constant(0.2)
mesh_dt = swe2d.getTimeStep(Umag=Umag)
dt = float(np.floor(mesh_dt.dat.data.min()/20.0))*0.8
dt = round(comm.allreduce(dt, dt, op=MPI.MIN))
print 'dt', dt
dt = float(TExport/np.ceil(TExport/dt))

if commrank == 0:
    print 'dt =', dt
    sys.stdout.flush()

#timeStepper2d = mode2d.DIRK3(swe2d, dt)
#timeStepper2d = mode2d.CrankNicolson(swe2d, dt, gamma=1.0)
timeStepper2d = mode2d.AdamsBashforth3(swe2d, dt)

U_2d_file = exporter(U_visu_2d, 'Depth averaged velocity', outputDir, 'Velocity2d.pvd')
eta_2d_file = exporter(P1_2d, 'Elevation', outputDir, 'Elevation2d.pvd')
uv_dav_2d_file = exporter(U_visu_2d, 'Depth Averaged Velocity', outputDir, 'DAVelocity2d.pvd')
uv_bot_2d_file = exporter(U_visu_2d, 'Bottom Velocity', outputDir, 'BotVelocity2d.pvd')

# assign initial conditions
uv2d, eta2d = swe2d.solution.split()
uv2d_old, eta2d_old = timeStepper2d.solution_old.split()
eta2d.assign(elev_init)
eta2d_old.assign(elev_init)

# Export initial conditions
U_2d_file.export(timeStepper2d.solution_old.split()[0])
eta_2d_file.export(timeStepper2d.solution_old.split()[1])

# The time-stepping loop
T_epsilon = 1.0e-5
cputimestamp = timeMod.clock()
t = 0
i = 0
iExp = 1
next_export_t = t + TExport


updateForcings = None


def compVolume(eta):
    val = assemble(eta * swe2d.dx)
    return op2.MPI.COMM.allreduce(val, op=MPI.SUM)

Vol_0 = compVolume(eta2d)
print 'Initial volume', Vol_0

from pyop2.profiling import timed_region, timed_function, timing

while t <= T + T_epsilon:

    # SSPRK33 time integration loop
    with timed_region('mode2d'):
        timeStepper2d.advance(t, dt, swe2d.solution, updateForcings)

    # Move to next time step
    t += dt
    i += 1

    # Write the solution to file
    if t >= next_export_t - T_epsilon:
        cputime = timeMod.clock() - cputimestamp
        cputimestamp = timeMod.clock()
        norm_h = norm(swe2d.solution.split()[1])
        norm_u = norm(swe2d.solution.split()[0])

        if commrank == 0:
            line = ('{iexp:5d} {i:5d} T={t:10.2f} '
                    'eta norm: {e:10.4f} u norm: {u:10.4f} {cpu:5.2f}')
            print(line.format(iexp=iExp, i=i, t=t, e=norm_h,
                              u=norm_u, cpu=cputime))
            line = 'Rel. vol. error {0:8.4e}'
            print(line.format((Vol_0 - compVolume(swe2d.solution.split()[1]))/Vol_0))

            sys.stdout.flush()
        U_2d_file.export(swe2d.solution.split()[0])
        eta_2d_file.export(swe2d.solution.split()[1])

        next_export_t += TExport
        iExp += 1

        if commrank == 0:
            labels = ['mode2d',]
            cost = {}
            relcost = {}
            totcost = 0
            for label in labels:
                value = timing(label, reset=True)
                cost[label] = value
                totcost += value
            for label in labels:
                c = cost[label]
                relcost = c/totcost
                print '{0:25s} : {1:11.6f} {2:11.2f}'.format(label, c, relcost)
                sys.stdout.flush()
