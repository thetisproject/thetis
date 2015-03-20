# Tracer advection in 3D
# ======================
#
# Consider tracer advetion equation in 3D domain :math:`\Omega` with
# impermeable boundaries
#
# .. math::
#
#    \frac{\partial C}{\partial t} + \nabla\cdot(\vec{u}C) = 0
#
#    n\cdot \vec{u} = 0 \ \textrm{on}\ \partial\Omega,
#
# where :math:`\vec{u}=(u,v,w)` is a prescribed vector field, and :math:`C` is
# the scalar field. The initial value of :math:`C` is known.
#
# The weak form of the advection equation is
#
# .. math::
#
#    \int_\Omega \! \phi \frac{\partial C}{\partial t} \, \mathrm{d} x +
#        \int_\Omega \! \phi \nabla\cdot(\vec{u}C) \, \mathrm{d} x = 0
#
#    \Leftrightarrow
#      \int_\Omega \! \phi \frac{\partial C}{\partial t} \, \mathrm{d} x +
#      \int_\Omega \! \phi \nabla_h\cdot(\vec{u}_h C) \, \mathrm{d} x +
#      \int_\Omega \! \phi \frac{\partial (wC)}{\partial z}\, \mathrm{d} x = 0
#
#    \Leftrightarrow
#       \int_\Omega \! \phi \frac{\partial C}{\partial t} \, \mathrm{d} x -
#       \int_\Omega \! C \nabla_h\phi \cdot \vec{u}_h \, \mathrm{d} x -
#       \int_\Omega \! C \frac{\partial \phi}{\partial z}w \, \mathrm{d} x = 0,
#
# where :math:`\phi` is a suitable test function. The last form was
# obtained by integration by parts and making use of the impermeable boundary
# conditions.
#
# First constraint for a tracer advection test is to construct a velocity field
# that is divergence-free in the weak sense. We therefore prescribe :math:`u`
# and :math:`v`, and solve :math:`w` from the continuity equation, similarly to
# hydrostatic models where :math:`w` is a diagnostic variable
#
# .. math::
#
#    \nabla\cdot(\vec{u}) = 0
#
#    \Leftrightarrow \frac{\partial u}{\partial x} +
#              \frac{\partial v}{\partial y}+ \frac{\partial w}{\partial z} = 0
#
#    \Rightarrow \frac{\partial w}{\partial z} = -\frac{\partial u}{\partial x}
#               -\frac{\partial v}{\partial y} = - \nabla_h \cdot \vec{u}_h,
#
# where :math:`\nabla_h` is horizontal gradient operator and :math:`\vec{u}_h`
# is restriction of :math:`\vec{u}` on the horizontal plane. The weak form for
# solving :math:`w` is
#
# .. math::
#
#    \int_\Omega \! \phi_w \frac{\partial w}{\partial z} \, \mathrm{d} x =
#             - \int_\Omega \! \phi_w \nabla_h \cdot \vec{u}_h \, \mathrm{d} x,
#
#    \Leftrightarrow
#    \int_\Omega \! \frac{\partial \phi_w}{\partial z}w \, \mathrm{d} x =
#            - \int_\Omega \! \nabla_h \phi_w  \cdot \vec{u}_h \, \mathrm{d} x,
#
# where :math:`\phi_w` is a suitable test function. Here :math:`\vec{u}_h` is
# prescribed to produce a gyre in the :math:`(x,z)` plane that vanishes on the
# boundaries. The aspect ratio of the mesh is :math:`L_x/L_z = 1\times10^4`,
# typical to ocean applications.
#
# The **first test** is to advect a constant tracer field. Setting :math:`C` to
# a constant in space in the weak formulation gives
#
# .. math::
#
#    \int_\Omega \! \phi \frac{\partial C}{\partial t} \, \mathrm{d} x -
#    C \Big[ \int_\Omega \! \nabla_h\phi \cdot \vec{u}_h \, \mathrm{d} x +
#    \int_\Omega \! \frac{\partial \phi}{\partial z} w \, \mathrm{d} x \Big]= 0
#
# which implies that :math:`\partial C/\partial t=0` iff the velocity field is
# diverence-free in the sense of the expression in square the bracets. This can
# be ensured by requiring that :math:`w`, :math:`C` and the test functions
# :math:`\phi_,\phi_w` belong to the same space. In this case :math:`C` should
# remain constant within the precision of the solver.
#
# The **second test** is to advect a non-trivial tracer field and assess
# numerical diffusion and monotonicity of the advection scheme. Here we start
# with tracer field initially set to 1 on the left hand side of the domain, 0
# elsewhere.
#
# Here we use continuous elements for :math:`\phi` and :math:`C`, and use SUPG
# to stabilize the advection term, i.e. we replace :math:`\phi` with
# :math:`\hat{\phi}`
#
# .. math::
#
#    \hat{\phi} = \phi + \alpha h \frac{\vec{u}}{|\vec{u}|} \cdot \nabla\phi,
#
# where :math:`\alpha` is constant and :math:`h` is the element size.
#
# The weak formulation of the advection equation then becomes
#
# .. math::
#
#    \int_\Omega \! \hat{\phi} \frac{\partial C}{\partial t} \, \mathrm{d} x -
#    \int_\Omega \! C \nabla_h\phi \cdot \vec{u}_h \, \mathrm{d} x -
#    \int_\Omega \! C \frac{\partial \phi}{\partial z}w \, \mathrm{d} x +
#    \int_\Omega \! \alpha h \frac{\vec{u}}{|\vec{u}|} \cdot \nabla\phi
#    \big[\nabla_hC \cdot \vec{u}_h + \frac{\partial C}{\partial z}w \big] = 0.
#
# Tuomas Karna 2014-09-04
#

from firedrake import *
import os
import numpy as np
import sys
import time as timeMod
from mpi4py import MPI
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

comm = op2.MPI.comm
commrank = op2.MPI.comm.rank
op2.init(log_level=WARNING)  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# order of basis functions
order = 2

mesh2d = Mesh('channel_mesh_p{0:d}.msh'.format(order))
mesh = ExtrudedMesh(mesh2d, layers=100/order, layer_height=-0.1*order)
commrank = op2.MPI.comm.rank
op2.init(log_level=CRITICAL)
outputDir = createDirectory('outputs_x' + '_p' + str(order))
# Define mesh dimensions

L_x = 1e5  # from 0 to L_x
L_y = 1e3  # from 0 to L_y
L_z = 10   # from -L_z to 0

# Function spaces
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
U_2d = VectorFunctionSpace(mesh2d, 'DG', 1)
U_visu_2d = VectorFunctionSpace(mesh2d, 'CG', 1)
U_scalar_2d = FunctionSpace(mesh2d, 'DG', 1)
H_2d = FunctionSpace(mesh2d, 'CG', order)
W_2d = MixedFunctionSpace([U_2d, H_2d])

solution2d = Function(W_2d, name='solution2d')
bathymetry2d = Function(P1_2d, name='Bathymetry')

use_wd = False
nonlin = False
swe2d = mode2d.freeSurfaceEquations(mesh2d, W_2d, solution2d, bathymetry2d,
                                    nonlin=nonlin, use_wd=use_wd)

P1 = FunctionSpace(mesh, 'CG', 1, vfamily='CG', vdegree=1)
U = VectorFunctionSpace(mesh, 'DG', 1, vfamily='CG', vdegree=1)
U_visu = VectorFunctionSpace(mesh, 'CG', 1, vfamily='CG', vdegree=1)
U_scalar = FunctionSpace(mesh, 'DG', 1, vfamily='CG', vdegree=1)
H = FunctionSpace(mesh, 'CG', order, vfamily='CG', vdegree=1)

#U = VectorFunctionSpace(mesh, 'CG', order, vfamily='CG', vdegree=order)
#U_visu = VectorFunctionSpace(mesh, 'CG', 1, vfamily='CG', vdegree=1)
#U_scalar = FunctionSpace(mesh, 'CG', order, vfamily='CG', vdegree=order)
#H = FunctionSpace(mesh, 'CG', order, vfamily='CG', vdegree=order)

eta3d = Function(H, name='Elevation')
bathymetry3d = Function(P1, name='Bathymetry')
uv3d = Function(U, name='Velocity')
w3d = Function(H, name='Vertical Velocity')
salt3d = Function(H, name='Salinity')

# Initial conditions
eta3d.assign(0.0)
bathymetry3d.assign(L_z)
bathymetry2d.assign(L_z)
# Horizontal velocity
u_mag = 1.0
velocity = Expression(('u_mag',
                       '0.0',
                       '0.0'),
                       L_x=L_x, L_z=L_z, u_mag=u_mag)

uv3d.project(velocity)

# Initial tracer field
density_mag = 1.0
# Test 1
#density = Expression(('4.5'), L_x=L_x, density_mag=density_mag)
# Test 2
density = Expression(('density_mag*(x[0] > 0.25*L_x) ? 0.0 : 1.0'), L_x=L_x,
                     density_mag=density_mag)
salt_init = Function(P1, name='salt init').interpolate(density)
#salt3d.interpolate(density)
salt3d.project(salt_init)

# Compute vertical velocity
computeVertVelocity(w3d, uv3d, bathymetry3d)

# outputs
uv_3d_file = exporter(U_visu, 'Velocity', outputDir, 'Velocity3d.pvd')
w_3d_file = exporter(P1, 'V.Velocity', outputDir, 'VertVelo3d.pvd')
salt_3d_file = exporter(P1, 'Salinity', outputDir, 'Salinity3d.pvd')

uv_3d_file.export(uv3d)
w_3d_file.export(w3d)
salt_3d_file.export(salt3d)

# Define tracer advection equations

T = 60*3600
TExport = 400.0
dt = 100.0 #133.3333  # TODO use CFL
mesh2d_dt = swe2d.getTimeStepAdvection(Umag=u_mag)
dt = float(np.floor(mesh2d_dt.dat.data.min()/20.0))
dt = round(comm.allreduce(dt, dt, op=MPI.MIN))
dt = float(TExport/np.ceil(TExport/dt))
if commrank == 0:
    print 'dt =', dt
    sys.stdout.flush()


def computeMagnitude(solution, u=None, w=None):
    """
    Computes magnitude of (u[0],u[1],w) and stores it in solution
    """

    function_space = solution.function_space()
    phi = TestFunction(function_space)
    magnitude = TrialFunction(function_space)

    a = phi*magnitude*dx
    s = 0
    if u is not None:
        s += u[0]**2 + u[1]**2
    if w is not None:
        s += w**2
    L = phi*sqrt(s)*dx

    solve(a == L, solution)
    solution.dat.data[:] = np.maximum(solution.dat.data[:], 1e-6)


def getHorzontalElemSize(P1_2d, P1_3d=None):
    """
    Computes horizontal element size from the 2D mesh, the copies it over a 3D
    field.
    """
    cellsize = CellSize(P1_2d.mesh())
    test = TestFunction(P1_2d)
    tri = TrialFunction(P1_2d)
    sol2d = Function(P1_2d)
    dx_2d = Measure('dx', domain=mesh2d, subdomain_id='everywhere')
    a = test * tri * dx_2d
    L = test * cellsize * dx_2d
    solve(a == L, sol2d)
    if P1_3d is None:
        return sol2d
    sol3d = Function(P1_3d)
    copy2dFieldTo3d(sol2d, sol3d)
    return sol3d


def getVerticalElemSize(P1_2d, P1_3d):
    """
    Computes horizontal element size from the 2D mesh, the copies it over a 3D
    field.
    """
    # compute total depth
    depth2d = Function(P1_2d)
    zbot2d = Function(P1_2d)
    zcoord3d = Function(P1_3d)
    project(Expression('x[2]'), zcoord3d)
    copy3dFieldTo2d(zcoord3d, depth2d, useBottomValue=False)
    copy3dFieldTo2d(zcoord3d, zbot2d, useBottomValue=True)
    #print depth2d.dat.data.min(), depth2d.dat.data.max()
    #print zbot2d.dat.data.min(), zbot2d.dat.data.max()
    depth2d += - zbot2d
    # divide by number of element layers
    n_layers = P1_3d.mesh().layers - 1
    depth2d /= n_layers
    copy2dFieldTo3d(depth2d, zcoord3d)
    return zcoord3d

hElemSize3d = getHorzontalElemSize(P1_2d, P1)
vElemSize3d = getVerticalElemSize(P1_2d, P1)

# SUPG stabilization parameters
alpha = Constant(0.3)  # cant be too large

u_mag_func = Function(U_scalar)
u_mag_func_h = Function(U_scalar)
u_mag_func_v = Function(U_scalar)
scale_h = Constant(1.0)  # can be increased
scale_v = Constant(1.0)
computeMagnitude(u_mag_func, u=uv3d, w=w3d)
computeMagnitude(u_mag_func_h, u=uv3d)
computeMagnitude(u_mag_func_v, w=w3d)
# k = (uh)_e/2 xi
# xi = coth(alpha) - 1/alpha
# alpha = uh/2k
# w_SUPG = w + k u_j/|u| w_,j /|u| = w + (uh)_e/2*xi*u_j*w_,j/|u|**2
# (uh)_e = velocity * elem size in the direction of u !!
# w_SUPG = w + k u_j/|u| w_,j /|u| = w + (h)_e/2*xi/|u|*u_j*w_,j
gamma_h = Function(U_scalar, name='gamma_h')
#gamma_v = Function(U_scalar, name='gamma_v')
gamma_h.project(hElemSize3d/2*alpha/u_mag_func)
#gamma_v.project(scale_v*vElemSize3d*alpha/u_mag_func_v)
test = TestFunction(H)
#test_supg_mass = alpha/u_mag_func*(hElemSize3d*uv3d[0]*Dx(test, 0) +
                                   #hElemSize3d*uv3d[1]*Dx(test, 1) +
                                   #vElemSize3d*w3d*Dx(test, 2))
test_supg_mass = gamma_h*(uv3d[0]*Dx(test, 0) +
                          uv3d[1]*Dx(test, 1))
test_supg_h = scale_h*gamma_h*(uv3d[0]*Dx(test, 0) + uv3d[1]*Dx(test, 1))
test_supg_v = None  # gamma_v*(w3d*Dx(test, 2))
#test_supg_mass = test_supg_h = test_supg_v = None

sol_old = Function(H)
sol_old.assign(salt3d)
dt_const = Constant(dt)
tri = TrialFunction(H)
a_RK = inner(tri, test+test_supg_mass)*dx
L_RK = -dt_const*(test + test_supg_h)*(uv3d[0]*Dx(sol_old, 0) + uv3d[1]*Dx(sol_old, 1))*dx

dsol0 = Function(H, name='drhodt_0')
dsol1 = Function(H, name='drhodt_1')
dsol2 = Function(H, name='drhodt_2')

probK0 = LinearVariationalProblem(a_RK, L_RK, dsol0)
solverK0 = LinearVariationalSolver(probK0)
probK1 = LinearVariationalProblem(a_RK, L_RK, dsol1)
solverK1 = LinearVariationalSolver(probK1)
probK2 = LinearVariationalProblem(a_RK, L_RK, dsol2)
solverK2 = LinearVariationalSolver(probK2)

# equations
salt_eq3d = mode3d.tracerEquation(mesh, H, salt3d, eta3d, uv3d, w=w3d,
                                  test_supg_h=test_supg_h,
                                  test_supg_v=test_supg_v,
                                  test_supg_mass=test_supg_mass,
                                  bnd_markers=swe2d.boundary_markers,
                                  bnd_len=swe2d.boundary_len)
ocean_salt_3d = {'value': Constant(1.0)}
river_salt_3d = {'value': Constant(0.0)}
salt_eq3d.bnd_functions = {2: ocean_salt_3d, 1: river_salt_3d}

timeStepper_salt3d = mode3d.SSPRK33(salt_eq3d, dt)
#timeStepper_salt3d = mode3d.ForwardEuler(salt_eq3d, dt)
timeStepper_salt3d.initialize(salt3d)

from pyop2.profiling import timed_region, timed_function

# The time-stepping loop
T_epsilon = 1.0e-5
t = 0
i = 0
iExp = 1
next_export_t = t + TExport
velocity_flipped = False
cputimestamp = timeMod.clock()

while t <= T + T_epsilon:

    with timed_region('saltEq'):
        timeStepper_salt3d.advance(t, dt, salt3d, None)
        #solverK0.solve()
        #sol_old.assign(salt3d + dsol0)
        #solverK1.solve()
        #sol_old.assign(salt3d + 0.25*dsol0 + 0.25*dsol1)
        #solverK2.solve()
        #salt3d += (1.0/6.0)*dsol0 + (1.0/6.0)*dsol1 + (2.0/3.0)*dsol2
        #sol_old.assign(salt3d)

    norm_C = norm(salt3d)
    norm_u = norm(uv3d)

    # Move to next time step
    t += dt
    i += 1

    if t >= next_export_t - T_epsilon:
        cputime = timeMod.clock() - cputimestamp
        if commrank == 0:
            print(('{iexp:5d} {i:5d} T={t:10.2f} C norm: {e:10.4f}'
                   ' u norm: {u:10.4f}  {cpu:5.2f}').format(iexp=iExp, i=i,
                                                            t=t, e=norm_C,
                                                            u=norm_u,
                                                            cpu=cputime))
            line = 'density {0:11.6f} {1:11.6f}'
            print line.format(salt3d.dat.data.min(), salt3d.dat.data.max()-1.0)
            sys.stdout.flush()
        uv_3d_file.export(uv3d)
        w_3d_file.export(w3d)
        salt_3d_file.export(salt3d)

        cputimestamp = timeMod.clock()
        next_export_t += TExport
        iExp += 1

    if not velocity_flipped and t > T/2:
        uv3d *= -1
        w3d *= -1
        velocity_flipped = True
