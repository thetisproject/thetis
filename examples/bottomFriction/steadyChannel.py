"""
Steady-state channel flow in 3D
===============================

Solves shallow water equations in open channel using log-layer bottom friction
and a constant volume flux.

This test case test the turbulence closure model and bottom boundary layer

Model setup is according to [1].

[1] Karna et al. (2012). Coupling of a discontinuous Galerkin finite element
    marine model with a finite difference turbulence closure model.
    Ocean Modelling, 47:55-64.
    http://dx.doi.org/10.1016/j.ocemod.2012.01.001

Tuomas Karna 2015-09-09
"""
from cofs import *

parameters['coffee'] = {}

physical_constants['z0_friction'] = 1.5e-3

outputdir = create_directory('outputs')
# set mesh resolution
dx = 2500.0
layers = 25

# generate unit mesh and transform its coords
x_max = 5.0e3
x_min = -5.0e3
Lx = (x_max - x_min)
n_x = Lx/dx
mesh2d = RectangleMesh(n_x, n_x, Lx, Lx, reorder=True)
# move mesh, center to (0,0)
mesh2d.coordinates.dat.data[:, 0] -= Lx/2
mesh2d.coordinates.dat.data[:, 1] -= Lx/2

print_info('Exporting to ' + outputdir)
# NOTE bottom friction (implicit mom eq) will blow up for higher dt
dt = 25.0
T = 12 * 3600.0  # 24 * 3600
TExport = 200.0
depth = 15.0
Umag = 1.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers)
options = solverObj.options
options.nonlin = False
options.solveSalt = False
options.solveVertDiffusion = True
options.useBottomFriction = True
options.useTurbulence = True
options.useALEMovingMesh = False
options.useLimiterForTracers = False
options.uvLaxFriedrichs = Constant(1.0)
options.tracerLaxFriedrichs = Constant(0.0)
# options.vViscosity = Constant(0.001)
# options.hViscosity = Constant(1.0)
# options.useSemiImplicit2D = False
# options.useModeSplit = False
options.TExport = TExport
options.dt = dt
options.T = T
options.outputdir = outputdir
options.uAdvection = Umag
options.checkSaltDeviation = True
options.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion', 'turbulence']
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d',
                            'baroc_head_3d', 'baroc_head_2d',
                            'uv_dav_2d', 'uv_bottom_2d',
                            'parab_visc_3d', 'eddy_visc_3d', 'shear_freq_3d',
                            'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]

# weak boundary conditions
left_tag = 1   # x=x_min plane
right_tag = 2  # x=x_max plane
surf_slope = 1.0e-5
left_elev = Constant(+0.5*Lx*surf_slope)
right_elev = Constant(-0.5*Lx*surf_slope)
right_funcs = {'elev': right_elev}
left_funcs = {'elev': left_elev}
solverObj.bnd_functions['shallow_water'] = {right_tag: right_funcs,
                                            left_tag: left_funcs}
solverObj.bnd_functions['momentum'] = {right_tag: right_funcs,
                                       left_tag: left_funcs}

solverObj.createEquations()
elev_init = Function(solverObj.function_spaces.H_2d, name='initial elev')
elev_init.interpolate(Expression('x[0]*slope', slope=-surf_slope))

solverObj.assignInitialConditions(elev=elev_init)
# sp = solverObj.timeStepper.timeStepper_vmom3d.solver_parameters
# sp['snes_monitor'] = True
# sp['ksp_monitor'] = True
# sp['ksp_monitor_true_residual'] = True
# sp['ksp_type'] = 'cg'
# sp['pc_type'] = 'ilu'
# sp['snes_converged_reason'] = True
# sp['ksp_converged_reason'] = True
# solverObj.timeStepper.timeStepper_vmom3d.updateSolver()
solverObj.iterate()
