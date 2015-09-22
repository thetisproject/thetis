"""
Tests implicit bottom friction formulation with COFS
====================================================

Tuomas Karna 2015-09-16
"""
from cofs import *
import time as timeMod

parameters['coffee'] = {}

physical_constants['z0_friction'] = 1.5e-3

outputDir = createDirectory('outputs')
# set mesh resolution
scale = 1000.0
reso = 2.5*scale
layers = 50
depth = 15.0

# generate unit mesh and transform its coords
x_max = 5.0*scale
x_min = -5.0*scale
Lx = (x_max - x_min)
n_x = int(Lx/reso)
mesh2d = RectangleMesh(n_x, n_x, Lx, Lx, reorder=True)

printInfo('Exporting to ' + outputDir)
dt = 25.0  # 25.0
T = 12 * 3600.0
TExport = 100.0
depth = 15.0
Umag = 1.0

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry2d = Function(P1_2d, name='Bathymetry')
bathymetry2d.assign(depth)

# create solver
solverObj = solver.flowSolver(mesh2d, bathymetry2d, layers)
solverObj.nonlin = False
solverObj.solveSalt = False
solverObj.solveVertDiffusion = True
solverObj.useBottomFriction = True
#solverObj.useParabolicViscosity = True
solverObj.useTurbulence = True
solverObj.useALEMovingMesh = False
solverObj.useLimiterForTracers = True
solverObj.uvLaxFriedrichs = Constant(1.0)
solverObj.tracerLaxFriedrichs = Constant(0.0)
#solverObj.vViscosity = Constant(0.001)
#solverObj.hViscosity = Constant(1.0)
solverObj.TExport = TExport
solverObj.dt = dt
solverObj.T = T
solverObj.outputDir = outputDir
solverObj.uAdvection = Umag
solverObj.checkSaltDeviation = True
solverObj.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion', 'turbulence']
#solverObj.fieldsToExport = []
solverObj.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                            'uv2d_dav', 'uv2d_bot',
                            'parabNuv3d', 'eddyNuv3d', 'shearFreq3d',
                            'tke3d', 'psi3d', 'eps3d', 'len3d', ]
solverObj.fieldsToExportNumpy = ['uv3d', 'eddyNuv3d', 'shearFreq3d',
                                 'tke3d', 'psi3d', 'eps3d', 'len3d']
solverObj.mightyCreator()

elev_slope = -1.0e-5
pressureGradientSource = Constant((-9.81*elev_slope, 0, 0))

s = solverObj
vertMomEq = module_3d.verticalMomentumEquation(
                s.mesh, s.U, s.U_scalar, s.uv3d, w=None,
                viscosity_v=s.tot_v_visc.getSum(),
                uv_bottom=s.uv_bottom3d,
                bottom_drag=s.bottom_drag3d,
                wind_stress=s.wind_stress3d,
                vElemSize=s.vElemSize3d,
                source=pressureGradientSource)

sp = {}
sp['ksp_type'] = 'gmres'
#sp['pc_type'] = 'lu'
#sp['snes_monitor'] = True
#sp['snes_converged_reason'] = True
#sp['snes_rtol'] = 1e-4  # to avoid stagnation
sp['snes_rtol'] = 1e-18  # to avoid stagnation
sp['ksp_rtol'] = 1e-22  # to avoid stagnation
timeStepper = timeIntegrator.DIRK_LSPUM2(vertMomEq, dt, solver_parameters=sp)

# TODO fix momemtum eq for parabolic visc
# TODO mimic gotm implementation

t = 0
nSteps = int(np.round(T/dt))
for it in range(nSteps):
    t = it*dt
    t0 = timeMod.clock()
    # momentumEq
    timeStepper.advance(t, dt, s.uv3d)
    s.uvP1_projector.project()
    # update bottom friction
    computeBottomFriction(
        s.uv3d_P1, s.uv_bottom2d,
        s.uv_bottom3d, s.z_coord3d,
        s.z_bottom2d, s.z_bottom3d,
        s.bathymetry2d, s.bottom_drag2d,
        s.bottom_drag3d,
        s.vElemSize2d, s.vElemSize3d)
    # update viscosity
    s.glsModel.preprocess()
    # NOTE psi must be solved first as it depends on tke
    s.timeStepper.timeStepper_psi3d.advance(t, s.dt, s.psi3d)
    s.timeStepper.timeStepper_tke3d.advance(t, s.dt, s.tke3d)
    s.glsModel.postprocess()
    t1 = timeMod.clock()
    # NOTE vtk exporter has a memory leak if output space is DG
    s.export()
    print '{:4d}  T={:9.1f} s  cpu={:.2f} s'.format(it, t, t1-t0)


def test_solution():
    target_u_min = 0.5
    target_u_max = 0.9
    target_u_tol = 5.0e-2
    target_zero = 1e-8
    solutionP1DG = Function(s.P1DGv, name='velocity p1dg')
    solutionP1DG.project(s.uv3d)
    uvw = solutionP1DG.dat.data
    w_max = np.max(np.abs(uvw[:, 2]))
    v_max = np.max(np.abs(uvw[:, 1]))
    print 'w', w_max
    print 'v', v_max
    assert w_max < target_zero, 'z velocity component too large'
    assert v_max < target_zero, 'y velocity component too large'
    u_min = uvw[:, 0].min()
    u_max = uvw[:, 0].max()
    print 'u', u_min, u_max
    assert np.abs(u_min - target_u_min) < target_u_tol, 'minimum u velocity is wrong {:} != {:}'.format(u_min, target_u_min)
    assert np.abs(u_max - target_u_max) < target_u_tol, 'maximum u velocity is wrong {:} != {:}'.format(u_max, target_u_max)
