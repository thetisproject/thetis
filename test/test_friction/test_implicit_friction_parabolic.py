"""
Tests implicit bottom friction formulation with COFS
====================================================

Tuomas Karna 2015-09-16
"""
from cofs import *
import time as timeMod

parameters['coffee'] = {}

physical_constants['z0_friction'] = 1.5e-3

outputDir = createDirectory('outputs_parab')
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
dt = 3600.0
T = 10 * 3600.0
TExport = 100.0
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
options.useParabolicViscosity = True
#options.useTurbulence = True
options.useALEMovingMesh = False
options.useLimiterForTracers = False
options.uvLaxFriedrichs = Constant(1.0)
options.tracerLaxFriedrichs = Constant(0.0)
#options.vViscosity = Constant(0.001)
#options.hViscosity = Constant(1.0)
options.TExport = TExport
options.dt = dt
options.T = T
options.outputDir = outputDir
options.uAdvection = Umag
options.checkSaltDeviation = True
options.timerLabels = ['mode2d', 'momentumEq', 'vert_diffusion', 'turbulence']
options.fieldsToExport = ['uv2d', 'elev2d', 'elev3d', 'uv3d',
                          'w3d', 'wMesh3d', 'salt3d',
                          'baroHead3d', 'baroHead2d',
                          'uvDav2d', 'uvBot2d',
                          'parabVisc3d', 'eddyVisc3d', 'shearFreq3d',
                          'tke3d', 'psi3d', 'eps3d', 'len3d',]
solverObj.createEquations()

elev_slope = -1.0e-5
pressureGradientSource = Constant((-9.81*elev_slope, 0, 0))
s = solverObj
uv3d_old = Function(s.U, name='Velocity old')
vertMomEq = momentumEquation.verticalMomentumEquation(
                s.fields.uv3d, w=None,
                viscosity_v=s.tot_v_visc.getSum(),
                uv_bottom=uv3d_old,
                bottom_drag=s.fields.bottom_drag3d,
                wind_stress=s.fields.get('windStress3d'),
                vElemSize=s.fields.vElemSize3d,
                source=pressureGradientSource)

sp = {}
sp['ksp_type'] = 'gmres'
#sp['pc_type'] = 'lu'
#sp['snes_monitor'] = True
#sp['snes_converged_reason'] = True
sp['snes_rtol'] = 1e-4  # to avoid stagnation
#timeStepper = timeIntegrator.DIRK_LSPUM2(vertMomEq, dt, solver_parameters=sp)
timeStepper = timeIntegrator.BackwardEuler(vertMomEq, dt, solver_parameters=sp)

t = 0
nSteps = int(np.round(T/dt))
for it in range(nSteps):
    t = it*dt
    try:
        s.uvP1_projector.project()
        computeBottomFriction(
            s.fields.uv3d_P1, s.fields.uv_bottom2d,
            s.fields.uv_bottom3d, s.fields.z_coord3d,
            s.fields.z_bottom2d,
            s.fields.bathymetry2d, s.fields.bottom_drag2d,
            s.fields.bottom_drag3d,
            s.fields.vElemSize2d, s.fields.vElemSize3d)
        computeParabolicViscosity(
            s.fields.uv_bottom3d, s.fields.bottom_drag3d,
            s.fields.bathymetry3d,
            s.fields.parabVisc3d)
        t0 = timeMod.clock()
        timeStepper.advance(t, dt, s.fields.uv3d)
        uv3d_old.assign(s.fields.uv3d)
        t1 = timeMod.clock()
        for key in s.exporters:
            s.exporters[key].export()
        print '{:4d}  T={:9.1f} s  cpu={:.2f} s'.format(it, t, t1-t0)
    except Exception as e:
        raise e


def test_solution():
    target_u_min = 0.5
    target_u_max = 0.9
    target_u_tol = 5.0e-2
    target_zero = 1e-6
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
