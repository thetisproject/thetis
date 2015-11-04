"""
Tests implicit vertical diffusion on a DG vector field
======================================================

Intended to be executed with pytest.

Tuomas Karna 2015-09-16
"""
from firedrake import *
import numpy as np

parameters['coffee'] = {}
exportSolution = False

# set mesh resolution
scale = 1000.0
reso = 2.5*scale
layers = 50
depth = 50

# generate unit mesh and transform its coords
x_max = 5.0*scale
x_min = -5.0*scale
Lx = (x_max - x_min)
n_x = int(Lx/reso)
mesh2d = RectangleMesh(n_x, n_x, Lx, Lx, reorder=True)
# move mesh, center at (0,0)
mesh2d.coordinates.dat.data[:, 0] -= Lx/2
mesh2d.coordinates.dat.data[:, 1] -= Lx/2

mesh = ExtrudedMesh(mesh2d, layers=50, layer_height=-depth/layers)

if exportSolution:
    outFile = File('test.pvd')

# define function spaces
fam = 'DG'
deg = 1
H = FunctionSpace(mesh, fam, degree=deg, vfamily=fam, vdegree=deg)
V = VectorFunctionSpace(mesh, fam, degree=deg, vfamily=fam, vdegree=deg)

solution = Function(V, name='velocity')
solution_new = Function(V, name='new velocity')
diffusivity_v = Constant(1.0)

test = TestFunction(V)
normal = FacetNormal(mesh)

# initial condition
solution.interpolate(Expression(['(x[2] > -25.0) ? 0.0 : 1.0', 0.0, 0.0]))


def RHS(solution):
    # vertical diffusion operator integrated by parts
    f = -diffusivity_v*inner(Dx(solution, 2), Dx(test, 2)) * dx
    # interface term
    diffFlux = diffusivity_v*Dx(solution, 2)
    f += (dot(avg(diffFlux), test('+'))*normal[2]('+') +
          dot(avg(diffFlux), test('-'))*normal[2]('-')) * dS_h
    # symmetric interior penalty stabilization
    L = Constant(depth/layers)
    nbNeigh = 2
    d = 3
    sigma = Constant((deg + 1)*(deg + d)/d * nbNeigh / 2) / L
    gamma = sigma*avg(diffusivity_v)
    jump_test = test('+')*normal[2]('+') + test('-')*normal[2]('-')
    f += gamma * dot(jump(solution), jump_test) * dS_h

    return f

# define solver
sp = {}
sp['snes_monitor'] = True
#sp['ksp_monitor'] = True
sp['ksp_monitor_true_residual'] = True
sp['snes_converged_reason'] = True
sp['ksp_converged_reason'] = True

dt_const = Constant(1000.0)
F = (inner(solution_new, test)*dx - inner(solution, test)*dx -
     dt_const*RHS(solution_new))
prob = NonlinearVariationalProblem(F, solution_new)
solver = LinearVariationalSolver(prob, solver_parameters=sp)

# solve

if exportSolution:
    outFile << solution
print 'sol', solution.dat.data[:, 0].min(), solution.dat.data[:, 0].max()
solver.solve()
solution.assign(solution_new)

if exportSolution:
    outFile << solution
print 'sol', solution.dat.data[:, 0].min(), solution.dat.data[:, 0].max()


def test_solution():
    target_u_min = 0.385
    target_u_max = 0.632
    target_u_tol = 1e-3
    target_zero = 1e-12
    uvw = solution.dat.data
    w_max = np.max(np.abs(uvw[:, 2]))
    v_max = np.max(np.abs(uvw[:, 1]))
    print 'w', w_max
    print 'v', v_max
    assert w_max < target_zero, 'z velocity component too large'
    assert v_max < target_zero, 'y velocity component too large'
    u_min = uvw[:, 0].min()
    u_max = uvw[:, 0].max()
    print 'u', u_min, u_max
    assert np.abs(u_min - target_u_min) < target_u_tol, 'minimum u velocity is wrong'
    assert np.abs(u_max - target_u_max) < target_u_tol, 'maximum u velocity is wrong'
