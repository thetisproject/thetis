"""
Tests implicit vertical diffusion on a DG vector field
======================================================

Intended to be executed with pytest.
"""
from firedrake import *
from thetis.utility import get_functionspace
import numpy

op2.init(log_level=WARNING)


def test_implicit_diffusion(do_export=False, do_assert=True):

    # set mesh resolution
    scale = 1000.0
    reso = 2.5*scale
    layers = 25  # 50
    depth = 50

    # generate unit mesh and transform its coords
    x_max = 5.0*scale
    x_min = -5.0*scale
    lx = (x_max - x_min)
    ly = 0.7*lx
    n_x = int(lx/reso)
    area = lx*ly
    mesh2d = RectangleMesh(n_x, n_x, lx, lx, reorder=True)
    # move mesh, center at (0,0)
    mesh2d.coordinates.dat.data[:, 0] -= lx/2
    mesh2d.coordinates.dat.data[:, 1] -= lx/2
    dz = depth/layers
    v_elem_size = Constant(dz)

    mesh = ExtrudedMesh(mesh2d, layers=layers, layer_height=-depth/layers)

    if do_export:
        sol_file = File('sol.pvd')
        ana_file = File('ana_sol.pvd')

    # define function spaces
    fam = 'DG'
    deg = 1
    fs = get_functionspace(mesh, fam, deg)

    solution = Function(fs, name='tracer')
    solution_new = Function(fs, name='new tracer')
    ana_sol = Function(fs, name='analytical tracer')

    nu_v = 1.0
    diffusivity_v = Constant(nu_v)

    test = TestFunction(fs)
    normal = FacetNormal(mesh)

    # setup simulation time
    t_init = 5.0
    t = t_init
    t_const = Constant(t)
    t_end = 100.0
    dt = dz*dz/nu_v / 10
    n_iter = numpy.ceil(t_end/dt)
    dt = t_end/n_iter
    print('dt {:}'.format(dt))
    dt_const = Constant(dt)

    # analytical solution
    u_max = 1.0
    u_min = -1.0
    z0 = -depth/2.0
    x, y, z = SpatialCoordinate(mesh)
    ana_sol_expr = 0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((z - z0)/sqrt(4*diffusivity_v*t_const))

    # initial condition
    solution.project(ana_sol_expr)
    ana_sol.project(ana_sol_expr)

    def rhs(solution):
        # vertical diffusion operator integrated by parts
        # f = -diffusivity_v*inner(Dx(solution, 2), Dx(test, 2)) * dx
        # interface term
        # diffFlux = diffusivity_v*Dx(solution, 2)
        # f += (dot(avg(diffFlux), test('+'))*normal[2]('+') +
        #      dot(avg(diffFlux), test('-'))*normal[2]('-')) * dS_h
        # symmetric interior penalty stabilization
        # L = Constant(depth/layers)
        # nbNeigh = 2
        # d = 3
        # sigma = Constant((deg + 1)*(deg + d)/d * nbNeigh / 2) / L
        # gamma = sigma*avg(diffusivity_v)
        # jump_test = test('+')*normal[2]('+') + test('-')*normal[2]('-')
        # f += gamma * dot(jump(solution), jump_test) * dS_h

        def grad_v(a):
            return as_vector((0, 0, Dx(a, 2)))

        n = as_vector((0, 0, normal[2]))
        grad_test = grad_v(test)
        diff_flux = diffusivity_v*grad_v(solution)
        diff_flux_jump = diffusivity_v*jump(solution, n)

        f = -inner(grad_test, diff_flux)*dx

        # Interior penalty as in horizontal case
        degree_h, degree_v = fs.ufl_element().degree()
        dim = 3.0
        sigma = (degree_v + 1.0)*(degree_v + dim)/dim/v_elem_size
        # sigma = 1.0/v_elem_size
        alpha = avg(sigma)
        ds_interior = (dS_h)
        f += -alpha*inner(jump(test, n), diff_flux_jump)*ds_interior
        f += +inner(avg(grad_test), diff_flux_jump)*ds_interior
        f += +inner(jump(test, n), avg(diff_flux))*ds_interior

        # symmetric boundary terms
        f += inner(test, dot(diff_flux, n))*(ds_t + ds_b)

        return f

    # define solver
    sp = {}
    sp['ksp_atol'] = 1e-20
    sp['ksp_rtol'] = 1e-20
    sp['snes_rtol'] = 1e-20
    sp['snes_atol'] = 1e-20

    # sp['ksp_monitor'] = True
    # sp['ksp_monitor_true_residual'] = True
    # sp['snes_converged_reason'] = True
    # sp['ksp_converged_reason'] = True

    # Cr-Ni
    # f = (inner(solution_new, test)*dx - inner(solution, test)*dx -
    #      dt_const*RHS(0.5*solution + 0.5*solution_new))
    # prob = NonlinearVariationalProblem(f, solution_new)
    # solver = LinearVariationalSolver(prob, solver_parameters=sp)

    # Backward Euler
    # f = (inner(solution_new, test)*dx - inner(solution, test)*dx -
    #      dt_const*RHS(solution_new))
    # prob = NonlinearVariationalProblem(f, solution_new)
    # solver = LinearVariationalSolver(prob, solver_parameters=sp)

    # From DIRK(2,3,2) IMEX scheme in Ascher et al. (1997)
    # This method has the Butcher tableau
    #
    # gamma   | gamma     0
    # 1       | 1-gamma  gamma
    # -------------------------
    #         | 0.5       0.5
    # with
    # gamma = (2 + sqrt(2))/2
    #
    solution_1 = Function(fs, name='tracer K1')
    solution_2 = Function(fs, name='tracer K2')
    gamma = Constant((2.0 + numpy.sqrt(2.0))/2.0)

    f1 = (inner(solution_1, test)*dx - inner(solution, test)*dx
          - gamma*dt_const*rhs(solution_1))
    prob1 = NonlinearVariationalProblem(f1, solution_1)
    solver1 = LinearVariationalSolver(prob1, solver_parameters=sp)

    f2 = (inner(solution_2, test)*dx - inner(solution, test)*dx
          - (1.0-gamma)*dt_const*rhs(solution_1)
          - (gamma)*dt_const*rhs(solution_2))
    prob2 = NonlinearVariationalProblem(f2, solution_2)
    solver2 = LinearVariationalSolver(prob2, solver_parameters=sp)

    f = (inner(solution_new, test)*dx - inner(solution, test)*dx
         - dt_const*(0.5*rhs(solution_1) + 0.5*rhs(solution_2)))
    prob = NonlinearVariationalProblem(f, solution_new)
    solver = LinearVariationalSolver(prob, solver_parameters=sp)

    if do_export:
        sol_file.write(solution)
        ana_file.write(ana_sol)
    print('sol {:} {:}'.format(solution.dat.data.min(), solution.dat.data.max()))
    print('ana {:} {:}'.format(ana_sol.dat.data.min(), ana_sol.dat.data.max()))
    # time loop
    while t < t_end + t_init:
        # solve
        solver1.solve()
        solver2.solve()
        solver.solve()
        solution.assign(solution_new)
        t += dt

    # update analytical solution
    t_const.assign(t)
    ana_sol.project(ana_sol_expr)
    if do_export:
        sol_file.write(solution)
        ana_file.write(ana_sol)
    print('sol {:} {:}'.format(solution.dat.data.min(), solution.dat.data.max()))
    print('ana {:} {:}'.format(ana_sol.dat.data.min(), ana_sol.dat.data.max()))

    l2_err = errornorm(ana_sol, solution)/area
    print('L2 error: {:}'.format(l2_err))
    if do_assert:
        l2_threshold = 1e-4
        assert l2_err < l2_threshold, 'L2 error exceeds threshold'


if __name__ == '__main__':
    test_implicit_diffusion(do_export=True, do_assert=True)
