"""
Tests implicit bottom friction formulation
==========================================

Intended to be executed with pytest.
"""
from firedrake import *
from thetis.utility import get_functionspace
import time as time_mod
import numpy

op2.init(log_level=WARNING)


def test_implicit_friction(do_export=False, do_assert=True):
    # set mesh resolution
    scale = 1000.0
    reso = 2.5*scale
    layers = 50
    depth = 15.0

    # generate unit mesh and transform its coords
    x_max = 5.0*scale
    x_min = -5.0*scale
    lx = (x_max - x_min)
    n_x = int(lx/reso)
    mesh2d = RectangleMesh(n_x, n_x, lx, lx, reorder=True)
    # move mesh, center to (0,0)
    mesh2d.coordinates.dat.data[:, 0] -= lx/2
    mesh2d.coordinates.dat.data[:, 1] -= lx/2

    mesh = ExtrudedMesh(mesh2d, layers=50, layer_height=-depth/layers)

    if do_export:
        out_file = File('implicit_bf_sol.pvd')

    # ----- define function spaces
    deg = 1
    p1dg = get_functionspace(mesh, 'DG', 1)
    p1dgv = get_functionspace(mesh, 'DG', 1, vector=True)
    u_h_elt = FiniteElement('RT', triangle, deg + 1, variant='point')
    u_v_elt = FiniteElement('DG', interval, deg, variant='equispaced')
    u_elt = HDiv(TensorProductElement(u_h_elt, u_v_elt))
    # for vertical velocity component
    w_h_elt = FiniteElement('DG', triangle, deg, variant='equispaced')
    w_v_elt = FiniteElement('CG', interval, deg + 1, variant='equispaced')
    w_elt = HDiv(TensorProductElement(w_h_elt, w_v_elt))
    # in deformed mesh horiz. velocity must actually live in U + W
    uw_elt = EnrichedElement(u_elt, w_elt)
    # final spaces
    v = FunctionSpace(mesh, uw_elt)  # uv

    solution = Function(v, name='velocity')
    solution_new = Function(v, name='new velocity')
    solution_p1_dg = Function(p1dgv, name='velocity p1dg')
    viscosity_v = Function(p1dg, name='viscosity')
    elev_slope = -1.0e-5
    source = Constant((-9.81*elev_slope, 0, 0))

    z0 = 1.5e-3
    kappa = 0.4
    drag = (kappa / numpy.log((depth/layers)/z0))**2
    bottom_drag = Constant(drag)
    u_bf = 0.035  # NOTE tuned to produce ~correct viscosity profile

    x, y, z = SpatialCoordinate(mesh)
    viscosity_v.project(kappa * u_bf * -z * (depth + z + z0) / (depth + z0))
    print('Cd {:}'.format(drag))
    print('u_bf {:}'.format(u_bf))
    print('nu {:} - {:}'.format(viscosity_v.dat.data.min(), viscosity_v.dat.data.max()))

    # --- solve mom eq
    test = TestFunction(v)
    normal = FacetNormal(mesh)

    def rhs(solution, sol_old):
        # source term (external pressure gradient
        f = inner(source, test)*dx
        # vertical diffusion (integrated by parts)
        f += -viscosity_v*inner(Dx(solution, 2), Dx(test, 2)) * dx
        # interface term
        diff_flux = viscosity_v*Dx(solution, 2)
        f += (dot(avg(diff_flux), test('+'))*normal[2]('+')
              + dot(avg(diff_flux), test('-'))*normal[2]('-')) * dS_h
        # symmetric interior penalty stabilization
        l = Constant(depth/layers)
        nb_neigh = 2
        o = 1
        d = 3
        sigma = Constant((o + 1)*(o + d)/d * nb_neigh / 2) / l
        gamma = sigma*avg(viscosity_v)
        f += gamma * dot(jump(solution), test('+')*normal[2]('+') + test('-')*normal[2]('-')) * dS_h
        # boundary term
        uv_bot_old = sol_old + Dx(sol_old, 2)*l*0.5
        uv_bot = solution + Dx(solution, 2)*l*0.5  # solver fails
        uv_mag = sqrt(uv_bot_old[0]**2 + uv_bot_old[1]**2) + Constant(1e-12)
        bnd_flux = bottom_drag*uv_mag*uv_bot
        ds_bottom = ds_t
        f += dot(bnd_flux, test)*normal[2] * ds_bottom

        return f

    # ----- define solver

    sp = {}

    dt = 3600.0
    time_steps = 13
    dt_const = Constant(dt)

    # Backward Euler
    f = (inner(solution_new, test)*dx - inner(solution, test)*dx
         - dt_const*rhs(solution_new, solution))
    prob = NonlinearVariationalProblem(f, solution_new)
    solver = LinearVariationalSolver(prob, solver_parameters=sp)

    if do_export:
        out_file.write(solution)
    # ----- solve
    t = 0
    for it in range(1, time_steps + 1):
        t = it*dt
        t0 = time_mod.perf_counter()
        solver.solve()
        solution.assign(solution_new)
        t1 = time_mod.perf_counter()

        if do_export:
            out_file.write(solution)
        print('{:4d}  T={:9.1f} s  cpu={:.2f} s'.format(it, t, t1-t0))

    if do_assert:
        target_u_min = 0.4
        target_u_max = 1.0
        target_u_tol = 5e-2
        target_zero = 1e-6
        solution_p1_dg.project(solution)
        uvw = solution_p1_dg.dat.data
        w_max = numpy.max(numpy.abs(uvw[:, 2]))
        v_max = numpy.max(numpy.abs(uvw[:, 1]))
        print('w {:}'.format(w_max))
        print('v {:}'.format(v_max))
        assert w_max < target_zero, 'z velocity component too large'
        assert v_max < target_zero, 'y velocity component too large'
        u_min = uvw[:, 0].min()
        u_max = uvw[:, 0].max()
        print('u {:} {:}'.format(u_min, u_max))
        assert numpy.abs(u_min - target_u_min) < target_u_tol, 'minimum u velocity is wrong'
        assert numpy.abs(u_max - target_u_max) < target_u_tol, 'maximum u velocity is wrong'
        print('*** PASSED ***')


if __name__ == '__main__':
    test_implicit_friction()
