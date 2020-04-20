from thetis import *
from firedrake.petsc import PETSc
import os


def bump(fs, locs, scale=1.0):
    """Scaled bump function for turbines."""
    x, y = SpatialCoordinate(fs.mesh())
    i = 0
    for j in range(len(locs)):
        x0 = locs[j][0]
        y0 = locs[j][1]
        r = locs[j][2]
        expr1 = (x-x0)*(x-x0) + (y-y0)*(y-y0)
        expr2 = scale*exp(1 - 1/(1 - (x-x0)*(x-x0)/r**2))*exp(1 - 1/(1 - (y-y0)*(y-y0)/r**2))
        i += conditional(lt(expr1, r*r), expr2, 0)
    return i


def setup_forward(mesh2d, **model_options):
    """
    Consider a simple test case with two turbines positioned in a channel. The mesh has been adapted
    with respect to fluid speed and so has strong anisotropy in the direction of flow.

    If the default SIPG parameter is used, this steady state problem fails to converge. However,
    using the automatic SIPG parameter functionality, it should converge.
    """

    # Create steady state solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(40.0))
    options = solver_obj.options
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.timestepper_type = 'SteadyState'
    options.timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_rtol': 1e-8,
        'snes_monitor': None,
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    options.output_directory = 'outputs'
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.use_grad_div_viscosity_term = False
    options.element_family = 'dg-dg'
    options.horizontal_viscosity = Constant(1.0)
    options.quadratic_drag_coefficient = Constant(0.0025)
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_grad_depth_viscosity_term = False
    options.use_automatic_sipg_parameter = True
    options.update(model_options)
    solver_obj.create_equations()

    # Apply boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {
        1: {'uv': Constant([3.0, 0.0])},
        2: {'elev': Constant(0.0)},
        3: {'un': Constant(0.0)},
    }

    # Set up turbine array
    L = 1000.0       # domain length
    W = 300.0        # domain width
    D = 18.0         # turbine diameter
    A = pi*(D/2)**2  # turbine area
    locs = [(L/2-8*D, W/2, D/2), (L/2+8*D, W/2, D/2)]  # turbine locations

    # NOTE: We include a correction to account for the fact that the thrust coefficient is based
    #       on an upstream velocity, whereas we are using a depth averaged at-the-turbine velocity
    #       (see Kramer and Piggott 2016, eq. (15)).
    correction = 4/(1 + sqrt(1-A/(40.0*D)))**2
    scaling = len(locs)/assemble(bump(solver_obj.function_spaces.P1DG_2d, locs)*dx)
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = bump(solver_obj.function_spaces.P1DG_2d, locs, scale=scaling)
    farm_options.turbine_options.diameter = D
    farm_options.turbine_options.thrust_coefficient = 0.8*correction
    solver_obj.options.tidal_turbine_farms['everywhere'] = farm_options

    # Apply initial guess of inflow velocity and solve
    solver_obj.assign_initial_conditions(uv=Constant([3.0, 0.0]))
    return solver_obj


def solve_adjoint(solution, J, solver_obj, label='adj'):
    ts = solver_obj.timestepper
    V = solution.function_space()
    adjoint_solution = Function(V)
    dFdu = derivative(ts.F, solution, TrialFunction(V))
    dFdu_form = adjoint(dFdu)
    dJdu = derivative(J(solution, solver_obj.options), solution, TestFunction(V))
    with timed_stage(label):
        solve(dFdu_form == dJdu, adjoint_solution, solver_parameters=ts.solver_parameters)
    z, zeta = adjoint_solution.split()
    z.rename("Adjoint fluid speed")
    zeta.rename("Adjoint elevation")
    return adjoint_solution


def power_functional(solution, options):
    uv, elev = split(solution)
    unorm = sqrt(dot(uv, uv))
    J = 0
    for subdomain_id, farm_options in options.tidal_turbine_farms.items():
        density = farm_options.turbine_density
        C_T = farm_options.turbine_options.thrust_coefficient
        A_T = pi * (farm_options.turbine_options.diameter/2.0)**2
        C_D = (C_T * A_T * density)/2.0
        J += C_D * unorm**3 * dx(subdomain_id)

    return J


# Load an anisotropic mesh from file
abspath = os.path.dirname(__file__)
plex = PETSc.DMPlex().create()
plex.createFromFile(os.path.join(abspath, 'anisotropic_plex.h5'))

# Construct base mesh and an iso-P2 refined space, along with transfer operators
mh = MeshHierarchy(Mesh(plex), 1)
mesh_c, mesh_f = mh
prolong, restrict, inject = dmhooks.get_transfer_operators(plex)

# Discretisation parameters, etc.
kwargs = {
    'element_family': 'dg-cg',
    'polynomial_degree': 1,
    'estimate_error': True,
}
di = os.path.join(abspath, 'outputs')
solve_fwd_f = True
# solve_fwd_f = False

# Solve forward
solver_c = setup_forward(mesh_c, output_directory=os.path.join(di, 'coarse'), **kwargs)
with timed_stage('fwd_c'):
    solver_c.iterate()
fwd_c = solver_c.fields.solution_2d

# Evaluate strong residual
error_estimator_c = solver_c.timestepper.error_estimator
residual = error_estimator_c.evaluate_strong_residual()
File(os.path.join(di, 'strong_residual.pvd')).write(*residual.split())

# Solve adjoint
adj_c = solve_adjoint(fwd_c, power_functional, solver_c, label='adj_c')
File(os.path.join(di, 'coarse', 'adjoint.pvd')).write(*adj_c.split())

# Evaluate difference quotient  # TODO: Account for flux term
flux_jump = error_estimator_c.evaluate_flux_jump(adj_c)
diff_quotient = assemble(error_estimator_c.p0test*inner(abs(residual), abs(flux_jump))*dx)
diff_quotient.rename("Difference quotient")
File(os.path.join(di, 'difference_quotient.pvd')).write(diff_quotient)

# Solve/prolong forward in iso-P2 refined space
solver_f = setup_forward(mesh_f, output_directory=os.path.join(di, 'fine'), **kwargs)
fwd_f = solver_f.fields.solution_2d
with timed_stage('fwd_f'):
    if solve_fwd_f:
        solver_f.iterate()     # Solve forward in refined space, or ...
    else:
        prolong(fwd_c, fwd_f)  # ... simply prolong coarse forward solution

# Solve adjoint in refined space
adj_f = solve_adjoint(fwd_f, power_functional, solver_f, label='adj_f')
File(os.path.join(di, 'fine', 'adjoint.pvd')).write(*adj_f.split())

# Prolong into refined space
fwd_proj = Function(solver_f.function_spaces.V_2d)
adj_proj = Function(solver_f.function_spaces.V_2d)
if solve_fwd_f:
    prolong(fwd_c, fwd_proj)
else:
    fwd_proj.assign(fwd_f)
prolong(adj_c, adj_proj)

# Take difference to approximate adjoint error
adj_error = adj_f.copy(deepcopy=True)
adj_error -= adj_proj
z, zeta = adj_error.split()
z.rename("Adjoint fluid speed error")
zeta.rename("Adjoint elevation error")
File(os.path.join(di, 'fine', 'adjoint_error.pvd')).write(z, zeta)

# Set up error estimator in fine space
bcs = solver_f.bnd_functions['shallow_water']
solver_f.timestepper.setup_error_estimator(fwd_proj, adj_error, bcs)
error_estimator = solver_f.timestepper.error_estimator
P0_f = error_estimator.P0_2d

# Evaluate element residual
indicator_c = Function(solver_c.function_spaces.P0_2d)
inject(interpolate(abs(error_estimator.element_residual()), P0_f), indicator_c)
indicator_c.rename("Element residual in modulus")
File(os.path.join(di, 'element_residual.pvd')).write(indicator_c)

# Evaluate inter-element flux
inject(interpolate(abs(error_estimator.inter_element_flux()), P0_f), indicator_c)
indicator_c.rename("Inter-element flux in modulus")
File(os.path.join(di, 'inter_element_flux.pvd')).write(indicator_c)

# Evaluate boundary flux
inject(interpolate(abs(error_estimator.boundary_flux()), P0_f), indicator_c)
indicator_c.rename("Boundary flux in modulus")
File(os.path.join(di, 'boundary_flux.pvd')).write(indicator_c)

# Assemble total error indicator
inject(interpolate(abs(error_estimator.weighted_residual()), P0), indicator_c)
indicator_c.rename("Dual weighted residual")
File(os.path.join(di, 'dwr.pvd')).write(indicator_c)
