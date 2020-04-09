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


def solve_forward(mesh2d, **model_options):
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
    solver_obj.iterate()
    return solver_obj


def solve_adjoint(solution, J, solver_obj):
    ts = solver_obj.timestepper
    V = solution.function_space()
    adjoint_solution = Function(V)
    dFdu = derivative(ts.F, solution, TrialFunction(V))
    dFdu_form = adjoint(dFdu)
    dJdu = derivative(J(solution, solver_obj.options), solution, TestFunction(V))
    solve(dFdu_form == dJdu, adjoint_solution, solver_parameters=ts.solver_parameters)
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
plex = PETSc.DMPlex().create()
plex.createFromFile(os.path.join(os.path.dirname(__file__), 'anisotropic_plex.h5'))
mh = MeshHierarchy(Mesh(plex), 1)
mesh = mh[0]
refined_mesh = mh[1]

# Solve forward
solver_obj_coarse = solve_forward(mesh, element_family='dg-cg', polynomial_degree=1,
                                  output_directory='outputs/coarse')
fwd_coarse = solver_obj_coarse.fields.solution_2d

# Solve adjoint
adj_coarse = solve_adjoint(fwd_coarse, power_functional, solver_obj_coarse)
z, zeta = adj_coarse.split()
z.rename("Adjoint fluid speed")
zeta.rename("Adjoint elevation")
File('outputs/coarse/adjoint.pvd').write(z, zeta)

# Solve forward in refined space
solver_obj_fine = solve_forward(refined_mesh, element_family='dg-cg', polynomial_degree=1,
                                output_directory='outputs/fine')
fwd_fine = solver_obj_fine.fields.solution_2d

# Solve adjoint in refined space
adj_fine = solve_adjoint(fwd_fine, power_functional, solver_obj_fine)
z, zeta = adj_fine.split()
z.rename("Adjoint fluid speed")
zeta.rename("Adjoint elevation")
File('outputs/fine/adjoint.pvd').write(z, zeta)

# Prolong into refined space
fwd_proj = Function(solver_obj_fine.function_spaces.V_2d)
adj_proj = Function(solver_obj_fine.function_spaces.V_2d)
prolong(fwd_coarse, fwd_proj)
prolong(adj_coarse, adj_proj)
adj_error = adj_fine.copy(deepcopy=True)
adj_error -= adj_proj
z, zeta = adj_error.split()
z.rename("Adjoint fluid speed")
zeta.rename("Adjoint elevation")
File('outputs/fine/adjoint_error.pvd').write(z, zeta)

# Set up error estimator
fields = solver_obj_fine.timestepper.fields
V_2d = solver_obj_fine.function_spaces.V_2d
bathymetry = solver_obj_fine.fields.bathymetry_2d
options = solver_obj_fine.options
error_estimator = error_estimation_2d.ShallowWaterErrorEstimator(V_2d, bathymetry, options)
args = ('all', fwd_proj, fwd_proj, adj_error, adj_error, fields, fields)

# Compute element residual
residual = error_estimator.element_residual(*args)
indicator_coarse = Function(solver_obj_coarse.function_spaces.P0_2d)
inject(interpolate(abs(residual), error_estimator.P0), indicator_coarse)
indicator_coarse.rename("Element residual in modulus")
File('outputs/element_residual.pvd').write(indicator_coarse)
dwr_fine = residual.copy(deepcopy=True)

# Compute inter-element flux
flux = error_estimator.inter_element_flux(*args)
inject(interpolate(abs(flux), error_estimator.P0), indicator_coarse)
indicator_coarse.rename("Inter-element flux in modulus")
File('outputs/inter_element_flux.pvd').write(indicator_coarse)
dwr_fine += flux

# Compute boundary flux
args += (solver_obj_fine.bnd_functions['shallow_water'],)
bnd_flux = error_estimator.boundary_flux(*args)
inject(interpolate(abs(bnd_flux), error_estimator.P0), indicator_coarse)
indicator_coarse.rename("Boundary flux in modulus")
File('outputs/boundary_flux.pvd').write(indicator_coarse)
dwr_fine += bnd_flux

# Assemble total error indicator
dwr_coarse = Function(solver_obj_coarse.function_spaces.P0_2d, name="Dual weighted residual")
inject(interpolate(abs(dwr_fine), error_estimator.P0), dwr_coarse)
File('outputs/dwr.pvd').write(dwr_coarse)
