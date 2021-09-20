"""
Tidal farm optimisation example
=======================================

This example is based on the OpenTidalFarm example:
http://opentidalfarm.readthedocs.io/en/latest/examples/headland-optimization/headland-optimization.html

It optimises the layout of a tidalfarm using the so called continuous approach where
the density of turbines within a farm (n/o turbines per unit area) is optimised. This
allows a.o to include a cost term based on the number of turbines which is computed as
the integral of the density. For more details, see:
  S.W. Funke, S.C. Kramer, and M.D. Piggott, "Design optimisation and resource assessment
  for tidal-stream renewable energy farms using a new continuous turbine approach",
  Renewable Energy 99 (2016), pp. 1046-1061, http://doi.org/10.1016/j.renene.2016.07.039

The optimisation is performed using a gradient-based optimisation algorithm (L-BFGS-B)
where the gradient is computed using the adjoint method. The adjoint is implemented, in
firedrake, via the dolfin-adjoint approach (http://http://www.dolfin-adjoint.org) which annotates
the forward model and automatically derives the adjoint.
"""

from thetis import *
# this import automatically starts the annotation:
from firedrake_adjoint import *
op2.init(log_level=INFO)

# setup the Thetis solver obj as usual:
mesh2d = Mesh('headland.msh')

tidal_amplitude = 5.
tidal_period = 12.42*60*60
H = 40
timestep = 800.

t_end = tidal_period
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # when run as a pytest test, only run 5 timesteps
    # and test the gradient
    t_end = 5*timestep
    test_gradient = True  # test gradient using Taylor test (see below)
    optimise = False  # skip actual gradient based optimisation
else:
    test_gradient = False
    optimise = True


# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
options.timestep = timestep
options.simulation_export_time = timestep
options.simulation_end_time = t_end
options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
options.element_family = 'dg-cg'
options.swe_timestepper_type = 'CrankNicolson'
options.swe_timestepper_options.implicitness_theta = 0.6
# using direct solver as PressurePicard does not work with dolfin-adjoint (due to .split() not being annotated correctly)
options.swe_timestepper_options.solver_parameters = {'snes_monitor': None,
                                                     'snes_rtol': 1e-9,
                                                     'ksp_type': 'preonly',
                                                     'pc_type': 'lu',
                                                     'pc_factor_mat_solver_type': 'mumps',
                                                     'mat_type': 'aij'
                                                     }
options.horizontal_viscosity = Constant(100.0)
options.quadratic_drag_coefficient = Constant(0.0025)

# assign boundary conditions
left_tag = 1
right_tag = 2
coasts_tag = 3
tidal_elev = Function(get_functionspace(mesh2d, "CG", 1), name='tidal_elev')
tidal_elev_bc = {'elev': tidal_elev}
# noslip currently doesn't work (vector Constants are broken in firedrake_adjoint)
freeslip_bc = {'un': Constant(0.0)}
solver_obj.bnd_functions['shallow_water'] = {
    left_tag: tidal_elev_bc,
    right_tag: tidal_elev_bc,
    coasts_tag: freeslip_bc
}

# a function to update the tidal_elev bc value every timestep
x = SpatialCoordinate(mesh2d)
g = 9.81
omega = 2 * pi / tidal_period


def update_forcings(t):
    print_output("Updating tidal elevation at t = {}".format(t))
    tidal_elev.project(tidal_amplitude*sin(omega*t + omega/pow(g*H, 0.5)*x[0]))


# a density function (to be optimised below) that specifies the number of turbines per unit area
turbine_density = Function(get_functionspace(mesh2d, "CG", 1), name='turbine_density')
# associate subdomain_id 2 (as in dx(2)) with a tidal turbine farm
# (implemented via a drag term) with specified turbine density
# Turbine characteristic can be specified via:
# - farm_options.turbine_options.thrust_coefficient (default 0.8)
# - farm_options.turbine_options.diameter (default 16.0)
farm_options = TidalTurbineFarmOptions()
farm_options.turbine_density = turbine_density
# amount of power produced per turbine (kW) on average to "break even" (cost = revenue)
# this is used to scale the cost, which is assumed to be linear with the number of turbines,
# in such a way that the cost is expressed in kW which can be subtracted from the profit
# which is calculated as the power extracted by the turbines
farm_options.break_even_wattage = 200
options.tidal_turbine_farms[2] = farm_options

# we first run the "forward" model with no turbines
turbine_density.assign(0.0)

# create a density restricted to the farm
# the turbine_density, which is the control that will be varied in the optimisation,
# is defined everywhere, but it's influence is restricted to the farm area -
# the turbine drag term is integrated over the farm area only
# Because the turbine density is CG, the nodal values at the farm boundaries
# introduces a jagged edge around the farm where these values are tapered to zero.
# These nonzero values outside the farm itself do not contribute in any of the
# computations. For visualisation purposes we therefore project to a DG field
# restricted to the farm.
farm_density = Function(get_functionspace(mesh2d, "DG", 1), name='farm_density')
projector = SubdomainProjector(turbine_density, farm_density, 2)
projector.project()

cb = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb, 'timestep')


# run as normal (this run will be annotated by firedrake_adjoint)
solver_obj.assign_initial_conditions(uv=as_vector((1e-7, 0.0)), elev=tidal_elev)
solver_obj.iterate(update_forcings=update_forcings)


# compute maximum turbine density (based on a minimum of 1.5D between
# turbines laterally, and 5D in the streamwise direction)
D = farm_options.turbine_options.diameter
max_density = 1./(2.5*D*5*D)
print_output("Maximum turbine density = {}".format(max_density))

# we rescale the functional such that the gradients are ~ order magnitude 1.
# the scaling is based on the maximum cost term
# also we multiply by -1 so that if we minimize the functional, we maximize profit
# (maximize is also availble from pyadjoint but currently broken)
scaling = -1./assemble(max(farm_options.break_even_wattage, 100) * max_density * dx(2, domain=mesh2d))
scaled_functional = scaling * cb.average_profit

# specifies the control we want to vary in the optimisation
c = Control(turbine_density)

# a number of callbacks to provide output during the optimisation iterations:
# - ControlsExportOptimisationCallback export the turbine_friction values (the control)
#            to outputs/control_turbine_friction.pvd. This can also be used to checkpoint
#            the optimisation by using the export_type='hdf5' option.
# - DerivativesExportOptimisationCallback export the derivative of the functional wrt
#            the control as computed by the adjoint to outputs/derivative_turbine_friction.pvd
# - UserExportOptimisationCallback can be used to output any further functions used in the
#            forward model. Note that only function states that contribute to the functional are
#            guaranteed to be updated when the model is replayed for different control values.
# - FunctionalOptimisationCallback simply writes out the (scaled) functional values
# - the TurbineOptimsationCallback outputs the average power, cost and profit (for each
#            farm if multiple are defined)
callback_list = optimisation.OptimisationCallbackList([
    optimisation.ControlsExportOptimisationCallback(solver_obj),
    optimisation.DerivativesExportOptimisationCallback(solver_obj),
    optimisation.UserExportOptimisationCallback(solver_obj, (farm_density,)),
    optimisation.FunctionalOptimisationCallback(solver_obj),
    turbines.TurbineOptimisationCallback(solver_obj, cb),
])

# anything that follows, is no longer annotated:
pause_annotation()

# this reduces the functional J(u, td) to a function purely of the control td:
# rf(td) = J(u(td), td) where the velocities u(td) of the entire simulation
# are computed by replaying the forward model for any provided turbine density td
rf = ReducedFunctional(scaled_functional, c, derivative_cb_post=callback_list)


if test_gradient:
    # whenever the forward model is changed - for example different terms in the equation,
    # different types of boundary conditions, etc. - it is a good idea to test whether the
    # gradient computed by the adjoint is still correct, as some steps in the model may
    # not have been annotated correctly. This can be done via the Taylor test.
    # Using the standard Taylor series, we should have (for a sufficiently smooth problem):
    #   rf(td0+h*dtd) - rf(td0) - < drf/dtd(rf0), h dtd> = O(h^2)

    # we choose a random point in the control space, i.e. a randomized turbine density with
    # values between 0 and 1 and choose a random direction dtd to vary it in
    td0 = Function(turbine_density)
    dtd = Function(turbine_density)
    numpy.random.seed(42)  # set seed to make test deterministic
    td0.dat.data[:] = numpy.random.random(td0.dat.data.shape)
    dtd.dat.data[:] = numpy.random.random(dtd.dat.data.shape)

    # this tests whether the above Taylor series residual indeed converges to zero at 2nd order in h as h->0
    minconv = taylor_test(rf, td0, dtd)
    print_output("Order of convergence with taylor test (should be 2) = {}".format(minconv))

    assert minconv > 1.95

if optimise:
    # Optimise the control for minimal functional (i.e. maximum profit)
    # with a gradient based optimisation algorithm using the reduced functional
    # to replay the model, and computing its derivative via the adjoint
    # By default scipy's implementation of L-BFGS-B is used, see
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    # options, such as maxiter and pgtol can be passed on.
    td_opt = minimise(rf, bounds=[0, max_density],
                      options={'maxiter': 100, 'pgtol': 1e-3})
    File('optimal_density.pvd').write(td_opt)
