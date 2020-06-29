"""
Discrete turbines optimisation example
=======================================
Test 1: Forward example
Test 2: Forward example (change interpolate to project)
Test 3: Taylor test
"""
# to enable a gradient-based optimisation using the adjoint to compute
# gradients, we need to import from thetis_adjoint instead of thetis. This
# ensure all firedrake operations in the Thetis model are annotated
# automatically, in such a way that we can rerun the model with different input
# parameters, and also derive the adjoint-based gradient of a specified input
# (the functional) with respect to a specified input (the control)
from thetis import *
from firedrake_adjoint import *
from pyadjoint import minimize
import numpy
op2.init(log_level=INFO)

#Comment for testing forward model
'''
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # when run as a pytest test, only run 5 timesteps
    # and test the gradient
    t_end = 5*timestep
    test_gradient = True  # test gradient using Taylor test (see below)
    optimise = False  # skip actual gradient based optimisation
else:
    test_gradient = False
    optimise = True
'''
test_gradient = False
optimise = True

### set up the Thetis solver obj as usual ###
mesh2d = Mesh('headland3.msh')

tidal_amplitude = 5.
tidal_period = 12.42*60*60
timestep = 100.
t_end = 5*timestep
#t_end = tidal_period

#set up depth
H = 40

#set viscosity bumps at in-flow boundaries.
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
x = SpatialCoordinate(mesh2d)
h_viscosity = Function(P1_2d).interpolate(conditional(le(x[0], 50), 50.1-x[0], conditional(ge(x[0],1950),x[0]-1949.9,0.1)))
File('outputs/viscosity.pvd').write(h_viscosity)



# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
options.timestep = timestep
options.simulation_export_time = timestep
options.simulation_end_time = t_end
options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
options.element_family = 'dg-cg'
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
# using direct solver as PressurePicard does not work with dolfin-adjoint (due to .split() not being annotated correctly)
options.timestepper_options.solver_parameters = {'snes_monitor': None,
                                                 'snes_rtol': 1e-9,
                                                 'ksp_type': 'preonly',
                                                 'pc_type': 'lu',
                                                 'pc_factor_mat_solver_type': 'mumps',
                                                 'mat_type': 'aij'
                                                 }
options.horizontal_viscosity =h_viscosity
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


# Initialise Discrete turbine farm characteristics
farm_options = DiscreteTidalTurbineFarmOptions()
farm_options.turbine_type = 'constant'
farm_options.turbine_options.thrust_coefficient = 0.6
farm_options.turbine_options.diameter = 20
farm_options.upwind_correction = False

# a list contains the coordinates of all turbines
farm_options.turbine_coordinates = [[Constant(x), Constant(y)] for x in numpy.arange(940, 1061, 60) for y in numpy.arange(260, 341, 40)]

#add turbines to SW_equations
options.discrete_tidal_turbine_farms[2] = farm_options

#set initial condition
solver_obj.assign_initial_conditions(elev=tidal_elev, uv=(as_vector((x[1]/1e5, 0.0))))

# Operation of tidal turbine farm through a callback
cb = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb, 'timestep')

# start computer forward model

solver_obj.iterate(update_forcings=update_forcings)


###set up interest functional and control###
power_output= sum(cb.integrated_power)
interest_functional = power_output

# specifies the control we want to vary in the optimisation
c = [Control(x) for xy in farm_options.turbine_coordinates for x in xy]

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
    #optimisation.ControlsExportOptimisationCallback(solver_obj),
    #optimisation.DerivativesExportOptimisationCallback(solver_obj),
    optimisation.UserExportOptimisationCallback(solver_obj, [solver_obj.fields.turbine_density_2d, solver_obj.fields.uv_2d]),
    optimisation.FunctionalOptimisationCallback(solver_obj),
    #turbines.TurbineOptimisationCallback(solver_obj, cb),
])

# callbacks to indicate start of forward and adjoint runs in log
def eval_cb_pre(controls):
    print_output("FORWARD RUN:")
    print_output("positions: {}".format([float(c) for c in controls]))

def derivative_cb_pre(controls):
    print_output("ADJOINT RUN:")
    print_output("positions: {}".format([float(c) for c in controls]))

# this reduces the functional J(u, m) to a function purely of the control m:
# rf(m) = J(u(m), m) where the velocities u(m) of the entire simulation
# are computed by replaying the forward model for any provided turbine coordinates m
rf = ReducedFunctional(-interest_functional, c, derivative_cb_post=callback_list,
        eval_cb_pre=eval_cb_pre, derivative_cb_pre=derivative_cb_pre)


print(interest_functional)

if test_gradient:
    # whenever the forward model is changed - for example different terms in the equation,
    # different types of boundary conditions, etc. - it is a good idea to test whether the
    # gradient computed by the adjoint is still correct, as some steps in the model may
    # not have been annotated correctly. This can be done via the Taylor test.
    # Using the standard Taylor series, we should have (for a sufficiently smooth problem):
    #   rf(td0+h*dtd) - rf(td0) - < drf/dtd(rf0), h dtd> = O(h^2)

    # we choose a random point in the control space, i.e. a randomized turbine density with
    # values between 0 and 1 and choose a random direction dtd to vary it in

    # this tests whether the above Taylor series residual indeed converges to zero at 2nd order in h as h->0
    m0 = [Constant(950), Constant(320), Constant(1080), Constant(300)]
    h0 = [Constant(10.), Constant(10.), Constant(10.), Constant(10.)]
    minconv = taylor_test(rf, m0, h0)
    print_output("Order of convergence with taylor test (should be 2) = {}".format(minconv))

    assert minconv > 1.95

if optimise:
    # Optimise the control for minimal functional (i.e. maximum profit)
    # with a gradient based optimisation algorithm using the reduced functional
    # to replay the model, and computing its derivative via the adjoint
    # By default scipy's implementation of L-BFGS-B is used, see
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    # options, such as maxiter and pgtol can be passed on.
    td_opt = minimize(rf,
            options={'maxiter': 100, 'pgtol': 1e-3, 'iprint': 1000})
    File('optimal_density.pvd').write(farm_options.turbine_density)
