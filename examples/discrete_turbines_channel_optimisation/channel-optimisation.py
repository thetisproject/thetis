"""
Basic example of discrete turbine optimisation in a channel
"""
from thetis import *
# this import automatically starts the annotation:
from firedrake_adjoint import *
from pyadjoint import minimize
import numpy
op2.init(log_level=INFO)

test_gradient = False
optimise = True

### set up the Thetis solver obj as usual ###
mesh2d = Mesh('mesh.msh')

# choose a depth
H = 50
h_viscosity = Constant(2.)

# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
# for a steady state solve the timestep is not relevant
# we just have to ensure that we perform exactly one timestep
options.timestep = 1.
options.simulation_export_time = 1.
options.simulation_end_time = 0.5

options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
options.element_family = 'dg-cg'
options.timestepper_type = 'SteadyState'
# for steady state we use a direct solve (preonly+lu) in combination with a Newton snes solver
# (this is partly the default, just switching on mumps here (TODO: which should probaly be added to defaults?)
#  and a snes monitor that displays the residual of the Newton iteration)
options.timestepper_options.solver_parameters = {'snes_monitor': None,
                                                 'snes_rtol': 1e-5,
                                                 'ksp_type': 'preonly',
                                                 'pc_type': 'lu',
                                                 'pc_factor_mat_solver_type': 'mumps',
                                                 'mat_type': 'aij'
                                                 }
options.horizontal_viscosity = h_viscosity

# TODO: check do I still need these:
options.use_automatic_sipg_parameter = False
options.sipg_parameter = Constant(100.)

options.quadratic_drag_coefficient = Constant(0.0025)

# assign boundary conditions
left_tag = 1
right_tag = 2
coasts_tag = 3

u_inflow = 2.0
inflow_bc = {'uv': Constant((u_inflow, 0.0))}
outflow_bc = {'elev': 0.}
freeslip_bc = {'un': 0.}

solver_obj.bnd_functions['shallow_water'] = {
    left_tag: inflow_bc,
    right_tag: outflow_bc,
    coasts_tag: freeslip_bc
}

# initialise discrete turbine farm characteristics
farm_options = DiscreteTidalTurbineFarmOptions()
farm_options.turbine_type = 'constant'
farm_options.turbine_options.thrust_coefficient = 0.8
farm_options.turbine_options.diameter = 20
# TODO: check, does this impact optimisation?
farm_options.upwind_correction = False

site_x = 320.
site_y = 160.
site_x_start = 160.
site_y_start = 80.
r = farm_options.turbine_options.diameter/2.

# a list contains the coordinates of all turbines: regular, non-staggered 4 x 2 layout
farm_options.turbine_coordinates = [[Constant(x+cos(y)), Constant(y+numpy.sin(x))]
        for x in np.linspace(site_x_start + 4*r, site_x_start + site_x - 4*r, 4)
        for y in np.linspace(site_y_start + 0.5*site_y-2*r, site_y_start + 0.5*site_y + 2*r, 2)]

# apply these turbine farm settings to subdomain 2
# this means that the mesh needs to have a Physical Surface id of 2 (see mesh.geo file)
# attached to the part of the domain where the turbines are operational
# Any turbines placed outside this subdomain will not produce any power (the corresponding
# integral is only performed over the subdomain). If you want the turbines to be operational
# everywhere, you can use the key "everywhere" instead of 2 below.
# Regardless of this, the area where the turbines are places need to have sufficiently high
# resolution to resolve the Gaussian bump functions that represent the turbines
options.discrete_tidal_turbine_farms[2] = farm_options

# set initial condition (initial velocity should not be exactly 0 to avoid failures in the Newton solve)
solver_obj.assign_initial_conditions(uv=(as_vector((1e-3, 0.0))))

# Operation of tidal turbine farm through a callback
# This can be used for diagnostic output (to the log and hdf5 file)
# but here is also used to define the functional that we optimised for
# through cb.integrated_power
cb = turbines.TurbineFunctionalCallback(solver_obj)
solver_obj.add_callback(cb, 'timestep')

# run forward model
solver_obj.iterate()


# set up interest functional and control
power_output= sum(cb.integrated_power)
interest_functional = power_output

print_output("Functional in forward model {}".format(interest_functional))

# specifies the control we want to vary in the optimisation
# this needs to be a flat list of controls corresponding to the x and y coordinates
# of the turbines
c = [Control(x) for xy in farm_options.turbine_coordinates for x in xy]

# interpolate the turbine density (a sum of Gaussian bump functions representing the turbines)
# to a P1 CG function. This is not used here, but will be used in the UserExportOptimisationCallback
# below to output the farm layouts as a series of vtus
turbine_density = Function(solver_obj.function_spaces.P1_2d, name='turbine_density')
turbine_density.interpolate(solver_obj.tidal_farms[0].turbine_density)

# a number of callbacks to provide output during the optimisation iterations:
# - ConstantControlOptimisationCallback - outputs the control values to
#    the log and hdf5 files (can be controled by append_to_log and export_to_hdf5 keyword arguments)
# - DerivativesExportOptimisationCallback - outputs the derivatives of the functional
#    with respect to the control values (here the x,y coordinates of the turbines) as calculated by the adjoint
# - UserExportOptimisationCallback can be used to output any further functions used in the
#            forward model. Note that only function states that contribute to the functional are
#            guaranteed to be updated when the model is replayed for different control values.
# - FunctionalOptimisationCallback simply writes out the functional values to log and hdf5
#
# finally, the OptimisationCallbackList combines multiple optimisation callbacks in one
callback_list = optimisation.OptimisationCallbackList([
    optimisation.ConstantControlOptimisationCallback(solver_obj, array_dim=len(c)),
    optimisation.DerivativeConstantControlOptimisationCallback(solver_obj, array_dim=len(c)),
    optimisation.UserExportOptimisationCallback(solver_obj, [turbine_density, solver_obj.fields.uv_2d]),
    optimisation.FunctionalOptimisationCallback(solver_obj),
])

# here we define some additional callbacks just to clearly indicate in the log what the model is doing:
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
# with rf.derivative() we can also compute the derivative with respect to the controls
# which is done through the automated adjoint
# Through the eval_cb_pre/post and derivative_cb_post arguments we can specify callbacks
# that are called at the beginning (_pre) and after (_post) the evaluation of the forward model (eval)
# and the derivative/adjoint model respectively
# NOTE, that we use -interest_functional so that we can *minimize* this reduced functional
# to maximize the power output
rf = ReducedFunctional(-interest_functional, c, derivative_cb_post=callback_list,
        eval_cb_pre=eval_cb_pre, derivative_cb_pre=derivative_cb_pre)

if test_gradient:
    # whenever the forward model is changed - for example different terms in the equation,
    # different types of boundary conditions, etc. - it is a good idea to test whether the
    # gradient computed by the adjoint is still correct, as some steps in the model may
    # not have been annotated correctly. This can be done via the Taylor test.
    # Using the standard Taylor series, we should have (for a sufficiently smooth problem):
    #   rf(td0+h*dtd) - rf(td0) - < drf/dtd(rf0), h dtd> = O(h^2)

    # we choose the same starting layout but with a small perturbation
    m0 = [Constant(x + random.uniform(-r, r)) for xy in farm_options.turbine_coordinates for x in xy]

    # the perturbation over which we test the Taylor approximation
    # (the taylor test below starts with a 1/100th of that, followed by a series of halvings
    h0 = [Constant(random.random(-1, 1)) for xy in farm_options.turbine_coordinates for x in xy]

    # this tests whether the above Taylor series residual indeed converges to zero at 2nd order in h as h->0
    minconv = taylor_test(rf, m0, h0)
    print_output("Order of convergence with taylor test (should be 2) = {}".format(minconv))

    assert minconv > 1.95

if optimise:
    # Optimise the control (turbine positions) for minimal functional (i.e. maximum profit)
    # with a gradient based optimisation algorithm using the reduced functional
    # to replay the model, and computing its derivative via the adjoint

    # There are two types of constraints we need here:
    # - first we don't want the turbines to leave the specified (high resolution)
    #   farm area. Since this area is rectangular we can specify this through
    #   simple box constraints: lb < c < ub
    lb = np.array([[site_x_start+r, site_y_start+r] for _ in farm_options.turbine_coordinates]).flatten()
    ub = np.array([[site_x_start+site_x-r, site_y_start+site_y-r] for _ in farm_options.turbine_coordinates]).flatten()

    # - secondly, we don't want the turbines to be placed arbitrary close. This is enforced
    #   through more general constraints of the form h(m)>0 where here
    #   we use h(m) = [dist((x1,y1) to (x2,y2))**2 - min_dist**2 for any combination of turbines]
    #   the MinimumDistanceConstraints implements this constraint (and its derivative), here
    #   with a minimum distantce min_dist=25
    mdc = turbines.MinimumDistanceConstraints(farm_options.turbine_coordinates, 25.)

    # finally, the optimisation call. Here, we can't use the default (L-BFGS-B) which only handles
    # box constraints, but use SLSQP which also handles more general constraints
    # see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
    # for further options
    td_opt = minimize(rf, method='SLSQP', constraints=mdc, bounds=[lb, ub],
            options={'maxiter': 300, 'ftol': 1e-06})
