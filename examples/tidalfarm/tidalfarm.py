# Tidal farm optimisation example
# =======================================
#
# This example is based on the OpenTidalFarm example:
# http://opentidalfarm.readthedocs.io/en/latest/examples/headland-optimization/headland-optimization.html
#
# It optimises the layout of a tidalfarm using the so called continuous approach where
# the density of turbines within a farm (n/o turbines per unit area) is optimised. This
# allows a.o to include a cost term based on the number of turbines which is computed as
# the integral of the density. For more details, see:
#   S.W. Funke, S.C. Kramer, and M.D. Piggott, "Design optimisation and resource assessment
#   for tidal-stream renewable energy farms using a new continuous turbine approach",
#   Renewable Energy 99 (2016), pp. 1046-1061, http://doi.org/10.1016/j.renene.2016.07.039

# to enable a gradient-based optimisation using the adjoint to compute gradients,
# we need to import from thetis_adjoint instead of thetis
from thetis_adjoint import *
op2.init(log_level=INFO)

parameters['coffee'] = {}  # temporarily disable COFFEE due to bug

test_gradient = False # whether to check the gradient computed by the adjoint
optimise = True

# setup the Thetis solver obj as usual:
mesh2d = Mesh('headland.msh')

tidal_amplitude = 5.
tidal_period = 12.42*60*60
H = 40
timestep = tidal_period/50

# create solver and set options
solver_obj = solver2d.FlowSolver2d(mesh2d, Constant(H))
options = solver_obj.options
options.timestep = timestep
options.simulation_export_time = timestep
options.simulation_end_time = tidal_period/2
options.output_directory = 'outputs'
options.check_volume_conservation_2d = True
solver_obj.options.element_family = 'dg-cg'
solver_obj.options.timestepper_type = 'PressureProjectionPicard'
solver_obj.options.timestepper_options.implicitness_theta = 0.6
options.horizontal_viscosity = Constant(2.0)
options.quadratic_drag_coefficient = Constant(0.0025)

# assign boundary conditions
left_tag = 1
right_tag = 2
coasts_tag = 3
tidal_elev = Function(FunctionSpace(mesh2d, "CG", 1), name='tidal_elev')
tidal_elev_bc = {'elev': tidal_elev}
noslip_bc = {'uv': Constant((0.0, 0.0))}
solver_obj.bnd_functions['shallow_water'] = {
        left_tag: tidal_elev_bc,
        right_tag: tidal_elev_bc,
        coasts_tag: noslip_bc
}


# first setup all the usual SWE terms
solver_obj.create_equations()

# defines an additional turbine drag term to the SWE
turbine_friction = Function(FunctionSpace(mesh2d, "CG", 1), name='turbine_friction')
class TurbineDragTerm(shallowwater_eq.ShallowWaterMomentumTerm):
    def residual(self, uv, eta, uv_old, eta_old, fields, fields_old, bnd_conditions=None):
        total_h = self.get_total_depth(eta_old)
        C_D = turbine_friction
        f = C_D * sqrt(dot(uv_old, uv_old)) * inner(self.u_test, uv) / total_h * self.dx(2)
        return -f

# add it to the shallow water equations
fs = solver_obj.fields.solution_2d.function_space()
u_test, eta_test = TestFunctions(fs)
u_space, eta_space = fs.split()
turbine_drag_term = TurbineDragTerm(u_test, u_space, eta_space,
        bathymetry=solver_obj.fields.bathymetry_2d)
solver_obj.eq_sw.add_term(turbine_drag_term, 'implicit')

turbine_friction.assign(0.0)
solver_obj.assign_initial_conditions(uv=as_vector((1e-7, 0.0)))

# Setup the functional. It computes a measure of the profit as the difference
# of the power output of the farm (the "revenue") minus the cost based on the number
# of turbines

u, eta = split(solver_obj.fields.solution_2d)
# should multiply this by density to get power in W - assuming rho=1000 we get kW instead
power_integral = turbine_friction * (u[0]*u[0] + u[1]*u[1])**1.5 * dx(2)

# turbine friction=C_T*A_T/2.*turbine_density
C_T = 0.8 # turbine thrust coefficient
A_T = pi * (16./2)**2 # turbine cross section
# cost integral is n/o turbines = \int turbine_density = \int c_t/(C_T A_T/2.)
cost_integral = 1./(C_T*A_T/2.) * turbine_friction * dx(2)

break_even_wattage = 10 # (kW) amount of power produced per turbine on average to "break even" (cost = revenue)

# we rescale the functional such that the gradients are ~ order magnitude 1.
# the scaling is chosen such that the gradient of break_even_wattage * cost_integral is of order 1
# the power-integral is assumed to be of the same order of magnitude
scaling = 1./assemble(break_even_wattage/(C_T*A_T/2.) * dx(2, domain=mesh2d))
combined_functional = Functional(scaling * (power_integral - break_even_wattage * cost_integral) * dt)


# a function to update the tidal_elev bc value every timestep
# we also use it to display the profit each time step (which will be a time-integrated into the functional)
x = SpatialCoordinate(mesh2d)
g = 9.81
omega = 2 * pi / tidal_period

def update_forcings(t, annotate=True):
    print "Updating tidal elevation at t = ", t
    P = assemble(power_integral)
    N = assemble(cost_integral)
    print "Power, N turbines, profit =", P, N, P-break_even_wattage*N
    # NOTE: we need to explicitly tell dolfin-adjoint to annotate this as by default it seems to
    # only annotate if the interpoland is a Function (is this reasonable?)
    tidal_elev.interpolate(tidal_amplitude*sin(omega*t + omega/pow(g*H, 0.5)*x[0]), annotate=annotate)
    # NOTE: it seems firedrake-adjoint (dolfin-adjoint in general?) cannot handle functions that are part of a
    # time-integrated Functional which do not change over time and are therefore not annotated
    turbine_friction.project(turbine_friction, annotate=annotate)


# run as normal (this run will be annotated by firedrake-adjoint)
solver_obj.iterate(update_forcings=update_forcings)


# everything relevant should be annotated now
parameters["adjoint"]["stop_annotating"] = True
# write out the annotation for debugging purposes
adj_html('forward.html', 'forward')

# our own version of a ReducedFunctional, which when asked
# to compute its derivative, calls the standard derivative()
# method of ReducedFunctional but additionaly outputs that
# gradient and the current value of the control to a .pvd
tfpvd = File('turbine_friction.pvd')
class MyReducedFunctional(ReducedFunctional):
    def derivative(self, **kwargs):
        dj = super(MyReducedFunctional, self).derivative(**kwargs)
        # need to make sure dj always has the same name in the output
        grad = dj[0].copy()
        grad.rename("Gradient")
        # same thing for the control
        tf = self.controls[0].data().copy()
        tf.rename('TurbineFriction')
        tfpvd.write(grad, tf)
        return dj

# this reduces the functional J(u, tf) to a function purely of
# rf(tf) = J(u(tf), tf) where the velocities u(tf) of the entire simulation
# are computed by replaying the forward model for any provided turbine friction tf
c = Control(turbine_friction)
rf = MyReducedFunctional(combined_functional, c)

if test_gradient:
    dJdc = compute_gradient(combined_functional, c, forget=False)
    File('dJdc.pvd').write(dJdc)
    J0 = rf(turbine_friction)
    minconv = taylor_test(rf, c, J0, dJdc, seed=1e-4)
    print "Order of convergence with taylor test (should be 2) =", minconv

    assert minconv > 1.95

if optimise:
    # compute maximum turbine density
    max_density = 1./(16.*2.5*16.*5)
    max_tf = C_T * A_T/2. * max_density
    print "Maximum turbine density =", max_tf

    tf_opt = maximise(rf, bounds=[0, max_tf],
            options={'maxiter': 100})
