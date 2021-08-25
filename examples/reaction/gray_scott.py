"""
Gray-Scott diffusion-reaction demo, taken from
[Hundsdorf & Verwer 2003]. It also appears as a
PETSc TS tutorial and can be found in your Firedrake
installation at

  $VIRTUAL_ENV/src/petsc/src/ts/tutorials/advection-diffusion-reaction/ex5.c

The test case consists of two tracer species, which
experience different diffusivities and react with one
another. The problem is interesting from a numerical
modelling perspective because of the nonlinear coupling
between the two equations in the system. Instead of
solving the equations as a monolithic system, we opt to
solve them alternately using a Picard iteration.

[Hundsdorf & Vermer 2003] W. Hundsdorf & J.G. Verwer
    (2003). "Numerical Solution of Time-Dependent
    Advection-Diffusion-Reaction Equations", Springer.
"""
from thetis import *


# Doubly periodic domain
mesh2d = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction='both')
x, y = SpatialCoordinate(mesh2d)

# Arbitrary bathymetry
P1_2d = get_functionspace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d).assign(1.0)

# Diffusion and reactivity constants
D1 = Constant(8.0e-05)
D2 = Constant(4.0e-05)
gamma = Constant(0.024)
kappa = Constant(0.06)

# Setup solver object
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solver_obj.options
options.tracer_only = True
options.tracer_element_family = 'cg'
options.use_supg_tracer = False
options.use_limiter_for_tracers = False
sim_end_time = 2000.0

# Specify iterative method
options.tracer_picard_iterations = 2
options.set_timestepper_type('CrankNicolson', implicitness_theta=1.0)

# Add tracer fields and the associated coefficients
a_2d = Function(P1_2d)
b_2d = Function(P1_2d)
options.add_tracer_2d(
    "a_2d", "Tracer A", "TracerA2d",
    function=a_2d,
    diffusivity=D1,
    source=gamma - a_2d*b_2d**2 - gamma*a_2d,
)
options.add_tracer_2d(
    "b_2d", "Tracer B", "TracerB2d",
    function=b_2d,
    diffusivity=D2,
    source=a_2d*b_2d**2 - (gamma + kappa)*b_2d,
)
options.fields_to_export = ["a_2d", "b_2d"]

# Define initial conditions
tracer_a_init = Function(P1_2d)
tracer_b_init = Function(P1_2d)
tracer_b_init.interpolate(
    conditional(
        And(And(1.0 <= x, x <= 1.5), And(1.0 <= y, y <= 1.5)),
        0.25*sin(4*pi*x)**2*sin(4*pi*y)**2, 0
    )
)
tracer_a_init.interpolate(
    1.0 - 2.0*tracer_b_init
)
solver_obj.assign_initial_conditions(a=tracer_a_init, b=tracer_b_init)

solver_obj.create_timestepper()

# Turn off exports and reduce time duration if regression testing
if os.getenv('THETIS_REGRESSION_TEST') is not None:
    options.no_exports = True
    sim_end_time = 500.0

# Spin up timestep
dt = 1.0e-04
end_time = 0.0
for i in range(4):
    dt *= 10
    end_time += 10*dt if i == 0 else 9*dt
    options.timestep = dt
    solver_obj.export_initial_state = i == 0
    options.simulation_export_time = 10*dt
    options.simulation_end_time = end_time
    solver_obj.create_timestepper()
    solver_obj.iterate()

# Run for duration
options.simulation_end_time = sim_end_time
solver_obj.create_timestepper()
solver_obj.iterate()
