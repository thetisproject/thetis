"""
In this version of the Gray-Scott demo, we solve the
two equations as a monolithic system.

The main difference is the use of
`ModelOptions2d.add_tracers_2d`, rather than
`ModelOptions2d.add_tracer_2d`.
"""
from thetis import *


# Doubly periodic domain
mesh2d = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction="both")
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
options.output_directory = "outputs_mixed"
options.tracer_only = True
options.tracer_element_family = "cg"
options.use_supg_tracer = False
options.use_limiter_for_tracers = False
sim_end_time = 2000.0
options.set_timestepper_type(
    "CrankNicolson",
    implicitness_theta=1.0,
    solver_parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_shift_type": "inblocks",
    },
)

# Add tracer fields and the associated coefficients
W = P1_2d * P1_2d
ab_2d = Function(W)
a_2d, b_2d = split(ab_2d)
options.add_tracer_system_2d(
    ["a_2d", "b_2d"],
    ["Tracer A", "Tracer B"],
    ["TracerA2d", "TracerB2d"],
    function=ab_2d,
    a_2d={
        "diffusivity": D1,
        "source": gamma - a_2d * b_2d**2 - gamma * a_2d,
    },
    b_2d={
        "diffusivity": D2,
        "source": a_2d * b_2d**2 - (gamma + kappa) * b_2d,
    },
)
options.fields_to_export = ["a_2d", "b_2d"]

# Define initial conditions
tracer_a_init = Function(P1_2d)
tracer_b_init = Function(P1_2d)
tracer_b_init.interpolate(
    conditional(
        And(And(1.0 <= x, x <= 1.5), And(1.0 <= y, y <= 1.5)),
        0.25 * sin(4 * pi * x) ** 2 * sin(4 * pi * y) ** 2,
        0,
    )
)
tracer_a_init.interpolate(1.0 - 2.0 * tracer_b_init)
solver_obj.assign_initial_conditions(a=tracer_a_init, b=tracer_b_init)

solver_obj.create_timestepper()

# Turn off exports and reduce time duration if regression testing
if os.getenv("THETIS_REGRESSION_TEST") is not None:
    options.no_exports = True
    sim_end_time = 500.0

# Spin up timestep
dt = 1.0e-04
end_time = 0.0
for i in range(4):
    dt *= 10
    end_time += 10 * dt if i == 0 else 9 * dt
    options.timestep = dt
    solver_obj.export_initial_state = i == 0
    options.simulation_export_time = 10 * dt
    options.simulation_end_time = end_time
    solver_obj.create_timestepper()
    solver_obj.iterate()

# Run for duration
options.simulation_end_time = sim_end_time
solver_obj.create_timestepper()
solver_obj.iterate()
