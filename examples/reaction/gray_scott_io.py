"""
This test case uses the same model as gray_scott.py.
The only difference here is that the model is loaded
from examples/reaction/reaction_models/gray-scott.yml
rather than being specified within the Python script.
"""
from thetis import *
import matplotlib.pyplot as plt
import networkx as nx


# Doubly periodic domain
mesh2d = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction='both')
x, y = SpatialCoordinate(mesh2d)

# Arbitrary bathymetry
P1_2d = get_functionspace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d).assign(1.0)

# Setup solver object
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solver_obj.options
options.output_directory = 'outputs_io'
options.tracer_only = True
options.tracer_element_family = 'cg'
options.use_supg_tracer = False
options.use_limiter_for_tracers = False
sim_end_time = 2000.0

# Specify iterative method
options.tracer_picard_iterations = 2
options.set_timestepper_type('CrankNicolson', implicitness_theta=1.0)

# Load tracers
solver_obj.create_function_spaces()
solver_obj.load_tracers_2d(
    "gray-scott", input_directory=os.path.join(
        os.path.dirname(__file__), "reaction_models"),
    use_conservative_form=False, append_dimension=True)
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

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    # Turn off exports and reduce time duration if regression testing
    options.no_exports = True
    sim_end_time = 500.0
else:
    # Plot the dependency graph
    G = solver_obj.adr_model.dependency_graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=200, with_labels=True)
    plt.savefig("dependency_graph.png", dpi=200, bbox_inches="tight")

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
