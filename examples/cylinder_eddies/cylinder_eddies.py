"""
2D flow around a cylinder
=========================

"""
from thetis import *


def snap_cylinder_coords(m, degree=2, circle_radius=500., circle_id=5):
    """
    Snap cylinder boundary nodes to circle arc.
    """
    # make new high degree coordinate function
    V = VectorFunctionSpace(m, 'CG', degree)
    new_coords = Function(V)
    xy = SpatialCoordinate(m)
    new_coords.interpolate(xy)
    # calculate new coordinates on circle arc
    xy_mag = sqrt(xy[0]**2 + xy[1]**2)
    circle_coords = as_vector((xy[0], xy[1]))/xy_mag*circle_radius
    bc = DirichletBC(V, circle_coords, circle_id)
    bc.apply(new_coords)
    # make a new mesh
    new_mesh = mesh.Mesh(new_coords)
    return new_mesh


mesh2d = Mesh('mesh_cylinder_coarse.msh')
mesh2d = snap_cylinder_coords(mesh2d)

t_end = 8 * 3600.
t_export = 2*60.
dt = 60.

if os.getenv('THETIS_REGRESSION_TEST') is not None:
    t_end = 5*t_export

flow_speed = 1.5

# bathymetry
depth = 20.0
P1_2d = get_functionspace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(depth)

# quadratic bottom friction
# increase friction along the cylinder wall, causes flow separation in the wake
drag_coeff_2d = Function(P1_2d, name='Cd')
cd_max = 1e-2
bc = DirichletBC(P1_2d, cd_max, 5)
bc.apply(drag_coeff_2d)

# create solver
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.quadratic_drag_coefficient = drag_coeff_2d
options.horizontal_viscosity = Constant(0.5)
options.horizontal_velocity_scale = Constant(flow_speed)
options.fields_to_export = ['uv_2d']
options.fields_to_export_hdf5 = []
options.swe_timestepper_type = 'DIRK22'
options.timestep = dt

# boundary conditions
flow_speed_ramped = Constant(0.0)
t_sim = Constant(0.0)
t_ramp = 1800.
u_ramp = flow_speed*conditional(le(t_sim, t_ramp), t_sim/t_ramp, Constant(1.0))
bnd_len = 7000.0
flux = Constant(depth * bnd_len) * flow_speed_ramped

inflow_tag = 1
outflow_tag = 2
inflow_bc = {'flux': -flux, 'elev': Constant(0.0)}
outflow_bc = {'flux': flux, 'elev': Constant(0.0)}
solver_obj.bnd_functions['shallow_water'] = {
    inflow_tag: inflow_bc,
    outflow_tag: outflow_bc
}

uv_init = Constant((1e-4, 0))
solver_obj.assign_initial_conditions(uv=uv_init)


def update_forcings(t):
    t_sim.assign(t)
    flow_speed_ramped.assign(u_ramp)


solver_obj.iterate(update_forcings=update_forcings)
