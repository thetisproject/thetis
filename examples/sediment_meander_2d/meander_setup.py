"""
Meander Test case
=======================
Solves the initial hydrodynamics simulation of flow around a 180 degree bend replicating
lab experiment 4 in Yen & Lee (1995).
Note this is not the main run-file and is just used to create an initial checkpoint for
the morphodynamic simulation.

For more details of the test case set-up see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *

# define mesh
mesh2d = Mesh("meander.msh")


def snap_mesh_bnd_to_circle_arc(m, circle_arc_list, degree=2):
    """
    Snap mesh boundary nodes to a circle arc.
    """
    # make new high degree coordinate function
    V = VectorFunctionSpace(m, 'CG', degree)
    new_coords = Function(V)
    x, y = SpatialCoordinate(m)
    new_coords.interpolate(as_vector((x, y)))
    for bnd_id, x0, y0, radius in circle_arc_list:
        # calculate new coordinates on circle arc
        xy_mag = sqrt((x - x0)**2 + (y - y0)**2)
        new_x = (x - x0)/xy_mag*radius + x0
        new_y = (y - y0)/xy_mag*radius + y0
        circle_coords = as_vector((new_x, new_y))
        bc = DirichletBC(V, circle_coords, bnd_id)
        bc.apply(new_coords)
    # make a new mesh
    new_mesh = mesh.Mesh(new_coords)
    return new_mesh


# define circle boundary arcs: bnd_id, x0, y0, radius
circle_arcs = [
    (4, 4.5, 2.5, 4.5),
    (5, 4.5, 2.5, 3.5),
]
mesh2d = snap_mesh_bnd_to_circle_arc(mesh2d, circle_arcs)

x, y = SpatialCoordinate(mesh2d)
# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
gradient = Constant(0.0035)
L_function = Function(V).interpolate(conditional(x > 5, pi*4*((pi/2)-acos((x-5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi,
                                                 pi*4*((pi/2)-acos((-x+5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi))
bathymetry_curve = Function(V).interpolate(conditional(y > 2.5,
                                           conditional(x < 5, (L_function*gradient),
                                                       -(L_function*gradient)), 0))
init = max(bathymetry_curve.dat.data[:])
final = min(bathymetry_curve.dat.data[:])
bathymetry_straight = Function(V).interpolate(conditional(x <= 5,
                                              conditional(y <= 2.5, gradient*abs(y - 2.5) + init, 0),
                                              conditional(y <= 2.5, - gradient*abs(y - 2.5) + final, 0)))
bathymetry_2d = Function(V).interpolate(-bathymetry_curve - bathymetry_straight)