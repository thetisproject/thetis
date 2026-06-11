"""
bathymetry_utils.py
=======================
Replicates the bathymetry for the tidal inlet (10.1016/j.cageo.2008.02.012)
and interpolates onto the Firedrake mesh

Domain geometry: 15x14 km

Depth profile:
   y = [0,      y_shelf]   ->  h = h_shelf  (flat)
   y = [y_shelf, y_ocean]  ->  h = linear   (slope)
   y > y_ocean             ->  h = h_ocean  (max depth)

Usage:
    pass Mesh and a FunctionSpace to generate_bathymetry()
"""

import numpy as np


def _depth_profile(y_coords,
                   h_shelf=4.0,
                   h_ocean=15.0,
                   y_shelf=6800.0,
                   y_ocean=14000.0):
    """
    Return water depth [m] at each point in y_coords using the piecewise
    linear profile of the tidal inlet domain

    Parameters:
        y_coords : cross-shore coordinate values [m] (array of node coords)
        h_shelf  : constant depth over the inner shelf / lagoon [m]
        h_ocean  : depth at the open-ocean boundary [m]
        y_shelf  : y-coordinate where the slope begins [m]
        y_ocean  : y-coordinate of the open boundary [m]
    """
    y = np.asarray(y_coords, dtype=float)
    depth = np.where(
        y <= y_shelf,
        h_shelf,  # flat shelf
        h_shelf + (h_ocean - h_shelf) * (y - y_shelf) / (y_ocean - y_shelf)  # linear ramp
    )
    # clamp to [h_shelf, h_ocean]
    depth = np.clip(depth, h_shelf, h_ocean)
    return depth


def generate_bathymetry(mesh2d, P1_2d,
                        h_shelf=4.0,
                        h_ocean=15.0,
                        y_shelf=6800.0,
                        y_ocean=14000.0,
                        name='Bathymetry'):
    """
    Generate the tidal inlet bathymetry
    Default values: [h_shelf=4, h_ocean=15, y_shelf=6800.0, y_ocean=14000]

    The depth depends only on the cross-shore coordinate y (uniform
    in x):
      - flat shelf of h_shelf metres from y = 0 to y = y_shelf
      - linear ramp from h_shelf to h_ocean over y = y_shelf to y_ocean

    Parameters:
        mesh2d   : firedrake.Mesh - The 2-D Firedrake mesh.
        P1_2d    : firedrake.FunctionSpace - CG1 function space on mesh2d
    ----------
        optionals  : h_shelf, h_ocean, y_shelf, y_ocean - parameters defining the depth profile
            (see _depth_profile)

    Returns
    -------
    bathymetry_2d : firedrake.Function aligned regardless of the internal
                    nodal reordering/optimization
    """
    from firedrake import Function

    # mesh2d.coordinates.dat.data has shape (n_nodes, 2): columns are (x, y)
    mesh_coords = mesh2d.coordinates.dat.data
    y_nodes = mesh_coords[:, 1]  # cross-shore coordinate

    bathy_values = _depth_profile(y_nodes,
                                  h_shelf=h_shelf,
                                  h_ocean=h_ocean,
                                  y_shelf=y_shelf,
                                  y_ocean=y_ocean)

    bathymetry_2d = Function(P1_2d, name=name)
    bathymetry_2d.dat.data[:] = bathy_values
    return bathymetry_2d
