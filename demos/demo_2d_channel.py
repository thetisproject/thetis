# 2D channel example
# ==================
#
# .. highlight:: python
#
# This example demonstrates a depth-averaged 2D simulation in a closed
# rectangular domain, where the flow is forced by an initial pertubation in the
# water elevation field.
#
# We begin by importing Thetis and creating a rectangular mesh with :py:func:`~.firedrake.utility_meshes.RectangleMesh`.
# The domain is 40 km long and 2 km wide.
# We generate 25 elements in the along-channel direction and 2 in the
# cross-channel direction::

from thetis import *

lx = 40e3
ly = 2e3
nx = 25
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)

# Next we define a bathymetry function in the 2D mesh, using continuous linear
# elements. In this example we set the bathymetry to constant 20 m depth::

P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
depth = 20.0
bathymetry_2d.assign(depth)

# .. note::
#
#     See
#     `Firedrake manual <http://firedrakeproject.org/variational-problems.html>`_
#     for more information on mesh generation, functions and function spaces.
#
# We are now ready to create a 2D solver object, and set some options::

# total duration in seconds
t_end = 2 * 3600
# export interval in seconds
t_export = 100.0

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end

# Here we simply define the total duration of the run, and the
# export interval. See :py:class:`~.ModelOptions` for more information about the
# available options.
#
# Next we define the used time integrator for the shallow water
# equations and set the time step::

options.swe_timestepper_type = 'CrankNicolson'
options.timestep = 50.0

# Because Crank-Nicolson is an uncondionally stable method, we can set
# the time step freely.
#
# We then define the initial condition for elevation. We begin by creating a
# function (in the same linear continous function space)::

elev_init = Function(P1_2d, name='initial elevation')

# We then need to define an analytical expression the the x,y coordinates of the
# mesh. To this end, we use
# :py:class:`~.ufl.classes.SpatialCoordinate` and define a `UFL <http://fenics-ufl.readthedocs.io/en/latest/>`_ expression (see
# `Firedrake's interpolation manual <http://firedrakeproject.org/interpolation.html>`_
# for more information)::

xy = SpatialCoordinate(mesh2d)
gauss_width = 4000.
gauss_ampl = 2.0
gauss_expr = gauss_ampl * exp(-((xy[0]-lx/2)/gauss_width)**2)

# This defines a 2 m tall Gaussian hill in the x-direction in the middle on the
# domain. We can then interpolate this expression on the function::

elev_init.interpolate(gauss_expr)

# and set this function as an initial condition to the elevation field::

solver_obj.assign_initial_conditions(elev=elev_init)

# Model setup is now complelete. We run the model by issuing::

solver_obj.iterate()

# While the model is running, Thetis prints some statistics on the command line:
#
# .. code-block:: none
#
#     0     0 T=      0.00 eta norm:  6251.2574 u norm:     0.0000  0.00
#     1     2 T=    100.00 eta norm:  5905.0262 u norm:  1398.1128  0.76
#     2     4 T=    200.00 eta norm:  5193.5227 u norm:  2377.8512  0.03
#     3     6 T=    300.00 eta norm:  4656.5334 u norm:  2856.5165  0.03
#     ...
#
# The first column is the export index, the second one the number of executed
# time steps, followed by the simulation time. ``eta norm`` and ``u norm`` are
# the L2 norms of the elevation and depth averaged velocity fields, respectively.
# The last column stands for the (approximate) wall-clock time between exports.
#
# The simulation terminates once the end time is reached.
# See :doc:`outputs and visualization <../outputs_and_visu>` page on how to
# visualize the results.
#
# This tutorial can be dowloaded as a Python script `here <demo_2d_channel.py>`__.
