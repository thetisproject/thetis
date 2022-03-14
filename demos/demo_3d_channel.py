# 3D tidal channel demo
# =====================
#
# .. highlight:: python
#
# This example demonstrates a 3D barotropic model in a tidal channel with sloping
# bathymetry. We also add a constant, passive salinity tracer to demonstrate local
# tracer conservation. This simulation uses the ALE moving mesh.
#
# We begin by defining the 2D mesh as before::

from thetis import *

lx = 100e3
ly = 6e3
nx = 33
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)

# In this case we define a linearly sloping bathymetry in the x-direction.
# The bathymetry function is defined as an
# `UFL <http://fenics-ufl.readthedocs.io/en/latest/>`_ expression making use of the
# coordinates of the 2D mesh.
# The expression is interpolated on the P1 bathymetry field::

P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
depth_oce = 20.0
depth_riv = 7.0
xy = SpatialCoordinate(mesh2d)
bath_ufl_expr = depth_oce - (depth_oce-depth_riv)*xy[0]/lx
bathymetry_2d.interpolate(bath_ufl_expr)

# Next we create the 3D solver. The 2D mesh will be extruded in the vertical
# direction using a constant number of layers::

n_layers = 6
solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)

# We then set some options (see :py:class:`~.ModelOptions` for more information)::

options = solver_obj.options
options.element_family = 'dg-dg'
options.timestepper_type = 'SSPRK22'
options.use_implicit_vertical_diffusion = False
options.use_bottom_friction = False
options.use_ale_moving_mesh = True
options.use_limiter_for_tracers = True
options.simulation_export_time = 900.0
options.simulation_end_time = 24 * 3600
options.fields_to_export = ['uv_2d', 'elev_2d', 'elev_3d', 'uv_3d',
                            'w_3d', 'w_mesh_3d', 'salt_3d', 'baroc_head_3d',
                            'uv_dav_2d']

# We set this simulation to be barotropic (i.e. salinity and temperature do not
# affect water density), but we still wish to simulate salinity as a passive
# tracer::

options.use_baroclinic_formulation = False
options.solve_salinity = True
options.solve_temperature = False

# We also want to see how much the salinity value deviates from its initial
# value::

options.check_salinity_overshoot = True

# In this simulation we do not set the time step manually but instead use the
# automatic time step estimation of Thetis. Time step is estimated based on the
# CFL number, used spatial discretization and time integration
# method. We only need to define maximal horizontal and vertical velocity scales::

u_max = 0.5
w_max = 2e-4
options.horizontal_velocity_scale = Constant(u_max)
options.vertical_velocity_scale = Constant(w_max)


# Next we define the boundary conditions. Note that in a 3D model there are
# multiple coupled equations, and we need to set boundary conditions to all of
# them.
#
# In this example we impose time dependent normal flux on both the deep (ocean)
# and shallow (river) boundaries. We begin by creating python functions that
# define the time dependent fluxes. Note that we use a linear ramp-up function on
# both boundaries::

ocean_bnd_id = 1
river_bnd_id = 2

un_amp = -0.5  # tidal normal velocity amplitude (m/s)
flux_amp = ly*depth_oce*un_amp
t_tide = 12 * 3600.  # tidal period (s)
un_river = -0.05  # constant river flow velocity (m/s)
flux_river = ly*depth_riv*un_river
t_ramp = 6*3600.0  # use linear ramp up for boundary forcings


def ocean_flux_func(t):
    return (flux_amp*sin(2 * pi * t / t_tide) - flux_river)*min(t/t_ramp, 1.0)


def river_flux_func(t):
    return flux_river*min(t/t_ramp, 1.0)


# We then define :py:class:`~.firedrake.constant.Constant` objects for the fluxes and
# use them as boundary conditions for the 2D shallow water model::

ocean_flux = Constant(ocean_flux_func(0))
river_flux = Constant(river_flux_func(0))

ocean_funcs = {'flux': ocean_flux}
river_funcs = {'flux': river_flux}

solver_obj.bnd_functions['shallow_water'] = {ocean_bnd_id: ocean_funcs,
                                             river_bnd_id: river_funcs}

# The volume fluxes are now defined in the 2D mode, so there's no need to impose
# anything in the 3D momentum equation. We therefore only use symmetry condition
# for 3D horizontal velocity::

ocean_funcs_3d = {'symm': None}
river_funcs_3d = {'symm': None}

solver_obj.bnd_functions['momentum'] = {ocean_bnd_id: ocean_funcs_3d,
                                        river_bnd_id: river_funcs_3d}

# For the salinity, we define a constant value and apply as inflow conditions
# at the open boundaries::

salt_init3d = Constant(4.5)
ocean_salt_3d = {'value': salt_init3d}
river_salt_3d = {'value': salt_init3d}

solver_obj.bnd_functions['salt'] = {ocean_bnd_id: ocean_salt_3d,
                                    river_bnd_id: river_salt_3d}

# As before, all boundaries where boundary conditions are not assigned are
# assumed to be impermeable land boundaries.
#
# We now need to define the callback functions that update all time dependent
# forcing fields. As the 2D and 3D modes may be treated separately in the time
# integrator we create a different call back for the two modes::


def update_forcings_2d(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 2D mode"""
    ocean_flux.assign(ocean_flux_func(t_new))
    river_flux.assign(river_flux_func(t_new))


def update_forcings_3d(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 3D mode"""
    pass


# Because the boundary conditions of the 3D equations do not depend on time, the
# 3d callback function does nothing (it could be omitted).
#
# We then assign the constant salinity value as an initial condition::

solver_obj.assign_initial_conditions(salt=salt_init3d)

# and run the simulation::

solver_obj.iterate(update_forcings=update_forcings_2d,
                   update_forcings3d=update_forcings_3d)

# As you run the simulation, Thetis prints out the normal simulation statistics
# and also prints out the over/undershoots in the salinity field:
#
# .. code-block:: none
#
#         0     0 T=      0.00 eta norm:     0.0000 u norm:     0.0000  0.00
#     salt_3d overshoot 0 0
#         1     5 T=    900.00 eta norm:    15.1764 u norm:     0.0000  1.23
#     salt_3d overshoot -1.00586e-11 2.58318e-11
#         2    10 T=   1800.00 eta norm:    83.4282 u norm:     0.0000  0.39
#     salt_3d overshoot -3.13083e-11 3.42579e-11
#         3    15 T=   2700.00 eta norm:   229.6974 u norm:     0.0000  0.35
#     salt_3d overshoot -6.35199e-11 6.6346e-11
#
# Note that here the ``u norm`` is the norm of :math:`\mathbf{u}'`, i.e. the prognostic 3D
# horizontal velocity field (3D velocity minus its vertical average).
#
# This tutorial can be dowloaded as a Python script `here <demo_3d_channel.py>`__.
