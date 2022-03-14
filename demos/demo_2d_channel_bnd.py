# 2D channel with time-dependent boundary conditions
# ==================================================
#
# .. highlight:: python
#
# Here we extend the :doc:`2D channel example <demo_2d_channel.py>` by adding constant and time
# dependent boundary conditions.
#
# We begin by defining the domain and solver as before::

from thetis import *

lx = 40e3
ly = 2e3
nx = 25
ny = 2
mesh2d = RectangleMesh(nx, ny, lx, ly)

P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
depth = 20.0
bathymetry_2d.assign(depth)

# total duration in seconds
t_end = 12 * 3600
# export interval in seconds
t_export = 300.0

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.swe_timestepper_type = 'CrankNicolson'
options.timestep = 50.0

# We will force the model with a constant volume flux at the right boundary
# (x=40 km) and impose a tidal volume flux on the left boundary (x=0 km).
# Note that we have increased ``t_end`` and ``t_export`` to better illustrate
# tidal dynamics.
#
# Boundary condtitions are defined for each external boundary using their ID.
# In this example we are using a
# :py:func:`~.firedrake.utility_meshes.RectangleMesh` which assigns IDs 1, 2, 3,
# and 4 for the four sides of the rectangle::

left_bnd_id = 1
right_bnd_id = 2

# At each boundary we need to define the external value of the prognostic
# variables, i.e. in this case the water elevation and velocity.
# The value should be either a Firedrake :py:class:`~.firedrake.constant.Constant` or
# :py:class:`~.firedrake.function.Function` (in case the boundary condition is not uniform in space).
#
# We store the boundary conditions in a dictionary::

swe_bnd = {}
in_flux = 1e3
swe_bnd[right_bnd_id] = {'elev': Constant(0.0),
                         'flux': Constant(-in_flux)}

# Above we set the water elevation to zero and prescribe a constant volume flux.
# The volume flux is defined as outward normal flux, i.e. a negative value stands
# for flow into the domain.
# Alternatively we could also prescribe the normal velocity (with key ``'un'``)
# or the 2D velocity vector (``'uv'``).
# For all supported boundary conditions, see module :py:mod:`~.shallowwater_eq`.
#
# In order to set time-dependent boundary conditions we first define a python
# function that evaluates the time dependent variable::


def timedep_flux(simulation_time):
    """Time-dependent flux function"""
    tide_amp = -2e3
    tide_t = 12 * 3600.
    flux = tide_amp*sin(2 * pi * simulation_time / tide_t) + in_flux
    return flux


# We then create a Constant object with the initial value,
# and assign it to the left boundary::

tide_flux_const = Constant(timedep_flux(0))
swe_bnd[left_bnd_id] = {'flux': tide_flux_const}

# Boundary conditions are now complete, and we assign them to the solver
# object::

solver_obj.bnd_functions['shallow_water'] = swe_bnd

# Note that if boundary conditions are not assigned for some boundaries
# (the lateral boundaries 3 and 4 in this case), Thetis assumes impermeable land
# conditions.
#
# The only missing piece is to add a mechanism that re-evaluates the boundary
# condition as the simulation progresses.
# For this purpose we use the optional ``update_forcings`` argument of the
# :py:meth:`~.FlowSolver2d.iterate` method.
# ``update_forcings`` is a python function that updates all time dependent
# :py:class:`~.firedrake.constant.Constant`\s or
# :py:class:`~.firedrake.function.Function`\s used to force the model.
# In this case we only need to update ``tide_flux_const``::


def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields"""
    tide_flux_const.assign(timedep_flux(t_new))


# and finally pass this callback to the time iterator::

solver_obj.iterate(update_forcings=update_forcings)

#
# This tutorial can be dowloaded as a Python script `here <demo_2d_channel_bnd.py>`__.
