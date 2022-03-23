# 2D setup with muliple tracers
# =============================
#
# The problem setup of this demo is almost identical to
# `the first tracer demo <./demo_2d_tracer.py.html>`__.
# The main difference is that it shows how to treat the
# three advected quantities as separate tracer fields.

from thetis import *

mesh2d = UnitSquareMesh(40, 40, quadrilateral=True)
P1_2d = FunctionSpace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d)
bathymetry2d.assign(1.0)

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)

# Again, tracer functionality is activated by adding tracer
# fields. In the previous demo, a single tracer field was
# used, with the default name `tracer_2d`. It is also
# possible to specify separate tracers, provided we provide
# labels, names and filenames for each, as well as any source
# terms, diffusion coefficients and boundary conditions. ::

labels = ['bell_2d', 'cone_2d', 'slot_cyl_2d']
names = ['Gaussian bell', 'Cone', 'Slotted cylinder']
filenames = ['GaussianBell2d', 'Cone2d', 'SlottedCylinder2d']
options = solver_obj.options
options.tracer_only = True
options.fields_to_export = labels
bc = {'value': {'on_boundary': Constant(1.0)}}
for label, name, filename in zip(labels, names, filenames):
    options.add_tracer_2d(label, name, filename, source=None, diffusivity=None)
    solver_obj.bnd_functions[label] = bc

# Note that tracer equations will be solved in the order
# in which they were added using `add_tracer_2d`. Most of
# the remaining model setup is as before.

options.tracer_timestepper_type = 'SSPRK33'
options.timestep = pi/300.0
options.simulation_end_time = 2*pi
options.simulation_export_time = pi/15.0
options.tracer_timestepper_options.use_automatic_timestep = False
options.use_lax_friedrichs_tracer = False
options.use_limiter_for_tracers = False

vP1_2d = VectorFunctionSpace(mesh2d, "CG", 1)
x, y = SpatialCoordinate(mesh2d)
uv_init = interpolate(as_vector([0.5 - y, x - 0.5]), vP1_2d)

# Initial conditions for each tracer are defined as before,
# but must be assigned separately. ::

bell_r0, bell_x0, bell_y0 = 0.15, 0.25, 0.5
cone_r0, cone_x0, cone_y0 = 0.15, 0.5, 0.25
cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.5, 0.75
slot_left, slot_right, slot_top = 0.475, 0.525, 0.85

bell = 0.25*(1+cos(pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cone_r0, 1.0)
slot_cyl = conditional(
    sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
    conditional(And(And(x > slot_left, x < slot_right), y < slot_top), 0.0, 1.0),
    0.0
)

bell_init = interpolate(1.0 + bell, P1_2d)
cone_init = interpolate(1.0 + cone, P1_2d)
slot_cyl_init = interpolate(1.0 + slot_cyl, P1_2d)
solver_obj.assign_initial_conditions(
    uv=uv_init, bell_2d=bell_init, cone_2d=cone_init, slot_cyl_2d=slot_cyl_init
)

# Finally, we solve the tracer transport problem and display
# the normalised :math:`L^2` error. ::

solver_obj.iterate()
for label, name, init in zip(labels, names, [bell_init, cone_init, slot_cyl_init]):
    q = solver_obj.fields[label]
    L2_err = sqrt(assemble((q - init)*(q - init)*dx))
    L2_init = sqrt(assemble(init*init*dx))
    print_output("Relative error {:8s}: {:.2f}%".format(name, L2_err/L2_init))

# This tutorial can be dowloaded as a Python script `here <demo_2d_multiple_tracers.py>`__.
