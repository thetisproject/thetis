"""
Option 2: rt-dg RT1/DG0 + DIRK22
==================================
Velocity: RT1 (Raviart-Thomas degree 1, H(div)-conforming, 3 DOFs/tri)
Elevation: DG0 (piecewise constant, 1 DOF/tri)
Total: 4 DOFs/tri vs 9 for the dg-dg P1DG baseline — more than 2x reduction.

RT1 guarantees exact pointwise mass conservation (div u = 0 on each element),
and normal-velocity continuity across element faces is built into the space.
Elevation is piecewise constant so horizontal wave propagation is first-order.

The default fieldsplit-multiplicative solver works for RT1-DG0.  See option 4
(perf_4_rt_hybridization.py) for a more efficient linear solver for this element
pair using Firedrake's HybridizationPC.

Run from the demos/ directory::

    python perf_2_rt_dg.py
"""
import perf_common as common
from thetis import *

mesh2d, bathymetry_2d = common.load_mesh_and_bathy()
manning_2d, coriolis_2d = common.make_physics_fields(mesh2d)
elev_tide_2d, update_forcings = common.setup_tidal_forcing(mesh2d)

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'rt-dg'
options.polynomial_degree = 0              # RT1 velocity, DG0 elevation
options.coriolis_frequency = coriolis_2d
options.manning_drag_coefficient = manning_2d
options.horizontal_velocity_scale = Constant(1.5)
# Lax-Friedrichs not needed for H(div)-conforming velocity (normal flux is exact)
options.use_lax_friedrichs_velocity = False
options.simulation_export_time = common.T_EXPORT
options.simulation_initial_date = common.START_DATE
options.simulation_end_date = common.END_DATE
options.swe_timestepper_type = 'DIRK22'
options.swe_timestepper_options.use_semi_implicit_linearization = True
options.timestep = common.DT
options.fields_to_export = ['elev_2d', 'uv_2d']
options.fields_to_export_hdf5 = []
options.output_directory = 'outputs_perf_2_rt_dg'

solver_obj.create_equations()

solver_obj.bnd_functions['shallow_water'] = {
    100: {'elev': elev_tide_2d, 'uv': Constant(as_vector([0, 0]))},
}

common.load_spinup(solver_obj, mesh2d)
common.add_station_callbacks(solver_obj)

wall_time = common.run_and_time(solver_obj, update_forcings)
common.print_performance(solver_obj, wall_time,
                         'Option 2: rt-dg RT1/DG0 + DIRK22')
