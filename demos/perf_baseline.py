"""
Baseline performance configuration
===================================
dg-dg P1DG + DIRK22 + semi-implicit linearisation.

This matches the setup in demo_2d_north_sea.py and is the reference against
which the other perf_*.py configurations are measured.

Run from the demos/ directory::

    python perf_baseline.py
"""
import perf_common as common
from thetis import *

mesh2d, bathymetry_2d = common.load_mesh_and_bathy()
manning_2d, coriolis_2d = common.make_physics_fields(mesh2d)
elev_tide_2d, update_forcings = common.setup_tidal_forcing(mesh2d)

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
options.coriolis_frequency = coriolis_2d
options.manning_drag_coefficient = manning_2d
options.horizontal_velocity_scale = Constant(1.5)
options.use_lax_friedrichs_velocity = True
options.simulation_export_time = common.T_EXPORT
options.simulation_initial_date = common.START_DATE
options.simulation_end_date = common.END_DATE
options.swe_timestepper_type = 'DIRK22'
options.swe_timestepper_options.use_semi_implicit_linearization = True
options.timestep = common.DT
options.fields_to_export = ['elev_2d', 'uv_2d']
options.fields_to_export_hdf5 = []
options.output_directory = 'outputs_perf_baseline'

solver_obj.create_equations()

solver_obj.bnd_functions['shallow_water'] = {
    100: {'elev': elev_tide_2d, 'uv': Constant(as_vector([0, 0]))},
}

common.load_spinup(solver_obj, mesh2d)
common.add_station_callbacks(solver_obj)

wall_time = common.run_and_time(solver_obj, update_forcings)
common.print_performance(solver_obj, wall_time, 'Baseline: dg-dg P1DG + DIRK22')
