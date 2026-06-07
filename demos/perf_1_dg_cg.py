"""
Option 1: dg-cg P1DG/P2CG + PressureProjectionPicard
======================================================
Velocity: P1DG (discontinuous, 3 DOFs/tri × 2 components = 6 DOFs/tri)
Elevation: P2CG (continuous, ~1.5 DOFs/tri — shared across elements)

PressureProjectionPicard splits the solve into:
  (a) A block-diagonal momentum solve for u* (Lax-Friedrichs + drag + Coriolis)
  (b) A wave-equation solve for (u^{n+1}, eta^{n+1}) using the assembled Schur
      complement preconditioner (AssembledSchurPC + GAMG).

The CG elevation space reduces DOFs by ~2× vs dg-dg, and the pressure-projection
splitting makes each timestep significantly cheaper than a monolithic solve.

With picard_iterations=1 and use_semi_implicit_linearization=True, each timestep
costs exactly two linear solves.

Run from the demos/ directory::

    python perf_1_dg_cg.py
"""
import perf_common as common
from thetis import *

mesh2d, bathymetry_2d = common.load_mesh_and_bathy()
manning_2d, coriolis_2d = common.make_physics_fields(mesh2d)
elev_tide_2d, update_forcings = common.setup_tidal_forcing(mesh2d)

solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-cg'
options.polynomial_degree = 1              # P1DG velocity, P2CG elevation
options.coriolis_frequency = coriolis_2d
options.manning_drag_coefficient = manning_2d
options.horizontal_velocity_scale = Constant(1.5)
options.use_lax_friedrichs_velocity = True
options.simulation_export_time = common.T_EXPORT
options.simulation_initial_date = common.START_DATE
options.simulation_end_date = common.END_DATE
options.swe_timestepper_type = 'PressureProjectionPicard'
options.swe_timestepper_options.use_semi_implicit_linearization = True
options.swe_timestepper_options.picard_iterations = 1
# Solver for the momentum step: block-diagonal (DG velocity mass is block-local)
options.swe_timestepper_options.solver_parameters_momentum = {
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
    'sub_ksp_type': 'preonly',
    'sub_pc_type': 'sor',
}
# Solver for the wave step: assembled Schur complement + GAMG on elevation block.
# AssembledSchurPC builds A10*M_u^{-1}*A01 explicitly and uses gamg on it.
# This is efficient because the DG velocity mass matrix M_u is block-diagonal
# and therefore trivially invertible element-by-element.
options.swe_timestepper_options.solver_parameters_pressure = {
    'ksp_type': 'preonly',
    'mat_type': 'matfree',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'fieldsplit_U_2d': {
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_ksp_type': 'preonly',
        'assembled_pc_type': 'bjacobi',
        'assembled_sub_pc_type': 'ilu',
    },
    'fieldsplit_H_2d': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'thetis.AssembledSchurPC',
        'schur_ksp_type': 'gmres',
        'schur_ksp_max_it': 100,
        'schur_pc_type': 'gamg',
    },
}
options.timestep = common.DT
options.fields_to_export = ['elev_2d', 'uv_2d']
options.fields_to_export_hdf5 = []
options.output_directory = 'outputs_perf_1_dg_cg'

solver_obj.create_equations()

solver_obj.bnd_functions['shallow_water'] = {
    100: {'elev': elev_tide_2d, 'uv': Constant(as_vector([0, 0]))},
}

common.load_spinup(solver_obj, mesh2d)
common.add_station_callbacks(solver_obj)

wall_time = common.run_and_time(solver_obj, update_forcings)
common.print_performance(solver_obj, wall_time,
                         'Option 1: dg-cg P1DG/P2CG + PressureProjectionPicard')
