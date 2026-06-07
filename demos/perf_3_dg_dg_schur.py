"""
Option 3: dg-dg P1DG + DIRK22 + Schur/AssembledSchurPC solver
===============================================================
Same discretisation as the baseline (dg-dg P1DG, DIRK22, semi-implicit), but
with a better linear solver.

The default solver (gmres + fieldsplit multiplicative, no explicit sub-block
preconditioners) does not exploit the saddle-point structure of the SWE
system.  This script replaces it with:

  • Velocity block: DG mass matrix is block-diagonal per element, so
    bjacobi+ILU is essentially exact (each element has an independent 6×6
    system for P1DG 2-component velocity).

  • Elevation/Schur block: AssembledSchurPC explicitly assembles
    S = A10 * M_u^{-1} * A01 + A11 and applies GAMG to it.  For the wave
    equation the Schur complement is a discrete Laplacian-like operator, which
    multigrid handles in O(N) work.

This requires the appctx change added to DIRKGeneric.update_solver so that
AssembledSchurPC can access the form to split into sub-blocks.

Run from the demos/ directory::

    python perf_3_dg_dg_schur.py
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
# Replace the default gmres+fieldsplit-multiplicative with an assembled Schur
# complement preconditioner.  mat_type='matfree' lets AssembledPC/AssembledSchurPC
# assemble the sub-blocks they need without assembling the full coupled matrix.
options.swe_timestepper_options.solver_parameters = {
    'ksp_type': 'preonly',
    'mat_type': 'matfree',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    # Velocity block: DG mass is block-local, so bjacobi+ILU is exact per element.
    'fieldsplit_U_2d': {
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_ksp_type': 'preonly',
        'assembled_pc_type': 'bjacobi',
        'assembled_sub_pc_type': 'ilu',
    },
    # Elevation/Schur block: AssembledSchurPC builds A10*Minv*A01+A11 explicitly.
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
options.output_directory = 'outputs_perf_3_dg_dg_schur'

solver_obj.create_equations()

solver_obj.bnd_functions['shallow_water'] = {
    100: {'elev': elev_tide_2d, 'uv': Constant(as_vector([0, 0]))},
}

common.load_spinup(solver_obj, mesh2d)
common.add_station_callbacks(solver_obj)

wall_time = common.run_and_time(solver_obj, update_forcings)
common.print_performance(solver_obj, wall_time,
                         'Option 3: dg-dg P1DG + DIRK22 + Schur/AssembledSchurPC')
