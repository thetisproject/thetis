"""
Option 4: rt-dg RT1/DG0 + DIRK22 + HybridizationPC
=====================================================
Same RT1/DG0 discretisation as option 2, but with Firedrake's HybridizationPC
as the linear solver.

Hybridization exploits the H(div)-conforming structure of the RT1 space.
Concretely it:
  1. Breaks RT1 into a fully-discontinuous H(div) space (one copy per element).
  2. Introduces Lagrange multipliers λ on interior facets to enforce the
     normal-velocity continuity that RT1 had built in.
  3. Eliminates the interior velocity and elevation unknowns by local static
     condensation, leaving a global system only for λ.
  4. Recovers velocity and elevation on each element independently (cheap local
     solves, no global communication beyond the λ solve).

The global system for λ has ~1.5 DOFs/tri (one per interior facet) vs 4 for
the full RT1-DG0 system — roughly 2.5× reduction in the global problem size,
on top of the 2× reduction from option 2 vs the baseline.

The inner solver for λ uses GAMG, which is well-suited to the elliptic
character of the hybridized wave equation.

This requires mat_type='matfree' so that HybridizationPC can work from the
variational form rather than an assembled matrix.  The appctx change added to
DIRKGeneric.update_solver provides the Jacobian form to the preconditioner.

Run from the demos/ directory::

    python perf_4_rt_hybridization.py
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
options.use_lax_friedrichs_velocity = False
options.simulation_export_time = common.T_EXPORT
options.simulation_initial_date = common.START_DATE
options.simulation_end_date = common.END_DATE
options.swe_timestepper_type = 'DIRK22'
options.swe_timestepper_options.use_semi_implicit_linearization = True
# HybridizationPC: reduces the global problem to trace/facet DOFs, then
# recovers element-interior unknowns by local solves.
# mat_type='matfree' is required — HybridizationPC works from the UFL form,
# not from an assembled matrix.
options.swe_timestepper_options.solver_parameters = {
    'snes_type': 'ksponly',
    'mat_type': 'matfree',
    'ksp_type': 'preonly',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    # Parameters for the global λ (Lagrange multiplier / trace) solve:
    'hybridization': {
        'ksp_type': 'gmres',
        'ksp_rtol': 1e-8,
        'ksp_max_it': 200,
        'pc_type': 'gamg',
        'mat_type': 'aij',   # assemble the trace system explicitly for GAMG
    },
}
options.timestep = common.DT
options.fields_to_export = ['elev_2d', 'uv_2d']
options.fields_to_export_hdf5 = []
options.output_directory = 'outputs_perf_4_rt_hybrid'

solver_obj.create_equations()

solver_obj.bnd_functions['shallow_water'] = {
    100: {'elev': elev_tide_2d, 'uv': Constant(as_vector([0, 0]))},
}

common.load_spinup(solver_obj, mesh2d)
common.add_station_callbacks(solver_obj)

wall_time = common.run_and_time(solver_obj, update_forcings)
common.print_performance(solver_obj, wall_time,
                         'Option 4: rt-dg RT1/DG0 + DIRK22 + HybridizationPC')
