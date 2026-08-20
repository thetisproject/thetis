"""
Test exporting in ensemble mode. Based on `channel2d` example.
"""
from pathlib import Path
import pytest
from thetis import *

@pytest.mark.parallel(nprocs=4)
def test_channel2d_ensemble_exports(tmp_path, monkeypatch):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        workdir = str(tmp_path)
    else:
        workdir = None
    workdir = comm.bcast(workdir, root=0)
    comm.barrier()
    monkeypatch.chdir(workdir)

    os.environ['THETIS_REGRESSION_TEST'] = '1'

    ensemble = Ensemble(comm, 2)
    ensemble_rank = ensemble.ensemble_rank
    spatial_comm = ensemble.comm

    lx = 100e3
    ly = 3750
    nx = 80
    ny = 3
    mesh2d = RectangleMesh(nx, ny, lx, ly, comm=spatial_comm)

    t_export = 100.0
    t_end = 5*t_export
    u_mag = Constant(6.0)

    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    x, y = SpatialCoordinate(mesh2d)
    depth_oce = 20.0
    depth_riv = 5.0
    bathymetry_2d.interpolate(depth_oce + (depth_riv - depth_oce)*x/lx)

    manning_values = (0.02, 0.03)
    manning_value = manning_values[ensemble_rank % len(manning_values)]
    manning_2d = Function(p1_2d, name='ManningCoefficient').assign(manning_value)

    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    options.output_directory = f'outputs_member_{ensemble_rank}'
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.horizontal_velocity_scale = u_mag
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
    options.swe_timestepper_type = 'SSPRK33'
    options.manning_drag_coefficient = manning_2d
    if not hasattr(options.swe_timestepper_options, 'use_automatic_timestep'):
        options.timestep = 10.0

    elev_init = Function(p1_2d)
    elev_height = 6.0
    elev_ramp_lx = 30e3
    elev_init.interpolate(conditional(x < elev_ramp_lx,
                                      elev_height*(1 - x/elev_ramp_lx),
                                      0.0))
    solver_obj.assign_initial_conditions(elev=elev_init)
    solver_obj.iterate()

    comm.barrier()

    output_dir = Path(workdir) / f'outputs_member_{ensemble_rank}'
    assert output_dir.is_dir()

    pvd_targets = {
        'Elevation2d': output_dir / 'Elevation2d' / 'Elevation2d.pvd',
        'Velocity2d': output_dir / 'Velocity2d' / 'Velocity2d.pvd',
    }
    for path in pvd_targets.values():
        assert path.is_file()

    hdf5_dir = output_dir / 'hdf5'
    assert hdf5_dir.is_dir()

    # Check that the .h5 files exist and can be loaded correctly
    hdf5_targets = {
        'elev_2d': 'Elevation2d',
        'uv_2d': 'Velocity2d',
    }

    expected_exports = int(round(t_end / t_export)) + 1

    for expected_name, prefix in hdf5_targets.items():
        files = [hdf5_dir / f'{prefix}_{i:05d}.h5' for i in range(expected_exports)]
        for h5path in files:
            assert h5path.is_file(), f'missing {expected_name} export {h5path}'
            with CheckpointFile(str(h5path), 'r', comm=spatial_comm) as chk:
                loaded_mesh = chk.load_mesh()
                try:
                    chk.load_function(loaded_mesh, expected_name)
                except Exception as err:
                    raise AssertionError(
                        f"Could not load '{expected_name}' from {h5path}: {err!r}"
                    ) from err
