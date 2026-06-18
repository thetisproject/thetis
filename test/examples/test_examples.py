"""
Runs all example scripts. Only tests whether examples can be executed.
"""
import pytest
import os
import subprocess
import glob
import sys
import shutil
import runpy

# set environment flag
# can be used in examples to reduce cpu cost
os.environ['THETIS_REGRESSION_TEST'] = "1"

exclude_files = [
    'baroclinic_eddies/diagnostics.py',
    'baroclinic_eddies/submitRuns.py',
    'bottomFriction/plot_results.py',
    'columbia_plume/atm_forcing.py',
    'columbia_plume/bathymetry.py',
    'columbia_plume/cre-plume.py',
    'columbia_plume/diagnostics.py',
    'columbia_plume/ncom_forcing.py',
    'columbia_plume/plot_elevation_ts.py',
    'columbia_plume/plot_salt_profile.py',
    'columbia_plume/roms_forcing.py',
    'columbia_plume/bath_smoothing_test.py',
    'columbia_plume/tidal_forcing.py',
    'columbia_plume/timeseries_forcing.py',
    'discrete_turbines/channel-optimisation.py',
    'dome/diagnostics.py',
    'dome/dome_setup.py',
    'dome/plot_histogram.py',
    'katophillips/plot_results.py',
    'lockExchange/diagnostics.py',
    'lockExchange/plotting.py',
    'lockExchange/submitRuns.py',
    'sediment_trench_2d/trench_example.py',
    'sediment_meander_2d/meander_example.py',
    'tidalfarm/tidalfarm.py',
    'tidal_barrage/plotting.py',
    'channel_inversion/plot_elevation_progress.py',
    'channel_inversion/inverse_problem.py',
    'headland_inversion/forward_run.py',
    'headland_inversion/inverse_problem.py',
    'headland_inversion/inversion_tools_vel.py',
    'headland_inversion/plot_velocity_progress.py',
    'tohoku_inversion/inverse_problem.py',
    'tohoku_inversion/okada.py',
    'tohoku_inversion/plot_convergence.py',
    'tohoku_inversion/plot_elevation_initial_guess.py',
    'tohoku_inversion/plot_elevation_progress.py',
    'tohoku_inversion/plot_optimized_source.py',
    'tohoku_inversion/plot_elevation_optimized.py',
    'north_sea/generate_mesh.py',
    'north_sea/model_config.py',
    'north_sea/plot_elevation.py',
    'north_sea/plot_setup.py',
    'north_sea/spinup.py',
    'north_sea/run.py',
]

cwd = os.path.abspath(os.path.dirname(__file__))
examples_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'examples'))

exclude_files = [os.path.join(examples_dir, f) for f in exclude_files]

all_examples = glob.glob(os.path.join(examples_dir, '*/*.py'))
all_examples = [f for f in all_examples if f not in exclude_files]

# Examples that should be exercised under MPI in the main parallel CI job.
# Keyed by path relative to `examples_dir` with the requested nprocs.
parallel_examples = {
    os.path.join('discrete_turbines', 'tidal_array.py'): 2,
}

for relpath, nprocs in parallel_examples.items():
    abspath = os.path.join(examples_dir, relpath)
    try:
        idx = all_examples.index(abspath)
    except ValueError:
        continue
    all_examples[idx] = pytest.param(abspath, marks=pytest.mark.parallel(nprocs))


@pytest.fixture(params=all_examples,
                ids=lambda x: os.path.relpath(x, examples_dir))
def example_file(request):
    return os.path.abspath(request.param)


def test_examples(example_file, tmp_path, monkeypatch, request):
    assert os.path.isfile(example_file), 'File not found {:}'.format(example_file)
    # copy mesh files
    source = os.path.dirname(example_file)

    if request.node.get_closest_marker("parallel") is None:
        # Serial example: copy mesh files and run in a subprocess.
        for f in glob.glob(os.path.join(source, '*.msh')):
            shutil.copy(f, str(tmp_path))
        # change workdir to temporary dir
        monkeypatch.chdir(tmp_path)
        subprocess.check_call([sys.executable, example_file])
        return

    # Parallel example: run the script under the current MPI communicator.
    # In CI this test is selected only in the main parallel job (outer mpiexec -n 2),
    # so we must not spawn mpiexec here (no nested MPI). We also coordinate a shared
    # working directory across ranks to avoid each rank creating its own tmp_path.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        workdir = tmp_path
        for f in glob.glob(os.path.join(source, '*.msh')):
            shutil.copy(f, str(workdir))
    else:
        workdir = None
    workdir = comm.bcast(str(workdir) if workdir is not None else None, root=0)
    comm.barrier()
    monkeypatch.chdir(workdir)
    # Make local imports like `import turbine_callback` work the same way they do
    # when running `python /abs/path/to/example.py` (which prepends the script dir
    # to sys.path).
    added_to_syspath = False
    if source not in sys.path:
        sys.path.insert(0, source)
        added_to_syspath = True
    try:
        runpy.run_path(example_file, run_name="__main__")
    finally:
        if added_to_syspath and sys.path and sys.path[0] == source:
            sys.path.pop(0)
