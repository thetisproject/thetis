"""
Runs all adjoint example scripts. Only tests whether examples can be executed.
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

# list of all adjoint examples to run
adjoint_files_serial = [
    'tidalfarm/tidalfarm.py',
    'channel_inversion/inverse_problem.py',
    'headland_inversion/inverse_problem.py',
    'tohoku_inversion/inverse_problem.py',
]

adjoint_files_parallel = [
    'discrete_turbines/channel-optimisation.py',
]

cwd = os.path.abspath(os.path.dirname(__file__))
examples_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'examples'))

all_examples = (
    [os.path.join(examples_dir, f) for f in adjoint_files_serial]
    + [
        pytest.param(os.path.join(examples_dir, f), marks=pytest.mark.parallel(2))
        for f in adjoint_files_parallel
    ]
)


@pytest.fixture(params=all_examples,
                ids=lambda x: os.path.relpath(x, examples_dir))
def example_file(request):
    return os.path.abspath(request.param)


def test_examples(example_file, tmp_path, tmp_path_factory, monkeypatch, request):
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
    # In CI this test is selected only in the adjoint-parallel job (outer mpiexec -n 2),
    # so we must not spawn mpiexec here (no nested MPI). We also coordinate a shared
    # working directory across ranks to avoid each rank creating its own tmpdir.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        workdir = tmp_path_factory.mktemp("thetis-adjoint-example-channel-optimisation")
        for f in glob.glob(os.path.join(source, '*.msh')):
            shutil.copy(f, str(workdir))
    else:
        workdir = None
    workdir = comm.bcast(str(workdir) if workdir is not None else None, root=0)
    comm.barrier()
    monkeypatch.chdir(workdir)
    runpy.run_path(example_file, run_name="__main__")
