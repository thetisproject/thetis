import glob
import os
import runpy
import shutil
import sys

import pytest
from mpi4py import MPI


os.environ['THETIS_REGRESSION_TEST'] = '1'

adjoint_files_ensemble = [
    'headland_inversion/inverse_problem.py',
]

cwd = os.path.abspath(os.path.dirname(__file__))
examples_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'examples'))
all_examples = [os.path.join(examples_dir, f) for f in adjoint_files_ensemble]


@pytest.fixture(params=all_examples, ids=lambda x: os.path.relpath(x, examples_dir))
def example_file(request):
    return os.path.abspath(request.param)


@pytest.mark.parallel(4)
def test_ensemble_examples(example_file, tmp_path_factory, monkeypatch):
    assert os.path.isfile(example_file), f'File not found {example_file}'

    source = os.path.dirname(example_file)
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        example_id = os.path.splitext(os.path.relpath(example_file, examples_dir))[0].replace(os.sep, '-')
        workdir = tmp_path_factory.mktemp(f'thetis-adjoint-ensemble-{example_id}')
        for f in glob.glob(os.path.join(source, '*.msh')):
            shutil.copy(f, str(workdir))
    else:
        workdir = None
    workdir = comm.bcast(str(workdir) if workdir is not None else None, root=0)
    comm.barrier()
    monkeypatch.chdir(workdir)

    added_to_syspath = False
    if source not in sys.path:
        sys.path.insert(0, source)
        added_to_syspath = True

    old_argv = sys.argv[:]
    sys.argv = [example_file, '--ensemble']
    try:
        runpy.run_path(example_file, run_name='__main__')
    finally:
        sys.argv = old_argv
        if added_to_syspath and sys.path and sys.path[0] == source:
            sys.path.pop(0)
