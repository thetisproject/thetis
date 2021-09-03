import pytest
import os
import subprocess
import glob


# set environment flag
# can be used in examples to reduce cpu cost
os.environ['THETIS_REGRESSION_TEST'] = "1"

cwd = os.path.abspath(os.path.dirname(__file__))
nb_dir = os.path.join(cwd, "..", "..", "demos")


# Discover the notebook files by globbing the notebook directory
@pytest.fixture(params=glob.glob(os.path.join(nb_dir, "*.ipynb")),
                ids=lambda x: os.path.basename(x))
def ipynb_file(request):
    return os.path.abspath(request.param)


def test_notebook_runs(ipynb_file, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    pytest = os.path.join(os.environ.get("VIRTUAL_ENV"), "bin", "pytest")
    subprocess.check_call([pytest, "--nbval-lax", ipynb_file])
