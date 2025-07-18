"""
Runs all demo scripts. Only tests whether demos can be executed.
"""
import pytest
import os
import subprocess
import glob
import sys
import shutil

# set environment flag
# can be used in demos to reduce cpu cost
os.environ['THETIS_REGRESSION_TEST'] = "1"

cwd = os.path.abspath(os.path.dirname(__file__))
demos_dir = os.path.abspath(os.path.join(cwd, '..', '..', 'demos'))

all_demos = glob.glob(os.path.join(demos_dir, '*.py'))

@pytest.fixture(params=all_demos)
def demo_file(request):
    return os.path.abspath(request.param)


def test_demos(demo_file, tmp_path, monkeypatch):
    assert os.path.isfile(demo_file), 'File not found {:}'.format(demo_file)
    # copy data/ directory
    source = os.path.dirname(demo_file)
    data_dir = os.path.join(source, 'runfiles')
    shutil.copytree(data_dir, tmp_path / 'runfiles', symlinks=False)
    spinup_dir = os.path.join(source, 'outputs_spinup')
    shutil.copytree(spinup_dir, tmp_path / 'outputs_spinup', symlinks=False)
    # change workdir to temporary dir
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DATA', data_dir)
    subprocess.check_call([sys.executable, demo_file])
