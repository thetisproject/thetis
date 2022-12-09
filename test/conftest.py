from subprocess import check_call
import pytest


def pytest_runtest_teardown(item, nextitem):
    """Clear Thetis caches after running a test"""
    from firedrake.tsfc_interface import TSFCKernel
    from pyop2.global_kernel import GlobalKernel

    # disgusting hack, clear the Class-Cached objects in PyOP2 and
    # Firedrake, otherwise these will never be collected.  The Kernels
    # get very big with bendy on.
    TSFCKernel._cache.clear()
    GlobalKernel._cache.clear()


def parallel(item):
    """
    Run a test in parallel.

    NOTE: Copied from firedrake/src/firedrake/tests/conftest.py

    :arg item: The test item to run.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size > 1:
        raise RuntimeError("Parallel test can't be run within parallel environment")
    marker = item.get_closest_marker("parallel")
    if marker is None:
        raise RuntimeError("Parallel test doesn't have parallel marker")
    nprocs = marker.kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise RuntimeError("Need at least two processes to run parallel test")

    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process
    call = [
        "mpiexec", "-n", "1", "python", "-m", "pytest", "--runxfail", "-s", "-q",
        "%s::%s" % (item.fspath, item.name)
    ]
    call.extend([
        ":", "-n", "%d" % (nprocs - 1), "python", "-m", "pytest", "--runxfail", "--tb=no", "-q",
        "%s::%s" % (item.fspath, item.name)
    ])
    check_call(call)


def pytest_configure(config):
    """
    Register an additional marker.

    NOTE: Copied from firedrake/src/firedrake/tests/conftest.py
    """
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors")


@pytest.fixture(autouse=True)
def old_pytest_runtest_setup(request):
    item = request.node
    """
    Special setup for parallel tests.

    NOTE: Copied from firedrake/src/firedrake/tests/conftest.py
    """
    if item.get_closest_marker("parallel"):
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            # Turn on source hash checking
            from firedrake import parameters
            from functools import partial

            def _reset(check):
                parameters["pyop2_options"]["check_src_hashes"] = check

            # Reset to current value when test is cleaned up
            item.addfinalizer(partial(_reset,
                                      parameters["pyop2_options"]["check_src_hashes"]))

            parameters["pyop2_options"]["check_src_hashes"] = True
        else:
            # Blow away function arg in "master" process, to ensure
            # this test isn't run on only one process.
            item.obj = lambda *args, **kwargs: None


def pytest_runtest_call(item):
    """
    Special call for parallel tests.

    NOTE: Copied from firedrake/src/firedrake/tests/conftest.py
    """
    from mpi4py import MPI
    if item.get_closest_marker("parallel") and MPI.COMM_WORLD.size == 1:
        # Spawn parallel processes to run test
        parallel(item)
