

def pytest_runtest_teardown(item, nextitem):
    """Clear Thetis caches after running a test"""
    from firedrake.tsfc_interface import TSFCKernel
    from pyop2.op2 import Kernel
    from pyop2.parloop import JITModule
    from pyadjoint import get_working_tape

    # disgusting hack, clear the Class-Cached objects in PyOP2 and
    # Firedrake, otherwise these will never be collected.  The Kernels
    # get very big with bendy on.
    Kernel._cache.clear()
    TSFCKernel._cache.clear()
    JITModule._cache.clear()
    # clear the adjoint tape, so subsequent tests don't interfere
    get_working_tape().clear_tape()
