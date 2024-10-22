

def pytest_runtest_teardown(item, nextitem):
    """Clear Thetis caches after running a test"""
    from pyadjoint import get_working_tape

    # clear the adjoint tape, so subsequent tests don't interfere
    get_working_tape().clear_tape()
