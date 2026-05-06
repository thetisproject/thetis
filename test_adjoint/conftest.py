

def pytest_runtest_teardown(item, nextitem):
    """Clear adjoint tape after running a test"""
    from pyadjoint import pause_annotation, get_working_tape

    # pause and then clear the adjoint tape, so subsequent tests don't interfere
    pause_annotation()
    get_working_tape().clear_tape()
