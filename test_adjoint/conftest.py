

def pytest_runtest_teardown(item, nextitem):
    """Clear adjoint tape after running a test"""
    from pyadjoint import get_working_tape

    # clear the adjoint tape, so subsequent tests don't interfere
    tape = get_working_tape()
    if tape is not None:
        tape.clear_tape()
