import pytest


def pytest_addoption(parser):
    parser.addoption("--travis", action="store_true", default=False,
                     help="Only run tests marked for Travis")


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "not_travis: Mark a test that should not be run on Travis")


def pytest_runtest_setup(item):
    not_travis = item.get_marker("not_travis")
    if not_travis is not None and item.config.getoption("--travis"):
        pytest.skip("Skipping test marked not for Travis")
