from __future__ import annotations

import pytest

from wdn_admm.data import load_problem_data, problem_data_path

NETWORKS = ["modena", "L_town", "bwfl_2022_05_hw"]


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="also run tests that solve full-size optimisation problems",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs a full optimisation solve")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def modena():
    if not problem_data_path("modena").exists():
        pytest.skip("modena problem data not available")
    return load_problem_data("modena")


@pytest.fixture(scope="session", params=NETWORKS)
def network(request):
    if not problem_data_path(request.param).exists():
        pytest.skip(f"{request.param} problem data not available")
    return load_problem_data(request.param)
