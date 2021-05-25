import pytest

def pytest_addoption(parser):
    parser.addoption("--processed_data",action="store",help="Path of data to be validated")

@pytest.fixture
def params(request):
    params = {}
    params["processed_data"] = request.config.getoption("--processed_data")
    if params["processed_data"] is None:
        pytest.xfail()

    return params
