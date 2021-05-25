import pytest

def pytest_addoption(parser):
    parser.addoption("--model",action="store",help="Path of the model to be validated")

@pytest.fixture
def params(request):
    params = {}
    params["model"] = request.config.getoption("--model")
    if params["model"] is None:
        pytest.xfail()

    return params
