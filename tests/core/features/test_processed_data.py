import pytest

import pandas as pd

@pytest.fixture
def data(params):
    path = params["processed_data"]
    data = pd.read_csv(f"{path}/data.csv")
    return data

def test_nan(data):
    assert not data.isnull().values.any()

def test_type_exists(data):
    assert "type" in data.columns

def test_number_columns(data):
    assert data.shape[1] == 13
