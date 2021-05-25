import pytest

import pandas as pd
import mlflow.sklearn

@pytest.fixture
def model(params):
    path = params["model"]
    model = mlflow.sklearn.load_model(f"{path}/model")
    return model

@pytest.fixture
def testData():
    return pd.DataFrame([{
            "fixed acidity":7.4,
            "volatile acidity": 0.26,
            "critic acidity":0.31,
            "residual sugar":2.4,
            "chlorides":0.043,
            "free sulfur dioxide":58,
            "total sulfur dioxide":178,
            "density":0.9941,
            "pH":3.42,
            "sulphates":0.68,
            "alcohol":10.6,
            "type":2}])

def test_model(model,testData):

    predictions = model.predict(testData)
    assert all(3 <= p <= 9 for  p in predictions)
