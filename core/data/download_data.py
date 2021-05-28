import os

import requests
import mlflow

PATH = os.environ["PYTHONPATH"]

def download():
    urls = {
            "red_wine": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "white_wine": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    }

    for key in urls:
        r = requests.get(urls[key])
        open(f"{PATH}/data/raw/tmp/{key}.csv","wb").write(r.content)

    with mlflow.start_run():
        mlflow.log_artifacts(f"{PATH}/data/raw/tmp")

if __name__ == "__main__":
    download()
