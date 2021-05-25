import os
import sys

import pandas as pd
import numpy as np
import mlflow

PATH = os.environ["PYTHONPATH"]

if __name__ == "__main__":
    raw_data = sys.argv[1]

    red = pd.read_csv(f"{raw_data}/red_wine.csv",sep=";")
    white = pd.read_csv(f"{raw_data}/white_wine.csv",sep=";")

    red["type"] = 1
    white["type"] = 2

    data = pd.concat([red,white])
    data = data.sample(frac=1)

    data.to_csv(f"{PATH}/data/processed/data.csv",index=False)

    with mlflow.start_run():
        mlflow.log_artifacts(f"{PATH}/data/processed")
