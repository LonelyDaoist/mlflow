import os
import sys
import shutil
from urllib.parse import urlparse

import mlflow

PATH = os.environ["PYTHONPATH"]

if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    version = sys.argv[3]

    with mlflow.start_run():
        print("Launching data download")
        download_run = mlflow.run(PATH,"download",parameters={})
        download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)
        raw_data = download_run.info.artifact_uri

        print("Launching data processing")
        etl_run = mlflow.run(PATH,"etl",parameters={"raw_data":raw_data})
        etl_run = mlflow.tracking.MlflowClient().get_run(etl_run.run_id)
        processed_data = etl_run.info.artifact_uri
        
        print("Launching processed data validation")
        test_etl_run = mlflow.run(PATH,"test_etl",parameters={"processed_data":processed_data})
        test_etl_run = mlflow.tracking.MlflowClient().get_run(test_etl_run.run_id)

        print("Launching model training")
        train_run = mlflow.run(PATH,"train",parameters={"alpha":alpha,
                                                        "l1_ratio":l1_ratio,
                                                        "processed_data":processed_data})
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)
        model = train_run.info.artifact_uri

        print("Launching models testing")
        test_models_run = mlflow.run(PATH,"test_models",parameters={"model":model})


        raw_path = urlparse(raw_data).path
        processed_path = urlparse(processed_data).path
        model_path = urlparse(model).path

        shutil.copytree(f"{raw_path}",f"{PATH}/data/raw/{version}")
        shutil.copytree(f"{processed_path}",f"{PATH}/data/processed/{version}")
        shutil.copytree(f"{model_path}/model",f"{PATH}/models/{version}")

