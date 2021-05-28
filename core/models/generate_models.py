import sys

import numpy as np
import mlflow

from core.models import train

def cross_validate(processed_data):
    with mlflow.start_run():
        alphas = np.linspace(0,1,5)
        l1_ratios = np.linspace(0,1,5)

        for alpha in alphas:
            for l1_ratio in l1_ratios:
                train.train(alpha,l1_ratio,processed_data)

def find_best_model():
    metric = "rmse"
    client = mlflow.tracking.client.MlflowClient()
    # Parametrizing the right experiment path using widgets
    experiment_name = 'Default'
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_ids = [experiment.experiment_id]
    print("Experiment IDs:", experiment_ids)

    # Setting the decision criteria for a best run
    query = f"metrics.{metric} < 0.8"
    runs = client.search_runs(experiment_ids, query, mlflow.entities.ViewType.ALL)

    # Searching throught filtered runs to identify the best_run and build the model URI to programmatically reference later
    best_metric = None
    best_run = None
    for run in runs:
      if (best_metric == None or run.data.metrics[metric] < best_metric):
        best_metric = run.data.metrics[metric]
        best_run = run
    run_id = best_run.info.run_id
    print(f'Best {metric}: ', best_metric)
    print('Run ID: ', run_id)

    return run_id

if __name__ == "__main__":
    processed_data = sys.argv[1]

    cross_validate(processed_data)
