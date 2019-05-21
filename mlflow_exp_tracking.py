import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


import click
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

experimentPath = "experiment-L3"

try:
  experimentID = mlflow.create_experiment(experimentPath)
except MlflowException:
  experimentID = MlflowClient().get_experiment_by_name(experimentPath).experiment_id
  mlflow.set_experiment(experimentPath)

print("The experiment can be found at the path `{}` and has an experiment_id of `{}`".format(experimentPath, experimentID))


@click.command()
@click.option("--data_path", default="airbnb-cleaned-mlflow.csv", type=str)
@click.option("--n_estimators", default=10, type=int)
@click.option("--max_depth", default=20, type=int)
@click.option("--max_features", default="auto", type=str)
@click.option("--experiment_id", default=experimentID, type=int)
@click.option("--run_name", default="experiment-L3", type=str)
def mlflow_rf(data_path, n_estimators, max_depth, max_features,experiment_id,run_name):

  with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    # Import the data
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
    
    # Create model, train it, and create predictions
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")
    
    # Log params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Log metrics
    mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
    mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
    mlflow.log_metric("r2", r2_score(y_test, predictions))  

## To run --n_estimators', 10, '--max_depth', 20, '--experiment_id', experimentID

if __name__ == "__main__":
   mlflow_rf() # Note that this does not need arguments thanks to click