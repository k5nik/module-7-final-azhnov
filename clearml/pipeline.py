import joblib
import pandas as pd

from clearml import Task
from clearml.automation.controller import PipelineDecorator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


PROJECT_NAME = "taxi-final"
DATA_PATH = "s3://r-mlops-bucket-8-1-4-35446443/processed/processed_uber.csv"


@PipelineDecorator.component(return_values=["data_path"], cache=False)
def load_data(data_path: str):
    print(f"Using dataset: {data_path}")
    return data_path


@PipelineDecorator.component(return_values=["result"], cache=False, task_type="training")
def train_models(data_path: str, n_estimators: int, max_depth: int):
    task = Task.current_task()
    df = pd.read_csv(data_path)

    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "hour",
        "day_of_week",
        "distance",
    ]
    target_col = "fare_amount"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    model_path = f"model_{n_estimators}_{max_depth}.joblib"
    joblib.dump(model, model_path)

    task.upload_artifact("model", model_path)
    task.get_logger().report_scalar("metrics", "rmse", iteration=0, value=rmse)

    return {
        "rmse": rmse,
        "model_path": model_path,
        "task_id": task.id,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }


@PipelineDecorator.component(return_values=["best_result"], cache=False)
def evaluate(results: list):
    best_result = min(results, key=lambda x: x["rmse"])
    print(f"Best result: {best_result}")
    return best_result


@PipelineDecorator.pipeline(
    name="taxi_pipeline",
    project=PROJECT_NAME,
    version="1.0"
)
def pipeline_logic():
    data_path = load_data(data_path=DATA_PATH)

    r1 = train_models(data_path=data_path, n_estimators=50, max_depth=8)
    r2 = train_models(data_path=data_path, n_estimators=100, max_depth=10)
    r3 = train_models(data_path=data_path, n_estimators=200, max_depth=12)

    best = evaluate(results=[r1, r2, r3])
    print("Best model info:", best)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline_logic()