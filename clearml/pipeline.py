import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from clearml import PipelineController, OutputModel, Task


PROJECT_NAME = "mlops"
PIPELINE_NAME = "Train taxi fare models"
PIPELINE_VERSION = "0.0.1"

QUEUE_NAME = "default"  
DATASET_PATH = "s3://r-mlops-bucket-8-1-4-35446443/processed/processed_uber.csv"
S3_ENDPOINT_URL = "https://storage.yandexcloud.net"


# Загрузка и подготовка данных
def load_data(dataset_path):
    storage_options = {
        "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}
    }

    df = pd.read_csv(dataset_path, storage_options=storage_options)

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

    X = df[feature_cols].values
    y = df[target_col].values

    splitted_data = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return splitted_data


# Обучение RandomForest с заданными гиперпараметрами
def train_rf_model(data, n_estimators, max_depth, model_name):
    X_train, X_test, y_train, y_test = data

    task = Task.current_task()
    task.connect(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "model_name": model_name,
        }
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


# Валидация модели
def validate_model(model, data, model_name):
    X_train, X_test, y_train, y_test = data

    y_pred = model.predict(X_test)
    rmse = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 5).item()

    task = Task.current_task()
    logger = task.get_logger()
    logger.report_scalar("metrics", "rmse", iteration=0, value=rmse)

    results = {
        "rmse": rmse,
        "model_name": model_name,
    }

    print(f"{model_name} RMSE: {rmse}")

    return model, results


# Выбор лучшей модели и сохранение
def select_best(
    model_small,
    model_medium,
    model_large,
    results_small,
    results_medium,
    results_large
):
    task = Task.current_task()

    candidates = [
        (model_small, results_small),
        (model_medium, results_medium),
        (model_large, results_large),
    ]

    best_model, best_results = min(candidates, key=lambda x: x[1]["rmse"])

    print(f"Best model: {best_results['model_name']}")
    print(f"Best RMSE: {best_results['rmse']}")

    output_model = OutputModel(
        task=task,
        framework="ScikitLearn",
        name=best_results["model_name"],
        comment=f"Best taxi fare prediction model, RMSE={best_results['rmse']}",
        tags=["best_model", "taxi", "random_forest"],
    )

    model_filename = f"{best_results['model_name']}.pkl"
    joblib.dump(best_model, model_filename, compress=True)
    output_model.update_weights(model_filename)

    return {
        "best_model_name": best_results["model_name"],
        "best_rmse": best_results["rmse"],
        "model_file": model_filename,
    }


# ---------------------------
# Создание PipelineController
# ---------------------------
pipe = PipelineController(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version=PIPELINE_VERSION,
    docker="python:3.12-slim",
)

# Все шаги по умолчанию идут в очередь агентов
pipe.set_default_execution_queue(default_execution_queue=QUEUE_NAME)

# Параметр пайплайна: путь до подготовленного датасета
pipe.add_parameter(
    name="dataset_path",
    description="Path to processed taxi dataset",
    default=DATASET_PATH,
)

# Загрузка данных
pipe.add_function_step(
    name="load_data",
    function=load_data,
    function_kwargs=dict(
        dataset_path="${pipeline.dataset_path}"
    ),
    function_return=["splitted_data"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "s3fs>=2024.0.0",
        "fsspec>=2024.0.0",
    ],
)

# Обучение трех моделей с разными гиперпараметрами
pipe.add_function_step(
    name="train_rf_small",
    function=train_rf_model,
    function_kwargs=dict(
        data="${load_data.splitted_data}",
        n_estimators=50,
        max_depth=8,
        model_name="RandomForest_small",
    ),
    function_return=["rf_model_small"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

pipe.add_function_step(
    name="train_rf_medium",
    function=train_rf_model,
    function_kwargs=dict(
        data="${load_data.splitted_data}",
        n_estimators=100,
        max_depth=10,
        model_name="RandomForest_medium",
    ),
    function_return=["rf_model_medium"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

pipe.add_function_step(
    name="train_rf_large",
    function=train_rf_model,
    function_kwargs=dict(
        data="${load_data.splitted_data}",
        n_estimators=200,
        max_depth=12,
        model_name="RandomForest_large",
    ),
    function_return=["rf_model_large"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

# Валидация
pipe.add_function_step(
    name="validate_rf_small",
    function=validate_model,
    function_kwargs=dict(
        model="${train_rf_small.rf_model_small}",
        data="${load_data.splitted_data}",
        model_name="RandomForest_small",
    ),
    function_return=["model_small", "results_small"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

pipe.add_function_step(
    name="validate_rf_medium",
    function=validate_model,
    function_kwargs=dict(
        model="${train_rf_medium.rf_model_medium}",
        data="${load_data.splitted_data}",
        model_name="RandomForest_medium",
    ),
    function_return=["model_medium", "results_medium"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

pipe.add_function_step(
    name="validate_rf_large",
    function=validate_model,
    function_kwargs=dict(
        model="${train_rf_large.rf_model_large}",
        data="${load_data.splitted_data}",
        model_name="RandomForest_large",
    ),
    function_return=["model_large", "results_large"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
    ],
)

# Выбор лучшей модели
pipe.add_function_step(
    name="select_best_model",
    function=select_best,
    function_kwargs=dict(
        model_small="${validate_rf_small.model_small}",
        model_medium="${validate_rf_medium.model_medium}",
        model_large="${validate_rf_large.model_large}",
        results_small="${validate_rf_small.results_small}",
        results_medium="${validate_rf_medium.results_medium}",
        results_large="${validate_rf_large.results_large}",
    ),
    function_return=["best_model_info"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]>=2.0.2",
        "joblib>=1.3.0",
        "scikit-learn>=1.4.0",
    ],
)

# Запуск пайплайна
pipe.start(queue=QUEUE_NAME)