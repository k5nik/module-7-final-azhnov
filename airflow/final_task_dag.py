from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import os
import pandas as pd
import pandera as pa
from pandera import Column, Check


AWS_CONNECTION_ID = "s3"  # Airflow connection id to s3
BUCKET_NAME = "r-mlops-bucket-8-1-4-35446443"
RAW_DATA_PREFIX = "raw"
TRANSFORMED_DATA_PREFIX = "processed"

RAW_FILE_NAME = "uber.csv"
PROCESSED_FILE_NAME = "processed_uber.csv"


def strict_datetime_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series,
        format="%Y-%m-%d %H:%M:%S UTC",
        errors="coerce"
    )
    return parsed.notna()


taxi_schema = pa.DataFrameSchema(
    columns={
        "key": Column(str, required=True),

        "pickup_datetime": Column(
            str,
            required=True,
            checks=Check(
                strict_datetime_column,
                error="pickup_datetime: не все значения соответствуют формату 'YYYY-MM-DD HH:MM:SS UTC'"
            )
        ),

        "fare_amount": Column(
            float,
            required=True,
            checks=[
                Check.ge(0),
                Check.le(100),
            ]
        ),

        "pickup_longitude": Column(
            float,
            required=True,
            checks=Check.in_range(-180, 180)
        ),

        "pickup_latitude": Column(
            float,
            required=True,
            checks=Check.in_range(-90, 90)
        ),

        "dropoff_longitude": Column(
            float,
            required=True,
            checks=Check.in_range(-180, 180)
        ),

        "dropoff_latitude": Column(
            float,
            required=True,
            checks=Check.in_range(-90, 90)
        ),

        "passenger_count": Column(
            int,
            required=True,
            checks=[
                Check.ge(1),
                Check.lt(10)
            ]
        ),
    },
    strict=True,
    ordered=False
)


def process_s3_data(**context):
    local_dir = "/tmp/worker_data"
    os.makedirs(local_dir, exist_ok=True)

    raw_local_path = os.path.join(local_dir, RAW_FILE_NAME)
    processed_local_path = os.path.join(local_dir, PROCESSED_FILE_NAME)

    raw_s3_key = f"{RAW_DATA_PREFIX}/{RAW_FILE_NAME}"
    processed_s3_key = f"{TRANSFORMED_DATA_PREFIX}/{PROCESSED_FILE_NAME}"

    s3_hook = S3Hook(aws_conn_id=AWS_CONNECTION_ID)

    # Скачивание исходного файла из S3
    file_content = s3_hook.read_key(
        key=raw_s3_key,
        bucket_name=BUCKET_NAME
    )

    with open(raw_local_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    print(f"Загружено из S3: s3://{BUCKET_NAME}/{raw_s3_key} -> {raw_local_path}")

    # Чтение исходных данных
    df = pd.read_csv(raw_local_path).drop(columns="Unnamed: 0", errors="ignore")
    print(f"Строк до очистки: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")

    # Очистка данных ДО валидации
    df = df.drop_duplicates()

    df = df.dropna(
        subset=[
            "key",
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    )

    df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
    df["pickup_longitude"] = pd.to_numeric(df["pickup_longitude"], errors="coerce")
    df["pickup_latitude"] = pd.to_numeric(df["pickup_latitude"], errors="coerce")
    df["dropoff_longitude"] = pd.to_numeric(df["dropoff_longitude"], errors="coerce")
    df["dropoff_latitude"] = pd.to_numeric(df["dropoff_latitude"], errors="coerce")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    )

    df["passenger_count"] = df["passenger_count"].astype(int)

    # фильтрация под схему (по заданию нужно взять taxi_schema из предыдущих уроков)
    df = df[
        (df["fare_amount"] >= 0) &
        (df["fare_amount"] <= 100) &
        (df["pickup_longitude"].between(-180, 180)) &
        (df["pickup_latitude"].between(-90, 90)) &
        (df["dropoff_longitude"].between(-180, 180)) &
        (df["dropoff_latitude"].between(-90, 90)) &
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] < 10)
    ].copy()

    print(f"Строк после очистки, до валидации: {len(df)}")

    # Валидация очищенных данных
    df = taxi_schema.validate(df, lazy=True)

    # Трансформация
    df["pickup_datetime_parsed"] = pd.to_datetime(
        df["pickup_datetime"],
        format="%Y-%m-%d %H:%M:%S UTC",
        errors="coerce"
    )
    df = df.dropna(subset=["pickup_datetime_parsed"])

    df["hour"] = df["pickup_datetime_parsed"].dt.hour
    df["day_of_week"] = df["pickup_datetime_parsed"].dt.dayofweek

    df["distance"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2 +
        (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    ) ** 0.5

    df = df[df["distance"] > 0].copy()

    print(f"Строк после преобразований: {len(df)}")

    df = df.drop(columns=["pickup_datetime_parsed"])

    # Сохранение локально
    df.to_csv(processed_local_path, index=False)

    # Загрузка результата обратно в S3
    s3_hook.load_file(
        filename=processed_local_path,
        key=processed_s3_key,
        bucket_name=BUCKET_NAME,
        replace=True
    )

    print(f"Загружен в S3: {processed_local_path} -> s3://{BUCKET_NAME}/{processed_s3_key}")

    return {
        "uploaded_files": [processed_s3_key],
        "bucket_name": BUCKET_NAME,
        "rows_after_processing": len(df),
    }


with DAG(
    dag_id="final_task_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    tags=["IvanA"],
    catchup=False,
    description="uber.csv pipeline: check_raw_data -> process_data -> check_transformed_data"
) as dag:

    check_raw_files_in_s3 = S3KeySensor(
        task_id="check_raw_files_in_s3",
        bucket_name=BUCKET_NAME,
        bucket_key=f"{RAW_DATA_PREFIX}/{RAW_FILE_NAME}",
        aws_conn_id=AWS_CONNECTION_ID,
        timeout=300,
        poke_interval=30,
        mode="poke"
    )

    process_data_from_s3 = PythonOperator(
        task_id="process_data_from_s3",
        python_callable=process_s3_data
    )

    check_transformed_files_in_s3 = S3KeySensor(
        task_id="check_transformed_files_in_s3",
        bucket_name=BUCKET_NAME,
        bucket_key=f"{TRANSFORMED_DATA_PREFIX}/{PROCESSED_FILE_NAME}",
        aws_conn_id=AWS_CONNECTION_ID,
        timeout=300,
        poke_interval=10,
        mode="poke"
    )

    check_raw_files_in_s3 >> process_data_from_s3 >> check_transformed_files_in_s3