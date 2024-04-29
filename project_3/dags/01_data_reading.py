import requests
import json
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sqlalchemy
import pandas as pd

def read_data_from_api() ->json:
    """Read data from API given
    @params
    group[int]: Corresponds to the group where you belong
    @utput
    json: json with the needed data
    """
    response = requests.request("GET", f"http://10.43.101.158/data-train-batches")
    print(response)
    data_json = json.loads(response.content.decode('utf-8'))
    # Save data as a file in the folder (uncomment if you need it)
    # with open(f"{os.getcwd()}/dags/corrida_{data_json['batch_number']}.json", 'w') as jf: 
    #     json.dump(response.json(), jf, ensure_ascii=False, indent=2)
    return data_json

def save_json_to_sql(**context) -> None:
    """"
    Save json read into MySQL
    """
    # Read data from previous step
    data_json = context["task_instance"].xcom_pull(
        task_ids="read_data_from_api"
    )

    # Transform JSON into a pd.DataFrame
    data = pd.DataFrame(data_json["data"])

    # Connect to MySQL
    DB_HOST = "10.43.101.158"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"
    PORT = 3306
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Save data, if exits append into the current table
    data.to_sql('train_database', con=engine, if_exists='append', index=False)
    print("Saved into MySQL!")




# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "00-Only-Read-And-Save-Train-Data-Batches",
    description='DAG that read from API and save in MySQL',
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",  
    catchup=False,
)

"""
Task 1: Read Data from API
"""
t1 = PythonOperator(
    task_id="read_data_from_api",
    provide_context=True,
    python_callable=read_data_from_api,
    # op_kwargs={"group": 9},
    dag=dag,
)

"""
Task 2: Save data in MySQL
"""
t2 = PythonOperator(
    task_id="save_json_to_sql",
    provide_context=True,
    python_callable=save_json_to_sql,
    dag=dag,
)

t1 >> t2