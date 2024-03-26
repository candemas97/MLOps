import requests
import json
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def read_data_from_api(group: int) ->json:
    """Read data from API given
    @params
    group[int]: Corresponds to the group where you belong
    @utput
    json: json with the needed data
    """
    response = requests.request("GET", f"http://10.43.101.149/data?group_number={group}")
    print(response)
    data_json = json.loads(response.content.decode('utf-8'))
    # Save data as a file in the folder
    with open(f"{os.getcwd()}/dags/corrida_{data_json['batch_number']}.json", 'w') as jf: 
        json.dump(response.json(), jf, ensure_ascii=False, indent=2)
    return data_json

# Creación del DAG y ejecución del mismo

"""
Se crea el dag y que se ejecute diariamente a las 00:00
"""
dag = DAG(
    "Cover-Type-Prediction",
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
    op_kwargs={"group": 9},
    dag=dag,
)

t1