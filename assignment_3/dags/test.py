from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.hooks.mysql_hook import MySqlHook
from airflow.operators.mysql_operator import MySqlOperator
import pandas as pd
import requests
import json

default_args = {
    "owner": "user",
    "depends_on_past": False,
    "start_date": datetime(2020, 4, 15),
    "email": ["user@mail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}
dag = DAG("Helloworld", schedule_interval="@once", default_args=default_args)


def get_data(**kwargs):
    url = "https://jsonplaceholder.typicode.com/albums"
    resp = requests.get(url)
    if resp.status_code == 200:
        res = resp.json()
        return res
    return -1


def save_db(**kwargs):
    query = "select * from test_table"
    mysql_hook = MySqlHook(mysql_conn_id="mysql_test_conn", schema="testdb")
    connection = mysql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        print("*********", result)


def check_table_exists(**kwargs):
    query = (
        'select count(*) from information_schema.tables where table_name="test_table"'
    )
    mysql_hook = MySqlHook(mysql_conn_id="mysql_test_conn", schema="testdb")
    connection = mysql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    return results


def store_data(**kwargs):
    res = get_data()
    table_status = check_table_exists()
    mysql_hook = MySqlHook(mysql_conn_id="mysql_test_conn", schema="testdb")
    if table_status[0][0] == 0:
        print("----- table does not exists, creating it")
        create_sql = "create table test_table(col1 varchar(100), col2 varchar(100))"
        mysql_hook.run(create_sql)
    else:
        print("----- table already exists")
    if res != -1:
        for data in res:
            sql = "insert into test_table values (%s, %s)"
            mysql_hook.run(sql, parameters=(data.get("title"), data.get("id")))


py = PythonOperator(
    task_id="py_opt",
    dag=dag,
    python_callable=save_db,
)
py1 = PythonOperator(
    task_id="store_opt",
    dag=dag,
    python_callable=store_data,
)
t1 = BashOperator(
    task_id="task_1", bash_command='echo "Hello World from Task 1"', dag=dag
)
t1 >> py1 >> py
