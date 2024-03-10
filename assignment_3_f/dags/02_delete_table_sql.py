from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.mysql_operator import MySqlOperator

default_args = {"owner": "airflow", "start_date": datetime(2024, 3, 10)}
with DAG(
    dag_id="02-delete-table-mysql", default_args=default_args, schedule_interval="@once"
) as dag:

    # Delete information in DAG
    delete_table = MySqlOperator(
        task_id="delete_table",
        mysql_conn_id="mysql_db1",
        sql="DROP TABLE IF EXISTS table_assignment_3;",
    )

    # Run DAG
    delete_table
