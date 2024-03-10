from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.mysql_operator import MySqlOperator

default_args = {"owner": "airflow", "start_date": datetime(2024, 3, 10)}
with DAG(
    dag_id="01-create-table-mysql", default_args=default_args, schedule_interval="@once"
) as dag:

    # Data Reading
    create_table = MySqlOperator(
        task_id="create_table",
        mysql_conn_id="mysql_db1",
        # sql="CREATE TABLE IF NOT EXISTS table_assignment_3(studyName VARCHAR(255) NULL, SampleNumber VARCHAR(255) NULL, Species VARCHAR(255) NULL, Region VARCHAR(255) NULL, Island VARCHAR(255) NULL, Stage VARCHAR(255) NULL, IndividualID VARCHAR(255) NULL, ClutchCompletion VARCHAR(50) NULL, DateEgg VARCHAR(50) NULL, CulmenLength_mm FLOAT NULL, CulmenDepth_mm FLOAT NULL, FlipperLength_mm FLOAT NULL, BodyMass_g FLOAT NULL, Sex VARCHAR(50) NULL, Delta15N_oo FLOAT NULL, Delta13C_oo FLOAT NULL, Comments TEXT NULL)",
        sql="CREATE TABLE IF NOT EXISTS table_assignment_3(studyName VARCHAR(255) NULL, SampleNumber VARCHAR(255) NULL, Species VARCHAR(255) NULL, Region VARCHAR(255) NULL, Island VARCHAR(255) NULL, Stage VARCHAR(255) NULL, IndividualID VARCHAR(255) NULL, ClutchCompletion VARCHAR(50) NULL, DateEgg VARCHAR(50) NULL, CulmenLength_mm VARCHAR(255) NULL, CulmenDepth_mm VARCHAR(255) NULL, FlipperLength_mm VARCHAR(255) NULL, BodyMass_g VARCHAR(255) NULL, Sex VARCHAR(50) NULL, Delta15N_oo VARCHAR(255) NULL, Delta13C_oo VARCHAR(255) NULL, Comments TEXT NULL)",
    )
    # Insert Data
    insert = MySqlOperator(
        task_id="insert_db",
        mysql_conn_id="mysql_db1",
        sql="""LOAD DATA  INFILE '/var/lib/mysql-files/penguins_lter.csv' INTO TABLE table_assignment_3 FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\r\n' IGNORE 1 ROWS;""",
    )

    # Run DAG
    create_table >> insert
