from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import sqlalchemy

# import pymysql


def dtype_mapping():
    return {
        "object": "TEXT",
        "int64": "INT",
        "float64": "FLOAT",
        "datetime64": "DATETIME",
        "bool": "TINYINT",
        "category": "TEXT",
        "timedelta[ns]": "TEXT",
    }


"""
Create a sqlalchemy engine
"""


def mysql_engine(
    user="root",
    password="helloworld",
    host="172.17.0.1",
    port="3307",
    database="testdb",
):
    engine = sqlalchemy.create_engine(
        "mysql://{0}:{1}@{2}:{3}/{4}?charset=utf8".format(
            user, password, host, port, database
        )
    )
    return engine


"""
Create a mysql connection from sqlalchemy engine
"""


def mysql_conn(engine):
    conn = engine.raw_connection()
    return conn


"""
Create sql input for table names and types
"""


def gen_tbl_cols_sql(df):
    dmap = dtype_mapping()
    sql = "pi_db_uid INT AUTO_INCREMENT PRIMARY KEY"
    df1 = df.rename(columns={"": "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for hl in hdrs_list:
        sql += " ,{0} {1}".format(hl[0], dmap[hl[1]])
    return sql


"""
Create a mysql table from a df
"""


def create_mysql_tbl_schema(df, conn, db, tbl_name):
    tbl_cols_sql = gen_tbl_cols_sql(df)
    sql = "USE {0}; CREATE TABLE {1} ({2})".format(db, tbl_name, tbl_cols_sql)
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()


"""
Write df data to newly create mysql table
"""


def df_to_mysql(df, engine, tbl_name):
    df.to_sql(tbl_name, engine, if_exists="replace")


def read_csv() -> pd.DataFrame:
    # Where we currently are, usually we are at -> /opt/***/
    root_airflow = os.getcwd()
    # Join path to access to the needed csv file
    csv_path = f"{root_airflow}/dags/data/penguins_lter.csv"
    # Read CSV file
    df = pd.read_csv(csv_path)

    db = "testdb"
    db_tbl_name = "penguins"

    create_mysql_tbl_schema(df, mysql_conn(mysql_engine()), db, db_tbl_name)
    df_to_mysql(df, mysql_engine(), db_tbl_name)

    # engine = sqlalchemy.create_engine(
    #     "mysql+pymysql://root:helloworld@172.17.0.1:3307/testdb"
    # )

    # # Conectar manualmente al engine
    # connection = engine.connect()

    # # Realizar operaciones con la conexión
    # df.to_sql(name="penguins", con=connection, index=False, if_exists="replace")

    # # Cerrar la conexión manualmente cuando hayas terminado
    # connection.close()

    print("Successfully done!")


with DAG(
    dag_id="01-readingcsv-to-mysql",
    description="Reading a CSV with Python Operator",
    schedule_interval="@once",
    start_date=datetime(2024, 3, 3),
) as dag:

    t1 = PythonOperator(task_id="read_csv_save_mysql", python_callable=read_csv)

    t1
