import sqlalchemy
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def delete_db():
    # Parámetros de conexión
    DB_HOST = "mysql"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_4"
    PORT = 3306

    # Crear conexión al motor de base de datos
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    for table in ["raw_data", "clean_data_train", "clean_data_val", "clean_data_test", "metrics_retraining_to_be_saved"]:
        # Conectar al motor
        with engine.connect() as connection:
            # Ejecutar comando para borrar la tabla
            connection.execute(f"DROP TABLE IF EXISTS {table};")
            print(f"La tabla '{table}' ha sido borrada.")

# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "05-Delete-All-Data",
    description="DAG that Preprocess data and save in SQL",
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",
    catchup=False,
)

"""
Task 1: Delete
"""
t1 = PythonOperator(
    task_id="delete_db",
    provide_context=True,
    python_callable=delete_db,
    # op_kwargs={"group": 9},
    dag=dag,
)

t1