import sqlalchemy
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def delete_db():
    # Parámetros de conexión
    DB_HOST = "10.43.101.158"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"
    PORT = 3306

    # Crear conexión al motor de base de datos
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Conectar al motor
    with engine.connect() as connection:
        # Ejecutar comando para borrar la tabla
        connection.execute("DROP TABLE IF EXISTS train_database;")
        print("La tabla 'train_database' ha sido borrada.")

# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "06-Delete-Train-Data",
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
