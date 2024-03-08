from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

def read_csv(path: str) -> pd.DataFrame:
    # df = pd.read_csv(path)
    # print(df.head())
    # Crea un DataFrame de Pandas como ejemplo
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
    })

    # Realiza alguna operación con el DataFrame, por ejemplo, imprimirlo
    print(df)

with DAG (dag_id="3-readingcsv",
         description="Reading a CSV with Python Operator",
         schedule_interval="@once",
         start_date=datetime(2024,6,3)) as dag:

        
    t1 = PythonOperator(task_id= "read_csv_python",
                        python_callable=read_csv("../data/penguins_lter.csv"))