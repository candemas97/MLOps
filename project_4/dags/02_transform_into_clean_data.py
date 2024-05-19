from airflow import DAG
from airflow.operators.python import PythonOperator
import json

import os
from datetime import datetime

import pandas as pd
import numpy as np
import sqlalchemy
import pymysql
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer # for dummies
from sklearn.pipeline import Pipeline # creating a pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def read_data(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str, PORT: int) -> json:
    """
    Read data from Raw Data
    """
    connection = pymysql.connect(host=DB_HOST,
                             user=DB_USER,
                             password=DB_PASSWORD,
                             db=DB_NAME,
                             cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_4.raw_data;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df = pd.DataFrame(result)
    except Exception as e:
        # If error returns the exact error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()
    # Show df
    print(f"The dataframe has {len(df)} rows")

    return df.to_json(orient="records")

    
def data_processing_and_save(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str, PORT: int, **context) -> None:
    """
    This step prepare the data and train the model
    """
    # Take data from previous step - data as JSON
    data = context["task_instance"].xcom_pull(task_ids="read_data") 
    df = pd.read_json(data, orient="records")

    # Removing not needed fields
    unique_columns_to_use = ["price", "bed", "bath", "acre_lot", "street", "city", "state", "house_size"]
    df = df[unique_columns_to_use]

    # Delete Nulls
    ## Putting "" as null 
    df.replace("", np.nan, inplace=True)
    df.replace("?", np.nan, inplace=True)
    df = df.dropna()

    # Division between y and the rest of variables
    y = df["price"]
    X = df.drop(columns="price")

    # Split train and test (80% train, 20% test)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.50, random_state=42)

    # Save data to clean data

    # Creating final DataFrame to Upload
    df_train_final = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    df_val_final = pd.concat([X_val, pd.DataFrame(y_val)], axis=1)
    df_test_final = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Save data, if exits append into the current table (TRAIN)
    df_train_final.to_sql('clean_data_train', con=engine, if_exists='replace', index=False)
    df_val_final.to_sql('clean_data_val', con=engine, if_exists='replace', index=False)
    df_test_final.to_sql('clean_data_test', con=engine, if_exists='replace', index=False)
    print("Saved into MySQL!")

# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "02-transform-to-clean-data",
    description='DAG that Read Store Data and prepare and Train Model',
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",  
    catchup=False,
)

"""
Task 3: Read Stored Data from MySQL
"""
t3 = PythonOperator(
    task_id="read_data",
    provide_context=True,
    python_callable=read_data,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    dag=dag,
)

"""
Task 4: Prepare and Train the ML Model
"""
t4 = PythonOperator(
    task_id="data_processing_and_save",
    provide_context=True,
    python_callable=data_processing_and_save,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    dag=dag,
)

t3 >> t4