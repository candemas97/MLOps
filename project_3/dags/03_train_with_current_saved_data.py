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

from sklearn.ensemble import RandomForestClassifier

def read_clean_data(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str) -> json:
    """Read data from MySQL where is stored all data 
    @params
    DB_HOST[str]: Database IP
    BD_USER[str]: Database username
    DB_PASSWORD[str]: Database password
    DB_NAME[str]: Database Name

    @output
    json: Json with the read data
    """
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute(f"SELECT * FROM {DB_NAME}.final_train_database;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_train_clean = pd.DataFrame(result)
    except Exception as e:
        # If error returns the exact error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()
    
    return df_train_clean.to_json(orient="records")

def train_model(**context) -> None:
    """
    This step prepare the data and train the model
    """
    # Take data from previous step - data as JSON
    data = context["task_instance"].xcom_pull(task_ids="read_clean_data") 
    df_train_clean = pd.read_json(data, orient="records")

    # Split Data
    y_train = df_train_clean['readmitted']
    X_train = df_train_clean.drop(columns = 'readmitted')

    # Look for categorical variables
    categorical_columns = X_train.select_dtypes(exclude = [int, float]).columns

    # Some changes
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                            categorical_columns),
                                        remainder='passthrough') # pass all the numeric values through the pipeline without any changes.

    pipe = Pipeline(steps=[("column_trans", column_trans),("scaler", StandardScaler(with_mean=False)), ("RandomForestClassifier", RandomForestClassifier())])

    # Training
    param_grid =  dict()
    param_grid["RandomForestClassifier__max_depth"] = [1,2,3,10] 
    param_grid['RandomForestClassifier__n_estimators'] = [10,11]

    search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=2)

    # Use MLFlow
    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.56.1.22:9000" 
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")
    mlflow.set_experiment("mlflow_project_3")

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name="modelo1")

    with mlflow.start_run(run_name="autolog_pipe_model_reg") as run:
        search.fit(X_train, y_train)
    
    # Showing where was saved the new model / model version.
    print('\n\n\ntracking uri:', mlflow.get_tracking_uri(), '\nartifact uri:', mlflow.get_artifact_uri(), "\n\n")


# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "04-Read-Clean-Data-And-Train-Model",
    description='DAG that Read Store Clean Data and Train Model',
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",  
    catchup=False,
)

"""
Task 1: Read Stored Data from MySQL
"""
t1 = PythonOperator(
    task_id="read_clean_data",
    provide_context=True,
    python_callable=read_clean_data,
    op_kwargs={"DB_HOST": "10.56.1.20", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_3"},
    dag=dag,
)

"""
Task 2: Prepare and Train the ML Model
"""
t2 = PythonOperator(
    task_id="train_model",
    provide_context=True,
    python_callable=train_model,
    dag=dag,
)

t1 >> t2