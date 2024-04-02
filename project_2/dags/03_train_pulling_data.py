from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import json

import os
from datetime import datetime

import pandas as pd
import numpy as np
import sqlalchemy
import pymysql
import mlflow
import requests
from sqlalchemy import create_engine


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer # for dummies
from sklearn.pipeline import Pipeline # creating a pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

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
    # Save data as a file in the folder (uncomment if you need it)
    # with open(f"{os.getcwd()}/dags/corrida_{data_json['batch_number']}.json", 'w') as jf: 
    #     json.dump(response.json(), jf, ensure_ascii=False, indent=2)
    return data_json

def save_json_to_sql(**context) -> None:
    """"
    Save json read into MySQL
    """
    # Read data from previous step
    data_json = context["task_instance"].xcom_pull(
        task_ids="read_data_from_api"
    )

    # Transform JSON into a pd.DataFrame
    data = pd.DataFrame(data_json["data"])

    # Connect to MySQL
    engine = create_engine('mysql://root:airflow@mysql:3306/project_2')

    # Save data, if exits append into the current table
    data.to_sql('dataset_covertype', con=engine, if_exists='append', index=False)
    print("Saved into MySQL!")

def read_data(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str) -> json:
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
            cursor.execute("SELECT * FROM project_2.dataset_covertype;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df = pd.DataFrame(result)
    except Exception as e:
        # If error returns the exact error
        print(f"Can't read data: {e}")
    finally:
        connection.close()
    
    return df.to_json(orient="records")

def data_processing_and_training(**context) -> None:
    """
    This step prepare the data and train the model
    """
    # Take data from previous step - data as JSON
    data = context["task_instance"].xcom_pull(task_ids="read_data") 
    df = pd.read_json(data, orient="records")

    # "12" is the variable to be predicted
    df.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

    # Convert from string to numeric

    for i in range(10):
        try:
            df[str(i)] = pd.to_numeric(df[str(i)], errors='raise')
        except Exception as e:
            print(f"Can't convert column {i} to number: {e}")
    df["12"] = pd.to_numeric(df["12"], errors='raise')

    # Split data in train and test

    ## Division between y and the rest of variables
    y = df["12"]
    X = df.drop(columns="12")

    ## Split train and test (99% train, 1% test) - Must of the data is gonna be tested with cross validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

    # Categorical to dummies

    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                            ["10", "11"]),
                                            remainder='passthrough') # pass all the numeric values through the pipeline without any changes.

    # Standarize and create pipeline

    pipe = Pipeline(steps=[("column_trans", column_trans),("scaler", StandardScaler(with_mean=False)), ("RandomForestClassifier", RandomForestClassifier())])

    # Hyperparameters
    param_grid =  dict()
    param_grid["RandomForestClassifier__max_depth"] = [1,2,3,10] 
    param_grid['RandomForestClassifier__n_estimators'] = [10,11]

    ## Create GridSearch for Hyperparameters
    search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=2)
    
    # Train model

    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTAINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.56.1.22:9000" # http://s3:9000 - container name
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://10.56.1.23:8087") # "http://mlflow:8087") - container name
    mlflow.set_experiment("mlflow_project_2")

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name="modelo1")

    with mlflow.start_run(run_name="autolog_pipe_model_reg") as run:
        search.fit(X_train, y_train)
    
    # Showing where was saved the new model / model version.
    print('\n\n\ntracking uri:', mlflow.get_tracking_uri(), '\nartifact uri:', mlflow.get_artifact_uri(), "\n\n")



# DAG creation and execution
dag = DAG(  'All-In-One-Read-Prepare-And-Train-Cover-Type', 
            description='DAG that read from API, Prepare and Train ML Model',
            start_date=datetime(2024, 3, 25, 0, 0, 00000),
            schedule_interval="@once",
            catchup=False
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

"""
Task 2: Save data in MySQL
"""
t2 = PythonOperator(
    task_id="save_json_to_sql",
    provide_context=True,
    python_callable=save_json_to_sql,
    dag=dag,
)

"""
Task 3: Read Stored Data from MySQL
"""
t3 = PythonOperator(
    task_id="read_data",
    provide_context=True,
    python_callable=read_data,
    op_kwargs={"DB_HOST": "10.56.1.20", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_2"},
    dag=dag,
)

"""
Task 4: Prepare and Train the ML Model
"""
t4 = PythonOperator(
    task_id="data_processing_and_training",
    provide_context=True,
    python_callable=data_processing_and_training,
    dag=dag,
)

t1 >> t2 >> t3 >> t4