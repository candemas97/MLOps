from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
import requests
import json

import pandas as pd

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
    engine = sqlalchemy.create_engine('mysql://root:airflow@mysql:3306/project_4')

    # Save data, if exits append into the current table
    data.to_sql('raw_data', con=engine, if_exists='append', index=False)
    print("Saved into MySQL!")

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

def read_clean_data(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str, PORT: int) -> json:
    """Read data from MySQL where is stored all data 
    @params
    DB_HOST[str]: Database IP
    BD_USER[str]: Database username
    DB_PASSWORD[str]: Database password
    DB_NAME[str]: Database Name
    PORT[int]: Port

    @output
    json: Json with the read data
    """
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute(f"SELECT * FROM {DB_NAME}.clean_data_train;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_train_clean = pd.DataFrame(result)

        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_4.clean_data_val;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_validation_clean = pd.DataFrame(result)

        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_4.clean_data_test;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_test_clean = pd.DataFrame(result)


    except Exception as e:
        # If error returns the exact error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()
    
    return df_train_clean.to_json(orient="records"), df_validation_clean.to_json(orient="records"), df_test_clean.to_json(orient="records")

def predict_mse_mae_actual_dataset(df_test: pd.DataFrame):
    """
    Predicts mse and mae with the data given
    """

    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    y_test = df_test["price"]
    X_test = df_test.drop(columns="price")
    
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:8084" # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")

    model_name = "model_final_project"

    # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

    print(model_production_uri)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    y_pred = loaded_model.predict(X_test)

    review_test_mse = mean_squared_error(y_test, y_pred)
    review_test_mae = mean_absolute_error(y_test, y_pred)

    print(f"Best model has an mse of {review_test_mse} and mae of {review_test_mae} in test")
    return review_test_mse, review_test_mae

def reviewing_if_needed_new_traing(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str, PORT: int, **context) -> bool:
    """
    Review if we need to re-train or not
    """
    # Take data from previous step - data as JSON

    data_train = context["task_instance"].xcom_pull(task_ids="read_data_clean", key="return_value")[0]
    data_val = context["task_instance"].xcom_pull(task_ids="read_data_clean", key="return_value")[1]
    data_test = context["task_instance"].xcom_pull(task_ids="read_data_clean", key="return_value")[2]

    if data_train is None or data_val is None or data_test is None:
        raise ValueError("No data returned from read_clean_data")

    #data_train, data_val, data_test = context["task_instance"].xcom_pull(task_ids="read_clean_data") 
    df_train_clean = pd.read_json(data_train, orient="records")
    df_validation_clean = pd.read_json(data_val, orient="records")
    df_test_clean = pd.read_json(data_test, orient="records")

    print(df_train_clean.head())

    # Creating stats (mean) to be store in SQL

    numerical_columns = df_train_clean.select_dtypes(include=['int64', 'float64']).columns.drop("price").to_list()
    mean_values = [df_train_clean[col].mean() for col in numerical_columns]
    categorical_columns = df_train_clean.select_dtypes(exclude=['int', 'float']).columns.to_list()
    count_values = [df_train_clean[col].nunique() for col in categorical_columns]
    mean_values.extend(count_values)
    numerical_columns.extend(categorical_columns)
    columns = numerical_columns.copy()
    mean_values.append(len(df_train_clean))
    columns.extend(["len_dataset"])

    # Save Metrics into SQL 

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    inspector = sqlalchemy.inspect(engine)
    table_name = 'metrics_retraining_to_be_saved'
    if table_name in inspector.get_table_names():
        # Search last element
        review_test_mse, review_test_mae = predict_mse_mae_actual_dataset(df_test_clean)
        mean_values.append(review_test_mse)
        mean_values.append(review_test_mae)
        columns.extend(["mse", "mae"])
        table_metrics = pd.DataFrame({key:mean_values[idx] for idx, key in enumerate(columns)}, index=[0])
        # Save data, if exits append into the current table
        table_metrics.to_sql(table_name, con=engine, if_exists='append', index=False)
        train = False
    else:
        mean_values.append(1000000000000000)
        mean_values.append(1000000000000000)
        columns.extend(["mse", "mae"])
        table_metrics = pd.DataFrame({key:mean_values[idx] for idx, key in enumerate(columns)}, index=[0])
        # Save data, if exits append into the current table
        table_metrics.to_sql("metrics_retraining_to_be_saved", con=engine, if_exists='append', index=False)
        train = True

    # Current model Accuracy - with new data

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Table to be sorted
    table_name = 'metrics_retraining_to_be_saved'

    # Search last element
    with engine.begin() as conn:
        query = sqlalchemy.text(f"SELECT * FROM {DB_NAME}.{table_name};") # ORDER BY {ordering_column} DESC LIMIT 1")
        last_element = pd.read_sql_query(query, conn)

    # Mostrar el último elemento
    save_last_element = last_element.tail(2).copy().reset_index(drop=True)

    # Veridying restrctions
    ## If it is the first time it will train
    if train == False:

        # Verify restriction 1 and 2

        if (save_last_element["mse"][0] * 1.15) < (save_last_element["mse"][1]) or (save_last_element["mae"][0] * 1.15) < (save_last_element["mae"][1]):
            train = True
            print("res 1 or res 2")

        # Verify restriction 3

        if (save_last_element["len_dataset"][0] * 1.5) < (save_last_element["len_dataset"][1]):
            train = True
            print("res 3")

        # Verify restriction 4

        r_bed = ((save_last_element["bed"][0] * 0.9) < (save_last_element["bed"][1]) and (save_last_element["bed"][0] * 1.1) > (save_last_element["bed"][1]))
        r_bath = ((save_last_element["bath"][0] * 0.9) < (save_last_element["bath"][1]) and (save_last_element["bath"][0] * 1.1) > (save_last_element["bath"][1]))
        r_acre_lot = ((save_last_element["acre_lot"][0] * 0.9) < (save_last_element["acre_lot"][1]) and (save_last_element["acre_lot"][0] * 1.1) > (save_last_element["acre_lot"][1]))
        r_street = ((save_last_element["street"][0] * 0.9) < (save_last_element["street"][1]) and (save_last_element["street"][0] * 1.1) > (save_last_element["street"][1]))
        r_house_size = ((save_last_element["house_size"][0] * 0.9) < (save_last_element["house_size"][1]) and (save_last_element["house_size"][0] * 1.1) > (save_last_element["house_size"][1]))

        if  r_bed == False or r_bath == False or r_acre_lot == False or r_street == False or r_house_size == False:
            train = True
            print("res 4")

        # Verify restriction 5

        r_city = (save_last_element["city"][0] * 1.05) > (save_last_element["city"][1])
        r_state = (save_last_element["state"][0] * 1.05) > (save_last_element["state"][1])

        if r_city == False and r_state == False:
            train = True
            print("res 5")
    
    if train == True:
        return "training_selecting_best_model_and_evaluate_solution"
    else:
        return "no_needed_new_trainig"

def no_needed_new_trainig():
    """
    Just printing a comment if not training is needed
    """
    print("It's not needed a new traing")

def training_selecting_best_model_and_evaluate_solution(DB_HOST: str, DB_USER: str, DB_PASSWORD: str, DB_NAME: str, PORT: int, **context) -> None:
    """
    
    """
    # Read data
    data_train, data_val, data_test = context["task_instance"].xcom_pull(task_ids="read_clean_data") 
    df_train_clean = pd.read_json(data_train, orient="records")
    df_validation_clean = pd.read_json(data_val, orient="records")
    df_test_clean = pd.read_json(data_test, orient="records")

    # Split data into y and X
    y_train = df_train_clean['price']
    X_train = df_train_clean.drop(columns = 'price')

    y_val = df_validation_clean['price']
    X_val = df_validation_clean.drop(columns = 'price')

    y_test = df_test_clean['price']
    X_test = df_test_clean.drop(columns = 'price')

    # Dummies
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                        ["city", "state"]),
                                      remainder='passthrough') # pass all the numeric values through the pipeline without any changes.

    # Standarization
    pipe = Pipeline(steps=[
        ("column_trans", column_trans),
        ("scaler", StandardScaler(with_mean=False)),
        ("RandomForestRegressor", RandomForestRegressor())
    ])

    # Hyperparameters
    param_grid =  dict()
    param_grid["RandomForestRegressor__max_depth"] = [1,2,3] 
    param_grid['RandomForestRegressor__n_estimators'] = [10,11]

    search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=2)

    # Train model
    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:8084" 
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")
    mlflow.set_experiment("mlflow_project_4")

    search.fit(X_train, y_train)
    # Mejor estimador después de la búsqueda
    best_estimator = search.best_estimator_

    # Uso del conjunto de prueba para evaluar el modelo final
    y_pred = best_estimator.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    # Log in MLflow
    with mlflow.start_run(run_name="autolog_pipe_model_reg") as run:
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(best_estimator, "model1")
        model_uri = f"runs:/{run.info.run_id}/model1"
        model_details = mlflow.register_model(model_uri=model_uri, name="model_final_project")
    
    # Determine the best model
    client = mlflow.tracking.MlflowClient()
    filter_string = "name='model_final_project'"
    all_model_versions = client.search_model_versions(filter_string)
    best_model = None
    best_mse = float("Inf")

    # Select best model
    for selected_model in all_model_versions:
        client.transition_model_version_stage(
            name="model_final_project",
            version=selected_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        for model in all_model_versions:
            if model.version != selected_model.version:
                client.transition_model_version_stage(
                    name="model_final_project",
                    version=model.version,
                    stage="Staging"
                )
        # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
        # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

        os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:8084" # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
        os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

        # connect to mlflow
        mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")

        model_name = "model_final_project"

        # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
        model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
        y_pred = loaded_model.predict(X_val)

        test_mse = mean_squared_error(y_val, y_pred)
        
        if test_mse < best_mse:
            best_mse = test_mse
            best_model = selected_model

    # add into production best model selected
    if best_model:
        client.transition_model_version_stage(
            name="model_final_project",
            version=best_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        for model in all_model_versions:
            if model.version != best_model.version:
                client.transition_model_version_stage(
                    name="model_final_project",
                    version=model.version,
                    stage="Staging"
                )

    print('tracking uri:', mlflow.get_tracking_uri())
    print('artifact uri:', mlflow.get_artifact_uri())

    ## Evaluation

    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:8084" # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")

    model_name = "model_final_project"

    # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

    print(model_production_uri)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    y_pred = loaded_model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"Best model has an mse of {test_mse} and mae of {test_mae} in test")

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')
    table_name = "metrics_retraining_to_be_saved"

    with engine.begin() as conn:
        query = sqlalchemy.text(f"""SELECT * FROM {table_name}""")
        df = pd.read_sql_query(query, conn)
        df = df.tail(1)
    dict_load = df.to_dict(orient="list")
    dict_load["mse"] = [test_mse]
    dict_load["mae"] = [test_mae]
    table_metrics = pd.DataFrame(dict_load)
    # Save data, if exits append into the current table
    table_metrics.to_sql(table_name, con=engine, if_exists='append', index=False)


# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "04-All-Together",
    description='DAG that read from API and save in MySQL',
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",  
    catchup=False,
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

"""
Task 5: Read Stored Data from MySQL
"""
t5 = PythonOperator(
    task_id="read_data_clean",
    provide_context=True,
    python_callable=read_clean_data,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    dag=dag,
)

"""
Task (Check if train): Review if we need to train or not 
"""

check_train = BranchPythonOperator(
    task_id='reviewing_if_needed_new_traing',
    python_callable=reviewing_if_needed_new_traing,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    provide_context=True,
    dag=dag,
)

"""
Task 6: Train, evaluate and store
"""
t6 = PythonOperator(
    task_id="training_selecting_best_model_and_evaluate_solution",
    provide_context=True,
    python_callable=training_selecting_best_model_and_evaluate_solution,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    dag=dag,
)

"""
Task 7: No training needed
"""

t7 = PythonOperator(
    task_id="no_needed_new_trainig",
    provide_context=True,
    python_callable=no_needed_new_trainig,
    op_kwargs={"DB_HOST": "mysql", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_4", "PORT": 3306},
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5 >> check_train
check_train >> t6
check_train >> t7