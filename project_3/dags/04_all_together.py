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
import requests
import math


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer # for dummies
from sklearn.pipeline import Pipeline # creating a pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

def read_data_from_api() ->json:
    """Read data from API given
    @params
    group[int]: Corresponds to the group where you belong
    @utput
    json: json with the needed data
    """
    response = requests.request("GET", f"http://10.43.101.158/data-train-batches")
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
    DB_HOST = "10.43.101.158"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"
    PORT = 3306
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Save data, if exits append into the current table
    data.to_sql('train_database', con=engine, if_exists='append', index=False)
    print("Saved into MySQL!")

def group_to_assign_diag_1(diagnostico):
    if diagnostico == "Data Filled Out":
        return "Data Filled Out"
    else:
        primera_letra = diagnostico[0].upper()
        if primera_letra.startswith(tuple('EFGHIJKLMNOPQRSTUV')):
            return "Other"
        else:
            if (390 <= math.floor(float(diagnostico)) <= 459) or math.floor(float(diagnostico)) == 785:
                return "Circulatory"
            elif (460 <= math.floor(float(diagnostico)) <= 519) or math.floor(float(diagnostico)) == 786:
                return "Respiratory"
            elif (520 <= math.floor(float(diagnostico)) <= 579) or math.floor(float(diagnostico)) == 787:
                return "Digestive"
            elif math.floor(float(diagnostico)) == 250:
                return "Diabetes"
            elif (800 <= math.floor(float(diagnostico)) <= 999):
                return "Injury"
            elif (710 <= math.floor(float(diagnostico)) <= 739):
                return "Musculoskeletal"
            elif (580 <= math.floor(float(diagnostico)) <= 629) or math.floor(float(diagnostico)) == 788:
                return "Genitourinary"
            elif (140 <= math.floor(float(diagnostico)) <= 239):
                return "Neoplasms"
            else:
                return "Other"

def preprocess_data() -> None:
    """
    Preprocess all data
    """
    # Rad Data for train, val and test

    DB_HOST = "10.43.101.158"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"
    PORT = 3306
 
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        ## Train
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_3.train_database;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_train = pd.DataFrame(result)

        ## Validation
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_3.validation_database;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_validation = pd.DataFrame(result)

        ## Test
        with connection.cursor() as cursor:
            # Query the database
            cursor.execute("SELECT * FROM project_3.test_database;")
            result = cursor.fetchall()
        # Convert into a pd.DataFrame
        df_test = pd.DataFrame(result)

    except Exception as e:
        # If error returns the exact error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

    # Show dfs
    print(f"The dataframe has {len(df_train)} rows")
    print(f"The dataframe has {len(df_validation)} rows")
    print(f"The dataframe has {len(df_test)} rows")

    # Rename columns
    columns_df = [
        "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
        "num_procedures", "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
        "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
        "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
        "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
        "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
        "readmitted"
    ]
    df_train.columns = columns_df
    df_validation.columns = columns_df
    df_test.columns = columns_df

    # Delete variables
    del_columns = [ "encounter_id", 
                    "patient_nbr", 
                    "weight", 
                    "payer_code", 
                    "medical_specialty", 
                    "max_glu_serum", 
                    "A1Cresult", 
                    "diag_2", 
                    "diag_3",]
    df_train = df_train.drop(columns=del_columns, axis=1)
    df_validation = df_validation.drop(columns=del_columns, axis=1)
    df_test = df_test.drop(columns=del_columns, axis=1)

    # Change "" and "?" to np.NaN

    # Putting "" as null 
    df_train.replace("", np.nan, inplace=True)
    # Putting ? as null
    df_train.replace("?", np.nan, inplace=True)

    # Putting "" as null 
    df_validation.replace("", np.nan, inplace=True)
    # Putting ? as null
    df_validation.replace("?", np.nan, inplace=True)

    # Putting "" as null 
    df_test.replace("", np.nan, inplace=True)
    # Putting ? as null
    df_test.replace("?", np.nan, inplace=True)

    # Variable Types

    numerical_variables = ["time_in_hospital", 
                        "num_lab_procedures", 
                        "num_procedures", 
                        "num_medications", 
                        "number_outpatient", 
                        "number_emergency", 
                        "number_inpatient",
                        "number_diagnoses",
                        ] 
    df_train[numerical_variables] = df_train[numerical_variables].astype(int)
    df_validation[numerical_variables] = df_validation[numerical_variables].astype(int)
    df_test[numerical_variables] = df_test[numerical_variables].astype(int)

    # Split X y 

    y_train = df_train['readmitted']
    X_train = df_train.drop(columns = 'readmitted')

    y_val = df_validation['readmitted']
    X_val = df_validation.drop(columns = 'readmitted')

    y_test = df_test['readmitted']
    X_test = df_test.drop(columns = 'readmitted')

    # Null Threatment

    # Look for categorical variables
    categorical_columns = X_train.select_dtypes(exclude = [int, float]).columns
    # Add new category
    X_train[categorical_columns] = X_train[categorical_columns].fillna("No Info")
    X_val[categorical_columns] = X_val[categorical_columns].fillna("No Info")
    X_test[categorical_columns] = X_test[categorical_columns].fillna("No Info")

    # Grouping some variables

    ## Group diag_1

    X_train['diag_1_group'] = X_train['diag_1'].apply(group_to_assign_diag_1)
    X_train = X_train.drop(columns=['diag_1'], axis=1)
    X_val['diag_1_group'] = X_val['diag_1'].apply(group_to_assign_diag_1)
    X_val = X_val.drop(columns=['diag_1'], axis=1)
    X_test['diag_1_group'] = X_test['diag_1'].apply(group_to_assign_diag_1)
    X_test = X_test.drop(columns=['diag_1'], axis=1)

    ## Category Grouping

    # To reduce the number of categories within each variable, we will create a group that if a category is less than 5% will join to a new group called "Group_Data"
    # Recalculate categorical variables
    categorical_columns = X_train.select_dtypes(exclude = [int, float]).columns
    for column in categorical_columns:
        categorical_counts = pd.value_counts(X_train[column]) # We based on results given by train
        grouper = (categorical_counts/categorical_counts.sum() * 100).lt(5) # Select the ones with less than 5%
        X_train[column] = np.where(X_train[column].isin(categorical_counts[grouper].index),'Group_Data',X_train[column])
        X_val[column] = np.where(X_val[column].isin(categorical_counts[grouper].index),'Group_Data',X_val[column])
        X_test[column] = np.where(X_test[column].isin(categorical_counts[grouper].index),'Group_Data',X_test[column])

    # SAVING CLEAN MODEL
    # Creating final DataFrame to Upload
    df_train_final = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    df_val_final = pd.concat([X_val, pd.DataFrame(y_val)], axis=1)
    df_test_final = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Save data, if exits replace
    df_train_final.to_sql('final_train_database', con=engine, if_exists='replace', index=False)
    df_val_final.to_sql('final_val_database', con=engine, if_exists='replace', index=False)
    df_test_final.to_sql('final_test_database', con=engine, if_exists='replace', index=False)
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
    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "8084"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.158:8084" # 9000" 
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://10.43.101.158:8087") # "http://0.0.0.0:8087")
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
    "05-All-Steps-Together",
    description='DAG that read from API and save in MySQL, later clean and store in a new table, finally train a model',
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
    # op_kwargs={"group": 9},
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
Task 3: Read and Save
"""
t3 = PythonOperator(
    task_id="read_and_save_clean_data",
    provide_context=True,
    python_callable=preprocess_data,
    # op_kwargs={"group": 9},
    dag=dag,
)

"""
Task 4: Read Stored Data from MySQL
"""
t4 = PythonOperator(
    task_id="read_clean_data",
    provide_context=True,
    python_callable=read_clean_data,
    op_kwargs={"DB_HOST": "10.43.101.158", "DB_USER": "root", "DB_PASSWORD": "airflow", "DB_NAME": "project_3", "PORT": 3306},
    dag=dag,
)

"""
Task 5: Prepare and Train the ML Model
"""
t5 = PythonOperator(
    task_id="train_model",
    provide_context=True,
    python_callable=train_model,
    dag=dag,
)

t1 >> t2 >> t3 >> t4 >> t5