import os

import pandas as pd
import numpy as np
import sqlalchemy
import pymysql
import math

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def group_to_assign_diag_1(diagnostico):
    if diagnostico == "Data Filled Out":
        return "Data Filled Out"
    else:
        primera_letra = diagnostico[0].upper()
        if primera_letra.startswith(tuple("EFGHIJKLMNOPQRSTUV")):
            return "Other"
        else:
            if (390 <= math.floor(float(diagnostico)) <= 459) or math.floor(
                float(diagnostico)
            ) == 785:
                return "Circulatory"
            elif (460 <= math.floor(float(diagnostico)) <= 519) or math.floor(
                float(diagnostico)
            ) == 786:
                return "Respiratory"
            elif (520 <= math.floor(float(diagnostico)) <= 579) or math.floor(
                float(diagnostico)
            ) == 787:
                return "Digestive"
            elif math.floor(float(diagnostico)) == 250:
                return "Diabetes"
            elif 800 <= math.floor(float(diagnostico)) <= 999:
                return "Injury"
            elif 710 <= math.floor(float(diagnostico)) <= 739:
                return "Musculoskeletal"
            elif (580 <= math.floor(float(diagnostico)) <= 629) or math.floor(
                float(diagnostico)
            ) == 788:
                return "Genitourinary"
            elif 140 <= math.floor(float(diagnostico)) <= 239:
                return "Neoplasms"
            else:
                return "Other"


def preprocess_data() -> None:
    """
    Preprocess all data
    """
    # Rad Data for train, val and test

    DB_HOST = "10.56.1.20"  # Using MySQL IP address (ipv4_address in docker-compose)
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"

    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
    )  # Using DictCursos to obtain results as dictionaries
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
        "index",
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "weight",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "payer_code",
        "medical_specialty",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesMed",
        "readmitted",
    ]
    df_train.columns = columns_df
    df_validation.columns = columns_df
    df_test.columns = columns_df

    # Delete variables
    del_columns = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
        "max_glu_serum",
        "A1Cresult",
        "diag_2",
        "diag_3",
        "index",
    ]
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

    numerical_variables = [
        "time_in_hospital",
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

    y_train = df_train["readmitted"]
    X_train = df_train.drop(columns="readmitted")

    y_val = df_validation["readmitted"]
    X_val = df_validation.drop(columns="readmitted")

    y_test = df_test["readmitted"]
    X_test = df_test.drop(columns="readmitted")

    # Null Threatment

    # Look for categorical variables
    categorical_columns = X_train.select_dtypes(exclude=[int, float]).columns
    # Add new category
    X_train[categorical_columns] = X_train[categorical_columns].fillna("No Info")
    X_val[categorical_columns] = X_val[categorical_columns].fillna("No Info")
    X_test[categorical_columns] = X_test[categorical_columns].fillna("No Info")

    # Grouping some variables

    ## Group diag_1

    X_train["diag_1_group"] = X_train["diag_1"].apply(group_to_assign_diag_1)
    X_train = X_train.drop(columns=["diag_1"], axis=1)
    X_val["diag_1_group"] = X_val["diag_1"].apply(group_to_assign_diag_1)
    X_val = X_val.drop(columns=["diag_1"], axis=1)
    X_test["diag_1_group"] = X_test["diag_1"].apply(group_to_assign_diag_1)
    X_test = X_test.drop(columns=["diag_1"], axis=1)

    ## Category Grouping

    # To reduce the number of categories within each variable, we will create a group that if a category is less than 5% will join to a new group called "Group_Data"
    # Recalculate categorical variables
    categorical_columns = X_train.select_dtypes(exclude=[int, float]).columns
    for column in categorical_columns:
        categorical_counts = pd.value_counts(
            X_train[column]
        )  # We based on results given by train
        grouper = (categorical_counts / categorical_counts.sum() * 100).lt(
            5
        )  # Select the ones with less than 5%
        X_train[column] = np.where(
            X_train[column].isin(categorical_counts[grouper].index),
            "Group_Data",
            X_train[column],
        )
        X_val[column] = np.where(
            X_val[column].isin(categorical_counts[grouper].index),
            "High level",
            X_val[column],
        )
        X_test[column] = np.where(
            X_test[column].isin(categorical_counts[grouper].index),
            "High level",
            X_test[column],
        )

    # SAVING CLEAN MODEL
    # Creating final DataFrame to Upload
    df_train_final = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    df_val_final = pd.concat([X_val, pd.DataFrame(y_val)], axis=1)
    df_test_final = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)

    # Connect to MySQL
    engine = sqlalchemy.create_engine("mysql://root:airflow@mysql:3306/project_3")

    # Save data, if exits append into the current table (TRAIN)
    df_train_final.to_sql(
        "final_train_database", con=engine, if_exists="append", index=False
    )
    df_val_final.to_sql(
        "final_val_database", con=engine, if_exists="replace", index=False
    )
    df_test_final.to_sql(
        "final_test_database", con=engine, if_exists="replace", index=False
    )
    print("Saved into MySQL!")


# DAG creation and execution

"""
Create dag and set the schedule interval
"""
dag = DAG(
    "03-Preprocess-And-Save-Into-Clean-Data",
    description="DAG that Preprocess data and save in SQL",
    start_date=datetime(2024, 3, 25, 0, 0, 00000),
    schedule_interval="@once",
    catchup=False,
)

"""
Task 1: Read and Save
"""
t1 = PythonOperator(
    task_id="read_and_save_clean_data",
    provide_context=True,
    python_callable=preprocess_data,
    # op_kwargs={"group": 9},
    dag=dag,
)

t1
