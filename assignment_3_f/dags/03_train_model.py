## Basic Libraries
import numpy as np
import pandas as pd

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from datetime import datetime
import os

# Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.mysql_hook import MySqlHook


def query_db_to_dataframe() -> pd.DataFrame:
    """Read MySQl database and return df

    Returns:
        pd.DataFrame: data within MySql
    """
    # Connect to the database
    mysql_hook = MySqlHook(mysql_conn_id="mysql_db1", schema="mysql")
    conn = mysql_hook.get_conn()
    # Run query
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM table_assignment_3")
    result = cursor.fetchall()
    # Close
    cursor.close()
    conn.close()

    # Convert result in a DataFrame
    df = pd.DataFrame(result, columns=[i[0] for i in cursor.description])

    # Change Column Name
    df.columns = [
        "studyName",
        "Sample Number",
        "Species",
        "Region",
        "Island",
        "Stage",
        "Individual ID",
        "Clutch Completion",
        "Date Egg",
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
        "Flipper Length (mm)",
        "Body Mass (g)",
        "Sex",
        "Delta 15 N (o/oo)",
        "Delta 13 C (o/oo)",
        "Comments",
    ]

    # Changing "" for np.nan
    df[
        [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
            "Delta 15 N (o/oo)",
            "Delta 13 C (o/oo)",
        ]
    ] = df[
        [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
            "Delta 15 N (o/oo)",
            "Delta 13 C (o/oo)",
        ]
    ].replace(
        {"": np.nan}
    )

    # Modify some data types
    df[
        [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
            "Delta 15 N (o/oo)",
            "Delta 13 C (o/oo)",
        ]
    ] = df[
        [
            "Culmen Length (mm)",
            "Culmen Depth (mm)",
            "Flipper Length (mm)",
            "Body Mass (g)",
            "Delta 15 N (o/oo)",
            "Delta 13 C (o/oo)",
        ]
    ].astype(
        "float64"
    )

    return df


def read_csv_train_model() -> pd.DataFrame:
    """Read data in Airflow and transform it to a DataFrame

    Returns:
        pd.DataFrame: data in csv
    """
    # Where we currently are, usually we are at -> /opt/***/
    root_airflow = os.getcwd()
    print(f"\nAQUI VER: {root_airflow}")

    print(os.listdir(root_airflow))
    # Join path to access to the needed csv file
    csv_path = f"{root_airflow}/op_files/penguins_lter.csv"  # f"{root_airflow}/dags/data/penguins_lter.csv"
    # Read CSV file
    df = pd.read_csv(csv_path)

    return df


def train_model():
    ## Reading Data

    data = query_db_to_dataframe()

    ## Transfoma Data
    data["Sex"] = data["Sex"].replace({".": None, np.nan: None})
    data_x = pd.DataFrame()
    for col in ["Island", "Clutch Completion", "Sex"]:
        idxs_none = data[data[col].isna()].index.tolist()
        dummies = pd.get_dummies(data[col], drop_first=True, prefix=col, dtype=int)
        dummies.loc[idxs_none, :] = None
        data_x = pd.concat([data_x, dummies], axis=1)
    for col in [
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
        "Flipper Length (mm)",
        "Body Mass (g)",
        "Delta 15 N (o/oo)",
        "Delta 13 C (o/oo)",
    ]:
        data_x = pd.concat([data_x, data[col]], axis=1)
    data_y = pd.factorize(data["Species"])[0]

    ## Data Preparation
    X, y = data_x.values, data_y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

    # We look every column type
    categorical_columns = X_train.select_dtypes(exclude=[int, float]).columns
    numerical_columns = X_train.select_dtypes(include=[int, float]).columns

    # Impute variables
    categoricas_imputer = SimpleImputer(missing_values=np.nan, strategy="constant")
    numericas_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

    # Training Imputers
    numericas_imputer.fit(X_train[numerical_columns])

    # Impute variables
    X_train[numerical_columns] = numericas_imputer.transform(X_train[numerical_columns])
    X_test[numerical_columns] = numericas_imputer.transform(X_test[numerical_columns])

    ## Training Model
    algorithm = DecisionTreeClassifier()
    h = algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)

    ## Print Result
    print(f"\n\nRESULT GIVEN BY THE MODEL:\n\n{y_pred}\n\n")


with DAG(
    dag_id="03-train-model",
    description="train model",
    schedule_interval="@once",
    start_date=datetime(2024, 3, 3),
) as dag:

    t1 = PythonOperator(task_id="03-train-ml-model", python_callable=train_model)

    t1
