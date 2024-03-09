## Importación
import numpy as np
import pandas as pd
# import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# def read_csv_train_model() -> pd.DataFrame:
#     # Where we currently are, usually we are at -> /opt/***/
#     root_airflow = os.getcwd()
#     # Join path to access to the needed csv file
#     csv_path = f"{root_airflow}/dags/data/penguins_lter.csv"
#     # Read CSV file
#     df = pd.read_csv(csv_path)

#     return df

def train_model():
    ## Lectura
    root_airflow = os.getcwd()
    # Join path to access to the needed csv file
    csv_path = f"{root_airflow}/dags/data/penguins_lter.csv"
    # Read CSV file
    data = pd.read_csv(csv_path)

    ## Transformación
    data["Sex"] = data["Sex"].replace({".":None,np.nan:None})
    data_x = pd.DataFrame()
    for col in ["Island", "Clutch Completion","Sex"]:
        idxs_none = data[data[col].isna()].index.tolist()
        dummies = pd.get_dummies(data[col],drop_first=True,prefix=col,dtype=int)
        dummies.loc[idxs_none,:] = None
        data_x = pd.concat([data_x,dummies],axis=1)
    for col in ['Culmen Length (mm)','Culmen Depth (mm)', 'Flipper Length (mm)', 
                'Body Mass (g)', 'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']:
        data_x = pd.concat([data_x,data[col]],axis=1)
    data_y = pd.factorize(data["Species"])[0]

    ## Preparación
    X, y = data_x.values, data_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ## Entrenamiento
    algorithm = DecisionTreeClassifier()
    h = algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test) 
    print(y_pred)

with DAG(
    dag_id="03-train-model",
    description="train model",
    schedule_interval="@once",
    start_date=datetime(2024, 3, 3),
) as dag:

    t1 = PythonOperator(task_id="train-model", python_callable=train_model)

    t1




