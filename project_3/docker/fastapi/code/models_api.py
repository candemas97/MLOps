from pydantic import BaseModel
from enum import Enum


class READMITTED(BaseModel):
    race: list[str] = ["Caucasian"]
    gender: list[str] = ["Female"]
    age: list[str] = ["High level"]
    admission_type_id: list[str] = ["2"]
    discharge_disposition_id: list[str] = ["1"]
    admission_source_id: list[str] = ["1"]
    time_in_hospital: list[int] = [10]
    num_lab_procedures: list[int] = [36]
    num_procedures: list[int] = [3]
    num_medications: list[int] = [16]
    number_outpatient: list[int] = [0]
    number_emergency: list[int] = [0]
    number_inpatient: list[int] = [1]
    number_diagnoses: list[int] = [5]
    metformin: list[str] = ["No"]
    repaglinide: list[str] = ["No"]
    nateglinide: list[str] = ["No"]
    chlorpropamide: list[str] = ["No"]
    glimepiride: list[str] = ["No"]
    acetohexamide: list[str] = ["No"]
    glipizide: list[str] = ["No"]
    glyburide: list[str] = ["No"]
    tolbutamide: list[str] = ["No"]
    pioglitazone: list[str] = ["No"]
    rosiglitazone: list[str] = ["No"]
    acarbose: list[str] = ["No"]
    miglitol: list[str] = ["No"]
    troglitazone: list[str] = ["No"]
    tolazamide: list[str] = ["No"]
    examide: list[str] = ["No"]
    citoglipton: list[str] = ["No"]
    insulin: list[str] = ["Steady"]
    glyburide_metformin: list[str] = ["No"]
    glipizide_metformin: list[str] = ["No"]
    glimepiride_pioglitazone: list[str] = ["No"]
    metformin_rosiglitazone: list[str] = ["No"]
    metformin_pioglitazone: list[str] = ["No"]
    change: list[str] = ["No"]
    diabetesMed: list[str] = ["Yes"]
    diag_1_group: list[str] = ["Other"]
    # readmitted: list[str] = ["NO"]


description_API = """
Level 3 for Machine Learning Operations (MLOps) 🚀

# Summary

The purpose of this API is to use the ML model that was trained and saved with MLFlow, MinIO, MySQL and Airflow.

All of the above using Kubernetes.

The possible API are:

* Test API.
* Watch all columns from training data and 5 rows of data
* Predict using the saved model and save the prediction and input in a SQL database
"""

tags_metadata = [
    dict(
        name="Testing API",
        description="Returns a Message to let the user know that the API is working",
    ),
    dict(
        name="Looking at the data",
        description="Shows the training, validation and test data stored in MySQL database (just 5 rows)",
    ),
    dict(
        name="Prediction Model",
        description="Shows the solution given by the model trained in MLFlow, MinIO, MySQL and Airflow and store input and prediction in SQL database",
    ),
    dict(
        name="Predictions saved",
        description="Shows input and prediction of historical predictions done by users",
    ),
]
