from pydantic import BaseModel
from enum import Enum

class COVERTYPE(BaseModel):
    var_0: list[int] = [3448]
    var_1: list[int] = [311]
    var_2: list[int] = [25]
    var_3: list[int] = [127]
    var_4: list[int] = [1]
    var_5: list[int] = [1518]
    var_6: list[int] = [146]
    var_7: list[int] = [214]
    var_8: list[int] = [204]
    var_9: list[int] = [1869]
    var_10: list[str] = ["Neota"]
    var_11: list[str] = ["C8772"]


description_API = """
Level 2 for Machine Learning Operations (MLOps) 🚀

# Summary

The purpose of this API is to use the ML model that was trained and saved with MLFlow, MinIO, MySQL and Airflow.

The possible API are:

* Test API.
* Watch all columns from training data and 5 rows of data
* Predict using the saved model
"""

tags_metadata = [
    dict(
        name="Testing API",
        description="Returns a Message to let the user know that the API is working",
    ),
    dict(
        name="Looking at the training data",
        description="Shows the training data stored in MySQL database (just 5 rows)",
    ),
    dict(
        name="Prediction Model",
        description="Shows the solution given by the model trained in MLFlow, MinIO, MySQL and Airflow",
    ),
]