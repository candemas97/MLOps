from pydantic import BaseModel
from enum import Enum

class PARAMETERS(BaseModel):
    brokered_by: list[int] = [10481.0]
    status: list[str] = ["for_sale"]
    bed: list[int] = [3.0]
    bath: list[int] = [2.0]
    acre_lot: list[int] = [1.0]
    street: list[int] = [1612297.0]
    city: list[str] = ["Airville"]
    state: list[str] = ["Pennsylvania"]
    zip_code: list[str] = ["17302.0"]
    house_size: list[str] = ["1792.0"]
    prev_sold_date: list[str] = ["2013-07-12"]


description_API = """
Level 4 for Machine Learning Operations (MLOps) 🚀

# Summary

The purpose of this API is to use the ML model that was trained and saved with MLFlow, MinIO, MySQL and Airflow, Github Actions.

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