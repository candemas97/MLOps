from pydantic import BaseModel
from enum import Enum


class models_to_predict(str, Enum):
    xgb = "xgb"
    random_forest = "random_forest"


class PENGUIN(BaseModel):
    studyName: list[str] = ["PAL0708", "PAL0708"]
    SampleNumber: list[float] = [1, 2]
    # Species: list[str] = ["Adelie Penguin (Pygoscelis adeliae)","Adelie Penguin (Pygoscelis adeliae)"],
    Region: list[str] = ["Anvers", "Anvers"]
    Island: list[str] = ["Torgersen", "Torgersen"]
    Stage: list[str] = ["Adult, 1 Egg Stage", "Adult, 1 Egg Stage"]
    IndividualID: list[str] = ["N1A1", "N1A2"]
    ClutchCompletion: list[str] = ["Yes", "Yes"]
    DateEgg: list[str] = ["39397", "39397"]
    CulmenLength: list[float] = [39.1, 39.5]
    CulmenDepth: list[float] = [18.7, 17.4]
    FlipperLength: list[float] = [181, 186]
    BodyMass: list[float] = [3750, 3800]
    Sex: list[str] = ["MALE", "FEMALE"]
    Delta15N: list[float] = [1.1, 8.94]
    Delta13C: list[float] = [3.5, 9.87]
    Comments: list[str] = ["Not enough blood for isotopes.", "Hola"]


description_API = """
Level 0 for Machine Learning Operations (MLOps) 🚀

# Summary

The purpose of this API is to identify the penguin specie, there are two ways to classify this:

* Using one single model with the default function.
* Choosing the model you want to use: **XGBoost (_xgb_) or Random Forest (_random_forest_)**
"""

tags_metadata = [
    dict(
        name="Testing API",
        description="Returns a Message to let the user know that the API is working",
    ),
    dict(
        name="Assignment 1 - Solution",
        description="Shows the solution to the assignmnt 1 (classify penguin specie)",
    ),
    dict(
        name="Assignment 1 - Bonus Solution",
        description="Shows the solution to assignmnt 1 bonus (classify penguin specie choosing different Machine Learning algorithms)",
    ),
]
