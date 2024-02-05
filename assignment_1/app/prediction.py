# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

## Import model from saved files
import joblib


def train_data() -> pd.DataFrame:
    """Having training data to data_preparation

    Returns:
        pd.DataFrame: X_train as it was trained
    """
    data = pd.read_csv(
        "../data/penguins_lter.csv"
    )  # "../data/penguins_lter.csv"
    data.columns = data.columns.str.lower()
    # Division between y and the rest of variables

    y = data["species"].map(
        {
            "Adelie Penguin (Pygoscelis adeliae)": 0,
            "Chinstrap penguin (Pygoscelis antarctica)": 1,
            "Gentoo penguin (Pygoscelis papua)": 2,
        }
    )
    X = data.drop(columns="species")

    # Split train and test (80% train, 20% test)
    X_train, _, _, _ = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train


def data_preparation(X_test: pd.DataFrame) -> pd.DataFrame:
    """Data Preparation to be able to predict with the trained models

    Args:
        X_test (pd.DataFrame): Variable to be predicted added by the user.

    Returns:
        pd.DataFrame: Clean data (data prepared) to be predicted
    """

    # Read Train data to do data preparation
    X_train = train_data()

    # Same column name
    X_test.columns = X_train.columns

    # Null treatment

    ## We look every column type
    categorical_columns = X_train.select_dtypes(exclude=[int, float]).columns
    numerical_columns = X_train.select_dtypes(include=[int, float]).columns

    ## Impute variables
    categoricas_imputer = SimpleImputer(missing_values=np.nan, strategy="constant")
    numericas_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

    ## Training Imputers
    categoricas_imputer.fit(X_train[categorical_columns])
    numericas_imputer.fit(X_train[numerical_columns])

    ## Impute variables
    X_train[categorical_columns] = categoricas_imputer.transform(
        X_train[categorical_columns]
    )
    X_test[categorical_columns] = categoricas_imputer.transform(
        X_test[categorical_columns]
    )

    X_train[numerical_columns] = numericas_imputer.transform(X_train[numerical_columns])
    X_test[numerical_columns] = numericas_imputer.transform(X_test[numerical_columns])

    # Dummies

    ## Creating dummies
    categorical_columns = X_train.select_dtypes(exclude=[int, float]).columns
    numerical_columns = X_train.select_dtypes(include=[int, float]).columns

    X_train = pd.get_dummies(
        X_train, columns=categorical_columns, drop_first=True, dtype=float
    )
    X_test = pd.get_dummies(
        X_test, columns=categorical_columns, drop_first=True, dtype=float
    )

    ## Aligning
    X_train, X_test = X_train.align(X_test, fill_value=0, axis=1, join="left")

    # Data Standarization
    columns = X_train.columns
    scaler = StandardScaler()

    scaler.fit(X_train)  # Se realiza el fit con la data de entrenamiento
    X_train.values[:] = scaler.transform(X_train)
    X_test.values[:] = scaler.transform(X_test)

    return X_test


def predict_model(model_to_be_used: int, X_test: pd.DataFrame) -> list[str]:
    """Prediction of the model with a dataset

    Args:
        model_to_be_used (int): model that is needed to predict
        X_test (pd.DataFrame): Data that will alow us to predict

    Returns:
        list[str]: predition of the model
    """

    if model_to_be_used == "random_forest":
        model = joblib.load(
            "../production_model/model_random_forest.joblib"
        )  # Aquí prodría ser sólo "model_random_forest.joblib"

    if model_to_be_used == "xgb":
        model = joblib.load(
            "../production_model/model_xgb.joblib"
        )  # Aquí prodría ser sólo "model_xgb.joblib"

    resul = model.predict(X_test)

    resul = (
        pd.DataFrame(resul)[0]
        .map(
            {
                0: "Adelie Penguin (Pygoscelis adeliae)",
                1: "Chinstrap penguin (Pygoscelis antarctica)",
                2: "Gentoo penguin (Pygoscelis papua)",
            }
        )
        .to_list()
    )

    return resul


def assign_penguin_specie(dictionary_to_predict: dict, model_to_be_used: str) -> list:
    """AI is creating summary for assign_penguin_specie

    Args:
        dictionary_to_predict (dict): Data that was added by the user

    Returns:
        list: Prediction of the model
    """

    X_test = pd.DataFrame(dictionary_to_predict)
    # X_test = X_test.drop(columns="model")
    X_test = data_preparation(X_test)

    resul = predict_model(model_to_be_used, X_test)

    return resul


if __name__ == "__main__":
    observation = {
        "culmenLen": [31.2, 12.3],
        "culmenDepth": [1.2, 3.4],
        "flipperLen": [1, 2],
        "bodyMass": [31, 12],
        "sex": ["MALE", "MALE"],
        "delta15N": [1.2, 2.45],
        "delta13C": [1, 34.5],
    }

    model_to_be_used = "xgb"
    sol = assign_penguin_specie(observation, model_to_be_used)
    print(f"\nWhat returns this code is an array (below see the result):\n\n{sol}")
