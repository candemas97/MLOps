import pandas as pd
import os
import mlflow


def assign_cover_type(dictionary_to_predict: dict) -> int:
    """Predict Cover Type

    Args:
        dictionary_to_predict (dict): Data that was added by the user

    Returns:
        int: Prediction of the model
    """
    X_test = pd.DataFrame(dictionary_to_predict)
    X_test.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = (
        "http://s3:9000"  # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    )
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "supersecret"

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8089")  # "http://mlflow:8087" - for all except VM Javeriana

    model_name = "modelo1"

    # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
    model_production_uri = "models:/{model_name}/production".format(
        model_name=model_name
    )

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)

    example_test = X_test  # .iloc[0].to_frame().T

    return loaded_model.predict(example_test)


if __name__ == "__main__":
    observation = {
        "var_0": [3448],
        "var_1": [311],
        "var_2": [25],
        "var_3": [127],
        "var_4": [1],
        "var_5": [1518],
        "var_6": [146],
        "var_7": [214],
        "var_8": [204],
        "var_9": [1869],
        "var_10": ["Neota"],
        "var_11": ["C8772"],
    }

    print(assign_cover_type(observation))
