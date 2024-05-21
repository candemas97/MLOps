import pandas as pd
import os
import mlflow
import sqlalchemy
import pymysql

def pre_process(value_to_predict):
    unique_columns_to_use = ["bed", "bath", "acre_lot", "street", "city", "state", "house_size"] # "price"
    value_to_predict = value_to_predict[unique_columns_to_use]
    return value_to_predict

def predict_and_save(dictionary_to_predict: dict) -> int:
    """Predict Cover Type

    Args:
        dictionary_to_predict (dict): Data that was added by the user

    Returns:
        int: Prediction of the model
    """
    X_input = pd.DataFrame(dictionary_to_predict)
    X_prediction = pre_process(X_input)

    # PREDICT

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://s3:8084" # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://mlflow:8087") # "http://0.0.0.0:8087")

    model_name = "model_final_project"

    # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

    print(model_production_uri)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    y_pred = loaded_model.predict(X_prediction)

    y_pred = pd.DataFrame(y_pred, columns=["price"])

    df_upload_predictions = pd.concat([X_input, y_pred], axis=1)

    # SAVE IN MYSQL

    # Parameters
    DB_HOST = "mysql" # "10.43.101.158" # "localhost" "10.43.101.158"  # Using INTERNET!
    DB_USER = "root"
    DB_PASSWORD = "airflow" 
    DB_NAME = "project_4"
    PORT= 3306

    # Connect to MySQL
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')


    # Save data, if exits append into the current table
    df_upload_predictions.to_sql('raw_data', con=engine, if_exists='append', index=False)

    # RETURN RESPONSE TO USER 
    print(y_pred)
    return y_pred["price"]

if __name__ == "__main__":
    observation = {
        "brokered_by": [10481.0],
        "status": ["for_sale"],
        "bed": [3.0],
        "bath": [2.0],
        "acre_lot": [1.0],
        "street": [1612297.0],
        "city": ["Airville"],
        "state": ["Pennsylvania"],
        "zip_code": ["17302.0"],
        "house_size": ["1792.0"],
        "prev_sold_date": ["2013-07-12"],
    }

    print(assign_cover_type(observation))