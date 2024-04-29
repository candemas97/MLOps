import pandas as pd
import os
import mlflow
import sqlalchemy

def save_user_input_and_model_prediction(input_: pd.DataFrame, y_pred) -> None:
    """Upload data that was used to predict to SQL

    Args:
        input_ (pd.DataFrame): Data that was added by the user
        y_pred: Model prediction according to user input
    """

    y_pred = pd.DataFrame(y_pred, columns=["prediction_readmitted"])
    df_upload_predictions = pd.concat([input_, y_pred], axis=1)
    # Connect to MySQL
    DB_HOST = "10.43.101.158"
    DB_USER = "root"
    DB_PASSWORD = "airflow"
    DB_NAME = "project_3"
    PORT = 3306
    engine = sqlalchemy.create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{PORT}/{DB_NAME}')

    # Save data, if exits append into the current table
    df_upload_predictions.to_sql('input_and_prediction_database', con=engine, if_exists='append', index=False)

def will_be_readmitted(dictionary_to_predict: dict) -> int:
    """Predict Cover Type

    Args:
        dictionary_to_predict (dict): Data that was added by the user

    Returns:
        int: Prediction of the model
    """
    X_test = pd.DataFrame(dictionary_to_predict)
    X_test.columns = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'diag_1_group']

    # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "8084"
    # WE ARE ALSO TAKING THE NETWORK VALUE NEVERTHELESS YOU CAN USE THE CONTEINER NAME (IN OUR CASE S3)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.158:8084" # YOU MUST TAKE THE API NOT THE WEBAPP IN MY CASE IT WAS "http://0.0.0.0:8083" BUT API "8084"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'

    # connect to mlflow
    mlflow.set_tracking_uri("http://10.43.101.158:8087") # "http://0.0.0.0:8087")

    model_name = "modelo1"

    # logged_model = 'runs:/71428bebed2b4feb9635714ea3cdb562/model'
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)

    y_pred = loaded_model.predict(X_test)

    save_user_input_and_model_prediction(X_test, y_pred)

    return y_pred

if __name__ == "__main__":
    observation = {
    "race": ["Caucasian"],
    "gender": ["Female"],
    "age": ["High level"],
    "admission_type_id": [2],
    "discharge_disposition_id": [1],
    "admission_source_id": [1],
    "time_in_hospital": [10.0],
    "num_lab_procedures": [36.0],
    "num_procedures": [3.0],
    "num_medications": [16.0],
    "number_outpatient": [0.0],
    "number_emergency": [0.0],
    "number_inpatient": [1.0],
    "number_diagnoses": [5.0],
    "metformin": ["No"],
    "repaglinide": ["No"],
    "nateglinide": ["No"],
    "chlorpropamide": ["No"],
    "glimepiride": ["No"],
    "acetohexamide": ["No"],
    "glipizide": ["No"],
    "glyburide": ["No"],
    "tolbutamide": ["No"],
    "pioglitazone": ["No"],
    "rosiglitazone": ["No"],
    "acarbose": ["No"],
    "miglitol": ["No"],
    "troglitazone": ["No"],
    "tolazamide": ["No"],
    "examide": ["No"],
    "citoglipton": ["No"],
    "insulin": ["Steady"],
    "glyburide_metformin": ["No"],
    "glipizide_metformin": ["No"],
    "glimepiride_pioglitazone": ["No"],
    "metformin_rosiglitazone": ["No"],
    "metformin_pioglitazone": ["No"],
    "change": ["No"],
    "diabetesMed": ["Yes"],
    "diag_1_group": ["Other"],
    "readmitted": ["NO"]
}


    print(will_be_readmitted(observation))