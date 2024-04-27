import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import pandas as pd


def load_lottieurl(url):
    r = requests.request("GET", url)
    if r.status_code != 200:
        return None
    return r.json()


header_image_lottie = load_lottieurl(
    "https://lottie.host/3ebc7ddd-12ec-4488-92e7-b5f40fd9187a/rpDKFcyZLf.json"
)

image_airflow = Image.open("./images/airflow.png")
image_minio = Image.open("./images/minio.png")
image_mlflow = Image.open("./images/mlflow.png")
image_fastapi = Image.open("./images/fastapi.png")

IP = "10.43.101.158"  # "10.43.101.158"  # "localhost"

url_request_api = f"http://{IP}:8085/prediction-readmission/"

# First Container
with st.container():
    st.title("Welcome to MLOps Project #3! 🚀")

# Second Container
with st.container():
    left_column, right_column = st.columns((1, 2))
    with left_column:
        st.write(
            """
            This interface is responsible for user interaction, with various functionalities such as:
            - Hyperlinks that will lead the user to: FastAPI, MLFlow, MinIO, and Airflow.
            - Module to request and allow input of information to perform inference and ultimately present prediction results.
"""
        )

    with right_column:
        st_lottie(header_image_lottie, height=300, key="coding")

with st.container():
    st.write(f"---")
    st.subheader(f"Hyperlinks to navegate through the different pages")
    st.write(
        f"Please click on the hyperlink below the image to go the webpage you want to see"
    )
    f1c1, f1c2, f1c3, f1c4 = st.columns((1, 1, 1, 1))
    with f1c1:
        st.image(image_airflow)
    with f1c2:
        st.image(image_minio)
    with f1c3:
        st.image(image_mlflow)
    with f1c4:
        st.image(image_fastapi)

    f2c1, f2c2, f2c3, f2c4 = st.columns((1, 1, 1, 1))
    with f2c1:
        url = f"http://{IP}:8080"
        # texto = f"[Airflow]({url})"
        text = "Airfow"
        st.markdown(
            f"<p style='text-align: center; font-size: 16px;'><a href='{url}'>{text}</a></p>",
            unsafe_allow_html=True,
        )
    with f2c2:
        url = f"http://{IP}:8083"
        # texto = f"[Minio]({url})"
        text = "MinIO"
        st.markdown(
            f"<p style='text-align: center; font-size: 16px;'><a href='{url}'>{text}</a></p>",
            unsafe_allow_html=True,
        )
    with f2c3:
        url = f"http://{IP}:8087"
        # texto = f"[MLFlow]({url})"
        text = "MLFlow"
        st.markdown(
            f"<p style='text-align: center; font-size: 16px;'><a href='{url}'>{text}</a></p>",
            unsafe_allow_html=True,
        )
    with f2c4:
        url = f"http://{IP}:8085/docs"
        # texto = f"[FastAPI]({url})"
        text = "FastAPI"
        st.markdown(
            f"<p style='text-align: center; font-size: 16px;'><a href='{url}'>{text}</a></p>",
            unsafe_allow_html=True,
        )

with st.container():
    st.write(f"---")
    st.subheader(f"Readmission Prediction")
    col1, col2, col3 = st.columns(3)
    var00 = col1.text_input("race", value="Caucasian")
    var01 = col1.text_input("gender", value="Female")
    var02 = col1.text_input("age", value="High level")
    var03 = col1.text_input("admission_type_id", value=2)
    var04 = col1.text_input("discharge_disposition_id", value=1)
    var05 = col1.text_input("admission_source_id", value=1)
    var06 = col1.number_input("time_in_hospital", min_value=-10000, value=10)
    var07 = col1.number_input("num_lab_procedures", min_value=-10000, value=36)
    var08 = col1.number_input("num_procedures", min_value=-10000, value=3)
    var09 = col1.number_input("num_medications", min_value=-10000, value=16)
    var10 = col1.number_input("number_outpatient", min_value=-10000, value=0)
    var11 = col1.number_input("number_emergency", min_value=-10000, value=0)
    var12 = col1.number_input("number_inpatient", min_value=-10000, value=1)
    var13 = col1.number_input("number_diagnoses", min_value=-10000, value=5)
    var14 = col2.text_input("metformin", value="No")
    var15 = col2.text_input("repaglinide", value="No")
    var16 = col2.text_input("nateglinide", value="No")
    var17 = col2.text_input("chlorpropamide", value="No")
    var18 = col2.text_input("glimepiride", value="No")
    var19 = col2.text_input("acetohexamide", value="No")
    var20 = col2.text_input("glipizide", value="No")
    var21 = col2.text_input("glyburide", value="No")
    var22 = col2.text_input("tolbutamide", value="No")
    var23 = col2.text_input("pioglitazone", value="No")
    var24 = col2.text_input("rosiglitazone", value="No")
    var25 = col2.text_input("acarbose", value="No")
    var26 = col2.text_input("miglitol", value="No")
    var27 = col2.text_input("troglitazone", value="No")
    var28 = col3.text_input("tolazamide", value="No")
    var29 = col3.text_input("examide", value="No")
    var30 = col3.text_input("citoglipton", value="No")
    var31 = col3.text_input("insulin", value="Steady")
    var32 = col3.text_input("glyburide-metformin", value="No")
    var33 = col3.text_input("glipizide-metformin", value="No")
    var34 = col3.text_input("glimepiride-pioglitazone", value="No")
    var35 = col3.text_input("metformin-rosiglitazone", value="No")
    var36 = col3.text_input("metformin-pioglitazone", value="No")
    var37 = col3.text_input("change", value="No")
    var38 = col3.text_input("diabetesMed", value="Yes")
    var39 = col3.text_input("diag_1_group", value="Other")
    # var40 = col3.text_input("readmitted", value="NO")

# Define JSON according to data
datos_json = {
    "race": [var00],
    "gender": [var01],
    "age": [var02],
    "admission_type_id": [var03],
    "discharge_disposition_id": [var04],
    "admission_source_id": [var05],
    "time_in_hospital": [var06],
    "num_lab_procedures": [var07],
    "num_procedures": [var08],
    "num_medications": [var09],
    "number_outpatient": [var10],
    "number_emergency": [var11],
    "number_inpatient": [var12],
    "number_diagnoses": [var13],
    "metformin": [var14],
    "repaglinide": [var15],
    "nateglinide": [var16],
    "chlorpropamide": [var17],
    "glimepiride": [var18],
    "acetohexamide": [var19],
    "glipizide": [var20],
    "glyburide": [var21],
    "tolbutamide": [var22],
    "pioglitazone": [var23],
    "rosiglitazone": [var24],
    "acarbose": [var25],
    "miglitol": [var26],
    "troglitazone": [var27],
    "tolazamide": [var28],
    "examide": [var29],
    "citoglipton": [var30],
    "insulin": [var31],
    "glyburide_metformin": [var32],
    "glipizide_metformin": [var33],
    "glimepiride_pioglitazone": [var34],
    "metformin_rosiglitazone": [var35],
    "metformin_pioglitazone": [var36],
    "change": [var37],
    "diabetesMed": [var38],
    "diag_1_group": [var39],
    # "readmitted": [var40]
}


with st.container():

    # Center buttom and response
    col0, col1, col2, col3 = st.columns([1, 1, 2, 1])
    with col2:
        # Streamlit Buttom
        if st.button("Readmission Prediction"):
            respuesta = requests.post(url_request_api, json=datos_json)
            # if response it's ok
            if respuesta.status_code == 200:
                respuesta_json = respuesta.json()
                df = pd.DataFrame(respuesta_json, index=["Prediction"])
                st.write(df)
            else:
                st.write("Error: ", respuesta.status_code)
