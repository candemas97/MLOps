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

IP = "localhost"  # "10.43.101.158"  # "localhost"

url_request_api = f"http://{IP}:8085/prediction_cover_type/"

# First Container
with st.container():
    st.title("Welcome to MLOps Project #2! 👋")

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
    st.subheader(f"Cover-Type Prediction")
    col1, col2, col3 = st.columns(3)
    var0 = col1.number_input("Elevation", min_value=-10000, value=3448)
    var1 = col1.number_input("Aspect", min_value=-10000, value=311)
    var2 = col1.number_input("Slope", min_value=-10000, value=25)
    var3 = col1.number_input(
        "Horizontal Distance To Hydrology", min_value=-10000, value=127
    )
    var4 = col2.number_input(
        "Vertical Distance To Hydrology", min_value=-10000, value=1
    )
    var5 = col2.number_input(
        "Horizontal Distance To Roadways", min_value=-10000, value=1518
    )
    var6 = col2.number_input("Hillshade 9am", min_value=-10000, value=146)
    var7 = col2.number_input("Hilldhase Noon", min_value=-10000, value=214)
    var8 = col3.number_input("Hillshade 3pm", min_value=-10000, value=204)
    var9 = col3.number_input(
        "Horizontal Distance To Fire Points", min_value=-10000, value=1869
    )
    var10 = col3.text_input("Widerness Area", value="Neota")
    var11 = col3.text_input("Soil Type", value="C8772")

# Define JSON according to data
datos_json = {
    "var_0": [var0],
    "var_1": [var1],
    "var_2": [var2],
    "var_3": [var3],
    "var_4": [var4],
    "var_5": [var5],
    "var_6": [var6],
    "var_7": [var7],
    "var_8": [var8],
    "var_9": [var9],
    "var_10": [var10],
    "var_11": [var11],
}

with st.container():

    # Center buttom and response
    col0, col1, col2, col3 = st.columns([1, 1, 2, 1])
    with col2:
        # Streamlit Buttom
        if st.button("Predict Cover Type"):
            respuesta = requests.post(url_request_api, json=datos_json)
            # if response it's ok
            if respuesta.status_code == 200:
                respuesta_json = respuesta.json()
                df = pd.DataFrame(respuesta_json, index=["Prediction"])
                st.write(df)
            else:
                st.write("Error: ", respuesta.status_code)
