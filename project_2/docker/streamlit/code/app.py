import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image


def load_lottieurl(url):
    r = requests.request("GET", url)
    if r.status_code != 200:
        return None
    return r.json()


header_image_lottie = load_lottieurl(
    "https://lottie.host/3ebc7ddd-12ec-4488-92e7-b5f40fd9187a/rpDKFcyZLf.json"
)
IP = "localhost"

# First Container
with st.container():
    st.title("Welcome to MLOps Project #2! 👋")

# Second Container
with st.container():
    left_column, right_column = st.columns((1, 2))
    with left_column:
        st.write("Text Here")
        st.write(f"[Airflow](http://{IP}:8080)")
    with right_column:
        st_lottie(header_image_lottie, height=300, key="coding")
