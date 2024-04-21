import requests
import pytest

IP = "localhost"  # "10.43.101.158"  # "localhost"

url_request_api = f"http://{IP}:8081/prediction_cover_type/"


def test_apimodel():
    # Define JSON according to data
    datos_json = {
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

    respuesta = requests.post(url_request_api, json=datos_json)
    # if response it's ok
    assert respuesta.status_code == 200


def test_apimodel_failing():
    # Define JSON according to data
    datos_json = {
        "var_0": ["Hola"],
        "var_1": [311],
        "var_2": [25],
        "var_3": [127],
        "var_4": [1],
        "var_5": [1518],
        "var_6": [146],
        "var_7": [214],
        "var_8": ["204"],
        "var_9": ["1869"],
        "var_10": ["Neota"],
        "var_11": ["C8772"],
    }

    respuesta = requests.post(url_request_api, json=datos_json)
    # if response it's ok
    assert not respuesta.status_code == 200


@pytest.mark.parametrize(
    "datos_json",
    [
        (
            {
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
        ),
        (
            {
                "var_0": [3448],
                "var_1": [311],
                "var_2": [50],
                "var_3": [128],
                "var_4": [1],
                "var_5": [1200],
                "var_6": [146],
                "var_7": [214],
                "var_8": [207],
                "var_9": [1000],
                "var_10": ["Neota"],
                "var_11": ["C8772"],
            }
        ),
    ],
)
def test_api_with_parameters(datos_json):
    respuesta = requests.post(url_request_api, json=datos_json)
    # if response it's ok
    assert respuesta.status_code == 200


@pytest.mark.parametrize(
    "datos_json",
    [
        (
            {
                "var_0": ["Hi"],
                "var_1": [311],
                "var_2": [25],
                "var_3": ["Wrong"],
                "var_4": [1],
                "var_5": [1518],
                "var_6": [146],
                "var_7": [214],
                "var_8": [204],
                "var_9": [1869],
                "var_10": ["Neota"],
                "var_11": ["C8772"],
            }
        ),
        (
            {
                "var_0": ["This"],
                "var_1": [311],
                "var_2": [50],
                "var_3": ["Will"],
                "var_4": ["BE"],
                "var_5": ["An"],
                "var_6": [146],
                "var_7": [214],
                "var_8": ["Error"],
                "var_9": [1000],
                "var_10": [2],
                "var_11": ["C8772"],
            }
        ),
    ],
)
def test_api_with_parameters_failing(datos_json):
    respuesta = requests.post(url_request_api, json=datos_json)
    # if response it's ok
    assert not respuesta.status_code == 200
