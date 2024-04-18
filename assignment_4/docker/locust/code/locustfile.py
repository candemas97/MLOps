from locust import HttpUser, task, constant
from pydantic import BaseModel

class COVERTYPE(BaseModel):
    var_0: list[int] = [3448]
    var_1: list[int] = [311]
    var_2: list[int] = [25]
    var_3: list[int] = [127]
    var_4: list[int] = [1]
    var_5: list[int] = [1518]
    var_6: list[int] = [146]
    var_7: list[int] = [214]
    var_8: list[int] = [204]
    var_9: list[int] = [1869]
    var_10: list[str] = ["Neota"]
    var_11: list[str] = ["C8772"]

class LoadTest(HttpUser):
    wait_time = constant(1)
    host = "http://inference:8081"

    @task
    def predict(self):
        request_body = {
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
        headers = {
            "Content-Type": "application/json",
        }
        self.client.post(
            "/prediction_cover_type/", json=request_body, headers=headers
        )