from fastapi import FastAPI
import uvicorn
import prediction
import models_api

app = FastAPI()


@app.get("/")
def index():
    return "MLOps: Assigment 1"


@app.post("/prediction_penguin")
async def predict(new_observation: models_api.PENGUIN):
    response = prediction.assign_penguin_specie(new_observation.model_dump(), "xgb")
    return {"penguins": response}


@app.post("/prediction_penguin/select_model/{model_to_be_used}")
async def predict(model_to_be_used: str, new_observation: models_api.PENGUIN):
    if model_to_be_used not in ["xgb", "random_forest"]:
        return {"error": f"No existe {model_to_be_used}. Selecione xgb o random_forest"}
    response = prediction.assign_penguin_specie(
        new_observation.model_dump(), model_to_be_used
    )
    return {"penguins": response}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
