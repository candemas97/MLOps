from fastapi import FastAPI
import uvicorn
import prediction
import models_api

app = FastAPI(
    title="MLOps - Assignment 1",
    description=models_api.description_API,
    openapi_tags=models_api.tags_metadata,
)


@app.get("/", tags=["Testing API"])
def index():
    return "MLOps: Assigment 1"


@app.post("/prediction_penguin", tags=["Assignment 1 - Solution"])
async def predict(new_observation: models_api.PENGUIN):
    response = prediction.assign_penguin_specie(new_observation.model_dump(), "xgb")
    return {"penguins": response}


@app.post(
    "/prediction_penguin/select_model_dropdown/{model_to_be_used}",
    tags=["Assignment 1 - Bonus Solution"],
)
async def predict(
    model_to_be_used: models_api.models_to_predict, new_observation: models_api.PENGUIN
):
    if model_to_be_used not in ["xgb", "random_forest"]:
        return {"error": f"No existe {model_to_be_used}. Selecione xgb o random_forest"}
    response = prediction.assign_penguin_specie(
        new_observation.model_dump(), model_to_be_used
    )
    return {"penguins": response}


@app.post(
    "/prediction_penguin/select_model_written/{model_to_be_used}",
    tags=["Assignment 1 - Bonus Solution"],
)
async def predict(model_to_be_used: str, new_observation: models_api.PENGUIN):
    if model_to_be_used not in ["xgb", "random_forest"]:
        return {
            "error_model_not_found": f"The model '{model_to_be_used}' does not exist. The only models that exist and you can write down are 'xgb' or 'random_forest'"
        }
    response = prediction.assign_penguin_specie(
        new_observation.model_dump(), model_to_be_used
    )
    return {"penguins": response}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
