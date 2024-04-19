from fastapi import FastAPI, HTTPException
import pymysql

# Own libraries
import models_api
import prediction

app = FastAPI(
    title="MLOps - Project 2",
    description=models_api.description_API,
    openapi_tags=models_api.tags_metadata,
)

# Database setup
DB_HOST = "10.56.1.20"  # MySQL IP Docker Network
DB_USER = "root"
DB_PASSWORD = "airflow"  
DB_NAME = "project_2"

@app.get("/", tags=["Testing API"])
async def root():
    return {"Project 2": "Hello World!"}

# Showing data
@app.get("/data", tags=["Looking at the training data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                 user=DB_USER,
                                 password=DB_PASSWORD,
                                 db=DB_NAME,
                                 cursorclass=pymysql.cursors.DictCursor)  # DictCursor for results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_2.dataset_covertype LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Predict using the model
@app.post("/prediction_cover_type/", tags=["Prediction Model"])
async def predict(new_observation: models_api.COVERTYPE):
    response = prediction.assign_cover_type(new_observation.model_dump())
    # Extract python variables (not numpy)
    if len(response) == 1:
        final_response = response.item()
    else:
        final_response = [response[i].item() for i, _ in enumerate(response)]

    return {"cover_type": final_response}