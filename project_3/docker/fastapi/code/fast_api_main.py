from fastapi import FastAPI, HTTPException
import pymysql

# Own libraries
import models_api
import prediction

app = FastAPI(
    title="MLOps - Project 3",
    description=models_api.description_API,
    openapi_tags=models_api.tags_metadata,
)

# Database setup
DB_HOST = "10.43.101.158"
DB_USER = "root"
DB_PASSWORD = "airflow"
DB_NAME = "project_3"
PORT = 3306

@app.get("/", tags=["Testing API"])
async def root():
    return {"Project 3": "Hello World!"}

# Showing data
@app.get("/data-train-raw", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.train_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Showing data
@app.get("/data-train-clean", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.final_train_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Showing data
@app.get("/data-validation-raw", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.validation_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Showing data
@app.get("/data-validation-clean", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.final_val_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Showing data
@app.get("/data-test-raw", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.test_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Showing data
@app.get("/data-test-clean", tags=["Looking at the data"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.final_test_database LIMIT 5;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()

# Predict using the model
@app.post("/prediction-readmission/", tags=["Prediction Model"])
async def predict(new_observation: models_api.READMITTED):
    response = prediction.will_be_readmitted(new_observation.model_dump())
    # Extract python variables (not numpy)
    if len(response) == 1:
        final_response = response.item()
    else:
        final_response = [response[i].item() for i, _ in enumerate(response)]

    return {"readmission": final_response}

# Showing data
@app.get("/prediction-data", tags=["Predictions saved"])
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                user=DB_USER,
                                password=DB_PASSWORD,
                                db=DB_NAME,
                                port=PORT,
                                cursorclass=pymysql.cursors.DictCursor)  # Using DictCursos to obtain results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_3.input_and_prediction_database;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()