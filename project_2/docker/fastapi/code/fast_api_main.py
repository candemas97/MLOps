from fastapi import FastAPI, HTTPException
import pymysql

app = FastAPI()

# Database setup
DB_HOST = "10.56.1.20"  # MySQL IP Docker Network
DB_USER = "root"
DB_PASSWORD = "airflow"  
DB_NAME = "project_2"

@app.get("/")
async def root():
    return {"Project 2": "Hello World!"}

# Showing data
@app.get("/data")
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                 user=DB_USER,
                                 password=DB_PASSWORD,
                                 db=DB_NAME,
                                 cursorclass=pymysql.cursors.DictCursor)  # DictCursor for results as dictionaries
    try:
        with connection.cursor() as cursor:
            query = "SELECT * FROM project_2.dataset_covertype;"
            cursor.execute(query)
            result = cursor.fetchall()  # fetchall() retrieve all the results
        return {"data": result}
    except Exception as e:
        # if error it shows the reason why
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()