from fastapi import FastAPI, HTTPException
import pymysql

app = FastAPI()

# Configura los detalles de conexión a la base de datos
DB_HOST = "10.56.1.20"  # Usa la dirección IP del servicio MySQL en tu red Docker
DB_USER = "root"
DB_PASSWORD = "airflow"  # O la que hayas configurado
DB_NAME = "project_2"

@app.get("/")
async def root():
    return {"Project 2": "Hello World!"}

@app.get("/data")
async def fetch_data():
    connection = pymysql.connect(host=DB_HOST,
                                 user=DB_USER,
                                 password=DB_PASSWORD,
                                 db=DB_NAME,
                                 cursorclass=pymysql.cursors.DictCursor)  # Usa DictCursor para obtener los resultados como diccionarios
    try:
        with connection.cursor() as cursor:
            # Asegúrate de que la consulta SQL refleje tu estructura de base de datos y nombre de tabla reales
            cursor.execute("SELECT * FROM project_2.dataset_covertype;")  # Reemplaza `table` con el nombre real de tu tabla
            result = cursor.fetchall()  # fetchall() recupera todos los resultados de la consulta
        return {"data": result}
    except Exception as e:
        # En caso de error, retorna el mensaje de error
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        connection.close()