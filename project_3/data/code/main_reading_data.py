from fastapi import FastAPI, HTTPException
from typing import Optional
import random
import json
import time
import csv
import os

MIN_UPDATE_TIME = 10  # Time in seconds
GROUP_NUMBER = "9"  # Group to use


app = FastAPI()

# Reading Data

## Train
data = []
with open("Diabetes/diabetes_train.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        data.append(row)

## Validation
data_val = []
with open("Diabetes/diabetes_val.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        data_val.append(row)

## Test
data_test = []
with open("Diabetes/diabetes_test.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        data_test.append(row)

# Batch size for training data
batch_size = len(data) // 6


# Function for random batch
def get_batch_data(batch_number: int, batch_size: int = batch_size):
    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    # Obtain data sorted randomly
    random_data = random.sample(data[start_index:end_index], batch_size)
    return random_data

if os.path.isdir(f"/Diabetes/timestamp") == False:
    os.mkdir(f"/Diabetes/timestamp")


if os.path.isfile("/Diabetes/timestamp/timestamps.json"):
    # Load JSON data in the dictionary
    with open("/Diabetes/timestamp/timestamps.json", "r") as f:
        timestamps = json.load(f)
else:
    # Create dictionary to be loaded in the file
    timestamps = {GROUP_NUMBER: [0, -1]}


@app.get("/")
async def root():
    return {"Project 3": "API for data batch extraction."}


# Get data in batches
@app.get("/data-train-batches")
async def read_data():
    global timestamps

    # All the dataset has been read
    if timestamps[GROUP_NUMBER][1] >= 6:
        raise HTTPException(
            status_code=400,
            detail="All the dataset has been read, if you want more data, start again.",
        )

    current_time = time.time()
    last_update_time = timestamps[GROUP_NUMBER][0]

    # Verify time needed to extract again
    if current_time - last_update_time > MIN_UPDATE_TIME:
        # Update timestamp and get new batch
        timestamps[GROUP_NUMBER][0] = current_time
        timestamps[GROUP_NUMBER][1] += 1

    # Take the data of the current batch
    random_data = get_batch_data(timestamps[GROUP_NUMBER][1])

    # Save time and batch number
    with open("/Diabetes/timestamp/timestamps.json", "w") as file:
        file.write(json.dumps(timestamps))

    return {
        "group_number": GROUP_NUMBER,
        "batch_number": timestamps[GROUP_NUMBER][1] + 1,
        "data": random_data,
    }


# Get data in batches
@app.get("/data-validation")
async def read_data_validation():
    return {"data": data_val}


# Get data in batches
@app.get("/data-test")
async def read_data_test():
    return {"data": data_test}


@app.get("/restart-data-train-generation")
async def restart_data():
    # Reset values as original
    timestamps[GROUP_NUMBER][0] = 0
    timestamps[GROUP_NUMBER][1] = -1
    # Write values in the JSON file
    with open("/Diabetes/timestamp/timestamps.json", "w") as file:
        file.write(json.dumps(timestamps))

    return {"ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=80)
