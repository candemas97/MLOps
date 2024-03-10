# MLOps - Assignment 3

## How to run this assignment

First of all you need to download this repo with the following command:

`git clone https://github.com/candemas97/MLOps.git`

After that, follow the next steps:

1. Open this repo in **_Visual Studio Code (VSCode)_**.
2. Go to the docker-compose.yaml file
3. Open a new terminal in your current folder
4. Add the following line in your new terminal: `cd assignment_3_f`. This will allow you to be in the current assignment (**assignment 3_f**) because the terminal, by default, takes the main folder as the path.

   > [!NOTE]
   >
   > If you only download the folder **assignment_3** you can skip this step (step 4)

5. Create and run the docker image: `docker-compose up`
6. Wait till all the images load.
7. Go to your browser and (if you are running this in your local machine) got to `localhost:8080`

### Enable MySQL in Airflow

1. Go to Admin >> Connections

![Image 1](https://raw.githubusercontent.com/candemas97/MLOps/main/assignment_3_f/images/pic1.png)

2. Then go to create or a plus (+) buttom.
3. Add the following information and save (_connection password = root_)

![Image 2](https://raw.githubusercontent.com/candemas97/MLOps/main/assignment_3_f/images/pic2.png)

4. Now you are ready to run all the DAG

## Task Objective

It is needed to:

1. Upload CSV data to a MySQL (using Airflow)
2. Delete data from MySQL (using Airflow)
3. Train a Machine Learning Algorithm (using MySQL data) with Airflow
