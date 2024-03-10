# MLOps - Assignment 2

## How to run this assignment

First of all you need to download this repo with the following command:

`git clone https://github.com/candemas97/MLOps.git`

After that, follow the next steps:

1. Open this repo in **_Visual Studio Code (VSCode)_**.
2. Go to the docker-compose.yaml file
3. Open a new terminal in your current folder
4. Add the following line in your new terminal: `cd assignment_3`. This will allow you to be in the current assignment (**_assignment 3_**) because the terminal, by default, takes the main folder as the path.

   > [!NOTE]
   >
   > If you only download the folder **assignment_3** you can skip this step (step 4)

5. Create and run the docker image: `docker-compose up`
6. Wait till all the images load.
7. Go to your browser and (if you are running this in your local machine) got to `localhost:8080`

### Enable MySQL in Airflow

1. Go to Admin >> Connections
2. Then go to create or a plus (+) buttom.
3. Add the following information and save (connection password = root)
4. Now you are ready to run all the DAG

## Task Objective

Transform a linux command into a docker-compose command. Below you can see the Linux command.

```
sudo docker run -it --name tfx --rm -p 8888:8888 -p 6006:6006 -v $PWD:/tfx/src --entrypoint /run_jupyter.sh  tensorflow/tfx:1.12.0
```

In the docker-compose, you will find the solution and the explanation of each step.
