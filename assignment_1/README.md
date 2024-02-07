# MLOps - Assignment 1

## How to run this assignment

First of all you need to download this repo with the following command:

`git clone https://github.com/candemas97/MLOps.git`

After that, follow the next steps:

1. Open this repo in **_Visual Studio Code (VSCode)_**.
2. Go to the Dockerfile
3. Open a new terminal in your current folder
4. Add the following line in your new terminal: `cd assignment_1`. This will allow you to be in the current assignment (**_assignment 1_**) because the terminal, by default, takes the main folder as the path.
   > [!NOTE]
   >
   > If you only download the folder **assignment_1** you can skip this step (step 4)
5. Create a docker image: `docker build -t assigment_1 .`
6. Run the container: `docker run --name container_assignment_1 -p 8989:8000 assigment_1`
7. Go to `http://localhost:8989/docs/`

## How to interact with the FastAPI interphase

In the following table you can find the description of each API:
| API | Description |
| :---: | :---: |
|`/` | Shows a message on the screen and shows if the app is working or not |
| `/prediction_penguin` | Predict the penguin specie. You just need to execute the code |
| `/prediction_penguin/select_model_dropdown/{model_to_be_used}` | Predict the penguin specie. First you need to select from a dropdown what model to use before you execute the code |
| `/prediction_penguin/select_model_written/{model_to_be_used}` | Predict the penguin specie. First you need to write down what model to use before you execute the code. There are only two models (**_xgb_** or **_random_forest_**) in case you write another model, the API will remember you that you can only write down two models |
