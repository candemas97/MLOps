# **MLOps - Project 1**

## How to run this assignment

### _1. Open Virtual Enviroment_

First of all you need to download this repo with the following command:

```
git clone https://github.com/candemas97/MLOps.git
```

After that, follow the next steps:

1. Open this repo in **_Visual Studio Code (VSCode)_**.
2. Go to the docker-compose.yaml file
3. Open a new terminal in your current folder
4. Add the following line in your new terminal: `cd project_1`. This will allow you to be in the current assignment (**_Project 1_**) because the terminal, by default, takes the main folder as the path.

   > [!NOTE]
   >
   > If you only download the folder **project_1** you can skip this step (step 4)

5. Create and run the docker image: `docker-compose up`
6. Wait till the following message appears

```
To access the server, open this file in a browser:
tfx  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
tfx  |     Or copy and paste one of these URLs:
tfx  |         http://4f70ff082008:8888/lab?token=e81180fa07981f14235d64ac89875200b16f6453a5db3e91
tfx  |      or http://127.0.0.1:8888/lab?token=e81180fa07981f14235d64ac89875200b16f6453a5db3e91
```

7. It is possible that you will have a different URL, nevertheless, copy the first part of the URL in a browser `http://127.0.0.1:8888` and press enter
8. Then, whitin the browser, add the given token that is after `lab?token=` in my case it was `e81180fa07981f14235d64ac89875200b16f6453a5db3e91`
9. Now you are in the enviroment

### _2. How to download the data_

Now that you have fullfilled the previous steps, you will need to download the data to be able to run all the files within this folder.

To be able to do that you need to do the following:

1. Open a new terminal in your current folder
2. Add the following line in your new terminal: `cd project_1`. This will allow you to be in the current assignment (**_Project 1_**) because the terminal, by default, takes the main folder as the path.

   > [!NOTE]
   >
   > If you only download the folder **project_1** you can skip this step (step 2)

3. Install all dependencies

   > [!WARNING]
   >
   > It is suposed that when you run the `docker-compose up`, you should have all the needed dependencies, nevertheless, run this line in case that any error appears

```
pip install -r ./requirements.txt
```

4. You need to import the data in the specific folder, please run the following lines:

```
dvc pull data/covertype/covertype_train.csv
dvc pull data/data_modified/covertype_train_v2.csv
dvc pull data/sv/serving_data.csv
```

5. Review the corresponding folders with the imported data

### _3. Run the code_

1. Go to `core/lab1_solution.ipynb`
2. Run the code to see the results

## Task Objective

The aim of this project was:

- Feature selection
- Dataset ingestion
- Dataset statistics generation
- Creation of a schema based on domain knowledge
- Schema environment creation
- Visualization of dataset anomalies
- Preprocessing, transformation, and feature engineering
- Tracking the lineage of your data flow using ML metadata

## Additional Comments

Within the .ipynb file you can find a step-by-step explanation of the process did.
