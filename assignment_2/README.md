# MLOps - Assignment 2

## How to run this assignment

First of all you need to download this repo with the following command:

`git clone https://github.com/candemas97/MLOps.git`

After that, follow the next steps:

1. Open this repo in **_Visual Studio Code (VSCode)_**.
2. Go to the docker-compose.yaml file
3. Open a new terminal in your current folder
4. Add the following line in your new terminal: `cd assignment_2`. This will allow you to be in the current assignment (**_assignment 2_**) because the terminal, by default, takes the main folder as the path.
   
   > [!NOTE]
   >
   > If you only download the folder **assignment_2** you can skip this step (step 4)

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
9. Run the .ipynb files

## Task Objective

Transform a linux command into a docker-compose command. Below you can see the Linux command.

```
sudo docker run -it --name tfx --rm -p 8888:8888 -p 6006:6006 -v $PWD:/tfx/src --entrypoint /run_jupyter.sh  tensorflow/tfx:1.12.0
```
In the docker-compose, you will find the solution and the explanation of each step.

## Run files within the graph terminal

If you want to download new jupyter notebooks that are in github, you should use the following commnad: `wget` and later go to the file in Github and press `raw`. Finally, copy and paste the link that is shown. Below you can see an example.

```
wget https://raw.githubusercontent.com/CristianDiazAlvarez/MLOPS_PUJ/main/Niveles/1/Validacion_de_datos/TF/TFDV.ipynb
```
Finally, run your code.
