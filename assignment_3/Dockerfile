FROM python:3.10

# Set the working directory
WORKDIR /

RUN apt-get update -y
RUN apt-get upgrade -y

COPY dags/data/penguins_lter.csv /penguins_lter.csv
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt