FROM python:3.10

# Set the working directory
WORKDIR /

RUN apt-get update -y
RUN apt-get upgrade -y

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt