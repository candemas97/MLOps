FROM python:3.10

# Set the working directory
WORKDIR /

RUN apt-get update -y
RUN apt-get upgrade -y
# RUN apt-get install -y --no-install-recommends python3-pip python3-dev

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./app /app
COPY ./production_model /production_model
COPY ./data /data

# docker container ls
# docker images
# docker build -t assigment_1 .
# docker run -p 8989:8000 assigment_1
WORKDIR /app
CMD uvicorn app:app --host 0.0.0.0 --port 8000