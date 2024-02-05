FROM python:3.10

# Set the working directory
WORKDIR /

RUN apt-get update -y
RUN apt-get upgrade -y

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./app /app
COPY ./production_model /production_model
COPY ./data /data

# docker container ls
# docker images

# First build the image 
## docker build -t [image_name] . 
## ASSIGNMENT EXAMPLE: docker build -t assigment_1 .

# Run Code (no matter container name)
## docker run -p [expose_port]:[publish_port] [image_name]  
## ASSIGNMENT EXAMPLE: docker run -p 8989:8000 assigment_1

# Run Code (assigning container name) -> () means optional
## docker run (-d) --name [container_name] -p [expose_port]:[publish_port] [image_name] 
## ASSIGNMENT EXAMPLE: docker run --name container_assignment_1 -p 8989:8000 assigment_1
WORKDIR /app
CMD uvicorn app:app --host 0.0.0.0 --port 8000