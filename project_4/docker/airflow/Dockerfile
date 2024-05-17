# It's mandatory to use this version to have the same version and libraries as jupyter if we do not use this version MLFlow will have issues related to
# model transcript because Python version must be the same as MLFlow version for this python.
FROM apache/airflow:2.6.0-python3.9

# Change as root user to be able to download needed packages (git)
USER root

# Install git and clean to reduce image size
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Changing to airflow user to do the next steps
USER airflow

# Install additional dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
# RUN pip install mlflow==2.3.0 pycaret scikit-learn xgboost scipy boto3 requests pymysql mysqlclient
# RUN pip install mlflow==2.3.0
# RUN pip install pycaret==3.0.0
# RUN pip install scikit-learn==1.2.2
# RUN pip install xgboost==2.0.3
# RUN pip install scipy==1.11.4
# RUN pip install boto3==1.26.121
# RUN pip install requests==2.28.2
# RUN pip install pymysql==1.1.0
# RUN pip install mysqlclient