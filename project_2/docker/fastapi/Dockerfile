FROM python:3.9
COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY ./code/fast_api_main.py /fast_api_main.py
CMD ["uvicorn", "fast_api_main:app", "--host", "0.0.0.0", "--port", "8085"]