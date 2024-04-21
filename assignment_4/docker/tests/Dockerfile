FROM python:3.9
COPY ./requirements.txt /requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt
COPY ./code/test_main.py test_main.py
CMD ["pytest", "-v", "test_main.py"]