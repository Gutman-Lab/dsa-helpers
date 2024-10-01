FROM python:3.11-slim-buster

ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip

RUN python -m pip install pymongo==4.9.1

RUN pip install --no-cache-dir -r requirements.txt

COPY dsa_helpers-0.4.dev1-py3-none-any.whl /app
RUN pip install dsa_helpers-0.4.dev1-py3-none-any.whl
