FROM python:3.9.13-slim-bullseye

MAINTAINER Onyekachukwu Okonji

WORKDIR /app

RUN apt-get install -y wget
RUN apt-get install libsm6 libxext6  -y


RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install pixellib
RUN pip install opencv-python

COPY pointrend_resnet50.pkl .

COPY parallelscore.py .

COPY deepsort_30sec.mp4 .

CMD ["python", "./parallelscore.py"]
