FROM python:3.9.7

MAINTAINER Onyekachukwu Okonji

WORKDIR /usr/src/app

RUN apt-get install -y wget
RUN apt-get install libsm6 libxext6  -y


RUN apt-get -y update
RUN apt-get -y upgrade

RUN pip install pixellib
RUN pip install opencv-python

WORKDIR /usr/src/app

COPY pointrend_resnet50.pkl .

COPY parallelscore.py .

COPY deepsort_30sec.mp4 .

EXPOSE 5000

CMD ["python", "./parallelscore.py"]
