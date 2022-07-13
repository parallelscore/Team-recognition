FROM python:3.9.7-slim-buster

RUN apt-get update

RUN pip install -U pip

RUN pip install numpy
RUN apt-get update -y
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt install -y build-essential
RUN pip install tensorflow
RUN pip install keras

WORKDIR ./app

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install pyyaml==5.1
RUN python -m pip install -e detectron2

COPY ["obdec.py","deepsort_30sec.mp4","team_model.h5","./"]



ENTRYPOINT [ "python", "./app/obdec.py"]