From python:3.9.6

WORKDIR /usr/src/app

COPY  res34_vgg16_256_bin20_5_2022_3c.h5 .
COPY  player_classification1.h5 .
COPY deepsort_30sec.mp4 .
COPY put_players_into_folders.py .



RUN pip install numpy==1.20.3
RUN pip install opencv-python
RUN pip install scikit-image==0.18.1
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install tensorflow


CMD ["python","./put_players_into_folders.py"]

