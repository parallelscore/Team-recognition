FROM python:3.8.13-slim

RUN pip install -U pip

WORKDIR /app

COPY [ "../requirements.txt", "./" ]

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

ARG video_url

# COPY ["../data/deepsort_30sec.mp4", "../data/deepsort_30sec.mp4"]

# COPY ["../data/deepsort_30sec.mp4", "../src/video_analytics.py", "../src/yolov5s.pt", "./" ]

COPY . .
EXPOSE 9696

CMD [ "python",  "src/video_analytics.py"]
