FROM python:latest

LABEL Maintainer="Opeyemi"

WORKDIR /parallel_score

COPY ./Team_identification.py ./Data/deepsort_30sec.mp4 ./requirements.txt ./run_script.sh ./yolov5s.pt ./

CMD [ "bash", "./run_script.sh"]