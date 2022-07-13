FROM python:latest

LABEL Maintainer="Opeyemi"

WORKDIR /parallel_score

COPY /parallel_score/Team_identification.py /parallel_score/Data/deepsort_30sec.mp4 /parallel_score/requirements.txt ./run_script.sh /parallel_score/yolov5s.pt ./

CMD [ "bash", "./run_script.sh"]