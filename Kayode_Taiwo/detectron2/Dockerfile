FROM python:3.7-slim-buster

RUN apt-get update -y
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install opencv-python
RUN pip3 install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN apt install -y build-essential

WORKDIR /usr/src/app

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

WORKDIR /usr/src/app/detectron2/projects

RUN git clone https://github.com/nwojke/deep_sort.git

WORKDIR /usr/src/app/detectron2/projects/deep_sort

COPY ./projects/deep_sort/requirements.txt ./
COPY ./projects/deep_sort/deep_sort/linear_assignment_.py ./deep_sort
COPY ./projects/deep_sort/deep_sort/linear_assignment.py ./deep_sort
COPY ./projects/deep_sort/predictor.py ./
COPY ./projects/deep_sort/test.py ./
RUN mkdir -p ./resources/networks
RUN mkdir -p ./data/input
RUN mkdir -p ./data/output
RUN mkdir -p ./data/output/blue
RUN mkdir -p ./data/output/yellow
RUN mkdir -p ./data/output/masks
RUN mkdir -p ./data/output/blue-yellow

COPY ./projects/deep_sort/resources/networks/mars-small128.pb ./resources/networks/
COPY ./projects/deep_sort/data/input/deepsort_30sec.mp4 ./data/input/

RUN pip3 install --no-cache-dir -r requirements.txt

# run detectron2 under user "appuser":
# RUN wget https://drive.google.com/file/d/1en2OKkAyv8nwkBXcCQ6pv9zIjyG3eMlO/view?usp=sharing -O ./resources/networks/mars-small128.pb
# RUN wget https://drive.google.com/file/d/1bB66hP9voDXuoBoaCcKYY7a8IYzMMs4P/view?usp=sharing -O ./resources/networks/mars-small128.pb
# RUN wget https://drive.google.com/file/d/1fqp2btzDh-_gIKe3aRodcvtfwPY5fOhO/view?usp=sharing -O ./data/input/deepsort_30sec.mp4

# CMD [ "python", "./test.py" ]
ENTRYPOINT [ "python", "./test.py" ]