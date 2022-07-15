## Problem Statement
	Build a Docker file that takes the video from the link above and extracts (images of) players in unique teams across each frame in the video and saves images in different folders for each team. <br>

## Solution/Approach:

[![Player classification approach](video_analytics_approach.jpg "Approach")](https://i.ibb.co/5Whw66Q/image.png) <br>

* **Split** the video into different frames for easy analysis. The default frame of the image is about 4 frames per second. But you might specify the second/per split to control the number of frames in the 30 seconds video
* **Extract person/human/player** in each image frame using pretrained **object detection** model from yolov5s. Using yolov5s is because it’s the current state of the art with light-weight resource and good accuracy.<br>

* How do you differentiate each player?

** Extract the **image representation** from each person detection using **resnet18** classification model. This runs the detected object on a model with the output of an hidden layer of vector size 512. This will try to detect colour in a player image.
** Run the image representation on a simple **K-means** clustering algorithm. This will try to demarcate the image based on color which the feature extraction from the image have strong correlation.
** The clustering will try to segment Team A, Team B and other humans(cameraman, referees, e.t.c) and save in different folder. The output is found in data>output>team <br>

## How to run?

### Locally:
The script `src>video_analytics.py`. Takes in url from the drive as an optional input, as well as path to save the detected team and the detected humans in the video.

> python video_analytics [Optional --video_url --team_dir_path --detected_object_path]

### Docker 
* Implement this dock script to build. 

>  docker build -t ride-duration-prediction-services .

* Then this try this command to run the out put. 

> docker run -it --rm -p  9696:9696 ride-duration-prediction-services


On the repo out of the image is shown bellow.

## Improvement:
* Annotating each team for better accuracy and traine with a pretrained model, but this will fail when inputting a new video with different team. This is the reason clustering was used in this work to handle issue.
* Use a higher classification mode like resnet150 for feature extraction, but there will be a trade-off for speed.
