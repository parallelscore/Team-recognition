# Ezinwanne Aka's Submittion 

This submition contains the dockerfile as well as the python script and image classification model which I built

The '''obdec.py''' is made up of basically 3 block of codes


## Block 1
This is where the video is ingested and its component image frames is extracted into the images folder. I built the code in such a way that the you can specify the steps of frames you want to extract, i.e you might not want every frame but my need every other 5 frames, in that case you can make the number of steps to be 5 and hence extract frame 1, frame 6, frame 11 ... and so on.

## Block 2
This block uses the pretrained detectron model to identify every person in each frame. With some additional twick, I was able to extract every part of the image where a human being was identified. This is then stored in the pic2 folder.

## Block 3
This block uses a model '''team_model.h5'''  which I trained with the images extracted from the previous block to classify the images into 3 classes. These are team1 (team in blue), team2(team in yellow), others(refreee, cameramen, every non player person). Then every classified player for either of the teams in put 

NB: running this script with a GPU in a collab or kaggle notebook after installing the detectron2 and pyaml dependecies takes approximately 20 mins to complete

The cummulation of the project is in the docker file which includes the environment and dependencies needed for this code to run effectively.
