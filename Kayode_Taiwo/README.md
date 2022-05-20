## How to run

My source code for this project is in the detectron2 folder. Specifically with the detection2/projects/deep_sort folder.

The resulting docker container is hosted online at Docker Hub. To run please do:

$ docker run -it --name pscore geek0075/pscore:thu_may19_v1

This has to process all 751 frames in the provided 30 seconds video. So it runs for a couple of hours. After it's run is over you can connect to it using the Docker Dashboard. Look under containers for pscore. Click on CLI. 

Then you can navigate to the folder ./data/output/blue, ./data/output/yellow, and ./data/output/blue-yellow, on the CLI.  

Please NOTE that the above is a work in progress until the deadline for the test elapses. Thanks.

## Person detection with the detectron2 library

I ran the person detection required to do this project with the Detectron2 package from Facebook: 

https://github.com/facebookresearch/detectron2

Detectron2 gives person detections across each frame of the input video.

## Person tracking with the deep sort algorithm

Next I deployed deep sort Tracking on the detections returned by Detectron2 to associate detections across frames. This is the problem of Multi Object Tracking (MOT) as described and solved in this github repository.

https://github.com/nwojke/deep_sort

## Team classification with the kmeans clustering algorithm

It is the tracks returned by deep_sort that I now classify into teams by detecting each teams colors in the image. The video features two teams - one wears a blue jersey, while the other team wears a yellow jersey.

Classifying into teams (blue or yellow) uses a KMeans clustering algorithm to detect the main colors and see if any is Blue or Yellow and save into appropriate folders. Some detections (images) from detectron2 contain a player from both teams. So KMeans will find both blue and yellow color and such detections are saved in the blue-yellow folder for now.

## Technical notes

The way to use detectron2 is to create your project in the projects folder of a detectron2 repository. This is why you find deep_sort in the projects folder of the detectron2 repository here. That is how detectron2 recommends it be used as a library.