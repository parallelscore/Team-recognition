## How to run

My source code for this project is in the detectron2 folder. Specifically with the detection2/projects/deep_sort folder.

The resulting docker container is hosted online at Docker Hub. To run please do:

$ docker run -it --name pscore geek0075/pscore:sun_may22_v1

This has to process all 751 frames in the provided 30 seconds video. So it runs for a couple of hours. After it's run is over you can connect to it using the Docker Dashboard. Look under containers for pscore. Click on CLI. 

Then you can navigate to the folder ./data/output/blue, ./data/output/yellow, and ./data/output/blue-yellow, on the CLI. 

or with the pscore container running you can copy the relevant files from the container to your local file system:

$ cd ~detectron2/projects/deep_sort/

$ docker cp pscore:/usr/src/app/detectron2/projects/deep_sort/data ./

You will find the results of my last run of the container under the '~detectron2/projects/deep_sort/data_mask' folder.

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

Detectron2 also returns a mask, which allows analysis to be more focused on the subject of the detection and not noise also in the image. This helps a lot to ONLY use relevant parts of the detection for fitting the KMeans clustering.

Team classification is actually where a lot of the work on this requirement can be done as there can be many approaches to the problem of sorting players into teams. Here I use KMeans clustering to extract the main colors in each detected and tracked image and perform the classification according to the presence of the colors blue, yellow, or both!

Classifying the Yellow class using this method is a breeze and is highly accurate. However the Blue class presented a challenge because so many unrelated persons unrelated to players also wore jerseys or attire that returned positive to a test for color Blue! So I find that I spent an inordinate amount of time adjusting shades of blue color to look for and the proportion of such shades to look for in an image before a classification of Blue Team can be made. And then reducing one shade of blue may remove more noise from my clue classification, but then it may also reduce the accuracy of my blue predictions. This can certainly use more brainstorming that I will be more than happy to deliver.

This is not the only way but rather one that I chose in order not to get bogged down on this task while there are other tasks to be done that have a finite completion time.

Other approaches are to train a neural network to accept the entire image and return a team classification. Basically this part of the tasks can easily be infinite as there are many approaches that can be tried and compared. The one week allotted for this project will be insufficient.

## Technical notes

The way to use detectron2 is to create your project in the projects folder of a detectron2 repository. This is why you find deep_sort in the projects folder of the detectron2 repository here. That is how detectron2 recommends it be used as a library.

## Questions and Support

Please kindly email me at kay_taylor@outlook.com for any questions. I shall look forward to hearing from you.