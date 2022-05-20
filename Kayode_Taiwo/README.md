My source code for this project is in the detectron2 folder. Specifically with the detection2/projects/deep_sort folder.

The resulting docker container is hosted online at Docker Hub. To run please do:

$ docker run -it --name pscore geek0075/pscore:thu_may19_v1

This has to process all 751 frames in the provided 30 seconds video. So it runs for a couple of hours. After it's run is over you can connect to it using the Docker Dashboard. Look under containers for pscore. Click on CLI. 

Then you can navigate to the folder ./data/output/blue, ./data/output/yellow, and ./data/output/blue-yellow, on the CLI.  

Please NOTE that the above is a work in progress until the deadline for the test elapses. Thanks.