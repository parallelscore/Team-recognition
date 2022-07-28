# OBJECTIVES:

The objective of the test is to identify, separate and store each team players from every frame in the given video clip.

# Approach used.

There are several approaches to achieve the objective of the proposed test. 

My approach follows the following proceedure:

1.  Read each frame from the video clip.
2.  For each frame, detect all the humans, crop them out and save the cropped image in a folder.
    I have used the pre-trained YOLO v5 model for the detection of players. 
    I tried to use the recently released YOLO v6, however, the current version is still very new and unstable.
    YOLO v5 have proven to be stable and successful in many deployed projects with decent performance. 
3.  I implement a K-means clustering algorithm to separate the players from each team. 
    Other approaches like, bag of visual words, linear binary pattern, SIFT, OrB, SURF, etc. could be used to extract the color and texture feature
    and classifiers like SVM can be used for classification. However, that would require a supervised training. Which is achievable but not an efficient approach,
    since it requires labelling some portion of the data. K-means algorithm is a light weight unspervised training approach that performs well for the task. 
    I used 3 clusters to identify NAP team, BAR team and Others (like referees, linesman etc.).
    
4. As requested, a dockerfile has been create for the test. The dockerfile hosts all the required scripts and files for this task to be achieved.
   The dockerfile is hosted on docker hub and can be run with the following command.
   
   ## docker build -t opeyemi2/parallel_score:p_score_updated ./
   
   ## docker run -t opeyemi2/parallel_score:p_score_updated
   
   The docker runs a bash script that install all the required dependencies and execute the python file.


I have chosen this approach while considering the following:
  
  1.    Efficiency (in terms of performance like accuracy).
  2.    Light weight for easy deployment, integration, and device-resource constraints (like mobile devices).
  3.    Time required to submission.
