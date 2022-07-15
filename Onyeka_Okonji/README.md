###### Steps Involved:
<li>
  <ol> Instance Segmentation using Pixellib which takes as input the video, generates 751 image frames and segments unique player images. Pointrend_resnet60 was used for the process. Can be downloaded at https://drive.google.com/file/d/1VTDyV7n-bS-0VJh0iDR4AwD74t0ECSOr/view?usp=sharing </ol>
  <ol> Generation of image features using InceptionV3 architecture from Tensorflow </ol>
  <ol> Using generated features to perform image classification using K-Means clustering for 2 clusters (Barcelona - yellow and Napoli - Blue) </ol>
  <ol> Containerization using Docker </ol>
</li>

###### Side note:
Team Classification performed considerably well, with 2 images mis-classified and also because the video clip contained images of the referee and coach, it captures those two. A step can be added to remove such images.
