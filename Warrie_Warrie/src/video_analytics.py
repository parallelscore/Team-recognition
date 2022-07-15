import os
import torch
import shutil
import requests
import logging
import argparse

import cv2 as cv
import numpy as np

from tqdm import tqdm

from sklearn.cluster import KMeans

import torchvision.models as model
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor



# +++++++++++++++++++++++++++ Helper function to download video from google drive +++++++++++++++++++++++++++
def extract_id(url):
    url.split("/")[-2]
    return (url.split("/")[-2])


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None
def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    root_list = destination.split("/")
    root_folder = "/".join(root_list[0:-1])

    if not os.path.exists(root_folder):
      os.makedirs(root_folder)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# +++++++++++++++++++++++++++ Other video analytics helper function +++++++++++++++++++++++++++


def detect_human_yolo(image,img_counter, saved_file_path):  
    """detect human from a video image using yolov5s pretrained model and save the detected
    human object in a file: saved_file_path

    Args:
        image (array): a 3D array of an image
        img_counter (int): the current frame attributed to the image in a video
        saved_file_path (str): The path defined to save the detected human object in the image
    """
    
    # Model to detect human
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
    model.classes = [0] # use only person detection
    results = model(image).pred[0]

    if results.shape:
        if not os.path.exists(saved_file_path):
            os.makedirs(saved_file_path)
        
        object_counter = 0
        
        for (x1,y1,x2,y2,_, _) in results:
            #  cv.rectangle(image, (int(x1),int(y1)), (int(x2), int(y2)),(0,0,255), 2)
            file_name = os.path.join(saved_file_path, f'{img_counter}_{object_counter}.jpg')
            detected_obj = image[int(y1):int(y2), int(x1):int(x2)]
            cv.imwrite(file_name, detected_obj) # write detected_object_image to file
            object_counter+=1


def capture_and_detect_human(video_path, saved_detected_obj_dir):

   """This captures images in a video from video_path and then it runs human detection on each images
   and save the output of the detection in path saved_detected_obj_dir

   Args:
      video_path: path to where video file is located
      saved_detected_obj_dir: direcotory/path to save the detected object.
   """
   
   second = 0 
   capture_rate = 0.4 # capture frame every 0.4 seconds
   counter = 0
   vidcapture = cv.VideoCapture(video_path)

   if not os.path.exists(saved_detected_obj_dir):
      os.makedirs(saved_detected_obj_dir)

   #  improvement in using threading to extract images from the video
   while True:
      success, frame = vidcapture.read()
      if success:
         vidcapture.set(cv.CAP_PROP_POS_MSEC,second*1000)
         detect_human_yolo(frame,counter, saved_detected_obj_dir)
      else:
         vidcapture.release()
         break

      counter +=1
      second  += capture_rate     # ensure the capture is done every 0.4 seconds
      second = round(second, 2)
      
      
      
    


def model_extractor(model):
  """creating a feature extraction model from a pretrained model

  Args:
      model (torch model): Pretrained pytorch classification model

  Returns:
      torch model: Pretrained pytorch classification model with hidden layer as an output
  """
  return_nodes = ['flatten']
  model_layer = create_feature_extractor(model, return_nodes=return_nodes)
  return model_layer


def extract_image_rep(img_file_path, pretrained_model):

  """Extract image representation from an image_object using pretrained classification model.
  This is also called feature extraction.

  Args:
    img_file_path: path to location of image for file extraction
    pretrained_model: Model used to extract the features from an image

  Returns:
      array: the features(hidden layer) from this image classification. This feature is the layer 
      before the final classification layer. 
  """

  image = cv.imread(img_file_path)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

  # Transform the image to tensor, so it becomes readable with the torch model
  transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(512),
    transforms.Resize(224),
    transforms.ToTensor()                              
  ])

  image = transform(image)
  test_image = image.reshape(1,3,224,224)   # test_image.shape
 
  model = model_extractor(pretrained_model)
  with torch.no_grad():
      feature = model(test_image)
      flatten_feat = feature['flatten'].cpu().detach().numpy().reshape(-1)
  return flatten_feat


def classification_model(folder_path, model):

    """returns the classification of each image in folder_path
    into different team and classes and returns the team_id and image_file name

    Args:
        folder_path: folder/path to where image file is located.
        model: The pretrained model to be used for feature extraction

    Returns:
        dict: dictionary of each file_name as key and the team_id as value
    """
  

    features = []
    file_names = []

    for (i, file_name) in enumerate(tqdm(os.listdir(folder_path))):
        # if i == 1:
        if file_name.endswith('.jpg'):
            img_path = os.path.join(folder_path,file_name)
            # print(i, file_name)
            extracted_feat = extract_image_rep(img_path, model)
            features.append(extracted_feat)
            file_names.append(file_name)

    # run a clustering on the feature vectors
    k_model = KMeans(n_clusters=4, random_state=2022)
    data = np.array(features)
    print(data.shape)
    k_model.fit(data)
    feat_classes = k_model.labels_  #estimate the class of each object


    pred_dict = dict(zip(file_names, feat_classes))
    return pred_dict

def save_team(pred_dict, detected_obj_dir, saved_team_dir):
        # save each team to different directory 
        for img_name in pred_dict:

            # select only the well segemented team
            if pred_dict[img_name] in [0,1]:
                team_dir =  os.path.join(saved_team_dir, str(pred_dict[img_name]))
                if not os.path.exists(team_dir):
                        os.makedirs(team_dir)
                # copy the image from the detected object directory
                shutil.copy(os.path.join(detected_obj_dir, img_name), os.path.join(team_dir, img_name) )


if __name__ == '__main__':
    
    destination = '../data/deep30secs_sort.mp4'  # path to download the youtube video
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--team_dir_path",
        default="../data/output/team",
        help = "the location where the each player will be classifed into different team folder"
    ) 
    parser.add_argument(
        "--video_url",
        default="https://drive.google.com/file/d/1fqp2btzDh-_gIKe3aRodcvtfwPY5fOhO/view",
        help = "the url where the video is located in the drive"
    ) 
    # parser.add_argument(
    #     "--video_path",
    #     default="../data/deepsort_30sec.mp4",
    #     help="the path where the football video file is located"
    # )

    parser.add_argument(
        "--detected_object_path",
        default="../data/detected_object",
        help="the path where all the detected human object in the video are saved"
    )

    args = parser.parse_args()

    # download video in a path 
    logging.info("Video downloading from drive to.... " +args.video_url )
    video_id =extract_id(args.video_url)
    download_file_from_google_drive(video_id, destination)


    # define resnet model for feature extraction
    res_model = model.resnet18(pretrained=True)

    # define the directory's to use   
    team_dir = args.team_dir_path
    detected_obj_dir =args.detected_object_path
    # video_dir = args.video_path

    #  Extract frames from video and store detected human object in a folder
    capture_and_detect_human(destination, detected_obj_dir)
    logging.info("Finished object detection")
    logging.info("============================")

    # print(video_dir)


    
    # Extract feature model from 

    logging.info("Starting Feature extraction...")
    pred_dict = classification_model(folder_path=detected_obj_dir, model=res_model)


    # save each detected object to their respective team based on the result of the classifier
    logging.info("Saving Team to their respective folder...")
    save_team(pred_dict, detected_obj_dir, team_dir)