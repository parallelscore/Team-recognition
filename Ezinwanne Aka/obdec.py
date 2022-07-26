import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import tensorflow as tf
import numpy as np
import os, json, cv2, random
from keras.models import load_model
import keras
from keras.layers import Dense, Flatten
from keras.models import Model 
from keras.preprocessing.image import ImageDataGenerator, load_img , img_to_array
from keras.applications.inception_v3 import preprocess_input
import glob
from PIL import Image
import os, os.path
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



#creatig the directories that will be used to store our outputs
os.mkdir('images')
os.mkdir('pic')
os.mkdir('team1')
os.mkdir('team2')


#This block of code is to split the video into picture frames
cap = cv2.VideoCapture('./deepsort_30sec.mp4')
i = 0

frame_skip = 1 #this determines the steps in the number of frames being extracted
frame_count = 0
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break
    if i > frame_skip - 1:
        frame_count += 1
        cv2.imwrite('./images/FRAME_'+str(frame_count*frame_skip)+'.jpg', frame) #saving the frames to the images directory
        i = 0
        continue
    i += 1

cap.release()
#cv2.destroyAllWindows()


#this block of code extracts every image of a person on each frame of the video
path = "./images"

for root, directories, files in os.walk(path):
    for name in files:
        nu=os.path.join(root, name)
        im=cv2.imread(nu)


        cfg = get_cfg()
        
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)


        for idx,val in enumerate (outputs["instances"].pred_classes):
            if val==0:
                masks = np.asarray(outputs["instances"].pred_masks.to("cpu"))
                item_mask = masks[idx]
                segmentation = np.where(item_mask == True)
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
                print(x_min, x_max, y_min, y_max)
                
                cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :], mode='RGB')

                mask = Image.fromarray((item_mask * 255).astype('uint8'))

                cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

                u=np.array(cropped)

                cv2.imwrite(f"./pic2/pic{name}_{idx}.png", u)


#This block applies the classification model trained on the images of both teams in other to classify yhe players
model= load_model("./team_model.h5")
path1='./pic2'
for root, directories, files in os.walk(path1):
    for name in files:
        nu=os.path.join(root, name)
        img=load_img(nu, target_size=(256,256))
        im=cv2.imread(nu)
        i= img_to_array(img)

        i=preprocess_input(i)

        input_arr=np.array([i])

        pred=np.argmax(model.predict(input_arr))
        
        #this saves the images of the players based on the team the belong to
        if pred == 1:
            cv2.imwrite(f"./team1/pic{name}.png", im)

        if pred == 2: 
            cv2.imwrite(f"./team2/pic{name}.png", im)
            
            
