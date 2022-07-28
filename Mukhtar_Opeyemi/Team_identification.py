# Install all the dependencies required with:
    # pip install -r requirements.txt


import os
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# # Step 1: Extract players from each frame
    # This function crops the specific location of each player.
    # The bounding box for each player is used for the array slicing.
    # Every extracted players are saved in their respective frame folders.

def crop_clip(r, frame, f_counter):
    for cnt in range(len(r)):
        elbl = r["class"][cnt]
        if elbl == 0:
            x,y,w,h = round(r.loc[cnt]["xmin"]),round(r.loc[cnt]["ymin"]), round(r.loc[cnt]["xmax"]), round(r.loc[cnt]["ymax"])

            w2 = w - x
            h2 = h - y
            
            cropped_image = frame[y:y+h2, x:x+w2]
            fframe = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            if not os.path.exists(f'cropped/f_counter_{f_counter}'):
                os.makedirs(f'cropped/f_counter_{f_counter}')
            plt.imsave(f'cropped/f_counter_{f_counter}/player{cnt}.png', fframe)


# This cell loads the video clip and YOLO v5 (for detection).
    # YOLO object detection model identifies all the players in each frame. 

vid = cv2.VideoCapture('Data/deepsort_30sec.mp4')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
frame_counter = 0

pbar = tqdm(total = 751, colour="green")
while vid.isOpened():
    pbar.set_description(f"Extracting players ")
    status, frame = vid.read()
    
    if not status:
        break


    results = model(frame)
    r = results.pandas().xyxy[0]
    
    crop_clip(r, frame, frame_counter)
    pbar.set_postfix(frame_counter=frame_counter)
    pbar.update(1)
    
    
    frame_counter += 1
        
vid.release()
pbar.close()


# # Prepare data for player's team clustering

data = []
base_path = "cropped"
IMG_SIZE = 32
for base in os.listdir(base_path):
    if base == '.DS_Store':
        continue        
    for file in os.listdir(os.path.join(base_path, base)):
        if file == '.DS_Store':
            continue
        
        img=cv2.imread(os.path.join(base_path, base, file))
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=img.astype('float32')

        data.append(img)

data = np.array(data)        
data = data/255.0
reshaped_data = data.reshape(len(data),-1)
# print(reshaped_data.shape)


# # Use K-means clustering algorithm to seperate players into different teams
    # 3 clusters was used.
    # Two clusters for each team 
    # The third cluster is to categorize others, like, referee, linesman, coach etc.
print('\nPerforming K-means clustering....')
kmeans = KMeans(n_clusters=3, random_state=0).fit(reshaped_data)

# This function preprocess (load, transform, normalize, flatten) individual player image for clustering
def process_img (base_path, frame_path, file_img):
    IMG_SIZE = 32
    img=cv2.imread(os.path.join(base_path, frame_path, file_img))
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img__=img.astype('float32')
    img__ = np.array(img__)
    img__ = img__/255.0
    return img__.reshape(1,-1)


# This cell compares and separate each players to their respective teams.
base_path = 'cropped'

# print("Separating players into individual teams...")
pbar = tqdm(total = 7941, colour="green")
for frame_dir in sorted(os.listdir(base_path)):
    if frame_dir == '.DS_Store': continue
    for file_img in sorted(os.listdir(os.path.join(base_path, frame_dir))):
        pbar.set_description(f"Separating players into individual teams")
        if file_img == '.DS_Store': continue
        prepare_file = process_img(base_path, frame_dir, file_img)
        
        if kmeans.predict(prepare_file) == 0:
            if not os.path.exists(os.path.join(base_path,frame_dir,'BAR')):
                os.makedirs(os.path.join(base_path,frame_dir,'BAR'))
            shutil.move(os.path.join(base_path, frame_dir, file_img), os.path.join(base_path, frame_dir, 'BAR', file_img))
            
        if kmeans.predict(prepare_file) == 1:
            if not os.path.exists(os.path.join(base_path,frame_dir,'NAP')):
                os.makedirs(os.path.join(base_path,frame_dir,'NAP'))
            shutil.move(os.path.join(base_path, frame_dir, file_img), os.path.join(base_path, frame_dir, 'NAP', file_img))

        
        if kmeans.predict(prepare_file) == 2:
            if not os.path.exists(os.path.join(base_path,frame_dir,'Others')):
                os.makedirs(os.path.join(base_path,frame_dir,'Others'))
            shutil.move(os.path.join(base_path, frame_dir, file_img), os.path.join(base_path, frame_dir, 'Others', file_img))
        pbar.update(1)




