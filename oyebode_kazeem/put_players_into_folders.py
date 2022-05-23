import numpy as np
import cv2
#from keras.models import load_model
import threading
#from keras.preprocessing.image import  img_to_array, load_img
#from tensorflow.keras.utils import img_to_array
from skimage.transform import resize
import random
import sys
import os
#import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import  load_model
im_width = 256
im_height = 256

blue_images_seg = []
yellow_images_seg = []

original_image = []
blue_images = []
yellow_images = []

try:
    base_path = sys._MEIPASS
except Exception:
    base_path = os.path.abspath(".")

new_model= os.path.join(base_path,'res34_vgg16_256_bin20_5_2022_3c.h5') # this gets the path to the multiclass segmentation
new_model_player= os.path.join(base_path,"player_classification1.h5") #this gets the path to the region proposal classification

model1 =load_model(new_model,compile=False)# load multiclass segmentation model
model = load_model(new_model_player) # load region proposal classification model

global TT

TT = True

def stop_video():  # this works if run via the GUI
    global TT
    TT = False
    print("STOPPED")

lock = threading.Lock()
def doforyellow(yellow_team,original_im): # this function cuts yellow players
        lock.acquire()

        print("doing yellow player")
        original_image = original_im # this is the original image      #[yellow_counter] #cv2.imread("C://Users//kazeem//Desktop//PAU//personal research//parallelscore//test//tester6.JPG")

        kernel = np.ones((2, 2), np.uint8) # create a 2 by 2 structuring element to erode or dilate
        #foreground = (np.array(y) > 1).sum()
        yt = cv2.dilate(yellow_team, kernel, iterations=4) # dilate the regions proposed, this removes errors
        contours, hierarchy = cv2.findContours(yt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get all the contours from the segmented output

        if (True):
        #if (realvalue[0][3] <= 0.09 and realvalue[0][3] >=0.001):

            for c in contours: # find all the contours
                    try:
                        x, y, w, h = cv2.boundingRect(c)
                        #if (cv2.contourArea(c)) > 100: # find contours greater than 50
                        if (cv2.contourArea(c)) > 100:  # find contours greater than 100
                        #cv2.rectangle(original_image, (x-12, y-6), (x + w, y + h+6), (255, 0, 0), 1)
                            crop_img = original_image[y-28:y + h+4, x-2:x + w] # crop player from the identified contour
                            crop_img=np.asarray(crop_img)
                            xx = crop_img.copy()
                            imagenp = np.array(cv2.resize(xx, (50, 50)) / 255)
                            expanded = np.expand_dims(imagenp, axis=0)

                            realvalue = model.predict(expanded)  # check if the proposed
                            value = np.argmax(realvalue)
                            print("value of yellow " + str(realvalue[0][3]))
                            #if (realvalue[0][3] > 0.0001):
                            #if (realvalue[0][3] > 0.0000001):
                            if (realvalue[0][3] < 0.00009): # get approved regions
                                s = realvalue[0][3]
                                v = random.randint(4,67777777777)
                                #filename = "C:\\Users\\kazeem\\Desktop\\PAU\\personal research\\parallelscore\\practice_code\\y\\"+"yellow_team"+str(yellow_team_counter) + ".png"

                                #filename = "y\\" + "yellow_team" + str(v) + ".png"

                                filename = ypath + "\\yellow_team" + str(s) +"--"+ str(v) + ".png" # save cropped player
                                #filename = "y\\" + "yellow_team" + str(s) +"--"+ str(v) + ".png"

                                cv2.imwrite(filename,crop_img)
                            #yellow_team_counter = yellow_team_counter+1
                    except:
                        continue
                #yellow_counter =  yellow_counter +1
        lock.release()

lock = threading.Lock()
def doforblue(blue_team, original_img): # this function cropped blue players
            lock.acquire()

            print("doing blue")
            original_image =original_img#get orignal image       [blue_counter]  # cv2.imread("C://Users//kazeem//Desktop//PAU//personal research//parallelscore//test//tester6.JPG")

            kernel = np.ones((2, 2), np.uint8)  # get a structuring element
            #foreground = (np.array(b) > 1).sum()
            #if(foreground > 2):

            bt = cv2.dilate(blue_team, kernel, iterations=3) # erode the proposed rejoins
            blue_t = cv2.erode(bt, kernel, iterations=2) # dilate

            contours, hierarchy = cv2.findContours(blue_t, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)  # find contours (players)


            if (True):
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    try:
                        if (cv2.contourArea(c)) > 100:  # find contours greater than 100

                            #cv2.rectangle(original_image, (x - 2, y - 6), (x + w, y + h + 6), (255, 0, 0), 1)
                            crop_img = original_image[y - 6:y + h + 4, x-4:x + w]  # crop players
                            crop_img = np.asarray(crop_img)
                            xx = crop_img.copy()
                            imagenp = np.array(cv2.resize(xx, (50, 50)) / 255) # check if proposed region is a valid region
                            expanded = np.expand_dims(imagenp, axis=0)

                            realvalue = model.predict(expanded)
                            value = np.argmax(realvalue)
                            if (value == 0): # if its not a valid region, do not save
                                v = random.randint(4, 67777777777)
                                #filename = os.path.join(base_path, "b\\" + "blue_team" + str(v) + ".png")
                                filename = bpath + "\\blue_team" + str(v) + ".png"   # save player
                                cv2.imwrite(filename, crop_img)
                                #blue_team_counter = blue_team_counter + 1
                    except:
                        continue
                #blue_counter = blue_counter+1
            lock.release()


def start_analysis():
    global TT
    print("Player Classification Started")
    cap = cv2.VideoCapture("deepsort_30sec.mp4")
    val, frame = cap.read()
    while(val and TT):
        val, frame = cap.read() # read frame after the other until the end

        if(val and TT):
            frame = cv2.resize(frame, (256, 256)) # resize each frame to 256 by 256 because the model takes this size

            #x = np.array(frame)
            #x = np.expand_dims(x, axis=0)
           # original_image.append(cv2.resize(frame,(500,500)))
            #oi = cv2.resize(frame, (500, 500))
            #ori = cv2.resize(frame, (500, 500))
            cv2.imwrite("orignalimage.jpg", frame)
            oi2 = cv2.imread('orignalimage.jpg')
            img = load_img('orignalimage.jpg')
            x_img = img_to_array(img)
            x_img = resize(x_img, (256, 256, 3), mode='constant', preserve_range=True)
            x = np.array(x_img)
            x = np.expand_dims(x, axis=0)

            original_image.append(cv2.resize(oi2,(256,256)))
            original_image.append(x_img)




            print(x.shape)

            predict = model1.predict(x) # the prediction gives a 256 by 256 by 3 result

            output_image = np.argmax(predict, axis=3) # now take the highest value
            output_image = np.array(output_image[0])


            yellow_players = np.where(output_image == 1, 255, output_image) # get the yellow section - players
            yellow_players = np.where(yellow_players < 255, 0, yellow_players)
            cv2.imwrite("segmented_yellow.png", yellow_players)
            yp = cv2.imread('segmented_yellow.png',0)
            #doforyellow(yp, oi2)

            process_yellow = threading.Thread(target=doforyellow, args=(yp, oi2)) # start a thread that crops yellow players
            process_yellow.start()

            blue_players = np.where(output_image == 2, 255, output_image) # get the blue section - players
            blue_players = np.where(blue_players < 255, 0, blue_players)

            cv2.imwrite("segmented_blue.png", blue_players)
            bp = cv2.imread('segmented_blue.png', 0)
            process_blue = threading.Thread(target=doforblue, args=(bp, oi2)) # start a thread that crops blue players
            process_blue.start()
        else:
            cap.release()


    cap.release()


# Path
ypath = os.path.join(base_path,"yellow_player")  # create this folders, if it exists , then ignore

isdir = os.path.isdir(ypath)
if(isdir==False):
    os.mkdir(ypath)

bpath = os.path.join(base_path,"blue_player")
isdir = os.path.isdir(bpath)
if(isdir==False):
    os.mkdir(bpath)


start_analysis()




