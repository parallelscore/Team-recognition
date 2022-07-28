import cv2
import numpy as np
import os

images =next( os.walk("C:\\Users\\kazeem\\Desktop\\PAU\\personal research\\parallelscore\\paintingfolder"))[2]

# this creates the groudtruths for the multiclass training process
for file_name in images:
    #file_name = "image" + str(t) +".JPG"
    path = "C:\\Users\\kazeem\\Desktop\\PAU\\personal research\\parallelscore\\paintingfolder\\" + file_name
    the_coloured_image = cv2.imread(path)
    #blank_image = np.zeros((300,671))
    blank_image = np.zeros((the_coloured_image.shape[0],the_coloured_image.shape[1]))

    for x in range(the_coloured_image.shape[0]):
        for y in range(the_coloured_image.shape[1]):
            if((the_coloured_image[x,y,2] >=210 and the_coloured_image[x,y,1] <=60 and the_coloured_image[x,y,0] <=60) or (the_coloured_image[x, y, 2] <= 60 and the_coloured_image[x, y, 1] >= 150 and the_coloured_image[x, y, 0] >= 60) ):
                #blank_image[x,y] = 255 #yellow team
                blank_image[x, y] = 1  # get yellow areas marked

            # get blue areas marked
            if(the_coloured_image[x, y, 2] <= 60 and the_coloured_image[x, y, 1] >= 150 and the_coloured_image[x, y, 0] >= 60):
                #blank_image[x, y] = 128 Blue team
                blank_image[x, y] = 2 # 128
    df =file_name.replace(".JPG", ".png")
    file_name = "C:/Users/kazeem/Desktop/PAU/personal research/parallelscore/gt_3d/"+df #"image" + str(t)  +".png"
    print(file_name)
    cv2.imwrite(file_name,blank_image)