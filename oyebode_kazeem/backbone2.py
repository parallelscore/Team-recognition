
import tensorflow as tf
import os
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

os.environ["SM_FRAMEWORK"] = "tf.keras" #before the import
import segmentation_models as sm
SM_FRAMEWORK=tf.keras
import glob
import cv2
import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



segmentation_classes = 3  # Number of classes for segmentation

# training images would be stored here
train_images = []
image_size = 256
# get all the sample images
for directory_path in glob.glob("C:/Users/kazeem/Desktop/PAU/personal research/parallelscore/images_3d/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        train_images.append(img)


train_images = np.array(train_images)


# testing images will be stored here
gt_container = []
# get corresponding groundtruths
for directory_path in glob.glob("C:/Users/kazeem/Desktop/PAU/personal research/parallelscore/gt_3d/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)  # read image in gray scale

        mask = cv2.resize(mask, (image_size, image_size))
        gt_container.append(mask)


train_gt = np.array(gt_container)

# expand the ground truth in the 3rd axis from 45,256,256 to  45,256,256,1, for example
train_gt = np.expand_dims(train_gt, axis=3)


# split dataset into 70 percent training and then 30 percent testing
X_train, X_test, y_train, y_test = train_test_split(train_images, train_gt, test_size=0.30, random_state=0)



# convert to a 3d categorical groundtruth
cat_gt = to_categorical(y_train, num_classes=segmentation_classes)
# reshape into 32,256,256,3 - where 32 is our training gt
y_train_cat = cat_gt.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], segmentation_classes))


test_masks = to_categorical(y_test, num_classes=segmentation_classes)
# reshape into 13,256,256,3 - where 13 is our testing gt
y_test_cat = test_masks.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], segmentation_classes))


preprocess_images = sm.get_preprocessing('vgg16')


# preprocess images
X_train_samples = preprocess_images(X_train)
X_test_samples = preprocess_images(X_test)


pretrained_unet = sm.Unet('vgg16', encoder_weights='imagenet', classes=3, activation='softmax')

segmentation_accuracy_measurement = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
pretrained_unet.compile(keras.optimizers.Adam(0.0001), sm.losses.CategoricalFocalLoss(), metrics=segmentation_accuracy_measurement)



print(pretrained_unet.summary())
# store the best model
callbacks = [EarlyStopping(patience=10, verbose=1),
             ReduceLROnPlateau(factor=0.15, patience=3, min_Ir=0.00001, verbose=1),
             ModelCheckpoint('res34_vgg16_256_bin20_5_2022_3f.h5', verbose=1, save_best_only=True, save_weights_only=False)]


pretrained_unet.fit(X_train_samples,y_train_cat,batch_size=8,epochs=50,verbose=1,callbacks=callbacks,validation_data=(X_test_samples, y_test_cat))










