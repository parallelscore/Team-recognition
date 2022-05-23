import keras
import tensorflow as tf
from keras.layers import Input, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,  Dropout
from keras.layers import Dense, Flatten


image_size = 50


path = "C:\\Users\\kazeem\\Desktop\\PAU\\personal research\\parallelscore\\player_training\\"
pathv = "C:\\Users\\kazeem\\Desktop\\PAU\\personal research\\parallelscore\\testing\\"
train_datagen = ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path,  # this is the target directory
        target_size=(image_size, image_size),
        class_mode='categorical',
        color_mode="rgb",
        batch_size=16
        )

validation_generator = test_datagen.flow_from_directory(
        pathv,
        target_size=(image_size, image_size),
        class_mode='categorical',
        color_mode="rgb",
        batch_size=16
        )


model = keras.models.Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3), padding='valid',activation='relu',input_shape=(image_size,image_size,3)))
model.add(BatchNormalization())
#model.add(Activation='relu')
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.025))

model.add(Conv2D(filters=8,kernel_size=(3,3), padding='valid',activation='relu'))
model.add(BatchNormalization())
#model.add(Activation='relu')
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.025))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(4, activation='softmax',kernel_regularizer=tf.keras.regularizers.l1(0.02),activity_regularizer=tf.keras.regularizers.l2(0.01)))


model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

callbacks = [EarlyStopping(patience=10, verbose=1),
             ReduceLROnPlateau(factor=0.15, patience=3, min_Ir=0.00001, verbose=1),
             ModelCheckpoint('player_classification1.h5', verbose=1, save_best_only=True, save_weights_only=False)]

model.fit(train_generator,validation_data=validation_generator,batch_size=64,epochs=300,callbacks=callbacks)

