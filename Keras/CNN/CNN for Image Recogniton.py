# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:26:03 2018

@author: aanishsingla
"""

#Import keras Packages
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout

img_size=64

#Initialize CNN
classifier = Sequential()

#Convoluton Layer 1
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),strides=(1,1), 
                             input_shape=(img_size,img_size,3),
                             activation="relu"))

#MaxPooling Layer 1
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Convoluton Layer 2
#classifier.add(Convolution2D(filters=32,kernel_size=(3,3),strides=(1,1),activation="relu"))

#MaxPooling Layer 2
#classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flatten
classifier.add(Flatten())

#Fully Connected layer
classifier.add(Dense(units=128, activation="relu"))
#classifier.add(Dropout(rate=0.1))

#classifier.add(Dense(units=128, activation="relu"))
#classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=1, activation="sigmoid"))

#Compile the Network
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
 #categorical_crossentropy for multi-class

#Image augumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

import os
print(os.getcwd())
os.chdir('E:\Python Dir\Deep_Learning_A_Z\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 2 - Convolutional Neural Networks (CNN)\Section 8 - Building a CNN\Convolutional_Neural_Networks') 

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(img_size,img_size),
        batch_size=32,
        class_mode='binary')

from keras.callbacks import EarlyStopping, ModelCheckpoint
    
cbk = [EarlyStopping(monitor="loss",min_delta=0.001,patience=3, restore_best_weights= True),
       ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)]

classifier.fit_generator(
            training_set,
            steps_per_epoch=(8000/32),
            epochs=30,
            validation_data=test_set,
            validation_steps=(2000/32),
            callbacks=cbk)

classifier.summary() 

#Predicting a single new sample
from keras.preprocessing import image
import numpy as np

img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', 
                    target_size=(img_size,img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.   #Re-scaling

images = np.vstack([x])
classes = classifier.predict_classes(images, batch_size=1)
print(classes)

print(training_set.class_indices)   # Encoding for classes

if classes[0][0] == 1:
    pred = "dog"
else:
    pred = "cat"
    
print("Prediction is " + pred)    