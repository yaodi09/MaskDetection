import cv2 as cv
import numpy as np

#import pandas as pd

from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

batch_size = 10
num_class = 2
epochs = 12

train = './dataset/train'
valid = './dataset/valid'
test = './dataset/test'

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   rotation_range = 40,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.15,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')
train_generator = train_datagen.flow_from_directory(train, 
                                                    batch_size = batch_size, 
                                                    target_size = (150, 150))

valid_datagen = ImageDataGenerator(rescale = 1.0/255)
valid_generator = valid_datagen.flow_from_directory(valid, 
                                                    batch_size = batch_size, 
                                                    target_size = (150, 150))


test_datagen = ImageDataGenerator(rescale = 1.0/255)
test_generator = test_datagen.flow_from_directory(test,
                                                  batch_size = batch_size,
                                                  target_size = (150, 150))

model = Sequential()
model.add(Conv2D(100, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(Conv2D(100, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


checkpoint = ModelCheckpoint('cp-{epoch:03d}.h5', monitor = 'loss', verbose = 1, save_best_only = True, mode = 'auto')

score = model.fit_generator(train_generator,
                            epochs = epochs,
                            validation_data = valid_generator,
                            callbacks = [checkpoint])

#model.save("model.h5")