# Cole Twitter Bot Project
# Training on Baby Yoda images

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import plotting

# Import images
PATH = os.path.join(os.path.dirname('/Users/colewood/PycharmProjects/TwitterBot2/Baby_Yoda'),'Baby_Yoda')
trainDir = os.path.join(PATH, 'Train')
validationDir = os.path.join(PATH, 'Validate')

numTrainingImages = len(os.listdir(trainDir))
numValidImages = len(os.listdir(validationDir))

print(numTrainingImages)
print(numValidImages)

# training variables from Tensorflow training website
batchSize = 128
epochs = 15
imageWidth = 150
imageHeight = 150

# Training Models
trainImageGenerator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validationImageGenerator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

trainDataGen = trainImageGenerator.flow_from_directory(batch_size=batchSize,
                                                       directory= trainDir,
                                                       shuffle= True,
                                                       target_size=(imageHeight, imageWidth),
                                                       class_mode='binary')

validationDataGen = validationImageGenerator.flow_from_directory(batch_size=batchSize,
                                                                 directory= validationDir,
                                                                 shuffle= True,
                                                                 target_size=(imageHeight, imageWidth),
                                                                 class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(imageHeight, imageWidth, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentrophy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    trainDataGen,
    steps_per_epoch=numTrainingImages,
    epochs=epochs,
    validation_data=validationDataGen,
    validation_steps=numValidImages
)


