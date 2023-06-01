from PIL import Image
import glob

import tensorflow as tf

from keras import utils, layers, models
import matplotlib.pyplot as plt
import numpy as np

import os

model = models.Sequential()
model.add(layers.Rescaling(1./255))
model.add(layers.Conv2D(200, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(400, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(400, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(2))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path).expect_partial()

path_to_image = "C:/Users/woprg/Downloads/Brain-MRI-Classification/dataset_uniform/"

while True:
    try:
        command = input("Image: ")
        if command.lower() == "exit":
            break
        results = model.predict(np.asarray(Image.open(path_to_image + command)).reshape((1,200,200,3)))[0]
        if results[0] > results[1]:
            print("No Tumor")
        else:
            print("Tumor")
    except ValueError:
        print("Pick Another Image, this one is broken for no reason . . .")