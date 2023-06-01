from PIL import Image
import glob

import tensorflow as tf

from keras import utils, layers, models
import matplotlib.pyplot as plt

import os

PATH="C:/Users/woprg/Downloads/Brain-MRI-Classification/dataset_uniform/"

PATH_TO_YES = "C:/Users/woprg/Downloads/Brain-MRI-Classification/datasets/yes/*"
PATH_TO_NO = "C:/Users/woprg/Downloads/Brain-MRI-Classification/datasets/no/*"

yes_images = [Image.open(i).resize((200,200)).save("C:/Users/woprg/Downloads/Brain-MRI-Classification/dataset_uniform/yes/" + str(i)[i.rfind("\\"):i.rfind(".")] + ".png") for i in glob.glob(PATH_TO_YES)]
no_images = [Image.open(i).resize((200,200)).save("C:/Users/woprg/Downloads/Brain-MRI-Classification/dataset_uniform/no/" + str(i)[i.rfind("\\"):i.rfind(".")] + ".png") for i in glob.glob(PATH_TO_NO)]

train_ds = tf.keras.utils.image_dataset_from_directory(
  PATH,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(200, 200),
  batch_size=32)

val_ds = utils.image_dataset_from_directory(
  PATH,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(200, 200),
  batch_size=32)

normalization_layer = layers.Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_ds, epochs=10, 
                    validation_data=val_ds,callbacks=[cp_callback])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.savefig("model_accuracy.png")

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(test_acc)