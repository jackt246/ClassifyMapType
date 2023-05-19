import os

#Codon environment setup

os.system('module load cuda-11.1.1-gcc-9.3.0-oqr2b7d')
os.system('module load cudnn-8.0.4.30-11.1-gcc-9.3.0-bbr3kjv')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mrcfile




#Load data function to be used by tf later on
def load_data(filepath_tensor, label):

    filepath = filepath_tensor.numpy().decode('utf-8')
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
    return data, label

# Define data directories
train_dir = 'Classes3D/Train/'
val_dir = 'Classes3D/Validation/'

#create empty lists for filepaths and labels
train_filepaths = []
train_labels = []
val_filepaths = []
val_labels = []

# Define mapping from class names to class indices
class_map = {'Tomograms': 0, 'NonTomograms': 1}

trainClassFolders = [folder for folder in os.listdir(train_dir) if folder != ".DS_Store"]
valClassFolders = [folder for folder in os.listdir(val_dir) if folder != ".DS_Store"]

for class_folder in trainClassFolders:
    class_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_path):
        file_names = os.listdir(class_path)

        # Iterate over the files in the class folder
        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            train_filepaths.append(file_path)
            train_labels.append(class_map[class_folder])  # Assuming class folder name represents the label
    else:
        print(f"Warning: Directory '{class_path}' doesn't exist or is not a directory.")

for class_folder in valClassFolders:
    class_path = os.path.join(val_dir, class_folder)
    if os.path.isdir(class_path):
        file_names = os.listdir(class_path)

        # Iterate over the files in the class folder
        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            val_filepaths.append(file_path)
            val_labels.append(class_map[class_folder])  # Assuming class folder name represents the label
    else:
        print(f"Warning: Directory '{class_path}' doesn't exist or is not a directory.")



print('making datasets')
datasetTraining = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labels))
datasetTraining = datasetTraining.map(lambda x, y: tf.py_function(load_data, [x, y], [tf.float32, y.dtype]), num_parallel_calls=tf.data.AUTOTUNE)

datasetValidation = tf.data.Dataset.from_tensor_slices((val_filepaths, val_labels))
datasetValidation = datasetValidation.map(lambda x, y: tf.py_function(load_data, [x, y], [tf.float32, y.dtype]), num_parallel_calls=tf.data.AUTOTUNE)

# Define input shape
input_shape = (200, 200, 200, 1)

#Define the model
model = Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(128, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'], run_eagerly=True)

model.summary()

#Define running conditions

batch_size = 1
epochs = 5
steps_per_epoch = len(train_filepaths) // batch_size
validation_steps = len(val_filepaths) // batch_size

# Configure the training and validation datasets
datasetTraining = datasetTraining.shuffle(len(train_filepaths)).repeat().batch(batch_size)
datasetValidation = datasetValidation.batch(batch_size)

# Train the model
history = model.fit(
    datasetTraining,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=datasetValidation,
    validation_steps=validation_steps
)

# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

figtitle = '3Dclassification.png'
plt.savefig('Outputs/{}'.format(figtitle))