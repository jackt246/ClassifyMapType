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
from scipy.ndimage.interpolation import zoom

#Load data

# Define data directories
train_dir = 'Classes/Train/'
val_dir = 'Classes/Validation/'

# Define mapping from class names to class indices
class_map = {'Tomogram': 0, 'NonTomogram': 1}

# Define data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and input shape
batch_size = 32
input_shape = (None, None, None, 1)

# Define train and validation data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:3],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:3],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary')

# Read MRC files into numpy arrays
train_data = []
train_labels = []
for subdir in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir, subdir)):
        class_name = subdir.split('/')[-1]
        class_idx = class_map[class_name]
        for filename in os.listdir(os.path.join(train_dir, subdir)):
            filepath = os.path.join(train_dir, subdir, filename)
            with mrcfile.open(filepath) as mrc:
                volume = mrc.data.astype(np.float32)
                volume = np.resize(volume, (64, 64, 64))
            train_data.append(volume)
            train_labels.append(class_idx)

val_data = []
val_labels = []
for subdir in os.listdir(val_dir):
    if os.path.isdir(os.path.join(val_dir, subdir)):
        class_name = subdir.split('/')[-1]
        class_idx = class_map[class_name]
        for filename in os.listdir(os.path.join(val_dir, subdir)):
            filepath = os.path.join(val_dir, subdir, filename)
            with mrcfile.open(filepath) as mrc:
                volume = mrc.data.astype(np.float32)
                volume = np.resize(volume, (64, 64, 64))
            val_data.append(volume)
            val_labels.append(class_idx)

print('data loaded')

# Convert data to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

# Define input shape
input_shape = (64, 64, 64, 1)

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
epochs = 25
#train and save as a history object for plotting.
history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(val_data, val_labels))

#Plot a bunch of stats
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
figtitle='Training_summary_3D.png'
plt.savefig('Outputs/{}'.format(figtitle))
print(figtitle)

# Convert the model to a tf lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#Output the graph values
df = pd.DataFrame(list(zip(acc, val_acc, loss, val_loss)), columns=['Accuracy', 'Val_Accuracy', 'Loss', 'Val_Loss'])
df.to_csv('Outputs/{}.csv'.format(figtitle.strip('.png')))

# Save the model.
#with open('Training_summary_ImgSize200_learningrate1e4_epoch5_SGD_noaug_dropout0_01ds_01val.tflite', 'wb') as f:
#  f.write(tflite_model)