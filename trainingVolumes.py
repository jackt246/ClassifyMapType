import os
import sys

sklearn_path = '/hps/software/users/gerard/emdb/miniconda3/envs/jack/lib/python3.10/site-packages'
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
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
import seaborn as sns



# Load data function to be used by tf later on
def load_data(filepath_tensor, label):
    filepath = filepath_tensor.numpy().decode('utf-8')
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
    return data, label

# Define data directories
train_dir = 'Classes3D/Train/'
val_dir = 'Classes3D/Validation/'

#_________ Define some variables that will be used for running _________#

batch_size = 1
epochs = 50
trainingRate = 1e-5
dropout = 0.2

# Filname of figure with accuracy and loss info

figtitle = '3Dclassification_1e-5_epoch1_dropout02.png'

# Filename for output model so we can reuse it if it is any good

modelFileName = 'Model_3D_1e-5_dropout02.tflite'

# Run in testing mode (only use subset of data) Y = 1 N = 0

TestingMode = 1

#____________________________________________________________________#

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
            n = 1
            file_path = os.path.join(class_path, file_name)
            train_filepaths.append(file_path)
            train_labels.append(class_map[class_folder])  # Assuming class folder name represents the label
            n = n+1
            if TestingMode == 1 and n > 50:
                break
    else:
        print(f"Warning: Directory '{class_path}' doesn't exist or is not a directory.")

for class_folder in valClassFolders:
    class_path = os.path.join(val_dir, class_folder)
    if os.path.isdir(class_path):
        file_names = os.listdir(class_path)

        # Iterate over the files in the class folder
        for file_name in file_names:
            n = 1
            file_path = os.path.join(class_path, file_name)
            val_filepaths.append(file_path)
            val_labels.append(class_map[class_folder])  # Assuming class folder name represents the label
            n = n+1
            if TestingMode == 1 and n > 50:
                break
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
    layers.Dropout(dropout),
    layers.Conv3D(64, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(128, (3, 3, 3), activation='relu'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=trainingRate),
              metrics=['accuracy'], run_eagerly=True)

model.summary()

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

# Initialize lists to accumulate predictions and true labels
y_pred_accumulated = []
y_true_accumulated = []

# Iterate over the dataset batch by batch
for x_batch, y_batch in datasetValidation:
    # Get predictions for the batch
    y_pred_batch = model.predict(x_batch)
    y_pred_accumulated.extend(y_pred_batch)

    # Convert true labels to numpy array and append to accumulated list
    y_true_batch = np.array(y_batch).flatten()
    y_true_accumulated.extend(y_true_batch)

# Convert accumulated lists to numpy arrays
y_pred_accumulated = np.array(y_pred_accumulated)
print('y_pred_accumulated is {} and the shape is {} wth a length of {}'.format(y_pred_accumulated,
                                                                               y_pred_accumulated.shape, len(y_pred_accumulated)))
y_true_accumulated = np.array(y_true_accumulated)
print('y_true_accumulated is {} and the shape is {} wth a length of {}'.format(y_true_accumulated,
                                                                               y_true_accumulated.shape, len(y_true_accumulated)))
y_pred_classes = np.argmax(y_pred_accumulated, axis=1)
# Convert one-hot encoded labels to single-label integers
y_true_single_label = np.argmax(y_true_accumulated)

# Calculate precision, recall, etc. using y_true_accumulated and y_pred_accumulated
# Assuming the second column contains probability for class 1 (tomogram)
y_pred_prob = y_pred_accumulated[:, 1]
precision, recall, thresholds = precision_recall_curve(y_true_accumulated, y_pred_prob)

# Plot the precision and recall curve
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('Outputs/{}_precisionRecall.png')

# Get the model's predictions for the validation set
y_val_pred_prob = model.predict(datasetValidation)
y_val_pred_classes = np.argmax(y_val_pred_prob, axis=1)

# Convert y_test to a NumPy array and flatten it
y_val_true = np.array(val_labels).flatten()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val_true, y_val_pred_classes)
# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tomograms', 'NonTomograms'],
            yticklabels=['Tomograms', 'NonTomograms'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Outputs/confusionMatrix.png')

# Print classification report
report = classification_report(y_val_true, y_val_pred_classes, target_names=['Tomograms', 'NonTomograms'])
print("Classification Report:\n", report)

# convert to tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()


#Output the graph values
df = pd.DataFrame(list(zip(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])), columns=['Accuracy', 'Val_Accuracy', 'Loss', 'Val_Loss'])
df.to_csv('Outputs/{}.csv'.format(figtitle.strip('.png')))

# Save the model.
with open(modelFileName, 'wb') as f:
  f.write(tflite_model)