import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PIL
import os
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import mrcfile as mrc
import sys

def OpenMrc(filename):
    # filename must be string
    map = mrc.read(filename)
    return map

class CentralSlicer():
    #Need standardised array shapes for machine learning
    def __init__(self, array):
        self.array = array

    def CentSliceZ(self):
        #takes numpy array
        array = self.array
        y,x,z = self.array.shape
        midz = z//2
        return self.array[:, :, midz]

    def CentSliceY(self):
        #takes numpy array
        y,x,z = self.array.shape
        midx = x//2
        midz = z//2
        return self.array[:, midx, midz]

    def CentSliceX(self):
        #takes numpy array
        y,x,z = self.array.shape
        midy = y//2
        midz = z//2
        return self.array[midy, :, midz]

def Projector(array, axis):
    arraySum = np.sum(array, axis=axis)
    return arraySum


#Open and pre-process map
MapLocation = sys.argv[1]
Map = OpenMrc(MapLocation)
MapEdit = CentralSlicer(Map)
Z = MapEdit.CentSliceZ()
Zresize = np.resize(Z, (100,100))
#Projected = Projector(Map, 2)

plt.imsave('testimage.png', Zresize, cmap='Greys')
ImageLoc = 'testimage.png'

#Set up information on the data
img_height = 100
img_width = 100

img = tf.keras.utils.load_img(
    ImageLoc, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


#Import network
TF_MODEL_FILE_PATH = 'model_resize.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite
predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
score_lite = tf.nn.softmax(predictions_lite)

class_names = ['Subtomogram Averaging', 'Tomogram']
print(
    "This map is likely a {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)



