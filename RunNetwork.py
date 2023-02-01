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


class emMap():
    # Need standardised array shapes for machine learning
    def __init__(self, filename):
        self.array = mrc.read(filename)

    def CentSliceZ(self):
        # takes numpy array
        array = self.array
        y, x, z = self.array.shape
        midz = z // 2
        return self.array[:, :, midz]

    def CentSliceY(self):
        # takes numpy array
        y, x, z = self.array.shape
        midx = x // 2
        midz = z // 2
        return self.array[:, midx, midz]

    def CentSliceX(self):
        # takes numpy array
        y, x, z = self.array.shape
        midy = y // 2
        midz = z // 2
        return self.array[midy, :, midz]


class runModel():
    def __init__(self, modelPath):
        self.modelPath = modelPath

    def imageSetup(self, imgLocation):
        img_height = 100
        img_width = 100

        img = tf.keras.utils.load_img(
            imgLocation, target_size=(img_height, img_width)
        )
        imgArray = tf.keras.utils.img_to_array(img)
        self.imgArray = tf.expand_dims(imgArray, 0)  # Create a batch

    def runPrediction(self):
        interpreter = tf.lite.Interpreter(model_path=self.modelPath)
        classify_lite = interpreter.get_signature_runner('serving_default')
        predictions_lite = classify_lite(rescaling_1_input=self.imgArray)['outputs']
        score_lite = tf.nn.softmax(predictions_lite)

        class_names = ['Non-Tomogram', 'Tomogram']
        print(
            "This map is likely a {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )

def Projector(array, axis):
    arraySum = np.sum(array, axis=axis)
    return arraySum


#Open and pre-process map
MapLocation = sys.argv[1]
MapEdit = emMap(MapLocation)
Z = MapEdit.CentSliceZ()

plt.imsave('ImageToClassify.png', Z, cmap='Greys')
ImageLoc = 'ImageToClassify.png'

#Set up information on the data

mapTypePredictor = runModel('Training_summary_fullDataSet_ImgSize100_learningrate1e3_epoch75_wdienetwork2_NETWORK.tflite')
mapTypePredictor.imageSetup(ImageLoc)
mapTypePredictor.runPrediction()



