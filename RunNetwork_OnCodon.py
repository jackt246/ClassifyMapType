import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mrcfile as mrc
import sys

class emMap():
    #Need standardised array shapes for machine learning
    def __init__(self, filename):
        self.array = mrc.read(filename)

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

class runModel():
    def __init__(self, modelPath, mapPath):
        self.modelPath = modelPath
        self.mapArray = mrc.read(mapPath)

    def centSlicer(self):
        array = self.mapArray
        y, x, z = self.mapArray.shape

        centZ = z // 2
        centX = x // 2
        centY = y // 2
        #model takes an image of 100,100 so resize the central slices
        self.centZSlice = np.resize((array[:, :, centZ]), (100,100))
        self.centXSlice = np.resize((array[:, centX, :]), (100,100))
        self.centYSlice = np.resize((array[centY, :, :]), (100,100))

    def imageSetup(self):
        # tensorflow is fussy and wants an RGB image. We have to trick it
        # when using greyscale and just duplicate the data into a 3D array
        centZSliceLarge = np.stack([self.centZSlice, self.centZSlice, self.centZSlice], axis=2)
        centXSliceLarge = np.stack([self.centXSlice, self.centXSlice, self.centXSlice], axis=2)
        centYSliceLarge = np.stack([self.centYSlice, self.centYSlice, self.centYSlice], axis=2)

        #Make the array a tensor
        self.imgArrayZ = tf.expand_dims(centZSliceLarge, 0)  # Create a batch
        self.imgArrayX = tf.expand_dims(centXSliceLarge, 0)  # Create a batch
        self.imgArrayY = tf.expand_dims(centYSliceLarge, 0)  # Create a batch

    def runPrediction(self):
        interpreter = tf.lite.Interpreter(model_path=self.modelPath)
        classify_lite = interpreter.get_signature_runner('serving_default')

        predictions_liteZ = classify_lite(sequential_input=self.imgArrayZ)['outputs']
        #predictions_liteX = classify_lite(rescaling_1_input=self.imgArrayX)['dense_1']
        #predictions_liteY = classify_lite(rescaling_1_input=self.imgArrayY)['dense_1']

        score_liteZ = tf.nn.softmax(predictions_liteZ)

        class_names = ['Non-Tomogram', 'Tomogram']
        print(
            "This map is likely a {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_liteZ)], 100 * np.max(score_liteZ))
        )

        data = pd.DataFrame({'Map': file, 'Expected Type': Folder, 'Predicted Type': class_names[np.argmax(score_liteZ)],
                             'Prediction score %': 100 * np.max(score_liteZ)}, index=[0])
        return data


Class = 'Tomograms'
Folder = '../{}/'.format(Class)
FilesList = os.listdir(Folder)

Results = pd.DataFrame(columns=['Map', 'Expected Type', 'Predicted Type', 'Prediction score %'])

#Open and pre-process map

for file in FilesList:
    # Open and pre-process map
    MapLocation = '{}/{}'.format(Folder, file)


    # Set up information on the data

    Model = runModel('model_SGD_subsetdata_10epoch_90val.tflite', MapLocation)
    Model = runModel('model_SGD_subsetdata_10epoch_90val.tflite', MapLocation)
    Model.centSlicer()
    Model.imageSetup()
    data = Model.runPrediction()

    Results = pd.concat([Results, data], ignore_index=True)

print(Results)
Results.to_csv('results_{}.csv'.format(Class))

