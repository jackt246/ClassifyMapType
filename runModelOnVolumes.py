import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mrcfile as mrc
import sys


class runModel():
    def __init__(self, modelPath, mapPath):
        self.modelPath = modelPath
        self.mapArray = mrc.read(mapPath)

    def runPrediction(self):
        interpreter = tf.lite.Interpreter(model_path=self.modelPath)
        #Tensorflow has a habit of changing the signatures so this is run to tell me what it is
        signatures = interpreter.get_signature_list()
        print('Signature: {}'.format(signatures))

        classify_lite = interpreter.get_signature_runner('serving_default')
        predictions = classify_lite(rescaling_1_input=self.preppedarray)['outputs']


        score = tf.nn.softmax(predictions)

        class_names = ['Non-Tomogram', 'Tomogram']
        print(
            "This map is likely a {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        data = pd.DataFrame({'Map': file, 'Expected Type': Folder, 'Predicted Type': class_names[np.argmax(score)],
                             'Prediction score %': 100 * np.max(score)}, index=[0])
        return data

    def cropAndPad(self):
        endArraySize = 200
        print(self.mapArray)
        #pad array to end size
        pad_width = [(max(endArraySize - shape, 0) // 2, max(endArraySize - shape, 0) - max(endArraySize - shape, 0) // 2) for shape in self.mapArray.shape]
        arrayPad = np.pad(self.mapArray, pad_width, 'constant')

        if any(x > 200 for x in arrayPad.shape):
            #then crop
            axis0start = arrayPad.shape[0] // 2 - (endArraySize//2)
            axis1start = arrayPad.shape[1] // 2 - (endArraySize//2)
            axis2start = arrayPad.shape[2] // 2 - (endArraySize//2)

            preppedarray = arrayPad[axis0start:axis0start+endArraySize, axis1start:axis1start+endArraySize,
                           axis2start:axis2start+endArraySize]

            return self.preppedarray
        else:
            return self.arrayPad

Folder = 'ValidationData_NotForTraining/Tomograms'
FilesList = os.listdir(Folder)

Results = pd.DataFrame(columns=['Map', 'Expected Type', 'Predicted Type', 'Prediction score %'])

#Open and pre-process map

for file in FilesList:
    # Open and pre-process map
    MapLocation = '{}/{}'.format(Folder, file)

    # Set up information on the data

    Model = runModel('3dconv.tflite', MapLocation)
    Model.cropAndPad()
    data = Model.runPrediction()

    Results = pd.concat([Results, data], ignore_index=True)

print(Results)
Results.to_csv('results_{}.csv'.format(Folder))
