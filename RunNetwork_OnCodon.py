import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

Folder = 'SubtomgramAverages'
FilesList = os.listdir(Folder)

Results = {}

#Open and pre-process map

for file in FilesList:
    MapLocation = '{}/{}'.format(Folder, file)
    print(MapLocation)
    Map = OpenMrc(MapLocation)
    MapEdit = CentralSlicer(Map)
    Z = MapEdit.CentSliceZ()


    plt.imsave('ImageToClassify_subtomo.png', Z, cmap='Greys')
    ImageLoc = 'ImageToClassify_subtomo.png'

    #Set up information on the data
    img_height = 100
    img_width = 100

    img = tf.keras.utils.load_img(
    ImageLoc, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch


    #Import network
    TF_MODEL_FILE_PATH = 'model.tflite' # The default path to the saved TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    classify_lite = interpreter.get_signature_runner('serving_default')
    predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

    class_names = ['Subtomogram Averaging', 'Tomogram']
    print(
        "This map is likely a {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    )

    data = {'Map': file, 'Expected Type': Folder, 'Predicted Type': class_names[np.argmax(score_lite)], 'Prediction score %': 100 * np.max(score_lite)}
    Results.update(data)

DataOut = pd.DataFrame(Results)
DataOut.to_csv('results_{}_subtomo.csv'.format(Folder))

