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
        predictions_lite = classify_lite(rescaling_1_input=self.imgArray)['dense_1']
        score_lite = tf.nn.softmax(predictions_lite)

    #Import network
    TF_MODEL_FILE_PATH = 'Training_summary_fullDataSet_ImgSize100_learningrate1e3_epoch75_wdienetwork2_NETWORK.tflite'
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    classify_lite = interpreter.get_signature_runner('serving_default')
    predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

        class_names = ['Subtomogram Averaging', 'Tomogram']
        print(
            "This map is likely a {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        )

        data = pd.DataFrame({'Map': file, 'Expected Type': Folder, 'Predicted Type': class_names[np.argmax(score_lite)],
                             'Prediction score %': 100 * np.max(score_lite)}, index=[0])

        return data


Folder = 'SubtomogramAverages'
FilesList = os.listdir(Folder)

Results = pd.DataFrame(columns=['Map', 'Expected Type', 'Predicted Type', 'Prediction score %'])

#Open and pre-process map

for file in FilesList:
    # Open and pre-process map
    MapLocation = '{}/{}'.format(Folder, file)
    MapEdit = emMap(MapLocation)
    Z = MapEdit.CentSliceZ()
    # Projected = Projector(Map, 2)

    plt.imsave('ImageToClassify_subtomo.png', Z, cmap='Greys')
    ImageLoc = 'ImageToClassify_subtomo.png'

    # Set up information on the data

    mapTypePredictor = runModel('model_resize.tflite')
    mapTypePredictor.imageSetup(ImageLoc)
    data = mapTypePredictor.runPrediction()
    Results = Results.append(data, ignore_index=True)

print(Results)
Results.to_csv('results_{}_subtomo.csv'.format(Folder))

