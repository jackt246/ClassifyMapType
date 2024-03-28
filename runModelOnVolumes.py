import os
import tensorflow as tf
import numpy as np
import pandas as pd
import mrcfile as mrc


class convModel():
    def __init__(self, modelPath):
        self.modelPath = modelPath

    def runPrediction(self, map):
        interpreter = tf.lite.Interpreter(model_path=self.modelPath)
        #Tensorflow has a habit of changing the signatures so this is run to tell me what it is
        signatures = interpreter.get_signature_list()
        print('Signature: {}'.format(signatures))

        try:
            classify_lite = interpreter.get_signature_runner('serving_default')
            predictions = classify_lite(conv3d_input=map)['dense_1']


            score = tf.nn.softmax(predictions)

            class_names = ['Tomogram', 'Non-Tomogram']
            print(
                "This map is likely a {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

            data = pd.DataFrame({'Map': file, 'Expected Type': Folder, 'Predicted Type': class_names[np.argmax(score)],
                             'Prediction score %': 100 * np.max(score)}, index=[0])
            return data

        except:
            print('unable to run model on this entry')



class mapObject():
    def __init__(self, mapPath):
        with mrc.open(mapPath, 'r') as mapfile:
            self.mapArray = mapfile.data
    def cropAndPad(self):
        endArraySize = 200
        #pad array to end size
        pad_width = [(max(endArraySize - shape, 0) // 2, max(endArraySize - shape, 0) - max(endArraySize - shape, 0) // 2) for shape in self.mapArray.shape]
        self.arrayPad = np.pad(self.mapArray, pad_width, 'constant')

        if any(x > 200 for x in self.arrayPad.shape):
            #then crop
            axis0start = self.arrayPad.shape[0] // 2 - (endArraySize//2)
            axis1start = self.arrayPad.shape[1] // 2 - (endArraySize//2)
            axis2start = self.arrayPad.shape[2] // 2 - (endArraySize//2)

            self.preppedarray = self.arrayPad[axis0start:axis0start+endArraySize, axis1start:axis1start+endArraySize,
                           axis2start:axis2start+endArraySize]

            # Reshape to add batch and channel dimensions
            self.preppedarray = np.expand_dims(self.preppedarray, axis=0)  # Add batch dimension (axis=0)
            self.preppedarray = np.expand_dims(self.preppedarray, axis=-1)  # Add channel dimension (axis=-1)
            return self.preppedarray
        else:
            self.preppedarray = self.arrayPad
            # Reshape to add batch and channel dimensions
            self.preppedarray = np.expand_dims(self.preppedarray, axis=0)  # Add batch dimension (axis=0)
            self.preppedarray = np.expand_dims(self.preppedarray, axis=-1)  # Add channel dimension (axis=-1)
            return self.preppedarray

Folder = 'ValidationData_NotForTraining/Tomograms'
FilesList = os.listdir(Folder)

Results = pd.DataFrame(columns=['Map', 'Expected Type', 'Predicted Type', 'Prediction score %'])

model = convModel('3dconv.tflite')

for file in FilesList:
    # Open and pre-process map
    MapLocation = '{}/{}'.format(Folder, file)
    print(MapLocation)
    map = mapObject(MapLocation)
    processedMap = map.cropAndPad()
    data = model.runPrediction(processedMap)

    Results = pd.concat([Results, data], ignore_index=True)

print(Results)
Results.to_csv('results_{}.csv'.format(Folder))
