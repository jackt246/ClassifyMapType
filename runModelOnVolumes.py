import os
import tensorflow as tf
import numpy as np
import pandas as pd
import mrcfile as mrc
import threading


class convModel():
    def __init__(self, modelPath):
        self.modelPath = modelPath

    def runPrediction(self, map, file, Folder):
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
    Folder = 'ValidationData_NotForTraining/SPA'
def run(Folder, CSVname, Model_name):
    FilesList = os.listdir(Folder)

    Results = pd.DataFrame(columns=['Map', 'Expected Type', 'Predicted Type', 'Prediction score %'])

    model = convModel(Model_name)

    for file in FilesList:
        # Open and pre-process map
        MapLocation = '{}/{}'.format(Folder, file)
        print(MapLocation)
        try:
            map = mapObject(MapLocation)
            processedMap = map.cropAndPad()
            data = model.runPrediction(processedMap, file, Folder)
            # Attempt to concatenate data with Results DataFrame
            try:
                Results = pd.concat([Results, data], ignore_index=True)
            except pd.errors.EmptyDataError:
                # Handle case where data DataFrame is empty
                print("Data DataFrame is empty for file: {}".format(file))
            except pd.errors.DtypeWarning:
                # Handle other potential errors related to DataFrame concatenation
                print("Error concatenating DataFrame for file: {}".format(file))

        except Exception as e:
            # Catch any other exceptions that might occur
            print('Error processing file {}: {}'.format(file, str(e)))

    Results.to_csv('{}.csv'.format(CSVname))

def run_task(data_folder, csv_name, Model_name):
    run(data_folder, csv_name, Model_name='3DConv_epoch100_trainingrate1e-5_dropout02.tflite')

file_name_descriptor = '_model_3d_1e-5_dropout2'

tasks = [
    ('ValidationData_NotForTraining/Tomograms', 'Tomograms{}'.format(file_name_descriptor)),
    ('ValidationData_NotForTraining/STA', 'STA{}'.format(file_name_descriptor)),
    ('ValidationData_NotForTraining/SPA', 'SPA{}'.format(file_name_descriptor)),
    ('ValidationData_NotForTraining/Ipets', 'Ipets{}'.format(file_name_descriptor)),
    ('ValidationData_NotForTraining/Helical', 'Helical{}'.format(file_name_descriptor))
]

# Create and start threads for each task
threads = []
for task in tasks:
    thread = threading.Thread(target=run_task, args=task)
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All tasks completed.")

# # run on Tomograms
# run('ValidationData_NotForTraining/Tomograms', 'Tomograms_Model_3D_1e-5_dropout04')
#
# # run on STA
# run('ValidationData_NotForTraining/STA', 'STA_Model_3D_1e-5_dropout04')
#
# # run on SPA
# run('ValidationData_NotForTraining/SPA', 'SPA_Model_3D_1e-5_dropout04')
#
# # run on IPET
# run('ValidationData_NotForTraining/Ipets', 'Ipets_Model_3D_1e-5_dropout04')
#
# # run on helical
# run('ValidationData_NotForTraining/Helical', 'Helical_Model_3D_1e-5_dropout04')
