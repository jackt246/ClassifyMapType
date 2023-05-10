import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import mrcfile

#Functions

def OpenMrc(filename):
    # filename must be string
    map = mrcfile.read(filename)
    return map

def saveMrc(array, location, name):
    with mrcfile.new("{}/{}".format(location, name), overwrite=True) as mrc:
        mrc.set_data(array)

#Import mrc file as a numpy array


class MapEditor():
    #Need standardised array shapes for machine learning
    def __init__(self, array):
        self.array = array

    def cropAndPad(self):
        endArraySize = 200

        #pad array to end size
        pad_width = [(max(endArraySize - shape, 0) // 2, max(endArraySize - shape, 0) - max(endArraySize - shape, 0) // 2) for shape in self.array.shape]
        arrayPad = np.pad(self.array, pad_width, 'constant')

        if any(x > 200 for x in arrayPad.shape):
            #then crop
            axis0start = arrayPad.shape[0] // 2 - (endArraySize//2)
            axis1start = arrayPad.shape[1] // 2 - (endArraySize//2)
            axis2start = arrayPad.shape[2] // 2 - (endArraySize//2)

            croppedArray = arrayPad[axis0start:axis0start+endArraySize, axis1start:axis1start+endArraySize,
                           axis2start:axis2start+endArraySize]

            return croppedArray
        else:
            return arrayPad

def PrepFiles(DirectoryName, OutputDir):
    Dir = os.listdir(DirectoryName)
    for file in Dir:
        if '.map' in file:
            map = OpenMrc('{}/{}'.format(DirectoryName, file))
            mapForEdit = MapEditor(map)
            newMap = mapForEdit.cropAndPad()
            saveMrc(newMap, OutputDir, file)




DirectoryName = ('/hps/nobackup/gerard/emdb/TomogramCheck/NonTomograms')
OutputDir = ('Classes3D/NonTomograms')
DirectoryName2 = ('/hps/nobackup/gerard/emdb/TomogramCheck/Tomograms')
OutputDir2 = ('Classes3D/Tomograms')

PrepFiles(DirectoryName, OutputDir)
PrepFiles(DirectoryName2, OutputDir2)
