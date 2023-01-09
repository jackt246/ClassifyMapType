import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import mrcfile as mrc

#Functions

def OpenMrc(filename):
    # filename must be string
    map = mrc.read(filename)
    return map


#Import mrc file as a numpy array


class MapEditor():
    #Need standardised array shapes for machine learning
    def __init__(self, array):
        self.array = array

    def StandardShapeZ(self):
        #takes numpy array
        array = self.array
        y,x,z = self.array.shape
        startx = x//2 - 100//2
        starty = y//2 - 100//2
        return self.array[starty:starty+100, startx:startx+100, :]

    def StandardShapeY(self):
        #takes numpy array
        y,x,z = self.array.shape
        startx = x//2 - 100//2
        startz = z//2 - 100//2
        return self.array[:, startx:startx+100, startz:startz+100]

    def StandardShapeX(self):
        #takes numpy array
        y,x,z = self.array.shape
        starty= y//2 - 100//2
        startz = z//2 - 100//2
        return self.array[starty:starty+100, :, startz:startz+100]

def SaveImages(array, axis, extension, Directory):
    length = array.shape
    print(array.shape[0])
    print(array.shape[1])
    print(array.shape[2])
    print(axis)
    if axis in ['Z', 'z']:
        print('doing z')
        #For each only save the central 80% of the data, this will hopefully remove any masked regions and horrible boundary crap
        for plane in range(int((length[2] / 10)), int(((length[2] / 100) * 90))):
            if array.shape[0] > 100 or array.shape[1] > 100:
                print(array.shape[0], array.shape[1])
                return False
            else:
                CurrentPlane = array[:,:,plane]
                plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlane, cmap='Greys')
    elif axis in ['Y', 'y']:
        print('doing y')
        for plane in range(int((length[0] / 10)), int(((length[0] / 100) * 90))):
            if array.shape[1]  > 100 or array.shape[2] > 100:
                print(array.shape[1], array.shape[2])
                return False
            else:
                CurrentPlane = array[plane,:,:]
                plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlane, cmap='Greys')
    elif axis in ['X', 'x']:
        print('doing x')
        for plane in range(int((length[1] / 10)), int(((length[1] / 100) * 90))):
            if array.shape[0] > 100 or array.shape[2] > 100:
                print(array.shape[0], array.shape[2])
                return False
            else:
                CurrentPlane = array[:,plane,:]
                plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlane, cmap='Greys')

def SaveAndResize(array, axis, extension, Directory):
    length = array.shape
    print(array.shape[0])
    print(array.shape[1])
    print(array.shape[2])
    print(axis)
    if axis in ['Z', 'z']:
        #print('doing z')
        #For each only save the central 80% of the data, this will hopefully remove any masked regions and horrible boundary crap
        for plane in range(int((length[2] / 10)), int(((length[2] / 100) * 90))):
            CurrentPlane = array[:,:,plane]
            CurrentPlaneResize = np.resize(CurrentPlane, (300,300))
            plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlaneResize, cmap='Greys')
    elif axis in ['Y', 'y']:
        #print('doing y')
        for plane in range(int((length[0] / 10)), int(((length[0] / 100) * 90))):
                CurrentPlane = array[plane,:,:]
                CurrentPlaneResize = np.resize(CurrentPlane, (300, 300))
                plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlaneResize, cmap='Greys')
    elif axis in ['X', 'x']:
        #print('doing x')
        for plane in range(int((length[1] / 10)), int(((length[1] / 100) * 90))):
            CurrentPlane = array[:,plane,:]
            CurrentPlaneResize = np.resize(CurrentPlane, (300, 300))
            plt.imsave('{}/{}_{}.png'.format(Directory, plane, extension), CurrentPlaneResize, cmap='Greys')

def SaveAndProject(array, extension, Directory):
    length = array.shape
    print(array.shape[0])
    print(array.shape[1])
    print(array.shape[2])

    Zsum = np.sum(array, axis=2)
    Ysum = np.sum(array, axis=0)
    Xsum = np.sum(array, axis=1)
    print('i projected')
    plt.imsave('{}/{}_{}.png'.format(Directory, 'z', extension), Zsum, cmap='Greys')
    plt.imsave('{}/{}_{}.png'.format(Directory, 'y', extension), Ysum, cmap='Greys')
    plt.imsave('{}/{}_{}.png'.format(Directory, 'x', extension), Xsum, cmap='Greys')

def PrepMapsCentralHundred(DirectoryName, OutputDir):

    Dir = os.listdir(DirectoryName, OutputDir)
    for file in Dir:
        if '.map' in file:
            map = OpenMrc('{}/{}'.format(DirectoryName, file))
            mapDimensions = map.shape
            #print(mapDimensions)
            MapEdit = MapEditor(map)
            if mapDimensions[0] > 100 and mapDimensions[1] > 100 and mapDimensions[2] > 100:
                ArrayForOutputX = MapEdit.StandardShapeX()
                nameX = '{}_X'.format(file)
                SaveImages(ArrayForOutputX, 'x', nameX, OutputDir)
                ArrayForOutputY = MapEdit.StandardShapeY()
                nameY = '{}_Y'.format(file)
                SaveImages(ArrayForOutputY, 'y', nameY, OutputDir)
                ArrayForOutputZ = MapEdit.StandardShapeZ()
                nameZ = '{}_Z'.format(file)
                SaveImages(ArrayForOutputZ, 'z', nameZ, OutputDir)
                print('Written images for {}'.format(file))
            else:
                print('{} map dimensions are {} and so it misses out on being part of EMDB Deep Learning'.format(file, mapDimensions))


def PrepMapsResize(DirectoryName, OutputDir):
    Dir = os.listdir(DirectoryName)
    for file in Dir:
        if '.map' in file:
            map = OpenMrc('{}/{}'.format(DirectoryName, file))
            nameX = '{}_X'.format(file)
            SaveAndResize(map, 'x', nameX, OutputDir)
            nameY = '{}_Y'.format(file)
            SaveAndResize(map, 'y', nameY, OutputDir)
            nameZ = '{}_Z'.format(file)
            SaveAndResize(map, 'z', nameZ, OutputDir)

def PrepMapsProject(DirectoryName, OutputDir):
    Dir = os.listdir(DirectoryName)
    for file in Dir:
        if '.map' in file:
            map = OpenMrc('{}/{}'.format(DirectoryName, file))
            SaveAndProject(map, file, OutputDir)

DirectoryName = ('NonTomograms')
OutputDir = ('Classes300/NonTomograms')

DirectoryName2 = ('Tomograms')
OutputDir2 = ('Classes300/Tomograms')

PrepMapsResize(DirectoryName, OutputDir)
PrepMapsResize(DirectoryName2, OutputDir2)
      
