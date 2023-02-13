#!/usr/bin/env python

import tensorflow as tf
import os

os.system('module load cudnn-8.0.4.30-11.1-gcc-9.3.0-bbr3kjv')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")