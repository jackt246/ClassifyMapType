#!/usr/bin/env python

import tensorflow as tf
import os

os.system('module load cudnn-8.0.4.30-11.1-gcc-9.3.0-bbr3kjv')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))