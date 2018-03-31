import os

import numpy as np
import cv2 as cv
import mxnet as mx
from mxnet import image
from mxnet import nd

from matplotlib import pyplot as plt


root_path = './train_dir'
label_path = '99'
image_path = '99_13536.jpg'

imageloader = mx.gluon.data.RecordFileDataset('./data/image.rec')
data = np.frombuffer(imageloader[0][0])
print(data.shape)