import os

import numpy as np
import cv2 as cv

from mxnet import image

root_path = './train_dir'
label_path = '99'
image_path = '99_13529.jpg'
print(os.path.join(root_path, label_path, image_path))
img = cv.imread('./train_dir/99/99_13529.jpg')
print(img)
cv.imshow('99', img)
cv.waitKey(0)

auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                                rand_crop=False, rand_resize=False, rand_mirror=True,
                                mean=np.array([0.4914, 0.4822, 0.4465]),
                                std=np.array([0.2023, 0.1994, 0.2010]),
                                brightness=0, contrast=0,
                                saturation=0, hue=0,
                                pca_noise=0, rand_gray=0, inter_method=2)
for aug in auglist:
    im = aug(im)