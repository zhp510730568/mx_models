import os

import shutil
import numpy as np
import cv2 as cv

from mxnet.tools import im2rec

root_path = './train_dir/'
train_ds_path = './valid_ds'
test_ds_path = './test_ds'
valid_ds_path = './train_ds'


def show(label, image_file):
    image_path = os.path.join(root_path, label, image_file)
    image = cv.imread(image_path, cv.IMREAD_COLOR)

    cv.imshow(label, image)
    cv.waitKey(0)


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def split(train_ratio = 0.10):
    label_dirs = ((os.path.join(root_path, label_dir), label_dir) for label_dir in os.listdir(root_path))
    for (label_dir, label) in label_dirs:
        train_label_dir = os.path.join(train_ds_path, label)
        valid_label_dir = os.path.join(valid_ds_path, label)
        makedir(train_label_dir)
        makedir(valid_label_dir)
        image_files = os.listdir(label_dir)
        count = len(image_files)
        x = np.arange(0, count)
        np.random.shuffle(x)
        train_count = int(count * train_ratio)

        for image_file in image_files[0: train_count]:
            shutil.copy(os.path.join(label_dir, image_file), os.path.join(train_label_dir, image_file))

        for image_file in image_files[train_count: -1]:
            shutil.copy(os.path.join(label_dir, image_file), os.path.join(valid_label_dir, image_file))


import mxnet.ndarray._internal as internal

def filterimage():
    for label_dir in os.listdir(train_ds_path)[0: -1]:
        print('label: ', label_dir)
        for image_file in os.listdir(os.path.join(train_ds_path, label_dir)):
            image_path = os.path.join(train_ds_path, label_dir, image_file)
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
                try:
                    print("normal: ", image_path)
                    image_nd = internal._cvimread(image_path, flag=1)
                except:
                    print("err: ", image_path)
                    os.remove(image_path)
            else:
                print("format error: ", image_path)
                os.remove(image_path)

    for label_dir in os.listdir('./train_ds'):
        print('label: ', label_dir)
        for image_file in os.listdir(os.path.join('./train_ds', label_dir)):
            image_path = os.path.join('./train_ds', label_dir, image_file)
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
                try:
                    print("normal: ", image_path)
                    image_nd = internal._cvimread(image_path, flag=1)
                except:
                    print("err: ", image_path)
                    os.remove(image_path)
            else:
                print("format error: ", image_path)
                os.remove(image_path)

def moveimage(fraction=0.5):
    for label_dir in os.listdir(train_ds_path)[0: -1]:
        print('label: ', label_dir)
        for image_file in os.listdir(os.path.join(train_ds_path, label_dir)):
            image_path = os.path.join(train_ds_path, label_dir, image_file)
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
                try:
                    print("normal: ", image_path)
                    image_nd = internal._cvimread(image_path, flag=1)
                except:
                    print("err: ", image_path)
                    os.remove(image_path)
            else:
                print("format error: ", image_path)
                os.remove(image_path)

    for label_dir in os.listdir('./train_ds'):
        print('label: ', label_dir)
        for image_file in os.listdir(os.path.join('./train_ds', label_dir)):
            image_path = os.path.join('./train_ds', label_dir, image_file)
            if image_file.endswith('.jpg') or image_file.endswith('.jpeg') or image_file.endswith('.png'):
                try:
                    print("normal: ", image_path)
                    image_nd = internal._cvimread(image_path, flag=1)
                except:
                    print("err: ", image_path)
                    os.remove(image_path)
            else:
                print("format error: ", image_path)
                os.remove(image_path)