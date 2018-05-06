import os
import sys

import numpy as np

import cv2 as cv

from mxnet import image

sys.path.append('..')
import utils

DATA_PATH = './train_dir'
TRAIN_PATH = './train_ds'
TEST_PATH = './test_ds'

IMG_EXTS = ['.jpg', '.jpeg', '.png']


def splitfilename(file_name):
    (name, extension) = os.path.splitext(file_name)
    return name, extension


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def augument(data_path, label, image_name, save_path, size=224, training = True):
    image_path = os.path.join(data_path, label, image_name)
    (name, extension) = splitfilename(image_name)
    extension = extension.lower()
    print(extension)
    if extension not in IMG_EXTS:
        print('filered image: %s' % image_name)
        return
    try:
        img = image.imdecode(open(image_path, 'rb').read()).astype('float32')
    except Exception as ex:
        print("error: ", ex)
        return
    label_path = os.path.join(save_path, label)
    mkdir(label_path)

    if training:
        aug1 = image.HorizontalFlipAug(0.5)
        aug2 = image.HorizontalFlipAug(.5)

        img = image.resize_short(img, size=384, interp=2)

        center_crop, _ = image.center_crop(img, size=(size, size))
        new_name = "%s_%s%s" % (name, "0", extension)
        cv.imwrite(os.path.join(label_path, new_name), center_crop.asnumpy())

        random_crop, _ = image.random_crop(img, size=(size, size))
        new_name = "%s_%s%s" % (name, "1", extension)
        cv.imwrite(os.path.join(label_path, new_name), random_crop.asnumpy())

        random_crop, _ = image.random_crop(img, size=(size, size))
        new_name = "%s_%s%s" % (name, "2", extension)
        cv.imwrite(os.path.join(label_path, new_name), random_crop.asnumpy())

        random_crop, _ = image.random_crop(img, size=(size, size))
        new_name = "%s_%s%s" % (name, "3", extension)
        cv.imwrite(os.path.join(label_path, new_name), random_crop.asnumpy())

        img_aug1 = aug1(random_crop).clip(0,255)
        new_name = "%s_%s%s" % (name, "4", extension)
        cv.imwrite(os.path.join(label_path, new_name), img_aug1.asnumpy())

        img_aug2 = aug2(center_crop).clip(0, 255)
        new_name = "%s_%s%s" % (name, "5", extension)
        cv.imwrite(os.path.join(label_path, new_name), img_aug2.asnumpy())

        img_resize = image.imresize(img, w=size, h=224, interp=2)
        new_name = "%s_%s%s" % (name, "6", extension)
        cv.imwrite(os.path.join(label_path, new_name), img_resize.asnumpy())
    else:
        img = image.resize_short(img, size=size)
        img, _ = image.center_crop(img, size=(size, size))
        new_name = "%s%s" % (name, extension)
        cv.imwrite(os.path.join(label_path, new_name), img.asnumpy())


def augument_dataset(data_path, train_path, test_path, train_ratio=0.95):
    total_count = 0
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        image_list = os.listdir(label_path)
        image_count = len(image_list)
        split = int(image_count * train_ratio)
        np.random.shuffle(image_list)
        for image_name in image_list[0: split]:
            img_path = os.path.join(label_path, image_name)
            print('train: %s' % img_path)
            augument(data_path, label, image_name, train_path, size=224, training=True)
            total_count += 1

        for image_name in image_list[split:]:
            img_path = os.path.join(label_path, image_name)
            augument(data_path, label, image_name, test_path, size=224, training=False)
            print('test: %s' % img_path)
            total_count += 1

    print('total count: %s' % total_count)


augument_dataset(DATA_PATH, TRAIN_PATH, TEST_PATH)
