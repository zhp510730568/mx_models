import os
import sys

import numpy as np

import cv2 as cv

from mxnet import image

sys.path.append('..')
np.random.seed(10)

DATA_PATH = './data/datasets'
TRAIN_PATH = './train_ds'
TEST_PATH = './test_ds'
VALID_PATH = './valid_ds'

IMG_EXTS = ['.jpg', '.jpeg', '.png']


def splitfilename(file_name):
    (name, extension) = os.path.splitext(file_name)
    return name, extension


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def augument(data_path, label, image_name, save_path, size=224, training = True):
    image_path = os.path.join(data_path, image_name)
    (name, extension) = splitfilename(image_name)
    extension = extension.lower()
    if extension not in IMG_EXTS:
        print('filered image: %s' % image_name)
        return
    try:
        img = image.imdecode(open(image_path, 'rb').read()).astype('float32')
    except Exception as ex:
        print("error: ", ex)
        return
    if label is not None:
        label_path = os.path.join(save_path, label)
    else:
        label_path = save_path
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

        img_resize = image.imresize(img, w=size, h=size, interp=2)
        new_name = "%s_%s%s" % (name, "6", extension)
        cv.imwrite(os.path.join(label_path, new_name), img_resize.asnumpy())
    else:
        img = image.resize_short(img, size=size)
        img, _ = image.center_crop(img, size=(size, size))
        new_name = "%s%s" % (name, extension)
        cv.imwrite(os.path.join(label_path, new_name), img.asnumpy())


def augument_dataset(train_ratio=0.95):
    mkdir(TRAIN_PATH)
    mkdir(TEST_PATH)
    mkdir(VALID_PATH)
    train_labels_path = os.path.join(DATA_PATH, 'train.txt')
    test_labels_path = os.path.join(DATA_PATH, 'test.txt')
    train_path = os.path.join(DATA_PATH, 'train')
    test_path = os.path.join(DATA_PATH, 'test')
    with open(train_labels_path, 'r') as f:
        content = f.read().strip().split('\n')
        np.random.shuffle(content)
        imgs_count = len(content)
        split_index = int(imgs_count * train_ratio)
        for line in content[0: split_index]:
            img_label = line.strip().split(' ')
            img_name = img_label[0]
            label = img_label[1]
            print('train: ', img_label)
            augument(train_path, label, img_name, TRAIN_PATH, size=112, training=True)
        for line in content[split_index: -1]:
            img_label = line.strip().split(' ')
            img_name = img_label[0]
            label = img_label[1]
            print('valid: ', img_label)
            augument(train_path, label, img_name, VALID_PATH, size=112, training=False)
    with open(test_labels_path, 'r') as f:
        for line in f:
            img_name = line.strip()
            augument(test_path, None, img_name, TEST_PATH, size=112, training=False)


if __name__ == '__main__':
    # augument_dataset()
    for idx in range(1, 101):
        mkdir(os.path.join(VALID_PATH, str(idx)))
